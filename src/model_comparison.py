"""Isolated topic experiments on the frozen Stage 1 corpus.

Every old artifact is read-only. Neural work is delegated to separate processes
by run_model_comparison.py, so a transformer and BERTopic never coexist in RAM.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache/numba'))

import numpy as np
import pandas as pd
import yaml
from .stage3_common import digest, signature, read_json, write_json

CONFIG = ROOT / 'configs/model_comparison.yaml'
BGE = ROOT / 'outputs/bertopic_chunked'


def cfg():
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def table(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def corpus():
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet')
    tokens = pd.read_parquet(ROOT / 'data/processed/lda_tokens.parquet')
    assert len(main) == cfg()['expected_documents'] and main.doc_id.is_unique
    assert main.doc_id.tolist() == tokens.doc_id.tolist()
    assert all(len(t) == n for t, n in zip(tokens.tokens, tokens.token_count))
    return main, [list(t) for t in tokens.tokens]


def validate_vectors(path, ids, expected_ids, dimension=None):
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise ValueError('Embedding document IDs/order do not match the frozen corpus')
    vectors = np.load(path, mmap_mode='r')
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError('Embedding matrix must have one row per ordered document ID')
    if dimension is not None and vectors.shape[1] != dimension:
        raise ValueError('Embedding dimension does not match the encoder')
    if not np.issubdtype(vectors.dtype, np.floating):
        raise ValueError('Embedding vectors must use a floating-point dtype')
    low, high = float('inf'), 0.
    for start in range(0, len(ids), 256):
        part = vectors[start:start + 256]
        if not np.isfinite(part).all():
            raise ValueError('Nonfinite embeddings')
        norms = np.linalg.norm(part, axis=1)
        if not np.allclose(norms, 1., atol=2e-5):
            raise ValueError('Embeddings are not L2 normalized')
        low, high = min(low, float(norms.min())), max(high, float(norms.max()))
    return dict(shape=list(vectors.shape), dtype=str(vectors.dtype),
                minimum_l2_norm=low, maximum_l2_norm=high, sha256=digest(path))


def old_files(out):
    for name in ('data', 'models', 'outputs'):
        for path in (ROOT / name).rglob('*'):
            if path.is_file() and not path.is_relative_to(out):
                yield path
    for path in (ROOT / 'configs').glob('*.yaml'):
        if path != CONFIG:
            yield path


def verify_protected(out):
    manifest = read_json(out / 'input_manifest.json')
    for name, expected in manifest['protected_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Protected artifact changed: ' + name)
    return len(manifest['protected_hashes'])


def preflight(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'input_manifest.json'
    if not manifest.exists():
        log('Hashing all pre-existing data/models/output artifacts (read-only)')
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in old_files(out)}
        write_json(manifest, dict(protected_hashes=hashes, config=cfg(),
                    config_sha256=digest(CONFIG), versions={n: importlib.metadata.version(n)
                    for n in ('numpy', 'pandas', 'gensim', 'torch', 'transformers',
                              'bertopic', 'umap-learn', 'hdbscan', 'scikit-learn')}))
    else:
        assert read_json(manifest)['config_sha256'] == digest(CONFIG)
        verify_protected(out)
    main, texts = corpus()
    from gensim import corpora
    dictionary = corpora.Dictionary.load(str(ROOT / 'models/lda/dictionary.gensim'))
    bow = corpora.MmCorpus(str(ROOT / 'data/stage2/corpus_bow.mm'))
    assert len(bow) == len(main)
    assert corpora.Dictionary(texts).token2id == dictionary.token2id
    assert all(dictionary.doc2bow(t) == list(b) for t, b in zip(texts, bow))
    pairs = pd.read_parquet(ROOT / 'data/stage3/contrastive_pairs.parquet')
    assert pairs.doc_id.is_unique and set(pairs.doc_id) <= set(main.doc_id)
    assert pairs.groupby('split').size().to_dict() == cfg()['batterybert']['expected_splits']
    assert pairs.groupby('split_group_id')['split'].nunique().max() == 1
    indexed = main.set_index('doc_id').loc[pairs.doc_id]
    assert indexed.split_group_id.tolist() == pairs.split_group_id.tolist()
    assert indexed.title.tolist() == pairs.title.tolist()
    assert indexed.abstract.tolist() == pairs.abstract.tolist()
    split_assignments = pd.read_parquet(ROOT / 'data/stage3/split_assignments.parquet')
    assert split_assignments.set_index('doc_id').loc[pairs.doc_id, 'split'].tolist() == pairs.split.tolist()
    ids = main.doc_id.tolist()
    vectors = validate_vectors(BGE / 'document_embeddings.npy',
                               read_json(BGE / 'ordered_doc_ids.json'), ids, 768)
    assert vectors['sha256'] == read_json(BGE / 'embedding_success.json')['document_embeddings_sha256']
    assert signature(ids) == read_json(BGE / 'embedding_success.json')['ordered_doc_ids_sha256']
    assert digest(BGE / 'model.pickle') == read_json(BGE / 'fit_success.json')['model_sha256']
    for folder in ('lda_k10', 'bertopic_bge_k10', 'batterybert_pretrained',
                   'batterybert_finetuned', 'bertopic_batterybert_k10', 'evaluation', 'reports'):
        (out / folder).mkdir(exist_ok=True)
    write_json(out / 'ordered_doc_ids.json', ids)
    write_json(out / 'config.json', cfg())
    write_json(out / 'preflight.json', dict(status='PASS', documents=len(main),
                vocabulary=len(dictionary), tokens=sum(map(len, texts)),
                split_counts=pairs.groupby('split').size().to_dict(),
                group_disjoint=True, bow_matches_every_document=True, bge_embeddings=vectors,
                protected_files=len(read_json(manifest)['protected_hashes'])))
    log('Preflight PASS: frozen corpus, every BoW row, group splits, BGE IDs/hash/norms')


def evaluate_words(words, texts, dictionary):
    from gensim import corpora
    from gensim.models import CoherenceModel
    if not words:
        return dict(topics=0, c_v=None, u_mass=None, topic_diversity=None), pd.DataFrame()
    keys = sorted(words)
    topics = [words[t] for t in keys]
    assert all(len(w) == 10 and all(x in dictionary.token2id for x in w) for w in topics)
    cv = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary,
        coherence='c_v', window_size=110, topn=10, processes=1).get_coherence_per_topic()
    um = CoherenceModel(topics=topics, corpus=corpora.MmCorpus(str(ROOT / 'data/stage2/corpus_bow.mm')),
        dictionary=dictionary, coherence='u_mass', topn=10, processes=1).get_coherence_per_topic()
    assert np.isfinite(cv).all() and np.isfinite(um).all()
    metrics = dict(topics=len(keys), c_v=float(np.mean(cv)), u_mass=float(np.mean(um)),
                   topic_diversity=len(set(w for topic in topics for w in topic)) / (10 * len(keys)))
    return metrics, pd.DataFrame(dict(topic_id=keys, c_v=cv, u_mass=um))


def interpretive_label(words, titles):
    """Evidence-derived tentative labels, never expert ground truth."""
    evidence = ' '.join(words + titles).lower()
    rules = [('Hard-carbon anodes and sodium storage', ('hard', 'carbon', 'anode')),
             ('Layered oxide cathodes', ('layered', 'oxide', 'cathode')),
             ('Polyanion cathode materials', ('phosphate', 'polyanion', 'vanadium')),
             ('Electrolytes and interfacial chemistry', ('electrolyte', 'interface', 'interphase')),
             ('Prussian-blue analogue cathodes', ('prussian', 'hexacyanoferrate')),
             ('Solid-state sodium batteries', ('solid', 'conductivity', 'ceramic'))]
    scores = [sum(term in evidence for term in terms) for _, terms in rules]
    best = int(np.argmax(scores))
    return rules[best][0] if scores[best] >= 2 else ' / '.join(words[:3])


def export_topics(folder, main, labels, keywords, scores, method, distribution=None,
                  distribution_topic_ids=None, distribution_semantics=None):
    folder.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int32)
    assert labels.shape == (len(main),) and np.isfinite(scores).all()
    assignments = main[['doc_id', 'source', 'year', 'split_group_id']].copy()
    assignments['topic_id'] = labels
    assignments['dominant_probability' if method == 'LDA posterior' else 'representative_score'] = scores
    assignments.to_parquet(folder / 'topic_assignments.parquet', index=False)
    table(folder / 'topic_assignments.csv', assignments)
    np.save(folder / 'assignments.npy', labels)
    write_json(folder / 'ordered_doc_ids.json', main.doc_id.tolist())
    key_rows, summaries, representatives, label_rows = [], [], [], []
    for topic in sorted(set(labels) | set(keywords)):
        indices = np.flatnonzero(labels == topic)
        pairs = keywords.get(int(topic), [])
        words = [w for w, _ in pairs]
        summaries.append(dict(topic_id=int(topic), documents=len(indices),
            document_share=len(indices) / len(main), is_outlier=topic == -1,
            top10_keywords='; '.join(words)))
        for rank, (word, weight) in enumerate(pairs, 1):
            key_rows.append(dict(topic_id=int(topic), rank=rank, token=word, weight=float(weight)))
        if topic == -1:
            continue
        order = indices[np.argsort(-scores[indices], kind='stable')][:5]
        titles = []
        for rank, index in enumerate(order, 1):
            row = main.iloc[index]
            titles.append(row.title)
            representatives.append(dict(topic_id=int(topic), rank=rank, doc_id=row.doc_id,
                title=row.title, abstract=row.abstract, source=row.source, year=row.year,
                score=float(scores[index]), selection_method=method))
        label_rows.append(dict(topic_id=int(topic), interpretive_label=interpretive_label(words, titles),
            status='Tentative evidence-based interpretation; requires domain expert review',
            keyword_evidence='; '.join(words), representative_title_evidence=' | '.join(titles[:2])))
    table(folder / 'topic_keywords.csv', pd.DataFrame(key_rows))
    table(folder / 'topic_summary.csv', pd.DataFrame(summaries))
    table(folder / 'representative_documents.csv', pd.DataFrame(representatives))
    table(folder / 'interpretive_labels.csv', pd.DataFrame(label_rows))
    table(folder / 'topic_years.csv', assignments.groupby(['topic_id', 'year'], dropna=False).size().rename('documents').reset_index())
    table(folder / 'topic_sources.csv', assignments.groupby(['topic_id', 'source'], dropna=False).size().rename('documents').reset_index())
    if distribution is not None:
        assert distribution.shape == (len(main), len(distribution_topic_ids))
        assert np.isfinite(distribution).all() and (distribution >= 0).all()
        np.save(folder / 'document_topic_distribution.npy', distribution)
        write_json(folder / 'distribution_metadata.json', dict(topic_ids=distribution_topic_ids,
            semantics=distribution_semantics, outliers='Assignments preserve -1; -1 is excluded from topic columns'))


def lda(out):
    from gensim import corpora
    from gensim.models import LdaModel
    from .stage2 import _fit_lda
    from threadpoolctl import threadpool_limits
    main, texts = corpus()
    folder = out / 'lda_k10'
    dictionary = corpora.Dictionary.load(str(ROOT / 'models/lda/dictionary.gensim'))
    bow = corpora.MmCorpus(str(ROOT / 'data/stage2/corpus_bow.mm'))
    source = ROOT / 'models/lda/k10_seed42'
    reused, reason = False, None
    try:
        marker = read_json(source / 'training_success.json')
        assert marker['k'] == 10 and marker['seed'] == 42 and marker['documents'] == len(main)
        for item in marker['model_files']:
            assert digest(source / item['file']) == item['sha256']
        model = LdaModel.load(str(source / 'model.gensim'))
        assert model.num_topics == 10 and model.id2word.token2id == dictionary.token2id
        assert np.isfinite(model.get_topics()).all()
        reused = True
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        reason = repr(exc)
        settings = yaml.safe_load((ROOT / 'configs/stage2.yaml').read_text())['lda']
        with threadpool_limits(limits=1):
            model = _fit_lda(bow, dictionary, 10, 42, settings)
    if reused:
        for path in source.glob('model.gensim*'):
            shutil.copy2(path, folder / path.name)
    else:
        model.save(str(folder / 'model.gensim'))
    model = LdaModel.load(str(folder / 'model.gensim'))
    model.random_state = np.random.RandomState(42)
    with threadpool_limits(limits=1):
        distribution = np.array([[p for _, p in model.get_document_topics(b, minimum_probability=0.)]
                                 for b in bow], dtype=np.float32)
    assert distribution.shape == (len(main), 10) and np.allclose(distribution.sum(axis=1), 1, atol=1e-5)
    np.save(folder / 'topic_word_distribution.npy', model.get_topics())
    table(folder / 'vocabulary.csv', pd.DataFrame(dict(token_id=list(range(len(dictionary))),
                                                      token=[dictionary[i] for i in range(len(dictionary))])))
    keywords = {i: model.show_topic(i, topn=10) for i in range(10)}
    export_topics(folder, main, distribution.argmax(axis=1), keywords, distribution.max(axis=1),
        'LDA posterior', distribution, list(range(10)), 'Full variational document-topic posterior; rows sum to one')
    metrics, per = evaluate_words({t: [w for w, _ in p] for t, p in keywords.items()}, texts, dictionary)
    metrics.update(outliers=None, outlier_ratio=None, seed=42, reused_checkpoint=reused,
                   fallback_reason=reason, coherence_reference='Frozen full Stage1 tokens / Stage2 BoW; top10; C_v window110')
    write_json(folder / 'metrics.json', metrics)
    table(folder / 'coherence_per_topic.csv', per)
    assert pd.read_csv(folder / 'representative_documents.csv').groupby('topic_id').size().ge(5).all()
    artifact_manifest(folder, dict(model_type='LDA', seed=42, reused_from=str(source) if reused else None))
    log('LDA K10 complete: ' + str(metrics))


def artifact_manifest(folder, provenance):
    hashes = {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob('*')
              if p.is_file() and p.name != 'manifest.json' and '.cache' not in p.parts}
    write_json(folder / 'manifest.json', dict(config=cfg(), provenance=provenance, sha256=hashes))


def reduced_topic_count(model, target):
    """Installed BERTopic includes the outlier slot in nr_topics."""
    return target + int(-1 in model.get_topics())


def export_bertopic(folder, model, main, texts, dictionary, embeddings, initial_labels, provenance):
    labels = np.asarray(model.topics_, dtype=np.int32)
    assert np.array_equal(labels == -1, initial_labels == -1), 'Reduction must preserve outliers'
    keywords = {int(t): [(w, float(p)) for w, p in model.get_topic(t)[:10] if w and p > 0]
                for t in model.get_topics() if t != -1}
    scores = np.zeros(len(main), dtype=np.float32)
    for topic in keywords:
        indices = np.flatnonzero(labels == topic)
        center = np.asarray(embeddings[indices]).mean(axis=0)
        center /= max(float(np.linalg.norm(center)), 1e-12)
        scores[indices] = embeddings[indices] @ center
    # c-TF-IDF affinities are an interpretable topic distribution, not posteriors.
    from sklearn.preprocessing import normalize
    topic_ids = sorted(keywords)
    if topic_ids:
        bow = model.vectorizer_model.transform([' '.join(t) for t in texts])
        selected = [list(sorted(model.get_topics())).index(t) for t in topic_ids]
        affinities = (normalize(bow) @ normalize(model.c_tf_idf_[selected]).T).toarray()
        distribution = normalize(affinities, norm='l1').astype(np.float32)
    else:
        distribution = np.zeros((len(main), 0), dtype=np.float32)
    export_topics(folder, main, labels, keywords, scores,
        'Cosine similarity to assigned merged-topic document-embedding centroid', distribution, topic_ids,
        'L1-normalized cosine affinities between frozen document BoW and topic c-TF-IDF; not calibrated probabilities. Zero rows have no affinity.')
    mapping = pd.DataFrame(dict(old_topic_id=initial_labels, new_topic_id=labels)).drop_duplicates()
    assert mapping.groupby('old_topic_id').new_topic_id.nunique().max() == 1
    table(folder / 'topic_mapping.csv', mapping.sort_values('old_topic_id'))
    metrics, per = evaluate_words({t: [w for w, _ in p] for t, p in keywords.items()}, texts, dictionary)
    sizes = pd.Series(labels[labels != -1]).value_counts()
    metrics.update(outliers=int((labels == -1).sum()), outlier_ratio=float((labels == -1).mean()),
        target_topics=10, target_reached=len(keywords) == 10,
        smallest_topic=int(sizes.min()) if len(sizes) else None,
        largest_topic=int(sizes.max()) if len(sizes) else None,
        largest_topic_corpus_share=float(sizes.max() / len(main)) if len(sizes) else None,
        coherence_reference='Frozen full Stage1 tokens / Stage2 BoW; top10; C_v window110; -1 excluded')
    write_json(folder / 'metrics.json', metrics)
    table(folder / 'coherence_per_topic.csv', per)
    model.save(str(folder / 'model.pickle'), serialization='pickle')
    from bertopic import BERTopic
    reloaded = BERTopic.load(str(folder / 'model.pickle'))
    assert np.array_equal(reloaded.topics_, labels)
    assert reloaded.vectorizer_model.vocabulary_ == dictionary.token2id
    assert np.isfinite(reloaded.c_tf_idf_.data).all()
    artifact_manifest(folder, provenance)
    log(f'Topic exports {folder.name}: {metrics}')


def topics(out, variant):
    from gensim import corpora
    from bertopic import BERTopic
    main, texts = corpus()
    dictionary = corpora.Dictionary.load(str(ROOT / 'models/lda/dictionary.gensim'))
    docs = [' '.join(t) for t in texts]
    if variant == 'bge':
        folder = out / 'bertopic_bge_k10'
        embedding_path, id_path = BGE / 'document_embeddings.npy', BGE / 'ordered_doc_ids.json'
        model = BERTopic.load(str(BGE / 'model.pickle'))
        assert np.array_equal(model.topics_, np.load(BGE / 'assignments.npy'))
        assert model.vectorizer_model.vocabulary_ == dictionary.token2id
        assert len(set(model.topics_) - {-1}) == read_json(BGE / 'metrics.json')['topics']
        initial = np.array(model.topics_, dtype=np.int32)
        provenance = dict(variant=variant, source_model=str(BGE / 'model.pickle'),
                          source_model_sha256=digest(BGE / 'model.pickle'), refit=False)
    else:
        embedding_folder = out / ('batterybert_' + variant)
        embedding_path, id_path = embedding_folder / 'document_embeddings.npy', embedding_folder / 'ordered_doc_ids.json'
        folder = out / 'bertopic_batterybert_k10' / variant
        folder.mkdir(parents=True, exist_ok=True)
        initial_folder = folder / 'initial'
        fit_marker = folder / 'initial_fit.json'
        expected = signature(dict(embedding=digest(embedding_path), ids=digest(id_path), settings=cfg()['bertopic']))
        if fit_marker.exists():
            info = read_json(fit_marker)
            assert info['fingerprint'] == expected
            assert digest(initial_folder / 'model.pickle') == info['model_sha256']
            model = BERTopic.load(str(initial_folder / 'model.pickle'))
            initial = np.array(model.topics_, dtype=np.int32)
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            from bertopic.vectorizers import ClassTfidfTransformer
            from umap import UMAP
            from hdbscan import HDBSCAN
            from joblib import parallel_backend
            from .stage3_bertopic_model import FrozenTokensBERTopic
            settings = cfg()['bertopic']
            model = FrozenTokensBERTopic(embedding_model=None,
                umap_model=UMAP(n_neighbors=settings['umap_neighbors'],
                    n_components=settings['umap_components'], min_dist=settings['umap_min_dist'], metric='cosine',
                    random_state=cfg()['seed'], n_jobs=1, low_memory=True),
                hdbscan_model=HDBSCAN(min_cluster_size=settings['min_cluster_size'], min_samples=settings['min_samples'],
                    cluster_selection_method=settings['cluster_selection_method'], prediction_data=True, core_dist_n_jobs=1),
                vectorizer_model=CountVectorizer(tokenizer=str.split, token_pattern=None,
                    lowercase=False, vocabulary=dictionary.token2id),
                ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=settings['reduce_frequent_words']),
                top_n_words=10, nr_topics=None, calculate_probabilities=False, verbose=True)
            embeddings = np.load(embedding_path, mmap_mode='r')
            validate_vectors(embedding_path, read_json(id_path), main.doc_id.tolist())
            with parallel_backend('sequential', n_jobs=1):
                model.fit_transform(docs, embeddings=embeddings)
            initial = np.array(model.topics_, dtype=np.int32)
            export_bertopic(initial_folder, model, main, texts, dictionary, embeddings, initial,
                             dict(variant=variant, embedding_sha256=digest(embedding_path), reduction=False))
            write_json(fit_marker, dict(fingerprint=expected, model_sha256=digest(initial_folder / 'model.pickle')))
        provenance = dict(variant=variant, refit=False, initial_model=str(initial_folder / 'model.pickle'))
    validate_vectors(embedding_path, read_json(id_path), main.doc_id.tolist())
    embeddings = np.load(embedding_path, mmap_mode='r')
    initial_count = len(set(initial) - {-1})
    api_target = reduced_topic_count(model, 10)
    if initial_count > 10:
        log(f'BERTopic reduction {initial_count} -> 10 research topics; API nr_topics={api_target}')
        model.reduce_topics(docs, nr_topics=api_target, use_ctfidf=False)
    provenance.update(initial_topics=initial_count, api_nr_topics=api_target,
        reduction_applied=initial_count > 10,
        reduction='BERTopic.reduce_topics with semantic topic embeddings; original outlier slot retained'
            if initial_count > 10 else None,
        target_unreachable_reason='Initial clustering has fewer than 10 research topics; reduction cannot create new topics'
            if initial_count < 10 else None, embedding_path=str(embedding_path), embedding_sha256=digest(embedding_path))
    export_bertopic(folder, model, main, texts, dictionary, embeddings, initial, provenance)
    if variant == 'bge':
        shutil.copy2(embedding_path, folder / 'document_embeddings.npy')
        artifact_manifest(folder, provenance)
    write_json(folder / 'inference.json', dict(embedding_path=str(embedding_path),
        encoder=str(ROOT / 'models/bert_finetuned') if variant == 'bge' else str(out / ('batterybert_' + variant) / 'model'),
        supply_explicit_embeddings=True, chunking='256 tokens; overlap64; see embedding manifest',
        assignments='Use saved topic mapper; -1 stays outlier'))


def comparison_rows(out):
    entries = [('LDA K=10', out / 'lda_k10'), ('BERTopic + BGE', out / 'bertopic_bge_k10'),
               ('BERTopic + BatteryBERT pretrained', out / 'bertopic_batterybert_k10/pretrained'),
               ('BERTopic + BatteryBERT fine-tuned', out / 'bertopic_batterybert_k10/finetuned')]
    rows = []
    for name, folder in entries:
        path = folder / 'metrics.json'
        metrics = read_json(path) if path.exists() else {}
        rows.append(dict(model=name, status='completed' if metrics else 'not_completed',
            topics=metrics.get('topics'), c_v=metrics.get('c_v'), u_mass=metrics.get('u_mass'),
            diversity=metrics.get('topic_diversity'),
            outlier_percent=100 * metrics['outlier_ratio'] if metrics.get('outlier_ratio') is not None else None,
            artifact_directory=str(folder)))
    return rows


def report(out):
    rows = comparison_rows(out)
    table(out / 'evaluation/model_comparison.csv', pd.DataFrame(rows))
    lines = ['# Model comparison on the frozen sodium-ion corpus', '',
        'All 13,702 Stage 1 documents are retained. Missing metrics mean the corresponding experiment has not completed.', '',
        '| Model | Status | Topics | C_v | U_Mass | Diversity | Outlier % |',
        '|---|---|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'N/A' if x is None else f'{x:.6f}' if isinstance(x, float) else str(x)
    for row in rows:
        lines.append('| ' + ' | '.join(fmt(row[k]) for k in ('model', 'status', 'topics', 'c_v', 'u_mass', 'diversity', 'outlier_percent')) + ' |')
    lines += ['', '## Method and evaluation', '',
        'LDA reuses K=10, seed42 if its saved hashes and dictionary are valid; otherwise one run uses the old Stage2 settings. '
        'BGE reuses the fitted 84-topic model and its normalized 13702 x 768 chunked embeddings. Neither BGE training nor its embedding/UMAP/HDBSCAN is repeated.', '',
        'BatteryBERT uses AutoTokenizer/AutoModel from batterydata/batterybert-cased, with pinned revision. '
        'Attention-mask mean pooling followed by L2 is used in both fine-tuning and inference. Full-document inference covers all abstract tokens with 256-token windows, overlap64; '
        'normalized chunk vectors are averaged and normalized. Title context may be capped to reserve abstract payload. '
        'Contrastive training and its proxy evaluation use single sequences capped at 128 tokens (as in the existing BGE training budget); this truncation differs from full-document chunking and is logged. '
        'One epoch updates only the final encoder block, AdamW lr2e-5, temperature0.05, microbatch2, accumulation2. '
        'Same-group pairs and title Jaccard>=0.8 are masked as false negatives. Frozen layers stay in evaluation mode. '
        'Validation Group MRR@10 selects only trained checkpoints; test is excluded from selection.', '',
        'BERTopic uses UMAP(15 neighbors, 5 components, min_dist0, cosine, seed42), HDBSCAN(30/10/eom), '
        'and c-TF-IDF(reduce_frequent_words=True) on the original frozen vocabulary/tokens. '
        'Reduction calls the installed BERTopic API with 10 plus its outlier slot; topic -1 never counts as a research topic. '
        'No labels are edited to manufacture ten topics. The original-to-reduced mapping and outlier mask are saved.', '',
        'C_v uses gensim indirect cosine coherence over Boolean sliding windows of 110 original Stage1 tokens (top10, processes1). '
        'U_Mass = mean over ordered keyword pairs i>j of log((D(w_i,w_j)+epsilon)/D(w_j)), computed by gensim on the unchanged Stage2 document BoW, epsilon=1e-12 in probability space. '
        'Diversity = number of unique top10 keywords / (10 x research topics). Outlier rate = count(topic=-1)/13702. '
        'Both coherence metrics exclude -1. Per-topic coherence and document sizes are exported.', '',
        'LDA distributions are variational posterior probabilities. BERTopic document-topic distributions are normalized cosine affinities of frozen BoW to c-TF-IDF topic representations; '
        'these are not calibrated probabilities. BERTopic representatives are the five assigned documents nearest the topic embedding centroid. '
        'Interpretive labels cite keyword and representative-title evidence and need expert review.', '',
        '## Actual execution state', '']
    state_path = out / 'run_status.json'
    if state_path.exists():
        lines += ['```json', json.dumps(read_json(state_path), ensure_ascii=False, indent=2), '```']
    for variant in ('pretrained', 'finetuned'):
        folder = out / ('batterybert_' + variant)
        metrics_path = folder / 'proxy_metrics.json'
        if metrics_path.exists():
            lines += ['', f'### BatteryBERT {variant} proxy', '', '```json',
                      json.dumps(read_json(metrics_path), ensure_ascii=False, indent=2), '```']
        marker = folder / 'training_success.json'
        if marker.exists():
            lines += ['', 'Fine-tuning was completed and verified: `' + str(marker) + '`.']
        elif variant == 'finetuned':
            lines += ['', 'Fine-tuning is not yet verified complete; no fine-tuned result is claimed.']
    done = [r for r in rows if r['status'] == 'completed']
    lines += ['', '## Interpretation and limitations', '',
        'No overall winner is selected from a single metric. Coherence, diversity, outlier coverage, topic balance and expert interpretation must be considered together. '
        'Topic counts alone do not establish quality. Proxy Title-to-Abstract retrieval checks embedding behavior only; it is not Search Top10 evaluation. '
        'The fixed Stage1 vocabulary can retain legacy placeholders or scientific formula tokenization issues; no token repair was applied. '
        'Topic stability across new seeds was not run to avoid additional configurations on the CPU budget; existing 84-topic or K20 stability is not attributed to these reduced models.', '']
    if len(done) >= 2:
        a, b = done[:2]
        lines += [f"Observed {b['model']} minus {a['model']}: C_v {b['c_v']-a['c_v']:+.6f}, "
                  f"U_Mass {b['u_mass']-a['u_mass']:+.6f}, diversity {b['diversity']-a['diversity']:+.6f}. "
                  'These differences are descriptive, not evidence of statistical significance.', '']
    lines += ['## Integration artifacts', '',
        'No Qdrant, Search V2/V3, dashboard or search evaluation code was changed. '
        'Use each model directory, ordered_doc_ids.json, document_embeddings.npy and inference/embedding metadata together. '
        'Only completed, verified branches are integration candidates.', '']
    for row in rows:
        lines.append(f"- {row['model']}: `{row['artifact_directory']}` ({row['status']})")
    (out / 'reports').mkdir(exist_ok=True)
    (out / 'reports/MODEL_COMPARISON_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Check report against the CSV representation, including missing values.
    reread = pd.read_csv(out / 'evaluation/model_comparison.csv', keep_default_na=False)
    assert reread.model.tolist() == [r['model'] for r in rows]
    content = (out / 'reports/MODEL_COMPARISON_REPORT.md').read_text(encoding='utf-8')
    for row in rows:
        assert '| ' + ' | '.join(fmt(row[k]) for k in ('model', 'status', 'topics', 'c_v', 'u_mass', 'diversity', 'outlier_percent')) + ' |' in content


def verify(out):
    count = verify_protected(out)
    from gensim.models import LdaModel
    from bertopic import BERTopic
    ids = read_json(out / 'ordered_doc_ids.json')
    checked = []
    for row in comparison_rows(out):
        folder = Path(row['artifact_directory'])
        if row['status'] != 'completed':
            continue
        assignment = pd.read_parquet(folder / 'topic_assignments.parquet')
        assert assignment.doc_id.tolist() == ids
        labels = assignment.topic_id.to_numpy()
        assert len(set(labels) - {-1}) == row['topics']
        assert pd.read_csv(folder / 'topic_summary.csv').documents.sum() == 13702
        reps = pd.read_csv(folder / 'representative_documents.csv')
        assert all(assignment.set_index('doc_id').loc[r.doc_id, 'topic_id'] == r.topic_id for r in reps.itertuples())
        counts = assignment[assignment.topic_id.ne(-1)].groupby('topic_id').size()
        assert reps.groupby('topic_id').size().to_dict() == {t: min(5, n) for t, n in counts.items()}
        model = LdaModel.load(str(folder / 'model.gensim')) if row['model'].startswith('LDA') else BERTopic.load(str(folder / 'model.pickle'))
        if not row['model'].startswith('LDA'):
            assert np.array_equal(model.topics_, labels)
        distribution = np.load(folder / 'document_topic_distribution.npy', mmap_mode='r')
        assert distribution.shape == (13702, row['topics']) and np.isfinite(distribution).all()
        manifest = read_json(folder / 'manifest.json')
        for name, sha in manifest['sha256'].items():
            assert digest(folder / name) == sha, name
        checked.append(row['model'])
        del model
    for variant in ('pretrained', 'finetuned'):
        folder = out / ('batterybert_' + variant)
        if (folder / 'embedding_success.json').exists():
            info = validate_vectors(folder / 'document_embeddings.npy', read_json(folder / 'ordered_doc_ids.json'), ids)
            assert info['sha256'] == read_json(folder / 'embedding_success.json')['sha256']
    report(out)
    write_json(out / 'verification.json', dict(status='PASS', protected_files_unchanged=count,
        checked_models=checked, all_models_completed=len(checked) == 4,
        note='PASS describes existing artifacts only; missing branches are not claimed complete'))
    log('Verification PASS: protected files unchanged; models reload; assignments/exports/report agree')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['preflight', 'lda', 'bge', 'topics_pretrained', 'topics_finetuned', 'report', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must stay inside the project outputs directory')
    if args.phase == 'bge':
        topics(out, 'bge')
    elif args.phase.startswith('topics_'):
        topics(out, args.phase.removeprefix('topics_'))
    else:
        globals()[args.phase](out)


if __name__ == '__main__':
    main()
