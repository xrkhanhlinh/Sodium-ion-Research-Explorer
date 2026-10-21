"""Isolated, resumable full-document BGE/BERTopic experiment.

No Stage 1/2, baseline, search, or fine-tuning write paths are used here.
Run through run_bertopic_chunked.py to release neural memory before clustering.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

import argparse
import bisect
import hashlib
import json
import re
import time
import numpy as np
import pandas as pd
from .stage3_common import digest, signature, read_json, write_json

SETTINGS = dict(chunk_size=256, overlap=64, batch_size=2, threads=4,
                dimension=768, seed=42, document_pooling='mean of normalized BGE CLS chunk embeddings, then L2',
                title_context='tokenizer pair; title capped only to reserve at least 65 abstract tokens',
                umap=dict(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
                hdbscan=dict(min_cluster_size=30, min_samples=10, cluster_selection_method='eom'),
                nr_topics=None, reassign_outliers=False)
MODEL = ROOT / 'models/bert_finetuned'


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def table(out, name, frame):
    frame.to_csv(out / name, index=False, encoding='utf-8-sig')


def windows(length, capacity, sentence_ends=(), overlap=64):
    """Half-open original-token offsets; exact overlap, full coverage, progress."""
    if capacity <= overlap:
        raise ValueError('Chunk payload capacity must exceed overlap')
    if length == 0:
        return []
    boundaries = sorted(set(sentence_ends))
    result = []
    start = 0
    while start < length:
        end = min(length, start + capacity)
        if end < length:
            at = bisect.bisect_right(boundaries, end) - 1
            if at >= 0 and boundaries[at] > start + overlap:
                end = boundaries[at]
        result.append((start, end))
        if end == length:
            break
        start = end - overlap
    return result


def coverage(length, spans):
    """Validate union coverage without a per-token bitmap."""
    covered = 0
    previous = 0
    for start, end in spans:
        assert 0 <= start < end <= length
        assert start <= previous, 'Uncovered token interval'
        covered += max(0, end - previous)
        previous = max(previous, end)
    assert covered == length and previous == length
    return covered


def bert_features(tokenizer, first, second=None):
    """BERT template from this checkpoint, without decode/re-tokenize token loss.

    Transformers 5 removed prepare_for_model; the preflight checks this template
    against the actual tokenizer's single/pair postprocessor.
    """
    inputs = [tokenizer.cls_token_id] + list(first) + [tokenizer.sep_token_id]
    types = [0] * len(inputs)
    if second is not None:
        inputs += list(second) + [tokenizer.sep_token_id]
        types += [1] * (len(second) + 1)
    return dict(input_ids=inputs, token_type_ids=types)


def chunk_document(tokenizer, title, abstract):
    title = '' if pd.isna(title) else str(title)
    abstract = '' if pd.isna(abstract) else str(abstract)
    title_ids = tokenizer(title, add_special_tokens=False, truncation=False)['input_ids']
    encoded = tokenizer(abstract, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    rows = []
    if ids:
        special = tokenizer.num_special_tokens_to_add(pair=True)
        retained = title_ids[:256 - special - 65]
        capacity = 256 - special - len(retained)
        token_ends = [end for start, end in encoded['offset_mapping']]
        # Sentence boundaries are a preference only; scientific strings are never edited.
        sentence_ends = [bisect.bisect_right(token_ends, match.start() + 1)
                         for match in re.finditer(r'[.!?](?:\s+|$)', abstract)]
        spans = windows(len(ids), capacity, sentence_ends)
        for start, end in spans:
            inputs = bert_features(tokenizer, retained, ids[start:end])
            rows.append(dict(token_start=start, token_end=end, offset_field='abstract',
                             char_start=encoded['offset_mapping'][start][0],
                             char_end=encoded['offset_mapping'][end - 1][1],
                             input_ids=inputs['input_ids'], token_type_ids=inputs.get('token_type_ids', []),
                             title_tokens_retained=len(retained), sentence_boundary=end in sentence_ends))
        covered = coverage(len(ids), spans)
    else:
        assert title_ids, 'Both title and abstract are empty; refusing fabricated content'
        spans = windows(len(title_ids), 256 - tokenizer.num_special_tokens_to_add(pair=False))
        for start, end in spans:
            inputs = bert_features(tokenizer, title_ids[start:end])
            rows.append(dict(token_start=start, token_end=end, offset_field='title', char_start=None,
                             char_end=None, input_ids=inputs['input_ids'],
                             token_type_ids=inputs.get('token_type_ids', []),
                             title_tokens_retained=end - start, sentence_boundary=False))
        coverage(len(title_ids), spans)
        covered = 0
    assert all(len(row['input_ids']) <= 256 for row in rows)
    for chunk_id, row in enumerate(rows):
        row.update(chunk_id=chunk_id, chunk_count=len(rows), input_token_count=len(row['input_ids']))
    return rows, dict(abstract_tokens=len(ids), covered_abstract_tokens=covered, title_tokens=len(title_ids),
                      title_truncated=bool(ids and len(retained) < len(title_ids)),
                      title_only=not bool(ids), chunk_count=len(rows))


def protected_paths():
    folders = ['data/interim', 'data/processed', 'data/stage2', 'models/lda', 'models/bert_finetuned',
               'outputs/tables/stage2', 'outputs/reports/stage2', 'outputs/figures/stage2',
               'outputs/reports/stage3', 'outputs/tables/stage3', 'models/bertopic']
    paths = [p for folder in folders for p in (ROOT / folder).rglob('*') if p.is_file()]
    paths += [ROOT / name for name in ['configs/config.yaml', 'configs/stage2.yaml', 'configs/stage3.yaml']]
    return sorted(set(paths))


def verify_inputs(out):
    from .artifact_history import verify_snapshot
    manifest = read_json(out / 'input_manifest.json')
    verify_snapshot(manifest['protected_hashes'])
    return manifest


def prepare(out):
    from transformers import AutoTokenizer
    import importlib.metadata
    import platform
    assert read_json(MODEL / 'training_success.json')['real_gradient_updates']
    if not (out / 'input_manifest.json').exists():
        assert not any(out.iterdir()), 'Use a new empty output directory; existing results must be preserved'
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in protected_paths()}
        write_json(out / 'input_manifest.json', dict(protected_hashes=hashes, settings=SETTINGS,
                    python=platform.python_version(), versions={name:importlib.metadata.version(name) for name in
                        ['torch','sentence-transformers','transformers','tokenizers','bertopic','umap-learn','hdbscan','gensim','scikit-learn']},
                    baseline_metrics=read_json(ROOT / 'outputs/reports/stage3/bertopic_metrics.json'),
                    baseline_embedding=read_json(ROOT / 'outputs/reports/stage3/embedding_manifest.json')))
    manifest = verify_inputs(out)
    assert manifest['settings'] == SETTINGS
    if (out / 'chunking_success.json').exists():
        for name, sha in read_json(out / 'chunking_success.json')['hashes'].items():
            assert digest(out / name) == sha
        log('Reusing validated chunk metadata')
        return
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet')
    tokens = pd.read_parquet(ROOT / 'data/processed/lda_tokens.parquet')
    assert len(main) == 13702 and main.doc_id.is_unique
    assert main.doc_id.tolist() == tokens.doc_id.tolist()
    assert all(len(t) == n for t, n in zip(tokens.tokens, tokens.token_count))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    assert tokenizer.is_fast and tokenizer.backend_tokenizer.model.__class__.__name__ == 'WordPiece'
    for first,second in [('Sodium battery',None),('Sodium battery','Hard carbon')]:
        a=tokenizer(first,add_special_tokens=False)['input_ids']
        b=tokenizer(second,add_special_tokens=False)['input_ids'] if second is not None else None
        actual=tokenizer(first,second,add_special_tokens=True)
        assert bert_features(tokenizer,a,b)['input_ids']==actual['input_ids']
        assert bert_features(tokenizer,a,b)['token_type_ids']==actual['token_type_ids']
    all_chunks, documents = [], []
    for index, row in enumerate(main.itertuples()):
        chunks, stats = chunk_document(tokenizer, row.title, row.abstract)
        documents.append(dict(doc_index=index, doc_id=row.doc_id, chunk_start=len(all_chunks), **stats))
        for chunk in chunks:
            all_chunks.append(dict(global_chunk_id=len(all_chunks), doc_index=index, doc_id=row.doc_id, **chunk))
        if (index + 1) % 2000 == 0:
            log(f'CHUNK {index + 1}/13702 documents; {len(all_chunks)} chunks')
    frame = pd.DataFrame(all_chunks)
    stats = pd.DataFrame(documents)
    frame.to_parquet(out / 'chunks.parquet', index=False)
    stats.to_parquet(out / 'document_chunk_counts.parquet', index=False)
    table(out, 'chunk_metadata.csv', frame.drop(columns=['input_ids', 'token_type_ids']))
    table(out, 'document_chunk_counts.csv', stats)
    write_json(out / 'ordered_doc_ids.json', main.doc_id.tolist())
    summary = dict(documents=len(main), chunks=len(frame), abstract_tokens=int(stats.abstract_tokens.sum()),
                   covered_abstract_tokens=int(stats.covered_abstract_tokens.sum()),
                   abstract_token_coverage=float(stats.covered_abstract_tokens.sum() / stats.abstract_tokens.sum()),
                   title_only_documents=int(stats.title_only.sum()), title_context_truncated_documents=int(stats.title_truncated.sum()),
                   maximum_chunk_tokens=int(frame.input_token_count.max()),
                   offsets='0-based half-open offsets in original abstract WordPiece tokens; title-only rows use title offsets',
                   ordered_doc_ids_sha256=signature(main.doc_id.tolist()))
    assert summary['abstract_token_coverage'] == 1.0
    summary['hashes'] = {name: digest(out / name) for name in
                         ['chunks.parquet', 'document_chunk_counts.parquet', 'ordered_doc_ids.json']}
    write_json(out / 'chunking_success.json', summary)
    log(f'Chunking complete: {summary}')


def encode_features(model, tokenizer, records):
    import torch
    features = []
    for row in records:
        item = dict(input_ids=[int(value) for value in row['input_ids']], attention_mask=[1] * len(row['input_ids']))
        if len(row['token_type_ids']):
            item['token_type_ids'] = [int(value) for value in row['token_type_ids']]
        features.append(item)
    padded = tokenizer.pad(features, padding=True, return_tensors='pt')
    assert padded['input_ids'].shape[1] <= 256
    with torch.inference_mode():
        result = model(dict(padded))['sentence_embedding']
        result = torch.nn.functional.normalize(result, p=2, dim=1)
    arr = result.cpu().numpy().astype('float32')
    assert arr.shape == (len(records), 768) and np.isfinite(arr).all()
    return arr


def preflight(out):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(MODEL), device='cpu', local_files_only=True)
    model.max_seq_length = 256
    model.eval()
    assert model.get_embedding_dimension() == 768 and model[1].pooling_mode == 'cls'
    title = 'Sodium-ion battery Na3V2(PO4)3'
    chunks, stats = chunk_document(model.tokenizer, title, 'Hard carbon stores sodium. ' * 150)
    assert len(chunks) > 1 and stats['covered_abstract_tokens'] == stats['abstract_tokens']
    a = encode_features(model, model.tokenizer, chunks[:2])
    b = np.vstack([encode_features(model, model.tokenizer, [row]) for row in chunks[:2]])
    assert np.allclose(a, b, atol=2e-5), 'Padding/batch size changes model embeddings'
    # Direct token features must agree with the checkpoint's public encode path.
    title_chunks, _ = chunk_document(model.tokenizer, title, '')
    direct = encode_features(model, model.tokenizer, title_chunks)
    reference = model.encode([title], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
    assert np.allclose(direct, reference, atol=2e-5)
    saved_records=pd.read_parquet(out / 'chunks.parquet').head(2).to_dict('records')
    saved_vectors=encode_features(model,model.tokenizer,saved_records)
    assert saved_vectors.shape==(2,768) and np.isfinite(saved_vectors).all()
    from .stage3_bertopic_model import FrozenTokensBERTopic
    frozen = FrozenTokensBERTopic(embedding_model=None)
    chemical = 'Na+ Na3V2(PO4)3 sodium_ion hard_carbon P2'
    assert frozen._preprocess_text([chemical]) == [chemical]
    write_json(out / 'preflight.json', dict(status='PASS', batch2_vs_batch1_max_error=float(np.max(np.abs(a-b))),
                public_encode_max_error=float(np.max(np.abs(direct-reference))), threads=torch.get_num_threads(),
                device='cpu', dimension=768, max_seq_length=256, query_instruction=False,
                checks=['actual checkpoint forward', 'batch/padding invariance', 'title-only public encode parity',
                        'full long-abstract coverage', 'persisted Parquet token IDs', 'frozen chemical tokens']))
    log('Technical preflight PASS; no configuration selection performed')


def embed(out):
    import gc
    import inspect
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    chunks = pd.read_parquet(out / 'chunks.parquet')
    docs = pd.read_parquet(out / 'document_chunk_counts.parquet')
    identity = signature(dict(chunking=read_json(out / 'chunking_success.json'), settings=SETTINGS,
                               encoder_implementation=inspect.getsource(encode_features),
                               weights=digest(MODEL / 'model.safetensors')))
    checkpoint = out / 'embedding_progress.json'
    path = out / 'chunk_embeddings.npy'
    state = read_json(checkpoint) if checkpoint.exists() else dict(fingerprint=identity, completed_chunks=0,
                total_chunks=len(chunks), batch_size=2, embedding_seconds=0.)
    assert state['fingerprint'] == identity
    if state['completed_chunks']:
        arr = np.lib.format.open_memmap(path, mode='r+')
    else:
        arr = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(chunks), 768))
    assert arr.shape == (len(chunks), 768)
    cursor = state['completed_chunks']
    if cursor < len(chunks):
        model = SentenceTransformer(str(MODEL), device='cpu', local_files_only=True)
        model.max_seq_length = 256
        model.eval()
        assert model.get_embedding_dimension() == 768
        batch_size = state['batch_size']
        tick = time.perf_counter()
        committed = cursor
        while cursor < len(chunks):
            end = min(len(chunks), cursor + batch_size)
            try:
                value = encode_features(model, model.tokenizer, chunks.iloc[cursor:end].to_dict('records'))
            except (MemoryError, RuntimeError) as exc:
                is_memory = isinstance(exc, MemoryError) or any(s in str(exc).lower() for s in
                            ['out of memory', 'not enough memory', "can't allocate memory", 'cannot allocate memory'])
                if not is_memory or batch_size == 1:
                    raise
                batch_size = 1
                gc.collect()
                log('RAM allocation failure: reducing batch size from 2 to 1; scientific settings unchanged')
                continue
            arr[cursor:end] = value
            cursor = end
            if cursor - committed >= 64 or cursor == len(chunks):
                arr.flush()
                state.update(completed_chunks=cursor, batch_size=batch_size,
                             embedding_seconds=state['embedding_seconds'] + time.perf_counter() - tick)
                write_json(checkpoint, state)
                rate = cursor / max(1., state['embedding_seconds'])
                log(f'EMBED {cursor}/{len(chunks)} chunks; {state["embedding_seconds"]:.1f}s; '
                    f'{rate:.3f} chunks/s; ETA {(len(chunks)-cursor)/max(rate,1e-9)/60:.1f} min')
                tick = time.perf_counter()
                committed = cursor
        del model
        gc.collect()
    assert np.isfinite(arr).all()
    assert np.allclose(np.linalg.norm(arr, axis=1), 1., atol=2e-5)
    document_vectors = np.lib.format.open_memmap(out / 'document_embeddings.npy', mode='w+', dtype='float32', shape=(13702, 768))
    for row in docs.itertuples():
        vector = np.asarray(arr[row.chunk_start:row.chunk_start + row.chunk_count]).mean(axis=0)
        norm = np.linalg.norm(vector)
        assert norm > 0 and np.isfinite(norm)
        document_vectors[row.doc_index] = vector / norm
    document_vectors.flush()
    assert document_vectors.shape == (13702, 768) and np.isfinite(document_vectors).all()
    assert np.allclose(np.linalg.norm(document_vectors, axis=1), 1., atol=1e-5)
    write_json(out / 'embedding_success.json', dict(status='PASS', shape=list(document_vectors.shape),
                chunks=len(chunks), seconds=state['embedding_seconds'], batch_size=state['batch_size'],
                ordered_doc_ids_sha256=signature(docs.doc_id.tolist()), normalization='L2',
                pooling=SETTINGS['document_pooling'], max_seq_length=256, query_instruction=False,
                model_sha256=digest(MODEL / 'model.safetensors'),
                document_embeddings_sha256=digest(out / 'document_embeddings.npy'),
                chunk_embeddings_sha256=digest(path)))
    log('All chunk and document embeddings validated and saved')


def fit(out):
    from joblib import parallel_backend
    from gensim import corpora
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP
    from .stage3_bertopic_model import FrozenTokensBERTopic
    if (out / 'fit_success.json').exists():
        assert digest(out / 'model.pickle') == read_json(out / 'fit_success.json')['model_sha256']
        log('Reusing completed chunked BERTopic fit')
        return
    tokens = pd.read_parquet(ROOT / 'data/processed/lda_tokens.parquet')
    assert tokens.doc_id.tolist() == read_json(out / 'ordered_doc_ids.json')
    docs = [' '.join(t) for t in tokens.tokens]
    embeddings = np.load(out / 'document_embeddings.npy', mmap_mode='r')
    assert embeddings.shape == (13702, 768) and len(docs) == 13702
    dictionary = corpora.Dictionary.load(str(ROOT / 'models/lda/dictionary.gensim'))
    vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, vocabulary=dictionary.token2id)
    model = FrozenTokensBERTopic(embedding_model=None,
                umap_model=UMAP(**SETTINGS['umap'], n_jobs=1, low_memory=True),
                hdbscan_model=HDBSCAN(**SETTINGS['hdbscan'], metric='euclidean', prediction_data=True, core_dist_n_jobs=1),
                vectorizer_model=vectorizer, ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                top_n_words=10, nr_topics=None, calculate_probabilities=False, verbose=True)
    start = time.perf_counter()
    with parallel_backend('sequential', n_jobs=1):
        labels, strengths = model.fit_transform(docs, embeddings=embeddings)
    seconds = time.perf_counter() - start
    assert np.isfinite(model.c_tf_idf_.data).all() and np.isfinite(model.ctfidf_model._idf_diag.data).all()
    assert model.vectorizer_model.vocabulary_ == dictionary.token2id
    model.save(str(out / 'model.pickle'), serialization='pickle')
    model.save(str(out / 'model_compact'), serialization='safetensors', save_ctfidf=True)
    np.save(out / 'assignments.npy', np.asarray(labels, dtype='int32'))
    np.save(out / 'membership_strength.npy', np.asarray(strengths))
    np.save(out / 'umap_5d.npy', model.umap_model.embedding_)
    write_json(out / 'fit_success.json', dict(seconds=seconds, topics=len(set(labels)-{-1}),
                outliers=int(np.sum(np.asarray(labels)==-1)), outlier_ratio=float(np.mean(np.asarray(labels)==-1)),
                model_sha256=digest(out / 'model.pickle'), settings=SETTINGS,
                representation='All frozen Stage 1 tokens; unchanged vocabulary and preprocessing; no truncation',
                inference='Supply external chunked document embeddings to transform'))
    log(f'BERTopic fit saved: {seconds:.1f}s')


def evaluate(out):
    from bertopic import BERTopic
    from gensim import corpora
    from gensim.models import CoherenceModel
    model = BERTopic.load(str(out / 'model.pickle'))
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet')
    tokens = pd.read_parquet(ROOT / 'data/processed/lda_tokens.parquet')
    assert main.doc_id.tolist() == tokens.doc_id.tolist() == read_json(out / 'ordered_doc_ids.json')
    texts = [list(t) for t in tokens.tokens]
    labels = np.load(out / 'assignments.npy')
    strengths = np.load(out / 'membership_strength.npy')
    assert np.isfinite(strengths).all()
    assert np.array_equal(labels, model.topics_)
    assignments = main[['doc_id', 'source', 'year', 'split_group_id']].copy()
    assignments['topic_id'] = labels
    assignments['membership_strength'] = strengths
    assignments.to_parquet(out / 'topic_assignments.parquet', index=False)
    table(out, 'topic_assignments.csv', assignments)
    summary, keywords, representatives, audits = [], [], [], []
    dictionary = corpora.Dictionary.load(str(ROOT / 'models/lda/dictionary.gensim'))
    vocabulary = dictionary.token2id
    topic_words = {}
    for topic in sorted(set(labels)):
        topic = int(topic)
        indices = np.flatnonzero(labels == topic)
        words = [(w, float(p)) for w, p in model.get_topic(topic)[:10] if w and p > 0]
        if topic != -1:
            assert len(words) == 10, f'Topic {topic} does not have ten positive keywords'
            topic_words[topic] = [w for w, p in words]
        summary.append(dict(topic_id=topic, documents=len(indices), document_share=len(indices)/len(main),
                            top10_keywords='; '.join(w for w,p in words), is_outlier=topic == -1))
        for rank, (word, weight) in enumerate(words, 1):
            keywords.append(dict(topic_id=topic, rank=rank, token=word, ctfidf_weight=weight))
            reasons = []
            if word not in vocabulary: reasons.append('not_in_frozen_vocabulary')
            if re.search(r'(?i)(placeholder|unknown|<unk>|\[unk\]|^nan$|^none$|^null$|^formula[_\d]|^chem[_\d])', word):
                reasons.append('possible_placeholder')
            if '\ufffd' in word or not any(c.isalnum() for c in word): reasons.append('unusual_characters')
            if len(word) > 60: reasons.append('unusually_long_keyword')
            if len(word) == 1: reasons.append('single_character_review')
            for reason in reasons: audits.append(dict(topic_id=topic, token=word, reason=reason))
        if topic == -1: continue
        order = indices[np.argsort(-strengths[indices], kind='stable')][:3]
        for rank, index in enumerate(order, 1):
            row = main.iloc[index]
            representatives.append(dict(topic_id=topic, rank=rank, doc_id=row.doc_id, source=row.source,
                title=row.title, abstract=row.abstract, membership_strength=float(strengths[index]),
                selection_method='Highest HDBSCAN membership strength within assigned topic, stable input-order ties; identical to baseline'))
    table(out, 'topic_summary.csv', pd.DataFrame(summary))
    table(out, 'topic_keywords.csv', pd.DataFrame(keywords))
    table(out, 'representative_documents.csv', pd.DataFrame(representatives))
    table(out, 'keyword_audit.csv', pd.DataFrame(audits, columns=['topic_id','token','reason']))
    # Prove each token string survived the representation path verbatim. This detects
    # newly transformed formulas; pre-existing corpus/tokenization errors need experts.
    assert all(row['token'] in vocabulary for row in keywords)
    start = time.perf_counter()
    if topic_words:
        cv = CoherenceModel(topics=list(topic_words.values()), texts=texts, dictionary=dictionary,
                            coherence='c_v', window_size=110, topn=10, processes=1).get_coherence_per_topic()
        bow = corpora.MmCorpus(str(ROOT / 'data/stage2/corpus_bow.mm'))
        um = CoherenceModel(topics=list(topic_words.values()), corpus=bow, dictionary=dictionary,
                            coherence='u_mass', topn=10, processes=1).get_coherence_per_topic()
        diversity = len(set(w for words in topic_words.values() for w in words)) / (10 * len(topic_words))
    else:
        cv, um, diversity = [], [], None
    table(out, 'coherence_per_topic.csv', pd.DataFrame([dict(topic_id=t,c_v=float(c),u_mass=float(u))
                                                     for t,c,u in zip(topic_words,cv,um)],
                                                    columns=['topic_id','c_v','u_mass']))
    fit_info = read_json(out / 'fit_success.json')
    metrics = dict(topics=len(topic_words), c_v=float(np.mean(cv)) if len(cv) else None,
                   u_mass=float(np.mean(um)) if len(um) else None, topic_diversity=diversity,
                   outliers=fit_info['outliers'], outlier_ratio=fit_info['outlier_ratio'],
                   coherence_seconds=time.perf_counter()-start, keyword_audit_flags=len(audits),
                   coherence_corpus='Identical full frozen Stage 1 tokens; top10; C_v window110; U_Mass Stage2 corpus_bow; outlier excluded',
                   ctfidf_reduce_frequent_words=True, expert_evaluation=False)
    assert all(v is None or np.isfinite(v) for v in [metrics['c_v'], metrics['u_mass'], diversity])
    write_json(out / 'metrics.json', metrics)
    baseline = read_json(out / 'input_manifest.json')['baseline_metrics']
    rows = [dict(model='BERTopic baseline (128 tokens)', **{key:baseline[key] for key in
            ['topics','outliers','outlier_ratio','c_v','u_mass','topic_diversity']}),
            dict(model='BERTopic full-document chunking (256 / overlap64)', **{key:metrics[key] for key in
            ['topics','outliers','outlier_ratio','c_v','u_mass','topic_diversity']})]
    table(out, 'comparison.csv', pd.DataFrame(rows))
    table(out, 'metric_deltas.csv', pd.DataFrame([dict(metric=k, baseline=baseline[k], chunked=metrics[k],
          delta=metrics[k]-baseline[k] if metrics[k] is not None else None)
          for k in ['topics','outliers','outlier_ratio','c_v','u_mass','topic_diversity']]))
    log(f'Evaluation saved: {metrics}')


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from umap import UMAP
    from joblib import parallel_backend
    folder = out / 'figures'
    folder.mkdir(exist_ok=True)
    topics = pd.read_csv(out / 'topic_summary.csv').sort_values('documents', ascending=False)
    keywords = pd.read_csv(out / 'topic_keywords.csv', keep_default_na=False)
    fig, ax = plt.subplots(figsize=(14,5))
    ax.bar(np.arange(len(topics)), topics.documents, color=['#999999' if t==-1 else '#217c9c' for t in topics.topic_id])
    step = max(1,len(topics)//35)
    ax.set(xticks=np.arange(len(topics))[::step], xticklabels=topics.topic_id.iloc[::step],
           xlabel='Topic ID (sorted by document count; -1 = outliers)', ylabel='Documents', title='Full-document BERTopic: all documents')
    fig.tight_layout(); fig.savefig(folder / 'topic_distribution.png', dpi=200); plt.close(fig)
    real = topics[topics.topic_id.ne(-1)]
    # Export every topic's top ten, in readable pages, plus a largest-topic overview.
    pages = [real.iloc[i:i+12] for i in range(0,len(real),12)]
    for page, subset in enumerate(pages,1):
        rows = max(1, int(np.ceil(len(subset)/3)))
        fig, axes = plt.subplots(rows,3,figsize=(17,3.5*rows),squeeze=False)
        for ax in axes.flat: ax.set_visible(False)
        for ax,row in zip(axes.flat,subset.itertuples()):
            ax.set_visible(True)
            sub = keywords[keywords.topic_id.eq(row.topic_id)].iloc[::-1]
            ax.barh(sub.token,sub.ctfidf_weight,color='#217c9c')
            ax.set(title=f'Topic {row.topic_id}: {row.documents:,} documents',xlabel='c-TF-IDF weight')
            ax.tick_params(axis='y',labelsize=8)
        fig.tight_layout(); fig.savefig(folder / f'top10_keywords_{page:02d}.png',dpi=180); plt.close(fig)
    path = out / 'umap_2d.npy'
    embeddings = np.load(out / 'document_embeddings.npy',mmap_mode='r')
    if path.exists(): points = np.load(path)
    else:
        with parallel_backend('sequential',n_jobs=1):
            points = UMAP(n_neighbors=15,n_components=2,min_dist=.1,metric='cosine',random_state=42,
                          n_jobs=1,low_memory=True).fit_transform(embeddings)
        np.save(path,points)
    labels = np.load(out / 'assignments.npy')
    scatter = pd.DataFrame(dict(doc_id=read_json(out / 'ordered_doc_ids.json'),topic_id=labels,x=points[:,0],y=points[:,1]))
    table(out,'umap_2d.csv',scatter)
    fig, ax = plt.subplots(figsize=(12,8))
    outliers = labels == -1
    ax.scatter(points[outliers,0],points[outliers,1],s=3,c='#b8b8b8',alpha=.35,label='Outliers (-1)')
    sc = ax.scatter(points[~outliers,0],points[~outliers,1],c=labels[~outliers],s=4,cmap='turbo',alpha=.7)
    if (~outliers).any(): fig.colorbar(sc,ax=ax,label='Topic ID')
    ax.legend(); ax.set(xlabel='UMAP 1',ylabel='UMAP 2',title='Chunked BGE document embeddings (separate 2-D visualization)')
    fig.tight_layout(); fig.savefig(folder / 'umap_2d.png',dpi=200); plt.close(fig)
    write_json(out / 'figures.json',dict(files=[p.name for p in sorted(folder.glob('*.png'))],
            projection='Separate 2D UMAP for display only; clustering uses the fixed 5D UMAP',
            keyword_pages_cover_all_non_outlier_topics=True))
    log('All figures exported')


def audit(out):
    """Trace suspicious keywords to frozen Stage 1 without changing any tokens."""
    flags = pd.read_csv(out / 'keyword_audit.csv',keep_default_na=False)
    tokens = pd.read_parquet(ROOT / 'data/processed/lda_tokens.parquet')
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet')
    labels = np.load(out / 'assignments.npy')
    assert tokens.doc_id.tolist()==main.doc_id.tolist()==read_json(out / 'ordered_doc_ids.json')
    rows=[]
    for flag in flags.itertuples():
        occurrences=np.array([sum(word==flag.token for word in text) for text in tokens.tokens])
        indices=np.flatnonzero(occurrences)
        assert len(indices)>0
        topic_indices=indices[labels[indices]==flag.topic_id]
        examples=[]
        for index in topic_indices[:3]:
            text=list(tokens.tokens.iloc[index]); at=text.index(flag.token)
            examples.append(dict(doc_id=main.doc_id.iloc[index],title=main.title.iloc[index],
                frozen_token_context=' '.join(text[max(0,at-4):at+5])))
        rows.append(dict(topic_id=int(flag.topic_id),token=flag.token,reason=flag.reason,
            preexisting_in_frozen_stage1=True,stage1_documents=len(indices),stage1_occurrences=int(occurrences.sum()),
            assigned_topic_documents=len(topic_indices),examples=json.dumps(examples,ensure_ascii=False),
            disposition='Flag for expert review; preserve frozen data/vocabulary; no automatic substitution'))
    table(out,'keyword_audit_provenance.csv',pd.DataFrame(rows))
    placeholders=[row for row in rows if row['reason']=='possible_placeholder']
    write_json(out / 'keyword_audit_details.json',dict(flags=len(rows),placeholder_keywords=placeholders,
        all_flagged_tokens_preexist_stage1=True,
        conclusion=f'{len(placeholders)} inherited compound phrase placeholders remain in top keywords; no new keyword transformation. Single characters require contextual review.',
        chemical_check='All keyword strings match the frozen vocabulary exactly; chemical validity and pre-existing formula truncation require expert review.'))
    log(f'Keyword provenance verified against Stage 1: {len(placeholders)} placeholder keywords; frozen inputs unchanged')


def verify(out):
    from bertopic import BERTopic
    from gensim import corpora
    manifest = verify_inputs(out)
    assert manifest['settings'] == SETTINGS
    ids = read_json(out / 'ordered_doc_ids.json')
    chunks = pd.read_parquet(out / 'chunks.parquet')
    docs = pd.read_parquet(out / 'document_chunk_counts.parquet')
    assignments = pd.read_parquet(out / 'topic_assignments.parquet')
    assert len(ids)==13702 and len(set(ids))==13702
    assert ids == docs.doc_id.tolist() == assignments.doc_id.tolist()
    assert chunks.global_chunk_id.tolist() == list(range(len(chunks)))
    chunk_embeddings = np.load(out / 'chunk_embeddings.npy',mmap_mode='r')
    embeddings = np.load(out / 'document_embeddings.npy',mmap_mode='r')
    assert chunk_embeddings.shape == (len(chunks),768) and embeddings.shape == (13702,768)
    for array in [chunk_embeddings,embeddings]:
        assert np.isfinite(array).all() and np.allclose(np.linalg.norm(array,axis=1),1,atol=2e-5)
    for row in docs.itertuples():
        sub = chunks.iloc[row.chunk_start:row.chunk_start+row.chunk_count]
        assert sub.doc_id.eq(row.doc_id).all() and sub.chunk_id.tolist()==list(range(row.chunk_count))
        length = row.title_tokens if row.title_only else row.abstract_tokens
        coverage(length,list(zip(sub.token_start,sub.token_end)))
        assert sub.input_token_count.le(256).all()
        assert all(a-b==64 for a,b in zip(sub.token_end.iloc[:-1],sub.token_start.iloc[1:]))
        vector = chunk_embeddings[row.chunk_start:row.chunk_start+row.chunk_count].mean(axis=0)
        vector /= np.linalg.norm(vector)
        assert np.allclose(vector,embeddings[row.doc_index],atol=1e-6)
    model = BERTopic.load(str(out / 'model.pickle'))
    assert np.array_equal(model.topics_,assignments.topic_id)
    assert model.vectorizer_model.vocabulary_ == corpora.Dictionary.load(str(ROOT / 'models/lda/dictionary.gensim')).token2id
    assert all(model.umap_model.get_params()[k]==v for k,v in SETTINGS['umap'].items())
    assert all(model.hdbscan_model.get_params()[k]==v for k,v in SETTINGS['hdbscan'].items())
    assert model.nr_topics is None and model.calculate_probabilities is False
    assert np.isfinite(model.c_tf_idf_.data).all()
    metrics = read_json(out / 'metrics.json')
    for name, sha in read_json(out / 'chunking_success.json')['hashes'].items():
        assert digest(out / name)==sha
    embedding_info=read_json(out / 'embedding_success.json')
    assert digest(out / 'document_embeddings.npy')==embedding_info['document_embeddings_sha256']
    assert digest(out / 'chunk_embeddings.npy')==embedding_info['chunk_embeddings_sha256']
    assert digest(out / 'model.pickle')==read_json(out / 'fit_success.json')['model_sha256']
    summary = pd.read_csv(out / 'topic_summary.csv')
    reps = pd.read_csv(out / 'representative_documents.csv')
    assert summary.documents.sum()==13702
    assert summary.topic_id.ne(-1).sum()==metrics['topics']
    assert assignments.topic_id.eq(-1).sum()==metrics['outliers']
    assert set(reps.topic_id)==set(summary.loc[summary.topic_id.ne(-1),'topic_id'])
    assert reps.groupby('topic_id').size().eq(3).all()
    assert all(assignments.set_index('doc_id').loc[r.doc_id,'topic_id']==r.topic_id for r in reps.itertuples())
    required = ['model.pickle','model_compact/config.json','document_embeddings.npy','chunk_embeddings.npy',
                'chunks.parquet','chunk_metadata.csv','document_chunk_counts.csv','topic_assignments.parquet',
                'topic_keywords.csv','representative_documents.csv','comparison.csv','metrics.json',
                'keyword_audit.csv','keyword_audit_provenance.csv','keyword_audit_details.json',
                'coherence_per_topic.csv','umap_2d.csv','figures/topic_distribution.png',
                'figures/umap_2d.png','figures/top10_keywords_01.png','preflight.json']
    assert all((out / name).is_file() and (out / name).stat().st_size>0 for name in required)
    assert read_json(out / 'preflight.json')['status']=='PASS'
    assert read_json(out / 'keyword_audit_details.json')['all_flagged_tokens_preexist_stage1']
    assert read_json(out / 'keyword_audit_details.json')['flags']==metrics['keyword_audit_flags']
    write_json(out / 'artifact_verification.json',dict(status='PASS',documents=13702,chunks=len(chunks),
               protected_files_verified=len(manifest['protected_hashes']),
               verified=['ID order','all token intervals/overlaps','all chunk lengths','all pooled vectors',
                         'finite unit vectors','model reload','vocabulary identity','fixed scientific configuration',
                         'topic counts','3 assigned representatives per topic','required artifacts'],
               artifact_hashes={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()
                                and p.suffix not in ['.log'] and p.name not in ['artifact_verification.json','run_status.json','REPORT.md']}))
    report(out)
    write_json(out / 'run_status.json',dict(status='COMPLETED',documents=13702,chunks=len(chunks),
               artifact_verification='PASS',protected_inputs='UNCHANGED'))
    log('COMPLETED: full pipeline, metrics, model reload, artifacts and protected inputs verified')


def report(out):
    chunking = read_json(out / 'chunking_success.json')
    embedding = read_json(out / 'embedding_success.json')
    fit_info = read_json(out / 'fit_success.json')
    metrics = read_json(out / 'metrics.json')
    audit_info=read_json(out / 'keyword_audit_details.json')
    placeholder_summary='; '.join(f"`{row['token']}` (topic {row['topic_id']})" for row in audit_info['placeholder_keywords'])
    comparison = pd.read_csv(out / 'comparison.csv')
    header = '| Model | Topics | Outliers | Outlier ratio | C_v | U_Mass | Diversity Top-10 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    lines = [f'| {r.model} | {r.topics} | {r.outliers} | {r.outlier_ratio:.6f} | {r.c_v:.6f} | {r.u_mass:.6f} | {r.topic_diversity:.6f} |'
             for r in comparison.itertuples()]
    content = f'''# BERTopic full-document chunking — COMPLETED

## Kết quả nghiệm thu

- Tài liệu: {chunking['documents']:,}; chunks: {chunking['chunks']:,}.
- Tokens Abstract: {chunking['covered_abstract_tokens']:,}/{chunking['abstract_tokens']:,}; độ bao phủ {chunking['abstract_token_coverage']:.2%}.
- Title-only: {chunking['title_only_documents']}; Title rút gọn làm ngữ cảnh: {chunking['title_context_truncated_documents']}.
- Embeddings: 13.702 × 768, float32, L2; thời gian inference chunk {embedding['seconds']:.2f} giây ({embedding['seconds']/60:.2f} phút).
- Fit BERTopic: {fit_info['seconds']:.2f} giây; {metrics['topics']} topics; {metrics['outliers']} outliers ({metrics['outlier_ratio']:.2%}).
- C_v: {metrics['c_v']}; U_Mass: {metrics['u_mass']}; Diversity Top-10: {metrics['topic_diversity']}.
- Kiểm tra artifacts và hashes inputs bảo vệ: PASS (`artifact_verification.json`).

## So sánh baseline đã lưu (không chạy lại)

{header}{chr(10).join(lines)}

C_v tăng {metrics['c_v']-comparison.c_v.iloc[0]:.6f}; U_Mass tăng {metrics['u_mass']-comparison.u_mass.iloc[0]:.6f} (ít âm hơn); Diversity tăng {metrics['topic_diversity']-comparison.topic_diversity.iloc[0]:.6f}. Outliers tăng {metrics['outliers']-int(comparison.outliers.iloc[0])} tài liệu, tương đương {(metrics['outlier_ratio']-comparison.outlier_ratio.iloc[0])*100:.3f} điểm phần trăm. Đây là đánh đổi giữa metrics, chưa đủ để kết luận mô hình tốt hơn tổng thể.

Nguồn baseline: `outputs/reports/stage3/bertopic_metrics.json`, giới hạn 128 tokens được xác minh từ embedding_manifest.json. Snapshot và SHA-256 trong input_manifest.json.

## Phương pháp

Checkpoint giữ nguyên `models/bert_finetuned/`, BGE CLS + Normalize theo modules gốc; không fine-tune, không query instruction. Abstract được tokenize nguyên bản bằng WordPiece của checkpoint, ưu tiên dấu kết thúc câu, overlap chính xác 64 token. Mỗi input gồm [CLS] Title [SEP] Abstract-window [SEP], tối đa 256 token. Title quá dài được giới hạn để dành tối thiểu 65 token Abstract và bảo đảm cửa sổ tiến lên; số trường hợp được ghi ở trên. Title-only được bao phủ bằng cửa sổ Title, không tạo nội dung. Mean của chunk embeddings đã L2, sau đó L2 cho mỗi tài liệu. Chunk embeddings, IDs, offsets và checkpoint tiến độ được lưu riêng.

BERTopic nhận đúng một embedding mỗi document. FrozenTokensBERTopic dùng toàn bộ tokens Stage 1 và nguyên vocabulary LDA; không cắt tokens c-TF-IDF. UMAP 15/5/0/cosine/42; HDBSCAN 30/10/eom. nr_topics=None, không gán lại outliers. CPU threads=4, batch khởi đầu=2, chỉ giảm=1 nếu thiếu RAM. Chỉ một cấu hình được chạy.

Metrics dùng chính cách tính baseline: C_v window110/top10/full Stage1 tokens, U_Mass Stage2 corpus_bow, Diversity unique keywords/(10×topics), bỏ topic -1 khi tính metrics. Đại diện là ba tài liệu có HDBSCAN membership strength cao nhất trong topic, tie theo thứ tự corpus, giống baseline.

## Hạn chế và kiểm tra keywords

Chưa có đánh giá chuyên gia. Không kết luận mô hình tốt hơn chỉ từ thay đổi embeddings hoặc số topics; các metrics có thể đánh đổi và không chứng minh tính đúng đắn ngữ nghĩa. Keyword audit có {metrics['keyword_audit_flags']} cờ cần xem trong keyword_audit.csv: {len(audit_info['placeholder_keywords'])} keywords chứa placeholder ({placeholder_summary}), cùng các ký tự đơn cần xem theo ngữ cảnh. Đã đối chiếu trực tiếp: tất cả cờ đều tồn tại từ tokens Stage 1, không do chunking hay c-TF-IDF mới tạo ra. Xem số lần xuất hiện, doc_id và ngữ cảnh trong keyword_audit_provenance.csv. Không tự thay thế keywords hoặc sửa dữ liệu đóng băng. Mọi keyword được kiểm chứng thuộc nguyên vocabulary Stage 1, preprocessing không sửa công thức; sai sót đã tồn tại trong dữ liệu gốc, công thức bị rút gọn từ trước và ý nghĩa hóa học vẫn cần đánh giá chuyên gia. Overlap và Title lặp lại làm tăng trọng số ngữ cảnh; mean pooling không loại trừ sự lặp này. Title context bị rút gọn nếu vượt ngân sách; toàn bộ Abstract vẫn được bao phủ. UMAP 2D chỉ dùng minh họa, tách khỏi UMAP 5D để clustering. Thời gian inference là tổng các đoạn đã commit, không gồm load model, tokenization hoặc phần phải làm lại sau gián đoạn.

## Artifacts

Tất cả outputs nằm tại `{out}`. Model đầy đủ `model.pickle`; bản compact `model_compact/`. Khi inference model đầy đủ, cần truyền embeddings tạo bằng cùng chunking vào transform.

'''
    content += '\n'.join(f'- `{p.relative_to(out).as_posix()}`' for p in sorted(out.rglob('*')) if p.is_file())
    (out / 'REPORT.md').write_text(content+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','preflight','embed','fit','evaluate','audit','figures','verify'])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to((ROOT / 'outputs').resolve()), 'Outputs must stay inside the project outputs directory'
    out.mkdir(parents=True,exist_ok=True)
    globals()[args.phase](out)


if __name__ == '__main__':
    main()
