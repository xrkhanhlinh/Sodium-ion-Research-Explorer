"""Train/validation selection followed by a single frozen final test.

No transformer, Qdrant write, source data modification or automatic fallback.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from src.batterybert_optimization import Geometry, TopicPipeline, keyword_model, document_npmi
from src.batterybert_topic_tools import sha, write_json, distribution, repair_phrase_placeholders

OLD = ROOT / 'outputs/model_comparison_v2'


def load(out):
    import yaml
    config = json.loads((ROOT / 'configs/batterybert_optimization.json').read_text())
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet')
    tokens = pd.read_parquet(ROOT / 'data/processed/lda_tokens.parquet')
    splits = pd.read_parquet(ROOT / 'data/stage3/split_assignments.parquet').set_index('doc_id').loc[main.doc_id]
    if tokens.doc_id.tolist() != main.doc_id.tolist() or not main.doc_id.is_unique:
        raise ValueError('Corpus order mismatch')
    if splits.groupby('split_group_id').split.nunique().max() != 1:
        raise ValueError('Group leakage')
    phrases = yaml.safe_load((ROOT / 'configs/config.yaml').read_text(encoding='utf8'))['preprocessing']['domain_phrases']
    texts, _ = repair_phrase_placeholders(tokens.tokens.tolist(), phrases)
    docs = np.asarray([' '.join(t) for t in texts], dtype=object)
    paths = dict(pretrained=OLD / 'batterybert_pretrained/document_embeddings.npy',
        finetuned=ROOT / 'outputs/final_system/batterybert_finetuned/document_embeddings.npy',
        bge=OLD / 'bertopic_bge_k10/document_embeddings.npy')
    orders = dict(pretrained=OLD / 'batterybert_pretrained/ordered_doc_ids.json',
        finetuned=ROOT / 'outputs/final_system/batterybert_finetuned/ordered_doc_ids.json',
        bge=OLD / 'bertopic_bge_k10/ordered_doc_ids.json')
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in [*paths.values(), *orders.values(),
        ROOT / 'data/processed/main_corpus.parquet', ROOT / 'data/processed/lda_tokens.parquet',
        ROOT / 'data/stage3/split_assignments.parquet', ROOT / 'configs/batterybert_optimization.json',
        ROOT / 'src/batterybert_optimization.py', Path(__file__)]}
    out.mkdir(parents=True, exist_ok=True)
    marker = out / 'inputs.json'
    if marker.exists() and json.loads(marker.read_text()) != fingerprints:
        raise ValueError('Inputs/code changed: use a fresh output directory')
    if not marker.exists() and any(out.iterdir()):
        raise ValueError('Refusing unrelated populated output')
    write_json(marker, fingerprints)
    vectors = {}
    for encoder, path in paths.items():
        if json.loads(orders[encoder].read_text()) != main.doc_id.tolist():
            raise ValueError(f'{encoder}: embedding order mismatch')
        x = np.load(path, mmap_mode='r')
        if x.shape != (len(main), 768) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-5):
            raise ValueError(f'{encoder}: invalid vectors')
        vectors[encoder] = x
    write_json(out / 'protocol.json', dict(config=config, counts=splits.split.value_counts().to_dict(),
        labels='No independent scientific topic labels; unsupervised development diagnostics only',
        selection='Passing BatteryBERT candidates first, then validation document NPMI; otherwise diagnostic best with explicit rejection',
        source_projection='One global rank-one train-only projection, no source required at inference; may remove real source-associated science',
        stopping='Fixed hypothesis family, one frozen final gate; stop on passing gate or disclose unresolved limits. No tuning against test.'))
    return config, main, docs, splits.split.to_numpy(), vectors


def describe(labels, main, keywords, docs):
    summary, sources, metrics = distribution(labels, main.source.to_numpy())
    metrics['smallest_topic_share'] = float(summary.corpus_share.min())
    metrics['keyword_diversity'] = keywords.token.nunique() / len(keywords)
    metrics.update(document_npmi(docs, keywords))
    return metrics, summary, sources


def passes(m, gate):
    checks = dict(largest=m['largest_topic_share'] <= gate['largest_topic_share_max'],
        top2=m['top2_topic_share'] <= gate['top2_topic_share_max'],
        smallest=m['smallest_topic_share'] >= gate['smallest_topic_share_min'],
        source=m['source_cramers_v'] <= gate['source_cramers_v_max'],
        stability=m['stability_ari'] >= gate['stability_ari_min'],
        diversity=m['keyword_diversity'] >= gate['keyword_diversity_min'],
        support=m['supported_keyword_fraction'] >= gate['keyword_support_min'],
        npmi=m['document_npmi'] >= gate['validation_npmi_min'], topics=m['topics'] == 10)
    return checks


def search(out):
    cfg, main, docs, splits, vectors = load(out)
    train, val = splits == 'train', splits == 'validation'
    rows = []
    for encoder, mode in cfg['candidates']:
        name = encoder + '_' + mode
        dest = out / 'experiments' / name; dest.mkdir(parents=True, exist_ok=True)
        if not (dest / 'metrics.json').exists():
            start = time.monotonic()
            geometry = Geometry(mode).fit(vectors[encoder][train], main.source.to_numpy()[train])
            xtrain = geometry.transform(vectors[encoder][train])
            xval = geometry.transform(vectors[encoder][val])
            cluster = KMeans(n_clusters=10, n_init=cfg['n_init'], max_iter=cfg['max_iter'], random_state=cfg['seed']).fit(xtrain)
            alternate = KMeans(n_clusters=10, n_init=cfg['n_init'], max_iter=cfg['max_iter'], random_state=cfg['stability_seed']).fit(xtrain)
            pipeline = TopicPipeline(geometry, cluster, encoder, main.doc_id[train].tolist())
            _, keyword_rows = keyword_model(docs[train].tolist(), cluster.labels_)
            keywords = pd.DataFrame(keyword_rows)
            predicted = cluster.predict(xval)
            metrics, summary, sources = describe(predicted, main[val], keywords, docs[val].tolist())
            metrics.update(stability_ari=float(adjusted_rand_score(predicted, alternate.predict(xval))),
                           encoder=encoder, geometry=mode, seconds=time.monotonic() - start)
            checks = passes(metrics, cfg['gates']); metrics.update(gates=checks, technical_gate_pass=all(checks.values()))
            keywords.to_csv(dest / 'keywords.csv', index=False)
            summary.to_csv(dest / 'validation_distribution.csv', index=False)
            sources.to_csv(dest / 'validation_sources.csv')
            joblib.dump(pipeline, dest / 'pipeline.joblib')
            np.save(dest / 'validation_predictions.npy', predicted)
            write_json(dest / 'metrics.json', metrics)
        m = json.loads((dest / 'metrics.json').read_text())
        rows.append(dict(candidate=name, **{k: v for k, v in m.items() if not isinstance(v, (list, dict))}))
        pd.DataFrame(rows).to_csv(out / 'validation_comparison.csv', index=False)
        print(name, json.dumps({k: m[k] for k in ('document_npmi', 'source_cramers_v', 'stability_ari', 'technical_gate_pass')}), flush=True)
    battery = [r for r in rows if r['encoder'] != 'bge']
    chosen = max(battery, key=lambda r: (r['technical_gate_pass'], r['document_npmi']))
    write_json(out / 'selection.json', dict(candidate=chosen['candidate'],
        technical_gate_pass=chosen['technical_gate_pass'], selected_on='validation only',
        independent_scientific_validation=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='outputs/batterybert_optimization_20261023_v1')
    parser.add_argument('--phase', choices=['search', 'final', 'all'], default='all')
    args = parser.parse_args(); output = ROOT / args.output
    if args.phase in ('search', 'all'): search(output)
    if args.phase in ('final', 'all'):
        from scripts.refine_batterybert_topics import run as refine
        from scripts.finalize_batterybert_optimization import run as export
        refine(output)
        export(output)
