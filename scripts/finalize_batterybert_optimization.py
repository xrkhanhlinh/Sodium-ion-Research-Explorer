"""Select refined words on validation, freeze once, then evaluate held-out test."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.optimize_batterybert import load, describe, passes
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from src.batterybert_topic_tools import distribution, sha, write_json


def run(out=None):
    out = Path(out) if out is not None else ROOT / 'outputs/batterybert_optimization_20261023_v1'
    cfg, main, docs, splits, vectors = load(out)
    refinement = out / 'representation_refinement'
    rows = []
    for p in sorted(refinement.glob('*/metrics.json')):
        rep = json.loads(p.read_text()); base = json.loads((out / 'experiments' / rep['base'] / 'metrics.json').read_text())
        combined = {**base, **rep}; checks = passes(combined, cfg['gates'])
        combined['technical_gate_pass'] = all(checks.values()); combined['gates'] = checks
        write_json(p.parent / 'combined_metrics.json', combined)
        rows.append(combined)
    chosen = max(rows, key=lambda r: (r['technical_gate_pass'], r['document_npmi']))
    selected = dict(candidate=chosen['base'], representation=chosen['candidate'], selected_on='validation only',
        technical_gate_pass=chosen['technical_gate_pass'], score=chosen['document_npmi'],
        frequency_power=chosen['frequency_power'], exclusivity_power=chosen['exclusivity_power'],
        gates=cfg['gates'], scientific_validation='Not independently validated')
    marker = out / 'refined_selection.json'
    if marker.exists() and json.loads(marker.read_text()) != selected:
        raise ValueError('Final selection is frozen; new candidates require a new experiment version')
    write_json(marker, selected)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows]).to_csv(
        out / 'refined_validation_comparison.csv', index=False)
    print('SELECTED', json.dumps(selected), flush=True)
    final_marker = out / 'final_test.json'
    if final_marker.exists():
        print('Retaining immutable final test'); return
    dest = out / 'experiments' / selected['candidate']
    model = joblib.load(dest / 'pipeline.joblib')
    keys = pd.read_csv(refinement / selected['representation'] / 'keywords.csv')
    labels = model.predict(vectors[model.encoder])
    train, test = splits == 'train', splits == 'test'
    m, _, _ = describe(labels[test], main[test], keys, docs[test].tolist())
    alt = KMeans(n_clusters=10, n_init=cfg['n_init'], max_iter=cfg['max_iter'], random_state=cfg['final_seed']).fit(
        model.geometry.transform(vectors[model.encoder][train]))
    m['stability_ari'] = float(adjusted_rand_score(labels[test], alt.predict(model.geometry.transform(vectors[model.encoder][test]))))
    checks = passes(m, cfg['gates'])
    m.update(gates=checks, technical_gate_pass=all(checks.values()), selection=selected,
        scientific_validation='Pending independent expert assessment; gates are engineering diagnostics',
        fit='10921 training documents only; no test refit; 52 title-only documents are inference only')
    summary, source, full = distribution(labels, main.source.tolist())
    summary.to_csv(out / 'topic_distribution.csv', index=False)
    source.to_csv(out / 'topic_source_distribution.csv')
    assigned = main[['doc_id', 'source', 'year']].copy(); assigned['topic_id'] = labels
    assigned.to_csv(out / 'topic_assignments.csv', index=False)
    keys.to_csv(out / 'topic_keywords.csv', index=False)
    joblib.dump(model, out / 'topic_pipeline.joblib')
    write_json(out / 'full_corpus_diagnostics.json', full)
    distances = model.cluster.transform(model.geometry.transform(vectors[model.encoder]))
    records = []
    for topic in range(10):
        ids = np.flatnonzero(labels == topic)
        central = ids[np.argsort(distances[ids, topic], kind='stable')[:3]]
        random = np.random.default_rng(20261023 + topic).choice(ids, min(3, len(ids)), replace=False)
        for role, indices in [('central', central), ('random', random)]:
            for i in indices:
                row = main.iloc[i]
                records.append(dict(topic_id=topic, sample_role=role, doc_id=row.doc_id, title=row.title,
                    abstract=row.abstract, source=row.source, year=row.year, split=splits[i],
                    distance_to_center=float(distances[i, topic])))
    pd.DataFrame(records).to_csv(out / 'representative_documents.csv', index=False)
    reloaded = joblib.load(out / 'topic_pipeline.joblib')
    np.testing.assert_array_equal(reloaded.predict(vectors[model.encoder][::97]), labels[::97])
    baseline_rows = []
    for name in ('pretrained_raw', 'finetuned_raw', 'bge_raw'):
        folder = out / 'experiments' / name
        baseline = joblib.load(folder / 'pipeline.joblib')
        baseline_keys = pd.read_csv(folder / 'keywords.csv')
        y = baseline.predict(vectors[baseline.encoder][test])
        metric, _, _ = describe(y, main[test], baseline_keys, docs[test].tolist())
        baseline_rows.append(dict(model=name, **{k: v for k, v in metric.items() if not isinstance(v, list)}))
    baseline_rows.append(dict(model='BatteryBERT_optimized', **{k: v for k, v in m.items() if not isinstance(v, (list, dict))}))
    pd.DataFrame(baseline_rows).to_csv(out / 'inductive_test_comparison.csv', index=False)
    write_json(out / 'export_validation.json', dict(reload_probes=len(labels[::97]), reload_predictions_match=True,
        all_documents_assigned=len(labels), unique_ids=main.doc_id.is_unique,
        pipeline_sha256=sha(out / 'topic_pipeline.joblib'), encoder=model.encoder,
        encoder_vectors_sha256=sha(ROOT / 'outputs/final_system/batterybert_finetuned/document_embeddings.npy')))
    write_json(final_marker, m)
    print('FINAL TEST', json.dumps(m), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='outputs/batterybert_optimization_20261023_v1')
    run(ROOT / parser.parse_args().output)
