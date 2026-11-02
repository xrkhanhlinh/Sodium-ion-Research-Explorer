"""Add phase-certified comparisons and formulas to the generated report.

Reads real metrics only. A metrics file without a committed phase is pending,
not a completed experiment. This script never writes encoder/topic artifacts.
"""
import argparse
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.stage3_common import read_json, digest
from src.model_comparison_io import write_json, write_csv, replace_with_retry

ENTRIES = [
    ('LDA K=10', 'lda', 'lda_k10'),
    ('BERTopic + BGE', 'bge', 'bertopic_bge_k10'),
    ('BERTopic + BatteryBERT pretrained', 'topics_pretrained', 'bertopic_batterybert_k10/pretrained'),
    ('BERTopic + BatteryBERT fine-tuned', 'topics_finetuned', 'bertopic_batterybert_k10/finetuned'),
]


def fmt(value):
    return 'N/A' if value is None else f'{value:.6f}' if isinstance(value, float) else str(value)


def certified_rows(out):
    rows, balance, hashes = [], [], {}
    for name, phase, relative in ENTRIES:
        folder, marker = out / relative, out / '.phases' / (phase + '.json')
        row = dict(model=name, status='not_completed', topics=None, c_v=None, u_mass=None,
                   diversity=None, outlier_percent=None, artifact_directory=str(folder))
        metrics_path = folder / 'metrics.json'
        if marker.exists() and metrics_path.exists():
            phase_info = read_json(marker)
            for filename in ('metrics.json', 'topic_summary.csv', 'topic_assignments.parquet'):
                path = folder / filename
                expected = phase_info['hashes'].get(path.relative_to(out).as_posix())
                if not expected or digest(path) != expected:
                    raise ValueError('Report input does not match committed phase: ' + str(path))
                hashes[path.relative_to(out).as_posix()] = expected
            metrics = read_json(metrics_path)
            summary = pd.read_csv(folder / 'topic_summary.csv')
            assert int(summary.documents.sum()) == 13702
            actual_topics = int(summary.topic_id.ne(-1).sum())
            assert actual_topics == metrics['topics']
            assert np.isfinite([metrics[k] for k in ('c_v', 'u_mass', 'topic_diversity')]).all()
            row.update(status='completed', topics=int(metrics['topics']), c_v=metrics['c_v'],
                u_mass=metrics['u_mass'], diversity=metrics['topic_diversity'],
                outlier_percent=None if metrics['outlier_ratio'] is None else 100 * metrics['outlier_ratio'])
            research = summary[summary.topic_id.ne(-1)]
            balance.append(dict(model=name, smallest_topic=int(research.documents.min()),
                largest_topic=int(research.documents.max()),
                largest_topic_corpus_percent=float(research.documents.max() / 13702 * 100),
                largest_topic_inlier_percent=float(research.documents.max() / research.documents.sum() * 100),
                target_10_reached=actual_topics == 10))
        rows.append(row)
    return rows, balance, hashes


def finalize(out):
    out = out.resolve()
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Report output must stay inside project outputs/')
    rows, balance, hashes = certified_rows(out)
    write_csv(out / 'evaluation/model_comparison.csv', pd.DataFrame(rows))
    write_csv(out / 'evaluation/topic_balance.csv', pd.DataFrame(balance))
    report = out / 'reports/MODEL_COMPARISON_REPORT.md'
    content = report.read_text(encoding='utf-8')
    begin = content.index('| Model | Status | Topics |')
    end = content.index('\n\n', begin)
    header = '| Model | Status | Topics | C_v | U_Mass | Diversity | Outlier % |\n|---|---|---:|---:|---:|---:|---:|\n'
    row_text = ['| ' + ' | '.join(fmt(row[k]) for k in ('model', 'status', 'topics', 'c_v', 'u_mass', 'diversity', 'outlier_percent')) + ' |' for row in rows]
    content = content[:begin] + header + '\n'.join(row_text) + content[end:]
    content = content.replace('log((D(w_i,w_j)+epsilon)/D(w_j))', 'log((P(w_i,w_j)+1e-12)/P(w_j))')
    content = content.replace('epsilon=1e-12 in probability space.', 'P denotes document-occurrence probability and epsilon=1e-12.')
    heading = '\n## Phase-certified comparison and evaluation formulas\n'
    if heading in content:
        content = content.split(heading)[0]
    lines = [heading, '',
        'A completed topic row requires its success marker plus matching SHA-256 for metrics, assignments and topic summary. '
        'An uncommitted metrics file is not a completed result.', '',
        'For K research topics and m=10 keywords per topic:', '',
        '- NPMI(i,j) = log((p_ij + 1e-12)/(p_i p_j)) / -log(p_ij + 1e-12), using Boolean sliding windows of 110 Stage1 tokens.',
        '- C_v = mean over topics of mean_i cosine(c_i, sum_j c_j), with context vector c_i[j]=NPMI(i,j) (gensim one-set segmentation, gamma=1).',
        '- U_Mass = mean over topics of mean_(i>j) log((P(w_i,w_j)+1e-12)/P(w_j)), using the original Stage2 document BoW.',
        '- Diversity = unique top10 keywords / (10 K). Outlier percent = 100 count(topic=-1)/13702.',
        '- BERTopic affinity A_dt = cosine(BoW_d,cTFIDF_t) / sum_s cosine(BoW_d,cTFIDF_s); zero-denominator rows remain zero. This is not a posterior.', '',
        'Topic -1 is excluded from K and coherence. Coherence is a macro-average over research topics, using the same corpus/setup for every branch.', '']
    completed = {r['model']: r for r in rows if r['status'] == 'completed'}
    battery_names = [r[0] for r in ENTRIES[2:]]
    for name in battery_names:
        if name not in completed:
            lines += [f'{name}: topic experiment pending; no topic-quality score or fine-tuning gain is inferred from the proxy results.', '']
            continue
        b = completed[name]
        for reference in ('BERTopic + BGE', 'LDA K=10'):
            if reference not in completed:
                continue
            a = completed[reference]
            delta = f"{name} minus {reference}: C_v {b['c_v']-a['c_v']:+.6f}, U_Mass {b['u_mass']-a['u_mass']:+.6f}, diversity {b['diversity']-a['diversity']:+.6f}"
            if a['outlier_percent'] is not None and b['outlier_percent'] is not None:
                delta += f", outliers {b['outlier_percent']-a['outlier_percent']:+.4f} percentage points"
            lines += [delta + '. These descriptive differences do not establish an overall winner or statistical significance.', '']
    if all(name in completed for name in battery_names):
        before, after = [completed[name] for name in battery_names]
        lines += [f"BatteryBERT fine-tuned minus pretrained topic model: C_v {after['c_v']-before['c_v']:+.6f}, "
            f"U_Mass {after['u_mass']-before['u_mass']:+.6f}, diversity {after['diversity']-before['diversity']:+.6f}, "
            f"outliers {after['outlier_percent']-before['outlier_percent']:+.4f} percentage points. "
            'Both branches use the same frozen token reference and clustering settings.', '']
    for item in balance:
        lines += [f"{item['model']}: smallest topic {item['smallest_topic']}, largest topic {item['largest_topic']} "
            f"({item['largest_topic_corpus_percent']:.4f}% of the full corpus, {item['largest_topic_inlier_percent']:.4f}% of assigned inliers); "
            f"ten-topic target reached: {item['target_10_reached']}.", '']
    proxies = {}
    for variant in ('pretrained', 'finetuned'):
        path = out / ('batterybert_' + variant) / 'proxy_metrics.json'
        marker = out / '.phases' / ('proxy_' + variant + '.json')
        if path.exists() and marker.exists():
            expected = read_json(marker)['hashes'].get(path.relative_to(out).as_posix())
            assert digest(path) == expected
            proxies[variant] = read_json(path)
            phase_hashes = read_json(marker)['hashes']
            for split in ('validation', 'test'):
                expected_rows = proxies[variant][split]['queries']
                ids = pd.read_csv(path.parent / 'proxy' / (split + '_ordered_doc_ids.csv'))
                assert len(ids) == expected_rows and ids.doc_id.is_unique
                for field in ('titles', 'abstracts'):
                    array_path = path.parent / 'proxy' / (split + '_' + field + '.npy')
                    relative = array_path.relative_to(out).as_posix()
                    actual_sha = digest(array_path)
                    assert actual_sha == phase_hashes[relative]
                    vectors = np.load(array_path, mmap_mode='r')
                    assert vectors.shape == (expected_rows, 768)
                    assert np.isfinite(vectors).all()
                    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)
                    hashes[relative] = actual_sha
    if len(proxies) == 2:
        for split in ('validation', 'test'):
            a, b = [proxies[v][split]['group_mrr_at_10'] for v in ('pretrained', 'finetuned')]
            lines += [f'Embedding proxy {split}: Group MRR@10 pretrained={a:.6f}, fine-tuned={b:.6f}, delta={b-a:+.6f}. This measures held-out Title-to-Abstract matching only.', '']
    verification = out / 'verification.json'
    if verification.exists():
        result = read_json(verification)
        lines += [f"Artifact verification: {result['status']}; protected files unchanged: {result['protected_files_unchanged']}; "
            f"all four topic branches completed: {result['all_models_completed']}.", '']
    else:
        lines += ['Final artifact/checksum verification is pending.', '']
    budget = out / 'evaluation/batterybert_training_proxy_token_budget.csv'
    if budget.exists():
        budget_rows = pd.read_csv(budget)
        hashes[budget.relative_to(out).as_posix()] = digest(budget)
        lines += ['Training/proxy truncation audit (length includes [CLS]/[SEP]):', '']
        for row in budget_rows[budget_rows.field.eq('abstract')].itertuples():
            lines += [f'- {row.split}: {row.truncated_sequences}/{row.documents} abstracts '
                f'({100*row.truncated_share:.2f}%) exceed the 128-token training/proxy cap.']
        lines += ['', 'These proxy scores concern capped views and must not be described as full-abstract retrieval quality. '
            'Full-document topic embeddings use complete 256-token/overlap64 chunk coverage; their abstract tails are retained.', '']
    lines += ['BGE and BatteryBERT also differ in tokenization and token pooling, so cross-encoder differences do not isolate '
        'the effect of battery-domain pretraining. The within-BatteryBERT comparison controls its adapter and clustering configuration. '
        'Overlapping windows and repeated title context can change the relative weight of document passages.', '']
    lines += ['The exact float32 full-document prefix cache reuses only frozen blocks; the current last block is recomputed. '
        'Each cached chunk has SHA-256 protection and sampled full-forward parity. This is a CPU optimization, not a new model configuration.', '',
        '[Interpretive topic notes](INTERPRETATION_NOTES.md) · [Execution and recovery record](RUN_HISTORY.md)', '',
        'BatteryBERT checkpoint source: [batterydata/batterybert-cased](https://huggingface.co/batterydata/batterybert-cased). '
        'The downloaded revision and model-file SHA-256 values are stored in download_provenance.json.', '']
    content += '\n'.join(lines)
    with tempfile.NamedTemporaryFile(mode='w', dir=report.parent, suffix='.tmp',
                                     encoding='utf-8', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, report)
    for line in row_text:
        assert line in report.read_text(encoding='utf-8')
    reread = pd.read_csv(out / 'evaluation/model_comparison.csv', keep_default_na=False)
    assert reread.model.tolist() == [r['model'] for r in rows]
    assert reread.status.tolist() == [r['status'] for r in rows]
    code_files = ['src/model_comparison.py', 'src/batterybert.py', 'src/model_comparison_io.py',
                  'configs/model_comparison.yaml', 'run_model_comparison.py',
                  'scripts/download_batterybert.py', 'scripts/monitor_model_comparison.py',
                  'scripts/finalize_model_comparison_report.py', 'tests/test_model_comparison_v2.py']
    write_json(out / 'evaluation/report_consistency.json', dict(status='PASS',
        completed_topic_models=sum(r['status'] == 'completed' for r in rows),
        metrics_and_assignment_sha256=hashes, csv_sha256=digest(out / 'evaluation/model_comparison.csv'),
        report_sha256=digest(report), code_sha256={name: digest(ROOT / name) for name in code_files}))
    print('Report certified against committed phase metrics and topic counts', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/model_comparison_v2')
    finalize(parser.parse_args().output)
