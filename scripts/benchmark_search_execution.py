"""Bounded CPU execution trials; never a relevance-parameter optimizer.

Frozen query splits, grid and admission rules are recorded in the run protocol.
Only threads and reranker batch size vary. No training, cache reuse, or relabeling.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLITS = {'dev': ['Q01', 'Q16', 'Q19'], 'gate': ['Q02', 'Q04', 'Q07'],
          'final': ['Q03', 'Q05', 'Q06'], 'baseline-gate': ['Q02', 'Q04', 'Q07']}
BASELINE = (4, 4)
GRID = [(threads, batch) for threads in (1, 2, 4) for batch in (4, 8, 16)]


def main():
    global ROOT
    import numpy as np
    import pandas as pd
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--phase', choices=SPLITS, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'src/search_v2.py')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/search_v2.json')
    parser.add_argument('--candidate', help='threads:batch, for coordinator gate/final only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.project_root.resolve()
    if args.output.exists():
        raise ValueError('Use a new output directory; trials are never overwritten')
    if args.phase in ('gate', 'final'):
        if not args.candidate:
            parser.error('Gate/final require a selected --candidate threads:batch')
        candidate = tuple(map(int, args.candidate.split(':')))
        if candidate not in GRID:
            parser.error('Candidate must belong to the frozen grid')
        settings = list(dict.fromkeys([BASELINE, candidate]))
    elif args.phase == 'dev':
        settings = GRID
    else:
        settings = [BASELINE]
    args.output.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location('execution_candidate', args.source.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    # Isolated mutable runtime index; scientific artifacts remain read-only.
    module.CACHE = args.output / 'runtime_cache'
    module.CACHE.mkdir()
    for name in ('lexical.pkl', 'lexical_manifest.json'):
        original = ROOT / 'data/search_v2' / name
        if original.exists():
            shutil.copyfile(original, module.CACHE / name)
    queries = pd.read_parquet(ROOT / 'data/stage4/queries.parquet').set_index('query_id')['query']
    query_ids = SPLITS[args.phase]
    saved = pd.read_csv(ROOT / 'outputs/search_upgrade_20261018/top5_comparison.csv')
    expected = {qid: saved[saved.query_id.eq(qid) & saved.method.eq('RERANK_PRETRAINED')]
                .sort_values('rank').doc_id.tolist() for qid in query_ids}
    module.write_json(args.output / 'protocol.json', {
        'phase': args.phase, 'queries': query_ids, 'settings': settings,
        'baseline': BASELINE, 'repetitions': 2, 'shuffle_seed': 81073,
        'source_sha256': module.digest(args.source), 'evaluator_sha256': module.digest(__file__),
        'config_sha256': module.digest(args.config), 'config': json.loads(args.config.read_text(encoding='utf-8')),
        'query_file_sha256': module.digest(ROOT / 'data/stage4/queries.parquet'),
        'candidate': args.candidate, 'scope': 'Warm serial CPU execution; no relevance claims',
        'guard': 'Top5 identical to saved baseline; scores within 1e-4 of same-phase baseline',
    })
    rows, outputs = [], []
    engine = module.SearchEngineV2(config_path=args.config, use_query_cache=False)
    try:
        def configure(setting):
            threads, batch = setting
            engine.config['threads'] = threads
            engine.config['reranker_batch_size'] = batch
            torch.set_num_threads(threads)

        # Exclude initialization and one request per setting from warm timings.
        for setting in settings:
            configure(setting)
            engine.search(queries[query_ids[0]], 'RERANK_PRETRAINED')
        rng = random.Random(81073)
        trials = [(qid, repeat, setting) for repeat in range(2)
                  for qid in query_ids for setting in settings]
        rng.shuffle(trials)
        for qid, repeat, setting in trials:
            configure(setting)
            begin = time.perf_counter()
            response = engine.search(queries[qid], 'RERANK_PRETRAINED')
            elapsed = time.perf_counter() - begin
            key = f'{setting[0]}:{setting[1]}'
            ids = [r['doc_id'] for r in response['results']]
            row = dict(query_id=qid, repeat=repeat, setting=key, seconds=elapsed,
                       same_saved_top5=ids == expected[qid], query_cache=False, pair_cache=False)
            rows.append(row)
            outputs.append(dict(query_id=qid, repeat=repeat, setting=key,
                                results=[dict(doc_id=r['doc_id'], score=r['score']) for r in response['results']]))
            # Preserve partial evidence if interrupted.
            module.write_json(args.output / 'raw.json', dict(rows=rows, outputs=outputs))
            print(args.phase, qid, repeat, key, round(elapsed, 3), flush=True)
    finally:
        engine.close()
    baseline_key = f'{BASELINE[0]}:{BASELINE[1]}'
    reference_times = {(r['query_id'], r['repeat']): r['seconds'] for r in rows if r['setting'] == baseline_key}
    reference_scores = {(r['query_id'], r['repeat']): [d['score'] for d in r['results']]
                        for r in outputs if r['setting'] == baseline_key}
    summaries = []
    for setting in settings:
        key = f'{setting[0]}:{setting[1]}'
        selected = [row for row in rows if row['setting'] == key]
        ratios = [r['seconds'] / reference_times[(r['query_id'], r['repeat'])] for r in selected]
        per_query = defaultdict(list)
        for row, ratio in zip(selected, ratios):
            per_query[row['query_id']].append(ratio)
        checks = [len(record['results']) == len(reference_scores[(record['query_id'], record['repeat'])]) and np.allclose([d['score'] for d in record['results']],
                             reference_scores[(record['query_id'], record['repeat'])],
                             atol=1e-4, rtol=0) for record in outputs if record['setting'] == key]
        ratio = statistics.median(ratios)
        guards = all(r['same_saved_top5'] for r in selected) and all(checks)
        worst_query = max(statistics.median(values) for values in per_query.values())
        summaries.append(dict(setting=key, requests=len(selected), median_seconds=statistics.median(r['seconds'] for r in selected),
                              median_paired_ratio=ratio, worst_query_median_ratio=worst_query,
                              ranking_score_guards_pass=guards,
                              admission_pass=guards and ratio <= 0.90 and worst_query <= 1.10))
    summaries.sort(key=lambda r: (not r['ranking_score_guards_pass'], r['median_paired_ratio'], r['setting']))
    module.write_json(args.output / 'summary.json', dict(phase=args.phase, candidates=summaries,
                      best_observed=summaries[0], scope='Grid-limited CPU execution evidence; not relevance quality'))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
