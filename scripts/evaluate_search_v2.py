"""Evaluate v2 plus historical baselines using strict human provenance guards.

With no labels this produces a real, validated PENDING evaluation package.
Never replaces missing judgments with model scores or generated human labels.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.research_evaluation import evaluate_frames, annotation_agreement
from src.search_v2 import ROOT, METHODS, write_json


def run(experiment, output, labels=None):
    experiment, output = Path(experiment), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty evaluation folder')
    queries = pd.read_csv(experiment / 'queries.csv')
    pool = pd.read_csv(experiment / 'shared_pool_UNVERIFIED.csv', keep_default_na=False)
    new = pd.read_csv(experiment / 'top5_comparison.csv', keep_default_na=False)
    old = pd.read_csv(ROOT / 'outputs/research_validation/retrieval/top5_seven_methods.csv', keep_default_na=False)
    legacy_methods = ['SE1', 'SE2', 'SE3', 'BM25']
    methods = METHODS + legacy_methods
    # The three dense baselines have identical saved Top5 in old and new runs;
    # use their v2 names once, not duplicate them as additional independent models.
    rankings = pd.concat([new, old[old.method.isin(legacy_methods)]], ignore_index=True)
    primary_ids = queries.loc[queries.is_edge_case.eq(False), 'query_id']
    primary = rankings[rankings.query_id.isin(primary_ids)]
    counts = primary.groupby(['query_id', 'method']).size().to_dict()
    runs = pd.DataFrame([dict(query_id=qid, method=m, results_count=counts.get((qid, m), 0))
                         for qid in primary_ids for m in methods])
    label_frame = pd.read_csv(labels, keep_default_na=False) if labels else None
    result = evaluate_frames(rankings, queries, pool, label_frame, methods=tuple(methods), run_manifest=runs)
    output.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output / 'top5_fourteen_methods.csv', index=False)
    runs.to_csv(output / 'run_manifest.csv', index=False)
    for name in ['per_query', 'summary', 'confidence_intervals', 'paired_deltas']:
        frame = result.pop(name)
        if len(frame):
            frame.to_csv(output / f'{name}.csv', index=False)
    result.update(methods=methods, labels_path=str(labels) if labels else None,
                  development_queries=['Q01', 'Q08', 'Q16', 'Q19', 'Q23'],
                  limitation='23 authored queries; five already used for development. No untouched independent benchmark claim. CIs do not include training-seed uncertainty; pairwise intervals are exploratory and unadjusted.',
                  provenance='Software validates declarations, not human expertise or independence.',
                  gain='2^grade-1', relevant_threshold=2,
                  idcg='Shared pool of old/new candidates; not whole-corpus recall',
                  old_run_source='outputs/research_validation/retrieval/top5_seven_methods.csv',
                  new_run_source=str(experiment / 'top5_comparison.csv'))
    write_json(output / 'scope.json', result)
    left = pd.read_csv(experiment / 'annotator_A_NEW.csv', keep_default_na=False)
    right = pd.read_csv(experiment / 'annotator_B_NEW.csv', keep_default_na=False)
    write_json(output / 'raw_annotation_agreement.json', annotation_agreement(left, right))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'outputs/search_upgrade_20261018')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    run(args.experiment, args.output, args.labels)
