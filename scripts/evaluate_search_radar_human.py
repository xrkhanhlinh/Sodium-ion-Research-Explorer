"""Import verified pilot qrels into a new evaluation version, never the old evaluator."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import json
import sys
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.final_system_evaluation import validate_labels
from src.final_system_artifacts import sha,read_json
from src.search_radar_retrieval import OUT,BRANCHES
from src.search_radar_core import METRICS,evaluate_ranking,paired_bootstrap
from src.model_comparison_io import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--split',choices=['dev','heldout','unassigned'],required=True)
    parser.add_argument('--run-dir',type=Path,help='Optional separate custom pilot directory exported by the app')
    parser.add_argument('--frozen-config',type=Path)
    parser.add_argument('--second-rater',type=Path)
    args=parser.parse_args()
    folder=args.run_dir.resolve() if args.run_dir else OUT/'pilot'
    pool=pd.read_csv(folder/'candidate_annotation_pool.csv',keep_default_na=False)
    ranks=pd.read_csv(folder/'rankings_top10.csv');candidates=pd.read_csv(folder/'candidates.csv')
    queries=pd.read_csv(folder/'queries.csv');ids=set(queries[queries.split.eq(args.split)].query_id)
    # Require a real review reference for query intent, as proposed families are not expert-verified.
    raw=pd.read_csv(args.labels,keep_default_na=False)
    raw=raw[raw.query_id.isin(ids)]
    labels=validate_labels(raw,pool[pool.query_id.isin(ids)],max_grade=2)
    if labels.empty:raise ValueError('No verified independent labels for the selected split')
    for field in ('reviewed_at','rationale','query_review_reference'):
        if field not in labels or labels[field].str.strip().eq('').any():raise ValueError('Missing '+field)
    for value in labels.reviewed_at:datetime.fromisoformat(value.replace('Z','+00:00'))
    frozen=None
    if args.split=='heldout':
        if args.frozen_config is None:raise ValueError('Held-out evaluation requires a development-selected frozen config')
        frozen=read_json(args.frozen_config)
        if frozen.get('status')!='ACCEPTED_INDEPENDENT_DEVELOPMENT_GATE':raise ValueError('Development gate not accepted')
        if frozen.get('heldout_labels_used',True):raise ValueError('Config selection must exclude held-out labels')
        if frozen.get('dev_qrels_sha256') is None or frozen.get('method') not in set(ranks.method):raise ValueError('Invalid frozen development selection')
        if not frozen.get('quality_gate_passed') or not frozen.get('isolated_cost_gate_passed'):raise ValueError('Quality and isolated-cost gates required')
        frozen_at=datetime.fromisoformat(frozen['frozen_at'].replace('Z','+00:00'))
        if any(datetime.fromisoformat(v.replace('Z','+00:00'))<=frozen_at for v in labels.reviewed_at):raise ValueError('Held-out labels must be reviewed after config freeze')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    target=OUT/'human_evaluations'/stamp;target.mkdir(parents=True,exist_ok=False)
    rows=[]
    for (qid,method,depth),g in ranks[ranks.query_id.isin(ids)].groupby(['query_id','method','depth']):
        if frozen and (method not in ('A0',frozen['method']) or depth!=frozen['depth']):continue
        q=labels[labels.query_id.eq(qid)]
        names=list(BRANCHES.get(method,BRANCHES['A3']))
        if method.startswith('A5_'):names.append('chunk_'+method.removeprefix('A5_'))
        if method=='A5':names.append('chunk')
        metrics=evaluate_ranking(g.sort_values('rank').doc_id,dict(zip(q.doc_id,q.relevance_grade)),
            candidates[candidates.query_id.eq(qid)&candidates.depth.eq(depth)&candidates.branch.isin(names)].doc_id)
        rows.append(dict(query_id=qid,method=method,depth=depth,**metrics))
    per=pd.DataFrame(rows);per.to_csv(target/'per_query.csv',index=False)
    summary=[];cis=[]
    for depth,dp in per.groupby('depth'):
        common=set(ids)
        for method,g in dp.groupby('method'):common &= set(g[g.status.eq('COMPLETE')].query_id)
        cp=dp[dp.query_id.isin(common)]
        for method,g in dp.groupby('method'):
            gc=g[g.query_id.isin(common)]
            summary.append(dict(method=method,depth=depth,queries=len(ids),evaluated=len(common),
                **{m:float(gc[m].mean()) if len(gc) else None for m in METRICS}))
        if len(cp) and 'A0' in set(cp.method):cis.extend(dict(depth=depth,**r) for r in paired_bootstrap(cp))
    pd.DataFrame(summary).to_csv(target/'summary.csv',index=False);pd.DataFrame(cis).to_csv(target/'paired_ci.csv',index=False)
    agreement=None
    if args.second_rater:
        second=validate_labels(pd.read_csv(args.second_rater,keep_default_na=False),pool,max_grade=2)
        joined=labels.merge(second,on=['query_id','doc_id'],suffixes=('_a','_b'),validate='one_to_one')
        if len(joined):
            if (joined.annotator_id_a==joined.annotator_id_b).any():raise ValueError('Agreement needs distinct annotators')
            from sklearn.metrics import cohen_kappa_score
            kappa=cohen_kappa_score(joined.relevance_grade_a,joined.relevance_grade_b,labels=[0,1,2])
            agreement=dict(pairs=len(joined),exact=float((joined.relevance_grade_a==joined.relevance_grade_b).mean()),
                cohen_kappa=float(kappa) if pd.notna(kappa) else None,second_rater_sha256=sha(args.second_rater))
    write_json(target/'manifest.json',dict(split=args.split,labels_sha256=sha(args.labels),verified_pairs=len(labels),
        created_at=datetime.now(timezone.utc).isoformat(),agreement=agreement,run_directory=str(folder),
        frozen_config_sha256=sha(args.frozen_config) if args.frozen_config else None,
        selection_status='NO_AUTOMATIC_PROMOTION; per-method isolated cost benchmark still required',
        historical_qrels_unchanged=True,production_changed=False))
    print(json.dumps(dict(output=str(target),summary=summary),ensure_ascii=False))


if __name__=='__main__':main()
