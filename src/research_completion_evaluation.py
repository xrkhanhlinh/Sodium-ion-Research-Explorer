"""Shared-pool 0..2 human relevance protocol for the four research methods."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .research_completion import OUT
from .final_search import RESEARCH_METHODS
from .final_system_artifacts import sha,read_json
from .final_system_evaluation import keys,validate_labels,metrics_at_10,LABELS
from .model_comparison_io import write_json


def create_pool():
    rankings=pd.read_csv(OUT/'rankings_top10.csv',keep_default_na=False)
    queries=pd.read_csv(OUT/'queries.csv',keep_default_na=False).set_index('query_id')
    pool=rankings[['query_id','query','doc_id','title','abstract','source','year','identifier']].drop_duplicates(['query_id','doc_id'])
    pool=keys(pool).sort_values(['query_id','doc_id']).reset_index(drop=True)
    if pool.groupby('query_id')['query'].nunique().max()!=1:raise ValueError('Conflicting query texts')
    if any(row.query!=queries.loc[row.query_id,'query'] for row in pool.itertuples()):raise ValueError('Query mismatch')
    # Deliberately exclude scores, method, ranks and topics from annotation.
    path=OUT/'annotation_pool.csv'
    if path.exists():
        old=pd.read_csv(path,keep_default_na=False)
        if not old.equals(pool):raise ValueError('Frozen annotation pool would change')
    else:pool.to_csv(path,index=False)
    template=pool.copy()
    for field in (*LABELS,'reviewer_notes'):template[field]=''
    template['label_status']='UNVERIFIED'
    if not (OUT/'annotation_template_0_2.csv').exists():template.to_csv(OUT/'annotation_template_0_2.csv',index=False)
    write_json(OUT/'annotation_protocol.json',dict(scale={'0':'Not relevant','1':'Partially relevant','2':'Directly relevant'},
        precision_and_mrr_threshold=2,ndcg_gain='2^grade - 1',idcg='Complete shared Top10 pool, not full-corpus qrels',
        recall='Not reported; corpus relevant set is not exhaustive',blank_is_zero=False,
        human_label_method='HUMAN_INDEPENDENT',verified_status='VERIFIED',independent_verifier_required=True,
        ranks_and_scores_hidden=True,query_selection='First20 frozen non-edge queries; no query rewriting or tuning using labels',
        evaluation_role='Locked final evaluation; these labels must not be used for representation or ranking tuning',
        pool_sha256=sha(path),queries_sha256=sha(OUT/'queries.csv'),rankings_sha256=sha(OUT/'rankings_top10.csv')))
    return pool


def evaluate(labels_path=None):
    return evaluate_dataset(OUT,RESEARCH_METHODS,labels_path)


def evaluate_dataset(OUT,RESEARCH_METHODS,labels_path=None,rankings_filename='rankings_top10.csv',runs_filename='run_manifest.csv'):
    """Apply the same locked human-label protocol to an isolated benchmark.

    Explicit directory and method arguments avoid changing module globals when
    the full-document and historical dashboards evaluate separate pools.
    """
    OUT=Path(OUT);RESEARCH_METHODS=tuple(RESEARCH_METHODS)
    protocol=read_json(OUT/'annotation_protocol.json')
    for file,key in [('annotation_pool.csv','pool_sha256'),('queries.csv','queries_sha256'),(rankings_filename,'rankings_sha256')]:
        if sha(OUT/file)!=protocol[key]:raise ValueError('Frozen evaluation artifact changed: '+file)
    pool=keys(pd.read_csv(OUT/'annotation_pool.csv',keep_default_na=False))
    ranks=pd.read_csv(OUT/rankings_filename,keep_default_na=False)
    runs=pd.read_csv(OUT/runs_filename,keep_default_na=False)
    queries=pd.read_csv(OUT/'queries.csv',keep_default_na=False)
    expected={(q,m) for q in queries.query_id for m in RESEARCH_METHODS}
    if runs.duplicated(['query_id','method']).any() or set(zip(runs.query_id,runs.method))!=expected or not runs.status.eq('OK').all():
        raise ValueError('Incomplete successful benchmark matrix')
    if ranks.duplicated(['query_id','method','doc_id']).any() or set(ranks.method)-set(RESEARCH_METHODS):raise ValueError('Invalid ranking IDs/methods')
    for run in runs.itertuples():
        selected=ranks[ranks.query_id.eq(run.query_id)&ranks.method.eq(run.method)]
        if len(selected)!=run.result_count or selected['rank'].tolist()!=list(range(1,len(selected)+1)) or len(selected)>10:
            raise ValueError('Invalid Top10 ranking/run count')
    if set(zip(ranks.query_id,ranks.doc_id))-set(zip(pool.query_id,pool.doc_id)):raise ValueError('Rankings outside annotation pool')
    if labels_path is None and (OUT/'evaluation_status.json').exists():
        previous=read_json(OUT/'evaluation_status.json')
        if previous.get('label_source'):
            labels_path=Path(previous['label_source'])
            if not labels_path.exists() or sha(labels_path)!=previous['label_source_sha256']:
                raise ValueError('Previously accepted human label source changed or disappeared')
    labels=validate_labels(pd.read_csv(labels_path,keep_default_na=False),pool,max_grade=2) if labels_path else pd.DataFrame(columns=['query_id','doc_id',*LABELS])
    labels.to_csv(OUT/'accepted_human_labels.csv',index=False)
    lookup={(r.query_id,r.doc_id):int(r.relevance_grade) for r in labels.itertuples()}
    rows,coverage=[],[]
    for q in queries.itertuples():
        docs=pool.loc[pool.query_id.eq(q.query_id),'doc_id'].tolist()
        judged=sum((q.query_id,d) in lookup for d in docs)
        complete=bool(docs) and judged==len(docs)
        status='COMPLETE_HUMAN_POOL' if complete else 'PENDING_HUMAN_ANNOTATION'
        coverage.append(dict(query_id=q.query_id,pool_documents=len(docs),judged_documents=judged,status=status))
        for method in RESEARCH_METHODS:
            row=dict(query_id=q.query_id,method=method,status=status,Precision_at_10=None,nDCG_at_10=None,MRR_at_10=None)
            if complete:
                selected=ranks[ranks.query_id.eq(q.query_id)&ranks.method.eq(method)].sort_values('rank')
                row.update(metrics_at_10([lookup[(q.query_id,d)] for d in selected.doc_id],[lookup[(q.query_id,d)] for d in docs]))
            rows.append(row)
    per=pd.DataFrame(rows);per.to_csv(OUT/'retrieval_metrics_per_query.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'annotation_coverage.csv',index=False)
    completed=per[per.status.eq('COMPLETE_HUMAN_POOL')];n=completed.query_id.nunique()
    status='PENDING_HUMAN_ANNOTATION' if not n else 'COMPLETE' if n==len(queries) else 'PARTIAL_QUERY_COVERAGE'
    summary=pd.DataFrame([dict(method=m,status=status,eligible_queries=n,total_queries=len(queries),
        **{metric:float(completed.loc[completed.method.eq(m),metric].mean()) if n else None for metric in ('Precision_at_10','nDCG_at_10','MRR_at_10')}) for m in RESEARCH_METHODS])
    summary.to_csv(OUT/'retrieval_evaluation.csv',index=False)
    write_json(OUT/'evaluation_status.json',dict(status=status,accepted_human_labels=len(labels),pool_pairs=len(pool),eligible_queries=n,
        total_queries=len(queries),label_scale=[0,1,2],missing_labels_are_zero=False,recall_reported=False,
        label_source=str(Path(labels_path).resolve()) if labels_path else None,label_source_sha256=sha(labels_path) if labels_path else None,
        selection_used_human_labels=False,metrics_are_retrieval_accuracy=False if not n else True))
    if n:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ax=summary.set_index('method')[['Precision_at_10','nDCG_at_10','MRR_at_10']].plot.bar(figsize=(12,6),ylim=(0,1))
        ax.set_title(f'Human judged Top10 | {n}/{len(queries)} complete query pools');ax.figure.tight_layout()
        ax.figure.savefig(OUT/'retrieval_evaluation.png',dpi=160);plt.close(ax.figure)
    return summary


def import_labels(raw):
    import io
    frame=pd.read_csv(io.BytesIO(raw),keep_default_na=False)
    validate_labels(frame,pd.read_csv(OUT/'annotation_pool.csv',keep_default_na=False),max_grade=2)
    path=OUT/('human_labels_'+hashlib.sha256(raw).hexdigest()[:16]+'.csv')
    if not path.exists():path.write_bytes(raw)
    return evaluate(path)
