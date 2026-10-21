"""Strict, isolated evaluation for saved retrieval runs; never manufactures qrels.

Unlike the frozen Stage4 evaluator this accepts all seven methods, validates the
complete query/method matrix, and writes only to an explicitly new output folder.
"""
from collections import Counter
from itertools import combinations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS=('SE1','SE2','SE3','BM25','BGE_PRETRAINED','BGE_FINETUNED','BGE_CHUNKED_OFFLINE')
METRICS=('Precision_at_5','nDCG_at_5','MRR_at_5')
PROVENANCE=('annotator_id','verifier_id','verification_reference')
NULL_TEXT={'','nan','none','null','n/a','na','unknown','tbd'}


def clean_text_column(values,name):
    if values.isna().any():raise ValueError(f'Missing {name}')
    result=values.map(lambda x:str(x).strip())
    if result.str.casefold().isin(NULL_TEXT).any():raise ValueError(f'Missing or placeholder {name}')
    return result


def clean_keys(frame):
    frame=frame.copy()
    for name in ('query_id','doc_id'):
        if name not in frame:raise ValueError(f'Missing {name}')
        frame[name]=clean_text_column(frame[name],name)
    if frame.duplicated(['query_id','doc_id']).any():raise ValueError('Duplicate query/document labels or pool records')
    return frame


def grades_array(values):
    raw=np.asarray(values,dtype=object)
    if raw.ndim!=1 or any(isinstance(x,(bool,np.bool_)) for x in raw):
        raise ValueError('Grades must be a one-dimensional list of integer grades, not booleans')
    try:arr=np.asarray(raw,dtype=float)
    except (TypeError,ValueError) as exc:raise ValueError('Invalid relevance grades') from exc
    if not np.isfinite(arr).all() or not np.isin(arr,[0,1,2,3]).all():
        raise ValueError('Grades must be finite integers 0..3')
    return arr.astype(int)


def validate_labels(frame):
    frame=clean_keys(frame)
    required=['relevance_grade',*PROVENANCE,'label_status','label_method']
    if any(name not in frame for name in required):raise ValueError('Missing grade or provenance columns')
    for name in PROVENANCE:frame[name]=clean_text_column(frame[name],name)
    if not frame.label_status.eq('VERIFIED').all():raise ValueError('Every evaluated label must be VERIFIED')
    if not frame.label_method.eq('HUMAN_INDEPENDENT').all():raise ValueError('Independent human labels required')
    if frame.annotator_id.eq(frame.verifier_id).any():raise ValueError('Annotator and verifier must differ after trimming')
    frame['grade']=grades_array(frame.relevance_grade)
    return frame


def metrics_at_5(grades,judged_pool_grades):
    g=grades_array(grades);pool=grades_array(judged_pool_grades)
    if len(g)>5:raise ValueError('@5 accepts at most five results; truncate explicitly upstream')
    if Counter(g)-Counter(pool):raise ValueError('Ranked grades must be included in the judged pool')
    ideal=np.sort(pool)[::-1][:5]
    dcg=float(np.sum((2.**g-1)/np.log2(np.arange(len(g))+2)))
    idcg=float(np.sum((2.**ideal-1)/np.log2(np.arange(len(ideal))+2)))
    relevant=np.flatnonzero(g>=2)
    result=dict(Precision_at_5=float(np.sum(g>=2)/5),nDCG_at_5=dcg/idcg if idcg else 0.,
                MRR_at_5=float(1/(relevant[0]+1)) if len(relevant) else 0.)
    if any(not 0<=x<=1+1e-12 for x in result.values()):raise ValueError('Metric outside [0,1]')
    return result


def expected_queries(queries):
    required={'query_id','is_edge_case'}
    if not required<=set(queries):raise ValueError('Query protocol missing ID or edge-case flag')
    if queries.query_id.isna().any() or queries.query_id.duplicated().any():raise ValueError('Invalid query protocol IDs')
    def boolean(value):
        if isinstance(value,(bool,np.bool_)):return bool(value)
        if isinstance(value,str) and value.casefold() in {'true','false'}:return value.casefold()=='true'
        raise ValueError('is_edge_case must be explicit true/false')
    edges=queries.is_edge_case.map(boolean)
    ids=clean_text_column(queries.query_id,'query_id')
    if ids.duplicated().any():raise ValueError('Duplicate normalized query IDs')
    return ids[~edges].tolist(),set(ids)


def validate_rankings(rankings,queries,pool,methods=METHODS,run_manifest=None):
    domain,all_queries=expected_queries(queries)
    if not domain or not methods or len(methods)!=len(set(methods)):
        raise ValueError('Need primary queries and distinct declared methods')
    required={'query_id','doc_id','method','rank'}
    if not required<=set(rankings):raise ValueError('Missing ranking columns')
    frame=rankings.copy()
    for name in ('query_id','doc_id','method'):frame[name]=clean_text_column(frame[name],name)
    if not set(frame.query_id)<=all_queries or not set(frame.method)<=set(methods):
        raise ValueError('Unknown query or method in rankings')
    if frame['rank'].map(lambda value:isinstance(value,(bool,np.bool_))).any():raise ValueError('Boolean rank is invalid')
    rank=pd.to_numeric(frame['rank'],errors='coerce')
    if not np.isfinite(rank).all() or not rank.between(1,5).all() or not rank.eq(np.floor(rank)).all():
        raise ValueError('Ranks must be integers 1..5')
    frame['rank']=rank.astype(int)
    if frame.duplicated(['query_id','method','rank']).any() or frame.duplicated(['query_id','method','doc_id']).any():
        raise ValueError('Duplicate rank or repeated document within a result list')
    primary=frame[frame.query_id.isin(domain)].copy()
    expected={(qid,method) for qid in domain for method in methods}
    counts=primary.groupby(['query_id','method']).size().to_dict()
    if run_manifest is None:
        if set(counts)!=expected:raise ValueError('Incomplete query/method coverage; declare empty runs explicitly')
    else:
        runs=run_manifest.copy()
        for name in ('query_id','method'):runs[name]=clean_text_column(runs[name],name)
        if runs.duplicated(['query_id','method']).any():raise ValueError('Duplicate run declarations')
        if set(map(tuple,runs[['query_id','method']].values))!=expected:raise ValueError('Incomplete run manifest')
        if 'results_count' not in runs:raise ValueError('Missing run result counts')
        if runs.results_count.map(lambda value:isinstance(value,(bool,np.bool_))).any():raise ValueError('Boolean result count is invalid')
        runs['results_count']=pd.to_numeric(runs.results_count,errors='coerce')
        for row in runs.itertuples():
            if isinstance(row.results_count,(bool,np.bool_)) or not np.isfinite(row.results_count) or int(row.results_count)!=row.results_count:
                raise ValueError('Run result count must be an integer')
            if not 0<=row.results_count<=5 or row.results_count!=counts.get((row.query_id,row.method),0):
                raise ValueError('Run manifest count disagrees with rankings')
    for _,group in primary.groupby(['query_id','method']):
        if sorted(group['rank'])!=list(range(1,len(group)+1)):raise ValueError('Ranking gaps')
    pool=clean_keys(pool)
    if not set(pool.query_id)<=all_queries:raise ValueError('Unknown query in pool')
    if not set(domain)<=set(pool.query_id):raise ValueError('Every primary query needs a declared judgment pool')
    keys=set(map(tuple,pool[['query_id','doc_id']].values))
    if not set(map(tuple,primary[['query_id','doc_id']].values))<=keys:raise ValueError('Ranked document absent from judged pool')
    return primary,domain,pool


def paired_bootstrap(per_query,domain,methods=METHODS,repetitions=2000,seed=42):
    if repetitions<100:raise ValueError('Use at least 100 bootstrap repetitions')
    samples=np.random.default_rng(seed).integers(0,len(domain),size=(repetitions,len(domain)))
    arrays={}
    for method in methods:
        frame=per_query[per_query.method.eq(method)].set_index('query_id')
        if frame.index.duplicated().any() or set(frame.index)!=set(domain):raise ValueError('Incomplete paired results')
        values=frame.loc[domain,list(METRICS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():raise ValueError('Cannot bootstrap missing relevance metrics')
        arrays[method]=values
    summary=[];pairs=[]
    for method,values in arrays.items():
        boot=values[samples].mean(axis=1)
        for i,metric in enumerate(METRICS):
            low,high=np.quantile(boot[:,i],[.025,.975])
            summary.append(dict(method=method,metric=metric,mean=float(values[:,i].mean()),
                                ci95_low=float(low),ci95_high=float(high),queries=len(domain)))
    for left,right in combinations(methods,2):
        delta=arrays[left]-arrays[right];boot=delta[samples].mean(axis=1)
        for i,metric in enumerate(METRICS):
            low,high=np.quantile(boot[:,i],[.025,.975])
            pairs.append(dict(left=left,right=right,metric=metric,mean_delta=float(delta[:,i].mean()),
                ci95_low=float(low),ci95_high=float(high),queries=len(domain),
                interpretation='Exploratory unadjusted paired bootstrap CI; no multiple-comparison claim'))
    return pd.DataFrame(summary),pd.DataFrame(pairs)


def evaluate_frames(rankings,queries,pool,labels=None,methods=METHODS,run_manifest=None,repetitions=2000):
    rankings,domain,pool=validate_rankings(rankings,queries,pool,methods,run_manifest)
    primary_pool=pool[pool.query_id.isin(domain)]
    expected=set(map(tuple,primary_pool[['query_id','doc_id']].values))
    if labels is not None:
        labels=clean_keys(labels)
        if not {'relevance_grade','label_status'}<=set(labels):raise ValueError('Missing annotation columns')
        _,all_queries=expected_queries(queries)
        if not set(labels.query_id)<=all_queries:raise ValueError('Unknown query in qrels')
        labels=labels[labels.query_id.isin(domain)]
    # Completely blank forms are pending, never implicitly zero-relevance labels.
    if labels is None or (labels.label_status.eq('UNVERIFIED').all()
            and labels.relevance_grade.fillna('').astype(str).str.strip().eq('').all()):
        per=pd.DataFrame([dict(query_id=qid,method=m,**{key:None for key in METRICS},status='PENDING_LABELS')
                          for qid in domain for m in methods])
        summary=pd.DataFrame([dict(method=m,status='PENDING_LABELS',verified_queries=0,**{key:None for key in METRICS}) for m in methods])
        return dict(status='PENDING_LABELS',domain_queries=len(domain),pool_pairs=len(expected),per_query=per,
                    summary=summary,confidence_intervals=pd.DataFrame(),paired_deltas=pd.DataFrame(),zero_relevant_queries=[],zero_idcg_queries=[])
    labels=validate_labels(labels)
    if set(map(tuple,labels[['query_id','doc_id']].values))!=expected:
        raise ValueError('Require exactly the predeclared primary judged pool: missing or extra qrels')
    qrels=labels.set_index(['query_id','doc_id']).grade
    rows=[];zero=[];zero_idcg=[]
    for qid in domain:
        ideal=labels.loc[labels.query_id.eq(qid),'grade'].tolist()
        if not any(x>=2 for x in ideal):zero.append(qid)
        if not any(x>0 for x in ideal):zero_idcg.append(qid)
        for method in methods:
            group=rankings[rankings.query_id.eq(qid)&rankings.method.eq(method)].sort_values('rank')
            grades=[qrels.loc[(qid,doc)] for doc in group.doc_id]
            rows.append(dict(query_id=qid,method=method,**metrics_at_5(grades,ideal),status='VERIFIED_DECLARED_PROVENANCE'))
    per=pd.DataFrame(rows)
    summary=per.groupby('method')[list(METRICS)].mean().reset_index()
    summary['status']='VERIFIED_DECLARED_PROVENANCE';summary['verified_queries']=len(domain)
    ci,deltas=paired_bootstrap(per,domain,methods,repetitions)
    return dict(status='VERIFIED_DECLARED_PROVENANCE',domain_queries=len(domain),pool_pairs=len(expected),
                per_query=per,summary=summary,confidence_intervals=ci,paired_deltas=deltas,
                zero_relevant_queries=zero,zero_idcg_queries=zero_idcg)


def annotation_agreement(left,right):
    """Agreement on paired raw ratings before adjudication; not verified qrels."""
    left=clean_keys(left);right=clean_keys(right)
    keys=['query_id','doc_id']
    if set(map(tuple,left[keys].values))!=set(map(tuple,right[keys].values)):
        raise ValueError('Both raters must receive the same declared pool')
    for frame in (left,right):
        if not {'relevance_grade','annotator_id'}<=set(frame):raise ValueError('Missing raw rating fields')
        missing=frame.relevance_grade.isna()|frame.relevance_grade.astype(str).str.strip().eq('')
        frame['rated']=~missing
        if (~missing).any():
            grades_array(frame.loc[~missing,'relevance_grade'])
            frame.loc[~missing,'annotator_id']=clean_text_column(frame.loc[~missing,'annotator_id'],'annotator_id')
    joined=left.merge(right,on=keys,validate='one_to_one',suffixes=('_a','_b'))
    paired=joined[joined.rated_a&joined.rated_b]
    result=dict(status='PENDING_ANNOTATIONS',pool_pairs=len(joined),paired_ratings=len(paired),
        exact_agreement=None,quadratic_weighted_kappa=None,kappa_note='No paired human ratings yet')
    if paired.empty:return result
    if paired.annotator_id_a.eq(paired.annotator_id_b).any():raise ValueError('Independent raters must differ')
    a=grades_array(paired.relevance_grade_a);b=grades_array(paired.relevance_grade_b)
    matrix=np.zeros((4,4),dtype=float)
    np.add.at(matrix,(a,b),1)
    weights=(np.arange(4)[:,None]-np.arange(4)[None,:])**2/9.
    observed=matrix/len(a);expected=np.outer(matrix.sum(axis=1),matrix.sum(axis=0))/len(a)**2
    denominator=float((weights*expected).sum())
    result.update(status='COMPLETE_RAW_RATINGS' if len(paired)==len(joined) else 'PARTIAL_RAW_RATINGS',
        exact_agreement=float(np.mean(a==b)),quadratic_weighted_kappa=1-float((weights*observed).sum())/denominator if denominator else None,
        kappa_note='Before adjudication; partial ratings may be selected/nonrepresentative; no expert identity authentication'
                   if denominator else 'Undefined kappa: no variation in rating marginals')
    return result


def paired_cluster_bootstrap(before,after,groups,metric_names,repetitions=2000,seed=42):
    """Paired bootstrap of groups, preserving dependent queries within each group.

    Reports document-averaged means with cluster-resampled uncertainty; does not
    estimate training-seed variability or external-query generalization.
    """
    before=np.asarray(before,dtype=float);after=np.asarray(after,dtype=float)
    group_series=pd.Series(groups)
    if before.ndim!=2 or before.shape!=after.shape or len(group_series)!=len(before) or not len(before):
        raise ValueError('Paired arrays/groups must align and be nonempty')
    if before.shape[1]!=len(metric_names) or repetitions<100:
        raise ValueError('Invalid metrics or bootstrap repetitions')
    if not np.isfinite(before).all() or not np.isfinite(after).all():raise ValueError('Nonfinite paired observations')
    group_series=clean_text_column(group_series,'split_group_id')
    unique,inverse=np.unique(group_series,return_inverse=True)
    if len(unique)<2:raise ValueError('Need at least two independent clusters')
    sizes=np.bincount(inverse).astype(float)
    sums=np.zeros((len(unique),before.shape[1]))
    np.add.at(sums,inverse,after-before)
    rng=np.random.default_rng(seed);boot=np.empty((repetitions,before.shape[1]))
    # Batch replicates to bound RAM on the 8GB machine.
    for start in range(0,repetitions,128):
        end=min(start+128,repetitions)
        sampled=rng.integers(0,len(unique),size=(end-start,len(unique)))
        boot[start:end]=sums[sampled].sum(axis=1)/sizes[sampled].sum(axis=1)[:,None]
    rows=[]
    for index,name in enumerate(metric_names):
        low,high=np.quantile(boot[:,index],[.025,.975])
        rows.append(dict(metric=name,before=float(before[:,index].mean()),after=float(after[:,index].mean()),
            mean_delta=float((after-before)[:,index].mean()),ci95_low=float(low),ci95_high=float(high),
            queries=len(before),clusters=len(unique),bootstrap_repetitions=repetitions,
            scope='Conditional on these fixed checkpoints and held-out proxy test; clustered by split_group_id'))
    return pd.DataFrame(rows)


def run(rankings_path,queries_path,pool_path,output,labels_path=None,run_manifest_path=None):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Choose an empty output directory; existing evaluation is preserved')
    output.mkdir(parents=True,exist_ok=True)
    read=lambda p:pd.read_csv(p,keep_default_na=False)
    result=evaluate_frames(read(rankings_path),read(queries_path),read(pool_path),
        read(labels_path) if labels_path else None,
        run_manifest=read(run_manifest_path) if run_manifest_path else None)
    for key in ('per_query','summary','confidence_intervals','paired_deltas'):
        frame=result.pop(key)
        if len(frame):frame.to_csv(output/(key+'.csv'),index=False,encoding='utf-8-sig')
    result.update(grade_threshold=2,ndcg_gain='2^grade-1',ndcg_reference='Shared predeclared judged pool per query',
        zero_ideal_policy='Keep all primary queries. nDCG=0 only if IDCG=0; zero-relevant query IDs listed separately.',
        ci_scope='Paired query bootstrap on this benchmark only; 2000 replicates/seed42; exploratory comparisons.',
        provenance_limit='Fields are validated; software cannot authenticate expert identity or independence.',
        labels_path=str(labels_path) if labels_path else None)
    (output/'scope.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('rankings','queries','pool','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--labels',type=Path)
    parser.add_argument('--run-manifest',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.rankings,args.queries,args.pool,args.output,args.labels,args.run_manifest),indent=2))
