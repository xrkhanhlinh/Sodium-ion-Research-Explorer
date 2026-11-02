"""Silver-only adapter; never writes to the locked human evaluation directory."""
from pathlib import Path
import hashlib,itertools,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.final_system_evaluation import metrics_at_10

OUT=ROOT/'outputs/ai_silver_labels_703_20261026_v1'
BASE=ROOT/'outputs/batterybert_full_document_20261024_v1'
METRICS=['Precision_at_10','nDCG_at_10','MRR_at_10']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,payload):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=payload:raise ValueError('Preserve existing output; differing result: '+str(p))
        return
    t=p.with_suffix(p.suffix+'.pending');t.write_bytes(payload);t.replace(p)
def js(p,v):write(p,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8'))
def csv(p,v):write(p,v.to_csv(index=False).encode('utf-8-sig'))
def evaluate(ranks,pool,labels,scenario='primary',threshold=2):
    qrels=labels.set_index(['query_id','doc_id']).ai_label.astype(int);rows=[]
    for (qid,method),g in ranks.groupby(['query_id','method'],sort=False):
        g=g.sort_values('rank');grades=[int(qrels.loc[(qid,d)]) for d in g.doc_id]
        pg=[int(qrels.loc[(qid,d)]) for d in pool.loc[pool.query_id.eq(qid),'doc_id']]
        v=metrics_at_10(grades,pg)
        if threshold!=2:
            rel=np.flatnonzero(np.array(grades)>=threshold)
            v.update(Precision_at_10=float(np.sum(np.array(grades)>=threshold)/10),MRR_at_10=float(1/(rel[0]+1)) if len(rel) else 0.)
        nrel=sum(x>=threshold for x in pg);hits=sum(x>=threshold for x in grades)
        rows.append({'query_id':qid,'method':method,'scenario':scenario,'threshold':threshold,**v,
          'relevant_in_top10':hits,'relevant_in_judged_pool':nrel,
          'Recall_pool_at_10':hits/nrel if nrel else np.nan,
          'returned':len(grades),'status':'AI_SILVER_NOT_HUMAN_VERIFIED'})
    return pd.DataFrame(rows)
def main():
    sys.stdout.reconfigure(encoding='utf8');target=OUT/'evaluation';target.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv(OUT/'silver_labels_703.csv',keep_default_na=False,dtype=str)
    pool=pd.read_csv(BASE/'annotation_pool.csv',keep_default_na=False,dtype=str)
    frozen=json.loads((OUT/'labels_frozen_manifest.json').read_text(encoding='utf8'))
    assert sha(OUT/'silver_labels_703.csv')==frozen['labels_sha256'] and sha(BASE/'annotation_pool.csv')==frozen['input_sha256']
    original_files=['annotation_pool.csv','comparison_rankings_top10.csv','comparison_run_manifest.csv','accepted_human_labels.csv','retrieval_evaluation.csv','retrieval_metrics_per_query.csv']
    before={n:sha(BASE/n) for n in original_files}
    ranks=pd.read_csv(BASE/'comparison_rankings_top10.csv',keep_default_na=False)
    runs=pd.read_csv(BASE/'comparison_run_manifest.csv',keep_default_na=False)
    queries=pd.read_csv(BASE/'queries.csv',keep_default_na=False)
    methods=list(ranks.method.drop_duplicates());qids=list(queries.query_id)
    assert len(labels)==len(pool)==703 and len(ranks)==1200 and len(methods)==6 and len(qids)==20
    assert list(zip(labels.query_id,labels.doc_id))==list(zip(pool.query_id,pool.doc_id))
    assert not ranks.duplicated(['query_id','method','doc_id']).any()
    assert set(zip(runs.query_id,runs.method))==set(itertools.product(qids,methods)) and runs.status.eq('OK').all()
    for (qid,m),g in ranks.groupby(['query_id','method']):
        assert sorted(g['rank'])==list(range(1,11)) and len(g)==10
        assert set(g.doc_id)<=set(pool.loc[pool.query_id.eq(qid),'doc_id'])
        source=pool.set_index(['query_id','doc_id'])
        for row in g.itertuples():
            assert row.title==source.loc[(qid,row.doc_id),'title'] and row.abstract==source.loc[(qid,row.doc_id),'abstract']
    protocol={'status':'FROZEN_AI_SILVER_EVALUATION','label_hash':frozen['labels_sha256'],'ranking_hash':before['comparison_rankings_top10.csv'],
      'queries':20,'methods':methods,'ranking_slots':1200,'distinct_query_document_labels':703,'primary_threshold':2,
      'precision':'sum(label>=2 in returned top10)/10','mrr':'1/rank of first label2, or0',
      'ndcg':'sum((2^label-1)/log2(rank+1))/IDCG@10; ideal grades sorted across entire shared pool of each query; zero IDCG gives0',
      'aggregation':'Equal-weight macro mean across all20 queries for every method, including queries with no grade2',
      'recall':'Supplementary pooled recall only: label2 hits / all label2 documents in shared query pool. Undefined when pool has no label2; not full-corpus Recall.',
      'bootstrap':'10000 paired query resamples, seed20261026, conditional on fixed AI labels; percentile95% intervals; not annotation uncertainty',
      'sensitivity':'Same labels, permissive threshold>=1 as secondary; review-flag labels all0/all2 as scenarios, not guaranteed bounds on nDCG.',
      'human_evaluation':'Unchanged; AI qrels never imported as HUMAN_INDEPENDENT. No reranking/retraining/model selection from these labels.'}
    js(target/'protocol.json',protocol)
    per=evaluate(ranks,pool,labels)
    summary=per.groupby('method',sort=False)[METRICS].mean().reset_index()
    summary['queries']=20;summary['label_method']='AI_SILVER_NOT_HUMAN_VERIFIED'
    recalls=per.groupby('method',sort=False).agg(Recall_pool_at_10=('Recall_pool_at_10','mean'),defined_recall_queries=('Recall_pool_at_10','count')).reset_index()
    csv(target/'metrics_per_query.csv',per);csv(target/'metrics_summary.csv',summary);csv(target/'pooled_recall_supplement.csv',recalls)
    joined=ranks.merge(labels,on=['query_id','doc_id'],validate='many_to_one',sort=False)
    csv(target/'rankings_with_silver_labels.csv',joined)
    csv(target/'top10_errors_label_0_1.csv',joined[joined.ai_label.astype(int).lt(2)])
    sensitivity=[]
    for scenario,value in [('review_flags_as_0',0),('review_flags_as_2',2)]:
        altered=labels.copy();altered.loc[altered.review_required.eq('TRUE'),'ai_label']=str(value)
        scenario_per=evaluate(ranks,pool,altered,scenario)
        agg=scenario_per.groupby('method',sort=False)[METRICS].mean().reset_index();agg['scenario']=scenario;sensitivity.append(agg)
    relaxed=evaluate(ranks,pool,labels,'partial_or_direct',1)
    relaxed_summary=relaxed.groupby('method',sort=False)[METRICS].mean().reset_index();relaxed_summary['scenario']='partial_or_direct_threshold1';sensitivity.append(relaxed_summary)
    csv(target/'sensitivity_summary.csv',pd.concat(sensitivity,ignore_index=True))
    csv(target/'threshold1_per_query.csv',relaxed)
    # Independent cross-check: sklearn uses precomputed linear gains, while
    # repository primitive computes exponential gain from the original grades.
    from sklearn.metrics import ndcg_score
    errors=[]
    for row in per.itertuples():
        q=labels[labels.query_id.eq(row.query_id)];ids=q.doc_id.tolist();gains=2.**q.ai_label.astype(int).to_numpy()-1
        scores=np.zeros(len(q));ordered=ranks[ranks.query_id.eq(row.query_id)&ranks.method.eq(row.method)].sort_values('rank')
        for rr in ordered.itertuples():scores[ids.index(rr.doc_id)]=11-rr.rank
        expected=float(ndcg_score([gains],[scores],k=10))
        errors.append(abs(expected-row.nDCG_at_10))
        gg=[int(q.set_index('doc_id').loc[d,'ai_label']) for d in ordered.doc_id]
        assert row.Precision_at_10==sum(x==2 for x in gg)/10
        expected_rr=next((1/(i+1) for i,x in enumerate(gg) if x==2),0.)
        assert row.MRR_at_10==expected_rr
    assert max(errors)<1e-12
    known=metrics_at_10([0,1,2],[0,1,2]);assert known['Precision_at_10']==.1 and known['MRR_at_10']==1/3
    manual_dcg=(1/np.log2(3)+3/np.log2(4))/(3+1/np.log2(3));assert abs(known['nDCG_at_10']-manual_dcg)<1e-12
    assert all(v==0 for v in metrics_at_10([0,0],[0,0]).values())
    # Paired query bootstrap: same samples for all methods and all metrics.
    matrix=np.array([per[per.method.eq(m)].set_index('query_id').loc[qids,METRICS].to_numpy(float) for m in methods])
    resamples=np.random.default_rng(20261026).integers(0,20,size=(10000,20))
    samples=matrix[:,resamples,:].mean(axis=2)
    ci=[];pairs=[]
    for i,m in enumerate(methods):
        for j,metric in enumerate(METRICS):
            lo,hi=np.quantile(samples[i,:,j],[.025,.975]);ci.append({'method':m,'metric':metric,'mean':float(matrix[i,:,j].mean()),'ci95_low':float(lo),'ci95_high':float(hi),'queries':20})
    for i,j in itertools.combinations(range(6),2):
        for k,metric in enumerate(METRICS):
            delta=samples[i,:,k]-samples[j,:,k];lo,hi=np.quantile(delta,[.025,.975])
            pairs.append({'left':methods[i],'right':methods[j],'metric':metric,'mean_delta_left_minus_right':float((matrix[i,:,k]-matrix[j,:,k]).mean()),'ci95_low':float(lo),'ci95_high':float(hi),
              'interpretation':'Exploratory query bootstrap;45 comparisons, no multiplicity adjustment; fixed single-AI labels'})
    csv(target/'confidence_intervals.csv',pd.DataFrame(ci));csv(target/'paired_deltas.csv',pd.DataFrame(pairs))
    no_direct=sorted(per.loc[per.relevant_in_judged_pool.eq(0),'query_id'].unique().tolist())
    validation={'status':'PASS','pairs':703,'queries':20,'methods':6,'ranking_slots':1200,'evaluated_query_method_groups':120,
      'labels_numeric_complete':True,'exact_input_pair_order':True,'all_evidence_exact_source_quotes':True,
      'ndcg_independent_sklearn_max_absolute_error':max(errors),'precision_mrr_independent_checks':120,'hand_computed_metric_fixtures':2,
      'shared_pool_IDCG_used':True,'no_label2_query_pools':no_direct,'original_files_unchanged':all(sha(BASE/n)==h for n,h in before.items()),
      'protected_file_hashes':before,'labels_sha256':sha(OUT/'silver_labels_703.csv'),'tests_are_not_semantic_label_validation':True}
    assert validation['original_files_unchanged'];js(target/'validation.json',validation)
    files=sorted(list(target.glob('*.csv'))+[target/'protocol.json',target/'validation.json'])
    js(target/'artifact_manifest.json',{'status':'completed','artifacts':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for p in files]})
    print(summary.to_string(index=False));print(json.dumps(validation,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
