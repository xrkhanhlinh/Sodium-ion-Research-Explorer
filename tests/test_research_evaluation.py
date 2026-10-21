"""Synthetic labels here test mathematics only; never real research qrels."""
import numpy as np
import pandas as pd
import pytest
import json
from src.research_evaluation import (validate_labels,metrics_at_5,evaluate_frames,
                                      validate_rankings,paired_bootstrap,annotation_agreement,paired_cluster_bootstrap,run,METHODS)


def label(**changes):
    row=dict(query_id='Q1',doc_id='d1',relevance_grade=3,annotator_id='alice',verifier_id='bob',
             verification_reference='synthetic-test-only',label_method='HUMAN_INDEPENDENT',label_status='VERIFIED')
    row.update(changes)
    return row


@pytest.mark.parametrize('field',['annotator_id','verifier_id','verification_reference','query_id','doc_id'])
@pytest.mark.parametrize('missing',[None,np.nan,'','  ','NaN','null','TBD'])
def test_missing_or_placeholder_provenance_rejected(field,missing):
    with pytest.raises(ValueError):validate_labels(pd.DataFrame([label(**{field:missing})]))


def test_self_verification_after_normalization_and_duplicate_ids():
    with pytest.raises(ValueError):validate_labels(pd.DataFrame([label(annotator_id=' bob ')]))
    with pytest.raises(ValueError):validate_labels(pd.DataFrame([label(),label(doc_id=' d1 ')]))


@pytest.mark.parametrize('value',[True,False,2.5,-1,4,np.inf,np.nan,'broken'])
def test_invalid_grades_rejected(value):
    with pytest.raises(ValueError):validate_labels(pd.DataFrame([label(relevance_grade=value)]))


def test_unverified_and_machine_generated_are_not_official():
    for change in [dict(label_status='UNVERIFIED'),dict(label_method='MODEL_GENERATED')]:
        with pytest.raises(ValueError):validate_labels(pd.DataFrame([label(**change)]))


def test_hand_computed_ranking_and_empty_results():
    result=metrics_at_5([0,2,3,0,1],[3,2,1,0,0])
    dcg=3/np.log2(3)+7/np.log2(4)+1/np.log2(6)
    idcg=7+3/np.log2(3)+1/np.log2(4)
    assert result==pytest.approx(dict(Precision_at_5=.4,MRR_at_5=.5,nDCG_at_5=dcg/idcg))
    assert metrics_at_5([],[])==dict(Precision_at_5=0.,nDCG_at_5=0.,MRR_at_5=0.)
    assert metrics_at_5([3],[3])['Precision_at_5']==.2
    assert metrics_at_5([0]*5,[0]*5)['nDCG_at_5']==0


def test_six_results_and_invalid_ideal_rejected():
    with pytest.raises(ValueError):metrics_at_5([3]*6,[3]*6)
    with pytest.raises(ValueError):metrics_at_5([3],[0])


@pytest.fixture
def example():
    queries=pd.DataFrame([dict(query_id='Q1',is_edge_case=False),dict(query_id='Q2',is_edge_case=False),
                          dict(query_id='QX',is_edge_case=True)])
    pool=pd.DataFrame([dict(query_id=q,doc_id=d) for q in ['Q1','Q2'] for d in ['d1','d2']])
    rankings=pd.DataFrame([dict(query_id=q,method=m,rank=r,doc_id=d)
        for q in ['Q1','Q2'] for m in ['A','B'] for r,d in enumerate(['d1','d2'] if m=='A' else ['d2','d1'],1)])
    labels=pd.DataFrame([label(query_id=q,doc_id=d,relevance_grade=3 if d=='d1' else 0)
                         for q in ['Q1','Q2'] for d in ['d1','d2']])
    return rankings,queries,pool,labels


def test_pending_has_no_invented_scores(example):
    rankings,queries,pool,labels=example
    result=evaluate_frames(rankings,queries,pool,methods=('A','B'))
    assert result['status']=='PENDING_LABELS' and result['domain_queries']==2
    assert result['per_query'][['Precision_at_5','nDCG_at_5','MRR_at_5']].isna().all().all()
    labels['label_status']='UNVERIFIED';labels['relevance_grade']=''
    assert evaluate_frames(rankings,queries,pool,labels,methods=('A','B'))['status']=='PENDING_LABELS'


def test_partial_labels_and_extra_qrels_rejected(example):
    r,q,p,l=example
    for labels in [l.iloc[:-1],pd.concat([l,pd.DataFrame([label(doc_id='unexpected')])],ignore_index=True)]:
        with pytest.raises(ValueError):evaluate_frames(r,q,p,labels,methods=('A','B'))


def test_full_metrics_and_paired_confidence_intervals(example):
    r,q,p,l=example
    result=evaluate_frames(r,q,p,l,methods=('A','B'),repetitions=200)
    assert result['status']=='VERIFIED_DECLARED_PROVENANCE'
    assert len(result['per_query'])==4 and len(result['confidence_intervals'])==6
    a=result['summary'].set_index('method').loc['A']
    b=result['summary'].set_index('method').loc['B']
    assert a.Precision_at_5==b.Precision_at_5==.2
    assert a.MRR_at_5==1. and b.MRR_at_5==.5
    ci=result['paired_deltas'].set_index('metric').loc['MRR_at_5']
    assert ci.mean_delta==ci.ci95_low==ci.ci95_high==.5


def test_missing_method_or_query_not_silently_dropped(example):
    r,q,p,l=example
    with pytest.raises(ValueError):evaluate_frames(r[r.method.eq('A')],q,p,l,methods=('A','B'))
    with pytest.raises(ValueError):evaluate_frames(r[r.query_id.eq('Q1')],q,p,l,methods=('A','B'))


def test_explicit_empty_run_is_included_with_zero_metrics(example):
    r,q,p,l=example
    r=r[~(r.query_id.eq('Q2')&r.method.eq('B'))]
    runs=pd.DataFrame([dict(query_id=qid,method=m,results_count=0 if (qid,m)==('Q2','B') else 2)
                       for qid in ['Q1','Q2'] for m in ['A','B']])
    result=evaluate_frames(r,q,p,l,methods=('A','B'),run_manifest=runs,repetitions=100)
    row=result['per_query'].query("query_id=='Q2' and method=='B'").iloc[0]
    assert row.Precision_at_5==row.MRR_at_5==row.nDCG_at_5==0
    runs.loc[0,'results_count']=0
    with pytest.raises(ValueError):evaluate_frames(r,q,p,l,methods=('A','B'),run_manifest=runs)


@pytest.mark.parametrize('change',['duplicate_rank','duplicate_document','gap','too_many','unknown_method','unknown_query','missing_pool'])
def test_ranking_structural_corruption_rejected(example,change):
    r,q,p,l=example
    r=r.copy();p=p.copy()
    if change=='duplicate_rank':r.loc[1,'rank']=1
    elif change=='duplicate_document':r.loc[1,'doc_id']='d1'
    elif change=='gap':r.loc[1,'rank']=3
    elif change=='too_many':r.loc[1,'rank']=6
    elif change=='unknown_method':r.loc[0,'method']='typo'
    elif change=='unknown_query':r.loc[0,'query_id']='unknown_query'
    else:p=p[~(p.query_id.eq('Q1')&p.doc_id.eq('d1'))]
    with pytest.raises(ValueError):validate_rankings(r,q,p,methods=('A','B'))


def test_bootstrap_rejects_partial_pairs(example):
    r,q,p,l=example
    data=evaluate_frames(r,q,p,l,methods=('A','B'),repetitions=100)['per_query']
    with pytest.raises(ValueError):paired_bootstrap(data.iloc[:-1],['Q1','Q2'],('A','B'))


def test_zero_relevant_queries_are_retained_and_disclosed(example):
    r,q,p,l=example
    l.loc[l.query_id.eq('Q2'),'relevance_grade']=0
    result=evaluate_frames(r,q,p,l,methods=('A','B'),repetitions=100)
    assert result['zero_relevant_queries']==['Q2']
    assert len(result['per_query'])==4


def test_annotation_agreement_pending_partial_and_full(example):
    _,_,_,a=example
    b=a.copy();b['annotator_id']='carol'
    assert annotation_agreement(a,b)['quadratic_weighted_kappa']==1
    a['relevance_grade']='';b['relevance_grade']=''
    assert annotation_agreement(a,b)['status']=='PENDING_ANNOTATIONS'
    a.loc[0,'relevance_grade']=3;b.loc[0,'relevance_grade']=3
    result=annotation_agreement(a,b)
    assert result['status']=='PARTIAL_RAW_RATINGS' and result['paired_ratings']==1
    assert result['quadratic_weighted_kappa'] is None


def test_annotation_agreement_independence_and_pool(example):
    _,_,_,a=example
    with pytest.raises(ValueError):annotation_agreement(a,a)
    with pytest.raises(ValueError):annotation_agreement(a,a.iloc[:-1])
    b=a.copy();b['annotator_id']='carol';b['relevance_grade']=b.relevance_grade.astype(float);b.loc[0,'relevance_grade']=1.5
    with pytest.raises(ValueError):annotation_agreement(a,b)


def test_annotation_agreement_matches_independent_library():
    from sklearn.metrics import cohen_kappa_score
    a=[0,1,2,3,2,1,3,0];b=[0,2,2,3,1,1,2,0]
    left=pd.DataFrame([label(doc_id=f'd{i}',relevance_grade=g) for i,g in enumerate(a)])
    right=pd.DataFrame([label(doc_id=f'd{i}',relevance_grade=g,annotator_id='carol') for i,g in enumerate(b)])
    assert annotation_agreement(left,right)['quadratic_weighted_kappa']==pytest.approx(cohen_kappa_score(a,b,weights='quadratic'))


def test_cluster_bootstrap_constant_paired_gain_and_reproducibility():
    a=np.array([[0.],[.1],[.2],[.4]])
    b=a+.2;groups=['g1','g1','g2','g3']
    first=paired_cluster_bootstrap(a,b,groups,['test_metric'],repetitions=100)
    second=paired_cluster_bootstrap(a,b,groups,['test_metric'],repetitions=100)
    pd.testing.assert_frame_equal(first,second)
    assert first.iloc[0].ci95_low==pytest.approx(.2)
    assert first.iloc[0].ci95_high==pytest.approx(.2)
    assert first.iloc[0].clusters==3 and first.iloc[0].queries==4


@pytest.mark.parametrize('a,b,groups',[
    ([[0]],[[1]],['only']),([[0],[1]],[[1]],['g1','g2']),
    ([[0],[1]],[[1],[2]],['g1',None]),([[0],[np.nan]],[[1],[2]],['g1','g2']),
])
def test_cluster_bootstrap_invalid_observations(a,b,groups):
    with pytest.raises(ValueError):paired_cluster_bootstrap(a,b,groups,['test_metric'],repetitions=100)


def test_file_evaluator_pending_and_no_overwrite(example,tmp_path):
    r,q,p,_=example
    frames=[]
    for method in METHODS:
        part=r[r.method.eq('A')].copy();part['method']=method;frames.append(part)
    pd.concat(frames).to_csv(tmp_path/'rankings.csv',index=False)
    q.to_csv(tmp_path/'queries.csv',index=False);p.to_csv(tmp_path/'pool.csv',index=False)
    output=tmp_path/'output'
    result=run(tmp_path/'rankings.csv',tmp_path/'queries.csv',tmp_path/'pool.csv',output)
    assert result['status']=='PENDING_LABELS'
    assert pd.read_csv(output/'summary.csv').Precision_at_5.isna().all()
    assert json.loads((output/'scope.json').read_text())['zero_idcg_queries']==[]
    with pytest.raises(ValueError):run(tmp_path/'rankings.csv',tmp_path/'queries.csv',tmp_path/'pool.csv',output)


@pytest.mark.parametrize('invalid',[True,'bad',float('inf'),1.5])
def test_invalid_run_count_rejected(example,invalid):
    r,q,p,l=example
    counts=r.groupby(['query_id','method']).size().rename('results_count').reset_index()
    counts['results_count']=counts.results_count.astype(object);counts.loc[0,'results_count']=invalid
    with pytest.raises(ValueError):evaluate_frames(r,q,p,l,methods=('A','B'),run_manifest=counts)


def test_no_primary_queries_rejected(example):
    r,q,p,_=example
    q['is_edge_case']=True
    with pytest.raises(ValueError):validate_rankings(r,q,p,methods=('A','B'))
