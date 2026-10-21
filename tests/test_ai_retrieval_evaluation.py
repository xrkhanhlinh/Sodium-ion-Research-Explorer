"""Synthetic unit fixtures only: never research labels or network calls."""
from copy import deepcopy
import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.ai_judge import (RUBRIC, PendingAPI, GeminiFreeClient, blind_input,
                          fingerprint, validate_judgment, response_json, read_json)
from src.ai_judge import judge
from src.ai_retrieval_evaluation import (AI_STATUS, agreement, consensus_record, evaluate_ai)
from src.ai_synthetic_benchmark import recovery_metrics, recovery_bootstrap, select_documents


DOC={'query_id':'Q01','doc_id':'fixture-doc','query':'aqueous sodium water-in-salt electrolyte',
     'title':'Water-in-salt sodium electrolytes','abstract':'Aqueous sodium cells use concentrated salt electrolytes.',
     'research_intent':'Fixture only, not a scientific judgment.'}


def vote(role='A',grade=3,doc=None):
    doc=doc or DOC
    return {'role':role,'requested_model':'fixture-model','model_version':'fixture-v1','evidence_valid':True,
            'request_sha256':fingerprint([role,doc]),'judgment':{
                'query_id':doc['query_id'],'doc_id':doc['doc_id'],'grade':grade,
                'evidence':[{'field':'abstract','quote':doc['abstract']}],
                'reason':'Synthetic unit fixture; not a research relevance label.',
                'confidence':.5,'model_id':'fixture-model',
                'axes':{key:'MATCH' for key in ('battery_system','material_formula','technical_condition','technical_relation')}}}


def test_blind_full_document_allowlist():
    original=dict(DOC,method='SECRET_METHOD',rank=1,score=.99,source='Patent',authors='Private Name')
    original['abstract']='Long full abstract '+('last part of text '*2000)
    result=blind_input(original,'intent')
    assert set(result)=={'query_id','doc_id','query','title','abstract','research_intent'}
    assert result['abstract']==original['abstract']
    assert 'SECRET_METHOD' not in json.dumps(result)


@pytest.mark.parametrize('grade',[True,False,-1,4,3.1,float('nan'),float('inf'),'nan',None,''])
def test_reject_invalid_grade(grade):
    value=vote()['judgment'];value['grade']=grade
    with pytest.raises(ValueError):validate_judgment(value,DOC,'fixture-model')


@pytest.mark.parametrize('quote',['fabricated quote','','   ','Aqueous sodium cells use concentrated salt electrolytes. EXTRA'])
def test_reject_fabricated_or_blank_evidence(quote):
    value=vote()['judgment'];value['evidence']=[{'field':'abstract','quote':quote}]
    with pytest.raises(ValueError):validate_judgment(value,DOC,'fixture-model')


@pytest.mark.parametrize('confidence',[True,-.1,1.1,float('nan'),'0.9'])
def test_invalid_confidence(confidence):
    value=vote()['judgment'];value['confidence']=confidence
    with pytest.raises(ValueError):validate_judgment(value,DOC,'fixture-model')


def test_unresolved_does_not_need_fabricated_evidence():
    value=vote()['judgment'];value.update(grade='UNRESOLVED',evidence=[])
    assert validate_judgment(value,DOC,'fixture-model')['grade']=='UNRESOLVED'


def test_numeric_needs_evidence_and_identity():
    for field,value in [('evidence',[]),('doc_id','another-doc'),('model_id','fake-model'),('axes',{})]:
        raw=vote()['judgment'];raw[field]=value
        with pytest.raises(ValueError):validate_judgment(raw,DOC,'fixture-model')


def test_consensus_and_separate_calls():
    a,b=vote('A',2),vote('B',2)
    result=consensus_record(DOC,a,b)
    assert result['label_status']==AI_STATUS and result['relevance_grade']==2
    assert result['expert_verified'] is False
    with pytest.raises(ValueError):consensus_record(DOC,a,a)


def test_disagreement_waits_for_C_no_majority_fabrication():
    a,b=vote('A',0),vote('B',3)
    assert consensus_record(DOC,a,b)['relevance_grade'] is None
    assert consensus_record(DOC,a,b)['label_status']=='PENDING_ADJUDICATION'
    c=vote('C','UNRESOLVED')
    assert consensus_record(DOC,a,b,c)['label_status']=='UNRESOLVED'
    assert consensus_record(DOC,a,b,c)['relevance_grade'] is None
    assert consensus_record(DOC,a,b,vote('C',1))['relevance_grade']==1


def test_agreement_reports_undefined_kappa_and_missing_pairs():
    votes={(r,'Q01','fixture-doc'):vote(r,3) for r in ('A','B')}
    value=agreement([DOC],votes)
    assert value['exact_agreement']==1 and value['quadratic_weighted_kappa'] is None
    assert agreement([DOC],{})['exact_agreement'] is None


def test_quadratic_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score
    docs=[];votes={};aa=[0,1,2,3,3];bb=[1,1,3,3,0]
    for i,(a,b) in enumerate(zip(aa,bb)):
        doc=dict(DOC,doc_id='fixture'+str(i));docs.append(doc)
        votes[('A','Q01',doc['doc_id'])]=vote('A',a,doc)
        votes[('B','Q01',doc['doc_id'])]=vote('B',b,doc)
    result=agreement(docs,votes)
    assert result['quadratic_weighted_kappa']==pytest.approx(cohen_kappa_score(aa,bb,weights='quadratic',labels=[0,1,2,3]))


def test_api_denied_before_key_read_or_network(tmp_path):
    with pytest.raises(PendingAPI,match='AUTHORIZATION'):
        GeminiFreeClient(tmp_path,{}, {'upload_authorized':False,'billing_disabled':True},tmp_path/'nonexistent.env')


@pytest.mark.parametrize('finish',['MAX_TOKENS','SAFETY',''])
def test_truncated_response_never_becomes_grade(finish):
    with pytest.raises(ValueError):response_json({'response':{'candidates':[{'finishReason':finish}]}})


@pytest.mark.parametrize('rank,expected',[(1,(1,1,1,1)),(5,(0,1,1,.2)),(10,(0,0,1,.1)),(None,(0,0,0,0))])
def test_exact_source_recovery(rank,expected):
    ids=['x'+str(i) for i in range(10)]
    if rank:ids[rank-1]='target'
    assert tuple(recovery_metrics(ids,'target').values())==expected


def test_recovery_duplicate_invalid():
    with pytest.raises(ValueError):recovery_metrics(['a','a'],'a')


def test_balanced_selection_frozen_and_no_missing_abstract():
    frame=pd.DataFrame([{'doc_id':s+str(i),'source':s,'title':'fixture title','abstract':'fixture abstract',
                         'split_group_id':s+str(i)} for s in ('Patent','Publication') for i in range(110)])
    result=select_documents(frame)
    assert len(result)==200 and result.source.value_counts().to_dict()=={'Patent':100,'Publication':100}
    assert result.equals(select_documents(frame))


def evaluation_fixture():
    queries=pd.DataFrame([dict(query_id=q,is_edge_case=False) for q in ('Q01','Q02')])
    pool=pd.DataFrame([dict(query_id=q,doc_id=q+'_'+str(i)) for q in ('Q01','Q02') for i in range(6)])
    ranks=pd.DataFrame([dict(query_id=q,doc_id=q+'_'+str(i),method=m,rank=i+1)
                        for q in ('Q01','Q02') for m in ('M1','M2') for i in range(5)])
    runs=pd.DataFrame([dict(query_id=q,method=m,results_count=5) for q in ('Q01','Q02') for m in ('M1','M2')])
    labels=[]
    for row in pool.to_dict('records'):
        doc=dict(DOC,**row)
        labels.append(consensus_record(doc,vote('A',2,doc),vote('B',2,doc)))
    return ranks,queries,pool,pd.DataFrame(labels),runs


def test_incomplete_pool_excludes_same_query_all_methods():
    ranks,queries,pool,labels,runs=evaluation_fixture()
    labels.loc[labels.index[0],['relevance_grade','label_status']]=[None,'UNRESOLVED']
    result=evaluate_ai(ranks,queries,pool,labels,runs,methods=('M1','M2'),repetitions=100)
    assert result['scope']['complete_query_pools']==['Q02']
    assert result['per_query'].query("query_id == 'Q01'").Precision_at_5.isna().all()
    assert result['summary'].query("split == 'development'").Precision_at_5.isna().all()
    assert result['summary'].query("split == 'remaining'").Precision_at_5.eq(1).all()


def test_missing_labels_are_not_zero():
    ranks,queries,pool,labels,runs=evaluation_fixture()
    labels['relevance_grade']=None;labels['label_status']='PENDING_JUDGES'
    result=evaluate_ai(ranks,queries,pool,labels,runs,methods=('M1','M2'),repetitions=100)
    assert result['per_query'].Precision_at_5.isna().all()
    assert not len(result['confidence_intervals'])


def test_ai_cannot_claim_human_ground_truth():
    ranks,queries,pool,labels,runs=evaluation_fixture();labels['expert_verified']=True
    with pytest.raises(ValueError):evaluate_ai(ranks,queries,pool,labels,runs,methods=('M1','M2'))
    labels['expert_verified']=False;labels['label_method']='HUMAN_INDEPENDENT'
    with pytest.raises(ValueError):evaluate_ai(ranks,queries,pool,labels,runs,methods=('M1','M2'))


def test_bootstrap_same_paired_queries_and_splits():
    ranks,queries,pool,labels,runs=evaluation_fixture()
    result=evaluate_ai(ranks,queries,pool,labels,runs,methods=('M1','M2'),repetitions=100)
    assert result['summary'].query("split == 'all'").queries_evaluated.eq(2).all()
    assert result['paired_deltas'].mean_delta.eq(0).all()


def fake_client(tmp_path):
    env=tmp_path/'fixture.env';env.write_text('GEMINI_API_KEY=unit-test-only-fake-credential',encoding='utf-8')
    config={'paid_fallback':False,'tools_enabled':False,'requests_per_minute':60,
            'max_calls_per_invocation':10,'max_output_tokens':512,'temperature':.2,
            'free_standard_models':['fixture-model']}
    client=GeminiFreeClient(tmp_path,config,{'upload_authorized':True,'billing_disabled':True,
        'credential_sha256':hashlib.sha256(b'unit-test-only-fake-credential').hexdigest()},env)
    client.available={'fixture-model'}
    return client


class FakeResponse:
    def __init__(self,status,body):self.status_code=status;self.body=body
    def json(self):return self.body


def test_stateless_call_cache_separates_A_B(tmp_path,monkeypatch):
    client=fake_client(tmp_path);calls=[]
    def request(method,url,**kwargs):
        calls.append((method,url,kwargs))
        assert '?' not in url and 'unit-test-only-fake-credential' not in url
        return FakeResponse(200,{'modelVersion':'fixture-v1','candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{}'}]}}]})
    monkeypatch.setattr(client.session,'request',request)
    monkeypatch.setattr('src.ai_judge.time.sleep',lambda _:None)
    a=client.generate('JUDGE_A','fixture-model','rubric',DOC,{})
    assert client.generate('JUDGE_A','fixture-model','rubric',DOC,{})==a
    b=client.generate('JUDGE_B','fixture-model','rubric',DOC,{})
    assert len(calls)==2 and a['request_sha256']!=b['request_sha256']
    assert 'unit-test-only-fake-credential' not in ''.join(p.read_text() for p in (tmp_path/'api_cache').glob('*.json'))


def test_error_redaction_and_no_paid_fallback(tmp_path,monkeypatch):
    client=fake_client(tmp_path)
    monkeypatch.setattr(client.session,'request',lambda *args,**kwargs:FakeResponse(403,{'error':{
        'status':'PERMISSION_DENIED','message':'unit-test-only-fake-credential was rejected'}}))
    with pytest.raises(PendingAPI,match='403'):client._request('GET','https://example.test')
    assert 'unit-test-only-fake-credential' not in (tmp_path/'last_api_error.json').read_text()
    assert '[REDACTED]' in (tmp_path/'last_api_error.json').read_text()


def test_daily_quota_stops_without_retry(tmp_path,monkeypatch):
    client=fake_client(tmp_path);calls=[]
    def request(*args,**kwargs):
        calls.append(1)
        return FakeResponse(429,{'error':{'status':'RESOURCE_EXHAUSTED','details':[{'violations':[
            {'quotaId':'GenerateRequestsPerDayPerProjectPerModel-FreeTier','quotaValue':'20'}]}]}})
    monkeypatch.setattr(client.session,'request',request)
    with pytest.raises(PendingAPI,match='429'):client._request('GET','https://example.test')
    assert len(calls)==1


def test_minute_quota_adapts_then_retries(tmp_path,monkeypatch):
    client=fake_client(tmp_path);client.rpm=60;calls=[];sleeps=[]
    def request(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:
            return FakeResponse(429,{'error':{'status':'RESOURCE_EXHAUSTED','details':[
                {'violations':[{'quotaId':'GenerateRequestsPerMinutePerProjectPerModel-FreeTier','quotaValue':'10'}]},
                {'retryDelay':'2s'}]}})
        return FakeResponse(200,{'models':[]})
    monkeypatch.setattr(client.session,'request',request)
    monkeypatch.setattr('src.ai_judge.time.sleep',sleeps.append)
    assert client._request('GET','https://example.test')=={'models':[]}
    assert client.rpm==10 and sleeps==[3.]


def test_zero_quota_never_retries_even_if_minute_dimension(tmp_path,monkeypatch):
    client=fake_client(tmp_path);calls=[]
    def request(*args,**kwargs):
        calls.append(1)
        return FakeResponse(429,{'error':{'status':'RESOURCE_EXHAUSTED','details':[{'violations':[
            {'quotaId':'GenerateRequestsPerMinutePerProjectPerModel-FreeTier','quotaValue':'0'}]}]}})
    monkeypatch.setattr(client.session,'request',request)
    with pytest.raises(PendingAPI):client._request('GET','https://example.test')
    assert len(calls)==1


def test_changed_key_requires_new_billing_attestation(tmp_path):
    env=tmp_path/'fixture.env';env.write_text('GEMINI_API_KEY=different-fake-key',encoding='utf-8')
    with pytest.raises(PendingAPI,match='CHANGED_CREDENTIAL'):
        GeminiFreeClient(tmp_path,{'paid_fallback':False,'tools_enabled':False},
            {'upload_authorized':True,'billing_disabled':True,'credential_sha256':'old-binding'},env)


def test_source_recovery_paired_cluster_bootstrap():
    selected=pd.DataFrame([{'synthetic_id':sid,'split_group_id':group} for sid,group in [('s1','g1'),('s2','g1'),('s3','g2')]])
    per=pd.DataFrame([dict(synthetic_id=sid,method=method,Hit_at_1=1,Hit_at_5=1,Hit_at_10=1,MRR_at_10=1.)
                      for sid in ('s1','s2','s3') for method in ('BM25_CHEMICAL','DENSE_PRETRAINED')])
    ci,pairs=recovery_bootstrap(per,selected,repetitions=100)
    assert ci.clusters.eq(2).all() and ci.queries.eq(3).all()
    assert ci.ci95_low.eq(1).all() and pairs.mean_delta.eq(0).all()


def test_synthetic_bootstrap_rejects_unequal_query_coverage():
    selected=pd.DataFrame([{'synthetic_id':'s1','split_group_id':'g1'},{'synthetic_id':'s2','split_group_id':'g2'}])
    per=pd.DataFrame([dict(synthetic_id=s,method=m,Hit_at_1=0,Hit_at_5=0,Hit_at_10=0,MRR_at_10=0.)
                      for s,m in [('s1','BM25_CHEMICAL'),('s2','BM25_CHEMICAL'),('s1','DENSE_PRETRAINED')]])
    with pytest.raises(ValueError,match='common complete'):recovery_bootstrap(per,selected,repetitions=100)


class SequenceJudge:
    def __init__(self,output,values):self.output=output;self.values=values;self.calls=[]
    def generate(self,stage,model,system,payload,schema):
        self.calls.append(deepcopy(payload))
        value=self.values[min(len(self.calls)-1,len(self.values)-1)]
        return {'request_sha256':fingerprint([stage,len(self.calls)]),'server_model_version':'fixture-v1',
                'prompt_sha256':fingerprint(system),'input_sha256':fingerprint(payload),'time':'fixture-time',
                'response':{'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(value)}]}}]}}


def test_first_valid_grade_zero_is_never_retried_for_higher_grade(tmp_path):
    client=SequenceJudge(tmp_path,[vote('A',0)['judgment'],vote('A',3)['judgment']])
    result=judge(client,DOC,'A',{'judge_a':'fixture-model'})
    assert result['judgment']['grade']==0 and len(client.calls)==1


def test_structural_repair_preserves_rejection_and_B_stays_blind(tmp_path):
    bad=vote('B',3)['judgment'];bad['evidence'][0]['quote']='invented wording'
    client=SequenceJudge(tmp_path,[bad,vote('B',1)['judgment']])
    result=judge(client,DOC,'B',{'judge_b':'fixture-model'})
    assert result['judgment']['grade']==1 and result['validation_attempt']==1
    assert len(list((tmp_path/'rejected_judgments').glob('*.json')))==1
    assert all(set(payload)==set(DOC) for payload in client.calls)


def test_three_invalid_responses_remain_unresolved(tmp_path):
    bad=vote()['judgment'];bad['evidence']=[]
    client=SequenceJudge(tmp_path,[bad])
    with pytest.raises(ValueError,match='Three structural'):judge(client,DOC,'A',{'judge_a':'fixture-model'})
    assert len(client.calls)==3


def test_service_503_retries_without_model_fallback(tmp_path,monkeypatch):
    client=fake_client(tmp_path);calls=[];sleeps=[]
    def request(method,url,**kwargs):
        calls.append(url)
        return FakeResponse(503,{'error':{'status':'UNAVAILABLE'}}) if len(calls)==1 else FakeResponse(200,{'ok':True})
    monkeypatch.setattr(client.session,'request',request)
    monkeypatch.setattr('src.ai_judge.time.sleep',sleeps.append)
    assert client._request('POST','https://example.test')=={'ok':True}
    assert calls==['https://example.test','https://example.test'] and sleeps==[5.]
