"""Contracts for the existing engine's pretrained/dense/TF-IDF RRF extension."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from src.final_search import FinalSearchEngine,RESEARCH_METHODS,RESEARCH_OUT
from src.final_system_artifacts import ROOT,corpus,sha
from src.search_v2 import reciprocal_rank_fusion
from src.final_system_evaluation import validate_labels,metrics_at_10


def test_rrf_uses_ranks_and_deduplicates_across_channels():
    ranks=reciprocal_rank_fusion([[('A',.2),('B',.1)],[('B',.99),('C',.98)]],k=60)
    assert [d for d,_ in ranks]==['B','A','C']
    assert ranks[0][1]==pytest.approx(1/62+1/61)
    assert ranks==reciprocal_rank_fusion([[('A',200),('B',100)],[('B',-99),('C',-100)]],k=60)
    with pytest.raises(ValueError):reciprocal_rank_fusion([[('A',1),('A',.5)]])


def label_frame(grade=2):
    return pd.DataFrame([dict(query_id='Q1',doc_id='A',relevance_grade=grade,annotator_id='fixture_A',
        verifier_id='fixture_B',verification_reference='SYNTHETIC UNIT FIXTURE, NOT REAL RESEARCH',
        label_method='HUMAN_INDEPENDENT',label_status='VERIFIED')])


def test_zero_to_two_scale_and_known_metric_case():
    pool=pd.DataFrame([dict(query_id='Q1',doc_id='A')])
    assert validate_labels(label_frame(),pool,max_grade=2).relevance_grade.tolist()==[2]
    with pytest.raises(ValueError):validate_labels(label_frame(3),pool,max_grade=2)
    with pytest.raises(ValueError):validate_labels(label_frame(True),pool,max_grade=2)
    assert validate_labels(label_frame(''),pool,max_grade=2).empty
    result=metrics_at_10([1,0,2],[2,1,0])
    assert result['Precision_at_10']==.1 and result['MRR_at_10']==pytest.approx(1/3)
    assert result['nDCG_at_10']==pytest.approx((1+3/np.log2(4))/(3+1/np.log2(3)))


def test_existing_memberships_and_chemical_vocabulary_preserved():
    current=pd.read_csv(RESEARCH_OUT/'topics/topic_assignments.csv')
    old=pd.read_csv(ROOT/'outputs/batterybert_topic_audit_20261022_v1/topic_assignments.csv')
    assert current.doc_id.tolist()==old.doc_id.tolist()==corpus().doc_id.tolist()
    assert current.topic_id.tolist()==old.topic_id.tolist()
    assert current.doc_id.nunique()==13702 and current.topic_id.nunique()==10


def test_real_tfidf_topic_filter_before_ranking():
    engine=FinalSearchEngine()
    try:
        response=engine.search('carbon anode','TFIDF_COSINE',topic_id=3)
        assert len(response['results'])==10
        assert all(r['topic_id']==3 for r in response['results'])
        assert all(r['rrf_score'] is None for r in response['results'])
        assert len({r['doc_id'] for r in response['results']})==10
        assert engine.search('carbon','TFIDF_COSINE',years=(1500,1501))['results']==[]
        with pytest.raises(ValueError):engine.search('carbon','TFIDF_COSINE',topic_id=10)
        with pytest.raises(ValueError):engine.search('carbon','TFIDF_COSINE',topic_id=True)
    finally:engine.close()


def test_semantic_does_not_require_lexical_support_and_hybrid_preserves_score_meaning(monkeypatch):
    # Model-free deterministic fixture exercises the actual production routing.
    engine=FinalSearchEngine()
    try:
        ids=engine.ids[:10]
        class DatabaseFixture:
            def search(self,*args,**kwargs):return [(doc,.99-i*.001) for i,doc in enumerate(ids)]
            def close(self):pass
        engine.research_database=DatabaseFixture()
        monkeypatch.setattr(engine,'encode_battery',lambda q,variant='finetuned':np.eye(1,768,dtype=np.float32)[0])
        response=engine.search('zzzxqvzzzz987654','BATTERYBERT_SEMANTIC')
        assert len(response['results'])==10
        assert response['checkpoint']=='batterybert_pretrained'
        hybrid=engine.search('zzzxqvzzzz987654','BATTERYBERT_HYBRID')
        assert len(hybrid['results'])==10
        assert hybrid['results'][0]['rrf_score']==pytest.approx(1/61)
        assert hybrid['results'][0]['semantic_cosine']==.99
        assert hybrid['results'][0]['lexical_rank'] is None
    finally:engine.close()


def fixture_evaluation(tmp_path,monkeypatch):
    from src import research_completion_evaluation as evaluation
    from src.model_comparison_io import write_json
    monkeypatch.setattr(evaluation,'OUT',tmp_path)
    queries=pd.DataFrame([dict(query_id=q,query=q) for q in ('Q1','Q2')]);queries.to_csv(tmp_path/'queries.csv',index=False)
    pool=pd.DataFrame([dict(query_id=q,doc_id=d) for q in ('Q1','Q2') for d in ('A','B')]);pool.to_csv(tmp_path/'annotation_pool.csv',index=False)
    ranks=[];runs=[]
    for q in ('Q1','Q2'):
        for m in RESEARCH_METHODS:
            ranks.extend([dict(query_id=q,method=m,doc_id=d,rank=i) for i,d in enumerate(('A','B'),1)])
            runs.append(dict(query_id=q,method=m,status='OK',result_count=2))
    pd.DataFrame(ranks).to_csv(tmp_path/'rankings_top10.csv',index=False);pd.DataFrame(runs).to_csv(tmp_path/'run_manifest.csv',index=False)
    write_json(tmp_path/'annotation_protocol.json',dict(pool_sha256=sha(tmp_path/'annotation_pool.csv'),
        queries_sha256=sha(tmp_path/'queries.csv'),rankings_sha256=sha(tmp_path/'rankings_top10.csv')))
    return evaluation


def test_missing_human_labels_are_pending_not_zero(tmp_path,monkeypatch):
    evaluation=fixture_evaluation(tmp_path,monkeypatch)
    summary=evaluation.evaluate()
    assert summary.status.eq('PENDING_HUMAN_ANNOTATION').all()
    assert summary.Precision_at_10.isna().all()
    frames=[]
    for q,d,g in [('Q1','A',2),('Q1','B',0),('Q2','A',1)]:
        f=label_frame(g);f['query_id']=q;f['doc_id']=d;frames.append(f)
    path=tmp_path/'unit_fixture_labels.csv';pd.concat(frames).to_csv(path,index=False)
    summary=evaluation.evaluate(path)
    assert summary.eligible_queries.eq(1).all()
    assert summary.nDCG_at_10.eq(1.).all()
    assert summary.Precision_at_10.eq(.1).all()
    coverage=pd.read_csv(tmp_path/'annotation_coverage.csv').set_index('query_id')
    assert coverage.loc['Q2','status']=='PENDING_HUMAN_ANNOTATION'
    # A refresh must retain accepted labels and record the exact source bytes.
    repeated=evaluation.evaluate()
    assert repeated.eligible_queries.eq(1).all()
    state=json.loads((tmp_path/'evaluation_status.json').read_text())
    assert state['label_source_sha256']==sha(path)
    assert len(pd.read_csv(tmp_path/'accepted_human_labels.csv'))==3


def test_evaluation_rejects_changed_frozen_queries(tmp_path,monkeypatch):
    evaluation=fixture_evaluation(tmp_path,monkeypatch)
    (tmp_path/'queries.csv').write_text('changed')
    with pytest.raises(ValueError,match='changed'):evaluation.evaluate()


def test_qdrant_dimension_validation_without_rebuilding():
    from src.final_system_vectors import VectorDatabase
    obj=VectorDatabase.__new__(VectorDatabase);obj.collections={'batterybert_pretrained':{}}
    with pytest.raises(ValueError,match='768'):obj.search('batterybert_pretrained',np.ones(12))
    with pytest.raises(ValueError,match='normalized'):obj.search('batterybert_pretrained',np.ones(768))


def test_top15_scientific_keywords_and_complete_representatives():
    dest=RESEARCH_OUT/'topics'
    keys=pd.read_csv(dest/'topic_keywords.csv')
    assert keys.groupby('topic_id').size().eq(15).all()
    assert not keys.token.str.contains('placeholder').any()
    reps=pd.read_csv(dest/'topic_representative_documents.csv')
    assert reps.groupby('topic_id').size().ge(5).all()
    assert set(reps.doc_id)<=set(corpus().doc_id)
    similarity=pd.read_csv(dest/'topic_similarity.csv',index_col=0).to_numpy()
    assert similarity.shape==(10,10) and np.allclose(similarity,similarity.T)
    assert np.allclose(np.diag(similarity),1.,atol=2e-6)


def test_committed_benchmark_all_four_methods_and_blind_pool():
    ranks=pd.read_csv(RESEARCH_OUT/'rankings_top10.csv')
    assert len(ranks)==800 and set(ranks.method)==set(RESEARCH_METHODS)
    assert ranks.groupby(['query_id','method']).size().eq(10).all()
    assert not ranks.duplicated(['query_id','method','doc_id']).any()
    pool=pd.read_csv(RESEARCH_OUT/'annotation_template_0_2.csv')
    assert not set(pool.columns)&{'rank','score','method','topic_id','topic_label','semantic_cosine','rrf_score'}
    assert pool.relevance_grade.isna().all()
