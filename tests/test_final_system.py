"""Meaningful final-system contracts; full-corpus checks run in the CLI too."""
import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient, models
from src.final_system_vectors import metadata_filter, VectorDatabase
from src.final_system_evaluation import validate_labels, metrics_at_10, keys
from src.final_system_artifacts import OUT, TOPICS, topic_data, corpus, vector_spec
from src.final_search import FinalSearchEngine, METHODS


def test_exact_qdrant_persistence_and_filters(tmp_path):
    path=tmp_path/'persistent'
    client=QdrantClient(path=str(path))
    client.create_collection('probe',vectors_config=models.VectorParams(size=768,distance=models.Distance.COSINE))
    vectors=np.zeros((3,768),dtype='float32');vectors[0,0]=1;vectors[1,1]=1;vectors[2,:2]=1/np.sqrt(2)
    client.upsert('probe',models.Batch(ids=[0,1,2],vectors=vectors.tolist(),payloads=[
        dict(doc_id='a',source='Publication',year=2021),dict(doc_id='b',source='Patent',year=2022),
        dict(doc_id='c',source='Publication',year=None)]))
    client.close()
    client=QdrantClient(path=str(path))
    try:
        hits=client.query_points('probe',query=vectors[0].tolist(),limit=3,search_params=models.SearchParams(exact=True)).points
        assert [h.id for h in hits]==[0,2,1]
        assert np.allclose([h.score for h in hits],(vectors@vectors[0])[[0,2,1]],atol=2e-6)
        hits=client.query_points('probe',query=vectors[0].tolist(),limit=3,query_filter=metadata_filter('Publication',(2020,2023))).points
        assert [h.id for h in hits]==[0] # Missing year excluded by numeric filter.
        hits=client.query_points('probe',query=vectors[0].tolist(),limit=3,query_filter=metadata_filter(doc_ids=['b'])).points
        assert [h.id for h in hits]==[1]
    finally:client.close()


@pytest.mark.parametrize('source,years',[('Other',None),('All',(2023,2020))])
def test_invalid_qdrant_filter(source,years):
    with pytest.raises(ValueError):metadata_filter(source,years)


def test_relevance_formula_known_case():
    result=metrics_at_10([0,3,2],[3,2,0])
    assert result['Precision_at_10']==.2
    assert result['MRR_at_10']==.5
    expected=(7/np.log2(3)+3/np.log2(4))/(7+3/np.log2(3))
    assert result['nDCG_at_10']==pytest.approx(expected)
    assert metrics_at_10([], [0])==dict(nDCG_at_10=0.,Precision_at_10=0.,MRR_at_10=0.)
    with pytest.raises(ValueError):metrics_at_10([3],[0])
    with pytest.raises(ValueError):metrics_at_10([4],[4])
    with pytest.raises(ValueError):metrics_at_10([True],[True])


def label_frame():
    return pd.DataFrame([dict(query_id='Q',doc_id='D',relevance_grade=3,annotator_id='A',verifier_id='B',
        verification_reference='Review record 1',label_method='HUMAN_INDEPENDENT',label_status='VERIFIED')])


@pytest.mark.parametrize('field,value',[('relevance_grade',-1),('relevance_grade',2.5),('relevance_grade',True),
    ('label_method','GEMINI_LLM_JUDGE'),('label_status','UNVERIFIED'),('verifier_id','A'),('verification_reference','tbd')])
def test_reject_invalid_and_nonindependent_labels(field,value):
    frame=label_frame();pool=frame[['query_id','doc_id']].copy();frame[field]=value
    with pytest.raises(ValueError):validate_labels(frame,pool)


def test_partial_labels_remain_absent_not_zero():
    frame=label_frame();pool=frame[['query_id','doc_id']].copy()
    frame['relevance_grade']=''
    assert validate_labels(frame,pool).empty
    frame=label_frame();assert validate_labels(frame,pool).relevance_grade.tolist()==[3]
    with pytest.raises(ValueError):validate_labels(pd.concat([frame,frame]),pool)
    frame['doc_id']='outside'
    with pytest.raises(ValueError):validate_labels(frame,pool)


def test_all_reused_topic_assignments_cover_real_corpus():
    ids=corpus().doc_id.tolist()
    for name in TOPICS:
        data=topic_data(name)
        assert data['assignments'].doc_id.tolist()==ids
        assert len(set(data['assignments'].topic_id)-{-1})==10
        assert data['summary'].documents.sum()==13702
        assert data['keywords'].groupby('topic_id').size().eq(10).all()
        assert set(data['keywords'].topic_id)==set(range(10))
        assert np.isfinite(data['keywords'].weight).all()
        assert data['sources'].documents.sum()==13702


def test_real_tfidf_filter_constraints_and_top10():
    engine=FinalSearchEngine()
    try:
        result=engine.search('Na2FeP2O7 cathode cycling stability','TFIDF_COSINE','Publication',(2020,2023),required_terms=['Na2FeP2O7'])
        assert 1<=len(result['results'])<=10
        assert len({r['doc_id'] for r in result['results']})==len(result['results'])
        assert all(r['source']=='Publication' and 2020<=r['year']<=2023 for r in result['results'])
        assert all(r['topic_label'] and r['evidence_passage'] for r in result['results'])
        expected_mask=engine.v2.evidence_filter(['Na2FeP2O7'])
        assert all(expected_mask[engine.v2.positions[r['doc_id']]] for r in result['results'])
        assert engine.search('zzzxqvzzzz987654','TFIDF_COSINE')['results']==[]
        assert engine.search('carbon','TFIDF_COSINE',years=(1500,1501))['results']==[]
        engine._load_tfidf()
        q=engine.tfidf.transform(['hard carbon sodium'])
        scores=(engine.tfidf_matrix@q.T).toarray().ravel()
        from sklearn.metrics.pairwise import cosine_similarity
        direct=cosine_similarity(engine.tfidf_matrix,q).ravel()
        assert np.allclose(scores,direct,atol=1e-6)
        assert engine.tfidf_matrix.shape[0]==13702
    finally:engine.close()


def test_search_v2_top10_extension_preserves_top5_without_changing_method(monkeypatch):
    # Exercise the real adapter with deterministic model-free V2 rankings. The
    # separate benchmark compares every original saved Top5 using real models.
    engine=FinalSearchEngine()
    try:
        ids=engine.ids[:20]
        def rank(query,method,source,years):
            assert method=='RERANK_PRETRAINED'
            return [(doc,20-i) for i,doc in enumerate(ids)],dict(status='OK',eligible_documents=13702)
        monkeypatch.setattr(engine.v2,'rank',rank)
        a=engine.search('carbon','SEARCH_V2',top_k=5)
        b=engine.search('carbon','SEARCH_V2',top_k=10)
        assert len(a['results'])==5 and len(b['results'])==10
        assert [r['doc_id'] for r in a['results']]==[r['doc_id'] for r in b['results'][:5]]
        assert b['original_v2_method']=='RERANK_PRETRAINED'
    finally:engine.close()


def test_main_embedding_and_collection_provenance_when_available():
    if not (OUT/'batterybert_finetuned/embedding_success.json').exists():
        pytest.skip('Full inference still running; full integration CLI must pass before acceptance')
    spec=vector_spec('batterybert_finetuned')
    assert spec['dimension']==768
    info=__import__('json').loads((OUT/'batterybert_finetuned/embedding_success.json').read_text())
    assert info['full_abstract_coverage'] and info['pretrained_finetuned_cosine_mean']<.9999


def test_evaluation_requires_entire_shared_pool_and_common_queries(tmp_path,monkeypatch):
    import src.final_system_evaluation as evaluation
    # Explicitly synthetic UNIT fixtures; never written into research outputs.
    root=tmp_path/'fixture';output=root/'outputs/final_system';output.mkdir(parents=True)
    (root/'data/stage4').mkdir(parents=True)
    monkeypatch.setattr(evaluation,'ROOT',root);monkeypatch.setattr(evaluation,'OUT',output)
    pd.DataFrame([dict(query_id='Q1',is_edge_case=False),dict(query_id='Q2',is_edge_case=False)]).to_parquet(root/'data/stage4/queries.parquet')
    pool=pd.DataFrame([dict(query_id=q,doc_id=d) for q in ('Q1','Q2') for d in ('A','B')])
    pool.to_csv(output/'annotation_pool.csv',index=False)
    runs=[];ranks=[]
    for q in ('Q1','Q2'):
        for m in evaluation.METHODS:
            runs.append(dict(query_id=q,method=m,status='OK',result_count=2))
            ranks.extend([dict(query_id=q,method=m,doc_id='A',rank=1),dict(query_id=q,method=m,doc_id='B',rank=2)])
    pd.DataFrame(runs).to_csv(output/'run_manifest.csv',index=False)
    pd.DataFrame(ranks).to_csv(output/'rankings_top10.csv',index=False)
    evaluation.evaluate()
    summary=pd.read_csv(output/'relevance_summary.csv')
    assert summary.status.eq('PENDING_INDEPENDENT_LABELS').all() and summary.nDCG_at_10.isna().all()
    frames=[]
    for q,d,grade in [('Q1','A',3),('Q1','B',0),('Q2','A',2)]:
        row=label_frame();row['query_id']=q;row['doc_id']=d;row['relevance_grade']=grade;frames.append(row)
    labels=output/'fixture_labels.csv';pd.concat(frames).to_csv(labels,index=False)
    evaluation.evaluate(labels)
    summary=pd.read_csv(output/'relevance_summary.csv')
    assert summary.eligible_queries.eq(1).all() and summary.nDCG_at_10.eq(1).all()
    coverage=pd.read_csv(output/'label_coverage.csv').set_index('query_id')
    assert coverage.loc['Q2','status']=='PENDING_INDEPENDENT_LABELS'
    assert pd.read_csv(output/'paired_bootstrap.csv').empty
