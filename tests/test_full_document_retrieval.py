from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import sparse
from src.full_document_retrieval import search


def engine_fixture():
    ids=['a','b','c']
    seen={}
    class Encoder:
        def encode(self,q):
            seen['query']=q
            return np.eye(1,768,dtype='float32')[0],dict(tokens=900,chunks=4,truncated=False,coverage=1.)
    class DB:
        def search(self,kind,vector,k,source,years,doc_ids=None):
            seen['doc_ids']=doc_ids
            return [(doc,score) for doc,score in [('b',.8),('a',.7),('c',.6)] if doc_ids is None or doc in doc_ids]
    engine=SimpleNamespace(ids=ids,full_document_encoder=Encoder(),full_document_database=DB())
    engine.meta=pd.DataFrame(dict(doc_id=ids,title=['A','B','C'],abstract=['aa','bb','cc'],source=['Publication']*3,year=[2020,pd.NA,2022],identifier=['x','y','z'])).set_index('doc_id')
    engine._research_topics=lambda:{'a':dict(topic_id=1),'b':dict(topic_id=2),'c':dict(topic_id=1)}
    engine.v2=SimpleNamespace(allowed=lambda s,y:np.ones(3,dtype=bool),evidence_filter=lambda x:np.array([True,False,True]),positions={d:i for i,d in enumerate(ids)})
    engine._release_bge=lambda:None;engine._release_battery=lambda:None
    engine._load_tfidf=lambda:None
    engine.tfidf=SimpleNamespace(transform=lambda q:sparse.csr_matrix([[1.,.5,0.]]))
    engine.tfidf_matrix=sparse.eye(3,format='csr')
    engine._attach=lambda response,meaning,research:dict(**response,score_method=meaning)
    return engine,seen


def test_full_document_dense_long_query_is_not_lexically_gated(monkeypatch):
    from src import full_document_inference as full
    monkeypatch.setattr(full,'ready',lambda:True)
    engine,seen=engine_fixture()
    r=search(engine,'q '*900,'BATTERYBERT_FULLDOC_SEMANTIC','All',None,(),3,None,0.)
    assert len(r['results'])==3 and r['results'][0]['doc_id']=='b'
    assert r['results'][0]['year'] is None
    assert r['query_coverage']['tokens']==900 and r['query_coverage']['truncated'] is False
    assert seen['query']=='q '*900


def test_full_document_hybrid_filters_before_both_channels_and_rrf_uses_ranks(monkeypatch):
    from src import full_document_inference as full
    monkeypatch.setattr(full,'ready',lambda:True)
    engine,seen=engine_fixture()
    r=search(engine,'query','BATTERYBERT_FULLDOC_HYBRID','All',None,['required'],3,1,0.)
    assert seen['doc_ids']==['a','c']
    assert [d['doc_id'] for d in r['results']]==['a','c']
    assert r['results'][0]['rrf_score']==1/61+1/61
    assert r['results'][1]['lexical_rank'] is None
    assert r['checkpoint']=='batterybert_fulldoc'


def test_search_topic_cache_refreshes_when_an_accepted_catalog_changes(monkeypatch):
    from src.final_search import FinalSearchEngine
    from src import optimized_topics,full_document_topics,revised_topics
    # This fixture tests the optimized -> full-document transition only.
    # Do not let an on-disk newer catalog override the synthetic one-ID corpus.
    monkeypatch.setattr(revised_topics,'ready',lambda:False)
    state={'new':False}
    monkeypatch.setattr(full_document_topics,'ready',lambda:state['new'])
    monkeypatch.setattr(optimized_topics,'ready',lambda:True)
    def frame(topic,label):return pd.DataFrame(dict(doc_id=['a'],topic_id=[topic],topic_label=[label]))
    monkeypatch.setattr(optimized_topics,'assignments',lambda ids:frame(1,'old'))
    monkeypatch.setattr(full_document_topics,'assignments',lambda ids:frame(8,'new'))
    engine=FinalSearchEngine.__new__(FinalSearchEngine);engine.ids=['a'];engine.research_topic_map=None
    assert engine._research_topics()['a']==dict(topic_id=1,label='old')
    state['new']=True
    assert engine._research_topics()['a']==dict(topic_id=8,label='new')
