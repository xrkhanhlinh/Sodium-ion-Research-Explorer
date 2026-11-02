"""Point-ID filtering must preserve the existing payload-filter semantics."""
import numpy as np
import pandas as pd
import pytest
from qdrant_client import models
from src import final_system_vectors as vectors


@pytest.fixture
def database(tmp_path,monkeypatch):
    meta=pd.DataFrame(dict(doc_id=['a','b','c','d'],source=['Publication','Patent','Publication','Publication'],
                           year=[2021,2022,np.nan,2019]))
    monkeypatch.setattr(vectors,'corpus',lambda:meta.copy())
    db=vectors.VectorDatabase(tmp_path/'qdrant',tmp_path)
    matrix=np.zeros((4,768),dtype='float32')
    matrix[0,0]=1;matrix[1,1]=1;matrix[2,:2]=1/np.sqrt(2);matrix[3,:2]=[.6,.8]
    db.client.create_collection('probe',vectors_config=models.VectorParams(size=768,distance=models.Distance.COSINE))
    payloads=[dict(doc_id=r.doc_id,source=r.source,year=None if pd.isna(r.year) else int(r.year)) for r in meta.itertuples()]
    db.client.upsert('probe',models.Batch(ids=list(range(4)),vectors=matrix.tolist(),payloads=payloads))
    db.collections['probe']=dict(name='probe',documents=4,corpus_sha256=vectors.sha(vectors.ROOT/'data/processed/main_corpus.parquet'))
    try:yield db,matrix[0]
    finally:db.close()


@pytest.mark.parametrize('source,years,expected',[
    ('All',None,['a','c','d','b']),('Publication',(2020,2023),['a']),
    ('Publication',None,['a','c','d']),('Publication',(2019,2019),['d'])])
def test_allowed_id_translation_matches_payload_filter_exactly(database,source,years,expected):
    db,query=database;ids=['d','c','b','a','unknown','a']
    old=db.client.query_points('probe',query=query.tolist(),limit=10,
        query_filter=vectors.metadata_filter(source,years,ids)).points
    new=db.search('probe',query,10,source,years,ids)
    assert [doc for doc,score in new]==expected==[hit.payload['doc_id'] for hit in old]
    assert np.allclose([score for doc,score in new],[hit.score for hit in old],atol=1e-7)


def test_unknown_allowed_ids_never_broaden_search(database):
    db,query=database
    assert db.search('probe',query,10,doc_ids=['unknown'])==[]
    assert db.search('probe',query,10,doc_ids=[])==[]


@pytest.mark.parametrize('field,value',[('corpus_sha256','wrong'),('documents',3)])
def test_id_translation_rejects_unverified_mapping(database,field,value):
    db,query=database;db.collections['probe'][field]=value
    with pytest.raises(ValueError,match='Cannot translate document IDs'):
        db.search('probe',query,10,doc_ids=['a'])
