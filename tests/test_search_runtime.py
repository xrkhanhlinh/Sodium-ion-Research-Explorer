"""Retrieval reuse must preserve filters, scores and request isolation."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import search_v2 as search
from src import search_v2_service as service


def bare_engine():
    engine = object.__new__(search.SearchEngineV2)
    engine._request_work = None
    engine.config = dict(candidate_depth=100, threads=1, title_weight=2, query_max_tokens=128)
    engine.ids = ['a', 'b', 'c']
    return engine


def test_live_engine_does_not_read_benchmark_query_file(monkeypatch):
    engine = bare_engine()
    engine.use_query_cache = False
    def unexpected(*args, **kwargs):
        pytest.fail('Live search opened benchmark query data')
    monkeypatch.setattr(pd, 'read_parquet', unexpected)
    engine._load_query_cache()
    assert engine.cached_queries == engine.query_vectors == {}


def test_offline_query_cache_keeps_normalized_query_mapping(monkeypatch):
    engine = bare_engine()
    engine.use_query_cache = True
    monkeypatch.setattr(pd, 'read_parquet', lambda *_: pd.DataFrame({'query':[' sodium  battery ', 'hard carbon']}))
    engine._load_query_cache()
    assert engine.cached_queries == {'sodium battery':0, 'hard carbon':1}


def test_request_work_is_released_on_failure_and_nested_calls():
    engine = bare_engine()
    with pytest.raises(RuntimeError):
        with engine.reuse_query_work():
            cache = engine._request_work
            cache['work'] = 1
            with engine.reuse_query_work():
                assert engine._request_work is cache
            raise RuntimeError('request failed')
    assert engine._request_work is None
    with engine.reuse_query_work():
        assert engine._request_work == {}


def test_encoder_reuses_same_checkpoint_only_inside_request():
    engine = bare_engine()
    calls = []
    engine.encoder_kind = 'finetuned'
    engine.use_query_cache = False
    engine.prefix = 'query: '
    def encode(texts, **kwargs):
        calls.extend(texts)
        return np.array([[1., 0.]])
    engine.encoder = SimpleNamespace(tokenizer=lambda *args, **kwargs: {'input_ids':[1,2]}, encode=encode)
    with engine.reuse_query_work():
        before = engine.encode('sodium battery', 'finetuned')
        after = engine.encode(' sodium  battery ', 'chunked')
        assert np.array_equal(before, after)
        assert len(calls) == 1
    engine.encode('sodium battery', 'finetuned')
    assert len(calls) == 2


def test_dense_reuse_reapplies_mask_and_separates_query_and_matrix():
    engine = bare_engine()
    calls = []
    engine.encode = lambda query, kind: calls.append((query,kind)) or np.array([1.,0.])
    engine._matrix = lambda kind: np.array([[3.,0.],[2.,0.],[1.,0.]])
    all_docs = np.ones(3,dtype=bool)
    with engine.reuse_query_work():
        assert engine.dense('sodium', 'pretrained', all_docs)[0] == ('a',3.)
        assert engine.dense('sodium', 'pretrained', np.array([False,True,True])) == [('b',2.),('c',1.)]
        assert len(calls) == 1
        engine.dense('carbon', 'pretrained', all_docs)
        engine.dense('sodium', 'chunked', all_docs)
        assert len(calls) == 3
    engine.dense('sodium', 'pretrained', all_docs)
    assert len(calls) == 4


def test_lexical_reuse_preserves_scores_and_applies_each_filter():
    engine = bare_engine()
    calls = []
    def scores(tokens):
        calls.append(tuple(tokens))
        return np.array([3.,2.,1.])
    engine.title_index = engine.abstract_index = SimpleNamespace(idf={'sodium':1.},get_scores=scores)
    with engine.reuse_query_work():
        assert engine.lexical('sodium', np.ones(3,dtype=bool)) == [('a',9.),('b',6.),('c',3.)]
        assert engine.lexical('sodium', np.array([False,True,False])) == [('b',6.)]
        assert len(calls) == 2
    engine.lexical('sodium', np.ones(3,dtype=bool))
    assert len(calls) == 4


def test_lexical_index_survives_unrelated_file_edit_and_tracks_token_rules(tmp_path,monkeypatch):
    monkeypatch.setattr(search, 'ROOT', tmp_path)
    monkeypatch.setattr(search, 'CACHE', tmp_path/'index')
    (tmp_path/'corpus.parquet').write_bytes(b'unchanged corpus')
    engine = bare_engine()
    engine.config.update(corpus='corpus.parquet',bm25_k1=1.5,bm25_b=.75)
    engine.meta = pd.DataFrame(dict(title=['sodium battery','hard carbon','electrolyte'],abstract=['storage','anode','salt']))
    engine._load_lexical()
    original_digest = search.digest
    monkeypatch.setattr(search, 'digest', lambda path: 'unrelated source edit' if Path(path)==Path(search.__file__) else original_digest(path))
    def rebuild(*args, **kwargs):
        pytest.fail('Unrelated source edit rebuilt lexical index')
    monkeypatch.setattr(search,'BM25Okapi',rebuild)
    engine._load_lexical()
    previous = search.lexical_fingerprint()
    monkeypatch.setattr(search,'SALT_ALIASES',{**search.SALT_ALIASES,'NaX':r'\bNaX\b'})
    assert search.lexical_fingerprint() != previous


def test_compare_reuses_work_on_worker_and_cleans_up_after_error(monkeypatch):
    calls = []
    class Engine:
        active = False
        @contextmanager
        def reuse_query_work(self):
            self.active = True
            try:
                yield
            finally:
                self.active = False
        def search(self, query, method, source, years, required_terms=()):
            calls.append((method,self.active,source,years,required_terms))
            if query == 'fail' and method == 'HYBRID_PRETRAINED':
                raise ValueError('failed branch')
            return dict(method=method,results=[])
        def close(self):
            pass
    monkeypatch.setattr(service,'EvidenceSearchEngineV2',Engine)
    adapter = service.SearchServiceV2()
    try:
        methods = ['BM25_CHEMICAL','HYBRID_PRETRAINED','RERANK_PRETRAINED']
        responses = adapter.compare('query',methods,'Publication',(2020,2023),('NaFSI',))
        assert all(row['retrieval_cache_scope']=='comparison_request' for row in responses)
        assert all(active and source=='Publication' and years==(2020,2023) and terms==('NaFSI',)
                   for _,active,source,years,terms in calls)
        with pytest.raises(ValueError):
            adapter.compare('fail',methods)
        assert not adapter.engine.active
        adapter.search('query','BM25_CHEMICAL')
        assert calls[-1][1] is False
    finally:
        adapter.close()
