"""Behavioral regressions; synthetic mini-corpora, not relevance labels."""
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.search_v2 import (lexical_tokens, normalize, reciprocal_rank_fusion,
                           ranked_scores, select_passages, SearchEngineV2)
from src.search_v2_constraints import EvidenceSearchEngineV2


def test_formulas_and_case_do_not_collapse_in_lexical_channel():
    assert 'chem:Na2FeP2O7' in lexical_tokens('Na₂FeP₂O₇ cathode')
    assert 'chem:Na3V2(PO4)3' in lexical_tokens('Na3V2(PO4)(3)/C')
    assert lexical_tokens('Co') != lexical_tokens('CO')
    assert lexical_tokens('FeS') != lexical_tokens('Fes')
    assert 'chem:Na2FeP2O7' in lexical_tokens('Na-2FeP2O7')


def test_salt_names_have_distinct_channels_and_no_placeholder_leak():
    assert 'chem:NaFSI' in lexical_tokens('sodium bis(fluorosulfonyl)imide')
    assert 'chem:NaTFSI' in lexical_tokens('sodium bis(trifluoromethanesulfonyl)imide')
    assert 'chem:NaFSI' not in lexical_tokens('NaTFSI')
    tokens = lexical_tokens('all-solid-state sodium-ion battery')
    assert not any('placeholder' in t for t in tokens)
    assert 'solid' in tokens and 'state' in tokens and 'sodium' in tokens


def test_field_ranking_filters_before_limit_and_breaks_ties_by_id():
    result = ranked_scores(np.array([.9, .8, .8, .7]), ['d', 'c', 'b', 'a'],
                           np.array([False, True, True, True]), 2)
    assert [x[0] for x in result] == ['b', 'c']
    assert ranked_scores(np.array([0., -1]), ['a', 'b'], np.ones(2, dtype=bool), 5, True) == []


def test_rrf_promotes_shared_candidate_without_score_scale_dependency():
    r = reciprocal_rank_fusion([[('a', 1000), ('b', 50)], [('b', -.1), ('c', -20)]])
    assert r[0][0] == 'b'
    assert dict(r)['b'] == pytest.approx(1 / 62 + 1 / 61)


class WhitespaceTokenizer:
    def __call__(self, text, **kwargs):
        spans = [m.span() for m in re.finditer(r'\S+', text)]
        return dict(input_ids=list(range(len(spans))), offset_mapping=spans)


def test_reranking_can_read_query_evidence_beyond_old_first_window():
    abstract = ' '.join(['background'] * 700 + ['NaFSI', 'ether', 'low-temperature'] + ['end'] * 30)
    passages = select_passages('electrolyte study', abstract, 'NaFSI ether low-temperature',
                               WhitespaceTokenizer(), width=100, overlap=20, maximum=2)
    assert passages[0]['token_start'] >= 600
    assert 'NaFSI ether low-temperature' in passages[0]['text']
    for p in passages:
        assert p['text'].endswith(abstract[p['start']:p['end']])


def test_passage_pair_budget_and_title_only_input():
    tok = WhitespaceTokenizer()
    title = ' '.join(['title'] * 500)
    passages = select_passages(title, ' '.join(['abstract'] * 700), ' '.join(['query'] * 128), tok)
    assert all(len(tok(p['text'])['input_ids']) + 128 + 3 <= 512 for p in passages)
    assert select_passages('title only', '', 'query', tok)[0]['text'] == 'title only'
    with pytest.raises(ValueError, match='128'):
        select_passages('title', 'abstract', ' '.join(['query'] * 129), tok)


@pytest.mark.parametrize('settings', [(64, 64, 1), (100, -1, 1), (100, 20, 0)])
def test_invalid_passage_settings(settings):
    with pytest.raises(ValueError):
        select_passages('title', 'abstract', 'query', WhitespaceTokenizer(), *settings)


def tiny_engine():
    engine = object.__new__(SearchEngineV2)
    engine.config = dict(candidate_depth=100, rrf_k=60)
    engine.ids = ['old', 'new', 'missing']
    engine.meta = pd.DataFrame(dict(source=['Patent', 'Publication', 'Patent'],
                                    year=[2010, 2022, np.nan]))
    return engine


def test_source_and_year_filters_include_only_eligible_documents():
    e = tiny_engine()
    assert e.allowed('Publication', (2020, 2023)).tolist() == [False, True, False]
    assert e.allowed('Patent', (2020, 2023)).tolist() == [False, False, False]
    with pytest.raises(ValueError):
        e.allowed('Other')
    with pytest.raises(ValueError):
        e.allowed(years=(2023, 2020))


def test_hybrid_abstains_without_lexical_support_before_neural_inference():
    e = tiny_engine()
    e.lexical = lambda *_: []
    def unexpected(*_):
        raise AssertionError('Dense inference should not be called')
    e.dense = unexpected
    ranks, details = e.rank('zzzxqvzzzz987654', 'HYBRID_PRETRAINED')
    assert ranks == [] and details['status'] == 'NO_LEXICAL_SUPPORT'
    assert 'May abstain' in details['limitation']


def test_no_result_after_filter_does_not_run_reranker():
    e = tiny_engine()
    ranks, details = e.rank('electrode', 'RERANK_PRETRAINED', 'Patent', (2020, 2023))
    assert ranks == [] and details['status'] == 'NO_DOCUMENTS_AFTER_FILTER'


@pytest.mark.parametrize('query', ['', ' ', '???', 'x' * 10001])
def test_invalid_query_rejected(query):
    with pytest.raises(ValueError):
        tiny_engine().rank(query, 'BM25_CHEMICAL')


def test_missing_abstract_normalization_does_not_index_nan():
    assert normalize(None) == normalize(np.nan) == ''
    assert 'nan' not in lexical_tokens(np.nan)


def test_required_formula_filter_uses_full_fields_and_preserves_salt_identity():
    e = object.__new__(EvidenceSearchEngineV2)
    e.ids = ['a', 'b', 'c']
    e.title_index = SimpleNamespace(doc_freqs=[{}, {'chem:NaTFSI': 1}, {}])
    e.abstract_index = SimpleNamespace(doc_freqs=[{'chem:NaFSI': 1, 'ether': 1},
                                                {'ether': 1}, {'chem:NaFSI': 1}])
    assert e.evidence_filter(['NaFSI', 'ether']).tolist() == [True, False, False]
    assert e.evidence_filter(['NaFSI']).tolist() == [True, False, True]
    with pytest.raises(ValueError):
        e.evidence_filter('NaFSI')
    with pytest.raises(ValueError):
        e.evidence_filter(['???'])


def test_evidence_filter_reset_after_failed_search():
    e = object.__new__(EvidenceSearchEngineV2)
    e.evidence_mask = None
    e.evidence_filter = lambda _: np.array([True])
    with pytest.raises(ValueError):
        e.search('query', required_terms=['NaFSI'], top_k=0)
    assert e.evidence_mask is None
