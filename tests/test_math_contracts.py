"""Bounded synthetic contracts extending the seven-group independent audit.

These tests establish input safety and exact arithmetic, not search relevance.
"""
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest
import sympy as sp

from src.search_v2 import SearchEngineV2, reciprocal_rank_fusion, select_passages, validate_config, ranked_scores


def test_topk_cutoff_uses_same_float_domain_as_sort():
    # These unequal int64 values tie when the baseline key converts to float.
    scores = np.array([2**53, 2**53+1], dtype=np.int64)
    assert ranked_scores(scores, ['a', 'z'], np.ones(2, dtype=bool), 1) == [('a', float(2**53))]


def test_topk_keeps_all_cutoff_ties_for_id_ordering():
    scores = np.array([4., 4., 4., 3.])
    assert ranked_scores(scores, ['c', 'a', 'b', 'd'], np.ones(4, dtype=bool), 2) == [('a', 4.), ('b', 4.)]


class Tokenizer:
    def __call__(self, text, **kwargs):
        offsets = [m.span() for m in re.finditer(r'\S+', text)]
        return dict(input_ids=list(range(len(offsets))), offset_mapping=offsets)

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2


def test_rrf_exact_scores_and_symbolic_monotonicity():
    scores = dict(reciprocal_rank_fusion([[('a', 99), ('b', 0)], [('b', -3), ('c', 8)]]))
    assert scores == pytest.approx(dict(a=float(sp.Rational(1, 61)),
                                      b=float(sp.Rational(1, 61) + sp.Rational(1, 62)),
                                      c=float(sp.Rational(1, 62))))
    k = sp.Symbol('k', nonnegative=True)
    r = sp.Symbol('r', integer=True, positive=True)
    delta = sp.factor(1 / (k + r) - 1 / (k + r + 1))
    assert delta.is_positive is True
    assert reciprocal_rank_fusion([[('b', 0), ('a', 0)], [('a', 0), ('b', 0)]], 0)[0][0] == 'a'


@pytest.mark.parametrize('k', [-1, -.5, float('nan'), float('inf'), -float('inf'), True, '60', None])
def test_invalid_rrf_k_rejected_even_for_empty_rankings(k):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        reciprocal_rank_fusion([], k)


def test_rrf_duplicate_rejected_per_list_only():
    with pytest.raises(ValueError, match='Duplicate'):
        reciprocal_rank_fusion([[('a', 1), ('a', 0)]])
    assert reciprocal_rank_fusion([[('a', 1)], [('a', 1)]]) == [('a', 2 / 61)]


def test_effective_width_rejects_original_stalled_loop_counterexample():
    # Requested width 400 > overlap 301; effective width is only 301.
    with pytest.raises(ValueError, match='Effective passage width'):
        select_passages(' '.join(['title'] * 80), ' '.join(['abstract'] * 600),
                        ' '.join(['query'] * 128), Tokenizer(), 400, 301)


@pytest.mark.parametrize('max_length', [256, 384, 512])
def test_pair_budget_matches_configured_length(max_length):
    tok = Tokenizer()
    title, query = ' '.join(['title'] * 80), ' '.join(['query'] * 128)
    # Deliberately use valid overlap for even the smallest effective width.
    passages = select_passages(title, ' '.join(['abstract'] * 700), query, tok,
                               overlap=20, max_length=max_length)
    assert all(len(tok(p['text'])['input_ids']) + 128 + 3 <= max_length for p in passages)
    assert passages[0]['token_end'] == min(320, max_length - 3 - 128 - 80)


def test_tokenizer_specific_pair_overhead_is_respected():
    class FourSpecialTokens(Tokenizer):
        def num_special_tokens_to_add(self, pair=False):
            return 4
    p = select_passages('title', ' '.join(['word'] * 400), 'query', FourSpecialTokens(),
                        overlap=0, max_length=128)[0]
    assert p['token_end'] == 122


@pytest.mark.parametrize('max_length', [256, 211])
def test_title_only_pair_needs_no_window_overlap_budget(max_length):
    title, query = ' '.join(['title'] * 80), ' '.join(['query'] * 128)
    assert select_passages(title, '', query, Tokenizer(), max_length=max_length) == [
        dict(text=title, start=0, end=0, token_start=0, token_end=0)]


def test_title_only_pair_rejects_insufficient_total_budget():
    with pytest.raises(ValueError, match='Title and query exceed'):
        select_passages(' '.join(['title'] * 80), '', ' '.join(['query'] * 128),
                        Tokenizer(), max_length=210)


def test_no_offset_abstract_returns_title_without_window_budget():
    class NoAbstractOffsets(Tokenizer):
        def __call__(self, text, **kwargs):
            return (dict(input_ids=[], offset_mapping=[]) if text == 'untokenized'
                    else super().__call__(text, **kwargs))
    title, query = ' '.join(['title'] * 80), ' '.join(['query'] * 128)
    p = select_passages(title, 'untokenized', query, NoAbstractOffsets(), max_length=211)
    assert p == [dict(text=title, start=0, end=0, token_start=0, token_end=0)]


@pytest.mark.parametrize('special_tokens', [True, -1, 1.5, float('nan')])
def test_invalid_tokenizer_special_token_count_is_rejected(special_tokens):
    tok = Tokenizer()
    tok.num_special_tokens_to_add = lambda **kwargs: special_tokens
    with pytest.raises(ValueError, match='pair special token count'):
        select_passages('title', '', 'query', tok)


@pytest.mark.parametrize('kwargs', [dict(width=True), dict(overlap=1.5), dict(maximum=0),
                                    dict(max_length=float('nan')), dict(max_length=1)])
def test_invalid_passage_numeric_domains(kwargs):
    with pytest.raises(ValueError):
        select_passages('title', 'abstract', 'query', Tokenizer(), **kwargs)


def test_rerank_passes_configured_maximum_to_selection():
    engine = object.__new__(SearchEngineV2)
    engine.config = dict(rerank_depth=1, passage_tokens=320, passage_overlap=20,
                         passages_per_document=2, reranker_max_length=256, reranker_batch_size=4)
    engine._score_cache, engine._pair_cache = {}, {}
    engine.meta = pd.DataFrame([dict(title='title', abstract=' '.join(['word'] * 600))], index=['a'])
    observed = []
    def predict(pairs, **kwargs):
        observed.extend(pairs)
        return [1.] * len(pairs)
    engine.reranker = SimpleNamespace(tokenizer=Tokenizer(), predict=predict)
    assert engine.rerank('query', [('a', 0)]) == [('a', 1.)]
    assert observed and all(len(Tokenizer()(q)['input_ids']) + len(Tokenizer()(p)['input_ids']) + 3 <= 256
                            for q, p in observed)


@pytest.mark.parametrize('name,value', [('rrf_k', -1), ('threads', False), ('bm25_b', 1.1),
                                      ('bm25_k1', 0), ('title_weight', float('inf')),
                                      ('passage_overlap', 320), ('reranker_max_length', 0)])
def test_invalid_config_is_rejected_before_corpus_io(tmp_path, monkeypatch, name, value):
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/search_v2.json').read_text())
    config[name] = value
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(config))
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid configuration reached corpus IO')
    monkeypatch.setattr(pd, 'read_parquet', unexpected)
    with pytest.raises(ValueError):
        SearchEngineV2(path)


def test_default_configuration_is_valid():
    validate_config(json.loads((Path(__file__).resolve().parents[1] / 'configs/search_v2.json').read_text()))
