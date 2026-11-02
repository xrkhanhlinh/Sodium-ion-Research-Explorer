import numpy as np
import pandas as pd
import pytest
from src.batterybert_optimization import Geometry, ContiguousScientificVectorizer, document_npmi


def test_no_phrase_invented_by_stopword_deletion():
    analyzer = ContiguousScientificVectorizer(ngram_range=(1, 2), stop_words=['and']).build_analyzer()
    terms = set(analyzer('hard and carbon ␞ carbon anode Na3V2(PO4)3'))
    assert 'hard carbon' not in terms
    assert 'carbon carbon' not in terms
    assert 'carbon anode' in terms
    assert 'Na3V2(PO4)3' in terms


@pytest.mark.parametrize('mode', ['raw', 'center', 'source', 'abtt3', 'whiten64', 'source_whiten64'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_geometry_is_train_fitted_and_finite(mode, dtype):
    rng = np.random.default_rng(17)
    train = rng.normal(size=(100, 12)).astype(dtype)
    original = train.copy()
    g = Geometry(mode, dimensions=5).fit(train, ['a'] * 50 + ['b'] * 50)
    before = g.mean_.copy()
    result = g.transform(rng.normal(size=(11, 12)))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(train, original)
    np.testing.assert_array_equal(g.mean_, before)


def test_source_projection_removes_only_training_mean_direction():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(100, 8)); train[:50, 0] += 30
    g = Geometry('source').fit(train, ['a'] * 50 + ['b'] * 50)
    # Projection is global and needs no source metadata when predicting.
    projected = g.transform(train)
    np.testing.assert_allclose(projected @ g.direction_, 0, atol=1e-6)


def test_npmi_missing_pairs_are_penalized_and_exact_phrases_required():
    keys = pd.DataFrame(dict(topic_id=[0, 0], rank=[1, 2], token=['hard carbon', 'anode']))
    m = document_npmi(['hard porous carbon anode', 'hard carbon'], keys)
    assert m['document_npmi'] == -1
    assert m['supported_keyword_fraction'] == 1


def test_npmi_positive_for_shared_distinct_terms():
    keys = pd.DataFrame(dict(topic_id=[0, 0], rank=[1, 2], token=['carbon', 'anode']))
    m = document_npmi(['carbon anode', 'electrolyte salt'], keys)
    assert m['document_npmi'] == pytest.approx(1)
