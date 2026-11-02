"""Validation contracts for the isolated topic experiments."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.batterybert_topic_tools import (BOUNDARY, PROTECTED, ScientificCountVectorizer,
    reserve_output, validate_labels, validate_mapping, distribution, stopword_evidence,
    sample_cosines, phrase_reference, centroid_scores, repair_phrase_placeholders)


def test_chemical_and_unit_tokens_survive():
    text = "hard carbon Na-ion Na3V2(PO4)3 Na_xCoO2 mAh Wh/kg cathode first-principles"
    analyzer = ScientificCountVectorizer(ngram_range=(1, 3), lowercase=False,
        stop_words=list(PROTECTED) + ["first"]).build_analyzer()
    features = set(analyzer(text))
    assert set(text.split()) <= features
    assert "hard carbon" in features
    assert "carbon Na-ion Na3V2(PO4)3" in features


def test_no_ngram_across_document_boundary():
    analyzer = ScientificCountVectorizer(ngram_range=(1, 2)).build_analyzer()
    assert "carbon electrolyte" not in set(analyzer(f"hard carbon {BOUNDARY} electrolyte salt"))


def test_df_on_original_documents_keeps_topic_specific_phrase():
    docs = ["Na3V2(PO4)3 cathode"] * 3 + ["hard carbon anode"] * 3
    vectorizer = ScientificCountVectorizer(ngram_range=(1, 2), min_df=3, lowercase=False)
    vectorizer.fit(docs)
    aggregate = ScientificCountVectorizer(ngram_range=(1, 2), vocabulary=vectorizer.vocabulary_, min_df=1, lowercase=False)
    matrix = aggregate.fit_transform([f" {BOUNDARY} ".join(docs[:3]), f" {BOUNDARY} ".join(docs[3:])])
    assert matrix[0, vectorizer.vocabulary_["Na3V2(PO4)3 cathode"]] == 3
    assert matrix[1, vectorizer.vocabulary_["Na3V2(PO4)3 cathode"]] == 0


def test_stopword_evidence_protects_technical_and_rare_terms():
    texts = [["obtain", "source", "sodium", "capacity", "mAh", "Na3V2(PO4)3"] for _ in range(40)]
    texts[0].append("embodiment")
    stop, evidence = stopword_evidence(texts, ["Patent"] * 20 + ["Publication"] * 20)
    assert "obtain" in stop
    assert not set(stop) & PROTECTED
    assert "source" not in stop
    assert not evidence.set_index("token").loc["embodiment", "remove"]


@pytest.mark.parametrize("labels", [[0, 1], [0, -2, 1], [0., 1., 2.]])
def test_invalid_assignments(labels):
    with pytest.raises(ValueError):
        validate_labels(labels, 3)


def test_outlier_counts_and_source_association():
    summary, table, metrics = distribution([0, 0, 1, 1, -1], ["Patent", "Patent", "Publication", "Publication", "Publication"])
    assert summary.documents.sum() == 5
    assert metrics["outliers"] == 1
    assert metrics["top2_topic_share"] == .8
    assert metrics["source_cramers_v"] == pytest.approx(1.)


@pytest.mark.parametrize("before,after", [([0, 0, 1], [0, 1, 1]), ([-1, 0, 1], [0, 0, 1])])
def test_reduction_cannot_split_or_reassign_outlier(before, after):
    with pytest.raises(ValueError):
        validate_mapping(before, after)


def test_valid_reduction_mapping():
    assert validate_mapping([-1, 0, 1, 2], [-1, 0, 0, 1]) == {-1: -1, 0: 0, 1: 0, 2: 1}


def test_safe_output_refuses_unrelated_results(tmp_path):
    (tmp_path / "baseline.csv").write_text("unchanged")
    with pytest.raises(FileExistsError):
        reserve_output(tmp_path)
    assert (tmp_path / "baseline.csv").read_text() == "unchanged"


def test_safe_output_resume(tmp_path):
    target = tmp_path / "new"
    reserve_output(target)
    reserve_output(target)
    assert json.loads((target / "run_identity.json").read_text())["training"] is False


def test_cosine_sampling_reproducible():
    vectors = np.eye(4, dtype=np.float32)
    assert sample_cosines(vectors, [0, 0, 1, 1], pairs=20) == sample_cosines(vectors, [0, 0, 1, 1], pairs=20)


def test_coherence_phrases_require_real_contiguous_occurrence():
    docs = list(phrase_reference([["hard", "carbon", "anode"], ["hard", "porous", "carbon"]], {"hard carbon"}))
    assert "hard\u2581carbon" in docs[0]
    assert "hard\u2581carbon" not in docs[1]


def test_recover_embedded_phrase_placeholders_without_changing_original():
    tokens = [["all-phraseplaceholdera", "Na3V2(PO4)3", "mAh", "Na_xCoO2"]]
    fixed, changes = repair_phrase_placeholders(tokens, ["solid state"])
    assert fixed == [["all-solid_state", "Na3V2(PO4)3", "mAh", "Na_xCoO2"]]
    assert tokens[0][0] == "all-phraseplaceholdera"
    assert len(changes) == 1
    assert repair_phrase_placeholders(fixed, ["solid state"])[0] == fixed


def test_unknown_placeholder_is_not_silently_exported():
    with pytest.raises(ValueError, match="Unresolved"):
        repair_phrase_placeholders([["chemplaceholderaa"]], ["solid state"])


def test_centroid_scores_are_assigned_topic_cosines():
    scores, redundancy = centroid_scores(np.eye(3, dtype=np.float32), [0, 0, 1])
    np.testing.assert_allclose(scores, [2 ** -.5, 2 ** -.5, 1], atol=1e-6)
    assert redundancy == 0.


def test_real_embedding_id_alignment_and_baseline_reduction():
    from src.model_comparison import corpus, validate_vectors
    root = Path(__file__).resolve().parents[1]
    main, _ = corpus()
    folder = root / "outputs/model_comparison_v2/batterybert_pretrained"
    ids = json.loads((folder / "ordered_doc_ids.json").read_text())
    info = validate_vectors(folder / "document_embeddings.npy", ids, main.doc_id.tolist(), 768)
    assert info["shape"] == [13702, 768]
    assert len(ids) == len(set(ids)) == 13702
    baseline = folder.parent / "bertopic_batterybert_k10/pretrained"
    labels = np.load(baseline / "assignments.npy")
    initial = np.load(baseline / "initial/assignments.npy")
    assert np.array_equal(labels, initial)
    assert len(set(labels) - {-1}) == 10
    with pytest.raises(ValueError):
        validate_vectors(folder / "document_embeddings.npy", ids[::-1], ids, 768)


def test_nonfinite_and_unnormalized_vectors_rejected(tmp_path):
    from src.model_comparison import validate_vectors
    path = tmp_path / "bad.npy"
    np.save(path, np.array([[np.nan, 0.]], dtype=np.float32))
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_vectors(path, ["a"], ["a"], 2)
    np.save(path, np.array([[2., 0.]], dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        validate_vectors(path, ["a"], ["a"], 2)


@pytest.mark.parametrize("matrix,dimension", [(np.eye(2), 3), (np.ones((3, 2)), 2), (np.ones(2), 2), (np.eye(2, dtype=np.int32), 2)])
def test_embedding_shape_and_dtype_errors_are_explicit(tmp_path, matrix, dimension):
    from src.model_comparison import validate_vectors
    target = tmp_path / "shape.npy"
    np.save(target, matrix)
    with pytest.raises(ValueError):
        validate_vectors(target, ["a", "b"], ["a", "b"], dimension)
