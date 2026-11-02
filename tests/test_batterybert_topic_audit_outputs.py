"""Integration checks for completed, real BatteryBERT audit exports.

Requires the versioned audit run documented in docs/BATTERYBERT_TOPIC_AUDIT.md.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from src.batterybert_topic_tools import sha, validate_mapping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/batterybert_topic_audit_20261022_v1"


def test_all_real_experiment_counts_and_ids():
    ids = pd.read_parquet(ROOT / "data/processed/main_corpus.parquet", columns=["doc_id"]).doc_id.tolist()
    markers = list((OUT / "experiments").glob("*/complete.json"))
    assert len(markers) == 15
    for marker in markers:
        folder = marker.parent
        labels = np.load(folder / "assignments.npy")
        frame = pd.read_csv(folder / "topic_assignments.csv")
        assert labels.shape == (13702,)
        assert frame.doc_id.tolist() == ids
        assert np.array_equal(labels, frame.topic_id)
        summary = pd.read_csv(folder / "topic_distribution.csv")
        assert summary.documents.sum() == 13702
        assert summary.corpus_share.sum() == pytest.approx(1.)
        assert summary.loc[summary.is_outlier, "documents"].sum() == (labels == -1).sum()
        recorded = json.loads(marker.read_text())
        assert sha(folder / "assignments.npy") == recorded["assignments_sha256"]


def test_representation_updates_preserve_every_membership():
    original = np.load(OUT / "experiments/baseline/assignments.npy")
    for folder in (OUT / "experiments").glob("repr*"):
        assert np.array_equal(np.load(folder / "assignments.npy"), original)
    for folder in (OUT / "experiments").glob("*_repaired"):
        config = json.loads((folder / "config.json").read_text())
        assert np.array_equal(np.load(folder / "assignments.npy"),
                              np.load(OUT / "experiments" / config["parent"] / "assignments.npy"))


def test_real_reduction_preserves_outliers_and_mapping():
    parent = OUT / "experiments/hdb_leaf30_5"
    child = OUT / "experiments/hdb_leaf30_5_reduced10"
    mapping = validate_mapping(np.load(parent / "assignments.npy"), np.load(child / "assignments.npy"))
    saved = pd.read_csv(child / "reduction_mapping.csv")
    assert mapping == dict(zip(saved.old_topic_id, saved.new_topic_id))


def test_placeholder_repair_and_scientific_labels_are_real():
    recovery = pd.read_csv(OUT / "placeholder_repairs.csv")
    assert len(recovery) == 716
    assert recovery.doc_id.nunique() == 355
    assert not recovery.after.str.contains("placeholder").any()
    for folder in (OUT / "experiments").glob("*_repaired"):
        assert not pd.read_csv(folder / "keywords.csv").token.str.contains("placeholder").any()
    reviewed = pd.read_csv(OUT / "scientific_review.csv")
    corpus = pd.read_parquet(ROOT / "data/processed/main_corpus.parquet").set_index("doc_id")
    assert corpus.loc[reviewed.doc_id, "title"].tolist() == reviewed.title.tolist()
    assert set(reviewed.review_status) >= {"well_matched", "questionable", "potentially_misplaced"}


def test_final_exports_are_complete_and_metrics_are_traceable():
    required = ["root_cause_audit.md", "experiment_comparison.csv", "topic_source_distribution.csv",
        "topic_distribution.csv", "topic_keywords_before_after.csv", "topic_assignments.csv",
        "representative_documents.csv", "topic_quality_metrics.json", "final_model_config.yaml",
        "bertopic_final_10_topics.png", "optimization_report.md"]
    assert all((OUT / name).is_file() and (OUT / name).stat().st_size for name in required)
    selection = json.loads((OUT / "final_selection.json").read_text())
    metrics = json.loads((OUT / "topic_quality_metrics.json").read_text())
    frame = pd.read_csv(OUT / "topic_assignments.csv")
    assert len(frame) == frame.doc_id.nunique() == 13702
    assert selection["topics"] == frame.loc[frame.topic_id >= 0, "topic_id"].nunique() == metrics["topics"]
    assert selection["outliers"] == int((frame.topic_id == -1).sum())
    assert metrics["training_invoked"] is False
    assert metrics["embeddings_regenerated"] is False
    assert json.loads((OUT / "preservation_check.json").read_text())["status"] == "PASS"


def test_reload_inference_and_original_files_unchanged():
    probe = json.loads((OUT / "model_reload_inference.json").read_text())
    assert probe["matching_saved_assignments"] == probe["documents"] == 12
    for relative, expected in json.loads((OUT / "protected_files.json").read_text()).items():
        assert sha(ROOT / relative) == expected, relative
