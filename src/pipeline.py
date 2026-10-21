import argparse
import importlib.metadata as packages
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import psutil
import yaml

from .data_io import SCHEMA, integrate, read_workbook, sha256
from .figures import build_figures
from .quality import apply_eligibility, assess_language, assess_text, duplicate_audit, missing_report
from .text_processing import formula_spans, lda_tokens, normalize_text

DIRECT_PACKAGES = ["pandas", "numpy", "openpyxl", "pyarrow", "scipy", "scikit-learn", "nltk", "matplotlib", "PyYAML", "pytest", "langdetect", "psutil", "Pillow"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def environment(root, config):
    gpu = {"status": "not_verified", "reason": "nvidia-smi unavailable; Windows CIM query denied in initial inspection. Stage 1 uses CPU."}
    binary = shutil.which("nvidia-smi")
    if binary:
        try:
            gpu = {"status": "queried", "output": subprocess.check_output([binary, "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=15, text=True).strip()}
        except (subprocess.SubprocessError, OSError) as exc:
            gpu = {"status": "not_verified", "reason": str(exc)}
    installed = {name: packages.version(name) for name in DIRECT_PACKAGES}
    report = dict(python=sys.version, executable=sys.executable, virtual_environment=sys.prefix != sys.base_prefix, system_site_packages_inherited=True, platform=platform.platform(), processor=platform.processor(), logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False), total_ram_bytes=psutil.virtual_memory().total, available_ram_bytes=psutil.virtual_memory().available, disk=shutil.disk_usage(root)._asdict(), gpu=gpu, packages=installed, seed=config["seed"], stage=1, wordnet_path=str(root / ".cache/nltk_data"), wordnet_sha256=sha256(root / ".cache/nltk_data/corpora/wordnet.zip"))
    write_json(root / "outputs/reports/environment_report.json", report)
    (root / "requirements.txt").write_text("# Stage 1 direct dependencies, pinned to the executed environment.\n" + "\n".join(f"{name}=={version}" for name, version in installed.items()) + "\n", encoding="utf-8")
    return report


def progress(root, phase, status="RUNNING", detail=""):
    now = datetime.now(timezone(timedelta(hours=7))).isoformat()
    write_json(root / "outputs/reports/stage1/progress.json", dict(stage=1, phase=phase, status=status, updated=now, detail=detail))
    with (root / "docs/EXPERIMENT_LOG.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n- {now} - Stage 1 {status} - {phase}: {detail}\n")


def validate(root, raw, clean, lda, bert, source_counts, hashes, config):
    checks = []

    def check(name, condition, detail=""):
        checks.append(dict(check=name, status="PASS" if condition else "FAIL", detail=detail))

    check("raw_counts_match_workbooks", raw.source.value_counts().to_dict() == source_counts)
    check("all_records_retained_in_clean_master", len(raw) == len(clean) and clean.source.value_counts().to_dict() == source_counts)
    check("unique_nonblank_clean_doc_ids", clean.doc_id.is_unique and clean.doc_id.str.len().gt(0).all())
    check("provenance_complete", clean[["source_file", "source_sheet", "source_row"]].notna().all().all())
    check("provenance_same_rows", set(zip(raw.source_file, raw.source_sheet, raw.source_row)) == set(zip(clean.source_file, clean.source_sheet, clean.source_row)))
    main = clean[clean.eligible_topic]
    check("nonempty_main_corpus", len(main) > 0 and set(main.source) == {"Patent", "Publication"})
    check("no_unresolved_structure_in_main", not main.structural_issue.any() and not main.text_structure_issue.any())
    check("no_retracted_in_main", not main.retracted.any())
    check("no_confirmed_duplicates_in_main", main.duplicate_of.eq("").all())
    check("relevance_policy_applied", main.relevance.isin(config["policy"]["main_relevance"]).all())
    check("language_policy_applied", main.language_eligible.all())
    check("no_fake_abstract_or_null_strings", not clean[["title", "abstract"]].isin(["nan", "None", "<NA>"]).any().any())
    check("corpus_ids_equal_in_order", lda.doc_id.tolist() == bert.doc_id.tolist() == main.doc_id.tolist())
    check("no_short_lda_documents", lda.tokens.map(len).ge(config["policy"]["minimum_lda_tokens"]).all())
    formula_failures = []
    token_map = dict(zip(lda.doc_id, lda.tokens))
    for row in main.itertuples():
        expected = {m.group() for m in formula_spans(row.text_for_model)}
        actual = set(token_map[row.doc_id])
        if not expected.issubset(actual):
            formula_failures.append(dict(doc_id=row.doc_id, missing=sorted(expected - actual)))
    check("all_detected_formula_tokens_preserved", not formula_failures, json.dumps(formula_failures[:10], ensure_ascii=False))
    sentinel = "Na3V2(PO4)3 NaFePO4 Na+ hard carbon cathode anode electrolyte"
    tokens = lda_tokens(sentinel, config["preprocessing"]["domain_phrases"], config["preprocessing"]["technical_stopwords"])
    check("required_chemical_and_domain_sentinels", {"Na3V2(PO4)3", "NaFePO4", "Na+", "hard_carbon", "cathode", "anode", "electrolyte"}.issubset(tokens) and normalize_text(sentinel) == sentinel)
    check("original_input_hashes_unchanged", all(sha256(root / config["inputs"][name]) == digest for name, digest in hashes.items()))
    check("exclusive_dispositions_reconcile", int(clean.groupby(["source", "disposition"]).size().sum()) == len(raw))
    check("all_primary_fields_available", set(SCHEMA).issubset(clean.columns))
    expected_data = ["interim/patent_raw.parquet", "interim/publication_raw.parquet", "interim/combined_raw.parquet", "processed/patent_clean.parquet", "processed/publication_clean.parquet", "processed/combined_clean.parquet", "processed/main_corpus.parquet", "processed/lda_tokens.parquet", "processed/bert_texts.parquet", "processed/metadata_extended.parquet"]
    check("required_datasets_exist", all((root / "data" / name).is_file() for name in expected_data))
    check("six_figures_and_tables_exist", len(list((root / "outputs/figures/data_preparation").glob("figure_*.png"))) >= 6 and len(list((root / "outputs/tables/data_preparation").glob("figure_*.csv"))) >= 6)
    for path in sorted((root / "data").rglob("*.parquet")):
        reopened = pd.read_parquet(path)
        check("parquet_readable:" + path.name, len(reopened) > 0, str(len(reopened)))
    from PIL import Image
    for path in sorted((root / "outputs/figures/data_preparation").glob("*.png")):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            check("png_dpi:" + path.name, min(image.info.get("dpi", (0, 0))) >= 299, str(image.size))
    result = dict(status="PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL", checks=checks, formula_failures=formula_failures)
    write_json(root / "outputs/reports/data_validation_report.json", result)
    write_csv(root / "outputs/reports/data_validation_report.csv", pd.DataFrame(checks))
    return result


def run(root, config):
    for folder in ["configs", "data/interim", "data/processed", "outputs/reports", "outputs/logs", "outputs/figures/data_preparation", "outputs/tables/data_preparation", "docs", ".cache"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(root / "outputs/logs/stage1.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("stage1")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    nltk.data.path.insert(0, str(root / ".cache/nltk_data"))
    # Fail instead of silently replacing actual lemmatization with a different method.
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()
    progress(root, "P0 environment")
    environment(root, config)
    input_hashes = {name: sha256(root / file) for name, file in config["inputs"].items()}
    log.info("Reading original workbooks (never opened for writing)")
    patent, patent_sheets = read_workbook(root / config["inputs"]["patent"], "Patent")
    publication, pub_sheets = read_workbook(root / config["inputs"]["publication"], "Publication")
    source_counts = {"Patent": len(patent), "Publication": len(publication)}
    manifest = dict(input_hashes=input_hashes, source_counts=source_counts, patent_sheets=patent_sheets, publication_sheets=pub_sheets, configuration=config)
    write_json(root / "outputs/reports/input_audit.json", manifest)
    patent.to_parquet(root / "data/interim/patent_raw.parquet", index=False)
    publication.to_parquet(root / "data/interim/publication_raw.parquet", index=False)
    pub_headers = next(s["headers"] for s in pub_sheets if s["record_count"])
    raw, _, _ = integrate(patent, publication, pub_headers, repair_enabled=False)
    raw.to_parquet(root / "data/interim/combined_raw.parquet", index=False)
    progress(root, "P1 integration complete", detail=json.dumps(source_counts))
    log.info("Raw counts: %s", source_counts)
    clean, metadata, actions = integrate(patent, publication, pub_headers)
    metadata["original_doc_id"] = raw.doc_id.values
    clean["original_doc_id"] = raw.doc_id.values
    text_fields = ["title", "abstract", "authors", "organization", "keywords", "classification", "identifier", "document_type", "source_name"]
    for idx, row in clean.iterrows():
        changes = {}
        for field in text_fields:
            before = row[field]
            after = normalize_text(before)
            if field == "title":
                import re
                after = re.sub(r"\s+NoAbstract\s+Dwg\s+\d+/\d+\s*$", "", after)
            clean.at[idx, field] = after
            if before != after:
                changes[field] = {"before": before, "after": after}
        if changes:
            actions.append(dict(doc_id=row.doc_id, source=row.source, source_file=row.source_file, source_sheet=row.source_sheet, source_row=row.source_row, action="normalize_text_fields", reason="whitespace_controls_NFC_subscript_digits_and_verified_export_suffix_only", before=json.dumps({k: v["before"] for k, v in changes.items()}, ensure_ascii=False), after=json.dumps({k: v["after"] for k, v in changes.items()}, ensure_ascii=False)))
    clean["text_for_model"] = [" ".join(v for v in (t, a) if v) for t, a in zip(clean.title, clean.abstract)]
    assess_text(clean)
    assess_language(clean, config, root / ".cache/language_detection.json", log)
    progress(root, "P2 text normalization, relevance and language complete")
    duplicates = duplicate_audit(clean, metadata, config, log)
    write_csv(root / "outputs/reports/duplicate_report.csv", duplicates)
    apply_eligibility(clean, config)
    log.info("Preprocessing eligible documents for LDA and BERT")
    token_rows, bert_rows = [], []
    for idx, row in clean[clean.eligible_topic].iterrows():
        tokens = lda_tokens(row.text_for_model, config["preprocessing"]["domain_phrases"], config["preprocessing"]["technical_stopwords"])
        if len(tokens) < config["policy"]["minimum_lda_tokens"]:
            clean.loc[idx, ["eligible_topic", "eligible_search", "eligible_finetune"]] = False
            clean.at[idx, "disposition"] = "TEXT_QUALITY_REVIEW"
            continue
        token_rows.append(dict(doc_id=row.doc_id, source=row.source, split_group_id=row.split_group_id, tokens=tokens, token_count=len(tokens)))
        bert_rows.append(dict(doc_id=row.doc_id, source=row.source, split_group_id=row.split_group_id, title=row.title, abstract=row.abstract, text=row.text_for_model, eligible_finetune=bool(row.eligible_finetune)))
    lda, bert = pd.DataFrame(token_rows), pd.DataFrame(bert_rows)
    clean.to_parquet(root / "data/processed/combined_clean.parquet", index=False)
    for source in ["Patent", "Publication"]:
        clean[clean.source == source].to_parquet(root / f"data/processed/{source.lower()}_clean.parquet", index=False)
    main = clean[clean.eligible_topic]
    main.to_parquet(root / "data/processed/main_corpus.parquet", index=False)
    lda.to_parquet(root / "data/processed/lda_tokens.parquet", index=False)
    bert.to_parquet(root / "data/processed/bert_texts.parquet", index=False)
    metadata["split_group_id"] = clean.split_group_id.values
    metadata.to_parquet(root / "data/processed/metadata_extended.parquet", index=False)
    quality_parts = [missing_report(raw, "raw_unified", SCHEMA), missing_report(clean, "clean_master", SCHEMA), missing_report(main, "main_corpus", SCHEMA)]
    for source, frame in [("Patent", patent), ("Publication", publication)]:
        fields = [c for c in frame if c not in {"source_file", "source_sheet", "source_row", "raw_fields_json", "unnamed_nonempty_json"}]
        quality_parts.append(missing_report(frame.assign(source=source), "original_source_columns", fields))
    quality = pd.concat(quality_parts, ignore_index=True)
    write_csv(root / "outputs/reports/data_quality_report.csv", quality)
    write_csv(root / "outputs/reports/relevance_audit.csv", clean[["doc_id", "source", "source_file", "source_sheet", "source_row", "title", "abstract", "relevance", "relevance_reason", "relevance_method", "disposition", "eligible_topic", "eligible_search", "eligible_finetune"]])
    review_mask = clean.disposition.str.contains("REVIEW")
    write_csv(root / "outputs/reports/needs_review.csv", clean[review_mask])
    write_csv(root / "outputs/reports/structural_issues.csv", clean[(clean.repair_status != "none") | clean.structural_issue])
    write_csv(root / "outputs/reports/retracted_records.csv", clean[clean.retracted])
    write_csv(root / "outputs/reports/language_audit.csv", clean[["doc_id", "source", "language_metadata", "detected_language", "language_confidence", "language_metadata_mismatch", "language_eligible"]])
    write_csv(root / "outputs/reports/text_quality_audit.csv", clean[["doc_id", "source", "title_chars", "abstract_chars", "title_words", "abstract_words", "character_issue", "abnormal_length", "text_structure_issue", "disposition"]])
    for row in clean[~clean.eligible_topic].itertuples():
        actions.append(dict(doc_id=row.doc_id, source=row.source, source_file=row.source_file, source_sheet=row.source_sheet, source_row=row.source_row, action="exclude_from_main_corpus_keep_in_clean_master", reason=row.disposition, before="retained_source_record", after="retained_in_clean_master_not_main_corpus"))
    for row in clean[clean.source == "Patent"].itertuples():
        actions.append(dict(doc_id=row.doc_id, source=row.source, source_file=row.source_file, source_sheet=row.source_sheet, source_row=row.source_row, action="derive_publication_year", reason=row.year_basis, before=row.year_source_raw, after=str(row.year)))
    write_csv(root / "outputs/reports/cleaning_log.csv", pd.DataFrame(actions))
    write_csv(root / "outputs/reports/cleaning_summary.csv", clean.groupby(["source", "disposition"]).size().reset_index(name="count"))
    progress(root, "P2 datasets, corpora and diagnostic reports saved")
    log.info("Rendering six Stage 1 figures from exported statistical tables")
    figures = build_figures(raw, clean, quality, root, config["visualization"]["dpi"])
    write_json(root / "outputs/reports/figure_manifest.json", figures)
    result = validate(root, raw, clean, lda, bert, source_counts, input_hashes, config)
    if result["status"] != "PASS":
        progress(root, "validation failed", "PARTIAL", "See outputs/reports/data_validation_report.json")
        raise RuntimeError("Data validation failed; inspect the saved report")
    summary = dict(status="VALIDATED_PENDING_TESTS", input_counts=source_counts, clean_master_counts=clean.source.value_counts().to_dict(), main_corpus_counts=main.source.value_counts().to_dict(), main_corpus_total=len(main), finetuning_eligible=int(clean.eligible_finetune.sum()), retracted=int(clean.retracted.sum()), repaired=int(clean.repair_status.eq("repaired").sum()), unresolved_structure=int(clean.structural_issue.sum()), confirmed_duplicate_exclusions=int(clean.disposition.eq("CONFIRMED_DUPLICATE").sum()), needs_review=int(review_mask.sum()), relevance=clean.groupby(["source", "relevance"]).size().reset_index(name="count").to_dict("records"), dispositions=clean.groupby(["source", "disposition"]).size().reset_index(name="count").to_dict("records"), validation_checks=len(result["checks"]), input_hashes=input_hashes)
    write_json(root / "outputs/reports/stage1_summary.json", summary)
    progress(root, "data validation PASS; unit/integration tests next", detail=f"Main corpus: {len(main)} documents; all {len(clean)} source rows retained.")
    log.info("Pipeline data checks PASS: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 1 only: prepare and validate Sodium-ion datasets")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    if config.get("stage") != 1:
        raise ValueError("This entry point permits Stage 1 only")
    try:
        run(root, config)
    except Exception as exc:
        if (root / "docs").exists():
            progress(root, "execution interrupted or failed", "PARTIAL", str(exc))
        raise
