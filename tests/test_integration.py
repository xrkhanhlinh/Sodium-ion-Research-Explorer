import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data_io import SCHEMA, sha256
from src.text_processing import formula_spans

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    path = ROOT / "data/processed/combined_clean.parquet"
    assert path.exists(), "Run python run_stage1.py first; integration tests require real outputs"
    return pd.read_parquet(path)


def test_originals_unchanged_and_dynamic_counts(data):
    audit = json.loads((ROOT / "outputs/reports/input_audit.json").read_text(encoding="utf-8"))
    for key in ["patent", "publication"]:
        assert sha256(ROOT / f"{key}.xlsx") == audit["input_hashes"][key]
        source = key.title()
        raw = pd.read_parquet(ROOT / f"data/interim/{key}_raw.parquet")
        assert len(raw) == audit["source_counts"][source] == int(data.source.eq(source).sum())
        sheets = audit[f"{key}_sheets"] if key == "patent" else audit["publication_sheets"]
        assert len(raw) == sum(s["record_count"] for s in sheets)


def test_counts_independently_from_workbook_xml(data):
    import zipfile
    import xml.etree.ElementTree as ET
    audit = json.loads((ROOT / "outputs/reports/input_audit.json").read_text(encoding="utf-8"))
    for source, key in [("Patent", "patent"), ("Publication", "publication")]:
        sheets = audit[f"{key}_sheets"]
        expected_headers = {s["name"]: s["header_row"] for s in sheets}
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
        with zipfile.ZipFile(ROOT / f"{key}.xlsx") as archive:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
            total = 0
            for sheet in workbook.find("s:sheets", ns):
                header = expected_headers[sheet.attrib["name"]]
                if header is None:
                    continue
                target = targets[sheet.attrib["{" + ns["r"] + "}id"]]
                target = target.lstrip("/") if target.startswith("/") else "xl/" + target
                with archive.open(target) as stream:
                    for event, elem in ET.iterparse(stream, events=("end",)):
                        if elem.tag == "{" + ns["s"] + "}row":
                            if int(elem.attrib["r"]) > header and any(c.find("s:v", ns) is not None or c.find("s:is", ns) is not None for c in elem):
                                total += 1
                            elem.clear()
            assert total == audit["source_counts"][source]
            assert total == int(data.source.eq(source).sum())


def test_ids_schema_and_provenance(data):
    assert set(SCHEMA).issubset(data)
    assert data.doc_id.is_unique and data.doc_id.str.len().gt(0).all()
    assert not data.duplicated(["source_file", "source_sheet", "source_row"]).any()
    meta = pd.read_parquet(ROOT / "data/processed/metadata_extended.parquet")
    assert meta.doc_id.tolist() == data.doc_id.tolist()
    assert meta.original_fields_json.map(json.loads).map(bool).all()


def test_real_broken_row_repaired_and_preserved(data):
    raw = pd.read_parquet(ROOT / "data/interim/publication_raw.parquet")
    broken = raw[(raw.source_sheet == "2001-2500") & (raw.source_row == 368)].iloc[0]
    assert "\t" in broken.TI and not broken.UT
    fixed = data[(data.source == "Publication") & (data.source_sheet == "2001-2500") & (data.source_row == 368)].iloc[0]
    assert fixed.repair_status == "repaired"
    assert fixed.identifier == "10.1002/elan.201600106"
    assert fixed.native_id == "WOS:000379039000010"
    assert fixed.year == 2016
    assert fixed.relevance == "OUT_OF_SCOPE" and not fixed.eligible_topic


def test_actual_duplicate_patents_not_deleted(data):
    pair = data[(data.source == "Patent") & data.source_row.isin([6326, 6327])]
    assert len(pair) == 2 and pair.native_id.nunique() == 2
    assert pair.split_group_id.nunique() == 1
    assert pair.duplicate_of.eq("").all()


def test_actual_patent_dates_keep_publication_and_priority_separate(data):
    row = data[(data.source == "Patent") & data.source_row.eq(4)].iloc[0]
    assert row.year == 1990 and row.priority_year == 1988
    assert row.year_basis == "earliest_publication_date_in_PD"


def test_corresponding_corpora_and_all_formula_tokens(data):
    lda = pd.read_parquet(ROOT / "data/processed/lda_tokens.parquet")
    bert = pd.read_parquet(ROOT / "data/processed/bert_texts.parquet")
    main = data[data.eligible_topic]
    assert lda.doc_id.tolist() == bert.doc_id.tolist() == main.doc_id.tolist()
    assert lda.token_count.eq(lda.tokens.map(len)).all()
    assert lda.token_count.ge(5).all()
    tokenmap = dict(zip(lda.doc_id, lda.tokens))
    for row in main.itertuples():
        assert {m.group() for m in formula_spans(row.text_for_model)}.issubset(set(tokenmap[row.doc_id]))
    assert main.language_eligible.all() and main.relevance.isin(["CORE", "RELATED"]).all()
    assert not main.retracted.any() and not main.structural_issue.any()


def test_missing_abstracts_not_synthesized(data):
    pat = data[(data.source == "Patent") & data.source_row.isin([3, 6, 8])]
    assert len(pat) == 3 and pat.abstract.eq("").all()
    assert not pat.eligible_finetune.any()
    raw = pd.read_parquet(ROOT / "data/interim/combined_raw.parquet")
    unchanged = data.repair_status.eq("none")
    assert data.loc[unchanged, "abstract"].eq("").tolist() == raw.loc[unchanged, "abstract"].str.strip().eq("").tolist()


def test_figures_tables_and_exclusive_totals(data):
    manifest = json.loads((ROOT / "outputs/reports/figure_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) >= 6
    for item in manifest:
        table = pd.read_csv(ROOT / item["table"])
        assert len(table) > 0
        with Image.open(ROOT / item["figure"]) as image:
            assert min(image.info["dpi"]) >= 299
            assert image.width >= 2000 and image.height >= 1000
    summary = pd.read_csv(ROOT / "outputs/reports/cleaning_summary.csv")
    assert summary["count"].sum() == len(data)
    assert summary.groupby("source")["count"].sum().to_dict() == data.source.value_counts().to_dict()
    relevance = pd.read_csv(ROOT / "outputs/tables/data_preparation/figure_06_domain_relevance.csv")
    assert relevance["count"].sum() == len(data)


def test_reports_and_runtime_validation(data):
    report = json.loads((ROOT / "outputs/reports/data_validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    for name in ["data_quality_report", "duplicate_report", "relevance_audit", "cleaning_log", "needs_review", "language_audit", "text_quality_audit"]:
        assert len(pd.read_csv(ROOT / f"outputs/reports/{name}.csv")) > 0
    assert data.retracted.sum() == data[data.document_type.str.contains("Retracted Publication", case=False)].shape[0]
