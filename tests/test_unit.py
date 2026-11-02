import json
import pandas as pd
import pytest

from src.data_io import extract_dates, integrate, read_workbook, repair_publication
from src.quality import apply_eligibility, doi_key, language_worker, UnionFind, duplicate_audit
from src.text_processing import lda_tokens, normalize_text, relevance


@pytest.mark.parametrize("text,expected", [
    ("  Na3V2(PO4)3\tNaFePO4\nNa+  ", "Na3V2(PO4)3 NaFePO4 Na+"),
    ("Na₃V₂(PO₄)₃", "Na3V2(PO4)3"),
    ("hard\u200b carbon\x00 electrolyte", "hard carbon electrolyte"),
    (None, ""),
])
def test_normalization_preserves_science(text, expected):
    assert normalize_text(text) == expected


def test_lda_real_lemmatization_and_protected_terms():
    tokens = lda_tokens("The electrodes were storing Na3V2(PO4)3 NaFePO4 Na+ in hard carbon cathode anode electrolyte", ["hard carbon"])
    assert {"electrode", "store", "Na3V2(PO4)3", "NaFePO4", "Na+", "hard_carbon", "cathode", "anode", "electrolyte"}.issubset(tokens)
    assert "the" not in tokens


@pytest.mark.parametrize("text,expected", [
    ("all-solid-state electrolyte", "all-solid_state"),
    ("quasi-solid-state sodium batteries", "quasi-solid_state"),
    ("hard-carbon-based anode Na3V2(PO4)3", "hard_carbon-based"),
])
def test_protected_phrases_in_hyphenated_compounds(text, expected):
    tokens = lda_tokens(text, ["solid state", "hard carbon"])
    assert expected in tokens
    assert not any("placeholder" in token for token in tokens)
    if "Na3V2(PO4)3" in text:
        assert "Na3V2(PO4)3" in tokens


@pytest.mark.parametrize("title,abstract,label", [
    ("Hard carbon anodes for sodium-ion batteries", "", "CORE"),
    ("Lithium-ion and sodium-ion batteries", "Comparative storage", "CORE"),
    ("Sodium intercalation in carbon electrodes", "Electrochemical sodium storage", "RELATED"),
    ("ELECTROCHEMICAL BEHAVIOR OF THE PHASES NAXCOO2", "", "RELATED"),
    ("Sodium sensor for sweat analysis", "The device contains a battery", "OUT_OF_SCOPE"),
    ("Zinc-ion batteries", "Aqueous zinc storage", "OUT_OF_SCOPE"),
    ("Sodium-metal chloride battery", "solid electrolyte", "UNCERTAIN"),
    ("Sodium metal anode", "Dendrite suppression and electrode plating", "UNCERTAIN"),
    ("New carbon material", "Mechanical properties", "UNCERTAIN"),
    ("Lithium-ion batteries", "Sodium-ion batteries are mentioned as a comparison.", "UNCERTAIN"),
    ("Sodium-sulfur battery", "Compared with sodium-ion batteries.", "UNCERTAIN"),
    ("SiB film", "A boride surface study.", "UNCERTAIN"),
])
def test_domain_context(title, abstract, label):
    assert relevance(title, abstract)[0] == label


def test_publication_dates_are_sorted_and_priority_separate():
    text = "WO-A 17 May 1990 English; EP-A 21 Aug 1991 English; bogus 31 Feb 1990"
    assert extract_dates(text) == ["1990-05-17", "1991-08-21"]
    assert extract_dates("DE3837721 07 Nov 1988") == ["1988-11-07"]


def test_repair_requires_exact_positions_and_anchors():
    headers = ["AU", "TI", "SO", "LA", "DT", "AB", "PY", "DI", "UT"]
    suffix = ["Title", "Journal", "English", "Article", "Abstract", "2020", "10.1234/example", "WOS:123"]
    raw = {h: "" for h in headers}
    raw["TI"] = "\t".join(suffix)
    fixed, status, _ = repair_publication(raw, headers)
    assert status == "repaired" and fixed["TI"] == "Title" and fixed["UT"] == "WOS:123"
    assert raw["TI"].count("\t") == 7  # original input dictionary remains intact
    broken = dict(raw, DI="10.9999/conflict")
    assert repair_publication(broken, headers)[1] == "needs_review"
    assert repair_publication(dict(raw, TI=raw["TI"] + "\textra"), headers)[1] == "needs_review"


def test_workbook_reader_detects_header_and_preserves_literal_NA(tmp_path):
    from openpyxl import Workbook
    book = Workbook()
    sheet = book.active
    sheet.append(["Column1", "Column2", "Column3", "Column4", "Column5"])
    sheet.append([" TI ", " AB ", " GA ", " PD ", " PI "])
    sheet.append(["Title", "NA", "record", "WO 17 May 1990", "DE 07 Nov 1988"])
    path = tmp_path / "fixture.xlsx"
    book.save(path)
    frame, audit = read_workbook(path, "Patent")
    assert frame.iloc[0]["AB"] == "NA"
    assert frame.iloc[0].source_row == 3
    assert audit[0]["header_row"] == 2


def test_group_transitivity_and_doi_normalization():
    groups = UnionFind(4)
    groups.union(0, 1)
    groups.union(1, 2)
    assert groups.find(0) == groups.find(2) != groups.find(3)
    assert doi_key(" HTTPS://doi.org/10.1234/ABC ") == "10.1234/abc"


def test_eligibility_precedence_is_exclusive():
    row = dict(structural_issue=False, text_structure_issue=False, title="Title", duplicate_of="", retracted=False, relevance="CORE", language_eligible=True, character_issue=False, abnormal_length=False, abstract="Abstract")
    records = [row, dict(row, retracted=True, relevance="OUT_OF_SCOPE"), dict(row, structural_issue=True, duplicate_of="other")]
    frame = pd.DataFrame(records)
    apply_eligibility(frame, {"policy": {"exclude_retracted": True, "main_relevance": ["CORE", "RELATED"]}})
    assert frame.disposition.tolist() == ["MAIN_CORPUS", "RETRACTED", "STRUCTURE_REVIEW"]
    assert frame.eligible_finetune.tolist() == [True, False, False]


def test_language_detection_deterministic():
    task = ("This research investigates hard carbon electrodes for sodium ion batteries and their electrochemical storage performance.", 42)
    assert language_worker(task) == language_worker(task)
    assert language_worker(task)[0] == "en"


def test_duplicate_policy_distinguishes_patents_and_doi_conflicts():
    import logging
    base = dict(title="Hard carbon electrodes for sodium ion storage", abstract="Reversible sodium storage in porous hard carbon electrodes with stable cycling.", source_sheet="test", source_row=2)
    frame = pd.DataFrame([
        dict(base, doc_id="patent:a", native_id="a", source="Patent", identifier="PN1"),
        dict(base, doc_id="patent:b", native_id="b", source="Patent", identifier="PN2"),
        dict(base, doc_id="publication:a", native_id="WOS:a", source="Publication", identifier="10.1234/same"),
        dict(base, doc_id="publication:b", native_id="WOS:b", source="Publication", identifier="10.1234/same"),
        dict(base, doc_id="publication:c", native_id="WOS:c", source="Publication", identifier="10.1234/same", abstract="Different scientific work about a lithium electrode and its chemistry."),
    ])
    meta = pd.DataFrame([dict(source=r.source, publication_numbers_json=json.dumps([r.identifier]), priority_information="") for r in frame.itertuples()])
    meta.at[0, "priority_information"] = "JP016346 29 Jan 2016"
    meta.at[1, "priority_information"] = "JP016346 29 Jan 2020"
    config = {"policy": {"near_duplicate_batch_size": 4, "near_duplicate_title_similarity": .88, "near_duplicate_abstract_jaccard": .8}}
    result = duplicate_audit(frame, meta, config, logging.getLogger("test"))
    assert frame.loc[:1, "duplicate_of"].eq("").all()
    assert frame.at[3, "duplicate_of"] == "publication:a"
    assert frame.at[4, "duplicate_of"] == ""
    assert frame.split_group_id.nunique() == 1
    assert "doi" in set(result.kind)
    assert "patent_shared_priority_application" not in set(result.kind)
