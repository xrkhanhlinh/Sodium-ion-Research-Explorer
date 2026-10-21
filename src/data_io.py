"""Workbook ingestion preserving raw values and physical Excel coordinates."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

SCHEMA = ["doc_id", "source", "title", "abstract", "text_for_model", "year", "authors", "organization", "keywords", "classification", "identifier", "document_type", "language", "source_name"]
DATE_PATTERN = re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+((?:19|20)\d{2})\b")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def value_string(value):
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def read_workbook(path, source):
    book = load_workbook(path, read_only=True, data_only=False)
    records, sheets = [], []
    try:
        for sheet in book.worksheets:
            iterator = sheet.iter_rows(values_only=True)
            physical_row, header, original_header = 0, None, None
            sheet_records = []
            for row in iterator:
                physical_row += 1
                if not any(v is not None and str(v).strip() for v in row):
                    continue
                if header is None:
                    if str(row[0]).strip().startswith("Column"):
                        continue
                    original_header = [value_string(v) for v in row]
                    header = [v.strip() for v in original_header]
                    named = [h for h in header if h]
                    if len(set(named)) != len(named):
                        raise ValueError(f"Duplicate normalized headers: {source}/{sheet.title}")
                    header_row = physical_row
                    continue
                rec = {name: value_string(row[i]) if i < len(row) else "" for i, name in enumerate(header) if name}
                unnamed_nonempty = {str(i + 1): value_string(v) for i, v in enumerate(row) if (i >= len(header) or not header[i]) and v is not None and str(v).strip()}
                rec.update(source_file=path.name, source_sheet=sheet.title, source_row=physical_row, raw_fields_json=json.dumps({str(i + 1): value_string(v) for i, v in enumerate(row)}, ensure_ascii=False), unnamed_nonempty_json=json.dumps(unnamed_nonempty, ensure_ascii=False))
                sheet_records.append(rec)
            if header:
                required = {"TI", "AB", "GA", "PD", "PI"} if source == "Patent" else {"TI", "AB", "UT", "DI", "PY"}
                if not required.issubset(header):
                    raise ValueError(f"Missing required columns in {sheet.title}: {required-set(header)}")
            sheets.append(dict(name=sheet.title, header_row=header_row if header else None, physical_columns=sheet.max_column, record_count=len(sheet_records), headers=header or [], original_headers=original_header or []))
            records.extend(sheet_records)
    finally:
        book.close()
    data_headers = [s["headers"] for s in sheets if s["record_count"]]
    if data_headers and any(h != data_headers[0] for h in data_headers):
        raise ValueError(f"Inconsistent nonempty sheet headers: {source}")
    return pd.DataFrame(records), sheets


def extract_dates(text):
    dates = []
    for match in DATE_PATTERN.finditer(text or ""):
        try:
            month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(match[2]) + 1
            dates.append(datetime(int(match[3]), month, int(match[1])).date().isoformat())
        except ValueError:
            pass
    return sorted(set(dates))


def repair_publication(row, headers):
    """Only restore a complete positional suffix with independently valid anchors."""
    original = dict(row)
    title = row.get("TI", "")
    if "\t" not in title:
        return original, "none", ""
    start = headers.index("TI")
    parts = title.split("\t")
    if len(parts) != len(headers) - start:
        return original, "needs_review", "tab_suffix_field_count_mismatch"
    suffix = dict(zip(headers[start:], parts))
    valid = (re.fullmatch(r"WOS:\S+", suffix.get("UT", "")) and re.fullmatch(r"(?:19|20)\d{2}", suffix.get("PY", "")) and re.fullmatch(r"10\.\d{4,9}/\S+", suffix.get("DI", "")) and bool(suffix.get("SO", "")) and bool(suffix.get("AB", "")))
    conflicts = [h for h, v in suffix.items() if h != "TI" and str(row.get(h, "")).strip() and str(row[h]) != v]
    if not valid or conflicts:
        return original, "needs_review", "invalid_suffix_anchors_or_conflicting_populated_cells"
    repaired = dict(row)
    repaired.update(suffix)
    return repaired, "repaired", "exact_header_suffix_alignment_validated_UT_DOI_PY_SO_AB"


def integrate(patent, publication, publication_headers, repair_enabled=True):
    rows, metadata, actions = [], [], []
    for source, frame in [("Patent", patent), ("Publication", publication)]:
        for raw in frame.to_dict("records"):
            record, repair, reason = (raw, "none", "") if source == "Patent" or not repair_enabled else repair_publication(raw, publication_headers)
            native = str(record.get("GA" if source == "Patent" else "UT", "")).strip()
            locator = f"{raw['source_sheet']}:{raw['source_row']}"
            doc_id = source.lower() + ":" + (native or "missing-" + hashlib.sha256(locator.encode()).hexdigest()[:16])
            date_list = extract_dates(record.get("PD", "")) if source == "Patent" else []
            priority_dates = extract_dates(record.get("PI", "")) if source == "Patent" else []
            py = record.get("PY", "").strip()
            year = int(date_list[0][:4]) if date_list else int(py) if re.fullmatch(r"(?:19|20)\d{2}", py) else None
            row = dict(doc_id=doc_id, source=source, title=record.get("TI", ""), abstract=record.get("AB", ""), text_for_model=" ".join(v for v in [record.get("TI", ""), record.get("AB", "")] if v).strip(), year=year, authors=record.get("AU" if source == "Patent" else "AF", ""), organization=record.get("AE" if source == "Patent" else "C3", ""), keywords=record.get("DE", "") if source == "Publication" else "", classification=record.get("IP" if source == "Patent" else "WC", ""), identifier=record.get("PN" if source == "Patent" else "DI", ""), document_type="Patent" if source == "Patent" else record.get("DT", ""), language=record.get("LA", "") if source == "Publication" else "", source_name="Derwent" if source == "Patent" else record.get("SO", ""))
            row.update(source_file=raw["source_file"], source_sheet=raw["source_sheet"], source_row=raw["source_row"], native_id=native, repair_status=repair, structural_issue=repair == "needs_review" or raw.get("unnamed_nonempty_json", "{}") != "{}", year_basis="earliest_publication_date_in_PD" if date_list else "PY" if year else "unresolved", year_source_raw=record.get("PD" if source == "Patent" else "PY", ""), priority_year=int(priority_dates[0][:4]) if priority_dates else None)
            rows.append(row)
            metadata.append(dict(doc_id=doc_id, source=source, source_file=raw["source_file"], source_sheet=raw["source_sheet"], source_row=raw["source_row"], data_provider="Derwent" if source == "Patent" else "Web of Science", venue=record.get("SO", "") if source == "Publication" else "", organization_role="assignee" if source == "Patent" else "author_affiliation", classification_system="IPC" if source == "Patent" else "WoS Categories", keywords_plus=record.get("ID", "") if source == "Publication" else "", priority_information=record.get("PI", "") if source == "Patent" else "", publication_dates_json=json.dumps(date_list), priority_dates_json=json.dumps(priority_dates), publication_numbers_json=json.dumps([v.strip() for v in record.get("PN", "").split(";") if v.strip()]) if source == "Patent" else "[]", original_fields_json=raw["raw_fields_json"], interpreted_fields_json=json.dumps(record, ensure_ascii=False), raw_language=record.get("LA", "") if source == "Publication" else "", patent_publication_details=record.get("PD", "") if source == "Patent" else ""))
            if repair != "none":
                actions.append(dict(doc_id=doc_id, source=source, source_file=raw["source_file"], source_sheet=raw["source_sheet"], source_row=raw["source_row"], action="repair_structure" if repair == "repaired" else "needs_review_structure", reason=reason, before=raw["TI"], after=record["TI"]))
    combined, meta = pd.DataFrame(rows), pd.DataFrame(metadata)
    # Preserve every occurrence if source exports repeat a native identifier.
    duplicate_ids = combined["doc_id"].duplicated(keep=False)
    combined["native_id_repeated"] = duplicate_ids
    for idx in combined.index[duplicate_ids]:
        suffix = hashlib.sha256(f"{combined.at[idx,'source_sheet']}:{combined.at[idx,'source_row']}".encode()).hexdigest()[:10]
        combined.at[idx, "doc_id"] += ":occurrence-" + suffix
        meta.at[idx, "doc_id"] = combined.at[idx, "doc_id"]
    for name in ["year", "priority_year"]:
        combined[name] = pd.array(combined[name], dtype="Int64")
    return combined, meta, actions
