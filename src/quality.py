"""Quality diagnostics, deterministic related-document groups, and eligibility."""
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_processing import fingerprint, normalize_text, relevance


def language_worker(task):
    text, seed = task
    DetectorFactory.seed = seed
    if sum(c.isalpha() for c in text) < 25:
        return "und", 0.0
    try:
        result = detect_langs(text)[0]
        return result.lang, float(result.prob)
    except LangDetectException:
        return "und", 0.0


def assess_language(frame, config, cache_path, logger):
    samples = frame.text_for_model.str.slice(0, config["policy"]["language_sample_chars"]).tolist()
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    prefix = f"langdetect1.0.9|seed={config['seed']}|"
    keys = [hashlib.sha256((prefix + text).encode()).hexdigest() for text in samples]
    pending = {key: sample for key, sample in zip(keys, samples) if key not in cache}
    logger.info("Language detection: %s uncached text samples", len(pending))
    def save_cache():
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache), encoding="utf-8")
        temporary.replace(cache_path)

    if pending:
        tasks = [(t, config["seed"]) for t in pending.values()]
        workers = config["policy"]["language_workers"]
        pending_keys = list(pending)
        if workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    values = list(pool.map(language_worker, tasks, chunksize=64))
                cache.update(dict(zip(pending_keys, values)))
            except PermissionError:
                logger.warning("Process creation denied; using serial language detection")
                workers = 1
        if workers == 1:
            for start in range(0, len(tasks), 256):
                values = list(map(language_worker, tasks[start:start + 256]))
                cache.update(dict(zip(pending_keys[start:start + 256], values)))
                save_cache()
                if start % 2048 == 0:
                    logger.info("Language detection completed %s/%s uncached samples", min(start + 256, len(tasks)), len(tasks))
        save_cache()
    frame["detected_language"] = [cache[k][0] for k in keys]
    frame["language_confidence"] = [cache[k][1] for k in keys]
    # Keep original source language separately; standardized language is detected text language.
    frame["language_metadata"] = frame.language
    frame["language"] = frame.detected_language
    frame["language_eligible"] = (frame.detected_language == "en") & (frame.language_confidence >= config["policy"]["minimum_language_confidence"])
    mapping = {"English": "en", "Chinese": "zh-cn", "Japanese": "ja", "Korean": "ko", "French": "fr", "Portuguese": "pt", "Polish": "pl", "German": "de"}
    expected = frame.language_metadata.map(mapping)
    frame["language_metadata_mismatch"] = expected.notna() & (expected != frame.detected_language)


def assess_text(frame):
    for field in ["title", "abstract"]:
        frame[f"{field}_chars"] = frame[field].str.len()
        frame[f"{field}_words"] = frame[field].str.split().str.len()
    frame["retracted"] = frame.document_type.str.contains("retracted", case=False, na=False) | frame.title.str.contains(r"retracted article|被撤回", case=False, na=False)
    frame["character_issue"] = frame.text_for_model.str.contains(r"\ufffd|Ã.|â€|Â\s", regex=True, na=False)
    frame["abnormal_length"] = (frame.title_chars < 15) | (frame.title_chars > 1000) | ((frame.abstract_chars > 0) & (frame.abstract_chars < 80)) | (frame.abstract_chars > 20000)
    frame["text_structure_issue"] = frame.title.str.contains("\t", regex=False) | frame.abstract.str.contains("\t", regex=False)
    labels = [relevance(t, a) for t, a in zip(frame.title, frame.abstract)]
    frame["relevance"] = [x[0] for x in labels]
    frame["relevance_reason"] = [x[1] for x in labels]
    frame["relevance_method"] = "context_rules_v1_not_expert_ground_truth"


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def doi_key(text):
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text.strip().casefold())


def duplicate_audit(frame, metadata, config, logger):
    n = len(frame)
    uf = UnionFind(n)
    events = []
    title = frame.title.map(fingerprint)
    abstract = frame.abstract.map(fingerprint)
    # Text equality alone never removes distinct patent records.
    frame["duplicate_of"] = ""
    keys = {
        "doc_id": frame.source + ":" + frame.native_id,
        "doi": pd.Series([doi_key(v) if s == "Publication" else "" for s, v in zip(frame.source, frame.identifier)]),
        "title": title,
        "abstract": abstract,
        "title_abstract": pd.Series([t + "\n" + a if t and a else "" for t, a in zip(title, abstract)]),
    }

    def add_group(kind, key, indices, confidence, action="retain_related"):
        gid = kind + ":" + hashlib.sha256(key.encode()).hexdigest()[:16]
        for i in indices[1:]:
            uf.union(indices[0], i)
        for i in indices:
            events.append(dict(group_id=gid, kind=kind, doc_id=frame.at[i, "doc_id"], source=frame.at[i, "source"], source_sheet=frame.at[i, "source_sheet"], source_row=int(frame.at[i, "source_row"]), related_to=frame.at[indices[0], "doc_id"], confidence=confidence, action=action, evidence=key[:500]))

    for kind, series in keys.items():
        nonblank = series.ne("")
        if kind == "doc_id":
            nonblank &= frame.native_id.ne("")
        for key, group in series[nonblank].groupby(series[nonblank], sort=True):
            indices = list(group.index)
            if len(indices) < 2:
                continue
            add_group(kind, key, indices, "exact_normalized_value")
            if kind in {"doi", "doc_id"}:
                # Only consolidate confirmed duplicate content, not DOI alone.
                signatures = {}
                for i in indices:
                    signature = (frame.at[i, "source"], title[i], abstract[i])
                    if not signature[1] or not signature[2]:
                        continue
                    if signature in signatures:
                        canonical = signatures[signature]
                        if frame.at[i, "source"] == "Publication" or frame.at[i, "native_id"] == frame.at[canonical, "native_id"]:
                            frame.at[i, "duplicate_of"] = frame.at[canonical, "doc_id"]
                    else:
                        signatures[signature] = i

    # Link individual patent publication/priority identifiers, not whole list strings.
    for field, kind in [("publication_numbers_json", "patent_shared_publication_number"), ("priority_information", "patent_shared_priority_application")]:
        mapping = {}
        for i, rec in metadata[metadata.source == "Patent"].iterrows():
            if field.endswith("_json"):
                values = json.loads(rec[field])
            else:
                # Short application serials can recur in different years.
                # Retain the full component, including its priority date.
                values = [normalize_text(item).upper() for item in rec[field].split(";") if item.strip()]
            for value in set(values):
                if value:
                    mapping.setdefault(value, []).append(i)
        for key, indices in sorted(mapping.items()):
            if len(indices) > 1:
                add_group(kind, key, indices, "candidate_relationship_not_verified_patent_family")

    logger.info("Near-duplicate detection: title TF-IDF candidate blocking + abstract Jaccard verification")
    valid = [i for i in range(n) if title[i] and abstract[i]]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=100000, lowercase=True, dtype=np.float32, token_pattern=r"(?u)\b\w\w+\b")
    matrix = vectorizer.fit_transform(title.iloc[valid])
    wordsets = {i: set(re.findall(r"[a-z0-9]+", abstract[i])) for i in valid}
    batch = config["policy"]["near_duplicate_batch_size"]
    threshold = config["policy"]["near_duplicate_title_similarity"]
    pairs = 0
    for start in range(0, len(valid), batch):
        scores = (matrix[start:start + batch] @ matrix.T).tocoo()
        selected = np.flatnonzero(scores.data >= threshold)
        for pos in selected:
            a, b = start + int(scores.row[pos]), int(scores.col[pos])
            if a >= b:
                continue
            i, j = valid[a], valid[b]
            if title[i] == title[j] and abstract[i] == abstract[j]:
                continue
            left, right = wordsets[i], wordsets[j]
            jac = len(left & right) / max(1, len(left | right))
            if jac >= config["policy"]["near_duplicate_abstract_jaccard"]:
                key = frame.at[i, "doc_id"] + "|" + frame.at[j, "doc_id"]
                add_group("near_duplicate", key, [i, j], f"title_cosine={scores.data[pos]:.6f};abstract_word_jaccard={jac:.6f}")
                pairs += 1
    logger.info("Near-duplicate candidate pairs retained for review: %s", pairs)
    roots = {}
    for i in range(n):
        roots.setdefault(uf.find(i), []).append(i)
    frame["split_group_id"] = ""
    for indices in roots.values():
        key = "|".join(sorted(frame.loc[indices, "doc_id"]))
        gid = "related:" + hashlib.sha256(key.encode()).hexdigest()[:20]
        frame.loc[indices, "split_group_id"] = gid
    columns = ["group_id", "kind", "doc_id", "source", "source_sheet", "source_row", "related_to", "confidence", "action", "evidence"]
    result = pd.DataFrame(events, columns=columns)
    if not result.empty:
        result.loc[result.doc_id.isin(frame.loc[frame.duplicate_of.ne(""), "doc_id"]), "action"] = "excluded_only_if_confirmed_duplicate_else_retained"
    return result


def apply_eligibility(frame, config):
    def disposition(row):
        if row.structural_issue or row.text_structure_issue or not row.title:
            return "STRUCTURE_REVIEW"
        if row.duplicate_of:
            return "CONFIRMED_DUPLICATE"
        if row.retracted and config["policy"]["exclude_retracted"]:
            return "RETRACTED"
        if row.relevance == "OUT_OF_SCOPE":
            return "OUT_OF_SCOPE"
        if row.relevance not in config["policy"]["main_relevance"]:
            return "RELEVANCE_REVIEW"
        if not row.language_eligible:
            return "LANGUAGE_REVIEW"
        if row.character_issue or row.abnormal_length:
            return "TEXT_QUALITY_REVIEW"
        return "MAIN_CORPUS"
    frame["disposition"] = [disposition(r) for r in frame.itertuples()]
    frame["eligible_search"] = frame.disposition.eq("MAIN_CORPUS")
    frame["eligible_topic"] = frame.eligible_search
    frame["eligible_finetune"] = frame.eligible_search & frame.title.ne("") & frame.abstract.ne("")


def missing_report(frame, stage, fields, source_column="source"):
    rows = []
    for source, group in frame.groupby(source_column):
        for field in fields:
            values = group[field]
            absent = values.isna() | values.astype(str).str.strip().eq("")
            rows.append(dict(stage=stage, source=source, feature=field, record_count=len(group), missing_count=int(absent.sum()), missing_rate=float(absent.mean()), completeness=float(1 - absent.mean())))
    return pd.DataFrame(rows)
