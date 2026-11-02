"""Scientific topic representations and bounded diagnostics; no neural inference.

Input strings are the existing immutable, chemical-aware Stage 1 token strings.
The explicit document separator prevents n-grams across aggregated documents.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
from scipy.stats import chi2_contingency

BOUNDARY = "\u241e"
PROTECTED = frozenset("sodium lithium cathode anode electrolyte electrode voltage capacity charge discharge carbon hard_carbon sodium_ion lithium_ion Na-ion Na3V2(PO4)3 Na_xCoO2 mAh mah Wh/kg".split())
GENERIC = frozenset("obtain obtained obtaining prepare prepared preparing use using used add added adding source according first second third plurality fixedly utility embodiment comprising comprise recent summarize summarized summarizing discuss discussed discussing portion review reviews progress future challenge challenges development present invention novelty advantage detailed description independent claim claims include includes including provide provided based following relate relates related thereof wherein said method high superior excellent performance research study studies material materials".split())
# Technical words in GENERIC are assessed but never automatically removed.
RETAIN = PROTECTED | frozenset("material materials performance method high source".split())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def reserve_output(path):
    path = Path(path).resolve()
    # A run may resume explicit phases but never adopts an unrelated directory.
    marker = path / "run_identity.json"
    if path.exists() and any(path.iterdir()) and not marker.exists():
        raise FileExistsError(f"Refusing existing non-experiment directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write_json(marker, {"pipeline": "batterybert_topic_audit_v1", "training": False,
                            "embeddings_recomputed": False})
    elif json.loads(marker.read_text(encoding="utf-8"))["pipeline"] != "batterybert_topic_audit_v1":
        raise ValueError("Output belongs to another pipeline")
    return path


class ScientificCountVectorizer(CountVectorizer):
    """Generate n-grams within original documents, preserving frozen tokens."""
    def build_analyzer(self):
        stop = set(self.stop_words or []) - PROTECTED
        low, high = self.ngram_range
        def analyze(document):
            for segment in document.split(BOUNDARY):
                tokens = [t for t in segment.split() if t not in stop]
                for size in range(low, high + 1):
                    for start in range(len(tokens) - size + 1):
                        yield " ".join(tokens[start:start + size])
        return analyze


def stopword_evidence(texts, sources):
    import pandas as pd
    counts = {source: Counter() for source in sorted(set(sources))}
    totals = Counter(sources)
    for tokens, source in zip(texts, sources):
        counts[source].update(set(tokens))
    rows = []
    selected = set(ENGLISH_STOP_WORDS) - RETAIN
    for term in sorted(GENERIC | PROTECTED):
        df = sum(c[term] for c in counts.values())
        rates = {s: counts[s][term] / totals[s] for s in counts}
        remove = term in GENERIC - RETAIN and df >= max(20, int(.002 * len(texts)))
        if remove:
            selected.add(term)
        rows.append(dict(token=term, document_frequency=df,
            patent_document_rate=rates.get("Patent", 0), publication_document_rate=rates.get("Publication", 0),
            remove=remove, reason="Frequent linguistic/writing boilerplate, inspected candidate list" if remove
                else "Protected technical/context-sensitive term" if term in RETAIN else "Insufficient corpus support"))
    return sorted(selected), pd.DataFrame(rows)


def repair_phrase_placeholders(texts, phrases):
    """Recover known phrase placeholders embedded in hyphenated frozen tokens.

    The legacy tokenizer restored only tokens exactly equal to a placeholder;
    `all-phraseplaceholderi` escaped that lookup. This repairs representation
    tokens only, without rewriting the frozen LDA corpus or guessing chemistry.
    """
    mapping = {"phraseplaceholder" + chr(97 + i): p.replace(" ", "_")
               for i, p in enumerate(sorted(phrases, key=len, reverse=True))}
    rows, repaired = [], []
    for doc_index, tokens in enumerate(texts):
        doc = []
        for token in tokens:
            fixed = re.sub(r"phraseplaceholder[a-z]", lambda m: mapping.get(m.group(), m.group()), token)
            if "placeholder" in fixed:
                raise ValueError(f"Unresolved preprocessing placeholder in document {doc_index}: {fixed}")
            doc.append(fixed)
            if fixed != token:
                rows.append({"doc_index": doc_index, "before": token, "after": fixed})
        repaired.append(doc)
    return repaired, rows


def validate_labels(labels, n):
    labels = np.asarray(labels)
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Assignments must have one integer label per document")
    if np.any(labels < -1):
        raise ValueError("Only -1 is an outlier label")
    return labels.astype(np.int32)


def validate_mapping(before, after):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or not np.array_equal(before == -1, after == -1):
        raise ValueError("Reduction changed document count or outlier mask")
    mapping = {}
    for old, new in zip(before, after):
        old, new = int(old), int(new)
        if old in mapping and mapping[old] != new:
            raise ValueError("Reduction split an original cluster")
        mapping[old] = new
    return mapping


def distribution(labels, sources):
    import pandas as pd
    labels = validate_labels(labels, len(sources))
    table = pd.crosstab(pd.Series(labels, name="topic_id"), pd.Series(sources, name="source"))
    sizes = table.sum(axis=1)
    summary = pd.DataFrame({"topic_id": sizes.index, "documents": sizes.values,
                           "corpus_share": sizes.values / len(labels), "is_outlier": sizes.index == -1})
    nonout = sizes[sizes.index != -1].sort_values(ascending=False)
    chi2 = float(chi2_contingency(table, correction=False)[0]) if min(table.shape) > 1 else 0.
    denom = len(labels) * min(table.shape[0] - 1, table.shape[1] - 1)
    metrics = dict(topics=len(nonout), documents=len(labels), outliers=int((labels == -1).sum()),
        outlier_ratio=float(np.mean(labels == -1)), largest_topic_share=float(nonout.iloc[0] / len(labels)) if len(nonout) else 0.,
        top2_topic_share=float(nonout.head(2).sum() / len(labels)),
        source_ami=float(adjusted_mutual_info_score(sources, labels)),
        source_cramers_v=float(np.sqrt(chi2 / denom)) if denom else 0.)
    return summary, table, metrics


def sample_cosines(vectors, labels, seed=42, pairs=3000):
    rng = np.random.default_rng(seed)
    a = rng.integers(len(labels), size=pairs * 4)
    b = rng.integers(len(labels), size=pairs * 4)
    labels = np.asarray(labels)
    cosine = np.einsum("ij,ij->i", vectors[a], vectors[b])
    groups = {"within_topic": (labels[a] == labels[b]) & (a != b) & (labels[a] != -1),
              "between_topic": (labels[a] != labels[b]) & (labels[a] != -1) & (labels[b] != -1)}
    return {name: {"pairs": int(mask[:].sum()), "mean": float(cosine[mask].mean()) if mask.any() else None,
                   "std": float(cosine[mask].std()) if mask.any() else None}
            for name, mask in groups.items()}


def centroid_scores(vectors, labels):
    labels = np.asarray(labels)
    scores = np.zeros(len(labels), dtype=np.float32)
    centers = []
    for topic in sorted(set(labels) - {-1}):
        ids = np.flatnonzero(labels == topic)
        center = normalize(np.asarray(vectors[ids]).mean(axis=0, keepdims=True))[0]
        scores[ids] = vectors[ids] @ center
        centers.append(center)
    redundancy = None
    if len(centers) > 1:
        sim = np.asarray(centers) @ np.asarray(centers).T
        redundancy = float(sim[np.triu_indices(len(centers), 1)].mean())
    return scores, redundancy


def phrase_reference(texts, feature_set):
    """All candidates share this exact reference; multiword features are atomic.

    Only contiguous occurrences in the ORIGINAL frozen tokens count, even if
    vectorizer stop filtering manufactured an adjacency. Coverage is reported.
    """
    phrases = {tuple(term.split()): term.replace(" ", "\u2581") for term in feature_set if " " in term}
    widths = sorted({len(p) for p in phrases})
    for tokens in texts:
        doc = []
        for i, token in enumerate(tokens):
            doc.append(token)
            for width in widths:
                phrase = tuple(tokens[i:i + width])
                if phrase in phrases:
                    doc.append(phrases[phrase])
        yield doc
