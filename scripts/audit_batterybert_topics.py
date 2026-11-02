"""Isolated BatteryBERT topic audit and staged CPU experiments.

Run phases sequentially to keep transformers and clustering out of shared RAM.
No BatteryBERT training, embedding generation, or production model replacement.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "4"
os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))

import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score
from src.batterybert_topic_tools import (BOUNDARY, GENERIC, PROTECTED, ScientificCountVectorizer,
    sha, write_json, reserve_output, stopword_evidence, validate_labels, validate_mapping,
    distribution, sample_cosines, centroid_scores, phrase_reference)
from src.batterybert_topic_tools import repair_phrase_placeholders

OLD = ROOT / "outputs/model_comparison_v2"
BASE = OLD / "bertopic_batterybert_k10/pretrained"
ENC = OLD / "batterybert_pretrained"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_inputs():
    from src.model_comparison import corpus, validate_vectors
    main, texts = corpus()
    if len(main) != 13702 or not main.doc_id.is_unique or len(texts) != len(main):
        raise ValueError("Expected 13,702 unique aligned corpus records")
    ids = json.loads((ENC / "ordered_doc_ids.json").read_text())
    stats = validate_vectors(ENC / "document_embeddings.npy", ids, main.doc_id.tolist(), 768)
    if stats["sha256"] != json.loads((ENC / "embedding_success.json").read_text())["sha256"]:
        raise ValueError("Embedding checksum does not match validated checkpoint")
    return main, texts, np.load(ENC / "document_embeddings.npy", mmap_mode="r"), stats


def assert_run_inputs(out):
    """Refuse stale representation/clustering caches after any frozen input change."""
    paths = [ROOT / "data/processed/main_corpus.parquet", ROOT / "data/processed/lda_tokens.parquet",
             ENC / "document_embeddings.npy", ENC / "ordered_doc_ids.json", BASE / "model.pickle"]
    actual = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    marker = out / "input_fingerprint.json"
    if marker.exists():
        if json.loads(marker.read_text()) != actual:
            raise ValueError("Frozen inputs changed; start a new versioned experiment directory")
    else:
        write_json(marker, actual)


def candidate(out, name, labels, keywords, main, vectors, config, model=None):
    dest = out / "experiments" / name
    if (dest / "complete.json").exists():
        saved = json.loads((dest / "config.json").read_text())
        if saved != config:
            raise ValueError(f"Stale experiment configuration: {name}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    labels = validate_labels(labels, len(main))
    summary, sources, metrics = distribution(labels, main.source.tolist())
    if int(summary.documents.sum()) != len(main):
        raise ValueError("Topic counts do not cover corpus")
    scores, redundancy = centroid_scores(vectors, labels)
    metrics.update(centroid_cosine_redundancy=redundancy, **sample_cosines(vectors, labels))
    rows = [dict(topic_id=int(t), rank=r, token=w, weight=float(p))
            for t, terms in sorted(keywords.items()) for r, (w, p) in enumerate(terms[:10], 1) if w and p > 0]
    keys = pd.DataFrame(rows, columns=["topic_id", "rank", "token", "weight"])
    metrics["keyword_diversity_at_10"] = keys.token.nunique() / len(keys) if len(keys) else 0.
    metrics["generic_keyword_count"] = int(keys.token.isin(GENERIC - PROTECTED).sum())
    metrics["multiword_keyword_count"] = int(keys.token.str.contains(" ").sum())
    metrics["keyword_slots"] = len(keys)
    keys.to_csv(dest / "keywords.csv", index=False)
    np.save(dest / "assignments.npy", labels)
    assignments = main[["doc_id", "source", "year"]].copy()
    assignments["topic_id"] = labels
    assignments["centroid_cosine"] = scores
    assignments.to_csv(dest / "topic_assignments.csv", index=False)
    summary.to_csv(dest / "topic_distribution.csv", index=False)
    sources.to_csv(dest / "topic_source_distribution.csv")
    reps = []
    for topic in sorted(set(labels) - {-1}):
        indices = np.flatnonzero(labels == topic)
        ordered = indices[np.argsort(-scores[indices], kind="stable")]
        rng = np.random.default_rng(42 + int(topic))
        selections = [("central", i) for i in ordered[:3]] + [("peripheral", i) for i in ordered[-2:]]
        selections += [("random", int(i)) for i in rng.choice(indices, min(2, len(indices)), replace=False)]
        for role, i in selections:
            row = main.iloc[i]
            reps.append(dict(topic_id=int(topic), sample_role=role, doc_id=row.doc_id,
                title=row.title, abstract=row.abstract, source=row.source, year=row.year,
                centroid_cosine=float(scores[i])))
    pd.DataFrame(reps).to_csv(dest / "representative_documents.csv", index=False)
    write_json(dest / "config.json", config)
    write_json(dest / "metrics.json", metrics)
    if model is not None:
        model.save(str(dest / "model.pickle"), serialization="pickle")
    write_json(dest / "complete.json", {"assignments_sha256": sha(dest / "assignments.npy"),
                                        "config_sha256": sha(dest / "config.json")})
    log(f"{name}: {metrics['topics']} topics; top2={metrics['top2_topic_share']:.3f}; outliers={metrics['outlier_ratio']:.3f}")


def get_keywords(model):
    return {int(t): model.get_topic(t)[:10] for t in model.get_topics() if t != -1}


def audit(out):
    from bertopic import BERTopic
    main, texts, vectors, stats = load_inputs()
    protected_paths = [ROOT / "data/processed/main_corpus.parquet", ROOT / "data/processed/lda_tokens.parquet",
        ENC / "document_embeddings.npy", ENC / "ordered_doc_ids.json", ENC / "model/pytorch_model.bin",
        ROOT / "src/search_v2.py", ROOT / "src/final_search.py", ROOT / "src/final_system_vectors.py"]
    for folder in (BASE, OLD / "lda_k10", OLD / "bertopic_bge_k10", ROOT / "models/lda"):
        protected_paths.extend(p for p in folder.rglob("*") if p.is_file())
    if not (out / "protected_files.json").exists():
        write_json(out / "protected_files.json", {str(p.relative_to(ROOT)): sha(p) for p in protected_paths if p.exists()})
    model = BERTopic.load(str(BASE / "model.pickle"))
    labels = np.load(BASE / "assignments.npy")
    if not np.array_equal(model.topics_, labels):
        raise ValueError("Saved model and exported assignments differ")
    if json.loads((BASE / "ordered_doc_ids.json").read_text()) != main.doc_id.tolist():
        raise ValueError("Topic IDs and corpus IDs are misaligned")
    initial = np.load(BASE / "initial/assignments.npy")
    mapping = validate_mapping(initial, labels)
    pd.DataFrame(mapping.items(), columns=["initial_topic_id", "final_topic_id"]).to_csv(out / "original_topic_mapping.csv", index=False)
    umap_vectors = np.asarray(model.umap_model.embedding_, dtype=np.float32)
    if umap_vectors.shape[0] != len(main) or not np.isfinite(umap_vectors).all():
        raise ValueError("Original UMAP rows invalid")
    np.save(out / "original_umap.npy", umap_vectors)
    stop, evidence = stopword_evidence(texts, main.source.tolist())
    evidence.to_csv(out / "stopword_evidence.csv", index=False)
    write_json(out / "stopwords.json", stop)
    model_meta = json.loads((ENC / "embedding_manifest.json").read_text())
    for filename, expected in model_meta["weights"].items():
        if sha(ENC / "model" / filename) != expected:
            raise ValueError("BatteryBERT weights differ from embedding provenance")
    chunks = pd.read_parquet(ENC / "document_chunk_counts.parquet")
    if chunks.doc_id.tolist() != main.doc_id.tolist():
        raise ValueError("Chunk metadata IDs mismatch")
    if not np.array_equal(chunks.abstract_tokens, chunks.covered_abstract_tokens):
        raise ValueError("Abstract truncation detected")
    normalized = (main.title.fillna("") + " " + main.abstract.fillna("")).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    duplicates = normalized.duplicated(keep=False)
    main.loc[duplicates, ["doc_id", "source", "title", "split_group_id"]].to_csv(out / "exact_duplicate_documents.csv", index=False)
    # Split-group repetitions are provenance warnings, not a near-duplicate ground truth.
    data = dict(documents=len(main), sources=main.source.value_counts().to_dict(), embeddings=stats,
        checkpoint=model_meta, missing_title=int(main.title.fillna("").str.strip().eq("").sum()),
        missing_abstract=int(main.abstract.fillna("").str.strip().eq("").sum()),
        empty_model_tokens=sum(not t for t in texts), exact_duplicate_documents=int(duplicates.sum()),
        duplicate_extra_rows=int(normalized.duplicated().sum()), repeated_split_group_rows=int(main.split_group_id.duplicated().sum()),
        initial_topics=len(set(initial) - {-1}), final_topics=len(set(labels) - {-1}),
        reduction_applied=not np.array_equal(initial, labels), mapping={str(k): v for k, v in mapping.items()},
        original_umap=model.umap_model.get_params(), original_hdbscan=model.hdbscan_model.get_params(),
        original_vectorizer=dict(fixed_vocabulary=bool(model.vectorizer_model.fixed_vocabulary_),
            vocabulary_size=len(model.vectorizer_model.vocabulary_), ngram_range=model.vectorizer_model.ngram_range,
            min_df=model.vectorizer_model.min_df, lowercase=model.vectorizer_model.lowercase),
        original_reduce_frequent_words=model.ctfidf_model.reduce_frequent_words,
        versions={p: importlib.metadata.version(p) for p in ("bertopic", "umap-learn", "hdbscan", "numpy", "scikit-learn", "gensim", "transformers", "torch")})
    # Estimators expose callables / numpy values: preserve exact readable parameter repr.
    def parameter(value):
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        return value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
    data["original_umap"] = {k: parameter(v) for k, v in data["original_umap"].items()}
    data["original_hdbscan"] = {k: parameter(v) for k, v in data["original_hdbscan"].items()}
    write_json(out / "audit_facts.json", data)
    candidate(out, "baseline", labels, get_keywords(model), main, vectors,
              {"algorithm": "saved HDBSCAN eom", "representation": "fixed LDA unigrams", "seed": 42})
    log(f"AUDIT PASS: norm range {stats['minimum_l2_norm']:.8f}..{stats['maximum_l2_norm']:.8f}; reduction={data['reduction_applied']}")


def vectorizer(out, docs, ngrams, repaired=False):
    vocab_path = out / f"vocabulary_{'repaired_' if repaired else ''}{ngrams}.json"
    stop = json.loads((out / "stopwords.json").read_text())
    if vocab_path.exists():
        vocabulary = json.loads(vocab_path.read_text())
    else:
        # DF is measured over 13,702 original documents, NOT 10 aggregated topics.
        fit = ScientificCountVectorizer(ngram_range=(1, ngrams), stop_words=stop,
            lowercase=False, min_df=3, max_features=60000)
        fit.fit(docs)
        vocabulary = {k: int(v) for k, v in fit.vocabulary_.items()}
        write_json(vocab_path, vocabulary)
    return ScientificCountVectorizer(ngram_range=(1, ngrams), stop_words=stop,
        lowercase=False, vocabulary=vocabulary, min_df=1)


def represent(out):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    main, texts, vectors, _ = load_inputs()
    docs = [" ".join(t) + " " + BOUNDARY for t in texts]
    for ngrams, frequent in ((1, True), (2, False), (2, True), (3, True)):
        name = f"repr_ng{ngrams}_sqrt{int(frequent)}"
        if (out / "experiments" / name / "complete.json").exists():
            continue
        model = BERTopic.load(str(BASE / "model.pickle"))
        log("START " + name)
        before = np.asarray(model.topics_).copy()
        start = time.perf_counter()
        model.update_topics(docs, vectorizer_model=vectorizer(out, docs, ngrams),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=frequent))
        if not np.array_equal(before, model.topics_):
            raise ValueError("Representation-only update changed assignments")
        model.topic_embeddings_ = np.asarray([np.asarray(vectors[before == topic]).mean(axis=0)
            for topic in sorted(set(before))])  # Keep semantic centroids for any later reduction.
        candidate(out, name, before, get_keywords(model), main, vectors,
            dict(algorithm="representation only", ngrams=ngrams, reduce_frequent_words=frequent,
                 assignments_unchanged=True, vocabulary_df_population="original documents", min_df=3,
                 seconds=time.perf_counter() - start), model)
        del model
        gc.collect()


def fit_candidate(out, name, main, texts, vectors, coordinates, labels, cluster, config):
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.cluster import BaseCluster
    from bertopic.vectorizers import ClassTfidfTransformer
    from src.stage3_bertopic_model import FrozenTokensBERTopic
    # Assignments already computed from the validated embeddings. BaseCluster is
    # supervised pass-through; no second clustering or neural loading occurs.
    docs = [" ".join(t) + " " + BOUNDARY for t in texts]
    model = FrozenTokensBERTopic(embedding_model=None, umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(), vectorizer_model=vectorizer(out, docs, 2),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True), top_n_words=10,
        nr_topics=None, calculate_probabilities=False, verbose=False)
    model.fit_transform(docs, embeddings=np.asarray(vectors), y=np.asarray(labels))
    # BERTopic sorts by frequency; IDs may change, memberships must not.
    if adjusted_rand_score(labels, model.topics_) != 1.:
        raise ValueError("BERTopic representation changed cluster memberships")
    # Restore real fitted inference components after the supervised pass-through.
    # topic_mapper_ retains the raw-estimator-label -> frequency-sorted-ID map.
    if cluster is not None:
        model.hdbscan_model = cluster
    if config.get("components") != 768:
        if config.get("neighbors") == 30:
            import joblib
            model.umap_model = joblib.load(out / "umap_n30_d10.joblib")
        else:
            model.umap_model = BERTopic.load(str(BASE / "model.pickle")).umap_model
    candidate(out, name, model.topics_, get_keywords(model), main, vectors, config, model)
    # Persist the genuine clustering estimator separately for reproducible inference.
    import joblib
    if cluster is not None:
        joblib.dump(cluster, out / "experiments" / name / "cluster.joblib")
    return model, docs


def cluster(out):
    from hdbscan import HDBSCAN
    from sklearn.cluster import KMeans
    from umap import UMAP
    from threadpoolctl import threadpool_limits
    main, texts, vectors, _ = load_inputs()
    original = np.load(out / "original_umap.npy")
    trials = [
        ("hdb_eom30_5", {"neighbors": 15, "components": 5, "min_cluster_size": 30, "min_samples": 5, "selection": "eom"}),
        ("hdb_leaf30_5", {"neighbors": 15, "components": 5, "min_cluster_size": 30, "min_samples": 5, "selection": "leaf"}),
        ("hdb_leaf_n30_d10", {"neighbors": 30, "components": 10, "min_cluster_size": 30, "min_samples": 5, "selection": "leaf"}),
        ("kmeans10_umap", {"neighbors": 15, "components": 5, "algorithm": "KMeans", "n_clusters": 10, "n_init": 10}),
        ("kmeans10_full", {"components": 768, "algorithm": "KMeans", "n_clusters": 10, "n_init": 10}),
    ]
    for name, config in trials:
        if (out / "experiments" / name / "complete.json").exists():
            continue
        start = time.perf_counter()
        log("START " + name)
        if name == "hdb_leaf_n30_d10":
            reducer = UMAP(n_neighbors=30, n_components=10, min_dist=0., metric="cosine", random_state=42,
                           n_jobs=1, low_memory=True)
            from joblib import parallel_backend
            # pynndescent internally requests n_jobs=-1 even when UMAP n_jobs=1.
            # Sequential backend avoids Windows named-pipe/ThreadPool creation.
            with parallel_backend("sequential", n_jobs=1):
                coordinates = reducer.fit_transform(np.asarray(vectors))
            import joblib
            joblib.dump(reducer, out / "umap_n30_d10.joblib")
            np.save(out / "umap_n30_d10.npy", coordinates)
        else:
            coordinates = np.asarray(vectors) if name == "kmeans10_full" else original
        with threadpool_limits(limits=4):
            if config.get("algorithm") == "KMeans":
                estimator = KMeans(n_clusters=10, random_state=42, n_init=10)
            else:
                estimator = HDBSCAN(min_cluster_size=30, min_samples=5,
                    cluster_selection_method=config["selection"], prediction_data=True, core_dist_n_jobs=1)
            labels = estimator.fit_predict(coordinates)
        config.update(seed=42, seconds=time.perf_counter() - start, embedding_sha256=sha(ENC / "document_embeddings.npy"))
        model, docs = fit_candidate(out, name, main, texts, vectors, coordinates, labels, estimator, config)
        if name == "hdb_leaf30_5" and len(set(model.topics_) - {-1}) > 10:
            before = np.asarray(model.topics_).copy()
            model.reduce_topics(docs, nr_topics=10 + int(-1 in before), use_ctfidf=False)
            mapping = validate_mapping(before, model.topics_)
            mapping_frame = pd.DataFrame(mapping.items(), columns=["old_topic_id", "new_topic_id"])
            candidate(out, "hdb_leaf30_5_reduced10", model.topics_, get_keywords(model), main, vectors,
                dict(algorithm="HDBSCAN leaf + semantic reduction", parent=name,
                     initial_topics=len(set(before) - {-1}), target=10, seed=42), model)
            mapping_frame.to_csv(out / "experiments/hdb_leaf30_5_reduced10/reduction_mapping.csv", index=False)
        del model
        gc.collect()
    log("Stage-one clustering completed")


def evaluate(out):
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    main, texts, vectors, _ = load_inputs()
    phrases = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))["preprocessing"]["domain_phrases"]
    texts, _ = repair_phrase_placeholders(texts, phrases)
    experiments = sorted(p for p in (out / "experiments").iterdir() if (p / "complete.json").exists())
    features = set()
    for folder in experiments:
        features.update(pd.read_csv(folder / "keywords.csv").token)
    reference = list(phrase_reference(texts, features))
    dictionary = Dictionary(reference)
    bow = [dictionary.doc2bow(t) for t in reference]
    write_json(out / "coherence_protocol.json", dict(reference="Frozen Stage1 tokens with deterministic phrase-placeholder recovery + exact contiguous occurrences of candidate keyword phrases",
        phrase_encoding="spaces replaced with U+2581; phrases inserted at occurrence position", c_v_window=110,
        processes=1, topn=10, documents=len(reference), reference_dictionary=len(dictionary),
        baseline_original_scores="Retained separately; never compared as same-protocol scores",
        validation="Development corpus; no independent scientific topic labels; seed43 stability is a robustness check"))
    rows = []
    for folder in experiments:
        metrics = json.loads((folder / "metrics.json").read_text())
        keys = pd.read_csv(folder / "keywords.csv")
        grouped = keys.sort_values(["topic_id", "rank"]).groupby("topic_id").token.apply(list)
        topics, eligible_ids, coverages = [], [], []
        for topic_id, words in grouped.items():
            converted = [w.replace(" ", "\u2581") for w in words]
            supported = [w for w in converted if w in dictionary.token2id]
            coverages.append(dict(topic_id=int(topic_id), keyword_count=len(words), supported=len(supported)))
            # Never silently evaluate shorter word lists as top10.
            if len(words) == 10 and len(supported) == 10:
                topics.append(converted)
                eligible_ids.append(int(topic_id))
        metrics["coherence_eligible_topics"] = len(topics)
        metrics["coherence_total_topics"] = len(grouped)
        if topics:
            cv = CoherenceModel(topics=topics, texts=reference, dictionary=dictionary,
                coherence="c_v", window_size=110, processes=1, topn=10).get_coherence_per_topic()
            um = CoherenceModel(topics=topics, corpus=bow, dictionary=dictionary,
                coherence="u_mass", processes=1, topn=10).get_coherence_per_topic()
            metrics.update(c_v=float(np.mean(cv)), u_mass=float(np.mean(um)))
            pd.DataFrame(dict(topic_id=eligible_ids, c_v=cv, u_mass=um)).to_csv(folder / "coherence_per_topic.csv", index=False)
        else:
            metrics.update(c_v=None, u_mass=None)
        # Mark partial-coverage aggregate explicitly; do not hide omitted topics.
        metrics["coherence_full_coverage"] = len(topics) == len(grouped)
        write_json(folder / "keyword_reference_coverage.json", coverages)
        write_json(folder / "quality_metrics.json", metrics)
        rows.append(dict(experiment=folder.name, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}))
        log(f"COHERENCE {folder.name}: Cv={metrics['c_v']}; eligible={len(topics)}/{len(grouped)}")
    pd.DataFrame(rows).to_csv(out / "experiment_comparison.csv", index=False)


def repair(out):
    """Stage-two representation refinement on shortlisted cluster memberships."""
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    main, texts, vectors, _ = load_inputs()
    phrases = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))["preprocessing"]["domain_phrases"]
    texts, changes = repair_phrase_placeholders(texts, phrases)
    recovery = pd.DataFrame(changes)
    recovery["doc_id"] = recovery.doc_index.map(main.doc_id)
    recovery.to_csv(out / "placeholder_repairs.csv", index=False)
    docs = [" ".join(t) + " " + BOUNDARY for t in texts]
    for parent in ("baseline", "kmeans10_full", "kmeans10_umap", "hdb_leaf30_5_reduced10"):
        name = parent + "_repaired"
        if (out / "experiments" / name / "complete.json").exists():
            continue
        path = BASE / "model.pickle" if parent == "baseline" else out / "experiments" / parent / "model.pickle"
        model = BERTopic.load(str(path))
        before = np.asarray(model.topics_).copy()
        log("REPAIR " + name)
        model.update_topics(docs, vectorizer_model=vectorizer(out, docs, 2, repaired=True),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True))
        if not np.array_equal(before, model.topics_):
            raise ValueError("Placeholder recovery changed memberships")
        config = dict(algorithm="representation refinement", parent=parent, ngrams=2,
            reduce_frequent_words=True, phrase_placeholder_recovery=True,
            repaired_tokens=len(changes), repaired_documents=int(recovery.doc_index.nunique()),
            assignments_unchanged=True, seed=42)
        candidate(out, name, before, get_keywords(model), main, vectors, config, model)
        del model
        gc.collect()


def stability(out, name):
    """Independent seed check, never used for iterative parameter search."""
    from hdbscan import HDBSCAN
    from sklearn.cluster import KMeans
    from umap import UMAP
    main, texts, vectors, _ = load_inputs()
    config = json.loads((out / "experiments" / name / "config.json").read_text())
    base_labels = np.load(out / "experiments" / name / "assignments.npy")
    if name.startswith("kmeans"):
        if name == "kmeans10_full":
            coords = np.asarray(vectors)
        else:
            from joblib import parallel_backend
            with parallel_backend("sequential", n_jobs=1):
                coords = UMAP(n_neighbors=config["neighbors"], n_components=config["components"], min_dist=0.,
                    metric="cosine", random_state=43, n_jobs=1, low_memory=True).fit_transform(np.asarray(vectors))
        estimator = KMeans(n_clusters=10, n_init=10, random_state=43)
    else:
        from joblib import parallel_backend
        with parallel_backend("sequential", n_jobs=1):
            coords = UMAP(n_neighbors=config.get("neighbors", 15), n_components=config.get("components", 5),
                min_dist=0., metric="cosine", random_state=43, n_jobs=1, low_memory=True).fit_transform(np.asarray(vectors))
        estimator = HDBSCAN(min_cluster_size=config.get("min_cluster_size", 30), min_samples=config.get("min_samples", 5),
            cluster_selection_method=config.get("selection", "leaf"), core_dist_n_jobs=1)
    alternate = estimator.fit_predict(coords)
    np.save(out / f"{name}_seed43_assignments.npy", alternate)
    write_json(out / f"{name}_stability.json", dict(seed42_vs_seed43_ari=float(adjusted_rand_score(base_labels, alternate)),
        seed43_topics=len(set(alternate) - {-1}), seed43_outliers=int((alternate == -1).sum()),
        note="Clustering robustness on same corpus, not held-out scientific validity"))


def diagnose(out):
    """Bounded nearest-neighbour duplicate audit; at most 64 x N similarities."""
    main, texts, vectors, _ = load_inputs()
    rng = np.random.default_rng(42)
    indices = rng.choice(len(main), 600, replace=False)
    pairs = []
    for start in range(0, len(indices), 64):
        chosen = indices[start:start + 64]
        similarities = np.asarray(vectors[chosen]) @ vectors.T
        similarities[np.arange(len(chosen)), chosen] = -1
        neighbours = similarities.argmax(axis=1)
        for i, j, sim in zip(chosen, neighbours, similarities.max(axis=1)):
            a, b = set(texts[i]), set(texts[j])
            overlap = len(a & b) / max(1, len(a | b))
            if overlap >= .8:
                pairs.append(dict(doc_id=main.doc_id.iloc[i], neighbour_doc_id=main.doc_id.iloc[j],
                    embedding_cosine=float(sim), token_set_jaccard=overlap,
                    title=main.title.iloc[i], neighbour_title=main.title.iloc[j]))
    pd.DataFrame(pairs, columns=["doc_id", "neighbour_doc_id", "embedding_cosine", "token_set_jaccard", "title", "neighbour_title"]).to_csv(out / "sampled_near_duplicate_candidates.csv", index=False)
    write_json(out / "duplicate_sampling_protocol.json", dict(sample_documents=600, seed=42,
        maximum_similarity_block=[64, len(main)], candidate_token_jaccard_threshold=.8,
        candidate_pairs=len(pairs), limitation="Sampled embedding-nearest neighbour screen; not exhaustive text deduplication; no documents excluded"))
    norms = np.linalg.norm(vectors, axis=1)
    write_json(out / "embedding_norm_statistics.json", dict(min=float(norms.min()), max=float(norms.max()),
        mean=float(norms.mean()), std=float(norms.std()), percentiles=np.percentile(norms, [0, 25, 50, 75, 100]).tolist()))
    raw = main.title.fillna("") + " " + main.abstract.fillna("")
    checks = [("Na3V2(PO4)3", "Na3V2(PO4)3"), ("Na_xCoO2", "Na_xCoO2"), ("NaxCoO2", "NaxCoO2"),
        ("Na-ion", "na-ion"), ("Na+", "Na+"), ("mAh", "mah"), ("Wh/kg", "wh"),
        ("hard carbon", "hard_carbon"), ("first-principles", "first-principles")]
    rows = []
    for original, token in checks:
        matched = raw.str.contains(original, regex=False)
        retained = np.asarray([token in t for t in texts])
        rows.append(dict(raw_expression=original, stage1_token_checked=token,
            raw_document_matches=int(matched.sum()), token_document_matches=int(retained.sum()),
            raw_matched_documents_without_expected_token=int((matched & ~retained).sum()),
            note="Stage1 uses lemmatized tokens; case/phrases/units may be normalized. Literal presence screen, not an exhaustive chemical parser test"))
    pd.DataFrame(rows).to_csv(out / "chemical_tokenization_audit.csv", index=False)


def render_topics(folder, destination, labels=None, subtitle=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    summary = pd.read_csv(folder / "topic_distribution.csv")
    keys = pd.read_csv(folder / "keywords.csv")
    topics = summary[~summary.is_outlier].sort_values("topic_id")
    n = len(topics)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(18, max(8, rows * 4.25)), squeeze=False)
    for ax, row in zip(axes.flat, topics.itertuples()):
        words = keys[keys.topic_id == row.topic_id].sort_values("rank")
        label = labels.get(int(row.topic_id), " / ".join(words.token.tolist()[:3])) if labels else " / ".join(words.token.tolist()[:3])
        ax.barh(range(len(words)), words.weight, color="#a56525")
        ax.set_yticks(range(len(words)), words.token, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(f"Topic {row.topic_id}: {textwrap.fill(label, 47)}\n{row.documents:,} documents ({row.corpus_share:.1%} of corpus)",
                     loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("c-TF-IDF weight", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.margins(x=.12)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    outliers = int(summary.loc[summary.is_outlier, "documents"].sum())
    total = int(summary.documents.sum())
    fig.suptitle(f"BatteryBERT pretrained + BERTopic | {n} topics", fontsize=21, fontweight="bold", y=.992)
    fig.text(.5, .966, f"{total:,} documents | Outliers: {outliers:,} ({outliers / total:.2%})\n{subtitle}", ha="center", va="top", fontsize=11)
    fig.text(.5, .012, "Top 10 actual model keywords; labels are tentative, based on title-abstract inspection. No documents excluded.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.15, right=.97, bottom=.05, top=.915, hspace=.85, wspace=.60)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def finalize(out, name):
    import shutil
    from bertopic import BERTopic
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main, texts, vectors, _ = load_inputs()
    provenance = json.loads((ENC / "embedding_manifest.json").read_text())
    for filename, expected in provenance["weights"].items():
        if sha(ENC / "model" / filename) != expected:
            raise ValueError("Current BatteryBERT weights differ from archived embedding provenance")
    shutil.copy2(BASE / "metrics.json", out / "baseline_original_metrics.json")
    folder = out / "experiments" / name
    labels = np.load(folder / "assignments.npy")
    quality = json.loads((folder / "quality_metrics.json").read_text())
    configs = json.loads((folder / "config.json").read_text())
    if configs.get("parent"):
        parent_config = json.loads((out / "experiments" / configs["parent"] / "config.json").read_text())
        configs = {**parent_config, **configs, "representation_operation": configs.get("algorithm"),
                   "algorithm": parent_config.get("algorithm"), "clustering_algorithm": parent_config.get("algorithm")}
    review_path = out / "scientific_review.csv"
    labels_path = out / "topic_labels.csv"
    if not review_path.exists() or not labels_path.exists():
        raise ValueError("Read real representative documents and save scientific_review.csv plus topic_labels.csv before finalizing")
    interpretations = pd.read_csv(labels_path)
    if set(interpretations.topic_id) != set(labels) - {-1}:
        raise ValueError("Every selected topic needs an evidence-based tentative label")
    reviews = pd.read_csv(review_path)
    if not set(reviews.doc_id) <= set(main.doc_id) or set(reviews.topic_id) != set(labels) - {-1}:
        raise ValueError("Scientific review must cover each selected topic with real corpus IDs")
    indexed = main.set_index("doc_id")
    if indexed.loc[reviews.doc_id, "title"].tolist() != reviews.title.tolist():
        raise ValueError("Reviewed titles do not match frozen corpus")
    name_map = interpretations.set_index("topic_id").topic_label.to_dict()
    for source, target in (("topic_assignments.csv", "topic_assignments.csv"),
        ("topic_distribution.csv", "topic_distribution.csv"), ("representative_documents.csv", "representative_documents.csv")):
        frame = pd.read_csv(folder / source)
        frame["topic_label"] = frame.topic_id.map(name_map).fillna("Outlier")
        frame.to_csv(out / target, index=False)
    sources = pd.read_csv(folder / "topic_source_distribution.csv")
    sources["documents"] = sources[[c for c in ("Patent", "Publication") if c in sources]].sum(axis=1)
    for col in ("Patent", "Publication"):
        if col in sources:
            sources[col + "_share"] = sources[col] / sources.documents
    sources["topic_label"] = sources.topic_id.map(name_map).fillna("Outlier")
    sources.to_csv(out / "topic_source_distribution.csv", index=False)
    keys = pd.read_csv(folder / "keywords.csv")
    before = pd.read_csv(out / "experiments/baseline/keywords.csv")
    comparison = pd.concat([before.assign(version="baseline"), keys.assign(version=name)], ignore_index=True)
    comparison["note"] = "Topic IDs are local; rows are not aligned scientific topics across models"
    comparison.to_csv(out / "topic_keywords_before_after.csv", index=False)
    # All representation-only changes share baseline IDs, so this table IS aligned.
    changes = []
    for rep in sorted((out / "experiments").glob("repr*")):
        after = pd.read_csv(rep / "keywords.csv")
        changes.append(before.rename(columns={"token": "before_token", "weight": "before_weight"}).merge(
            after.rename(columns={"token": "after_token", "weight": "after_weight"}), on=["topic_id", "rank"]).assign(experiment=rep.name))
    pd.concat(changes, ignore_index=True).to_csv(out / "representation_only_keywords_before_after.csv", index=False)
    original_labels = np.load(BASE / "assignments.npy")
    for rep in (out / "experiments").glob("repr*"):
        if not np.array_equal(np.load(rep / "assignments.npy"), original_labels):
            raise ValueError("Representation-only assignment preservation failed")
    model = BERTopic.load(str(folder / "model.pickle"))
    if not np.array_equal(model.topics_, labels):
        raise ValueError("Selected model reload changed labels")
    if not np.isfinite(model.c_tf_idf_.data).all():
        raise ValueError("Selected c-TF-IDF is nonfinite")
    sample = np.random.default_rng(42).choice(len(main), 12, replace=False)
    docs = [" ".join(texts[i]) + " " + BOUNDARY for i in sample]
    from joblib import parallel_backend
    with parallel_backend("sequential", n_jobs=1):
        predicted, _ = model.transform(docs, embeddings=np.asarray(vectors[sample]))
    inference = dict(documents=12, matching_saved_assignments=int(np.sum(np.asarray(predicted) == labels[sample])),
        expected=labels[sample].tolist(), predicted=[int(x) for x in predicted],
        note="HDBSCAN approximate prediction may differ at density boundaries; KMeans should agree")
    if name.startswith("kmeans") and inference["matching_saved_assignments"] != 12:
        raise ValueError("Restored KMeans BERTopic inference does not reproduce assignments")
    write_json(out / "model_reload_inference.json", inference)
    shutil.copy2(folder / "model.pickle", out / "bertopic_final.pickle")
    quality.update(selected_experiment=name, scientific_validation="Agent inspection of saved real title-abstract samples; not independent expert validation",
        scientific_acceptance="No satisfactory validated ten-theme model established; fixed-10 development comparison only",
        reviewed_documents=len(reviews), review_counts=reviews.review_status.value_counts().to_dict(),
        review_sampling="One central and one peripheral example per topic plus two scope checks; purposive examples, not a precision estimate",
        representation_only_assignments_unchanged=True, training_invoked=False, embeddings_regenerated=False)
    write_json(out / "topic_quality_metrics.json", quality)
    configs.update(selected_experiment=name, model_id="batterydata/batterybert-cased",
        revision="0106ed8cca65ec6302ada7123048be9c37e31a7d", actual_topics=quality["topics"],
        vectorizer=dict(ngram_range=[1, 2], min_df_original_documents=3, min_df_aggregated_topics=1,
            max_features=60000, stopwords="stopwords.json", vocabulary="vocabulary_repaired_2.json" if name.endswith("_repaired") else "vocabulary_2.json", chemical_case_preserved=True),
        ctfidf=dict(reduce_frequent_words=True), seed=42, embeddings=str((ENC / "document_embeddings.npy").relative_to(ROOT)),
        coherence_protocol="coherence_protocol.json", runtime_versions=json.loads((out / "audit_facts.json").read_text())["versions"])
    (out / "final_model_config.yaml").write_text(yaml.safe_dump(configs, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # Flag possible scope issues for human review; retain every original record.
    flags = []
    import re
    for i, row in enumerate(main.itertuples()):
        reasons = []
        title = str(row.title).lower()
        if re.search(r"superconduct|superfluid|magnetic suscept", title):
            reasons.append("Superconductivity/magnetism: verify sodium-ion battery relevance")
        if re.search(r"fixture|housing|cover|terminal|inspection|protection|mechanical|clamping|welding", title) and not re.search(r"cathode|anode|electrolyte|intercalat|sodium.storage", title):
            reasons.append("Hardware/process document: check specific SIB relevance")
        if reasons:
            flags.append(dict(doc_id=row.doc_id, topic_id=int(labels[i]), title=row.title,
                source=row.source, review_reason="; ".join(reasons), excluded=False))
    pd.DataFrame(flags).to_csv(out / "potentially_irrelevant_documents.csv", index=False)
    render_topics(folder, out / ("bertopic_final_10_topics.png" if quality["topics"] == 10 else "bertopic_final_topics.png"),
                  name_map, subtitle="Provisional fixed-10 comparison: KMeans on original vectors; scientific validation remains incomplete")
    render_comparison(out, name)
    protected = json.loads((out / "protected_files.json").read_text())
    changed = [p for p, expected in protected.items() if not (ROOT / p).exists() or sha(ROOT / p) != expected]
    write_json(out / "preservation_check.json", dict(files_checked=len(protected), changed=changed, status="PASS" if not changed else "FAIL"))
    if changed:
        raise ValueError("Protected artifacts changed: " + repr(changed))
    write_json(out / "final_selection.json", dict(experiment=name, topics=quality["topics"], outliers=quality["outliers"],
        scientific_status="No satisfactory validated ten-theme model; provisional fixed-10 development comparison",
        production_promoted=False, exclusions=0))
    log("FINAL EXPORT COMPLETE: " + name)


def render_comparison(out, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = pd.read_csv(out / "experiment_comparison.csv")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, field, title in zip(axes.flat, ("c_v", "keyword_diversity_at_10", "outlier_ratio", "top2_topic_share"),
        ("Coherence C_v (shared reference)", "Keyword diversity@10", "Outlier fraction", "Two largest topics / full corpus")):
        colors = ["#a56525" if x == name else "#668fad" for x in table.experiment]
        ax.barh(table.experiment, table[field], color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if field == "c_v":
            for bar, complete in zip(ax.patches, table.coherence_full_coverage):
                if not complete:
                    bar.set_hatch("//")
    fig.suptitle("BatteryBERT topic experiments | Development evidence", fontsize=19, fontweight="bold")
    fig.text(.5, .015, "Shared frozen corpus/vectors. Hatched coherence bars have partial topic coverage (see CSV). Balanced sizes/zero outliers do not establish scientific quality.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .035, 1, .96])
    fig.savefig(out / "experiment_comparison.png", dpi=150, facecolor="white")
    plt.close(fig)


def report(out):
    import xml.etree.ElementTree as ET
    selected = json.loads((out / "final_selection.json").read_text())["experiment"]
    table = pd.read_csv(out / "experiment_comparison.csv").set_index("experiment")
    base, best = table.loc["baseline"], table.loc[selected]
    reviews = pd.read_csv(out / "scientific_review.csv")
    quality = json.loads((out / "topic_quality_metrics.json").read_text())
    quality.update(reviewed_documents=len(reviews), review_counts=reviews.review_status.value_counts().to_dict(),
        review_sampling="Central/peripheral purposive samples and two scope checks; not a precision estimate",
        stability_seed42_vs43_ari=json.loads((out / "kmeans10_full_stability.json").read_text())["seed42_vs_seed43_ari"])
    write_json(out / "topic_quality_metrics.json", quality)
    config_path = out / "final_model_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["algorithm"] = config.get("clustering_algorithm", config["algorithm"])
    config["dimensionality_reduction"] = "none; original normalized 768-dimensional vectors"
    config["scientific_status"] = "Provisional development comparison; not a validated scientific winner"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    render_comparison(out, selected)
    shortlist = ["baseline", "baseline_repaired", "hdb_eom30_5", "hdb_leaf30_5", "hdb_leaf_n30_d10",
                 "hdb_leaf30_5_reduced10_repaired", "kmeans10_umap_repaired", selected]
    lines = ["| Experiment | Topics | C_v | Diversity@10 | Outliers | Top-2 share | C_v coverage |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for key in shortlist:
        row = table.loc[key]
        lines.append(f"| {key} | {int(row.topics)} | {row.c_v:.4f} | {row.keyword_diversity_at_10:.4f} | {row.outlier_ratio:.2%} | {row.top2_topic_share:.2%} | {int(row.coherence_eligible_topics)}/{int(row.coherence_total_topics)} |")
    tests = []
    for filename in ("tests.xml", "integration_tests.xml", "regression_tests.xml"):
        path = out / filename
        if path.exists():
            tree = ET.parse(path)
            suite = tree.getroot().find("testsuite")
            tests.append(f"- `{filename}`: {suite.get('tests')} tests, {suite.get('failures')} failures, {suite.get('errors')} errors, {suite.get('skipped')} skipped.")
    stable_full = json.loads((out / "kmeans10_full_stability.json").read_text())
    stable_umap = json.loads((out / "kmeans10_umap_stability.json").read_text())
    preservation = json.loads((out / "preservation_check.json").read_text())
    sizes = pd.read_csv(out / "topic_distribution.csv")
    counts = ", ".join(f"T{r.topic_id}={r.documents:,}" for r in sizes.itertuples())
    text = f"""# BatteryBERT topic optimization report

## Outcome and selection boundary

Code defects were repaired and 15 saved variants were compared on all 13,702 documents using the same validated pretrained BatteryBERT embeddings. The exported fixed-10 comparison is **{selected}**, using KMeans(n_clusters=10, random_state=42, n_init=10) directly on the original normalized 768-dimensional vectors, without UMAP or forced balancing. Representation uses its own original-document vocabulary, ngram_range=(1,2), corpus-document min_df=3, max_features=60000, aggregated-topic min_df=1, evidence-filtered stopwords and c-TF-IDF reduce_frequent_words=True. Phrase-placeholder recovery is applied only in the isolated topic-representation branch.

**No candidate establishes a satisfactory, independently validated set of ten scientific themes.** This is a provisional fixed-10 development comparison, not a scientific winner and not a production replacement. The natural HDBSCAN solutions (71 or 60 topics) reveal more specific themes but leave 59.56% or 57.85% of the corpus unassigned. Their separate 10-topic reduction does not recover these outliers. Both natural and reduced results remain available in experiments/.

## Initial diagnosis and changes

The baseline collapse began at HDBSCAN EOM, not at topic reduction: initial and final assignments are identical. Source association is extremely strong. Vector counts, dimensions, ID alignment, finite values, attention-masked mean pooling, normalization and chunk coverage pass the audited contracts. No BatteryBERT retraining or embedding regeneration was needed or executed.

Confirmed fixes:

- `src/model_comparison.py`: configured UMAP/HDBSCAN fields now drive constructors; reduction provenance records whether reduction actually occurred; dimension/shape/dtype validation raises explicit errors rather than relying on assertions.
- `src/batterybert_topic_tools.py`: scientific vectorizer preserves frozen chemical tokens and blocks cross-document n-grams; stopword evidence protects technical and context-sensitive terms; phrase-placeholder recovery fixes 716 tokens across 355 documents without editing frozen inputs; validation checks assignments, merge mappings, safe output adoption, bounded cosine sampling and exact phrase occurrence support.
- `scripts/audit_batterybert_topics.py`: isolated audit, staged representations, HDBSCAN/KMeans comparisons, no-refit keyword repair, same-reference coherence, real-document exports, seed checks, reload inference and preservation checks. Sequential joblib prevents the actual Windows WinError 5 encountered inside pynndescent's internal thread-pool path.
- `tests/test_batterybert_topic_audit.py` and `tests/test_batterybert_topic_audit_outputs.py`: numerical, semantic and real-artifact contracts.

## Controlled experimental comparison

{chr(10).join(lines)}

The complete table contains all 15 variants. C_v uses one shared repaired Stage 1 reference with exact observed phrases encoded atomically (U+2581 separator), window110, top10 and one process. U_Mass uses that same reference's BoW. Refer to coherence_protocol.json. Baseline C_v recomputed under this protocol is {base.c_v:.6f}; the old reported {json.loads((out / 'baseline_original_metrics.json').read_text())['c_v']:.6f} is retained separately. Scores from differing preprocessing are not substituted into this table.

Natural unrepaired HDBSCAN variants have partial top-10 coherence coverage because a leaked placeholder is absent from the repaired reference; eligible/total coverage is explicit and those aggregates are not treated as full-model scores. The repaired reduced HDBSCAN comparison has complete coverage. Coherence scores are computed on the development corpus; there is no independent scientific ground truth or held-out performance claim.

Representation-only changes reduce the declared generic keyword count from 23/100 to 5/100 without changing any document assignment. Ngram(1,3) obtains a high C_v but retains the 92.76% top-two collapse, illustrating why coherence alone cannot select the clustering. Frequent-word square-root weighting was already active in the baseline. Removing it reduced diversity to 0.65; it was retained. The blacklist metric includes context-sensitive inspected words deliberately retained for technical use, so it is a transparent lexical diagnostic, not a scientific interpretability score.

For the exported fixed-10 comparison, C_v changes from {base.c_v:.4f} to {best.c_v:.4f}, diversity from {base.keyword_diversity_at_10:.2f} to {best.keyword_diversity_at_10:.2f}, and the top-two share from {base.top2_topic_share:.2%} to {best.top2_topic_share:.2%}. These are trade-offs, not across-the-board improvements. Source Cramer's V remains {best.source_cramers_v:.6f}, versus {base.source_cramers_v:.6f} initially: source separation is **not solved**. Zero outliers is imposed by KMeans membership rules and is not evidence of superior quality.

Final topic counts: {counts}. They sum to 13,702; outliers are zero. Sizes were not equalized. Mean centroid cosine redundancy changes from {base.centroid_cosine_redundancy:.4f} to {best.centroid_cosine_redundancy:.4f}; high similarity and overlapping anode groups remain concerns.

Seed42/43 ARI is {stable_full['seed42_vs_seed43_ari']:.4f} for full-vector KMeans and {stable_umap['seed42_vs_seed43_ari']:.4f} for UMAP+KMeans (both UMAP and KMeans seeds changed for the latter). These are different robustness interventions, not strictly identical perturbations. Full-vector robustness is moderate. The UMAP version has stronger seed agreement and clearer P2/phosphate separation in examples, but retains a small formatting/style cluster and a broad anode cluster. Full-vector KMeans was exported as a compact alternative that avoids that 37-document style cluster and UMAP dependence; this does not justify claiming that it is scientifically best.

## Scientific inspection

Real title-abstract inspection covers {len(reviews)} purposively selected records across all ten exported topics, with central, peripheral and additional scope checks. `scientific_review.csv` records the agent's assessment, evidence, abstract availability and non-expert status. Labels are tentative and grounded in these documents. These samples are not a random relevance evaluation and their status counts must not be reported as precision.

- Well-matched example: `publication:WOS:000465409000001` describes FeS2 encapsulated in N-doped carbon nanosheets and actual sodium-storage anode tests.
- Potentially misplaced example: `patent:202319674A` is a flame-retardant electrolyte formulation assigned to the inorganic electrode-composition topic.
- Scope issue: `publication:WOS:000186422400023` studies superconducting transition temperature under pressure without battery performance context. It is flagged, not deleted.
- Counterexample to keyword-based exclusion: `patent:201969241M` explicitly claims an aqueous sodium-ion battery application for a sodium-aluminium-titanium phosphate even though its title says superconducting. The title word alone cannot justify exclusion.
- Generic hardware records may list sodium-ion alongside lithium and other chemistries. Abstracts can be absent (52 corpus records), and some patent polarity terminology is ambiguous. These limitations are preserved, not silently corrected or removed.

## Tests and preservation

{chr(10).join(tests)}

The selected saved model was reloaded and supplied explicit existing embeddings for 12 real document probes: all 12 predicted topics agree with saved assignments. Every representation-only branch preserves all memberships. Topic reduction preserves the -1 mask and maps each original cluster to a single new cluster. Chemical names, hyphens, charges, units and marker restoration have automated checks. {preservation['files_checked']} protected input/model/export/search files were SHA-256 checked: {preservation['status']}, {len(preservation['changed'])} changes. No Qdrant build/write or production Search change occurs in this run; live Qdrant files are not read for ownership checks or modified by this branch.

## Resource use, recovery and reproducibility

Experiments were sequential at the expensive clustering stage on CPU. Vectors are read-only memmaps (about42 MB); sampled similarity blocks are at most64x13702, never13702x13702. Sparse vectorizers cap vocabulary at60,000. Existing original UMAP coordinates are reused for controlled HDBSCAN/UMAP+KMeans comparisons. One additional UMAP(30 neighbours,10 components,cosine,seed42) and one seed43 robustness projection were fitted; no Cartesian search was run.

The initial new-UMAP attempt failed at Windows named-pipe/thread-pool creation; it was repaired using joblib's sequential backend and rerun successfully. This is a recovered execution failure, not an unrun experiment. No required experiment remains blocked by local resources. The original two-hour neural embedding computation was reused; no claim is made that pooled pretrained BERT is optimal for sentence similarity. The present evidence does not require BatteryBERT retraining. A future study could compare representations under independent scientific labels before authorizing expensive encoder changes.

Run the full audit/experiment/evaluation stages into a fresh directory:

```powershell
.\\.venv\\Scripts\\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_NEW --phase all
.\\.venv\\Scripts\\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_NEW --phase stability --candidate kmeans10_full
.\\.venv\\Scripts\\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_NEW --phase stability --candidate kmeans10_umap
```

Then inspect the actual representative documents and record that run's `topic_labels.csv` and `scientific_review.csv` before final export. Do not blindly copy labels to unrelated cluster IDs. For this frozen run the finalization command is:

```powershell
.\\.venv\\Scripts\\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_20261022_v1 --phase finalize --candidate kmeans10_full_repaired
.\\.venv\\Scripts\\python.exe -m pytest tests/test_batterybert_topic_audit.py tests/test_model_comparison_v2.py -q
.\\.venv\\Scripts\\python.exe -m pytest tests/test_batterybert_topic_audit_outputs.py -q
.\\.venv\\Scripts\\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_20261022_v1 --phase report
```

`bertopic_final.pickle` requires this project import path and the pinned local package versions; `transform` requires explicitly supplied embeddings from the same BatteryBERT checkpoint and pooling protocol. It does not auto-load another encoder. Every experiment has its own config, membership array, summaries, source table, keyword weights, representative documents, metrics and checkpoint. Vocabulary/stopword snapshots, input fingerprints and artifact hashes are retained. The web continues to use its previous topic artifacts; this audit does not change the production model choice.

## Deliverables

All files are under `{out.relative_to(ROOT).as_posix()}/`. Required exports: root_cause_audit.md, experiment_comparison.csv, topic_source_distribution.csv, topic_distribution.csv, topic_keywords_before_after.csv, topic_assignments.csv, representative_documents.csv, topic_quality_metrics.json, final_model_config.yaml, bertopic_final_10_topics.png and this optimization_report.md. Additional outputs include experiment_comparison.png, scientific_review.csv, topic_labels.csv, potentially_irrelevant_documents.csv, placeholder_repairs.csv, chemical_tokenization_audit.csv, the original/reduced mappings, seed checks, preserved baseline metrics and a final artifact manifest.
"""
    (out / "optimization_report.md").write_text(text, encoding="utf-8")
    # Hash all final outputs and implementation sources; exclude self and live files.
    files = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}
    sources = [ROOT / p for p in ("src/model_comparison.py", "src/batterybert_topic_tools.py", "scripts/audit_batterybert_topics.py", "tests/test_batterybert_topic_audit.py", "tests/test_batterybert_topic_audit_outputs.py", "docs/BATTERYBERT_TOPIC_AUDIT.md")]
    write_json(out / "artifact_manifest.json", dict(files_sha256=files,
        implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        training_invoked=False, embeddings_regenerated=False, production_promoted=False))
    log("Report and manifest complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["audit", "diagnose", "represent", "cluster", "repair", "evaluate", "stability", "finalize", "report", "all"], default="all")
    parser.add_argument("--candidate", default="kmeans10_full")
    args = parser.parse_args()
    target = (ROOT / args.output).resolve()
    if not target.is_relative_to(ROOT / "outputs"):
        raise ValueError("Experiment output must stay in outputs/")
    out = reserve_output(target)
    assert_run_inputs(out)
    if args.phase == "stability":
        stability(out, args.candidate)
    elif args.phase == "finalize":
        finalize(out, args.candidate)
    elif args.phase == "report":
        report(out)
    else:
        for phase in ("audit", "diagnose", "represent", "cluster", "repair", "evaluate"):
            if args.phase in ("all", phase):
                globals()[phase](out)


if __name__ == "__main__":
    main()
