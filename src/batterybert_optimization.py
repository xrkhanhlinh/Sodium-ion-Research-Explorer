"""Inductive BatteryBERT topic experiments without transformer imports.

Transforms, vocabulary, keywords and clustering are fitted on training groups.
Validation chooses a recipe; test documents are opened only by final evaluation.
Topic similarities are descriptive and are never presented as probabilities.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

from .batterybert_topic_tools import BOUNDARY, GENERIC, PROTECTED


class ContiguousScientificVectorizer(CountVectorizer):
    """Keep original adjacency: deleting a stopword must not invent a phrase."""
    def build_analyzer(self):
        stop = set(self.stop_words or []) - PROTECTED
        low, high = self.ngram_range
        def analyze(document):
            for segment in document.split(BOUNDARY):
                tokens = segment.split()
                for size in range(low, high + 1):
                    for i in range(len(tokens) - size + 1):
                        term = tokens[i:i + size]
                        if all(t not in stop for t in term):
                            yield ' '.join(term)
        return analyze


@dataclass
class Geometry:
    mode: str = 'raw'
    dimensions: int = 64
    def fit(self, vectors, sources):
        x = np.array(vectors, dtype=np.float64, copy=True)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite 2D embeddings')
        self.mean_ = x.mean(axis=0) if self.mode != 'raw' else np.zeros(x.shape[1])
        x -= self.mean_
        self.direction_ = np.zeros(x.shape[1])
        if 'source' in self.mode:
            sources = np.asarray(sources)
            values = np.unique(sources)
            if len(values) != 2:
                raise ValueError('Source projection requires exactly two training sources')
            direction = x[sources == values[0]].mean(0) - x[sources == values[1]].mean(0)
            self.direction_ = direction / max(np.linalg.norm(direction), 1e-12)
            x -= (x @ self.direction_)[:, None] * self.direction_
        self.basis_, self.scale_, self.remove_ = None, None, None
        if 'whiten' in self.mode or self.mode == 'abtt3':
            values, basis = np.linalg.eigh(x.T @ x / max(1, len(x) - 1))
            order = np.argsort(values)[::-1]
            values, basis = np.maximum(values[order], 0), basis[:, order]
            if self.mode == 'abtt3':
                self.remove_ = basis[:, :3]
            else:
                n = min(self.dimensions, x.shape[1], len(x) - 1)
                self.basis_ = basis[:, :n]
                self.scale_ = np.sqrt(np.maximum(values[:n], max(values[0] * .01, 1e-12)))
        return self

    def transform(self, vectors):
        x = np.asarray(vectors, dtype=np.float64) - self.mean_
        x -= (x @ self.direction_)[:, None] * self.direction_
        if self.remove_ is not None:
            x -= (x @ self.remove_) @ self.remove_.T
        if self.basis_ is not None:
            x = (x @ self.basis_) / self.scale_
        return normalize(x).astype(np.float32)


def keyword_model(docs, labels, topn=10):
    # Standalone generic words are unsuitable names; technical terms are kept.
    stop = sorted((set(ENGLISH_STOP_WORDS) | set(GENERIC)) - PROTECTED)
    vectorizer = ContiguousScientificVectorizer(lowercase=False, ngram_range=(1, 2),
        min_df=5, max_df=.60, max_features=40000, stop_words=stop)
    counts = vectorizer.fit_transform(docs).tocsr()
    binary = counts.copy(); binary.data[:] = 1
    labels = np.asarray(labels)
    topic_ids = sorted(set(labels) - {-1})
    membership = sparse.csr_matrix((np.ones(len(labels)),
        (labels, np.arange(len(labels)))), shape=(max(topic_ids) + 1, len(labels)))
    tf = (membership @ counts).toarray()
    topic_df = (membership @ binary).toarray()
    corpus_df = np.asarray(binary.sum(0)).ravel()
    # c-TF-IDF style sqrt frequency, IDF and class exclusivity.
    tf /= np.maximum(tf.sum(1, keepdims=True), 1)
    exclusivity = (topic_df + 1) / (corpus_df[None, :] + len(topic_ids))
    weights = np.sqrt(tf) * np.log1p(len(docs) / np.maximum(corpus_df, 1))[None, :] * np.sqrt(exclusivity)
    words = vectorizer.get_feature_names_out()
    rows = []
    for topic in topic_ids:
        chosen = []
        for j in np.argsort(-weights[topic], kind='stable'):
            if topic_df[topic, j] < 5 or weights[topic, j] <= 0:
                continue
            tokens = set(words[j].split())
            # Avoid filling the list with nested copies of a phrase.
            if any(tokens <= old or old <= tokens for old in chosen):
                continue
            chosen.append(tokens)
            rows.append(dict(topic_id=int(topic), rank=len(chosen), token=words[j],
                weight=float(weights[topic, j]), train_document_support=int(topic_df[topic, j])))
            if len(chosen) == topn:
                break
        if len(chosen) != topn:
            raise ValueError(f'Topic {topic} has fewer than {topn} supported distinct keywords')
    return vectorizer, rows


def document_npmi(docs, keywords):
    """Macro topic document-level NPMI. Missing co-occurrence scores -1.

    This is an unsupervised lexical diagnostic, not scientific ground truth.
    Exact contiguous phrases are required on the evaluation documents.
    """
    words = sorted(set(keywords.token))
    vectorizer = ContiguousScientificVectorizer(vocabulary={w: i for i, w in enumerate(words)},
        ngram_range=(1, 2), lowercase=False, binary=True, dtype=np.int32)
    binary = vectorizer.transform(docs).tocsr()
    df = np.asarray(binary.sum(0)).ravel()
    result, supported = [], 0
    for _, group in keywords.groupby('topic_id', sort=True):
        indices = [vectorizer.vocabulary[w] for w in group.sort_values('rank').token]
        co = (binary[:, indices].T @ binary[:, indices]).toarray()
        scores = []
        for i in range(len(indices)):
            supported += int(df[indices[i]] > 0)
            for j in range(i):
                if co[i, j] == 0:
                    scores.append(-1.)
                else:
                    p = min(float(co[i, j]) / len(docs), 1 - 1e-12)
                    scores.append(float(np.log(p / ((df[indices[i]] / len(docs)) *
                        (df[indices[j]] / len(docs)))) / -np.log(p)))
        result.append(float(np.mean(scores)))
    return dict(document_npmi=float(np.mean(result)), per_topic_npmi=result,
                supported_keyword_fraction=supported / len(keywords))


@dataclass
class TopicPipeline:
    geometry: Geometry
    cluster: KMeans
    encoder: str
    ordered_training_ids: list

    def predict(self, vectors):
        return self.cluster.predict(self.geometry.transform(vectors)).astype(np.int32)
