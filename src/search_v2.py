"""Versioned full-corpus retrieval; isolated caches, no training or Qdrant writes.

Chemical lexical channels complement an uncased neural encoder. A public frozen
cross-encoder reranks a bounded candidate union using query-aware passages.
Scores and lexical support checks are never presented as verified relevance.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
from functools import lru_cache
import hashlib
import html
import inspect
from importlib.metadata import version
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import pickle
import re
import time
import unicodedata

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs/search_v2.json'
CACHE = ROOT / 'data/search_v2'
METHODS = ['BM25_CHEMICAL', 'DENSE_PRETRAINED', 'DENSE_FINETUNED', 'DENSE_CHUNKED',
           'HYBRID_PRETRAINED', 'HYBRID_FINETUNED', 'HYBRID_CHUNKED',
           'RERANK_PRETRAINED', 'RERANK_FINETUNED', 'RERANK_CHUNKED']
STEM = SnowballStemmer('english')
WORD = re.compile(r'[A-Za-z][A-Za-z0-9]*(?:\([A-Za-z0-9]+\)[0-9]*)*(?:[+-])?')
SALT_ALIASES = {
    'NaFSI': r'\bsodium\s+bis\s*\(?\s*fluorosulfonyl\s*\)?\s*imide\b|\bNaFSI\b',
    'NaTFSI': r'\bsodium\s+bis\s*\(?\s*trifluoromethanesulfonyl\s*\)?\s*imide\b|\bNaTFSI\b',
}


@lru_cache(maxsize=100000)
def stem_word(word):
    return STEM.stem(word)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def normalize(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ''
    text = unicodedata.normalize('NFKC', html.unescape(str(value)))
    text = re.sub(r'[\u2010-\u2015\u2212]', '-', text)
    return re.sub(r'\s+', ' ', text).strip()


def chemical_like(word):
    # Lexical heuristic, not a chemistry parser or formula equivalence solver.
    return bool(re.search(r'[0-9()]', word) or
                (re.fullmatch(r'(?:[A-Z][a-z]?){2,}', word) and len(word) <= 16))


def lexical_tokens(text):
    text = normalize(text)
    # Repair explicit source typography Na-2FeP2O7 / Na3V2(PO4)(3), only in
    # the lexical view. Keep original cleaned Title/Abstract untouched.
    text = re.sub(r'(?<=[A-Za-z])-(?=\d)', '', text)
    text = re.sub(r'\((\d+)\)', r'\1', text)
    tokens = []
    for word in WORD.findall(text):
        word = word.rstrip('+-')
        if len(word) < 2:
            continue
        if chemical_like(word):
            tokens.append('chem:' + word)
        else:
            low = word.casefold()
            if low not in ENGLISH_STOP_WORDS:
                tokens.append(stem_word(low))
    for canonical, pattern in SALT_ALIASES.items():
        if re.search(pattern, text, re.I) and 'chem:' + canonical not in tokens:
            tokens.append('chem:' + canonical)
    return tokens


def lexical_fingerprint():
    """Index identity follows tokenization, not unrelated search/UI edits."""
    identity = dict(
        schema=1,
        functions=[inspect.getsource(f) for f in
                   (normalize, chemical_like, stem_word, lexical_tokens)],
        word_pattern=WORD.pattern, word_flags=WORD.flags, salt_aliases=SALT_ALIASES,
        stopwords=sorted(ENGLISH_STOP_WORDS),
        stemmer=type(getattr(STEM.stem, '__self__', STEM)).__qualname__,
        stemmer_stopwords=sorted(STEM.stopwords),
        unicode_version=unicodedata.unidata_version,
        dependencies={name: version(name) for name in ('nltk', 'scikit-learn', 'rank-bm25')},
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def _integer_at_least(value, minimum, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def validate_config(config):
    """Reject invalid numeric domains before loading corpus or neural models."""
    for name in ('candidate_depth', 'rerank_depth', 'passage_tokens',
                 'passages_per_document', 'reranker_max_length',
                 'reranker_batch_size', 'query_max_tokens', 'threads'):
        _integer_at_least(config[name], 1, name)
    _integer_at_least(config['passage_overlap'], 0, 'passage_overlap')
    if config['passage_overlap'] >= config['passage_tokens']:
        raise ValueError('passage_overlap must be less than passage_tokens')
    for name in ('rrf_k', 'title_weight', 'bm25_k1', 'bm25_b'):
        _finite_nonnegative(config[name], name)
    if config['bm25_k1'] == 0 or config['bm25_b'] > 1:
        raise ValueError('BM25 requires k1 > 0 and 0 <= b <= 1')


def reciprocal_rank_fusion(rankings, k=60):
    """Fuse unique document IDs per list; scores do not affect rank weights."""
    _finite_nonnegative(k, 'k')
    scores = Counter()
    for ranking in rankings:
        seen = set()
        for rank, (doc, _) in enumerate(ranking, 1):
            if doc in seen:
                raise ValueError('Duplicate document ID in one RRF input ranking')
            seen.add(doc)
            scores[doc] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def ranked_scores(scores, ids, allowed, limit, positive_only=False):
    candidates = np.flatnonzero(allowed & np.isfinite(scores))
    if positive_only:
        candidates = candidates[scores[candidates] > 0]
    if isinstance(limit, (int, np.integer)) and 0 < limit < len(candidates):
        # Match the baseline Python-float sort key before selecting a cutoff.
        with np.errstate(over='ignore'):
            values = np.asarray(scores[candidates], dtype=np.float64)
        cutoff = np.partition(values, len(values) - limit)[len(values) - limit]
        # Keep every cutoff tie so the original score/ID sort decides membership.
        candidates = candidates[values >= cutoff]
    order = sorted(candidates, key=lambda i: (-float(scores[i]), ids[i]))[:limit]
    return [(ids[i], float(scores[i])) for i in order]


def select_passages(title, abstract, query, tokenizer, width=320, overlap=64, maximum=2,
                    max_length=512):
    """Consider every abstract token; select lexical-overlap windows, not qrels.

    Offsets preserve chemical text. Selection is a lossy, query-aware view for
    reranking; full lexical indexing still covers the entire original abstract.
    """
    for value, minimum, name in ((width, 1, 'width'), (overlap, 0, 'overlap'),
                                  (maximum, 1, 'maximum'), (max_length, 1, 'max_length')):
        _integer_at_least(value, minimum, name)
    if width <= overlap:
        raise ValueError('Invalid passage window settings')
    title, abstract = normalize(title), normalize(abstract)
    title_offsets = tokenizer(title, add_special_tokens=False, truncation=False,
                              return_offsets_mapping=True, verbose=False)['offset_mapping']
    if len(title_offsets) > 80:
        title = title[:title_offsets[79][1]]
    query_length = len(tokenizer(query, add_special_tokens=False, truncation=False)['input_ids'])
    if query_length > 128:
        raise ValueError('Reranker query exceeds 128 tokens; shorten the query')
    # The production WordPiece tokenizer uses 3 pair special tokens. Ask other
    # tokenizers for their own pair overhead; newline adds no WordPiece token.
    special_tokens = (tokenizer.num_special_tokens_to_add(pair=True)
                      if hasattr(tokenizer, 'num_special_tokens_to_add') else 3)
    _integer_at_least(special_tokens, 0, 'pair special token count')
    remaining_budget = max_length - special_tokens - query_length - min(80, len(title_offsets))
    if remaining_budget < 0:
        raise ValueError('Title and query exceed reranker token budget')
    if not abstract:
        return [dict(text=title, start=0, end=0, token_start=0, token_end=0)]
    encoded = tokenizer(abstract, add_special_tokens=False, truncation=False, return_offsets_mapping=True, verbose=False)
    offsets = encoded['offset_mapping']
    if not offsets:
        return [dict(text=title, start=0, end=0, token_start=0, token_end=0)]
    width = min(width, remaining_budget)
    if width <= overlap:
        raise ValueError('Effective passage width must exceed overlap within reranker token budget')
    qterms = set(lexical_tokens(query))
    windows = []
    start = 0
    while start < len(offsets):
        end = min(start + width, len(offsets))
        left, right = int(offsets[start][0]), int(offsets[end - 1][1])
        passage = abstract[left:right]
        terms = set(lexical_tokens(passage))
        # Exact chemical channels count as ordinary terms; no query-ID rules.
        hit = len(qterms & terms)
        windows.append((hit, start, dict(text=title + '\n' + passage,
                        start=left, end=right, token_start=start, token_end=end)))
        if end == len(offsets):
            break
        start = end - overlap
    windows.sort(key=lambda value: (-value[0], value[1]))
    return [value[2] for value in windows[:maximum]]


class SearchEngineV2:
    def __init__(self, config_path=CONFIG_PATH, use_query_cache=False):
        self.config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        validate_config(self.config)
        self.meta = pd.read_parquet(ROOT / self.config['corpus'])
        if len(self.meta) != self.config['expected_documents'] or not self.meta.doc_id.is_unique:
            raise ValueError('Expected exactly the cleaned 13,702-document corpus')
        self.ids = self.meta.doc_id.tolist()
        self.positions = {doc: i for i, doc in enumerate(self.ids)}
        self.meta = self.meta.set_index('doc_id', drop=False)
        self.encoder = self.encoder_kind = self.reranker = None
        self.vectors = {}
        self._score_cache = {}
        self._pair_cache = {}
        self.use_query_cache = use_query_cache
        self._active_query = None
        self._request_work = None
        self.prefix = json.loads((ROOT / 'outputs/reports/stage3/inference_contract.json').read_text(encoding='utf-8'))['query_prefix']
        self._load_lexical()
        self._load_query_cache()

    def _load_lexical(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        identity = dict(corpus_sha256=digest(ROOT / self.config['corpus']),
                        tokenizer_source_sha256=lexical_fingerprint(),
                        k1=self.config['bm25_k1'], b=self.config['bm25_b'])
        marker, path = CACHE / 'lexical_manifest.json', CACHE / 'lexical.pkl'
        if marker.exists():
            state = json.loads(marker.read_text(encoding='utf-8'))
            if state['identity'] == identity and path.exists() and digest(path) == state['sha256']:
                with path.open('rb') as stream:
                    self.title_index, self.abstract_index = pickle.load(stream)
                return
        title_tokens = [lexical_tokens(t) or ['__empty_title__'] for t in self.meta.title]
        abstract_tokens = [lexical_tokens(a) or ['__empty_abstract__'] for a in self.meta.abstract]
        # Keep BM25Okapi for interpretability; clip negative field scores.
        self.title_index = BM25Okapi(title_tokens, k1=identity['k1'], b=identity['b'])
        self.abstract_index = BM25Okapi(abstract_tokens, k1=identity['k1'], b=identity['b'])
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            pickle.dump((self.title_index, self.abstract_index), stream, protocol=4)
        os.replace(tmp, path)
        write_json(marker, dict(identity=identity, sha256=digest(path), documents=len(self.ids),
                              corpus_changed=False, note='Fresh lexical view from full cleaned Title/Abstract, no Stage1 token repair or vocabulary mutation.'))

    def _load_query_cache(self):
        self.cached_queries = {}
        self.query_vectors = {}
        if not self.use_query_cache:
            return
        queries = pd.read_parquet(ROOT / 'data/stage4/queries.parquet')
        self.cached_queries = {normalize(q): i for i, q in enumerate(queries['query'])}

    @contextmanager
    def reuse_query_work(self):
        """Share retrieval work within one serial request, never across requests."""
        previous = self._request_work
        if previous is None:
            self._request_work = {}
        try:
            yield
        finally:
            self._request_work = previous

    def _cached_vectors(self, kind):
        if kind not in self.query_vectors:
            path = ROOT / f'data/stage4/{kind}_query_embeddings.npy'
            marker = ROOT / f'outputs/reports/stage4/{kind}_query_embedding_manifest.json'
            state = json.loads(marker.read_text(encoding='utf-8'))
            folder = 'models/bge_pretrained' if kind == 'pretrained' else 'models/bert_finetuned'
            if (state['identity']['model_sha256'] != digest(ROOT / folder / 'model.safetensors') or
                    state['identity']['query_sha256'] != digest(ROOT / 'data/stage4/queries.parquet') or
                    state['embedding_sha256'] != digest(path)):
                raise ValueError('Cached query embedding provenance mismatch')
            self.query_vectors[kind] = np.load(path, mmap_mode='r')
        return self.query_vectors[kind]

    def _matrix(self, kind):
        if kind not in self.vectors:
            if kind == 'pretrained':
                path = ROOT / 'data/stage4/pretrained_embeddings.npy'
                order = pd.read_parquet(ROOT / 'data/stage4/pretrained_embedding_ids.parquet').doc_id.tolist()
                state = json.loads((ROOT / 'outputs/reports/stage4/pretrained_embedding_manifest.json').read_text(encoding='utf-8'))
                expected = state['sha256']
            elif kind == 'finetuned':
                path = ROOT / 'data/stage3/embeddings.npy'
                order = pd.read_parquet(ROOT / 'data/stage3/embeddings.parquet', columns=['doc_id']).doc_id.tolist()
                expected = json.loads((ROOT / 'outputs/reports/stage3/embedding_manifest.json').read_text(encoding='utf-8'))['numpy_sha256']
            else:
                path = ROOT / 'outputs/bertopic_chunked/document_embeddings.npy'
                order = json.loads((ROOT / 'outputs/bertopic_chunked/ordered_doc_ids.json').read_text(encoding='utf-8'))
                expected = json.loads((ROOT / 'outputs/bertopic_chunked/embedding_success.json').read_text(encoding='utf-8')).get('document_embeddings_sha256')
            if order != self.ids:
                raise ValueError(f'{kind} embedding document order mismatch')
            if expected is not None and digest(path) != expected:
                raise ValueError(f'{kind} embedding SHA mismatch')
            matrix = np.load(path, mmap_mode='r')
            if matrix.shape != (len(self.ids), 768) or not np.isfinite(matrix).all():
                raise ValueError('Invalid document matrix')
            if not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-5):
                raise ValueError('Document embeddings must be normalized')
            self.vectors[kind] = matrix
        return self.vectors[kind]

    def encode(self, query, kind):
        kind = 'finetuned' if kind == 'chunked' else kind
        query = normalize(query)
        work = getattr(self, '_request_work', None)
        key = ('encoding', kind, query)
        if work is not None and key in work:
            return work[key]
        if self.use_query_cache and query in self.cached_queries:
            vector = self._cached_vectors(kind)[self.cached_queries[query]]
            if work is not None:
                work[key] = vector
            return vector
        if self.encoder_kind != kind:
            from sentence_transformers import SentenceTransformer
            import torch
            self.encoder = None
            gc.collect()
            torch.set_num_threads(self.config['threads'])
            folder = 'models/bge_pretrained' if kind == 'pretrained' else 'models/bert_finetuned'
            self.encoder = SentenceTransformer(str(ROOT / folder), device='cpu', local_files_only=True)
            self.encoder.max_seq_length = self.config['query_max_tokens']
            self.encoder_kind = kind
        if len(self.encoder.tokenizer(self.prefix + query, truncation=False)['input_ids']) > self.config['query_max_tokens']:
            raise ValueError('Dense query exceeds 128 tokens; shorten the query')
        vector = self.encoder.encode([self.prefix + query], normalize_embeddings=True, show_progress_bar=False)[0]
        if work is not None:
            work[key] = vector
        return vector

    def allowed(self, source='All', years=None):
        if source not in ['All', 'Patent', 'Publication']:
            raise ValueError('Invalid source')
        mask = np.ones(len(self.ids), dtype=bool)
        if source != 'All':
            mask &= self.meta.source.eq(source).to_numpy()
        if years is not None:
            lo, hi = map(int, years)
            if lo > hi:
                raise ValueError('Invalid year interval')
            mask &= self.meta.year.between(lo, hi).fillna(False).to_numpy(dtype=bool)
        return mask

    def lexical(self, query, mask):
        work = getattr(self, '_request_work', None)
        key = ('lexical_scores', normalize(query))
        if work is not None and key in work:
            scores = work[key]
        else:
            tokens = lexical_tokens(query)
            if not tokens or not any(t in self.title_index.idf or t in self.abstract_index.idf for t in tokens):
                return []
            scores = (self.config['title_weight'] * np.maximum(self.title_index.get_scores(tokens), 0) +
                      np.maximum(self.abstract_index.get_scores(tokens), 0))
            if work is not None:
                work[key] = scores
        return ranked_scores(scores, self.ids, mask, self.config['candidate_depth'], positive_only=True)

    def dense(self, query, kind, mask):
        work = getattr(self, '_request_work', None)
        key = ('dense_scores', kind, normalize(query))
        if work is not None and key in work:
            scores = work[key]
        else:
            vector = self.encode(query, kind)
            with threadpool_limits(limits=self.config['threads']):
                scores = self._matrix(kind) @ vector
            if work is not None:
                work[key] = scores
        return ranked_scores(scores, self.ids, mask, self.config['candidate_depth'])

    def _load_reranker(self):
        if self.reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            torch.set_num_threads(self.config['threads'])
            self.reranker = CrossEncoder(str(ROOT / self.config['reranker']), device='cpu',
                                        local_files_only=True, trust_remote_code=False,
                                        max_length=self.config['reranker_max_length'])

    def rerank(self, query, candidates):
        self._load_reranker()
        rows, pairs = [], []
        for doc, _ in candidates[:self.config['rerank_depth']]:
            key = (query, doc)
            if key in self._score_cache:
                continue
            r = self.meta.loc[doc]
            passages = select_passages(r.title, r.abstract, query, self.reranker.tokenizer,
                                      self.config['passage_tokens'], self.config['passage_overlap'],
                                      self.config['passages_per_document'],
                                      max_length=self.config['reranker_max_length'])
            for passage in passages:
                rows.append((doc, passage))
                pairs.append((query, passage['text']))
        if pairs:
            scores = self.reranker.predict(pairs, batch_size=self.config['reranker_batch_size'],
                                          show_progress_bar=False, convert_to_numpy=True)
            scores = np.asarray(scores).reshape(-1)
            if len(scores) != len(rows) or not np.isfinite(scores).all():
                raise ValueError('Invalid cross-encoder outputs')
            for (doc, passage), score in zip(rows, scores):
                key = (query, doc)
                if key not in self._score_cache or score > self._score_cache[key]:
                    self._score_cache[key] = float(score)
                    self._pair_cache[key] = passage
        ranked = [(doc, self._score_cache[(query, doc)]) for doc, _ in candidates[:self.config['rerank_depth']]]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))

    def rank(self, query, method, source='All', years=None):
        query = normalize(query)
        if not query or len(query) > 10000 or not re.search(r'[A-Za-z0-9]', query):
            raise ValueError('Enter a nonempty English query of at most 10,000 characters')
        if method not in METHODS:
            raise ValueError('Invalid v2 method')
        mask = self.allowed(source, years)
        if not mask.any():
            return [], dict(status='NO_DOCUMENTS_AFTER_FILTER', eligible_documents=0)
        eligible = int(mask.sum())
        if method == 'BM25_CHEMICAL':
            ranks = self.lexical(query, mask)
        elif method.startswith('DENSE_'):
            ranks = self.dense(query, method.removeprefix('DENSE_').lower(), mask)
        else:
            lexical = self.lexical(query, mask)
            # Conservative support gate, not calibrated OOD/relevance detection.
            if not lexical:
                return [], dict(status='NO_LEXICAL_SUPPORT', eligible_documents=eligible,
                                limitation='May abstain on valid paraphrases with no indexed token overlap')
            kind = method.split('_', 1)[1].lower()
            dense = self.dense(query, kind, mask)
            ranks = reciprocal_rank_fusion([lexical, dense], self.config['rrf_k'])
            if method.startswith('RERANK_'):
                ranks = self.rerank(query, ranks)
        return ranks, dict(status='OK' if ranks else 'NO_LEXICAL_SUPPORT', eligible_documents=eligible)

    def search(self, query, method=None, source='All', years=None, top_k=5):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError('top_k must be an integer 1..100')
        method = method or self.config['default_method']
        start = time.perf_counter()
        query = normalize(query)
        if not self.use_query_cache or self._active_query != query:
            self.clear_scores()
        self._active_query = query
        ranks, details = self.rank(query, method, source, years)
        results = []
        for rank, (doc, score) in enumerate(ranks[:top_k], 1):
            r = self.meta.loc[doc]
            passage = self._pair_cache.get((query, doc)) if method.startswith('RERANK_') else None
            results.append(dict(rank=rank, doc_id=doc, title=r.title, abstract=r.abstract,
                                source=r.source, year=None if pd.isna(r.year) else int(r.year),
                                identifier=str(r.identifier), score=score,
                                evidence_passage=passage['text'] if passage else '',
                                passage_offset_field='normalized_abstract' if passage else None,
                                passage_char_start=passage['start'] if passage else None,
                                passage_char_end=passage['end'] if passage else None))
        return dict(query=query, method=method, results=results, details=details,
                    seconds=time.perf_counter() - start, source_filter=source, year_filter=years,
                    eligible_documents=details['eligible_documents'],
                    quality_status='PENDING_INDEPENDENT_LABELS',
                    benchmark_query_cache=self.use_query_cache,
                    score_meaning='Method-specific ranking score; not a relevance probability',
                    uses_project_finetune=method.endswith(('FINETUNED', 'CHUNKED')),
                    version='search_v2')

    def clear_scores(self):
        self._score_cache.clear()
        self._pair_cache.clear()

    def close(self):
        self.encoder = self.reranker = None
        self._request_work = None
        self.vectors.clear()
        self.clear_scores()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query')
    parser.add_argument('--method', choices=METHODS, default='RERANK_PRETRAINED')
    parser.add_argument('--source', choices=['All', 'Patent', 'Publication'], default='All')
    parser.add_argument('--output')
    args = parser.parse_args()
    engine = SearchEngineV2()
    try:
        result = engine.search(args.query, args.method, args.source)
        if args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
