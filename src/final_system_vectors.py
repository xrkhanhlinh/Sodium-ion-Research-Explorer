"""Persistent Qdrant Local collections, exact retrieval and mapping validation."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from qdrant_client import QdrantClient, models
from .final_system_artifacts import ROOT, OUT, corpus, vector_spec, read_json, sha
from .model_comparison_io import write_json


def collection_name(spec):
    return f"sodium_{spec['kind']}_v1_{spec['weights_sha256'][:8]}_{spec['embeddings_sha256'][:8]}"


def metadata_filter(source='All', years=None, doc_ids=None):
    if source not in ('All', 'Patent', 'Publication'):
        raise ValueError('Invalid source')
    must = []
    if source != 'All':
        must.append(models.FieldCondition(key='source', match=models.MatchValue(value=source)))
    if years is not None:
        lo, hi = map(int, years)
        if lo > hi:
            raise ValueError('Invalid year interval')
        must.append(models.FieldCondition(key='year', range=models.Range(gte=lo, lte=hi)))
    if doc_ids is not None:
        # Used only for explicit chemical/required-term constraints. Their
        # matching semantics are exactly the existing Search V2 lexical view.
        must.append(models.FieldCondition(key='doc_id', match=models.MatchAny(any=list(doc_ids))))
    return models.Filter(must=must) if must else None


class VectorDatabase:
    def __init__(self, path=None, artifact_dir=None, topics=None):
        self.path = Path(path or OUT / 'qdrant')
        self.artifact_dir = Path(artifact_dir or OUT)
        self.registry_path = self.artifact_dir / 'qdrant_collections.json'
        self.topics = topics
        self.client = QdrantClient(path=str(self.path), force_disable_check_same_thread=True)
        self.collections = read_json(self.registry_path) if self.registry_path.exists() else {}

    def close(self):
        self.client.close()

    def build(self, kind):
        spec = vector_spec(kind)
        name = collection_name(spec)
        meta = corpus()
        vectors = np.load(spec['matrix'], mmap_mode='r')
        if not self.client.collection_exists(name):
            self.client.create_collection(name, vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
        marker = self.artifact_dir / f'{name}_upsert.json'
        identity = dict(corpus_sha256=sha(ROOT / 'data/processed/main_corpus.parquet'),
            vector_sha256=spec['embeddings_sha256'], weights_sha256=spec['weights_sha256'], collection=name)
        if self.topics is not None:
            if self.topics.doc_id.tolist() != meta.doc_id.tolist():
                raise ValueError('Qdrant topic metadata document order mismatch')
            from .stage3_common import signature
            identity['topic_metadata_sha256'] = signature(self.topics[['doc_id','topic_id','topic_label']].to_dict('records'))
        state = read_json(marker) if marker.exists() else dict(identity=identity, completed=0)
        if state['identity'] != identity:
            raise ValueError('Collection provenance mismatch')
        if state['completed']==len(meta) and kind in self.collections:
            if (any(self.collections[kind].get(k)!=v for k,v in identity.items())
                    or self.client.count(name,exact=True).count!=len(meta)):
                raise ValueError('Completed collection metadata/count mismatch')
            # Keep the original measured build duration when reusing a
            # completed collection; the caller profiles this reuse separately.
            return name
        tick = time.perf_counter()
        for start in range(state['completed'], len(meta), 64):
            frame = meta.iloc[start:start + 64]
            payloads = [dict(doc_id=row.doc_id, source=row.source,
                year=None if pd.isna(row.year) else int(row.year), title=row.title,
                identifier=str(row.identifier), corpus_row=start + i, encoder=kind)
                for i, row in enumerate(frame.itertuples())]
            if self.topics is not None:
                for i, payload in enumerate(payloads):
                    topic = self.topics.iloc[start+i]
                    payload.update(abstract='' if pd.isna(frame.iloc[i].abstract) else str(frame.iloc[i].abstract),
                                   topic_id=int(topic.topic_id), topic_label=str(topic.topic_label))
            self.client.upsert(name, points=models.Batch(ids=list(range(start, start + len(frame))),
                vectors=np.asarray(vectors[start:start + len(frame)]).tolist(), payloads=payloads), wait=True)
            state['completed'] = start + len(frame)
            write_json(marker, state)
            if start % 2048 == 0:
                print(f'QDRANT {kind} {state["completed"]}/13702', flush=True)
        if self.client.count(name, exact=True).count != len(meta):
            raise ValueError('Qdrant document count mismatch')
        self.collections[kind] = dict(name=name, **identity, documents=len(meta), dimension=768,
            distance='Cosine', mode='Local persistent exact (no HNSW)', seconds=time.perf_counter() - tick,
            vector_path=str(spec['matrix'].relative_to(ROOT)), model_path=str(spec['model'].relative_to(ROOT)))
        write_json(self.registry_path, self.collections)
        return name

    def search(self, kind, vector, top_k=100, source='All', years=None, doc_ids=None):
        if kind not in self.collections:
            raise FileNotFoundError(f'Qdrant collection not built: {kind}; run run_final_system.py qdrant')
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (768,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=2e-5):
            raise ValueError('Query vector must be finite, 768-dimensional and L2-normalized')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError('top_k must be 1..100')
        if doc_ids is not None and len(doc_ids) == 0:
            return []
        query_filter=metadata_filter(source,years)
        if doc_ids is not None:
            # Collections built here use corpus row numbers as point IDs. The
            # caller's allowed doc IDs plus source/year can therefore become a
            # single ID condition. Qdrant Local otherwise repeatedly parses
            # payload JSON paths and scans a MatchAny list for every point.
            # Keep filtering before ranking and check the mapping provenance.
            if not hasattr(self,'_filter_metadata'):
                self._filter_metadata=corpus().reset_index(drop=True)
                self._filter_corpus_sha256=sha(ROOT/'data/processed/main_corpus.parquet')
            if self.collections[kind].get('corpus_sha256')!=self._filter_corpus_sha256:
                raise ValueError('Cannot translate document IDs: collection corpus differs')
            meta=self._filter_metadata
            if self.collections[kind].get('documents')!=len(meta):
                raise ValueError('Cannot translate document IDs: collection size differs')
            allowed=meta.doc_id.isin(set(doc_ids)).to_numpy()
            if source!='All':allowed&=meta.source.eq(source).to_numpy()
            if years is not None:
                lo,hi=map(int,years)
                allowed&=meta.year.between(lo,hi).fillna(False).to_numpy(dtype=bool)
            rows=np.flatnonzero(allowed).tolist()
            if not rows:return []
            query_filter=models.Filter(must=[models.HasIdCondition(has_id=rows)])
        with threadpool_limits(limits=4):
            hits = self.client.query_points(self.collections[kind]['name'], query=vector.tolist(),
                query_filter=query_filter, limit=top_k, with_payload=True,
                with_vectors=False, search_params=models.SearchParams(exact=True)).points
        return [(hit.payload['doc_id'], float(hit.score)) for hit in hits]

    def verify(self):
        """Check EVERY payload/vector and sampled Top100 exact retrieval incl filters.

        Qdrant and NumPy can order float32 cutoff ties differently. Report raw
        overlap plus a score/tie-aware exactness check, never silently relabel.
        """
        rows = []
        mapping = []
        meta = corpus()
        ids = meta.doc_id.tolist()
        for kind, info in self.collections.items():
            spec = vector_spec(kind)
            if info['name'] != collection_name(spec):
                raise ValueError('Collection encoder/vector identity mismatch')
            matrix = np.load(spec['matrix'], mmap_mode='r')
            name = info['name']
            parameters=self.client.get_collection(name).config.params.vectors
            assert parameters.size==768 and parameters.distance==models.Distance.COSINE
            seen = set()
            max_error = 0.
            offset = None
            while True:
                records, offset = self.client.scroll(name, offset=offset, limit=128, with_payload=True, with_vectors=True)
                for record in records:
                    i = int(record.id)
                    if i in seen or not 0 <= i < len(ids):
                        raise ValueError('Duplicate or invalid Qdrant point ID')
                    seen.add(i)
                    p = record.payload
                    r = meta.iloc[i]
                    assert p['doc_id'] == ids[i] and p['corpus_row'] == i and p['source'] == r.source
                    assert p['year'] == (None if pd.isna(r.year) else int(r.year))
                    assert p['title'] == r.title and p['encoder'] == kind
                    assert p['identifier'] == str(r.identifier)
                    if self.topics is not None:
                        assert p['abstract'] == ('' if pd.isna(r.abstract) else str(r.abstract))
                        assert p['topic_id'] == int(self.topics.iloc[i].topic_id)
                        assert p['topic_label'] == str(self.topics.iloc[i].topic_label)
                    err = float(np.max(np.abs(np.asarray(record.vector) - matrix[i])))
                    max_error = max(max_error, err)
                    assert err < 2e-6, 'Stored vector differs from source'
                if offset is None:
                    break
            assert len(seen) == len(meta)
            mapping.append(dict(encoder=kind, points=len(seen), payload_mapping='PASS', max_vector_error=max_error))
            rng = np.random.default_rng(42)
            probes = [(f'doc_{i}', np.asarray(matrix[i])) for i in (0, 100, 1000, 5000, 10000, 13701)]
            for i in range(3):
                q = rng.normal(size=768).astype('float32'); q /= np.linalg.norm(q)
                probes.append((f'random_{i}', q))
            for label, vector in probes:
                for source, years in [('All', None), ('Patent', None), ('Publication', (2020, 2023))]:
                    allowed = np.ones(len(ids), dtype=bool)
                    if source != 'All': allowed &= meta.source.eq(source).to_numpy()
                    if years: allowed &= meta.year.between(*years).fillna(False).to_numpy(dtype=bool)
                    # Explicit cosine normalization matches Qdrant's cosine distance.
                    scores = (matrix @ vector) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))
                    candidates = np.flatnonzero(allowed)
                    exact = sorted(candidates, key=lambda i: (-float(scores[i]), ids[i]))[:100]
                    hits = self.search(kind, vector, 100, source, years)
                    pos = {doc:i for i,doc in enumerate(ids)}
                    got = [pos[doc] for doc,_ in hits]
                    cutoff = float(scores[exact[-1]]) if exact else 0.
                    tolerance = 2e-6
                    tie_correct = (len(got) == len(exact) and len(set(got)) == len(got)
                        and all(allowed[i] and float(scores[i]) >= cutoff - tolerance for i in got)
                        and all(i in got for i in candidates if float(scores[i]) > cutoff + tolerance)
                        and all(abs(score - float(scores[pos[doc]])) <= tolerance for doc,score in hits))
                    assert tie_correct, (kind, label, source)
                    rows.append(dict(encoder=kind, probe=label, source=source, years=str(years), k=len(exact),
                        id_overlap=len(set(got)&set(exact))/len(exact) if exact else 1.,
                        exact_score_tie_aware_pass=tie_correct, tolerance=tolerance))
            none = self.search(kind, probes[0][1], 10, 'Publication', (1600,1601))
            assert none == []
            subset = ids[100:140]
            assert all(doc in subset for doc,_ in self.search(kind, probes[0][1], 10, doc_ids=subset))
        pd.DataFrame(rows).to_csv(self.artifact_dir / 'qdrant_exact_parity.csv', index=False)
        write_json(self.artifact_dir / 'qdrant_validation.json', dict(status='PASS', collections=mapping, cases=len(rows),
            min_top100_overlap=min(r['id_overlap'] for r in rows), exact_tolerance=2e-6,
            notes='Float32 cutoff ties accepted only within score tolerance; all non-tied better points required'))
