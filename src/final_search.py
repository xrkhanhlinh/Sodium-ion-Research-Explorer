"""Unified research retrieval and historical V2/V3; topics never enter scores.

Search V2 is delegated unchanged to the frozen implementation. V3 changes only
the dense channel to the explicitly named fine-tuned BatteryBERT/Qdrant space.
The research methods use pretrained BatteryBERT or BGE, and TF-IDF/RRF without
reranking. Explicit historical method names preserve the previous algorithms.
"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import gc
import json
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor
import atexit
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from .search_v2 import lexical_tokens, normalize, lexical_fingerprint, ranked_scores, reciprocal_rank_fusion, select_passages
from .search_v2_constraints import EvidenceSearchEngineV2
from .final_system_artifacts import ROOT, OUT, CORPUS, corpus, sha, read_json, topic_data
from .final_system_vectors import VectorDatabase
from .model_comparison_io import write_json

METHODS = ('TFIDF_COSINE', 'SEARCH_V2', 'SEARCH_V3')
RESEARCH_METHODS = ('TFIDF_COSINE', 'BATTERYBERT_SEMANTIC', 'BATTERYBERT_HYBRID', 'BGE_SEMANTIC')
RESEARCH_OUT = ROOT / 'outputs/research_completion_v1'
TFIDF = OUT / 'tfidf'


def build_tfidf():
    TFIDF.mkdir(parents=True, exist_ok=True)
    identity = dict(corpus_sha256=sha(CORPUS), tokenizer=lexical_fingerprint(),
        settings=dict(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False, dtype='float32'))
    marker = TFIDF / 'manifest.json'
    if marker.exists():
        info = read_json(marker)
        if info['identity'] != identity or sha(TFIDF / 'vectorizer.pkl') != info['vectorizer_sha256'] or sha(TFIDF / 'matrix.npz') != info['matrix_sha256']:
            raise ValueError('Existing TF-IDF artifacts differ; use a new output version')
        return
    meta = corpus()
    texts = (meta.title.fillna('') + '\n' + meta.abstract.fillna('')).tolist()
    tick = time.perf_counter()
    vectorizer = TfidfVectorizer(tokenizer=lexical_tokens, token_pattern=None, lowercase=False,
                                 dtype=np.float32, norm='l2')
    matrix = vectorizer.fit_transform(texts).tocsr()
    sparse.save_npz(TFIDF / 'matrix.npz', matrix)
    with (TFIDF / 'vectorizer.pkl').open('wb') as stream:
        pickle.dump(vectorizer, stream, protocol=4)
    write_json(TFIDF / 'ordered_doc_ids.json', meta.doc_id.tolist())
    write_json(marker, dict(identity=identity, vectorizer_sha256=sha(TFIDF / 'vectorizer.pkl'),
        matrix_sha256=sha(TFIDF / 'matrix.npz'), documents=matrix.shape[0], features=matrix.shape[1],
        nonzeros=matrix.nnz, seconds=time.perf_counter() - tick,
        note='Full cleaned Title–Abstract; TF-IDF baseline, distinct from BERTopic c-TF-IDF'))


class FinalSearchEngine:
    def __init__(self):
        tick = time.perf_counter()
        # Existing lexical index must validate and be reused. Never rebuild a
        # historical index implicitly while serving the final system.
        state = read_json(ROOT / 'data/search_v2/lexical_manifest.json')
        cfg = read_json(ROOT / 'configs/search_v2.json')
        expected = dict(corpus_sha256=sha(CORPUS), tokenizer_source_sha256=lexical_fingerprint(),
                        k1=cfg['bm25_k1'], b=cfg['bm25_b'])
        if state['identity'] != expected or sha(ROOT / 'data/search_v2/lexical.pkl') != state['sha256']:
            raise ValueError('Frozen V2 lexical index missing or invalid; no silent rebuild')
        self.v2 = EvidenceSearchEngineV2(use_query_cache=False)
        self.ids = self.v2.ids
        self.meta = self.v2.meta
        self.database = None
        self.research_database = None
        self.battery = self.battery_tokenizer = None
        self.battery_kind = None
        self.research_topic_map = None
        self.tfidf = self.tfidf_matrix = None
        self.topic_maps = None
        self.init_seconds = time.perf_counter() - tick

    def _topics(self):
        if self.topic_maps is None:
            self.topic_maps = {}
            for name in ('LDA', 'BERTopic–BGE', 'BERTopic–BatteryBERT (pretrained)'):
                data = topic_data(name)
                self.topic_maps[name] = {r.doc_id:dict(topic_id=int(r.topic_id), label=data['labels'][int(r.topic_id)])
                    for r in data['assignments'].itertuples()}
        return self.topic_maps

    def _load_tfidf(self):
        if self.tfidf is None:
            info = read_json(TFIDF / 'manifest.json')
            if info['identity']['corpus_sha256'] != sha(CORPUS) or info['identity']['tokenizer'] != lexical_fingerprint():
                raise ValueError('TF-IDF corpus/tokenizer mismatch')
            if sha(TFIDF / 'vectorizer.pkl') != info['vectorizer_sha256'] or sha(TFIDF / 'matrix.npz') != info['matrix_sha256']:
                raise ValueError('TF-IDF artifact checksum mismatch')
            if read_json(TFIDF / 'ordered_doc_ids.json') != self.ids:
                raise ValueError('TF-IDF document order mismatch')
            with (TFIDF / 'vectorizer.pkl').open('rb') as stream:
                self.tfidf = pickle.load(stream)
            self.tfidf_matrix = sparse.load_npz(TFIDF / 'matrix.npz')

    def _release_bge(self):
        self.v2.encoder = self.v2.encoder_kind = None
        gc.collect()

    def _release_battery(self):
        self.battery = self.battery_tokenizer = None
        self.battery_kind = None
        gc.collect()

    def encode_battery(self, query, variant='finetuned'):
        import torch
        if variant not in ('pretrained','finetuned'):
            raise ValueError('Unknown BatteryBERT variant')
        if self.battery is None or self.battery_kind != variant:
            self._release_bge()
            self._release_battery()
            from transformers import AutoTokenizer, AutoModel
            folder = ROOT / f'outputs/model_comparison_v2/batterybert_{variant}/model'
            if variant == 'pretrained':
                provenance = read_json(folder / 'download_provenance.json')
                names = ('pytorch_model.bin','config.json','tokenizer.json','tokenizer_config.json')
                for filename in names:
                    if sha(folder / filename) != provenance['hashes'][filename]:
                        raise ValueError('Pretrained query checkpoint/tokenizer checksum mismatch: '+filename)
            else:
                info = read_json(OUT / 'batterybert_finetuned/embedding_success.json')
                if sha(folder / 'model.safetensors') != info['identity']['weights']['model.safetensors']:
                    raise ValueError('Query checkpoint differs from document checkpoint')
                frozen = read_json(OUT / 'protected_artifacts.json')
                for filename in ('config.json','tokenizer.json','tokenizer_config.json'):
                    path = folder / filename
                    expected = frozen[path.relative_to(ROOT).as_posix()]['sha256']
                    if sha(path) != expected:
                        raise ValueError('Query tokenizer/config differs from frozen document inference: '+filename)
            torch.set_num_threads(4)
            self.battery_tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True)
            self.battery = AutoModel.from_pretrained(folder, local_files_only=True, add_pooling_layer=False,
                                                    trust_remote_code=False).eval()
            self.battery_kind = variant
        tokens = self.battery_tokenizer(query, truncation=False, add_special_tokens=True)
        maximum = 256 if variant == 'pretrained' else 128
        if len(tokens['input_ids']) > maximum:
            raise ValueError(f'BatteryBERT query exceeds {maximum} tokens; shorten the query')
        features = {k:torch.tensor([v]) for k,v in tokens.items()}
        with torch.inference_mode():
            h = self.battery(**features).last_hidden_state
            mask = features['attention_mask'].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, dim=1)[0].numpy()

    def _research_topics(self):
        from . import optimized_topics,full_document_topics,revised_topics
        catalog=('revised' if revised_topics.ready() else 'full_document' if full_document_topics.ready() else 'optimized' if optimized_topics.ready() else 'legacy')
        if self.research_topic_map is None or getattr(self,'research_topic_catalog',None)!=catalog:
            if catalog=='revised':
                assignments = revised_topics.assignments(self.ids)
            elif catalog=='full_document':
                assignments = full_document_topics.assignments(self.ids)
            elif catalog=='optimized':
                assignments = optimized_topics.assignments(self.ids)
            else:
                path = RESEARCH_OUT / 'topics/topic_assignments.csv'
                assignments = pd.read_csv(path)
            if assignments.doc_id.tolist() != self.ids or not assignments.doc_id.is_unique:
                raise ValueError('Research topic assignments do not match corpus order')
            self.research_topic_map = {r.doc_id:dict(topic_id=int(r.topic_id),label=r.topic_label) for r in assignments.itertuples()}
            self.research_topic_catalog = catalog
        return self.research_topic_map

    def _attach(self, response, score_method, research=False):
        maps = self._topics()
        response['score_method'] = score_method
        for r in response['results']:
            r['score_method'] = score_method
            r['topics'] = {name: mapping[r['doc_id']] for name,mapping in maps.items()}
            # Flatten the main topic label for CSV clients.
            r['topic_label'] = r['topics']['BERTopic–BatteryBERT (pretrained)']['label']
            r['topic_encoder_variant'] = 'BatteryBERT pretrained; topic exploration only'
            if research:
                topic = self._research_topics()[r['doc_id']]
                r['topic_id'], r['topic_label'] = topic['topic_id'], topic['label']
                from . import optimized_topics,full_document_topics,revised_topics
                r['topic_encoder_variant'] = ('BatteryBERT full-document + source projection + KMeans10 n_init50; source-corrected keywords; AI provisional labels'
                    if revised_topics.ready() else 'BatteryBERT full-document trained + train-fitted geometry + KMeans10; provisional labels'
                    if full_document_topics.ready() else 'BatteryBERT fine-tuned + train-fitted source projection + KMeans10; tentative labels'
                    if optimized_topics.ready() else 'BatteryBERT pretrained + fixed KMeans; tentative scientific labels')
            if not r.get('evidence_passage'):
                r['evidence_passage'] = self.lexical_passage(r['title'], r['abstract'], response['query'])
                r['passage_method'] = 'Lexical-overlap excerpt; not a neural relevance explanation'
            else:
                r['passage_method'] = 'Query-aware passage selected for MiniLM reranking'
        return response

    @staticmethod
    def lexical_passage(title, abstract, query):
        words = set(lexical_tokens(query))
        pieces = [p.strip() for p in re.split(r'(?<=[.!?])\s+', str(abstract)) if p.strip()]
        if not pieces: return title
        return max(pieces, key=lambda p:len(set(lexical_tokens(p)) & words))

    def search(self, query, method='SEARCH_V3', source='All', years=None, required_terms=(), top_k=10, topic_id=None):
        from . import full_document_inference
        if method not in set(METHODS) | set(RESEARCH_METHODS) | set(full_document_inference.METHODS): raise ValueError('Unknown search method')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError('top_k must be 1..10')
        query = normalize(query)
        maximum=1000000 if method in full_document_inference.METHODS else 10000
        if not query or len(query) > maximum or not re.search(r'[A-Za-z0-9]',query):
            raise ValueError(f'Enter a nonempty English query of at most {maximum:,} characters')
        tick = time.perf_counter()
        if method in full_document_inference.METHODS:
            from .full_document_retrieval import search as full_document_search
            return full_document_search(self,query,method,source,years,required_terms,top_k,topic_id,tick)
        if hasattr(self,'full_document_encoder'):
            self.full_document_encoder=None;gc.collect()
        if method in RESEARCH_METHODS and (method != 'TFIDF_COSINE' or (RESEARCH_OUT/'topics/topic_assignments.csv').exists()):
            return self._research_search(query,method,source,years,required_terms,top_k,topic_id,tick)
        if topic_id is not None:
            raise ValueError('Topic filters require one of the research search methods')
        if method == 'SEARCH_V2':
            self._release_battery()
            response = self.v2.search(query, method='RERANK_PRETRAINED', source=source, years=years,
                                      required_terms=required_terms, top_k=top_k)
            response['original_v2_method'] = response['method']
            response['method'] = 'SEARCH_V2'
            return self._attach(response, 'MiniLM cross-encoder score (V2)')
        mask = self.v2.allowed(source, years)
        if required_terms:
            mask &= self.v2.evidence_filter(required_terms)
        eligible = int(mask.sum())
        ranks = []
        status = 'NO_DOCUMENTS_AFTER_FILTER' if not eligible else 'NO_LEXICAL_SUPPORT'
        self.v2.clear_scores()
        if eligible and method == 'TFIDF_COSINE':
            self._load_tfidf()
            q = self.tfidf.transform([query])
            scores = (self.tfidf_matrix @ q.T).toarray().ravel()
            ranks = ranked_scores(scores, self.ids, mask, top_k, positive_only=True)
        elif eligible:
            lexical = self.v2.lexical(query, mask)
            if lexical:
                vector = self.encode_battery(query)
                if self.database is None: self.database = VectorDatabase()
                dense = self.database.search('batterybert_finetuned', vector, 100, source, years,
                    doc_ids=[doc for doc,allowed in zip(self.ids,mask) if allowed] if required_terms else None)
                fused = reciprocal_rank_fusion([lexical, dense], k=60)
                ranks = self.v2.rerank(query, fused[:50])
        if ranks: status = 'OK'
        results = []
        for rank,(doc,score) in enumerate(ranks[:top_k],1):
            r = self.meta.loc[doc]
            passage = self.v2._pair_cache.get((query,doc))
            results.append(dict(rank=rank,doc_id=doc,title=r.title,abstract=r.abstract,source=r.source,
                year=None if pd.isna(r.year) else int(r.year), identifier=str(r.identifier),score=float(score),
                evidence_passage=passage['text'] if passage else '',
                passage_char_start=passage['start'] if passage else None,
                passage_char_end=passage['end'] if passage else None,
                passage_offset_field='normalized_abstract' if passage else None))
        response = dict(query=query,method=method,results=results,seconds=time.perf_counter()-tick,
            source_filter=source,year_filter=years,required_terms=list(required_terms),eligible_documents=eligible,
            details=dict(status=status),quality_status='PENDING_INDEPENDENT_LABELS',version='final_system_v1',
            score_meaning='Method-specific ranking score; not a probability of relevance',
            checkpoint='batterybert_finetuned' if method=='SEARCH_V3' else None)
        return self._attach(response, 'TF-IDF cosine similarity' if method=='TFIDF_COSINE' else 'MiniLM cross-encoder score (V3)')

    def _research_search(self, query, method, source, years, required_terms, top_k, topic_id, tick):
        mask = self.v2.allowed(source, years)
        if required_terms:
            mask &= self.v2.evidence_filter(required_terms)
        topics = self._research_topics()
        if topic_id is not None:
            if isinstance(topic_id,bool) or not isinstance(topic_id,(int,np.integer)) or int(topic_id) not in range(10):
                raise ValueError('Topic ID must be an integer 0..9')
            mask &= np.asarray([topics[doc]['topic_id']==int(topic_id) for doc in self.ids])
        eligible = int(mask.sum())
        lexical, dense, ranks = [], [], []
        scores_by_doc = {}
        self.v2.clear_scores()
        if eligible and method in ('TFIDF_COSINE','BATTERYBERT_HYBRID'):
            self._load_tfidf()
            scores = (self.tfidf_matrix @ self.tfidf.transform([query]).T).toarray().ravel()
            lexical = ranked_scores(scores,self.ids,mask,100,positive_only=True)
            scores_by_doc = dict(lexical)
            ranks = lexical
        if eligible and method != 'TFIDF_COSINE':
            if method == 'BGE_SEMANTIC':
                self._release_battery()
                dense = self.v2.dense(query,'pretrained',mask)
            else:
                if self.research_database is None:
                    self.research_database = VectorDatabase(RESEARCH_OUT/'qdrant',artifact_dir=RESEARCH_OUT)
                vector = self.encode_battery(query,'pretrained')
                dense = self.research_database.search('batterybert_pretrained',vector,100,source,years,
                    doc_ids=[doc for doc,allowed in zip(self.ids,mask) if allowed] if (required_terms or topic_id is not None) else None)
            ranks = reciprocal_rank_fusion([lexical,dense],k=60) if method=='BATTERYBERT_HYBRID' else dense
        cosine = dict(dense)
        if method=='BATTERYBERT_HYBRID':
            # A lexical-only RRF result may lie outside dense Top100. Display
            # its actual cosine without inventing a dense candidate rank.
            missing=[doc for doc,_ in ranks[:top_k] if doc not in cosine]
            if missing:
                path=ROOT/'outputs/model_comparison_v2/batterybert_pretrained/document_embeddings.npy'
                matrix=np.load(path,mmap_mode='r')
                for doc in missing:
                    cosine[doc]=float(matrix[self.v2.positions[doc]] @ vector)
        lexical_rank = {doc:i for i,(doc,_) in enumerate(lexical,1)}
        dense_rank = {doc:i for i,(doc,_) in enumerate(dense,1)}
        results = []
        for rank,(doc,score) in enumerate(ranks[:top_k],1):
            row = self.meta.loc[doc]
            results.append(dict(rank=rank,doc_id=doc,title=row.title,abstract=row.abstract,source=row.source,
                year=None if pd.isna(row.year) else int(row.year),identifier=str(row.identifier),score=float(score),
                semantic_cosine=cosine.get(doc),tfidf_cosine=scores_by_doc.get(doc),
                rrf_score=float(score) if method=='BATTERYBERT_HYBRID' else None,
                lexical_rank=lexical_rank.get(doc),semantic_rank=dense_rank.get(doc)))
        meaning = 'RRF score (TF-IDF + BatteryBERT pretrained, k=60)' if method=='BATTERYBERT_HYBRID' else 'TF-IDF cosine similarity' if method=='TFIDF_COSINE' else 'Dense cosine similarity'
        response = dict(query=query,method=method,results=results,seconds=time.perf_counter()-tick,
            source_filter=source,year_filter=years,topic_filter=topic_id,required_terms=list(required_terms),
            eligible_documents=eligible,details=dict(status='OK' if results else 'NO_RESULTS'),
            quality_status='PENDING_HUMAN_ANNOTATION',version='research_completion_v1',score_meaning=meaning,
            checkpoint='batterybert_pretrained' if method.startswith('BATTERYBERT') else 'bge_pretrained' if method=='BGE_SEMANTIC' else None)
        return self._attach(response,meaning,research=True)

    def close(self):
        if getattr(self,'full_document_database',None) is not None:
            self.full_document_database.close();self.full_document_database=None
        if hasattr(self,'full_document_encoder'):self.full_document_encoder=None
        if self.research_database is not None:
            self.research_database.close(); self.research_database = None
        if self.database is not None:
            self.database.close(); self.database = None
        self._release_battery()
        self.v2.close()


class FinalSearchService:
    """One worker bounds model RAM and owns the persistent Qdrant connection."""
    def __init__(self):
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='final-search')
        self.engine = None
        self.closed = False
        atexit.register(self.close)

    def _search(self, *args, **kwargs):
        if self.engine is None: self.engine = FinalSearchEngine()
        return self.engine.search(*args, **kwargs)

    def search(self, *args, **kwargs):
        tick = time.perf_counter()
        result = self.worker.submit(self._search,*args,**kwargs).result()
        result['system_seconds'] = time.perf_counter() - tick
        return result

    def compare(self, query, source='All', years=None, required_terms=()):
        return [self.search(query,method,source,years,required_terms) for method in METHODS]

    def close(self):
        if self.closed: return
        self.closed = True
        # Drain requests before freeing resources. At interpreter shutdown the
        # executor may already reject submit(), so cleanup must not enqueue a
        # new task. Qdrant was opened with its explicit cross-thread option.
        self.worker.shutdown(wait=True)
        if self.engine is not None:
            self.engine.close()
            self.engine = None
