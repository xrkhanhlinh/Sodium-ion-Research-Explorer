"""Isolated exact retrieval over verified existing embeddings; never opens Qdrant."""
import json
import time

import numpy as np
import pandas as pd

from .final_search import FinalSearchEngine
from .final_system_artifacts import ROOT, sha, read_json
from .full_document_inference import OUT as FULL, FullDocumentEncoder, ready
from .search_v2 import normalize, ranked_scores
from .search_radar_core import VERSION, parse_query, condition_evidence, weighted_rrf, aggregate_chunks

OUT=ROOT/'outputs'/VERSION
BRANCHES={'A0':('tfidf','fulldoc'),'A1':('tfidf','bge'),'A2':('bge','fulldoc'),
          'A3':('tfidf','bge','fulldoc'),'A3_weighted':('tfidf','bge','fulldoc')}


class RadarRetriever:
    def __init__(self):
        tick=time.perf_counter()
        if not ready():raise ValueError('Full-document deployment not ready')
        self.engine=FinalSearchEngine(); self.ids=self.engine.ids; self.meta=self.engine.meta
        self.engine._load_tfidf()
        self.bge=self.engine.v2._matrix('pretrained')
        info=read_json(FULL/'embedding_success.json')
        if read_json(FULL/'ordered_doc_ids.json')!=self.ids or sha(FULL/'document_embeddings.npy')!=info['sha256']:
            raise ValueError('Full-document order/hash mismatch')
        self.fulldoc=np.load(FULL/'document_embeddings.npy',mmap_mode='r')
        if self.fulldoc.shape!=(len(self.ids),768) or not np.isfinite(self.fulldoc).all() or not np.allclose(np.linalg.norm(self.fulldoc,axis=1),1,atol=2e-5):
            raise ValueError('Invalid full-document matrix')
        bge_info=read_json(ROOT/'outputs/reports/stage4/pretrained_embedding_manifest.json')
        if sha(ROOT/'models/bge_pretrained/model.safetensors')!=bge_info['identity']['model_sha256']:
            raise ValueError('BGE query checkpoint mismatch')
        protected=read_json(ROOT/'outputs/reports/stage4/input_manifest.json')
        # Historical Stage4 manifest contains the exact tokenizer/config hashes.
        hashes=protected.get('protected_hashes',protected.get('hashes',{}))
        if not hashes:
            hashes=next((v for v in protected.values() if isinstance(v,dict) and 'models/bge_pretrained/tokenizer.json' in v),{})
        if not hashes:raise ValueError('Missing BGE tokenizer provenance')
        for name in ('tokenizer.json','tokenizer_config.json','config.json','1_Pooling/config.json','modules.json'):
            if 'models/bge_pretrained/'+name not in hashes:raise ValueError('Missing BGE provenance: '+name)
        for rel,digest in hashes.items():
            if rel.startswith('models/bge_pretrained/') and not rel.endswith('model.safetensors'):
                if sha(ROOT/rel)!=digest:raise ValueError('BGE tokenizer/pooling mismatch: '+rel)
        self.full_encoder=None;self.chunk_vectors=None;self.chunk_records=None
        self.init_seconds=time.perf_counter()-tick

    def mask(self,source='All',years=None,required_terms=(),topic_id=None):
        mask=self.engine.v2.allowed(source,years)
        if required_terms:mask &= self.engine.v2.evidence_filter(required_terms)
        if topic_id is not None:
            if isinstance(topic_id,bool) or not isinstance(topic_id,(int,np.integer)) or topic_id not in range(10):raise ValueError('Topic ID must be 0..9')
            topics=self.engine._research_topics();mask &= np.array([topics[d]['topic_id']==topic_id for d in self.ids])
        return mask

    def encode(self,query,names=('bge','fulldoc')):
        query=normalize(query);parse_query(query)
        vectors={};timing={}
        for name in names:
            if name=='tfidf':continue
            start=time.perf_counter()
            if name=='bge':vectors[name]=self.engine.v2.encode(query,'pretrained')
            elif name=='fulldoc':
                if self.full_encoder is None:self.full_encoder=FullDocumentEncoder()
                vectors[name],timing['coverage']=self.full_encoder.encode(query)
            else:raise ValueError('Unknown encoder branch')
            timing[name]=time.perf_counter()-start
        return vectors,timing

    def branches(self,query,depth=100,source='All',years=None,required_terms=(),topic_id=None,vectors=None,names=('tfidf','bge','fulldoc')):
        if depth not in (50,100,200):raise ValueError('Candidate depth must be 50, 100 or 200')
        query=normalize(query);parsed=parse_query(query)
        mask=self.mask(source,years,required_terms,topic_id);latency={};rankings={}
        if not mask.any():return {name:[] for name in names},{},mask,parsed
        if vectors is None:vectors,latency=self.encode(query,names)
        if 'tfidf' in names:
            start=time.perf_counter()
            scores=(self.engine.tfidf_matrix@self.engine.tfidf.transform([query]).T).toarray().ravel()
            rankings['tfidf']=ranked_scores(scores,self.ids,mask,depth,positive_only=True);latency['tfidf_retrieval']=time.perf_counter()-start
        for name,matrix in (('bge',self.bge),('fulldoc',self.fulldoc)):
            if name not in names:continue
            start=time.perf_counter();vector=np.asarray(vectors[name])
            if vector.shape!=(768,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=2e-5):raise ValueError('Invalid query vector')
            rankings[name]=ranked_scores(matrix@vector,self.ids,mask,depth);latency[name+'_retrieval']=time.perf_counter()-start
        return rankings,latency,mask,parsed

    def load_chunks(self):
        if self.chunk_vectors is not None:return
        folder=OUT/'chunks'; info=read_json(folder/'manifest.json')
        if info['weights_sha256']!=sha(FULL/'model/model.safetensors') or info['ordered_doc_ids_sha256']!=sha(FULL/'ordered_doc_ids.json'):
            raise ValueError('Chunk model/order mismatch')
        for name in ('vectors.npy','records.parquet'):
            if sha(folder/name)!=info['files'][name]:raise ValueError('Chunk artifact mismatch')
        self.chunk_vectors=np.load(folder/'vectors.npy',mmap_mode='r')
        self.chunk_records=pd.read_parquet(folder/'records.parquet').to_dict('records')
        if self.chunk_vectors.shape!=(len(self.chunk_records),768) or not np.isfinite(self.chunk_vectors).all() or not np.allclose(np.linalg.norm(self.chunk_vectors,axis=1),1,atol=2e-5):raise ValueError('Invalid chunk matrix')
        if set(r['doc_id'] for r in self.chunk_records)!=set(self.ids):raise ValueError('Chunk ID mismatch')

    def chunk_branch(self,vector,mask,depth=100,mode='max'):
        self.load_chunks();scores=self.chunk_vectors@vector
        rows=aggregate_chunks(scores,self.chunk_records,mode)
        allowed={d for d,use in zip(self.ids,mask) if use}
        return [(d,s,r) for d,s,r in rows if d in allowed][:depth]

    def rerank(self,query,fused,depth=30,mode='evidence'):
        if depth not in (20,30,50):raise ValueError('Rerank depth must be 20/30/50')
        parsed=parse_query(query); annotated=[]
        for rank,(doc,score,provenance) in enumerate(fused[:depth],1):
            row=self.meta.loc[doc];evidence=condition_evidence(parsed,row.title,row.abstract)
            if mode=='conditions':key=(evidence['supported'],)
            elif mode=='evidence':key=(-evidence['contradicted'],int(evidence['same_sentence_lexical_support']),evidence['supported'])
            else:raise ValueError('Unknown reranking mode')
            annotated.append((key,rank,doc,score,provenance,evidence))
        annotated.sort(key=lambda r:(tuple(-v for v in r[0]),r[1]))
        return [(doc,score,{**provenance,'evidence':evidence,'original_rank':rank,
            'rank_change':rank-i,'rerank_reason':f'{mode}: supported={evidence["supported"]}, contradicted={evidence["contradicted"]}, same_sentence={evidence["same_sentence_lexical_support"]}'})
            for i,(_,rank,doc,score,provenance,evidence) in enumerate(annotated,1)]+fused[depth:]

    def search(self,query,method='A3',depth=100,rerank_depth=30,weights=None,**filters):
        tick=time.perf_counter()
        base='A3' if method in ('A4','A5','A6') else method
        if base not in BRANCHES:raise ValueError('Unknown pilot method')
        mask=self.mask(**filters)
        vectors,encode_latency=self.encode(query,BRANCHES[base]) if mask.any() else ({},{})
        branches,latency,mask,parsed=self.branches(query,depth,vectors=vectors,names=BRANCHES[base],**filters)
        latency.update(encode_latency)
        selected={b:branches[b] for b in BRANCHES[base]}
        chunk_lookup={}
        if method=='A5':
            chunks=self.chunk_branch(vectors['fulldoc'],mask,depth) if mask.any() else []
            selected['chunk']=[(d,s) for d,s,_ in chunks]
            chunk_lookup={d:record for d,s,record in chunks};branches['chunk']=selected['chunk']
        if method=='A3_weighted' and weights is None:weights=dict(tfidf=1.,bge=1.,fulldoc=2.)
        fused=weighted_rrf(selected,weights)
        if method in ('A4','A6'):fused=self.rerank(query,fused,rerank_depth,'conditions' if method=='A4' else 'evidence')
        results=[]
        for rank,(doc,score,provenance) in enumerate(fused[:10],1):
            row=self.meta.loc[doc]
            if doc in chunk_lookup:
                record=chunk_lookup[doc]
                provenance['best_chunk']={k:record[k] for k in ('global_chunk_id','offset_field','char_start','char_end')}
                for key,value in list(provenance['best_chunk'].items()):
                    if isinstance(value,float) and not np.isfinite(value):provenance['best_chunk'][key]=None
            results.append(dict(rank=rank,doc_id=doc,title=row.title,abstract=row.abstract,source=row.source,
                year=None if pd.isna(row.year) else int(row.year),score=score,provenance=provenance,
                evidence=condition_evidence(parsed,row.title,row.abstract)))
        return dict(query=query,method=method,depth=depth,results=results,candidates=branches,query_decomposition=parsed,
            seconds=time.perf_counter()-tick,branch_latency=latency,eligible_documents=int(mask.sum()),
            version=VERSION,quality_status='EXPERIMENTAL_PENDING_INDEPENDENT_LABELS',
            score_meaning='RRF before lexical condition/evidence reranking; inspect provenance for rank changes' if method in ('A4','A6') else 'Weighted RRF; not relevance probability')

    def close(self):
        self.full_encoder=None; self.engine.close()
