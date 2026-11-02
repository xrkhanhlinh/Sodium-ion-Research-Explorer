"""Additive full-document search channel; legacy benchmarks stay frozen."""
import time
import numpy as np
import pandas as pd
from . import full_document_inference as full
from .search_v2 import ranked_scores,reciprocal_rank_fusion
from .final_system_vectors import VectorDatabase,collection_name
from .final_system_artifacts import vector_spec


def search(engine,query,method,source,years,required_terms,top_k,topic_id,tick):
    if not full.ready():raise ValueError('Full-document model/index has not passed its deployment checks')
    mask=engine.v2.allowed(source,years)
    if required_terms:mask&=engine.v2.evidence_filter(required_terms)
    topics=engine._research_topics()
    if topic_id is not None:
        if isinstance(topic_id,bool) or not isinstance(topic_id,(int,np.integer)) or topic_id not in range(10):
            raise ValueError('Topic ID must be an integer 0..9')
        mask&=np.asarray([topics[doc]['topic_id']==topic_id for doc in engine.ids])
    eligible=int(mask.sum());lexical=[];dense=[];scores={};query_info={}
    if eligible:
        engine._release_bge();engine._release_battery()
        if getattr(engine,'full_document_encoder',None) is None:engine.full_document_encoder=full.FullDocumentEncoder()
        vector,query_info=engine.full_document_encoder.encode(query)
        if getattr(engine,'full_document_database',None) is None:
            spec=vector_spec('batterybert_fulldoc')
            database=VectorDatabase(full.OUT/'qdrant',artifact_dir=full.OUT)
            info=database.collections.get('batterybert_fulldoc',{})
            if (info.get('name')!=collection_name(spec) or info.get('weights_sha256')!=spec['weights_sha256']
                    or info.get('vector_sha256')!=spec['embeddings_sha256'] or info.get('documents')!=len(engine.ids)):
                database.close();raise ValueError('Full-document collection does not match the verified checkpoint/vectors/corpus')
            engine.full_document_database=database
        dense=engine.full_document_database.search('batterybert_fulldoc',vector,100,source,years,
            doc_ids=[doc for doc,use in zip(engine.ids,mask) if use] if (required_terms or topic_id is not None) else None)
        if method=='BATTERYBERT_FULLDOC_HYBRID':
            engine._load_tfidf();raw=(engine.tfidf_matrix@engine.tfidf.transform([query]).T).toarray().ravel()
            lexical=ranked_scores(raw,engine.ids,mask,100,positive_only=True);scores=dict(lexical)
    hybrid=method=='BATTERYBERT_FULLDOC_HYBRID'
    ranks=reciprocal_rank_fusion([lexical,dense],k=60) if hybrid else dense
    cosine=dict(dense);lex_rank={doc:i for i,(doc,s) in enumerate(lexical,1)};den_rank={doc:i for i,(doc,s) in enumerate(dense,1)}
    if eligible:
        matrix=None
        for doc,score in ranks[:top_k]:
            if doc not in cosine:
                if matrix is None:matrix=np.load(full.OUT/'document_embeddings.npy',mmap_mode='r')
                cosine[doc]=float(matrix[engine.v2.positions[doc]]@vector)
    results=[]
    for rank,(doc,score) in enumerate(ranks[:top_k],1):
        row=engine.meta.loc[doc]
        results.append(dict(rank=rank,doc_id=doc,title=row.title,abstract=row.abstract,source=row.source,
            year=None if pd.isna(row.year) else int(row.year),identifier=str(row.identifier),score=float(score),
            semantic_cosine=cosine.get(doc),tfidf_cosine=scores.get(doc),rrf_score=float(score) if hybrid else None,
            lexical_rank=lex_rank.get(doc),semantic_rank=den_rank.get(doc)))
    meaning='RRF score (TF-IDF + full-document trained BatteryBERT, k=60)' if hybrid else 'Full-document BatteryBERT cosine similarity'
    response=dict(query=query,method=method,results=results,seconds=time.perf_counter()-tick,
        source_filter=source,year_filter=years,topic_filter=topic_id,required_terms=list(required_terms),eligible_documents=eligible,
        details=dict(status='OK' if results else 'NO_RESULTS'),score_meaning=meaning,query_coverage=query_info,
        quality_status='PENDING_HUMAN_ANNOTATION',version='batterybert_full_document_20261024_v1',checkpoint='batterybert_fulldoc')
    return engine._attach(response,meaning,research=True)
