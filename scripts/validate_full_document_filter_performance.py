"""Compare old/new pre-ranking filters on the real fixed collection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import time
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from src.final_system_artifacts import corpus
from src.final_system_vectors import VectorDatabase,metadata_filter
from src.full_document_inference import OUT
from src.full_document_topics import assignments
from src.model_comparison_io import write_json


def main():
    meta=corpus();topics=assignments(meta.doc_id.tolist())
    allowed=meta.source.eq('Publication')&meta.year.between(2020,2023)
    chosen=int(topics.loc[allowed].groupby('topic_id').size().idxmax())
    allowed&=topics.topic_id.eq(chosen)
    ids=meta.loc[allowed,'doc_id'].tolist()
    matrix=np.load(OUT/'document_embeddings.npy',mmap_mode='r');query=np.array(matrix[np.flatnonzero(allowed)[0]])
    db=VectorDatabase(OUT/'qdrant',OUT);name=db.collections['batterybert_fulldoc']['name']
    records=[]
    try:
        for trial in range(3):
            # Alternate order to avoid consistently giving the new path warm caches.
            timings={};outputs={}
            for method in (['payload_matchany','verified_point_ids'] if trial%2==0 else ['verified_point_ids','payload_matchany']):
                tick=time.perf_counter()
                if method=='payload_matchany':
                    with threadpool_limits(limits=4):
                        hits=db.client.query_points(name,query=query.tolist(),limit=100,
                            query_filter=metadata_filter('Publication',(2020,2023),ids),with_payload=True).points
                    outputs[method]=[(h.payload['doc_id'],h.score) for h in hits]
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',UserWarning)
                        outputs[method]=db.search('batterybert_fulldoc',query,100,'Publication',(2020,2023),ids)
                timings[method]=time.perf_counter()-tick
            assert [d for d,s in outputs['payload_matchany']]==[d for d,s in outputs['verified_point_ids']]
            assert np.allclose([s for d,s in outputs['payload_matchany']],[s for d,s in outputs['verified_point_ids']],atol=1e-7)
            records.append(dict(trial=trial,**timings));print(records[-1],flush=True)
    finally:db.close()
    old=float(np.median([r['payload_matchany'] for r in records]));new=float(np.median([r['verified_point_ids'] for r in records]))
    result=dict(status='PASS',eligible_documents=len(ids),top100_order_and_scores_identical=True,
        median_old_seconds=old,median_new_seconds=new,speedup=old/new,trials=records,
        scope='Real Qdrant exact search with identical source/year/allowed IDs; vector reused, not encoder or relevance evaluation')
    write_json(OUT/'filter_optimization.json',result);print(result,flush=True)


if __name__=='__main__':main()
