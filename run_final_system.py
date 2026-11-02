"""Build and validate the final system from existing checkpoints; NO training.

Run prepare, then embeddings (only if missing), qdrant, validate, benchmark,
evaluate and report. Qdrant Local must have one owning process at a time.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['USE_TF'] = os.environ['USE_FLAX'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
import pandas as pd
import psutil
from src.final_system_artifacts import ROOT, OUT, OLD, TOPICS, snapshot, export_topics, verify_protected, sha, read_json, corpus
from src.model_comparison_io import write_json


@contextmanager
def profile():
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.05): peak[0] = max(peak[0],process.memory_info().rss)
    thread = threading.Thread(target=sample,daemon=True); thread.start()
    tick = time.perf_counter()
    result = {}
    try: yield result
    finally:
        stop.set(); thread.join()
        result.update(seconds=time.perf_counter()-tick,peak_rss_mib=max(peak[0],process.memory_info().rss)/2**20)


def prepare():
    snapshot()
    export_topics()
    from src.final_search import build_tfidf
    build_tfidf()


def qdrant():
    from src.final_system_vectors import VectorDatabase
    database = VectorDatabase()
    try:
        for kind in ('bge_pretrained','batterybert_finetuned'):
            with profile() as perf: name = database.build(kind)
            write_json(OUT/(kind+'_qdrant_build_profile.json'),dict(collection=name,**perf))
        database.verify()
    finally: database.close()


def validate():
    import subprocess
    result = subprocess.run([sys.executable,'-m','pytest','tests/test_final_system.py','tests/test_search_v2.py',
        'tests/test_search_runtime.py','tests/test_math_contracts.py','-q','--junitxml=outputs/final_system/tests.xml'],cwd=ROOT)
    if result.returncode: raise RuntimeError('Tests failed')
    from src.final_system_vectors import VectorDatabase
    database = VectorDatabase()
    try: database.verify()
    finally: database.close()


def benchmark():
    from src.final_search import FinalSearchEngine, METHODS
    queries = pd.read_parquet(ROOT/'data/stage4/queries.parquet')
    dest = OUT/'query_runs'; dest.mkdir(exist_ok=True)
    # Fingerprint every model/config/source used in live ranking.
    source_paths = ['src/final_search.py','src/final_system_vectors.py','src/search_v2.py',
                    'src/search_v2_constraints.py','configs/search_v2.json']
    identity = {p:sha(ROOT/p) for p in source_paths}
    identity['collections'] = {kind:{k:v for k,v in info.items() if k!='seconds'}
        for kind,info in read_json(OUT/'qdrant_collections.json').items()}
    identity['corpus'] = sha(ROOT/'data/processed/main_corpus.parquet')
    marker = OUT/'benchmark_identity.json'
    if marker.exists() and read_json(marker) != identity: raise ValueError('Benchmark identity changed; preserve old run and use a new version')
    write_json(marker,identity)
    engine = None
    responses = []
    try:
        for method in METHODS:
            for i,q in enumerate(queries.itertuples()):
                path = dest/f'{method}_{q.query_id}.json'
                if path.exists():
                    response = read_json(path)
                    if response['query'] != q.query or response['method'] != method: raise ValueError('Cached benchmark mismatch')
                else:
                    if engine is None:
                        with profile() as init: engine = FinalSearchEngine()
                        write_json(OUT/'engine_initialization_profile.json',init)
                    with profile() as perf: response = engine.search(q.query,method)
                    response.update(query_id=q.query_id,system_seconds=perf['seconds'],peak_rss_mib=perf['peak_rss_mib'],
                        first_request_for_method=i==0,uses_query_embedding_cache=False)
                    write_json(path,response)
                responses.append(response)
                print(method,q.query_id,len(response['results']),round(response['system_seconds'],3),flush=True)
    finally:
        if engine: engine.close()
    rows, manifest = [], []
    for response in responses:
        manifest.append({k:response[k] for k in ('query_id','query','method','system_seconds','peak_rss_mib','first_request_for_method')} |
            dict(status='OK',retrieval_status=response['details']['status'],result_count=len(response['results'])))
        for result in response['results']:
            row = {k:v for k,v in result.items() if k!='topics'}
            row.update(query_id=response['query_id'],query=response['query'],method=response['method'])
            rows.append(row)
    ranks = pd.DataFrame(rows)
    ranks.to_csv(OUT/'rankings_top10.csv',index=False)
    runs = pd.DataFrame(manifest)
    runs.to_csv(OUT/'run_manifest.csv',index=False)
    runs.groupby('method').agg(requests=('system_seconds','size'),mean_seconds=('system_seconds','mean'),
        median_seconds=('system_seconds','median'),max_peak_rss_mib=('peak_rss_mib','max')).to_csv(OUT/'latency_all_requests.csv')
    warm = runs[~runs.first_request_for_method]
    warm.groupby('method').agg(requests=('system_seconds','size'),mean_seconds=('system_seconds','mean'),
        median_seconds=('system_seconds','median'),p95_seconds=('system_seconds',lambda x:x.quantile(.95)),
        max_peak_rss_mib=('peak_rss_mib','max')).to_csv(OUT/'latency_warm.csv')
    # Concrete regression against saved original V2 Top5. No old files changed.
    old = pd.read_csv(ROOT/'outputs/search_upgrade_20261018/rankings_top10.csv',keep_default_na=False)
    original = old[old.method.eq('RERANK_PRETRAINED')&old['rank'].le(5)]
    actual = ranks[ranks.method.eq('SEARCH_V2')&ranks['rank'].le(5)]
    checks=[]
    for q in queries.query_id:
        left = original[original.query_id.eq(q)].sort_values('rank')
        right = actual[actual.query_id.eq(q)].sort_values('rank')
        matched = left.doc_id.tolist() == right.doc_id.tolist()
        score_error = float(np.max(np.abs(left.score.to_numpy(float)-right.score.to_numpy(float)))) if matched and len(left) else 0.
        checks.append(dict(query_id=q,top5_ids_identical=matched,score_max_abs_error=score_error,
                           original_count=len(left),actual_count=len(right)))
    pd.DataFrame(checks).to_csv(OUT/'v2_top5_regression.csv',index=False)
    if not all(r['top5_ids_identical'] and r['score_max_abs_error']<2e-4 for r in checks):
        raise ValueError('V2 regression differs; inspect artifacts before claiming parity')
    # Pair overlap/rank changes have descriptive meaning without relevance labels.
    overlap=[]
    for q in queries.query_id:
        selected=ranks[ranks.query_id.eq(q)]
        a=selected[selected.method.eq('SEARCH_V2')].set_index('doc_id')['rank'].to_dict()
        b=selected[selected.method.eq('SEARCH_V3')].set_index('doc_id')['rank'].to_dict()
        for doc in sorted(set(a)|set(b)):
            overlap.append(dict(query_id=q,doc_id=doc,v2_rank=a.get(doc),v3_rank=b.get(doc),in_both=doc in a and doc in b))
    pd.DataFrame(overlap).to_csv(OUT/'v2_v3_rank_changes.csv',index=False)
    from src.final_system_evaluation import create_pool
    create_pool()
    write_json(OUT/'benchmark_success.json',dict(status='PASS',queries=len(queries),methods=list(METHODS),
        rankings_sha256=sha(OUT/'rankings_top10.csv'),runs_sha256=sha(OUT/'run_manifest.csv'),
        v2_original_top5_regression_pass=True,query_embeddings_cached=False))


def optimize():
    """Measure real existing bottlenecks without changing ranking algorithms."""
    from src.final_search import FinalSearchEngine
    from src.search_v2 import ranked_scores, select_passages
    engine=FinalSearchEngine()
    profiles=[]
    try:
        mask=engine.v2.allowed()
        rng=np.random.default_rng(42)
        scores=rng.normal(size=len(mask)).astype('float32')
        scores[10:15]=scores[20]  # Include cutoff/ordinary ties in parity tests too.
        with profile() as before:
            for _ in range(200):
                baseline=sorted([(engine.ids[i],float(scores[i])) for i in np.flatnonzero(mask)],key=lambda x:(-x[1],x[0]))[:100]
        with profile() as after:
            for _ in range(200): optimized=ranked_scores(scores,engine.ids,mask,100)
        assert baseline==optimized
        profiles.extend([dict(operation='Top100 selection / 200 repetitions',variant='full sort',**before),
                         dict(operation='Top100 selection / 200 repetitions',variant='partition + boundary tie sort',**after)])
        engine.v2._load_reranker()
        query='Hard carbon anodes with high initial Coulombic efficiency'
        candidates=engine.v2.lexical(query,mask)[:12]
        pairs=[]
        for doc,_ in candidates:
            r=engine.meta.loc[doc]
            for p in select_passages(r.title,r.abstract,query,engine.v2.reranker.tokenizer): pairs.append((query,p['text']))
        # Identical trained checkpoint and pairs; only batch scheduling differs.
        outputs=[]
        for batch in (1,4):
            with profile() as perf:
                values=np.asarray(engine.v2.reranker.predict(pairs,batch_size=batch,show_progress_bar=False)).reshape(-1)
            outputs.append(values)
            profiles.append(dict(operation=f'MiniLM / {len(pairs)} real query-passages',variant=f'batch {batch}',**perf))
        delta=float(np.max(np.abs(outputs[0]-outputs[1])))
        assert np.allclose(outputs[0],outputs[1],atol=2e-4)
        assert np.argsort(-outputs[0],kind='stable').tolist()==np.argsort(-outputs[1],kind='stable').tolist()
        write_json(OUT/'optimization_parity.json',dict(status='PASS',topk_exact=True,
            minilm_max_abs_error=delta,minilm_pair_order_identical=True,existing_batch_size_unchanged=4,
            note='Microbenchmarks use fixed scores/real passages, not relevance judgments; RSS is process-wide'))
    finally: engine.close()
    pd.DataFrame(profiles).to_csv(OUT/'optimization_profiles.csv',index=False)


def smoke():
    from src.final_search import FinalSearchEngine, METHODS
    from src.search_v2 import lexical_tokens
    engine=FinalSearchEngine()
    rows=[]
    try:
        for method in METHODS:
            result=engine.search('Na2FeP2O7 cathode cycling stability',method,'Publication',(2020,2023),
                                required_terms=['Na2FeP2O7'])
            assert result['results'] and len(result['results'])<=10
            seen=set()
            for row in result['results']:
                assert row['doc_id'] not in seen;seen.add(row['doc_id'])
                assert row['source']=='Publication' and 2020<=row['year']<=2023
                assert 'chem:Na2FeP2O7' in lexical_tokens(row['title']+' '+row['abstract'])
                assert row['evidence_passage'] and row['topic_label'] and np.isfinite(row['score'])
                rows.append({k:v for k,v in row.items() if k!='topics'}|dict(method=method))
            if method=='SEARCH_V2':
                original=engine.v2.search('Na2FeP2O7 cathode cycling stability','RERANK_PRETRAINED',
                    'Publication',(2020,2023),top_k=5,required_terms=['Na2FeP2O7'])
                assert [r['doc_id'] for r in original['results']]==[r['doc_id'] for r in result['results'][:5]]
    finally:engine.close()
    pd.DataFrame(rows).to_csv(OUT/'filtered_search_smoke.csv',index=False)
    write_json(OUT/'filtered_search_smoke.json',dict(status='PASS',methods=list(METHODS),
        source='Publication',years=[2020,2023],required_terms=['Na2FeP2O7'],
        original_v2_top5_prefix_preserved=True,duplicates=False))


def model_loading():
    # Isolated subprocesses keep only one large neural/topic model resident.
    commands={
      'lda':"import numpy as np; from gensim.models import LdaModel; m=LdaModel.load('outputs/model_comparison_v2/lda_k10/model.gensim'); assert m.num_topics==10 and m.num_terms==35098; assert np.allclose(m.get_topics(),np.load('outputs/model_comparison_v2/lda_k10/topic_word_distribution.npy'),atol=1e-6)",
      'bertopic_bge':"import numpy as np; from bertopic import BERTopic; m=BERTopic.load('outputs/model_comparison_v2/bertopic_bge_k10/model.pickle'); assert len(set(m.topics_)-{-1})==10; assert np.array_equal(m.topics_,np.load('outputs/model_comparison_v2/bertopic_bge_k10/assignments.npy'))",
      'bertopic_batterybert':"import numpy as np; from bertopic import BERTopic; m=BERTopic.load('outputs/model_comparison_v2/bertopic_batterybert_k10/pretrained/model.pickle'); assert len(set(m.topics_)-{-1})==10; assert np.array_equal(m.topics_,np.load('outputs/model_comparison_v2/bertopic_batterybert_k10/pretrained/assignments.npy'))",
      'bge':"from sentence_transformers import SentenceTransformer; m=SentenceTransformer('models/bge_pretrained',device='cpu',local_files_only=True); assert m.get_sentence_embedding_dimension()==768",
      'batterybert':"from transformers import AutoModel; m=AutoModel.from_pretrained('outputs/model_comparison_v2/batterybert_finetuned/model',local_files_only=True,add_pooling_layer=False); assert m.config.hidden_size==768"}
    rows=[]
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    for name,code in commands.items():
        tick=time.perf_counter()
        with (logs/(name+'_reload.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        rows.append(dict(model=name,status='PASS' if result.returncode==0 else 'FAIL',seconds=time.perf_counter()-tick))
        write_json(OUT/'model_loading.json',rows)
        if result.returncode: raise RuntimeError('Model reload failed: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','embeddings','topics','qdrant','validate','benchmark','evaluate','optimize','smoke','models','preserve','report'])
    parser.add_argument('--labels',type=Path)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='embeddings':
        subprocess.run([sys.executable,'scripts/prepare_final_battery_vectors.py'],cwd=ROOT,check=True)
    elif args.phase=='topics': export_topics()
    elif args.phase=='evaluate':
        from src.final_system_evaluation import evaluate
        evaluate(args.labels)
    elif args.phase=='models': model_loading()
    elif args.phase=='preserve': verify_protected()
    elif args.phase=='report':
        from src.final_system_reports import report
        report()
    else: globals()[args.phase]()


if __name__=='__main__': main()
