"""Sequential per-method process benchmark: warm wall latency and sampled peak RSS."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import psutil
from src.search_radar_retrieval import RadarRetriever,OUT
from src.final_system_artifacts import sha
from src.model_comparison_io import write_json


def worker(method,repetitions):
    target=OUT/'benchmark';target.mkdir(parents=True,exist_ok=True)
    queries=pd.read_csv(OUT/'historical/queries.csv')
    process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.02):peak[0]=max(peak[0],process.memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start();engine=None
    try:
        init=time.perf_counter();engine=RadarRetriever();init=time.perf_counter()-init
        first=time.perf_counter();engine.search(queries.iloc[0]['query'],method=method);cold=time.perf_counter()-first
        rows=[]
        for repetition in range(repetitions):
            for q in queries.itertuples():
                start=time.perf_counter();result=engine.search(q.query,method=method)
                rows.append(dict(method=method,query_id=q.query_id,repetition=repetition,
                    seconds=time.perf_counter()-start,result_count=len(result['results']),rss_bytes=process.memory_info().rss))
        pd.DataFrame(rows).to_csv(target/(method+'_runs.csv'),index=False)
        write_json(target/(method+'_process.json'),dict(method=method,pid=process.pid,init_seconds=init,
            first_search_seconds=cold,peak_rss_bytes=peak[0],rss_sample_interval_ms=20,
            query_runs=len(rows),scope='fresh process per method; initialization + first query excluded from warm latency; serial CPU',
            source_sha256=sha(ROOT/'src/search_radar_retrieval.py')))
    finally:
        stop.set();monitor.join()
        if engine is not None:engine.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--repetitions',type=int,default=2)
    args=parser.parse_args()
    if args.worker:worker(args.worker,args.repetitions);return
    if args.repetitions<1:raise ValueError('Positive repetitions required')
    methods=['A0','A1','A2','A3','A4','A5','A6'];rows=[]
    for method in methods:
        result=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)), '--worker',method,'--repetitions',str(args.repetitions)],cwd=ROOT)
        if result.returncode:raise RuntimeError('Benchmark worker failed: '+method)
        target=OUT/'benchmark';runs=pd.read_csv(target/(method+'_runs.csv'))
        info=json.loads((target/(method+'_process.json')).read_text(encoding='utf8'))
        rows.append(dict(method=method,depth=100,P50_seconds=runs.seconds.median(),P95_seconds=runs.seconds.quantile(.95),
            rss_max_bytes=info['peak_rss_bytes'],query_runs=len(runs),init_seconds=info['init_seconds'],first_search_seconds=info['first_search_seconds']))
        pd.DataFrame(rows).to_csv(target/'summary.csv',index=False)
        print('BENCHMARK',method,rows[-1],flush=True)
    write_json(OUT/'benchmark/manifest.json',dict(methods=methods,queries=20,repetitions=args.repetitions,
        source_sha256=sha(Path(__file__)),production_changed=False,
        caveat='Sequential fresh processes; shared OS page cache; observed P95 of query/repetition mix, not load-test SLA'))


if __name__=='__main__':main()
