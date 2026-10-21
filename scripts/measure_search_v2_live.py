"""Direct local request timings: actual query encoding + uncached CE inference."""
from pathlib import Path
import argparse
from datetime import datetime
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.search_v2 import ROOT, SearchEngineV2, digest, write_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/search_runs' /
                        ('latency_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    output = parser.parse_args().output
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty latency output folder')
    output.mkdir(parents=True, exist_ok=True)
    queries = pd.read_parquet(ROOT / 'data/stage4/queries.parquet')
    selected = queries[queries.query_id.isin(['Q01', 'Q16', 'Q19'])]
    begin = time.perf_counter()
    engine = SearchEngineV2(use_query_cache=False)
    initialization = time.perf_counter() - begin
    rows = []
    try:
        # Actual noncached query encoding and reranker warm-up; excluded explicitly.
        engine.search(selected['query'].iloc[0], 'RERANK_PRETRAINED')
        for q in selected.itertuples():
            reference = None
            for repeat in range(2):
                response = engine.search(q.query, 'RERANK_PRETRAINED')
                ids = [r['doc_id'] for r in response['results']]
                if reference is not None and reference != ids:
                    raise AssertionError('Repeated live rankings changed')
                reference = ids
                assert response['benchmark_query_cache'] is False
                rows.append(dict(query_id=q.query_id, method='RERANK_PRETRAINED', repeat=repeat,
                                 seconds=response['seconds'], results=len(ids), query_cache=False,
                                 reranker_pair_cache=False))
                print(q.query_id, repeat, round(response['seconds'], 3), 'seconds', flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output / 'live_latency_uncached.csv', index=False)
        write_json(output / 'live_latency_scope.json', dict(
            requests=len(rows), queries=selected.query_id.tolist(), repetitions=2,
            method='RERANK_PRETRAINED', initialization_seconds=initialization,
            mean_seconds=float(frame.seconds.mean()), median_seconds=float(frame.seconds.median()),
            max_seconds=float(frame.seconds.max()), source_sha256=digest(ROOT / 'src/search_v2.py'),
            scope='Direct local warm search requests: real BGE query encoding, dense and lexical retrieval, full uncached reranking and result formatting. Excludes engine/model loading and service queue. No simultaneous benchmark process.',
            limitation='Six requests on three development queries; not p95 population estimate, concurrent-user throughput, or evidence of retrieval quality.',
        ))
    finally:
        engine.close()
