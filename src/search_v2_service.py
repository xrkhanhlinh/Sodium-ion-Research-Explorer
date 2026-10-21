"""Single-worker adapter for the current search methods."""
import atexit
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from .search_v2 import METHODS
from .search_v2_constraints import EvidenceSearchEngineV2


class SearchServiceV2:
    def __init__(self):
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='search-v2')
        self.engine = None
        self.closed = False
        atexit.register(self.close)

    def _search(self, query, method, source, years, required_terms=()):
        if method not in METHODS:
            raise ValueError('Unknown search method')
        if self.engine is None:
            self.engine = EvidenceSearchEngineV2()
        return self.engine.search(query, method, source, years, required_terms=required_terms)

    def search(self, query, method='RERANK_PRETRAINED', source='All', years=None, required_terms=()):
        return self.worker.submit(self._search, query, method, source, years, required_terms).result()

    def compare(self, query, methods, source='All', years=None, required_terms=()):
        if not methods or any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
            raise ValueError('Select distinct valid methods')
        def run():
            if self.engine is None:
                self.engine = EvidenceSearchEngineV2()
            with self.engine.reuse_query_work():
                responses = [self._search(query, m, source, years, required_terms) for m in methods]
                for response in responses:
                    response['retrieval_cache_scope'] = 'comparison_request'
                return responses
        return self.worker.submit(run).result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        def release():
            if self.engine is not None:
                self.engine.close()
        try:
            self.worker.submit(release).result()
        except RuntimeError:
            pass
        self.worker.shutdown(wait=True)


def rank_matrix(responses):
    rows = {}
    for response in responses:
        for result in response['results']:
            doc = result['doc_id']
            row = rows.setdefault(doc, dict(doc_id=doc, title=result['title'],
                                            source=result['source'], appearances=0))
            row['appearances'] += 1
            row[response['method']] = result['rank']
    return pd.DataFrame(sorted(rows.values(), key=lambda row: (-row['appearances'], row['doc_id'])))
