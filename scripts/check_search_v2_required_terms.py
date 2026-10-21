"""Exploratory development checks of explicit filters, not quality metrics."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.search_v2 import ROOT, lexical_tokens, write_json
from src.search_v2_constraints import EvidenceSearchEngineV2


if __name__ == '__main__':
    output = ROOT / 'outputs/search_upgrade_20261018'
    queries = pd.read_parquet(ROOT / 'data/stage4/queries.parquet').set_index('query_id')
    engine = EvidenceSearchEngineV2(use_query_cache=True)
    checks = []
    try:
        for qid, terms in [('Q08', ['NaFSI', 'ether']), ('Q23', ['Na2FeP2O7'])]:
            response = engine.search(queries.loc[qid, 'query'], 'RERANK_PRETRAINED', required_terms=terms)
            for result in response['results']:
                text = set(lexical_tokens(result['title'] + ' ' + (result['abstract'] or '')))
                assert all(set(lexical_tokens(t)) <= text for t in terms)
            assert engine.evidence_mask is None
            write_json(output / f'{qid}_explicit_required_terms.json', response)
            checks.append(dict(query_id=qid, required_terms=terms, eligible=response['eligible_documents'],
                               results=len(response['results']), all_returned_documents_have_required_tokens=True,
                               relevance_verified=False))
            print(qid, 'eligible', response['eligible_documents'], flush=True)
            for r in response['results']:
                print(r['rank'], r['title'], flush=True)
        write_json(output / 'required_terms_checks.json', dict(
            status='PASS_TOKEN_PRESENCE_ONLY', cases=checks,
            development_only=True, no_relevance_grades_created=True,
            limitation='Explicit token filters are user constraints, not independent evidence that results satisfy all query aspects.',
        ))
    finally:
        engine.close()
