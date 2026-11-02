"""Profile one real long, filtered request without altering saved rankings."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cProfile
import pstats
import time
import json
import faulthandler
from src.final_search import FinalSearchEngine
from src import full_document_topics


def main():
    out=ROOT/'outputs/batterybert_full_document_20261024_v1'
    assigned=full_document_topics.assignments()
    subset=assigned[assigned.source.eq('Publication')&assigned.year.between(2020,2023)]
    selected=int(subset.groupby('topic_id').size().idxmax())
    query='Sodium ion battery electrodes and energy storage. '*40+'Hard carbon anodes from biomass. '*20
    engine=FinalSearchEngine();profile=cProfile.Profile()
    faulthandler.dump_traceback_later(45,repeat=True)
    try:
        tick=time.perf_counter()
        response=profile.runcall(engine.search,query,'BATTERYBERT_FULLDOC_HYBRID',
            'Publication',(2020,2023),['sodium'],10,selected)
        elapsed=time.perf_counter()-tick
        profile.dump_stats(str(out/'long_filtered_query.prof'))
        with (out/'long_filtered_query_profile.txt').open('w',encoding='utf8') as stream:
            pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(30)
        (out/'long_filtered_query_timing.json').write_text(json.dumps(dict(seconds=elapsed,
            eligible_documents=response['eligible_documents'],query_coverage=response['query_coverage'],
            results=len(response['results']),scope='Cold engine, cProfile overhead; not benchmark latency'),indent=2),encoding='utf8')
        print((out/'long_filtered_query_profile.txt').read_text(encoding='utf8'),flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later();engine.close()


if __name__=='__main__':main()
