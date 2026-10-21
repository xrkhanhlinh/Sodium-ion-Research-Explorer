"""CPU-only worker; writes synthetic artifacts, never invokes a remote API."""
import pandas as pd
import atexit
import sys
from src.ai_judge import DEFAULT_OUTPUT, read_json, write_json, utc_now
from src.ai_synthetic_benchmark import retrieve_synthetic
from scripts.run_retrieval_ai_evaluation import synthetic_generations


def main():
    output=DEFAULT_OUTPUT
    if sys.platform=='win32':
        import msvcrt
        lock=(output/'synthetic/.worker.lock').open('a+b')
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Synthetic CPU worker already running')
        atexit.register(lock.close)
    status=output/'synthetic/worker_status.json'
    write_json(status,{'status':'RUNNING','started_at':utc_now()})
    try:
        selected=pd.DataFrame(read_json(output/'synthetic/selection.json'))
        retrieve_synthetic(output,selected,synthetic_generations(output))
    except Exception as exc:
        write_json(status,{'status':'FAILED','error_type':type(exc).__name__,'message':str(exc),'ended_at':utc_now()})
        raise
    write_json(status,{'status':'COMPLETE','ended_at':utc_now()})


if __name__=='__main__':main()
