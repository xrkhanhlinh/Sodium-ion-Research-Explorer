"""Start the real Streamlit HTTP server on loopback, probe it, clean up own PIDs.

Interactive data/model behavior is separately tested with Streamlit AppTest.
No persistent server or occupied port is left behind by this smoke test.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import psutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/final_system'


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT))
    args=parser.parse_args()
    output=Path(args.output)
    if not output.is_absolute():output=ROOT/output
    (output/'logs').mkdir(parents=True,exist_ok=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
    command=[sys.executable,'-m','streamlit','run',str(ROOT/'app.py'),
             '--server.headless=true','--server.address=127.0.0.1',f'--server.port={port}',
             '--browser.gatherUsageStats=false']
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             USE_TF='0',USE_FLAX='0',PYTHONIOENCODING='utf-8')
    result=dict(status='FAIL',url=f'http://127.0.0.1:{port}',command=command,
                scope='Real HTTP startup; separate AppTest covers interactive data/real V3 query')
    started=time.perf_counter()
    with (output/'logs/dashboard_server_smoke.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            while time.perf_counter()-started<40:
                if process.poll() is not None:raise RuntimeError('Streamlit exited before startup')
                try:
                    with urllib.request.urlopen(result['url'],timeout=2) as response:
                        html=response.read().decode('utf-8')
                        if response.status==200 and 'streamlit' in html.casefold():
                            result.update(status='PASS',http_status=200,seconds=time.perf_counter()-started)
                            break
                except (OSError,TimeoutError):pass
                time.sleep(.5)
            if result['status']!='PASS':raise RuntimeError('Streamlit HTTP startup timed out')
        finally:
            # Windows venv launchers can have an actual interpreter child.
            # Only descendants of this exact newly created process are owned.
            try: children=psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess: children=[]
            for child in reversed(children):
                try:
                    if 'streamlit' in child.cmdline():child.terminate()
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if process.poll() is None:process.terminate()
            process.wait(timeout=10)
            psutil.wait_procs(children,timeout=5)
            result['owned_processes_stopped']=all(not p.is_running() for p in children)
            (output/'dashboard_server_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert result['owned_processes_stopped']
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
