"""Execute each phase in a fresh process to fit the 8 GB CPU-only machine."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime
from src.stage3_common import write_json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true',help='Explicitly resume only this output folder')
    args = parser.parse_args()
    out = (args.output or ROOT / 'outputs/bertopic_chunked').resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        if args.output:
            parser.error('Existing output folder: use --resume or select a new directory')
        out = ROOT / ('outputs/bertopic_chunked_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    assert out.is_relative_to((ROOT / 'outputs').resolve())
    out.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    for phase in ['prepare','preflight','embed','fit','evaluate','audit','figures','verify']:
        # prepare requires an empty folder on its first run.
        if phase != 'prepare': write_json(out / 'run_status.json',dict(status='RUNNING',phase=phase))
        print(f'PHASE {phase}: {out}',flush=True)
        # Keep even prepare's log without populating its initially empty directory.
        with tempfile.TemporaryFile(mode='w+',encoding='utf-8') as capture:
            process = subprocess.Popen([sys.executable,'-u','-m','src.bertopic_chunked',phase,'--output',str(out)],
                cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                print(line,end='',flush=True)
                capture.write(line)
            code = process.wait()
            capture.seek(0)
            with (out / f'{phase}.log').open('a',encoding='utf-8') as logfile:
                logfile.write(capture.read())
        if code:
            write_json(out / 'run_status.json',dict(status='FAILED',phase=phase,exit_code=code,
                resume=f'python run_bertopic_chunked.py --output "{out}" --resume'))
            return code
    return 0


if __name__ == '__main__':
    sys.exit(main())
