"""Run/resume Stage 3 phases in separate processes to release model memory."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
os.environ['PYTHONUTF8']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['NUMBA_NUM_THREADS']='2'
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')

def main():
    phases=['prepare','download','audit','baseline','train','after','embed','topics','figures','tests','complete']
    parser=argparse.ArgumentParser();parser.add_argument('--from-phase',choices=phases,default='prepare');parser.add_argument('--only',choices=phases)
    parser.add_argument('--wait-for-baseline',action='store_true',help='Wait up to 30 minutes for an already running baseline phase before continuing')
    args=parser.parse_args()
    (ROOT/'outputs/logs').mkdir(parents=True,exist_ok=True)
    if args.wait_for_baseline:
        deadline=time.monotonic()+1800
        print('Waiting for the existing pretrained baseline; no duplicate inference/training.',flush=True)
        while not (ROOT/'outputs/reports/stage3/baseline_success.json').exists():
            if time.monotonic()>deadline:raise RuntimeError('Baseline not complete after 30 minutes; preserve caches and inspect its log')
            time.sleep(10)
        time.sleep(2)
    for phase in ([args.only] if args.only else phases[phases.index(args.from_phase):]):
        if phase=='baseline' and (ROOT/'outputs/reports/stage3/baseline_success.json').exists():
            print('REUSE successful pretrained baseline',flush=True);continue
        modules={'prepare':'src.stage3_common','download':'src.stage3_download','audit':'src.stage3_audit','topics':'src.stage3_topics','figures':'src.stage3_figures','complete':'src.stage3_complete'}
        if phase in modules:command=[sys.executable,'-m',modules[phase]]
        elif phase=='tests':command=[sys.executable,'-m','pytest','tests/test_stage3_unit.py','tests/test_stage3_integration.py','-q','--junitxml=outputs/reports/stage3/tests.xml']
        else:command=[sys.executable,'-m','src.stage3_neural',phase]
        print('STAGE 3 PHASE',phase,flush=True)
        with (ROOT/'outputs/logs/stage3_execution.log').open('a',encoding='utf-8') as log:
            log.write('\nPHASE '+phase+'\n');log.flush()
            proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
            for line in proc.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=proc.wait()
        if code:
            from src.stage3_common import progress
            progress(f'Phase {phase} exited {code}; preserve successful outputs/checkpoints. Resume using python run_stage3.py --from-phase {phase}. No completion claim.',status='PARTIAL')
            raise SystemExit(code)

if __name__=='__main__':main()
