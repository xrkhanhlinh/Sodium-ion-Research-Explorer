"""One-command local demo. No preprocessing, training or Stage 4 execution."""
import os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONIOENCODING='utf-8',PYTHONUTF8='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    return subprocess.call([sys.executable,'-m','streamlit','run',str(ROOT/'app.py'),*sys.argv[1:]],cwd=ROOT,env=env)
if __name__=='__main__':raise SystemExit(main())
