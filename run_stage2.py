"""Run the saved-corpus LDA experiment; separate from application inference."""
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'
from src.stage2 import main

if __name__=='__main__':
    main()
