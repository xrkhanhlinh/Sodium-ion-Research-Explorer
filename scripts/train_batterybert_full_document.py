"""Versioned CPU runner for full available-text BatteryBERT continuation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='4'
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from src.batterybert_full_document import prepare,training,export_and_embed,smoke_train,OUT

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','smoke','pilot','train','embed','all']);parser.add_argument('--output',default=str(OUT))
    args=parser.parse_args();out=Path(args.output)
    if args.phase in ('prepare','all'):prepare(out)
    if args.phase=='smoke':smoke_train(out)
    if args.phase in ('pilot','all'):training(out,'pilot')
    if args.phase in ('train','all'):training(out,'full')
    if args.phase in ('embed','all'):export_and_embed(out)
