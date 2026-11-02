"""Read actual complete Title-Abstract samples for the final qualitative review."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/batterybert_full_document_20261024_v1/topics'
parser=argparse.ArgumentParser();parser.add_argument('--topics',type=int,nargs='+',required=True)
args=parser.parse_args()
frame=pd.read_csv(OUT/'representative_documents.csv',keep_default_na=False)
for row in frame[frame.topic_id.isin(args.topics)].to_dict('records'):
    print(json.dumps(row,ensure_ascii=False),flush=True)
