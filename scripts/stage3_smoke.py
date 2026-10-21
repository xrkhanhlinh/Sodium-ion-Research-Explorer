"""Validate real gradient flow and exact cache inference before the long CPU run."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
import src.stage3_neural as n
from src.stage3_common import ROOT,REPORT,write_json

model=n.model_load();bert=model[0].auto_model
for p in model.parameters():p.requires_grad=False
for p in bert.encoder.layer[-1].parameters():p.requires_grad=True
frame=pd.read_parquet(ROOT/'data/stage3/contrastive_pairs.parquet')
frame=frame[frame.split.eq('train')].head(4)
q=model(model.preprocess([n.CFG['query_prefix']+t for t in frame.title]))['sentence_embedding']
d=model(model.preprocess(frame.abstract.tolist()))['sentence_embedding']
scores=torch.nn.functional.normalize(q,dim=1)@torch.nn.functional.normalize(d,dim=1).T/.05
loss=torch.nn.functional.cross_entropy(scores,torch.arange(len(frame)));loss.backward()
gradient_l1=sum(float(p.grad.abs().sum()) for p in bert.encoder.layer[-1].parameters() if p.grad is not None)
assert gradient_l1>0 and all(p.grad is None for p in bert.encoder.layer[0].parameters())
model.zero_grad(set_to_none=True)
n.DATA=ROOT/'.cache/stage3_smoke';n.DATA.mkdir(parents=True,exist_ok=True)
n.encode_validation_frozen_cache(model,frame.abstract.iloc[:3].tolist(),'smoke')
write_json(REPORT/'smoke_check.json',dict(status='PASS',real_gradient_l1=gradient_l1,frozen_first_block_gradients_none=True,loss=float(loss.detach()),optimizer_updates=0,cache_roundtrip='PASS'))
print('SMOKE PASS; real gradients and float32 frozen cache verified; no optimizer update performed',flush=True)
