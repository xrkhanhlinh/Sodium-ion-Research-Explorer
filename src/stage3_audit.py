"""Check model-visible token identities across splits without loading BGE weights."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import hashlib
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from .stage3_common import ROOT,REPORT,DATA,config,write_json,csv

def main():
    cfg=config();pairs=pd.read_parquet(DATA/'contrastive_pairs.parquet')
    tokenizer=Tokenizer.from_file(str(ROOT/'models/bge_pretrained/tokenizer.json'))
    tokenizer.enable_truncation(max_length=cfg['max_seq_length'])
    summary=[];details=[]
    for field in ['title','abstract']:
        texts=pairs[field].tolist()
        if field=='title':texts=[cfg['query_prefix']+s for s in texts]
        hashes=[hashlib.sha256(np.asarray(e.ids,dtype='int32').tobytes()).hexdigest() for e in tokenizer.encode_batch(texts)]
        frame=pairs[['doc_id','split_group_id','split']].copy();frame['token_hash']=hashes
        counts=frame.groupby('token_hash').split.nunique();bad=set(counts[counts>1].index)
        subset=frame[frame.token_hash.isin(bad)]
        summary.append(dict(field=field,max_tokens=cfg['max_seq_length'],cross_split_identical_token_sequences=len(bad),affected_records=len(subset)))
        for row in subset.itertuples():details.append(dict(field=field,doc_id=row.doc_id,split_group_id=row.split_group_id,split=row.split,token_hash=row.token_hash))
    write_json(REPORT/'token_sequence_leakage_audit.json',summary)
    csv('cross_split_token_sequences.csv',pd.DataFrame(details,columns=['field','doc_id','split_group_id','split','token_hash']))
    assert all(row['cross_split_identical_token_sequences']==0 for row in summary),'Review model-visible cross-split duplicates before training'
    print('Model-visible split audit PASS',summary,flush=True)

if __name__=='__main__':main()
