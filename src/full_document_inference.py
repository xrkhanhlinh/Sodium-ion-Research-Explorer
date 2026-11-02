"""Versioned full-text BatteryBERT query inference, with no 128-token cutoff."""
from pathlib import Path
import json
import os
import numpy as np

os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/batterybert_full_document_20261024_v1'
METHODS=('BATTERYBERT_FULLDOC_SEMANTIC','BATTERYBERT_FULLDOC_HYBRID')


def ready():
    marker=OUT/'deployment.json'
    return marker.exists() and json.loads(marker.read_text(encoding='utf8')).get('activate_search',False)


def query_windows(tokenizer,text,capacity=256,overlap=64):
    """Native-token windows cover every query token, without decode/retokenize."""
    # The complete sequence is only tokenized here, then split before BERT.
    tokens=tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
    payload=capacity-tokenizer.num_special_tokens_to_add(pair=False)
    if not tokens:raise ValueError('Query has no tokenizer content')
    if payload<=overlap:raise ValueError('Window payload must exceed overlap')
    rows=[];start=0;previous_end=0
    while start<len(tokens):
        end=min(start+payload,len(tokens))
        ids=[tokenizer.cls_token_id]+[int(t) for t in tokens[start:end]]+[tokenizer.sep_token_id]
        rows.append(dict(input_ids=ids,token_type_ids=[0]*len(ids),token_start=start,token_end=end,
            unique_tokens=end-max(start,previous_end)))
        previous_end=end
        if end==len(tokens):break
        start=end-overlap
    if sum(r['unique_tokens'] for r in rows)!=len(tokens):raise ValueError('Query token coverage mismatch')
    return rows,len(tokens)


class FullDocumentEncoder:
    def __init__(self,out=OUT):
        import torch
        from transformers import AutoTokenizer,AutoModel
        from .final_system_artifacts import sha
        self.out=Path(out);self.info=json.loads((self.out/'embedding_success.json').read_text(encoding='utf8'))
        if sha(self.out/'model/model.safetensors')!=self.info['weights_sha256']:
            raise ValueError('Full-document query checkpoint differs from corpus checkpoint')
        for filename,digest in self.info['tokenizer_files'].items():
            if sha(self.out/'model'/filename)!=digest:raise ValueError('Full-document tokenizer/config mismatch')
        torch.set_num_threads(4)
        self.tokenizer=AutoTokenizer.from_pretrained(self.out/'model',local_files_only=True,use_fast=True)
        self.model=AutoModel.from_pretrained(self.out/'model',local_files_only=True,add_pooling_layer=False,trust_remote_code=False).eval()

    def encode(self,text):
        import torch
        from torch.nn import functional as F
        records,count=query_windows(self.tokenizer,text)
        vectors=[];weights=[]
        with torch.inference_mode():
            for at in range(0,len(records),2):
                group=records[at:at+2]
                rows=[{k:r[k] for k in ('input_ids','token_type_ids')} for r in group]
                features=self.tokenizer.pad(rows,padding=True,return_tensors='pt')
                hidden=self.model(**features).last_hidden_state
                mask=features['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                vectors.append(F.normalize((hidden*mask).sum(1)/mask.sum(1),dim=1))
                weights.extend(r['unique_tokens'] for r in group)
        matrix=torch.cat(vectors);w=torch.tensor(weights,dtype=matrix.dtype)
        pooled=F.normalize((matrix*w[:,None]).sum(0)/w.sum(),dim=0).numpy().astype(np.float32)
        return pooled,dict(tokens=count,chunks=len(records),truncated=False,coverage=1.,checkpoint='batterybert_full_document_20261024_v1')

    def encode_document(self,title,body):
        """Encode every available body token, including text beyond an abstract.

        Training data scope stays Title-Abstract; this API does not claim that
        inference on a supplied full article has independent quality validation.
        """
        import torch
        from torch.nn import functional as F
        from .bertopic_chunked import chunk_document
        if not isinstance(title,str) or not isinstance(body,str):raise ValueError('Title and body must be text')
        if not title.strip() and not body.strip():raise ValueError('Document has no text')
        rows,stats=chunk_document(self.tokenizer,title,body)
        values=[];weights=[];previous=0
        with torch.inference_mode():
            for at in range(0,len(rows),2):
                group=rows[at:at+2]
                features=self.tokenizer.pad([{k:[int(v) for v in r[k]] for k in ('input_ids','token_type_ids')} for r in group],padding=True,return_tensors='pt')
                hidden=self.model(**features).last_hidden_state
                mask=(features['token_type_ids']==1)&(features['input_ids']!=self.tokenizer.sep_token_id)&features['attention_mask'].bool()
                for j in range(len(mask)):
                    if not mask[j].any():mask[j]=features['attention_mask'][j].bool()&(features['input_ids'][j]!=self.tokenizer.cls_token_id)&(features['input_ids'][j]!=self.tokenizer.sep_token_id)
                weight=mask.unsqueeze(-1).to(hidden.dtype)
                values.append(F.normalize((hidden*weight).sum(1)/weight.sum(1),dim=1))
                for row in group:
                    weights.append(row['token_end']-max(previous,row['token_start']));previous=row['token_end']
            vectors=torch.cat(values);w=torch.tensor(weights,dtype=vectors.dtype)
            pooled=F.normalize((vectors*w[:,None]).sum(0)/w.sum(),dim=0).numpy().astype(np.float32)
        if stats['covered_abstract_tokens']!=stats['abstract_tokens']:raise ValueError('Incomplete document body coverage')
        return pooled,dict(body_tokens=stats['abstract_tokens'],covered_body_tokens=stats['covered_abstract_tokens'],
            chunks=len(rows),body_truncated=False,title_context_truncated=stats['title_truncated'],
            training_scope='Title-Abstract; full-article inference quality not independently validated')
