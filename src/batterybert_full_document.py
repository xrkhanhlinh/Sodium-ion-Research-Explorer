"""Full-document last-block training with exact two-level gradient caching.

Every document chunk contributes to the representation and encoder gradient.
First eleven BERT blocks remain frozen; the original cache is read-only.
Title context in document chunks is explicit: the title-pair evaluation is a
proxy and must not be described as independent query relevance accuracy.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '4')
os.environ.update(USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import gc
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .model_comparison_io import write_json, replace_with_retry

ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'outputs/model_comparison_v2/batterybert_pretrained'
OUT=ROOT/'outputs/batterybert_full_document_20261024_v1'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def log(message):print(time.strftime('%H:%M:%S'),message,flush=True)


def save_torch(path,value):
    temporary=Path(path).with_suffix('.pending')
    torch.save(value,temporary);replace_with_retry(temporary,path)


def configuration():
    import yaml
    return yaml.safe_load((ROOT/'configs/batterybert_full_document.yaml').read_text(encoding='utf8'))


def normal_pool(values,weights=None):
    if weights is None:weights=torch.ones(len(values),device=values.device,dtype=values.dtype)
    if values.ndim!=2 or weights.shape!=(len(values),) or torch.any(weights<=0):
        raise ValueError('Positive weight required for every document chunk')
    return F.normalize((values*weights[:,None]).sum(0)/weights.sum(),dim=0)


def unique_token_weights(records):
    end=0;weights=[]
    for row in records:
        start,stop=int(row['token_start']),int(row['token_end'])
        if start>end or stop<=end:raise ValueError('Incomplete or non-progressing document coverage')
        weights.append(stop-max(end,start));end=stop
    return np.asarray(weights,dtype=np.float32)


def stratified_batches(indices,sources,batch_size,seed):
    """One appearance per train row, source composition spread across batches."""
    indices=np.asarray(indices);sources=np.asarray(sources)
    rng=np.random.default_rng(seed)
    order=[];positions=[]
    for source in np.unique(sources[indices]):
        part=rng.permutation(indices[sources[indices]==source])
        order.extend(part);positions.extend((np.arange(len(part))+.5)/len(part))
    ordered=np.asarray(order)[np.argsort(positions,kind='stable')]
    batches=[rng.permutation(ordered[at:at+batch_size]) for at in range(0,len(ordered),batch_size)]
    if len(batches[-1])==1:batches[-2]=np.concatenate([batches[-2],batches.pop()])
    if min(map(len,batches))<2:raise ValueError('Contrastive batch needs at least two documents')
    if sorted(np.concatenate(batches).tolist())!=sorted(indices.tolist()):raise ValueError('Training sampler lost/duplicated documents')
    return batches


def symmetric_loss(q,d,mask,temperature):
    scores=q@d.T/temperature
    scores=scores.masked_fill(torch.as_tensor(mask,dtype=torch.bool,device=scores.device),-1e4)
    target=torch.arange(len(q),device=scores.device)
    return (F.cross_entropy(scores,target)+F.cross_entropy(scores.T,target))/2


def embedding_gradients(q,d,mask,temperature,reference=None,distillation_weight=0.):
    """Differentiate the entire contrastive batch, independent of microbatching."""
    q=q.detach().requires_grad_(True);d=d.detach().requires_grad_(True)
    loss=symmetric_loss(q,d,mask,temperature)
    if reference is not None and distillation_weight:
        old_q,old_d=reference
        loss=loss+distillation_weight*((q-old_q).square().sum(1).mean()+(d-old_d).square().sum(1).mean())
    if not torch.isfinite(loss):raise ValueError('Nonfinite contrastive objective')
    loss.backward()
    return float(loss.detach()),q.grad.detach(),d.grad.detach()


def pooling_gradients(chunks,weights,gradient):
    chunks=chunks.detach().requires_grad_(True)
    if weights.is_inference():weights=weights.clone()
    normal_pool(chunks,weights).backward(gradient)
    return chunks.grad.detach()


class LastBlock(torch.nn.Module):
    def __init__(self,folder):
        super().__init__()
        from transformers import BertConfig
        from transformers.models.bert.modeling_bert import BertLayer
        config=BertConfig.from_pretrained(folder,local_files_only=True)
        config._attn_implementation='sdpa'
        self.layer=BertLayer(config,layer_idx=config.num_hidden_layers-1)
        self.eval()

    def forward(self,hidden,attention,pooling):
        # Exact deterministic replay: eval disables dropout but retains gradients.
        if self.training:raise ValueError('Gradient-cache replay requires dropout disabled')
        mask=(1-attention[:,None,None,:].to(hidden.dtype))*torch.finfo(hidden.dtype).min
        value=self.layer(hidden,attention_mask=mask)
        if isinstance(value,tuple):value=value[0]
        weight=pooling.unsqueeze(-1).to(value.dtype)
        if torch.any(weight.sum(1)<=0):raise ValueError('Cannot pool an empty document/query')
        return F.normalize((value*weight).sum(1)/weight.sum(1),dim=1)


class FeatureStore:
    def __init__(self,out=OUT):
        self.out=Path(out);self.cache=self.out/'features'
        self.main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
        self.splits=pd.read_parquet(ROOT/'data/stage3/split_assignments.parquet').set_index('doc_id').loc[self.main.doc_id]
        if self.splits.groupby('split_group_id').split.nunique().max()!=1:raise ValueError('Split-group leakage')
        self.chunks=pd.read_parquet(LEGACY/'chunks.parquet')
        self.docs=pd.read_parquet(LEGACY/'document_chunk_counts.parquet')
        if self.docs.doc_id.tolist()!=self.main.doc_id.tolist():raise ValueError('Document cache order mismatch')
        self.old_hidden=np.load(LEGACY/'frozen_document_prefix/hidden.npy',mmap_mode='r')
        self.old_valid=np.load(LEGACY/'frozen_document_prefix/valid.npy',mmap_mode='r')
        self.old_hashes=np.load(LEGACY/'frozen_document_prefix/row_sha256.npy',mmap_mode='r')
        self.queries=pd.read_parquet(self.cache/'query_records.parquet')
        if self.queries.doc_id.tolist()!=self.main.doc_id.tolist():raise ValueError('Query prefix order mismatch')
        if sha(LEGACY/'frozen_document_prefix/valid.npy')!=read(self.cache/'identity.json')['old_valid_sha256']:
            raise ValueError('Legacy cache validity map changed since preparation')
        self.query_hidden=np.load(self.cache/'query_hidden.npy',mmap_mode='r')
        self.query_valid=np.load(self.cache/'query_valid.npy',mmap_mode='r')
        self.query_hashes=np.load(self.cache/'query_sha256.npy',mmap_mode='r')
        self.extra=pd.read_parquet(self.cache/'extra_records.parquet').set_index('chunk_index')
        self.extra_hidden=np.load(self.cache/'extra_hidden.npy',mmap_mode='r')
        self.extra_valid=np.load(self.cache/'extra_valid.npy',mmap_mode='r')
        self.extra_hashes=np.load(self.cache/'extra_sha256.npy',mmap_mode='r')
        self.checked=set()
        self.doc_starts=self.docs.chunk_start.to_numpy(dtype=np.int64)
        self.doc_counts=self.docs.chunk_count.to_numpy(dtype=np.int64)
        self.chunk_stops=self.chunks.token_end.to_numpy(dtype=np.int64)
        self.chunk_lengths=self.chunks.input_token_count.to_numpy(dtype=np.int64)
        self.weight_cache={}

    @staticmethod
    def verify(array,expected):
        if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest().encode('ascii')!=expected:
            raise ValueError('Frozen-prefix SHA-256 mismatch')
        if not np.isfinite(array).all():raise ValueError('Nonfinite frozen prefix')

    def get(self,kind,index):
        key=(kind,int(index))
        if kind=='query':
            row=self.queries.iloc[index];start=int(row.offset);end=start+int(row.length)
            if not self.query_valid[index]:raise ValueError('Incomplete query prefix cache')
            raw=self.query_hidden[start:end]
            if key not in self.checked:self.verify(raw,self.query_hashes[index])
            pooling=np.ones(end-start,dtype=np.int64)
        else:
            record=self.chunks.iloc[index];length=len(record.input_ids)
            if self.old_valid[index]:
                raw=self.old_hidden[index]
                if key not in self.checked:self.verify(raw,self.old_hashes[index])
                raw=raw[:length]
            else:
                row=self.extra.loc[index];start=int(row.offset);end=start+int(row.length);slot=int(row.slot)
                if not self.extra_valid[slot]:raise ValueError('Incomplete supplementary prefix cache')
                raw=self.extra_hidden[start:end]
                if key not in self.checked:self.verify(raw,self.extra_hashes[slot])
            # Pool the abstract segment, not the repeated title or trailing SEP.
            types=np.asarray(record.token_type_ids)
            pooling=(types==1).astype(np.int64)
            pooling[-1]=0
            if not pooling.any():
                pooling=np.ones(length,dtype=np.int64);pooling[[0,-1]]=0
        self.checked.add(key)
        return np.array(raw,copy=True),pooling

    def packed(self,kind,indices):
        rows=[self.get(kind,int(i)) for i in indices];width=max(len(h) for h,p in rows)
        hidden=np.zeros((len(rows),width,768),dtype=np.float32)
        attention=np.zeros((len(rows),width),dtype=np.int64);pool=attention.copy()
        for j,(h,p) in enumerate(rows):hidden[j,:len(h)]=h;attention[j,:len(h)]=1;pool[j,:len(h)]=p
        return tuple(torch.from_numpy(t) for t in (hidden,attention,pool))

    def chunk_ids(self,doc):
        start=int(self.doc_starts[doc]);return list(range(start,start+int(self.doc_counts[doc])))

    def weights(self,doc):
        if doc not in self.weight_cache:
            ends=self.chunk_stops[self.chunk_ids(doc)]
            weights=np.diff(np.r_[0,ends]).astype(np.float32)
            if np.any(weights<=0):raise ValueError('Non-progressing full-document chunks')
            # Validation may populate this cache under inference_mode; weights
            # are reused later by autograd replay and must be normal tensors.
            with torch.inference_mode(False):self.weight_cache[doc]=torch.from_numpy(weights)
        return self.weight_cache[doc]

    def document_chunks(self,block,doc,microbatch=2):
        ids=self.chunk_ids(doc);vectors=[]
        for at in range(0,len(ids),microbatch):vectors.append(block(*self.packed('chunk',ids[at:at+microbatch])))
        return torch.cat(vectors)

    def representations(self,block,indices,microbatch=2,include_queries=True):
        # The no-grad pass may use a wider microbatch than gradient replay.
        # Bucket chunks by length to avoid wasting work on padding.
        q=[];d=[];chunk_values={};forward_batch=8
        with torch.no_grad():
            if include_queries:
                for at in range(0,len(indices),forward_batch):q.append(block(*self.packed('query',indices[at:at+forward_batch])))
            flat=[j for i in indices for j in self.chunk_ids(int(i))]
            ordered=sorted(flat,key=lambda j:self.chunk_lengths[j]);all_values={}
            for at in range(0,len(ordered),forward_batch):
                batch=ordered[at:at+forward_batch];value=block(*self.packed('chunk',batch))
                all_values.update({j:value[k] for k,j in enumerate(batch)})
            for i in indices:
                value=torch.stack([all_values[j] for j in self.chunk_ids(int(i))])
                d.append(normal_pool(value,self.weights(int(i))));chunk_values[int(i)]=value
        return torch.cat(q) if q else None,torch.stack(d),chunk_values


def prepare(out=OUT):
    """Create only missing prefixes; never write the legacy 25GB memmap."""
    from .batterybert import BatteryEncoder,tensor_hash
    out=Path(out);cache=out/'features';cache.mkdir(parents=True,exist_ok=True)
    cfg=configuration();torch.set_num_threads(cfg['threads'])
    source=ROOT/cfg['initial_checkpoint'];adapter=BatteryEncoder(source)
    frozen=[(n,p) for n,p in adapter.model.named_parameters() if not n.startswith('encoder.layer.11.')]
    frozen_hash=tensor_hash(frozen)
    original=read(LEGACY/'frozen_document_prefix/manifest.json')
    if frozen_hash!=original['frozen_hash']:raise ValueError('Initial checkpoint differs from frozen cached backbone')
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    chunks=pd.read_parquet(LEGACY/'chunks.parquet');docs=pd.read_parquet(LEGACY/'document_chunk_counts.parquet')
    if sha(LEGACY/'chunks.parquet')!=original['chunks_sha256']:raise ValueError('Legacy chunk file changed')
    if docs.doc_id.tolist()!=main.doc_id.tolist() or not np.array_equal(docs.abstract_tokens,docs.covered_abstract_tokens):
        raise ValueError('Coverage/order contract failed')
    # Longest title has <256 tokens in this corpus. No truncation is requested.
    records=[];offset=0
    for row in main.itertuples():
        tokens=adapter.tokenizer(row.title,add_special_tokens=True,truncation=False)
        if len(tokens['input_ids'])>adapter.maximum:raise ValueError('Long title requires its own windowing; never truncate')
        records.append(dict(doc_id=row.doc_id,offset=offset,length=len(tokens['input_ids']),**tokens));offset+=len(tokens['input_ids'])
    query=pd.DataFrame(records);query.to_parquet(cache/'query_records.parquet',index=False)
    old_valid=np.load(LEGACY/'frozen_document_prefix/valid.npy',mmap_mode='r')
    extra=[];at=0
    for slot,index in enumerate(np.flatnonzero(~old_valid)):
        length=len(chunks.iloc[index].input_ids)
        extra.append(dict(chunk_index=int(index),slot=slot,offset=at,length=length));at+=length
    extra=pd.DataFrame(extra);extra.to_parquet(cache/'extra_records.parquet',index=False)
    identity=dict(corpus=sha(ROOT/'data/processed/main_corpus.parquet'),chunks=sha(LEGACY/'chunks.parquet'),
        frozen_hash=frozen_hash,query_tokens=signature([{k:list(map(int,r[k])) for k in ('input_ids','token_type_ids')} for r in records]),
        old_valid_sha256=sha(LEGACY/'frozen_document_prefix/valid.npy'),precision='float32',
        qshape=[offset,768],extra_shape=[at,768])
    marker=cache/'identity.json'
    if marker.exists() and read(marker)!=identity:raise ValueError('Frozen cache inputs differ: use new experiment directory')
    if not marker.exists():write_json(marker,identity)
    def arrays(name,shape,count):
        exists=(cache/f'{name}_hidden.npy').exists()
        hidden=np.lib.format.open_memmap(cache/f'{name}_hidden.npy',mode='r+' if exists else 'w+',dtype='float32',shape=tuple(shape))
        valid=np.lib.format.open_memmap(cache/f'{name}_valid.npy',mode='r+' if exists else 'w+',dtype='bool',shape=(count,))
        hashes=np.lib.format.open_memmap(cache/f'{name}_sha256.npy',mode='r+' if exists else 'w+',dtype='S64',shape=(count,))
        if not exists:valid[:]=False;valid.flush()
        return hidden,valid,hashes
    qh,qv,qs=arrays('query',identity['qshape'],len(query));eh,ev,es=arrays('extra',identity['extra_shape'],len(extra))
    old_queries=pd.read_parquet(ROOT/'data/stage3/contrastive_pairs.parquet')
    old_queries=old_queries[old_queries.split.eq('validation')].reset_index(drop=True)
    old_query_map={row.doc_id:i for i,row in old_queries.iterrows()}
    old_qh=np.load(LEGACY/'validation_frozen/titles.npy',mmap_mode='r')
    old_qmask=np.load(LEGACY/'validation_frozen/titles_mask.npy',mmap_mode='r')
    initial=LastBlock(source);initial.layer.load_state_dict(adapter.model.encoder.layer[-1].state_dict());initial.eval()
    save_torch(out/'initial_last_block.pt',initial.state_dict())
    parity=read(out/'forward_parity.json')['checks'] if (out/'forward_parity.json').exists() else []
    with torch.inference_mode():
        for kind,frame,hidden,valid,hashes in [('query',query,qh,qv,qs),('extra',extra,eh,ev,es)]:
            pending=[];started=time.monotonic()
            def write_row(index,value):
                row=frame.iloc[index];start=int(row.offset);end=start+int(row.length)
                hidden[start:end]=value[:int(row.length)]
                hashes[index]=hashlib.sha256(np.ascontiguousarray(hidden[start:end]).tobytes()).hexdigest().encode('ascii');pending.append(index)
            # Reuse complete, nontruncated validation-title prefixes first.
            if kind=='query':
                for index,row in enumerate(frame.itertuples()):
                    if not valid[index] and row.doc_id in old_query_map:
                        old_index=old_query_map[row.doc_id]
                        if int(old_qmask[old_index].sum())!=row.length:raise ValueError('Validation title cache truncated')
                        write_row(index,np.array(old_qh[old_index,:row.length],copy=True))
                if pending:hidden.flush();hashes.flush();valid[pending]=True;valid.flush();pending=[]
            remaining=np.flatnonzero(~valid).tolist()
            batch_size=8 if kind=='query' else 2
            for at_index in range(0,len(remaining),batch_size):
                batch_indices=remaining[at_index:at_index+batch_size]
                selected_records=[records[i] if kind=='query' else chunks.iloc[int(frame.iloc[i].chunk_index)].to_dict() for i in batch_indices]
                features=adapter.record_features(selected_records);values=adapter.prefix(features)
                if not parity or kind not in {r['kind'] for r in parity}:
                    att=features['attention_mask']
                    actual=initial(values,att,att);expected=adapter.from_prefix(values,att)
                    error=float((actual-expected).abs().max())
                    if error>2e-5:raise ValueError('Standalone BERT final block parity failed')
                    parity.append(dict(kind=kind,max_error=error,padded_batch=len(batch_indices)))
                    write_json(out/'forward_parity.json',dict(status='PASS',checks=parity))
                for j,index in enumerate(batch_indices):write_row(index,values[j].numpy())
                if len(pending)>=64 or at_index+batch_size>=len(remaining):
                    hidden.flush();hashes.flush();valid[pending]=True;valid.flush();pending=[]
                    write_json(cache/'progress.json',dict(phase=kind,complete=int(valid.sum()),total=len(frame)))
                    log(f'PREFIX {kind}: {int(valid.sum())}/{len(frame)}; elapsed={time.monotonic()-started:.1f}s')
    adapter.tokenizer.save_pretrained(out/'tokenizer')
    adapter.model.config.save_pretrained(out/'tokenizer')
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/processed/main_corpus.parquet',ROOT/'data/stage3/split_assignments.parquet',
        LEGACY/'chunks.parquet',LEGACY/'document_chunk_counts.parquet',source/'model.safetensors']}
    write_json(out/'input_manifest.json',inputs)
    write_json(out/'coverage_audit.json',dict(data_scope=cfg['data_scope'],documents=len(main),
        title_tokens_covered=int(query.length.sum()),query_truncation=False,maximum_title_sequence=int(query.length.max()),
        abstract_tokens=int(docs.abstract_tokens.sum()),covered_abstract_tokens=int(docs.covered_abstract_tokens.sum()),
        chunk_count=len(chunks),maximum_chunk=256,old_cached_chunks=int(old_valid.sum()),supplementary_chunks=len(extra),
        first_11_blocks_frozen=True,frozen_hash=frozen_hash,precision='float32',
        caveat='Title context is in each document chunk. Paired-title metrics are a proxy, not independent search quality.'))
    del adapter,initial;gc.collect();log('PREFIX PREPARATION COMPLETE')


def train_step(block,store,indices,optimizer,cfg,reference_block=None):
    from .batterybert import false_negative_mask
    block.eval();optimizer.zero_grad(set_to_none=True)
    q,d,chunk_values=store.representations(block,indices,cfg['microbatch_chunks'])
    reference=None
    if reference_block is not None:
        rq,rd,_=store.representations(reference_block,indices,cfg['microbatch_chunks']);reference=(rq,rd)
    mask=false_negative_mask(store.main.iloc[indices],cfg['false_negative_title_jaccard'])
    loss,gq,gd=embedding_gradients(q,d,mask,cfg['temperature'],reference,cfg['distillation_weight'])
    bs=cfg['microbatch_chunks'];query_batch=8
    for at in range(0,len(indices),query_batch):
        pred=block(*store.packed('query',indices[at:at+query_batch]));pred.backward(gq[at:at+query_batch])
    chunks_seen=0;payload=0
    for j,i in enumerate(indices):
        i=int(i);chunk_grad=pooling_gradients(chunk_values[i],store.weights(i),gd[j]);ids=store.chunk_ids(i)
        for at in range(0,len(ids),bs):
            block(*store.packed('chunk',ids[at:at+bs])).backward(chunk_grad[at:at+bs]);chunks_seen+=len(ids[at:at+bs])
        payload+=int(store.docs.iloc[i].abstract_tokens)
    norm=float(torch.nn.utils.clip_grad_norm_(block.parameters(),cfg['max_grad_norm']))
    if not np.isfinite(norm):raise ValueError('Nonfinite encoder gradient')
    optimizer.step()
    return dict(loss=loss,gradient_norm=norm,pairs=len(indices),negatives_pool=len(indices)-1,
        chunks_backpropagated=chunks_seen,abstract_tokens_covered=payload,masked_pairs=int(mask.sum()))


def evaluate(block,store,split,dest):
    from .stage3_common import group_retrieval_metrics
    ids=np.flatnonzero(store.splits.split.to_numpy()==split)
    q=[];d=[];started=time.monotonic()
    with torch.inference_mode():
        for at in range(0,len(ids),16):
            qq,dd,_=store.representations(block,ids[at:at+16]);q.append(qq.numpy());d.append(dd.numpy())
            if at%256==0:log(f'EVAL {split} {at}/{len(ids)}')
    q=np.concatenate(q);d=np.concatenate(d)
    m,ranks=group_retrieval_metrics(q,d,store.main.iloc[ids].split_group_id.to_numpy(),store.main.iloc[ids].doc_id.to_numpy())
    m.update(split=split,seconds=time.monotonic()-started,protocol='paired title -> title-conditioned full-document proxy',
        query_truncated=False,full_abstract_coverage=True,document_pooling=configuration()['document_aggregation'],
        independent_relevance_accuracy=False)
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);write_json(dest/'metrics.json',m)
    np.save(dest/'query_vectors.npy',q);np.save(dest/'document_vectors.npy',d);ranks.to_csv(dest/'ranks.csv',index=False)
    log(f'EVAL {split}: MRR@10={m["group_mrr_at_10"]:.6f}')
    return m


def training(out=OUT,phase='pilot'):
    out=Path(out);cfg=configuration();torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    if cfg['epochs']!=1 or cfg['trainable_blocks']!=1 or cfg['chunk_tokens']!=256 or cfg['overlap']!=64:
        raise ValueError('This validated cache/trainer supports one train pass, one block and 256/64 windows only')
    store=FeatureStore(out);indices=np.flatnonzero(store.splits.split.to_numpy()=='train')
    if not store.query_valid.all() or not store.extra_valid.all():raise ValueError('Complete feature preparation before training')
    batches=stratified_batches(indices,store.main.source.to_numpy(),cfg['effective_batch_pairs'],cfg['seed'])
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','pandas')}
    snapshot=out/'code_snapshot';snapshot.mkdir(exist_ok=True)
    source_files=[Path(__file__),ROOT/'scripts/train_batterybert_full_document.py',ROOT/'configs/batterybert_full_document.yaml']
    protocol=dict(config=cfg,versions=versions,inputs=read(out/'input_manifest.json'),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files},
        batches=len(batches),train_ids_sha256=signature(store.main.iloc[indices].doc_id.tolist()),
        validation_independent_query_relevance=False,test_used_for_selection=False)
    if (out/'training_protocol.json').exists() and read(out/'training_protocol.json')!=protocol:
        raise ValueError('Frozen training protocol changed; preserve this run and use a new experiment version')
    write_json(out/'training_protocol.json',protocol)
    for source_file in source_files:
        target=snapshot/source_file.name
        if not target.exists():target.write_bytes(source_file.read_bytes())
    if phase=='pilot' and (out/'pilot_selection.json').exists():
        log('Retaining frozen pilot selection; no post-training reselection')
        return
    initial=torch.load(out/'initial_last_block.pt',map_location='cpu',weights_only=True)
    reference=LastBlock(out/'tokenizer');reference.load_state_dict(initial);reference.eval()
    for param in reference.parameters():param.requires_grad=False
    baseline_path=out/'baseline/validation/metrics.json'
    baseline=read(baseline_path) if baseline_path.exists() else evaluate(reference,store,'validation',out/'baseline/validation')
    candidates=[]
    if phase=='full':
        selected=read(out/'pilot_selection.json');rates=[selected['learning_rate']]
    else:rates=cfg['learning_rates']
    for lr in rates:
        name=f'lr_{lr:.0e}';folder=out/'experiments'/name;folder.mkdir(parents=True,exist_ok=True)
        block=LastBlock(out/'tokenizer');block.load_state_dict(initial);block.eval()
        optimizer=torch.optim.AdamW(block.parameters(),lr=lr,weight_decay=cfg['weight_decay'])
        state=dict(next_batch=0,rows_seen=0,seconds=0.,optimizer_updates=0,best_score=baseline['group_mrr_at_10'],best_step=0)
        history=[];validations=[];resume=folder/'resume.pt'
        identity=signature(dict(config=cfg,versions=versions,inputs=read(out/'input_manifest.json'),lr=lr,
            trainer=sha(Path(__file__)),batches=[b.tolist() for b in batches]))
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            if saved['identity']!=identity:raise ValueError('Training fingerprint mismatch')
            block.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer'])
            state=saved['state'];history=saved['history'];validations=saved['validations']
        limit=min(cfg['pilot_steps'],len(batches)) if phase=='pilot' else len(batches)
        def save():
            save_torch(resume,dict(identity=identity,weights=block.state_dict(),optimizer=optimizer.state_dict(),state=state,history=history,validations=validations))
            pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
            write_json(folder/'validation_history.json',validations)
            write_json(folder/'progress.json',dict(**state,total_batches=len(batches),phase=phase,learning_rate=lr))
        for at in range(state['next_batch'],limit):
            tick=time.monotonic()
            # Linear warmup then decay; validation never updates parameters.
            scale=min(1.,(at+1)/max(1,int(len(batches)*.05))) * max(.1,1-at/len(batches))
            for group in optimizer.param_groups:group['lr']=lr*scale
            metric=train_step(block,store,batches[at],optimizer,cfg,reference)
            metric.update(step=at+1,seconds=time.monotonic()-tick,learning_rate=lr*scale)
            history.append(metric);state.update(next_batch=at+1,rows_seen=state['rows_seen']+len(batches[at]),
                optimizer_updates=state['optimizer_updates']+1,seconds=state['seconds']+metric['seconds'])
            if (at+1)%4==0 or at==0:log(f'TRAIN {name} {at+1}/{limit}; loss={metric["loss"]:.5f}; step={metric["seconds"]:.2f}s; chunks={metric["chunks_backpropagated"]}')
            if (at+1)%cfg['checkpoint_every_steps']==0:save()
            if (at+1)%cfg['validation_every_steps']==0 or at+1==limit:
                measured=evaluate(block,store,'validation',folder/f'validation_step_{at+1}')
                validations.append(dict(step=at+1,**measured))
                if measured['group_mrr_at_10']>state['best_score']:
                    state.update(best_score=measured['group_mrr_at_10'],best_step=at+1)
                    save_torch(folder/'best_last_block.pt',block.state_dict())
                save()
        current=validations[-1] if validations else baseline
        candidates.append(dict(name=name,learning_rate=lr,score=current['group_mrr_at_10'],best_score=state['best_score'],best_step=state['best_step']))
        del block,optimizer;gc.collect()
    if phase=='pilot':
        chosen=max(candidates,key=lambda r:r['score']);write_json(out/'pilot_selection.json',dict(**chosen,selected_on='validation only',candidates=candidates))
    else:
        chosen=candidates[0];baseline_score=baseline['group_mrr_at_10']
        improved=chosen['best_score']>=baseline_score+cfg['minimum_validation_gain'] and chosen['best_step']>0
        seen=np.concatenate(batches[:state['next_batch']])
        counts=np.bincount(seen,minlength=len(store.main))
        expected_chunks=int(store.docs.iloc[indices].chunk_count.sum())
        expected_tokens=int(store.docs.iloc[indices].abstract_tokens.sum())
        if not (np.all(counts[indices]==1) and counts.sum()==len(indices)
                and sum(r['chunks_backpropagated'] for r in history)==expected_chunks
                and sum(r['abstract_tokens_covered'] for r in history)==expected_tokens):
            raise ValueError('Full training-pass coverage does not match actual backprop history')
        coverage=store.main[['doc_id','source']].copy();coverage['split']=store.splits.split.to_numpy()
        coverage['optimizer_appearances']=counts;coverage['abstract_tokens']=store.docs.abstract_tokens.to_numpy()
        coverage['chunks']=store.docs.chunk_count.to_numpy();coverage.to_csv(out/'training_document_coverage.csv',index=False)
        write_json(out/'training_coverage.json',dict(status='PASS',train_documents=len(indices),
            actual_unique_abstract_tokens=expected_tokens,actual_chunks_backpropagated=expected_chunks,
            validation_and_test_optimizer_appearances=int(counts[store.splits.split.to_numpy()!='train'].sum())))
        selected_seen=np.concatenate(batches[:chosen['best_step']]) if improved else np.asarray([],dtype=int)
        write_json(out/'selection.json',{**chosen,'candidate_best_score':chosen['best_score'],'candidate_best_step':chosen['best_step'],
            'best_score':chosen['best_score'] if improved else baseline_score,'best_step':chosen['best_step'] if improved else 0,
            'baseline_score':baseline_score,'accepted':improved,'test_used_for_selection':False,
            'gradient_updates_real':True,'train_documents':state['rows_seen'],'all_training_abstracts_covered':state['rows_seen']==len(indices),
            'selected_checkpoint_rows_seen':len(selected_seen),
            'selected_checkpoint_abstract_tokens_seen':int(store.docs.iloc[selected_seen].abstract_tokens.sum()),
            'selected_checkpoint':str((out/'experiments'/chosen['name']/'best_last_block.pt') if improved else out/'initial_last_block.pt')})
    log(f'{phase.upper()} COMPLETE')


def smoke_train(out=OUT):
    """Real optimizer update on complete cached train rows; no serving change."""
    out=Path(out);cfg=configuration();torch.set_num_threads(cfg['threads']);store=FeatureStore(out)
    usable=[]
    for i in np.flatnonzero((store.splits.split.to_numpy()=='train') & store.query_valid):
        if all(store.old_valid[j] for j in store.chunk_ids(int(i))):usable.append(int(i))
        if len(usable)==cfg['effective_batch_pairs']:break
    if len(usable)<2:raise ValueError('Not enough committed train rows for a smoke update')
    block=LastBlock(out/'tokenizer' if (out/'tokenizer/config.json').exists() else ROOT/cfg['initial_checkpoint'])
    original=torch.load(out/'initial_last_block.pt',map_location='cpu',weights_only=True);block.load_state_dict(original);block.eval()
    # Verify bucketing/padding does not change any document representation.
    with torch.inference_mode():
        _,batched,_=store.representations(block,usable[:4])
        single=torch.stack([normal_pool(store.document_chunks(block,i,2),store.weights(i)) for i in usable[:4]])
        error=float((batched-single).abs().max())
        if error>2e-5:raise ValueError('Bucketed/full-document inference parity failed')
    optimizer=torch.optim.AdamW(block.parameters(),lr=cfg['learning_rates'][0],weight_decay=cfg['weight_decay'])
    tick=time.monotonic();metrics=train_step(block,store,np.asarray(usable),optimizer,cfg)
    delta=float(torch.sqrt(sum((v-original[k]).square().sum() for k,v in block.state_dict().items())))
    if delta<=0:raise ValueError('No actual trainable-block update occurred')
    metrics.update(seconds=time.monotonic()-tick,weight_delta_l2=delta,padding_bucket_parity_error=error,train_ids=store.main.iloc[usable].doc_id.tolist(),
        serving_model_changed=False,training_smoke_only=True)
    write_json(out/'real_training_smoke.json',metrics)
    log(f'REAL TRAIN SMOKE PASS: delta={delta:.6f}; {metrics["seconds"]:.2f}s')


def export_and_embed(out=OUT):
    out=Path(out);cfg=configuration();selection=read(out/'selection.json');store=FeatureStore(out)
    block=LastBlock(out/'tokenizer');block.load_state_dict(torch.load(selection['selected_checkpoint'],map_location='cpu',weights_only=True));block.eval()
    if not (out/'final_test/metrics.json').exists():evaluate(block,store,'test',out/'final_test')
    ids=np.arange(len(store.main));path=out/'document_embeddings.npy';marker=out/'embedding_progress.json'
    identity=signature(dict(checkpoint=sha(selection['selected_checkpoint']),features=read(out/'features/identity.json'),pooling=cfg['document_aggregation']))
    state=read(marker) if marker.exists() else dict(identity=identity,completed=0)
    if state['identity']!=identity:raise ValueError('Stale embedding resume')
    matrix=np.lib.format.open_memmap(path,mode='r+' if marker.exists() else 'w+',dtype='float32',shape=(len(ids),768))
    with torch.inference_mode():
        for at in range(state['completed'],len(ids),32):
            end=min(at+32,len(ids));_,values,_=store.representations(block,ids[at:end],include_queries=False)
            matrix[at:end]=values.numpy()
            if end%64==0 or end==len(ids):
                matrix.flush();state['completed']=end;write_json(marker,state);log(f'EMBED {end}/{len(ids)}')
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=2e-5):raise ValueError('Invalid final vectors')
    write_json(out/'ordered_doc_ids.json',store.main.doc_id.tolist())
    probes=[]
    for i in sorted(set([0,int(store.docs.abstract_tokens.argmax()),len(ids)-1])):
        probes.append((i,store.chunks.iloc[store.chunk_ids(i)].to_dict('records'),store.weights(i).numpy(),np.array(matrix[i],copy=True)))
    weights=block.layer.state_dict();initial=torch.load(out/'initial_last_block.pt',map_location='cpu',weights_only=True)
    delta=float(torch.sqrt(sum((block.state_dict()[k]-v).square().sum() for k,v in initial.items())))
    del block,store;gc.collect()
    from .batterybert import BatteryEncoder,tensor_hash
    encoder=BatteryEncoder(ROOT/cfg['initial_checkpoint']);encoder.model.encoder.layer[-1].load_state_dict(weights)
    encoder.model.save_pretrained(out/'model',safe_serialization=True);encoder.tokenizer.save_pretrained(out/'model')
    frozen_after=tensor_hash([(n,p) for n,p in encoder.model.named_parameters() if not n.startswith('encoder.layer.11.')])
    if frozen_after!=read(out/'features/identity.json')['frozen_hash']:raise ValueError('Frozen encoder blocks changed during continuation')
    parity=[]
    with torch.inference_mode():
        for i,records,chunk_weights,expected in probes:
            direct=[]
            for at in range(0,len(records),2):
                features=encoder.record_features(records[at:at+2]);hidden=encoder.model(**features).last_hidden_state
                mask=(features['token_type_ids']==1)&(features['input_ids']!=encoder.tokenizer.sep_token_id)&features['attention_mask'].bool()
                for row in range(len(mask)):
                    if not mask[row].any():mask[row]=features['attention_mask'][row].bool()&(features['input_ids'][row]!=encoder.tokenizer.sep_token_id)&(features['input_ids'][row]!=encoder.tokenizer.cls_token_id)
                weight=mask.unsqueeze(-1).to(hidden.dtype)
                direct.append(F.normalize((hidden*weight).sum(1)/weight.sum(1),dim=1))
            actual=normal_pool(torch.cat(direct),torch.from_numpy(chunk_weights)).numpy()
            error=float(np.max(np.abs(actual-expected)))
            if error>2e-5:raise ValueError('Exported full BERT disagrees with cached last-block inference')
            parity.append(dict(doc_index=i,chunks=len(records),max_error=error))
    write_json(out/'model_reload_parity.json',dict(status='PASS',probes=parity,frozen_hash_after=frozen_after))
    write_json(out/'embedding_success.json',dict(status='PASS',documents=len(ids),dimension=768,sha256=sha(path),
        weights_sha256=sha(out/'model/model.safetensors'),pooling=cfg['document_aggregation'],
        tokenizer_files={p.name:sha(p) for p in (out/'model').iterdir() if p.name in ('config.json','tokenizer.json','tokenizer_config.json')},
        content_pooling=cfg['document_content_pooling'],full_available_abstract_coverage=True,
        article_full_text_available=False,new_training_accepted=selection['accepted'],
        selected_last_block_delta_l2=delta,pretrained_fallback=False,title_context=True))
    log('MODEL EXPORT AND FULL-CORPUS EMBEDDINGS COMPLETE')
