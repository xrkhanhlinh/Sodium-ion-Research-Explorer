"""Real, resumable partial BGE fine-tuning and memory-bounded inference."""
import os
os.environ.setdefault('HF_HOME',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')
import argparse
import hashlib
import random
import time
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from .stage3_common import ROOT,REPORT,TABLE,DATA,config,read_json,write_json,csv,signature,digest,progress,verify_frozen,group_retrieval_metrics,validate_frozen_validation_settings

MODEL=ROOT/'models/bert_finetuned'
CFG=config()
torch.set_num_threads(CFG['threads'])
torch.set_num_interop_threads(1)

def setup_seed():
    random.seed(CFG['seed']); np.random.seed(CFG['seed']); torch.manual_seed(CFG['seed'])

def model_load(finetuned=False):
    if (REPORT/'cpu_benchmark.json').exists():
        torch.set_num_threads(read_json(REPORT/'cpu_benchmark.json')['selected_threads'])
    info=read_json(REPORT/'pretrained_model.json')
    path=MODEL if finetuned else ROOT/info['local_path']
    model=SentenceTransformer(str(path),device=CFG['device'],local_files_only=True)
    model.max_seq_length=CFG['max_seq_length']
    assert model.get_embedding_dimension()==768
    return model

def benchmark_cpu(model):
    """Choose a CPU thread count using training texts only, never test metrics."""
    import psutil
    path=REPORT/'cpu_benchmark.json'
    if path.exists():
        torch.set_num_threads(read_json(path)['selected_threads']);return
    pairs=pd.read_parquet(DATA/'contrastive_pairs.parquet')
    sample=pairs[pairs.split.eq('train')].head(16)
    texts=[CFG['query_prefix']+s for s in sample.title]+sample.abstract.tolist()
    model.encode(texts[:8],batch_size=8,normalize_embeddings=True,show_progress_bar=False)
    results=[];reference=None
    for threads in sorted(set([1,2,min(4,psutil.cpu_count(logical=False) or 2)])):
        torch.set_num_threads(threads)
        start=time.perf_counter();arr=model.encode(texts,batch_size=8,normalize_embeddings=True,show_progress_bar=False);seconds=time.perf_counter()-start
        if reference is None:reference=arr
        error=float(np.max(np.abs(arr-reference)))
        assert np.allclose(arr,reference,atol=2e-5)
        results.append(dict(threads=threads,documents=len(texts),seconds=seconds,documents_per_second=len(texts)/seconds,max_abs_difference=error))
        print('CPU BENCHMARK',results[-1],flush=True)
    selected=min(results,key=lambda r:r['seconds'])['threads']
    write_json(path,dict(selected_threads=selected,method='One warmed real BGE inference pass on 16 train titles + 16 train abstracts, batch8; no optimizer updates/test data. Select fastest measured 1/2/up-to-4 physical threads.',results=results))
    torch.set_num_threads(selected)

def fetch():
    from huggingface_hub import HfApi,snapshot_download
    if (REPORT/'pretrained_model.json').exists():
        info=read_json(REPORT/'pretrained_model.json')
        assert all(digest(ROOT/p)==h for p,h in info['files'].items())
        print('REUSE pinned pretrained snapshot',info['revision'],flush=True);return
    revision=HfApi().model_info(CFG['model_id']).sha
    folder=ROOT/'models/bge_pretrained'
    snapshot_download(CFG['model_id'],revision=revision,local_dir=str(folder),allow_patterns=['*.json','*.txt','model.safetensors','1_Pooling/*'],max_workers=2)
    files={p.relative_to(ROOT).as_posix():digest(p) for p in folder.rglob('*') if p.is_file() and '.cache' not in p.parts}
    assert (folder/'model.safetensors').exists(),'Missing real pretrained weights'
    write_json(REPORT/'pretrained_model.json',dict(model=CFG['model_id'],revision=revision,local_path=folder.relative_to(ROOT).as_posix(),files=files))
    progress(f'Downloaded pinned real BGE pretrained weights: revision={revision}.')

def tensor_hash(named):
    h=hashlib.sha256()
    for name,param in named:
        h.update(name.encode()); h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def runtime():
    import psutil
    hardware=dict(torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),cuda_devices=torch.cuda.device_count(),selected_device=CFG['device'],threads=torch.get_num_threads(),ram=psutil.virtual_memory()._asdict(),versions={n:importlib.metadata.version(n) for n in ['torch','sentence-transformers','transformers','tokenizers','huggingface-hub']})
    write_json(REPORT/'neural_environment.json',hardware)
    print(hardware,flush=True)

def encode_cached(model,texts,path,phase):
    """Durable prefix progress + memmap; never cache by text without model identity."""
    path=__import__('pathlib').Path(path)
    stamp=path.with_suffix('.progress.json')
    identity=signature(dict(phase=phase,texts=texts,config=CFG,model_revision=read_json(REPORT/'pretrained_model.json')['revision'],finetuned=digest(MODEL/'model.safetensors') if 'pretrained' not in phase else None))
    if stamp.exists():
        marker=read_json(stamp);assert marker['fingerprint']==identity,'Cached embeddings fingerprint mismatch'
        done=marker['completed']
        arr=np.lib.format.open_memmap(path,mode='r+')
    else:
        done=0;arr=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(texts),768))
    start=time.perf_counter()
    for i in range(done,len(texts),CFG['encoding']['chunk_size']):
        end=min(len(texts),i+CFG['encoding']['chunk_size'])
        arr[i:end]=model.encode(texts[i:end],batch_size=CFG['encoding']['batch_size'],normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
        arr.flush()
        write_json(stamp,dict(fingerprint=identity,completed=end,total=len(texts),seconds_this_session=time.perf_counter()-start))
        print(f'ENCODE {phase} {end}/{len(texts)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    assert np.isfinite(arr).all() and np.allclose(np.linalg.norm(arr,axis=1),1,atol=1e-5)
    return arr

def encode_validation_frozen_cache(model,texts,field):
    """Reuse exact float32 frozen-encoder states, then evaluate the current last block."""
    validate_frozen_validation_settings(CFG['training'])
    bert=model[0].auto_model;model.eval();length=CFG['max_seq_length']
    path=DATA/f'validation_frozen_{field}.npy';maskpath=DATA/f'validation_frozen_{field}_mask.npy'
    marker=path.with_suffix('.progress.json')
    identity=signature(dict(texts=texts,max_length=length,revision=read_json(REPORT/'pretrained_model.json')['revision'],frozen_layers=11,dtype='float32',transformers=importlib.metadata.version('transformers')))
    if marker.exists():
        state=read_json(marker);assert state['fingerprint']==identity
        hidden_cache=np.lib.format.open_memmap(path,mode='r+');mask_cache=np.lib.format.open_memmap(maskpath,mode='r+');done=state['completed']
    else:
        hidden_cache=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(texts),length,768))
        mask_cache=np.lib.format.open_memmap(maskpath,mode='w+',dtype='int64',shape=(len(texts),length));done=0
    bs=CFG['encoding']['batch_size']
    with torch.no_grad():
        for start in range(done,len(texts),bs):
            end=min(len(texts),start+bs);features=model.preprocess(texts[start:end])
            h=bert.embeddings(input_ids=features['input_ids'],token_type_ids=features.get('token_type_ids'))
            att,_=bert._create_attention_masks(features['attention_mask'],None,h,None,None)
            for layer in bert.encoder.layer[:-1]:h=layer(h,attention_mask=att)
            width=h.shape[1]
            hidden_cache[start:end]=0;mask_cache[start:end]=0
            hidden_cache[start:end,:width]=h.numpy();mask_cache[start:end,:width]=features['attention_mask'].numpy()
            if end%128==0 or end==len(texts):
                hidden_cache.flush();mask_cache.flush();write_json(marker,dict(fingerprint=identity,completed=end,total=len(texts)))
                print(f'VALIDATION FROZEN CACHE {field} {end}/{len(texts)}',flush=True)
        result=[]
        for start in range(0,len(texts),bs):
            end=min(len(texts),start+bs)
            width=int(mask_cache[start:end].sum(axis=1).max())
            h=torch.from_numpy(np.array(hidden_cache[start:end,:width],copy=True))
            attention=torch.from_numpy(np.array(mask_cache[start:end,:width],copy=True))
            att,_=bert._create_attention_masks(attention,None,h,None,None)
            h=bert.encoder.layer[-1](h,attention_mask=att)
            result.append(torch.nn.functional.normalize(h[:,0],dim=1).numpy())
        result=np.concatenate(result)
        reference=model.encode(texts[:3],batch_size=3,normalize_embeddings=True,show_progress_bar=False)
        error=float(np.max(np.abs(result[:3]-reference)))
        assert np.allclose(result[:3],reference,atol=2e-5),'Frozen-cache inference differs from full BGE'
        write_json(REPORT/f'validation_frozen_cache_{field}_check.json',dict(status='PASS',dtype='float32',max_abs_error=error,checked_documents=3,cache_documents=len(texts),cached_layers=11,trainable_layer_evaluated_fresh=True))
        return result

def evaluate(model,split,phase,cache=True):
    pairs=pd.read_parquet(DATA/'contrastive_pairs.parquet')
    frame=pairs[pairs.split.eq(split)].reset_index(drop=True)
    titles=[CFG['query_prefix']+x for x in frame.title]
    abstracts=frame.abstract.tolist()
    if cache:
        q=encode_cached(model,titles,DATA/f'{phase}_{split}_titles.npy',phase+'_'+split+'_titles')
        d=encode_cached(model,abstracts,DATA/f'{phase}_{split}_abstracts.npy',phase+'_'+split+'_abstracts')
    else:
        assert split=='validation','Only validation uses frozen-state acceleration'
        q=encode_validation_frozen_cache(model,titles,'titles')
        d=encode_validation_frozen_cache(model,abstracts,'abstracts')
    metrics,rows=group_retrieval_metrics(q,d,frame.split_group_id.to_numpy(),frame.doc_id.to_numpy())
    metrics.update(split=split,phase=phase,candidate_scope='same held-out split only; all group matches relevant',max_seq_length=CFG['max_seq_length'])
    if cache:
        write_json(REPORT/f'{phase}_{split}_metrics.json',metrics)
        csv(f'{phase}_{split}_retrieval_ranks.csv',rows)
    print('EVAL',phase,split,metrics,flush=True)
    return metrics

def baseline():
    progress('Evaluating real pretrained BGE on fixed validation/test pairs; no training or checkpoint selection uses test.')
    runtime();setup_seed();model=model_load()
    assert model[1].pooling_mode=='cls','Expected BGE CLS pooling'
    start=time.perf_counter()
    stats=[]
    frame=pd.read_parquet(ROOT/'data/processed/bert_texts.parquet')
    for field in ['title','abstract','text']:
        truncated=0;total=0;maximum=0
        for i in range(0,len(frame),128):
            texts=frame[field].iloc[i:i+128].tolist()
            if field=='title': texts=[CFG['query_prefix']+t for t in texts]
            lengths=[len(ids) for ids in model.tokenizer(texts,add_special_tokens=True,truncation=False)['input_ids']]
            truncated+=sum(n>CFG['max_seq_length'] for n in lengths);total+=len(lengths);maximum=max(maximum,max(lengths))
        stats.append(dict(field=field,documents=total,truncated_at_128=truncated,truncation_share=truncated/total,max_tokens=maximum))
    csv('tokenizer_truncation.csv',pd.DataFrame(stats))
    for split in ['validation','test']: evaluate(model,split,'pretrained')
    write_json(REPORT/'baseline_success.json',dict(seconds=time.perf_counter()-start,weights_hash=digest(ROOT/'models/bge_pretrained/model.safetensors')))
    progress('Pretrained BGE evaluated on fixed validation/test Title–Abstract splits before training; baseline saved. Test scores will not select checkpoints.')

def false_negative_mask(frame,threshold):
    words=[set(__import__('re').findall(r'\w+',t.casefold())) for t in frame.title]
    n=len(frame);mask=np.zeros((n,n),dtype=bool)
    for i in range(n):
        for j in range(i):
            same=frame.iloc[i].split_group_id==frame.iloc[j].split_group_id
            lexical=len(words[i]&words[j])/max(1,len(words[i]|words[j]))>=threshold
            mask[i,j]=mask[j,i]=same or lexical
    return mask

def train():
    validate_frozen_validation_settings(CFG['training'])
    setup_seed();verify_frozen()
    assert (REPORT/'baseline_success.json').exists()
    import inspect
    trainer_signature=signature(dict(code=[inspect.getsource(f) for f in [train,false_negative_mask,encode_validation_frozen_cache,evaluate]],config=CFG,revision=read_json(REPORT/'pretrained_model.json')['revision'],versions=read_json(REPORT/'neural_environment.json')['versions']))
    if (MODEL/'training_success.json').exists():
        success=read_json(MODEL/'training_success.json')
        assert success['config_sha256']==digest(ROOT/'configs/stage3.yaml')
        assert success['trainer_signature']==trainer_signature,'Trainer changed; preserve successful model rather than silently reusing/retraining'
        assert digest(MODEL/'model.safetensors')==success['model_sha256']
        print('REUSE completed real fine-tuning checkpoint',flush=True);return
    model=model_load();benchmark_cpu(model);bert=model[0].auto_model
    for p in model.parameters(): p.requires_grad=False
    for layer in bert.encoder.layer[-CFG['training']['trainable_encoder_layers']:]:
        for p in layer.parameters(): p.requires_grad=True
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    frozen=[(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    before={n:p.detach().cpu().clone() for n,p in trainable}
    frozen_before=tensor_hash(frozen)
    frame=pd.read_parquet(DATA/'contrastive_pairs.parquet')
    frame=frame[frame.split.eq('train')].reset_index(drop=True)
    settings=CFG['training'];bs=settings['batch_size']
    batches=[]
    for epoch in range(settings['epochs']):
        order=np.random.default_rng(CFG['seed']+epoch).permutation(len(frame))
        chunks=[order[i:i+bs] for i in range(0,len(order),bs)]
        if len(chunks[-1])==1:
            chunks[-2]=np.concatenate([chunks[-2],chunks[-1]]);chunks.pop()
        batches += [(epoch,chunk) for chunk in chunks]
    optimizer=torch.optim.AdamW([p for n,p in trainable],lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    warmup=max(1,int(len(batches)*settings['warmup_fraction']))
    def lr_factor(step):
        return step/warmup if step<warmup else max(0.,(len(batches)-step)/max(1,len(batches)-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    last=MODEL/'resume.pt';best=MODEL/'best_trainable.pt'
    state=dict(step=0,best_score=-1.,best_step=None,training_seconds=0.,rows_seen=0,masked_pairs=0)
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        assert saved['config_sha256']==digest(ROOT/'configs/stage3.yaml')
        assert saved['trainer_signature']==trainer_signature,'Trainer differs from resumable experiment'
        for n,p in trainable:p.data.copy_(saved['weights'][n])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);state=saved['state']
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        print('RESUME real training at step',state['step'],flush=True)
    parameter_info=dict(total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for n,p in trainable),trainable_names=[n for n,p in trainable],frozen_hash_before=frozen_before,planned_steps=len(batches),training_pairs=len(frame),settings=settings,device=CFG['device'],torch_threads=torch.get_num_threads(),max_seq_length=CFG['max_seq_length'],config_sha256=digest(ROOT/'configs/stage3.yaml'),trainer_signature=trainer_signature)
    write_json(REPORT/'training_plan.json',parameter_info)
    losses=read_json(REPORT/'training_history.json') if (REPORT/'training_history.json').exists() else []
    validations=read_json(REPORT/'validation_history.json') if (REPORT/'validation_history.json').exists() else []
    losses=[x for x in losses if x['step']<=state['step']]
    validations=[x for x in validations if x['step']<=state['step']]
    def save_resume():
        temp=last.with_suffix('.tmp')
        torch.save(dict(config_sha256=digest(ROOT/'configs/stage3.yaml'),trainer_signature=trainer_signature,weights={n:p.detach().cpu() for n,p in trainable},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),state=state,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),temp)
        os.replace(temp,last)
        write_json(REPORT/'training_history.json',losses);write_json(REPORT/'validation_history.json',validations)
        csv('training_loss.csv',pd.DataFrame(losses));csv('validation_metrics.csv',pd.DataFrame(validations))
    progress(f'Real BGE CPU fine-tuning started/resumed: {len(frame)} train pairs, {len(batches)} steps, final encoder block trainable; test excluded from checkpoint selection.')
    for step in range(state['step'],len(batches)):
        start=time.perf_counter();epoch,indices=batches[step];batch=frame.iloc[indices]
        model.eval()
        for layer in bert.encoder.layer[-settings['trainable_encoder_layers']:]: layer.train()
        optimizer.zero_grad(set_to_none=True)
        queries=model(model.preprocess([CFG['query_prefix']+s for s in batch.title]))['sentence_embedding']
        documents=model(model.preprocess(batch.abstract.tolist()))['sentence_embedding']
        queries=torch.nn.functional.normalize(queries,dim=1);documents=torch.nn.functional.normalize(documents,dim=1)
        scores=queries@documents.T/settings['temperature']
        mask=false_negative_mask(batch,settings['title_jaccard_false_negative_threshold'])
        scores=scores.masked_fill(torch.as_tensor(mask),-1e4)
        target=torch.arange(len(batch))
        loss=(torch.nn.functional.cross_entropy(scores,target)+torch.nn.functional.cross_entropy(scores.T,target))/2
        if not torch.isfinite(loss):raise RuntimeError('Non-finite real training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for n,p in trainable],settings['max_grad_norm']);optimizer.step();scheduler.step()
        seconds=time.perf_counter()-start
        state.update(step=step+1,training_seconds=state['training_seconds']+seconds,rows_seen=state['rows_seen']+len(batch),masked_pairs=state['masked_pairs']+int(mask.sum()))
        losses.append(dict(step=step+1,epoch=epoch+1,loss=float(loss.detach()),learning_rate=optimizer.param_groups[0]['lr'],batch_pairs=len(batch),masked_off_diagonal=int(mask.sum()),seconds=seconds))
        if step==0 or (step+1)%25==0: print(f'TRAIN {step+1}/{len(batches)} loss={float(loss.detach()):.6f} seconds={seconds:.2f} total_train_seconds={state["training_seconds"]:.1f}',flush=True)
        if (step+1)%settings['eval_every_steps']==0 or step+1==len(batches):
            model.eval();val_start=time.perf_counter()
            metrics=evaluate(model,'validation',f'step_{step+1}',cache=False)
            validations.append(dict(step=step+1,epoch=epoch+1,seconds=time.perf_counter()-val_start,**metrics))
            selection_key=settings['selection_metric'].removeprefix('validation_')
            if metrics[selection_key]>state['best_score']:
                state.update(best_score=metrics[selection_key],best_step=step+1)
                best_temp=best.with_suffix('.tmp')
                torch.save({n:p.detach().cpu() for n,p in trainable},best_temp);os.replace(best_temp,best)
            save_resume()
            progress(f'Fine-tuning step {step+1}/{len(batches)}; validation MRR@10={metrics["group_mrr_at_10"]:.6f}, best step={state["best_step"]}. Test not inspected for selection.')
        elif (step+1)%settings['checkpoint_every_steps']==0:
            save_resume()
            progress(f'Real fine-tuning checkpoint persisted at step {step+1}/{len(batches)}, rows seen={state["rows_seen"]}, optimizer time={state["training_seconds"]:.1f}s. Validation/test policy unchanged.')
    assert state['best_step'] is not None
    selected=torch.load(best,map_location='cpu',weights_only=True)
    for n,p in trainable:p.data.copy_(selected[n])
    frozen_after=tensor_hash(frozen);assert frozen_before==frozen_after
    l2=sum(float(torch.sum((p.detach().cpu()-before[n])**2)) for n,p in trainable)**.5
    assert l2>0,'No real pretrained weight update occurred'
    model.eval();model.save_pretrained(str(MODEL),safe_serialization=True,create_model_card=False)
    success=dict(**parameter_info,**state,model_sha256=digest(MODEL/'model.safetensors'),selected_weight_delta_l2=l2,selected_weights_hash=tensor_hash(trainable),frozen_hash_after=frozen_after,selection_uses_test=False,real_gradient_updates=True)
    write_json(MODEL/'training_success.json',success)
    progress(f'Real BGE fine-tuning complete: selected validation checkpoint step={state["best_step"]}; nonzero pretrained weight delta L2={l2:.6f}. Saved full 768-dimensional model.')

def after():
    assert read_json(MODEL/'training_success.json')['real_gradient_updates']
    model=model_load(True)
    for split in ['validation','test']: evaluate(model,split,'finetuned')
    comparison=[]
    for phase in ['pretrained','finetuned']:
        for split in ['validation','test']:comparison.append(read_json(REPORT/f'{phase}_{split}_metrics.json'))
    csv('bert_before_after.csv',pd.DataFrame(comparison))
    progress('Before/after BGE evaluated on identical group-disjoint held-out test; results saved, no test-based checkpoint selection.')

def embed():
    assert (MODEL/'training_success.json').exists()
    model=model_load(True)
    frame=pd.read_parquet(ROOT/'data/processed/bert_texts.parquet')
    arr=encode_cached(model,frame.text.tolist(),DATA/'embeddings.npy','finetuned_full_corpus')
    table=pd.DataFrame(arr,columns=[f'emb_{i:03d}' for i in range(768)])
    table.insert(0,'source',frame.source.to_numpy());table.insert(0,'doc_id',frame.doc_id.to_numpy())
    table.to_parquet(DATA/'embeddings.parquet',index=False)
    write_json(REPORT/'embedding_manifest.json',dict(shape=list(arr.shape),ordered_doc_ids_sha256=signature(frame.doc_id.tolist()),model_sha256=digest(MODEL/'model.safetensors'),parquet_sha256=digest(DATA/'embeddings.parquet'),numpy_sha256=digest(DATA/'embeddings.npy'),normalization='L2',text_field='Stage 1 bert_texts.text unchanged',query_prefix_for_documents=False,max_seq_length=CFG['max_seq_length']))
    verify_frozen();progress('Saved real fine-tuned BGE embeddings for all 13,702 documents: 768 dimensions, L2 normalized, doc_id aligned. Ready for BERTopic.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['fetch','baseline','train','after','embed']);args=parser.parse_args()
    globals()[args.phase]()
