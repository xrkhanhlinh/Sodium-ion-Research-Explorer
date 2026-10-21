"""Small shared utilities; importing this module does not load neural libraries."""
import hashlib
import json
import os
import sys
from numbers import Integral
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

for _stream in [sys.stdout,sys.stderr]:
    if hasattr(_stream,'reconfigure'):
        _stream.reconfigure(encoding='utf-8',errors='replace')

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'outputs/reports/stage3'
TABLE=ROOT/'outputs/tables/stage3'
DATA=ROOT/'data/stage3'
FIGURE=ROOT/'outputs/figures/stage3'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

def require_cache_fingerprint(path, expected, metadata_path=None):
    """Refuse legacy or mismatched caches without changing historical files."""
    path=Path(path)
    if not path.exists():
        return None
    metadata=Path(metadata_path) if metadata_path is not None else path
    payload=read_json(metadata) if metadata.exists() else {}
    if payload.get('cache_fingerprint') != expected:
        raise ValueError(f'Unverified or mismatched cache: {path}. Preserve historical artifacts and recompute in an isolated output directory.')
    return payload

def validate_frozen_validation_settings(settings):
    """The current validation acceleration supports exactly one updated block."""
    layers=settings['trainable_encoder_layers']
    if isinstance(layers,bool) or not isinstance(layers,Integral) or layers != 1:
        raise ValueError('Frozen validation cache requires trainable_encoder_layers=1; implement generic frozen-prefix validation before changing this setting.')

def config():
    return yaml.safe_load((ROOT/'configs/stage3.yaml').read_text(encoding='utf-8'))

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)

def csv(name,frame):
    TABLE.mkdir(parents=True,exist_ok=True)
    frame.to_csv(TABLE/name,index=False,encoding='utf-8-sig')

def progress(message,status='RUNNING'):
    stamp=datetime.now(timezone(timedelta(hours=7))).isoformat()
    REPORT.mkdir(parents=True,exist_ok=True)
    print(stamp,status,message,flush=True)
    write_json(REPORT/'progress.json',dict(stage=3,status=status,updated=stamp,message=message))
    with (ROOT/'outputs/logs/stage3.log').open('a',encoding='utf-8') as stream:
        stream.write(f'{stamp} {status}: {message}\n')
    with (ROOT/'docs/EXPERIMENT_LOG.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n- {stamp} - Stage 3 {status}: {message}\n')

def verify_frozen():
    from .artifact_history import verify_snapshot
    original=read_json(REPORT/'input_manifest.json')
    result=verify_snapshot(original['protected_hashes'])
    return result['checked']

def group_retrieval_metrics(queries,documents,groups,ids=None):
    """Title->abstract proxy, with all same-group abstracts relevant.

    Success@k measures whether any relevant abstract is reached by k. Legacy
    group_recall_at_k keys are deprecated aliases, NOT document Recall@k.
    Group rank pessimistically places non-relevant ties before the best match;
    exact rank pessimistically places every tied other document before self.
    """
    groups=np.asarray(groups)
    ranks=[]; exact=[]; rows=[]
    for start in range(0,len(queries),64):
        scores=queries[start:start+64]@documents.T
        for j,sim in enumerate(scores):
            i=start+j
            relevant=groups==groups[i]
            best=sim[relevant].max()
            # Pessimistic ties; no arbitrary input-order advantage.
            rank=1+int(np.sum((sim>=best)&~relevant))
            r_exact=1+int(np.sum((sim>=sim[i])&(np.arange(len(sim))!=i)))
            ranks.append(rank);exact.append(r_exact)
            rows.append(dict(doc_id=str(ids[i]) if ids is not None else str(i),group_rank=rank,exact_rank=r_exact,candidates=len(sim),relevant_documents=int(relevant.sum())))
    ranks=np.asarray(ranks);exact=np.asarray(exact)
    metrics=dict(queries=len(ranks),candidates=len(documents),group_mrr_at_10=float(np.mean(np.where(ranks<=10,1/ranks,0))),exact_mrr_at_10=float(np.mean(np.where(exact<=10,1/exact,0))),exact_recall_at_10=float(np.mean(exact<=10)))
    for k in (1,5,10):
        metrics[f'group_success_at_{k}']=float(np.mean(ranks<=k))
        metrics[f'group_recall_at_{k}']=metrics[f'group_success_at_{k}']
    metrics['metric_semantics']={
        'group_success_at_k':'Fraction of queries with at least one relevant same-group document by k; not document Recall@k',
        'deprecated_aliases':{f'group_recall_at_{k}':f'group_success_at_{k}' for k in (1,5,10)},
        'group_ties':'All non-relevant ties precede the highest-scored relevant document',
        'exact_ties':'All tied other documents precede the paired document',
    }
    return metrics,pd.DataFrame(rows)

def audit_exact_text_leakage(pairs):
    import re
    rows=[];summary=[]
    for field in ['title','abstract']:
        frame=pairs[['doc_id','split_group_id','split',field]].copy()
        frame['text_hash']=frame[field].map(lambda t:hashlib.sha256(re.sub(r'\s+',' ',t.casefold()).strip().encode()).hexdigest())
        counts=frame.groupby('text_hash').split.nunique()
        subset=frame[frame.text_hash.isin(set(counts[counts>1].index))]
        summary.append(dict(field=field,cross_split_exact_text_hashes=int((counts>1).sum()),affected_records=len(subset)))
        for row in subset.itertuples():rows.append(dict(field=field,doc_id=row.doc_id,split_group_id=row.split_group_id,split=row.split,text_hash=row.text_hash))
    write_json(REPORT/'exact_text_leakage_audit.json',summary)
    csv('cross_split_exact_text.csv',pd.DataFrame(rows,columns=['field','doc_id','split_group_id','split','text_hash']))
    assert all(item['cross_split_exact_text_hashes']==0 for item in summary),'Unexpected exact cross-split texts; review split grouping before training'

def audit_windows_hardware():
    info=dict(method='Read-only Windows registry; supplementary to psutil/PyTorch capability probe')
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            info['cpu_name']=winreg.QueryValueEx(key,'ProcessorNameString')[0]
        display=r'SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}'
        names=[]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,display) as key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                name=winreg.EnumKey(key,i)
                if not name.isdigit():continue
                try:
                    with winreg.OpenKey(key,name) as subkey:names.append(winreg.QueryValueEx(subkey,'DriverDesc')[0])
                except OSError:pass
        info['display_adapters']=names
    except (ImportError,OSError) as exc:info['probe_error']=str(exc)
    write_json(REPORT/'hardware_registry.json',info)
    return info

def prepare():
    import psutil, shutil, platform
    from sklearn.model_selection import GroupShuffleSplit
    for path in [REPORT,TABLE,DATA,FIGURE,ROOT/'models/bert_finetuned',ROOT/'models/bertopic',ROOT/'outputs/logs']:
        path.mkdir(parents=True,exist_ok=True)
    cfg=config()
    frozen=[]
    for folder in ['data/interim','data/processed','data/stage2','models/lda','outputs/tables/stage2','outputs/figures/stage2','outputs/reports/stage2']:
        frozen += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
    frozen += [ROOT/n for n in ['patent.xlsx','publication.xlsx','configs/config.yaml','configs/stage2.yaml']]
    manifest=REPORT/'input_manifest.json'
    if manifest.exists():
        verify_frozen()
    else:
        hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(frozen))}
        write_json(manifest,dict(protected_hashes=hashes,config=cfg,config_sha256=digest(ROOT/'configs/stage3.yaml')))
    assert read_json(manifest)['config_sha256']==digest(ROOT/'configs/stage3.yaml'),'Experiment config changed; preserve previous run'
    b=pd.read_parquet(ROOT/'data/processed/bert_texts.parquet')
    m=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    assert len(b)==cfg['expected_documents'] and b.doc_id.tolist()==m.doc_id.tolist()
    assert b.split_group_id.ne('').all() and not b.split_group_id.isna().any()
    pairs=b[b.eligible_finetune].copy().reset_index(drop=True)
    assert len(pairs)==cfg['expected_pairs'] and pairs.title.ne('').all() and pairs.abstract.ne('').all()
    outer=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=cfg['seed'])
    train_idx,hold_idx=next(outer.split(pairs,groups=pairs.split_group_id))
    hold=pairs.iloc[hold_idx]
    val_local,test_local=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=cfg['seed']).split(hold,groups=hold.split_group_id))
    pairs['split']='train'
    pairs.loc[hold_idx[val_local],'split']='validation'
    pairs.loc[hold_idx[test_local],'split']='test'
    # Includes singleton title-only documents, grouped consistently for future evaluation.
    group_map=dict(zip(pairs.split_group_id,pairs.split))
    assignments=b[['doc_id','source','split_group_id','eligible_finetune']].copy()
    assignments['split']=assignments.split_group_id.map(group_map).fillna('embedding_only')
    pairs.to_parquet(DATA/'contrastive_pairs.parquet',index=False)
    assignments.to_parquet(DATA/'split_assignments.parquet',index=False)
    csv('split_assignments.csv',assignments)
    summary=pairs.groupby(['split','source']).agg(documents=('doc_id','size'),groups=('split_group_id','nunique')).reset_index()
    csv('split_summary.csv',summary)
    split_groups={s:set(g.split_group_id) for s,g in pairs.groupby('split')}
    assert not(split_groups['train']&split_groups['validation'] or split_groups['train']&split_groups['test'] or split_groups['test']&split_groups['validation'])
    audit_exact_text_leakage(pairs)
    write_json(REPORT/'hardware_initial.json',dict(ram=psutil.virtual_memory()._asdict(),cpu_logical=psutil.cpu_count(),cpu_physical=psutil.cpu_count(logical=False),disk_free_bytes=shutil.disk_usage(ROOT).free,platform=platform.platform(),python=platform.python_version()))
    audit_windows_hardware()
    write_json(REPORT/'split_audit.json',dict(documents=len(b),pairs=len(pairs),title_only=len(b)-len(pairs),group_overlap=0,pair_counts=pairs.split.value_counts().to_dict(),groups=pairs.groupby('split').split_group_id.nunique().to_dict(),source_counts=summary.to_dict('records'),limitation='Stage 1 split_group_id captures known candidate links, not all semantic relationships. Title-abstract metrics are proxy supervision, not expert query judgments.'))
    progress('Prepared frozen corpus and group-disjoint Title–Abstract splits; Stage 1/2 hashes captured. Awaiting real CPU benchmark and BGE training.')

if __name__=='__main__': prepare()
