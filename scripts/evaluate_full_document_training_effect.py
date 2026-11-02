"""Paired post-selection proxy comparison; no checkpoint/ranking tuning."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import numpy as np
import pandas as pd
import torch
from src.batterybert_full_document import OUT,FeatureStore,LastBlock,evaluate,read
from src.model_comparison_io import write_json


def compare(baseline,selected,main,split):
    if baseline.doc_id.tolist()!=selected.doc_id.tolist():raise ValueError('Paired evaluation ID order differs')
    data=baseline[['doc_id','group_rank']].rename(columns={'group_rank':'baseline_rank'}).copy()
    data['selected_rank']=selected.group_rank
    data=data.join(main.set_index('doc_id')[['split_group_id','source']],on='doc_id',validate='many_to_one')
    for name in ['baseline','selected']:
        rank=data[name+'_rank'].to_numpy();data[name+'_rr']=np.where(rank<=10,1/rank,0.)
    data['rr_difference']=data.selected_rr-data.baseline_rr
    group=data.groupby('split_group_id').rr_difference.agg(['sum','count'])
    rng=np.random.default_rng(20261024);replicates=[]
    values=group.to_numpy()
    for at in range(0,10000,100):
        sample=values[rng.integers(0,len(values),size=(100,len(values)))].sum(axis=1)
        replicates.extend(sample[:,0]/sample[:,1])
    ci=np.quantile(replicates,[.025,.975])
    result=dict(split=split,documents=len(data),groups=len(group),baseline_mrr_at_10=float(data.baseline_rr.mean()),
        selected_mrr_at_10=float(data.selected_rr.mean()),paired_gain=float(data.rr_difference.mean()),
        cluster_bootstrap_percentile_95_interval=ci.tolist(),bootstrap_replicates=10000,
        resampling_unit='split_group_id',test_used_for_selection=False,
        interpretation='Paired title-conditioned full-document proxy; not independent relevance',
        validation_selection_bias=(split=='validation'),
        source_groups=[dict(source=s,documents=len(g),gain=float(g.rr_difference.mean())) for s,g in data.groupby('source')])
    return result,data


def main():
    torch.set_num_threads(4)
    selection=read(OUT/'selection.json');store=FeatureStore(OUT)
    block=LastBlock(OUT/'tokenizer');block.load_state_dict(torch.load(OUT/'initial_last_block.pt',map_location='cpu',weights_only=True));block.eval()
    if not (OUT/'baseline/test/metrics.json').exists():evaluate(block,store,'test',OUT/'baseline/test')
    results=[]
    for split in ['validation','test']:
        if split=='test':selected=OUT/'final_test'
        elif selection['best_step']:
            selected=OUT/'experiments'/selection['name']/f"validation_step_{selection['best_step']}"
        else:selected=OUT/'baseline/validation'
        metric,paired=compare(pd.read_csv(OUT/f'baseline/{split}/ranks.csv'),pd.read_csv(selected/'ranks.csv'),store.main,split)
        paired.to_csv(OUT/f'paired_{split}_ranks.csv',index=False);results.append(metric)
    write_json(OUT/'paired_training_effect.json',results)
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
