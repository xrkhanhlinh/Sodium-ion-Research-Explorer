"""Reconcile interrupted Arbor bookkeeping from immutable measured outputs."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/'outputs/research_revision_20261025_v1'
TREE=Path('C:/Users/Admin/.agents/skills/arbor/scripts/tree.py')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def command(run,*args):
    result=subprocess.run([sys.executable,str(TREE),'--run-dir',str(run),*map(str,args)],capture_output=True,text=True,encoding='utf8',errors='replace')
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    return result.stdout
def nodes(run):return read(run/'.arbor/tree.json')['nodes']
def prepare(run,baseline,cycles):
    config=read(run/'.arbor/run.json')
    if config['best_test_score'] is None:command(run,'baseline','--test-score',baseline,'--branch-ref','cf0b428')
    while read(run/'.arbor/run.json')['cycles_used']<cycles:
        command(run,'cycle');command(run,'observe')
def evidence(run,node,score,result,insight,branch):
    if nodes(run)[node]['status'] not in ['pending','running']:return
    command(run,'set-evidence','--node',node,'--dev-score',score,'--result',result,'--insight',insight,'--branch-ref',branch)
    command(run,'propagate','--node',node,'--insight',insight,'--to-root')
def prune(run,node,reason):
    if nodes(run)[node]['status']!='pruned':command(run,'prune','--node',node,'--reason',reason)


def main():
    geom=REV/'arbor_geometry';decision=read(REV/'decisions/geometry.json')
    prepare(geom,decision['baseline']['stability_ari'],1)
    for node,name in [('n2','source_n50'),('n3','source_whiten32_n50'),('n4','source_whiten64_n50')]:
        metric=read(REV/'geometry/evidence/geometry'/name/'metrics.json')
        insight='More initializations stabilize the same encoder geometry without losing keyword coherence' if node=='n2' else 'Whitening improves agreement but harms lexical coherence; reject by predeclared guard'
        evidence(geom,node,metric['stability_ari'],name+' measured from saved development artifacts; bookkeeping resumed after usage interruption',insight,'b1dff40')
        if node!='n2':prune(geom,node,'Failed frozen coherence/balance development gate; no final gate used')
    if nodes(geom)['n2']['status']!='merged':command(geom,'merge','--node','n2','--test-score',decision['candidate']['stability_ari'],'--branch-ref','b1dff40')
    keys=REV/'arbor_keywords';base=read(REV/'coordinator_gate/keywords_baseline/gate_metrics.json')
    prepare(keys,base['macro_f1'],2)
    for node,name in [('n2','contrastive_df'),('n3','title_evidence'),('n4','contrastive_phrases')]:
        metric=read(REV/'keywords/experiment_outputs'/name/'dev_metrics.json')
        evidence(keys,node,metric['macro_f1'],name+' failed declared coverage/coherence safeguards',
          'Specific-looking words can improve lexical classification while losing representative coverage and cooccurrence','b67ee60')
        prune(keys,node,'Development safeguards failed; coordinator gate was not opened')
    current=nodes(keys)
    if 'n5' not in current:command(keys,'add-node','--parent','n0','--hypothesis','Conservative original frequency weighting may replace procedural terms without coverage loss')
    if 'n6' not in nodes(keys):command(keys,'add-node','--parent','n5','--hypothesis','Original TF1 with conservative suppression preserves coverage and improves F1')
    if 'n7' not in nodes(keys):command(keys,'add-node','--parent','n5','--hypothesis','Square root TF with conservative suppression improves summaries without coherence loss')
    for node,metric in zip(['n6','n7'],read(REV/'keywords/experiment_outputs/cycle2/comparison.json')):
        evidence(keys,node,metric['macro_f1'],metric['candidate']+' rejected under frozen cycle2 protocol',
          'Removing generic/alias slots alone did not meet the combined F1, coherence and coverage objective','b67ee60')
        prune(keys,node,'At least one frozen development safeguard failed; preserve original statistical recipe')
    search=REV/'arbor_search';decision=read(REV/'decisions/search_gate.json')
    prepare(search,decision['baseline']['group_mrr_at_10'],1)
    for node,name in [('n2','abstract_rerank30'),('n3','abstract_rerank50')]:
        value=read(REV/'search/experiment_outputs/dev'/(name+'_summary.json'))
        evidence(search,node,value['group_mrr_at_10'],name+' saved dev results',
          'General-purpose cross encoder gains on development did not establish transfer or independent relevance','efc1e69')
    if nodes(search)['n2']['status']=='executed':command(search,'merge','--node','n2','--test-score',decision['candidate']['group_mrr_at_10'],'--branch-ref','efc1e69')
    prune(search,'n2','Reserved gate MRR decreased .927083 to .850694; do not promote or retune on this gate')
    prune(search,'n3','Lower dev MRR and greater additional latency than selected30; not gated')
    result={}
    for name in ['arbor_geometry','arbor_keywords','arbor_search']:
        path=REV/name;command(path,'validate');result[name]=read(path/'.arbor/run.json')
        (path/'FINAL_TREE.txt').write_text(command(path,'status'),encoding='utf8')
    (REV/'arbor_reconciliation.json').write_text(json.dumps({'status':'PASS','no_models_reexecuted':True,
       'reason':'Bookkeeping reconciliation after interruption; protocols and numerical evidence remain unchanged','runs':result},indent=2),encoding='utf8')
    print('PASS Arbor records reconciled; no evaluation rerun')


if __name__=='__main__':main()
