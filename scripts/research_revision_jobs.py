"""Coordinator jobs for the resumable research revision; no original data writes."""
from pathlib import Path
import argparse
import json
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.resumable_pipeline import atomic_json,sha256
REV=ROOT/'outputs/research_revision_20261025_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Refusing to replace completed output: '+str(path))
    else:atomic_json(path,value)


def run_script(path,args):
    previous=sys.argv
    try:
        sys.argv=[str(path),*map(str,args)]
        runpy.run_path(str(path),run_name='__main__')
    finally:sys.argv=previous


def geometry_decision():
    directory=REV/'geometry/evidence/geometry'
    baseline=read(directory/'baseline_source_n10/gate_metrics.json')
    candidate=read(directory/'source_n50/gate_metrics.json')
    protocol=read(directory/'protocol.json');limits=protocol['gates']
    gates={
      'stability_improvement':candidate['stability_ari']>=baseline['stability_ari']+limits['minimum_ari_improvement'],
      'source':candidate['source_cramers_v']<=limits['source_cramers_v_max'],
      'largest':candidate['largest_topic_share']<=limits['largest_topic_share_max'],
      'top2':candidate['top2_topic_share']<=limits['top2_topic_share_max'],
      'smallest':candidate['smallest_topic_share']>=limits['smallest_topic_share_min'],
      'support':candidate['supported_keyword_fraction']>=limits['supported_keyword_fraction_min'],
      'diversity':candidate['keyword_diversity']>=limits['keyword_diversity_min'],
      'coherence':candidate['document_npmi']>=max(0.,baseline['document_npmi']-limits['document_npmi_max_drop_from_baseline'])}
    value={'accepted':all(gates.values()),'gates':gates,'baseline':baseline,'candidate':candidate,
      'selected':'source_n50','selection_scope':'Historical validation development; coordinator seed20261029 gate on historical test',
      'expert_ground_truth':False,'scientific_accuracy_claim':False,
      'tradeoffs':['Test source V and top2 share are slightly higher than current baseline, still within frozen bounds',
          'Stability and document NPMI improve; no universal model superiority is inferred'],
      'inputs':{name:sha256(directory/name) for name in ['protocol.json','development_selection.json','baseline_source_n10/gate_metrics.json','source_n50/gate_metrics.json']}}
    immutable_json(REV/'decisions/geometry.json',value)
    if not value['accepted']:raise ValueError('Frozen geometry candidate failed coordinator gate; no automatic retuning')
    print(json.dumps({'accepted':True,'ari_before':baseline['stability_ari'],'ari_after':candidate['stability_ari'],'npmi_before':baseline['document_npmi'],'npmi_after':candidate['document_npmi']},indent=2))


def search_select():
    folder=REV/'search/experiment_outputs'
    selection=read(folder/'selection.json')
    baseline=read(folder/'dev/baseline_summary.json')
    candidates={name:read(folder/'dev'/(name+'_summary.json')) for name in ['abstract_rerank30','abstract_rerank50']}
    valid=[(name,row) for name,row in candidates.items() if row['success_at_10']>=baseline['success_at_10']]
    name,winner=max(valid,key=lambda pair:(pair[1]['group_mrr_at_10'],-int(pair[0].replace('abstract_rerank',''))))
    if name!=selection['selected_method']:raise ValueError('Executor selection differs from frozen deterministic rule')
    if sha256(REV/'search/src/full_document_rerank.py')!=selection['candidate_source_sha256']:
        raise ValueError('Candidate implementation changed after dev selection')
    immutable_json(REV/'decisions/search_development.json',{'selected':name,'depth':int(name.replace('abstract_rerank','')),
       'baseline':baseline,'candidate':winner,'candidates':candidates,
       'protocol_sha256':sha256(folder/'protocol.json'),'candidate_source_sha256':selection['candidate_source_sha256'],
       'gate_opened_by_selection':False,'independent_relevance_claim':False})


def search_gate():
    selected=read(REV/'decisions/search_development.json')
    folder=REV/'search/experiment_outputs'
    if sha256(REV/'search/src/full_document_rerank.py')!=selected['candidate_source_sha256']:
        raise ValueError('Frozen candidate changed before gate')
    run_script(REV/'search/scripts/evaluate_search_revision.py',['candidate','--project-root',ROOT,'--split','gate','--depth',selected['depth']])
    baseline=read(folder/'gate/baseline_summary.json')
    candidate=read(folder/'gate'/(selected['selected']+'_summary.json'))
    gain=candidate['group_mrr_at_10']-baseline['group_mrr_at_10']
    accepted=gain>=.005 and candidate['success_at_10']>=baseline['success_at_10']
    immutable_json(REV/'decisions/search_gate.json',{'accepted':accepted,'selected':selected['selected'],'depth':selected['depth'],
      'baseline':baseline,'candidate':candidate,'mrr_gain':gain,
      'gates':{'mrr_gain_at_least_0_005':gain>=.005,'success10_no_loss':candidate['success_at_10']>=baseline['success_at_10']},
      'optional_method_only':True,'natural_query_relevance_validated':False,
      'scope':'Newly reserved title anchors from historically reused corpus. Corpus index contains Title; no human query relevance labels.'})
    print(json.dumps({'accepted':accepted,'mrr_gain':gain,'method':selected['selected']},indent=2))


def scientific_stream():
    run_script(REV/'keywords/scripts/export_scientific_stream_v2.py',['--project-root',ROOT,'--output',REV/'scientific_stream_v2'])


def final_topics():
    decision=read(REV/'decisions/geometry.json')
    if not decision['accepted']:raise ValueError('No accepted geometry')
    run_script(REV/'geometry/scripts/finalize_revision_topics.py',['--project-root',ROOT,'--output',REV/'final_topics',
        '--export',REV/'geometry/evidence/export_source_n50','--stream',REV/'scientific_stream_v2/tokens.parquet'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',choices=['geometry_decision','search_select','search_gate','scientific_stream','final_topics','semantic_review','comparison','visualizations','verification','deployment','routing_smoke','report','report_v2'])
    args=parser.parse_args()
    if args.job in ['semantic_review','comparison','visualizations','verification','deployment','routing_smoke','report','report_v2']:
        from revision_reporting import run_job
        run_job(args.job)
    else:globals()[args.job]()


if __name__=='__main__':main()
