"""Resume the 25 October research revision from verified stage checkpoints.

Run with project Python. No encoder training or corpus embedding is repeated
when the frozen 24 October weights/vectors and coverage contracts are valid.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.resumable_pipeline import Pipeline,Stage,atomic_json,sha256,utc_now

REV=ROOT/'outputs/research_revision_20261025_v1'
BASE=ROOT/'outputs/batterybert_full_document_20261024_v1'
EXEC=REV/'execution'
GEOMETRY=REV/'geometry/evidence/geometry'
KEYWORDS=REV/'keywords/experiment_outputs'
SEARCH=REV/'search/experiment_outputs'
PYTHON=str(ROOT/'.venv/Scripts/python.exe')
JOBS=ROOT/'scripts/research_revision_jobs.py'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def json_valid(path,required=()):
    value=read(path)
    if str(value.get('status','')).upper() in ('FAIL','FAILED','ERROR','PARTIAL'):
        raise ValueError('Failed/partial artifact is not a valid completed checkpoint: '+str(path))
    for key in required:
        if key not in value:raise ValueError('Missing field '+key+' in '+str(path))
    return {'valid':True,'path':str(path),'keys_checked':list(required)}


def inventory():
    protected=read(REV/'protected_artifacts_before.json')
    changed=[item['path'] for item in protected if not (ROOT/item['path']).is_file() or sha256(ROOT/item['path'])!=item['sha256']]
    if changed:raise ValueError('Protected historical inputs changed: '+str(changed))
    geometry=read(GEOMETRY/'development_selection.json')
    search_completed=[name for name in ['baseline','abstract_rerank30','abstract_rerank50'] if (SEARCH/'dev'/(name+'_summary.json')).is_file()]
    result={'status':'PASS','checked_at':utc_now(),'protected_files':len(protected),'historical_inputs_changed':[],
        'encoder_training':'Completed and accepted 24/10; do not repeat',
        'embeddings':'Existing 13702 x 768 vectors require validation, not recomputation',
        'geometry_development':'Completed 3 candidates; selected '+str(geometry['selected']),
        'keyword_cycle1':'Completed; all three candidates rejected',
        'keyword_cycle2':'Completed; both candidates rejected' if (KEYWORDS/'cycle2/comparison.json').is_file() else 'Pending',
        'search_dev_completed':search_completed,
        'last_valid_before_interruption':'Geometry development complete; Search baseline and Top30 complete',
        'interrupted_work':'Search Top50 and final scientific-parser correction; final gate/export/comparison/report not yet completed',
        'recovery_policy':'Adopt semantically checked existing outputs; hash every committed stage; save per-query/per-shard work; never overwrite historical outputs'}
    atomic_json(EXEC/'resume_inventory.json',result)


def validate_embeddings():
    info=read(BASE/'embedding_success.json');selection=read(BASE/'selection.json')
    coverage=read(BASE/'training_coverage.json')
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet',columns=['doc_id'])
    ids=read(BASE/'ordered_doc_ids.json')
    if ids!=main.doc_id.tolist() or len(ids)!=len(set(ids)):raise ValueError('Embedding doc_id identity/order mismatch')
    matrix=np.load(BASE/'document_embeddings.npy',mmap_mode='r')
    if matrix.shape!=(13702,768):raise ValueError('Embedding matrix shape mismatch')
    error=0.
    for start in range(0,len(matrix),1024):
        values=matrix[start:start+1024]
        if not np.isfinite(values).all():raise ValueError('Nonfinite embeddings')
        error=max(error,float(np.max(np.abs(np.linalg.norm(values,axis=1)-1))))
    if error>2e-5:raise ValueError('Embeddings not L2 normalized')
    if sha256(BASE/'document_embeddings.npy')!=info['sha256']:raise ValueError('Embedding SHA256 mismatch')
    if sha256(BASE/'model/model.safetensors')!=info['weights_sha256']:raise ValueError('Model/checkpoint identity mismatch')
    if not selection['accepted'] or coverage['status']!='PASS' or coverage['validation_and_test_optimizer_appearances']!=0:
        raise ValueError('Historical training/coverage not accepted')
    atomic_json(EXEC/'embedding_validation.json',{'status':'PASS','documents':len(ids),'dimension':768,
       'maximum_norm_error':error,'vector_sha256':info['sha256'],'model_sha256':info['weights_sha256'],
       'encoder_forward_calls':0,'fit_calls':0,'origin':'Reused verified frozen artifacts from 24/10',
       'training_checkpoint_step':selection['best_step'],'trained_documents':coverage['train_documents'],
       'abstract_tokens_backpropagated':coverage['actual_unique_abstract_tokens']})


def validate_geometry():
    protocol=read(GEOMETRY/'protocol.json')
    for spec in [protocol['baseline']]+protocol['candidates']:
        folder=GEOMETRY/spec['name'];metric=read(folder/'metrics.json')
        if metric['spec']!=spec:raise ValueError('Geometry recipe changed')
        prediction=pd.read_csv(folder/'validation_predictions.csv')
        keywords=pd.read_csv(folder/'keywords.csv')
        if len(prediction)!=1361 or not prediction.doc_id.is_unique or len(keywords)!=100:
            raise ValueError('Geometry output dimensions invalid')
        for artifact in read(folder/'development_artifact_manifest.json'):
            if sha256(folder/artifact['path'])!=artifact['sha256']:raise ValueError('Geometry artifact changed')
    selection=read(GEOMETRY/'development_selection.json')
    if selection['selected']!='source_n50':raise ValueError('Unexpected frozen geometry choice')
    return {'valid':True,'fits_repeated':0,'winner':selection['selected'],'candidates':3}


def validate_keyword_cycle1():
    protocol=read(KEYWORDS/'protocol.json')
    for name in ['baseline_dev','contrastive_df','title_evidence','contrastive_phrases']:
        metrics=read(KEYWORDS/name/'dev_metrics.json');keys=pd.read_csv(KEYWORDS/name/'keywords.csv')
        if len(keys)!=100 or metrics['documents']!=protocol['dev_documents']:raise ValueError('Keyword experiment dimensions invalid')
        if not np.isfinite(metrics['macro_f1']):raise ValueError('Invalid keyword proxy')
    return {'valid':True,'candidates':3,'decision':'All rejected on frozen development gates'}


def validate_keyword_cycle2():
    values=read(KEYWORDS/'cycle2/comparison.json')
    if len(values)!=2 or any(row['passes_dev'] for row in values):raise ValueError('Unexpected cycle2 outcome')
    return {'valid':True,'candidates':2,'decision':'Both rejected; no gate was optimized against'}


def search_bundle(name,split='dev'):
    folder=SEARCH/split
    summary=read(folder/(name+'_summary.json'))
    queries=pd.read_csv(folder/(name+'_per_query.csv'));ranks=pd.read_csv(folder/(name+'_rankings.csv'))
    expected=[item['case_id'] for item in read(SEARCH/'protocol.json')['cases'] if item['role']==split]
    if queries.case_id.tolist()!=expected or set(ranks.case_id)!=set(expected):raise ValueError('Search query manifest mismatch')
    if summary['protocol_sha256']!=sha256(SEARCH/'protocol.json'):raise ValueError('Search protocol changed')
    if not np.isclose(summary['group_mrr_at_10'],queries.reciprocal_rank.mean()):raise ValueError('Search aggregate mismatch')
    if name!='baseline':
        evidence=read(folder/(name+'_evidence.json'))
        if [item['case_id'] for item in evidence]!=expected:raise ValueError('Search evidence mismatch')
    return {'valid':True,'queries':len(expected),'method':name,'scope':'Biased title-pair proxy, not natural-query relevance'}


def manifest_valid(path):
    manifest=read(path)
    records=manifest.get('artifacts',manifest.get('outputs',[]))
    if isinstance(records,dict):records=[{'path':name,'sha256':digest} for name,digest in records.items()]
    records=list(records)+[{'path':name,'sha256':digest} for name,digest in manifest.get('shard_manifests',{}).items()]
    if not records:raise ValueError('Empty artifact manifest: '+str(path))
    for record in records:
        artifact=Path(record['path'])
        if not artifact.is_absolute():artifact=Path(path).parent/artifact
        if not artifact.is_file() or sha256(artifact)!=record['sha256']:raise ValueError('Artifact manifest mismatch: '+str(artifact))
    return {'valid':True,'artifacts':len(records)}


def build_stages():
    command=lambda *parts:[PYTHON,*map(str,parts)]
    geometry_outputs=[GEOMETRY/'development_selection.json',GEOMETRY/'protocol.json']
    for name in ['baseline_source_n10','source_n50','source_whiten32_n50','source_whiten64_n50']:
        geometry_outputs.extend(GEOMETRY/name/file for file in ['metrics.json','pipeline.joblib','keywords.csv','development_artifact_manifest.json'])
    stages=[
      Stage('resume_inventory',[EXEC/'resume_inventory.json'],lambda:json_valid(EXEC/'resume_inventory.json',['status','protected_files']),action=inventory,
            inputs=[REV/'protected_artifacts_before.json'],note='Validate original data and identify completed/interrupted phases'),
      Stage('embeddings',[EXEC/'embedding_validation.json'],lambda:json_valid(EXEC/'embedding_validation.json',['vector_sha256','encoder_forward_calls']),action=validate_embeddings,
            inputs=[BASE/'document_embeddings.npy',BASE/'model/model.safetensors',BASE/'embedding_success.json',BASE/'ordered_doc_ids.json'],dependencies=('resume_inventory',),
            note='Reuse valid full-document embeddings, never retrain/encode an already accepted corpus'),
      Stage('baseline_audit',[REV/'baseline_audit.json',REV/'protected_artifacts_before.json'],lambda:json_valid(REV/'baseline_audit.json',['formula_parser_examples','frozen_relevance_status']),
            command=command(ROOT/'scripts/audit_research_revision_20261025.py'),dependencies=('resume_inventory',),note='Adopt audit completed before usage interruption'),
      Stage('geometry_development',geometry_outputs,validate_geometry,
            command=command(REV/'geometry/scripts/geometry_revision.py','candidates','--project-root',ROOT),dependencies=('embeddings',),
            note='Adopt existing baseline and all three completed clustering candidates'),
      Stage('keyword_cycle1',[KEYWORDS/name/'dev_metrics.json' for name in ['baseline_dev','contrastive_df','title_evidence','contrastive_phrases']],validate_keyword_cycle1,
            dependencies=('baseline_audit',),note='Preserve three failed keyword hypotheses; no rerun'),
      Stage('keyword_cycle2',[KEYWORDS/'cycle2/comparison.json'],validate_keyword_cycle2,
            command=command(REV/'keywords/scripts/fit_keyword_cycle2.py','--project-root',ROOT),dependencies=('keyword_cycle1',),note='Preserve two conservative keyword hypotheses; no promotion when gates fail'),
    ]
    for name,phase,depth in [('baseline','baseline',None),('abstract_rerank30','candidate',30),('abstract_rerank50','candidate',50)]:
        outputs=[SEARCH/'dev'/(name+suffix) for suffix in ['_summary.json','_per_query.csv','_rankings.csv']]
        if depth:outputs.append(SEARCH/'dev'/(name+'_evidence.json'))
        cmd=command(REV/'search/scripts/evaluate_search_revision.py',phase,'--project-root',ROOT,'--split','dev')
        if depth:cmd+=['--depth',str(depth)]
        stages.append(Stage('search_dev_'+name,outputs,lambda name=name:search_bundle(name),command=cmd,
            dependencies=('embeddings',),note='Per-query resume validates previous query vectors, candidates, rankings and evidence'))
    for name in ['baseline_source_n10','source_n50']:
        path=GEOMETRY/name/'gate_metrics.json'
        stages.append(Stage('geometry_gate_'+name,[path,GEOMETRY/name/'gate_predictions.csv'],lambda path=path:json_valid(path,['stability_ari','document_npmi','seed']),
             command=command(REV/'geometry/scripts/geometry_revision.py','gate','--project-root',ROOT,'--candidate',name),
             dependencies=('geometry_development',),note='Coordinator-only new seed; exactly once per frozen baseline/winner'))
    stages.extend([
      Stage('geometry_decision',[REV/'decisions/geometry.json'],lambda:json_valid(REV/'decisions/geometry.json',['accepted','baseline','candidate']),
            command=command(JOBS,'geometry_decision'),dependencies=('geometry_gate_baseline_source_n10','geometry_gate_source_n50')),
      Stage('geometry_export',[REV/'geometry/evidence/export_source_n50/artifact_manifest.json'],lambda:manifest_valid(REV/'geometry/evidence/export_source_n50/artifact_manifest.json'),
            command=command(REV/'geometry/scripts/export_geometry_candidate.py','--project-root',ROOT,'--candidate-folder',GEOMETRY/'source_n50','--output-dir',REV/'geometry/evidence/export_source_n50'),
            dependencies=('geometry_decision',),note='Prediction only; preserve native partition; align presentation IDs on train overlap'),
      Stage('search_development_selection',[REV/'decisions/search_development.json'],lambda:json_valid(REV/'decisions/search_development.json',['selected','baseline','candidate']),
            command=command(JOBS,'search_select'),dependencies=('search_dev_baseline','search_dev_abstract_rerank30','search_dev_abstract_rerank50')),
      Stage('search_gate_baseline',[SEARCH/'gate/baseline_summary.json',SEARCH/'gate/baseline_per_query.csv',SEARCH/'gate/baseline_rankings.csv'],lambda:search_bundle('baseline','gate'),
            command=command(REV/'search/scripts/evaluate_search_revision.py','baseline','--project-root',ROOT,'--split','gate'),dependencies=('search_development_selection',)),
      Stage('search_gate_selected',[REV/'decisions/search_gate.json'],lambda:json_valid(REV/'decisions/search_gate.json',['accepted','baseline','candidate']),
            command=command(JOBS,'search_gate'),dependencies=('search_gate_baseline',)),
      Stage('scientific_stream_v2',[REV/'scientific_stream_v2/artifact_manifest.json'],lambda:manifest_valid(REV/'scientific_stream_v2/artifact_manifest.json'),
            command=command(JOBS,'scientific_stream'),dependencies=('keyword_cycle2',),note='Versioned per-shard tokenization; immutable raw fields and historical stream'),
      Stage('final_topics',[REV/'final_topics/artifact_manifest.json'],lambda:manifest_valid(REV/'final_topics/artifact_manifest.json'),
            command=command(JOBS,'final_topics'),dependencies=('geometry_export','scientific_stream_v2'),note='Canonical formula keywords, full-corpus predictions and honest semantic labels'),
      Stage('topic_semantic_review',[REV/'review/topic_labels.csv',REV/'review/summary.json'],lambda:json_valid(REV/'review/summary.json',['topics_reviewed','human_ground_truth']),
            command=command(JOBS,'semantic_review'),dependencies=('final_topics',),note='AI interpretation of keywords and complete central abstracts; no expert accuracy claim'),
      Stage('model_comparison',[REV/'comparison/artifact_manifest.json'],lambda:manifest_valid(REV/'comparison/artifact_manifest.json'),
            command=command(JOBS,'comparison'),dependencies=('topic_semantic_review','search_gate_selected'),note='Compare archived/current/new models with explicit metric protocols'),
      Stage('visualizations',[REV/'visualizations/artifact_manifest.json'],lambda:manifest_valid(REV/'visualizations/artifact_manifest.json'),
            command=command(JOBS,'visualizations'),dependencies=('model_comparison',)),
      Stage('code_verification',[REV/'verification/summary.json'],lambda:json_valid(REV/'verification/summary.json',['status','tests','preservation']),
            command=command(JOBS,'verification'),dependencies=('topic_semantic_review','search_gate_selected')),
      Stage('deployment',[REV/'deployment.json'],lambda:json_valid(REV/'deployment.json',['activate_revised_topics','activate_search_rerank']),
            command=command(JOBS,'deployment'),dependencies=('code_verification',),note='Activate accepted topic catalog; explicitly keep rejected reranker inactive'),
      Stage('routing_smoke',[REV/'verification/routing_smoke.json'],lambda:json_valid(REV/'verification/routing_smoke.json',['status','all_returned_doc_ids_match_revised_catalog']),
            command=command(JOBS,'routing_smoke'),dependencies=('deployment',),note='Real lexical retrieval with revised topic IDs, no neural inference'),
      Stage('final_report',[REV/'report/artifact_manifest.json'],lambda:manifest_valid(REV/'report/artifact_manifest.json'),
            command=command(JOBS,'report'),dependencies=('visualizations','routing_smoke'),note='New version only; includes failed hypotheses, resume inventory and metric limitations'),
      Stage('report_layout_revision',[REV/'report_v2/artifact_manifest.json'],lambda:manifest_valid(REV/'report_v2/artifact_manifest.json'),
            command=command(JOBS,'report_v2'),dependencies=('final_report',),note='Keep first PDF unchanged; correct physical page numbers and plot-label collisions in separate version'),
      Stage('report_visual_qa',[REV/'report_v2/validation_final.json'],lambda:json_valid(REV/'report_v2/validation_final.json',['status','visual_review','page_count']),
            command=command(ROOT/'scripts/validate_revision_report_20261026.py'),dependencies=('report_layout_revision',),note='PDF text/layout checks and explicit PNG inspection before delivery'),
    ])
    return stages


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--through');parser.add_argument('--stage',action='append');parser.add_argument('--status',action='store_true')
    args=parser.parse_args();stages=build_stages();pipeline=Pipeline(EXEC,stages,ROOT)
    if args.status:
        print(json.dumps(pipeline.state,ensure_ascii=True,indent=2));return
    selected=args.stage
    if args.through:
        names=[stage.name for stage in stages]
        if args.through not in names:parser.error('Unknown --through stage')
        selected=names[:names.index(args.through)+1]
    summary=pipeline.run(selected);print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
