"""Resume only missing Silver annotation/evaluation stages, preserving old runs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.resumable_pipeline import Stage,Pipeline,sha256
OUT=ROOT/'outputs/ai_silver_labels_703_20261026_v1';E=OUT/'evaluation'
BASE=ROOT/'outputs/batterybert_full_document_20261024_v1'
RUNTIME='C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
def record(p):
    d=json.loads(p.read_text(encoding='utf8'))
    if d.get('status')=='FAIL':raise ValueError('Failed validation: '+str(p))
    return d
def manifest(p):
    d=record(p)
    for a in d['artifacts']:
        assert sha256(ROOT/a['path'])==a['sha256'],a['path']
    return {'valid':True,'artifacts':len(d['artifacts'])}
def batch(qid):
    r=record(OUT/'checkpoints'/f'{qid}.json')
    assert r['output_sha256']==sha256(OUT/'batches'/f'{qid}_labels.csv')
    assert r['manual_decisions_sha256']==sha256(OUT/'decisions'/f'{qid}.json')
    assert r['source_sha256']==sha256(BASE/'annotation_pool.csv')
    return {'valid':True,'pairs':r['pairs']}
def frozen():
    r=record(OUT/'labels_frozen_manifest.json');assert r['labels_sha256']==sha256(OUT/'silver_labels_703.csv')
    assert r['input_sha256']==sha256(BASE/'annotation_pool.csv')
    for n,v in r['batch_sha256'].items():assert sha256(OUT/'batches'/n)==v
    return {'valid':True,'pairs':703}
def main():
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');a=p.parse_args();stages=[]
    for i in range(1,21):
        q=f'Q{i:02d}';stages.append(Stage('labels_'+q,[OUT/'batches'/f'{q}_labels.csv',OUT/'checkpoints'/f'{q}.json'],lambda q=q:batch(q),
          command=[sys.executable,str(ROOT/'scripts/silver_labels_703.py'),'commit',q],inputs=[BASE/'annotation_pool.csv',OUT/'decisions'/f'{q}.json'],note='Explicit full-text assistant judgments; no score-based grade generation'))
    stages.extend([
      Stage('freeze_labels',[OUT/'silver_labels_703.csv',OUT/'labels_frozen_manifest.json',OUT/'labels_with_source_703.csv',OUT/'needs_review.csv',OUT/'label_summary.json'],frozen,
        command=[sys.executable,str(ROOT/'scripts/finalize_silver_703_20261026.py')],dependencies=tuple(s.name for s in stages)),
      Stage('metrics',[E/'artifact_manifest.json'],lambda:manifest(E/'artifact_manifest.json'),command=[sys.executable,str(ROOT/'scripts/evaluate_silver_703_20261026.py')],
        inputs=[OUT/'silver_labels_703.csv',BASE/'comparison_rankings_top10.csv',BASE/'comparison_run_manifest.csv'],dependencies=('freeze_labels',)),
      Stage('report',[E/'report_manifest_v2.json'],lambda:manifest(E/'report_manifest_v2.json'),command=[sys.executable,str(ROOT/'scripts/report_silver_703_20261026.py'),'--revision','v2'],dependencies=('metrics',)),
      Stage('pdf_qa',[E/'pdf_validation_v2.json'],lambda:{'valid':record(E/'pdf_validation_v2.json')['status']=='PASS'},
        command=[RUNTIME,str(ROOT/'scripts/validate_silver_report_703_20261026.py')],dependencies=('report',))])
    pipeline=Pipeline(OUT/'execution',stages,ROOT)
    if a.status:print(json.dumps(pipeline.state,ensure_ascii=True,indent=2))
    else:print(json.dumps(pipeline.run(),ensure_ascii=True,indent=2))
if __name__=='__main__':main()
