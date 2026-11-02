"""Deduplicate final pytest evidence and hash the delivered run artifacts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
import importlib.metadata
import xml.etree.ElementTree as ET
from src.final_system_artifacts import sha
from src.model_comparison_io import write_json

OUT=ROOT/'outputs/batterybert_full_document_20261024_v1'


def main():
    cases={};sources=[];web_durations={}
    # The targeted rerun follows a serving/filter change; deduplicate shared
    # cases rather than summing repeated executions as extra coverage.
    for name in ['backend_tests.xml','post_filter_backend_tests.xml','web_tests.xml']:
        path=OUT/name;sources.append(dict(path=name,sha256=sha(path)))
        for case in ET.parse(path).getroot().iter('testcase'):
            if name=='web_tests.xml':web_durations[case.get('name')]=float(case.get('time','0'))
            identity=(case.get('classname'),case.get('name'))
            status='error' if case.find('error') is not None else 'failure' if case.find('failure') is not None else 'skipped' if case.find('skipped') is not None else 'passed'
            if identity in cases and cases[identity]!=status:raise ValueError('Conflicting test outcomes')
            cases[identity]=status
    summary=dict(status='PASS',distinct_tests=len(cases),passed=sum(s=='passed' for s in cases.values()),
        failures=sum(s=='failure' for s in cases.values()),errors=sum(s=='error' for s in cases.values()),
        skipped=sum(s=='skipped' for s in cases.values()),xml_sources=sources,
        technical_tests_are_not_relevance_accuracy=True,web_test_case_seconds=web_durations)
    if summary['failures'] or summary['errors']:raise ValueError('Final verification has failures/errors')
    deploy=json.loads((OUT/'deployment.json').read_text())
    if deploy['activate_search'] and summary['skipped']:raise ValueError('Accepted serving branch needs its real tests, no skips')
    for name in ['dashboard_server_smoke.json','model_reload_parity.json','inference_api_parity.json',
                 'validation_query_api_parity.json','long_query_tail_probe.json','preservation_check.json','training_coverage.json']:
        value=json.loads((OUT/name).read_text())
        if value.get('status')!='PASS':raise ValueError('Verification artifact failed: '+name)
    write_json(OUT/'verification_summary.json',summary)
    versions={}
    for name in ['torch','transformers','numpy','pandas','scipy','scikit-learn','gensim','nltk','qdrant-client','streamlit','pytest']:
        versions[name]=importlib.metadata.version(name)
    write_json(OUT/'package_versions.json',versions)
    code=[p for folder in ['src','scripts','tests'] for p in (ROOT/folder).glob('*full_document*.py')]
    code+=[ROOT/'src/final_search.py',ROOT/'src/final_dashboard.py',ROOT/'src/final_system_artifacts.py',
        ROOT/'src/scientific_keyword_stream.py',ROOT/'src/research_completion_evaluation.py',ROOT/'src/final_system_vectors.py',ROOT/'tests/test_scientific_keyword_stream.py',
        ROOT/'configs/batterybert_full_document.yaml',ROOT/'scripts/export_current_topic_catalogs.py']
    write_json(OUT/'final_code_manifest.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(code))})
    paths=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or p.name=='artifact_manifest.json' or p.is_relative_to(OUT/'qdrant'):continue
        if p.name.endswith('_hidden.npy') or p.suffix in ['.pending','.log']:continue
        paths.append(p)
    write_json(OUT/'artifact_manifest.json',dict(
        files={p.relative_to(OUT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)},
        excluded='Large prefix hidden arrays use per-row SHA-256/validity; mutable Qdrant storage is verified by all-point vector/payload and exact search checks; logs/temp files excluded'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
