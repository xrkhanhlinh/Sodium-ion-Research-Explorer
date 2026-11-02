"""Synthetic fixtures only: expanded pools cannot fabricate relevance scores."""
import json
import pandas as pd
import pytest
from src import research_completion_evaluation as evaluation
from src.final_system_artifacts import sha


def dataset(tmp_path):
    pd.DataFrame([dict(query_id=q,query=q) for q in ['unit-Q1','unit-Q2']]).to_csv(tmp_path/'queries.csv',index=False)
    pairs=pd.DataFrame([dict(query_id=q,doc_id=d) for q in ['unit-Q1','unit-Q2'] for d in ['unit-A','unit-B']])
    pairs.to_csv(tmp_path/'annotation_pool.csv',index=False)
    ranks=[];runs=[]
    for q in pairs.query_id.unique():
        for method in ['legacy','candidate']:
            docs=['unit-A','unit-B'] if method=='legacy' else ['unit-B','unit-A']
            ranks.extend(dict(query_id=q,method=method,doc_id=d,rank=i) for i,d in enumerate(docs,1))
            runs.append(dict(query_id=q,method=method,status='OK',result_count=2))
    pd.DataFrame(ranks).to_csv(tmp_path/'expanded.csv',index=False)
    pd.DataFrame(runs).to_csv(tmp_path/'runs.csv',index=False)
    (tmp_path/'annotation_protocol.json').write_text(json.dumps(dict(pool_sha256=sha(tmp_path/'annotation_pool.csv'),
        queries_sha256=sha(tmp_path/'queries.csv'),rankings_sha256=sha(tmp_path/'expanded.csv'))))
    return pairs


def test_expanded_evaluation_is_isolated_and_requires_all_pool_judgments(tmp_path):
    pairs=dataset(tmp_path);historical_directory=evaluation.OUT
    def run(labels=None):
        return evaluation.evaluate_dataset(tmp_path,['legacy','candidate'],labels,'expanded.csv','runs.csv')
    empty=run()
    assert empty.MRR_at_10.isna().all() and empty.eligible_queries.eq(0).all()
    labels=pairs.iloc[:3].copy();labels['relevance_grade']=[2,0,1]
    labels=labels.assign(annotator_id='unit-annotator',verifier_id='unit-verifier',
        verification_reference='SYNTHETIC TEST FIXTURE ONLY',label_method='HUMAN_INDEPENDENT',label_status='VERIFIED')
    path=tmp_path/'synthetic_labels.csv';labels.to_csv(path,index=False)
    scored=run(path).set_index('method')
    assert scored.eligible_queries.eq(1).all()
    assert scored.loc['legacy','MRR_at_10']==1. and scored.loc['candidate','MRR_at_10']==.5
    assert run().eligible_queries.eq(1).all()
    assert evaluation.OUT==historical_directory


def test_expanded_evaluation_rejects_missing_method_runs(tmp_path):
    dataset(tmp_path)
    with pytest.raises(ValueError,match='Incomplete successful benchmark'):
        evaluation.evaluate_dataset(tmp_path,['legacy','candidate','absent'],None,'expanded.csv','runs.csv')
