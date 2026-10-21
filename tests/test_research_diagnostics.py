import pandas as pd
import pytest
from src.research_diagnostics import potential_aqueous_contradiction,diagnose_rankings,topic_review_sample


@pytest.mark.parametrize('title',['non-aqueous solvent','Non aqueous sodium electrolytes','non-water solvent','NONWATER solvent','non‑aqueous solvent','nonaqueous electrolyte'])
def test_explicit_potential_title_contradictions(title):
    assert potential_aqueous_contradiction('Aqueous sodium-ion batteries',title)


@pytest.mark.parametrize('query,title',[
    ('Non-aqueous sodium batteries','non-aqueous electrolyte'),
    ('Nonaqueous sodium batteries','non-aqueous electrolyte'),
    ('Aqueous sodium batteries','Aqueous solvent with nonaqueously phrased comparison'),
    ('Hard carbon sodium storage','non-aqueous solvent'),
    ('Aqueous sodium batteries','Water-in-salt aqueous electrolyte'),
    ('Aqueous sodium batteries','Review of batteries'),
])
def test_no_flag_from_absent_terms_or_negative_query(query,title):
    assert not potential_aqueous_contradiction(query,title)


def test_group_repetition_not_declared_as_relevance():
    metadata=pd.DataFrame([dict(doc_id='d1',split_group_id='g1',source='Patent'),
                           dict(doc_id='d2',split_group_id='g1',source='Publication')])
    rankings=pd.DataFrame([dict(query_id='Q1',method='A',rank=i+1,doc_id=d,title='non-aqueous electrolyte') for i,d in enumerate(['d1','d2'])])
    queries=pd.DataFrame([dict(query_id='Q1',query='Aqueous batteries')])
    rows,flags=diagnose_rankings(rankings,metadata,queries)
    assert rows.iloc[0].repeated_known_group_slots==1
    assert len(flags)==2 and flags.label_status.eq('UNVERIFIED_DIAGNOSTIC_ONLY').all()
    assert 'relevance_grade' not in flags


def test_topic_sample_is_reproducible_assigned_and_not_only_top_members():
    assignments=pd.DataFrame([dict(doc_id=f'd{i}',topic_id=0) for i in range(12)])
    representatives=pd.DataFrame([dict(topic_id=0,doc_id=f'd{i}',rank=i+1) for i in range(3)])
    summary=pd.DataFrame([dict(topic_id=0,top10_keywords='carbon; sodium')])
    metadata=pd.DataFrame([dict(doc_id=f'd{i}',title='title',abstract='abstract',source='Publication') for i in range(12)])
    a=topic_review_sample(assignments,representatives,summary,metadata)
    b=topic_review_sample(assignments,representatives,summary,metadata)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==a.doc_id.nunique()==5
    assert a.selection_method.eq('random_additional_member_seed42').sum()==2
    assert a.label_status.eq('UNVERIFIED').all() and a.topic_fit_grade.eq('').all()
    representatives.loc[0,'doc_id']='wrong'
    with pytest.raises(ValueError):topic_review_sample(assignments,representatives,summary,metadata)
