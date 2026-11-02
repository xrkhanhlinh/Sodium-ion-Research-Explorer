"""Contracts for fusion, missing judgments, offsets, filters and parser distinctions."""
import numpy as np
import pandas as pd
import pytest
from src.search_radar_core import (weighted_rrf,parse_query,condition_evidence,
    aggregate_chunks,evaluate_ranking,paired_bootstrap)


def test_rrf_uses_ranks_preserves_identity_and_rejects_duplicates():
    fused=weighted_rrf({'a':[('x',.001),('y',999.)],'b':[('y',-.5)]},{'a':1.,'b':2.})
    assert fused[0][0]=='y'
    assert fused[0][1]==pytest.approx(1/62+2/61)
    assert fused[0][2]['a']['score']==999.
    with pytest.raises(ValueError):weighted_rrf({'a':[('x',1.),('x',2.)]})
    with pytest.raises(ValueError):weighted_rrf({'a':[]},{'a':float('nan')})


def test_parser_preserves_formulas_and_constraint_types():
    query='Na3V2(PO4)3 P2/O3 cathodes with carbon coating without electrolyte additives'
    parsed=parse_query(query)
    assert 'Na3V2(PO4)3' in parsed['chemical_formulas']
    assert set(['P2','O3'])<=set(parsed['chemical_formulas'])
    assert parsed['electrode_role']==['cathode']
    assert 'electrolyte additive' in parsed['negated_concepts']
    assert 'particle coating' in parsed['processes']
    parsed=parse_query('NaFSI ether electrolytes for low-temperature sodium-ion batteries')
    assert parsed['salt']==['NaFSI'] and parsed['materials']==['ether']
    assert parsed['experimental_conditions']==['low temperature']
    assert parsed['required_relationships'][0]['scope']=='same_document'
    parsed=parse_query('Hard carbon without NaFSI')
    assert 'NaFSI' in parsed['negated_concepts']


@pytest.mark.parametrize('query,concept,category',[
    ('Anode half cell','anode','electrode_role'),('Cathode full cell','cathode','electrode_role'),
    ('Electrode coating equipment','electrode coating','processes'),
    ('Sacrificial cathode additive','sacrificial cathode additive','processes'),
    ('Recycling spent sodium-ion batteries','spent battery recycling','processes'),
    ('Lithium-ion recycling','lithium-ion','battery_system'),
    ('Thermal management of battery packs','thermal management','processes'),
    ('Voltage balancing','voltage balancing','processes')])
def test_parser_distinctions(query,concept,category):
    assert concept in parse_query(query)[category]


def test_evidence_missing_is_not_contradiction_and_offsets_are_exact():
    parsed=parse_query('NaFSI ether low-temperature sodium-ion')
    abstract='NaFSI is used in ether. Sodium-ion cells were evaluated.'
    evidence=condition_evidence(parsed,'',abstract)
    assert evidence['contradicted']==0
    assert 'low temperature' in evidence['missing_conditions']
    assert not evidence['same_sentence_lexical_support']
    for condition in evidence['conditions']:
        for span in condition['spans']:
            assert abstract[span['start']:span['end']]==span['text']
    contradiction=condition_evidence(parse_query('ether'),'','Without ether, cells operate.')
    assert contradiction['conditions'][0]['status']=='contradicted'
    assert condition_evidence(parsed,'NaFSI','')['abstract_available'] is False


def test_chunk_overlap_does_not_double_count_and_outputs_unique_docs():
    rows=[dict(doc_id='a',token_start=0,token_end=10),dict(doc_id='a',token_start=5,token_end=15),
          dict(doc_id='b',token_start=0,token_end=10)]
    assert len(aggregate_chunks([.6,.9,.7],rows))==2
    mean=dict((d,s) for d,s,r in aggregate_chunks([.6,.9,.7],rows,'unique_mean'))
    assert mean['a']==pytest.approx((.6*10+.9*5)/15)
    with pytest.raises(ValueError):aggregate_chunks([.1,.2],[dict(doc_id='a',token_start=0,token_end=2),dict(doc_id='a',token_start=3,token_end=4)])


def test_unjudged_metrics_withheld_zero_grade2_recall_undefined():
    missing=evaluate_ranking(['a','unjudged'],{'a':2},['a'])
    assert missing['Precision_at_10'] is None and missing['status']=='PENDING_LABELS'
    assert missing['Candidate_Recall_pool']==1
    valid=evaluate_ranking(['a','b','c'],{'a':0,'b':1,'c':2},['b'])
    assert valid['Precision_at_10']==.1 and valid['MRR_at_10']==pytest.approx(1/3)
    q08=evaluate_ranking(['x'],{'x':1},['x'])
    assert q08['nDCG_at_10']==1 and q08['Precision_at_10']==0
    assert q08['Candidate_Recall_pool'] is None
    assert evaluate_ranking([], {})['Precision_at_10'] is None


def test_bootstrap_requires_paired_common_queries():
    frame=pd.DataFrame([dict(method='A0',query_id='q1',Precision_at_10=.1,nDCG_at_10=.5,MRR_at_10=.2),
                        dict(method='A1',query_id='q2',Precision_at_10=.1,nDCG_at_10=.5,MRR_at_10=.2)])
    with pytest.raises(ValueError):paired_bootstrap(frame)
    frame.loc[1,'query_id']='q1'
    assert all(r['delta']==0 for r in paired_bootstrap(frame,repetitions=10))


def test_pilot_query_families_disjoint_and_never_human_labeled():
    from scripts.run_search_radar import pilot_queries
    queries=pilot_queries()
    assert set(queries[queries.split.eq('dev')].family).isdisjoint(queries[queries.split.eq('heldout')].family)
    assert queries.query_id.is_unique and not queries.expert_verified.any()


def test_same_mask_for_all_branches_with_source_year_topic_and_required_terms():
    from src.search_radar_retrieval import RadarRetriever
    from types import SimpleNamespace
    fake=RadarRetriever.__new__(RadarRetriever)
    fake.ids=['a','b','c','d']
    fake.engine=SimpleNamespace(v2=SimpleNamespace(allowed=lambda source,years:np.array([True,True,False,True]),
        evidence_filter=lambda terms:np.array([True,False,True,True])),
        _research_topics=lambda:{d:dict(topic_id=i) for d,i in zip(fake.ids,[1,1,1,2])})
    assert fake.mask('Publication',(2020,2023),['NaFSI'],1).tolist()==[True,False,False,False]
    with pytest.raises(ValueError):fake.mask(topic_id=True)
