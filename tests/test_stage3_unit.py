import numpy as np
import pytest
from src.stage3_common import group_retrieval_metrics,signature

def test_group_retrieval_perfect():
    q=np.eye(3,dtype='float32')
    m,rows=group_retrieval_metrics(q,q,['a','b','c'])
    assert m['group_mrr_at_10']==m['group_recall_at_1']==1
    assert rows.group_rank.tolist()==[1,1,1]

def test_related_abstract_is_relevant_and_ties_are_pessimistic():
    q=np.array([[1.,0.],[1.,0.],[0.,1.]])
    d=np.array([[0.,1.],[1.,0.],[0.,1.]])
    m,rows=group_retrieval_metrics(q,d,['family','family','other'])
    assert rows.group_rank.tolist()==[1,1,2]
    assert m['group_mrr_at_10']==pytest.approx(5/6)
    assert m['group_recall_at_1']==pytest.approx(2/3)

def test_fingerprint_stable_and_sensitive():
    assert signature({'a':1,'b':2})==signature({'b':2,'a':1})
    assert signature({'a':1})!=signature({'a':2})

def test_configured_checkpoint_metric_maps_to_validation_result_key():
    from src.stage3_common import config
    key=config()['training']['selection_metric'].removeprefix('validation_')
    metrics,_=group_retrieval_metrics(np.eye(2),np.eye(2),['a','b'])
    assert key in metrics and metrics[key]==1

def test_false_negative_mask_controls_groups_and_near_titles():
    import pandas as pd
    from src.stage3_neural import false_negative_mask
    f=pd.DataFrame(dict(title=['hard carbon sodium storage','hard carbon sodium storage','organic electrolyte','different words'],split_group_id=['a','b','c','c']))
    mask=false_negative_mask(f,.8)
    assert mask[0,1] and mask[2,3]
    assert np.array_equal(mask,mask.T) and not mask.diagonal().any()
    assert not mask[0,2]

def test_bertopic_preserves_stage1_domain_tokens():
    from src.stage3_bertopic_model import FrozenTokensBERTopic
    model=FrozenTokensBERTopic(embedding_model=None)
    text='sodium_ion hard_carbon Na+ P2 Na3V2(PO4)3 na-ion'
    assert model._preprocess_text(np.array([text]))==[text]
