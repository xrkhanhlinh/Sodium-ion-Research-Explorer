import numpy as np
import pytest
from src.stage2 import match_stability, fingerprint


def test_stability_invariant_to_topic_permutation():
    topics=[{'Na+','cathode','oxide'},{'hard_carbon','anode','storage'}]
    value,pairs=match_stability(topics,list(reversed(topics)))
    assert value==1.0
    assert {(a,b) for a,b,v in pairs}=={(0,1),(1,0)}


def test_stability_disjoint_topics_and_size_guard():
    value,_=match_stability([{'a'},{'b'}],[{'c'},{'d'}])
    assert value==0.0
    with pytest.raises(ValueError): match_stability([{'a'}],[{'a'},{'b'}])


def test_fingerprint_mapping_order_independent():
    assert fingerprint({'seed':42,'k':10})==fingerprint({'k':10,'seed':42})
    assert fingerprint({'seed':42,'k':10})!=fingerprint({'seed':123,'k':10})
