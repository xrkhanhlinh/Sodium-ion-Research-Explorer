"""Synthetic counterexamples; no models, training, or historical output writes."""
import ast
from pathlib import Path

import numpy as np
import pytest

from src.stage3_common import (
    group_retrieval_metrics, require_cache_fingerprint, signature,
    validate_frozen_validation_settings, write_json,
)


def test_success_is_not_multi_positive_document_recall():
    queries=np.array([[.9,.2,.8],[.2,.9,.8],[.2,.3,.9]])
    metrics,rows=group_retrieval_metrics(queries,np.eye(3),['A','A','B'])
    # Each query's highest-scored document is relevant; only one of two A
    # documents can occupy position one. Hand-counted Recall@1 is 2/3.
    assert metrics['group_success_at_1']==1
    assert metrics['group_success_at_1']!=pytest.approx((.5+.5+1)/3)
    assert rows.relevant_documents.tolist()==[2,2,1]
    for k in (1,5,10):
        assert metrics[f'group_recall_at_{k}']==metrics[f'group_success_at_{k}']
        assert metrics['metric_semantics']['deprecated_aliases'][f'group_recall_at_{k}']==f'group_success_at_{k}'


def test_pessimistic_group_ties_exclude_other_relevant_documents():
    q=np.ones((3,1));d=np.ones((3,1))
    metrics,rows=group_retrieval_metrics(q,d,['A','A','B'])
    assert rows.group_rank.tolist()==[2,2,3]
    assert rows.exact_rank.tolist()==[3,3,3]
    assert metrics['group_success_at_1']==0
    assert metrics['group_mrr_at_10']==pytest.approx((.5+.5+1/3)/3)


@pytest.mark.parametrize('layers',[0,2,11,12,True,1.0,None,'1'])
def test_unsupported_trainable_layers_rejected(layers):
    with pytest.raises(ValueError,match='requires trainable_encoder_layers=1'):
        validate_frozen_validation_settings({'trainable_encoder_layers':layers})


def test_default_single_trainable_layer_allowed():
    validate_frozen_validation_settings({'trainable_encoder_layers':1})
    validate_frozen_validation_settings({'trainable_encoder_layers':np.int64(1)})


@pytest.mark.parametrize('function_name',['train','encode_validation_frozen_cache'])
def test_neural_entrypoints_reject_before_model_access(function_name):
    # Compile the actual function body without importing torch or loading models.
    tree=ast.parse((Path(__file__).parents[1]/'src/stage3_neural.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function_name)
    scope={'CFG':{'training':{'trainable_encoder_layers':2}},
           'validate_frozen_validation_settings':validate_frozen_validation_settings}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<neural-entrypoint>','exec'),scope)
    with pytest.raises(ValueError,match='requires trainable_encoder_layers=1'):
        scope[function_name](*([] if function_name=='train' else [None,[],'titles']))


def test_absent_cache_needs_no_legacy_metadata(tmp_path):
    assert require_cache_fingerprint(tmp_path/'new.json','expected') is None


def test_matching_metrics_cache_preserved(tmp_path):
    path=tmp_path/'metrics.json'
    payload={'cache_fingerprint':signature({'model':'M1','tokens':'T1'}),'c_v':.75}
    write_json(path,payload);before=path.read_bytes()
    assert require_cache_fingerprint(path,payload['cache_fingerprint'])==payload
    assert path.read_bytes()==before


@pytest.mark.parametrize('payload',[{'c_v':.75},{'cache_fingerprint':'other','c_v':.75}])
def test_legacy_or_mismatched_metrics_refused_without_rewrite(tmp_path,payload):
    path=tmp_path/'metrics.json';write_json(path,payload);before=path.read_bytes()
    with pytest.raises(ValueError,match='isolated output directory'):
        require_cache_fingerprint(path,'expected')
    assert path.read_bytes()==before


def test_projection_requires_matching_sidecar(tmp_path):
    path=tmp_path/'projection.npy';np.save(path,np.eye(2))
    marker=path.with_suffix('.cache.json');before=path.read_bytes()
    with pytest.raises(ValueError,match='Unverified'):
        require_cache_fingerprint(path,'embedding-order-params',marker)
    write_json(marker,{'cache_fingerprint':'embedding-order-params'})
    assert require_cache_fingerprint(path,'embedding-order-params',marker)['cache_fingerprint']=='embedding-order-params'
    with pytest.raises(ValueError,match='mismatched'):
        require_cache_fingerprint(path,'changed-embedding-or-order',marker)
    assert path.read_bytes()==before


@pytest.mark.parametrize('component',['model','representation','dictionary','bow','implementation','version'])
def test_topic_metric_identity_tracks_actual_evaluation_inputs(component):
    from types import SimpleNamespace
    tree=ast.parse((Path(__file__).parents[1]/'src/stage3_topics.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics_cache_identity')
    state={'dictionary':'D1','bow':'B1','implementation':'code1','version':'1'}
    scope={'ROOT':Path('/synthetic'),'signature':signature,
           'digest':lambda p:state['dictionary' if p.name=='dictionary.gensim' else 'bow'],
           'inspect':SimpleNamespace(getsource=lambda f:state['implementation']),
           'importlib':SimpleNamespace(metadata=SimpleNamespace(version=lambda n:state['version'])),
           'run':lambda:None,'topwords':lambda:None}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<topic-identity>','exec'),scope)
    identity=scope['metrics_cache_identity']
    before=identity('M1','R1')
    if component=='model':after=identity('M2','R1')
    elif component=='representation':after=identity('M1','R2')
    else:
        state[component]='changed';after=identity('M1','R1')
    assert before!=after
