"""Scientific contracts: no group leakage, padding contamination or lost rows."""
import os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'

import numpy as np
import pandas as pd
import pytest
import torch
from src.batterybert import masked_mean, false_negative_mask, contrastive_loss, batches_for
from src.model_comparison import validate_vectors, reduced_topic_count


def test_pooling_ignores_padding_and_is_unit_length():
    hidden = torch.tensor([[[1., 0.], [0., 1.], [900., -900.]]])
    mask = torch.tensor([[1, 1, 0]])
    vector = masked_mean(hidden, mask)
    assert torch.allclose(vector, torch.tensor([[2 ** -.5, 2 ** -.5]]))
    assert torch.allclose(vector, masked_mean(hidden[:, :2], mask[:, :2]))


def test_real_tokenizer_accepts_parquet_numpy_token_ids():
    from src.batterybert import BatteryEncoder
    from src.model_comparison import ROOT
    from transformers import AutoTokenizer
    adapter = BatteryEncoder.__new__(BatteryEncoder)
    adapter.tokenizer = AutoTokenizer.from_pretrained(
        ROOT / 'outputs/model_comparison_v2/batterybert_pretrained/model', local_files_only=True)
    real = adapter.tokenizer('Sodium-ion storage in hard carbon')
    records = [{key: np.asarray(real[key], dtype=np.int64) for key in ('input_ids', 'token_type_ids')}]
    features = adapter.record_features(records)
    assert features['input_ids'].tolist() == [real['input_ids']]
    assert features['attention_mask'].sum().item() == len(real['input_ids'])


@pytest.mark.parametrize('size', [2, 3, 5, 9, 10921])
def test_epoch_visits_every_pair_once_even_with_odd_tail(size):
    batches = batches_for(size, 2)
    assert sorted(np.concatenate(batches).tolist()) == list(range(size))
    assert all(2 <= len(b) <= 3 for b in batches)
    assert all(np.array_equal(a, b) for a, b in zip(batches, batches_for(size, 2)))


def test_same_group_and_near_duplicate_titles_are_not_negatives():
    frame = pd.DataFrame(dict(title=['Hard carbon sodium storage', 'Different title',
                                    'Hard carbon sodium storage', 'Electrolyte chemistry'],
                              split_group_id=['a', 'a', 'b', 'c']))
    mask = false_negative_mask(frame)
    assert mask[0, 1] and mask[0, 2] and not mask[0, 3]
    assert np.array_equal(mask, mask.T) and not mask.diagonal().any()


def test_masked_false_negative_contributes_no_gradient_to_similarity():
    q = torch.eye(2, requires_grad=True)
    d = torch.eye(2, requires_grad=True)
    all_same_group = np.array([[False, True], [True, False]])
    loss = contrastive_loss(q, d, all_same_group, .05)
    loss.backward()
    assert loss.item() == 0 and torch.equal(q.grad, torch.zeros_like(q))
    q.grad = None
    real_loss = contrastive_loss(q, d, np.zeros((2, 2), dtype=bool), 1.)
    real_loss.backward()
    assert real_loss.item() > 0 and q.grad.abs().sum() > 0


def test_vector_validation_rejects_order_nonfinite_and_unnormalized(tmp_path):
    path = tmp_path / 'embeddings.npy'
    ids = ['docA', 'docB']
    np.save(path, np.eye(2, dtype=np.float32))
    assert validate_vectors(path, ids, ids, 2)['shape'] == [2, 2]
    with pytest.raises(ValueError, match='IDs/order'):
        validate_vectors(path, ids[::-1], ids)
    np.save(path, np.array([[np.nan, 0.], [0., 1.]]))
    with pytest.raises(ValueError, match='Nonfinite'):
        validate_vectors(path, ids, ids)
    np.save(path, np.array([[2., 0.], [0., 1.]]))
    with pytest.raises(ValueError, match='normalized'):
        validate_vectors(path, ids, ids)


def test_reduction_api_target_reserves_outlier_slot_only_when_present():
    class Model:
        def __init__(self, topics):
            self.topics = topics
        def get_topics(self):
            return self.topics
    assert reduced_topic_count(Model({-1: [], 0: []}), 10) == 11
    assert reduced_topic_count(Model({0: [], 1: []}), 10) == 10


def test_document_prefix_cache_reuses_frozen_states_and_detects_corruption(tmp_path):
    from types import SimpleNamespace
    from src.batterybert import DocumentPrefixCache
    class TinyAdapter:
        dimension = 2
        calls = 0
        def __init__(self):
            self.folder = tmp_path / 'model'
            self.folder.mkdir()
            (self.folder / 'tokenizer.json').write_text('test-tokenizer')
            self.model = torch.nn.Module()
            self.model.embeddings = torch.nn.Embedding(8, 2)
            self.model.encoder = torch.nn.Module()
            self.model.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        def prefix(self, features):
            self.calls += 1
            return features['input_ids'].float().unsqueeze(-1).repeat(1, 1, 2)
    adapter = TinyAdapter()
    features = dict(input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]))
    cache = DocumentPrefixCache(tmp_path / 'cache', adapter, 'frozen-chunks', 3)
    first = cache.get(adapter, features, [0, 1])
    assert adapter.calls == 1 and not cache.valid.any()
    cache.flush()
    assert cache.valid[:2].all()
    assert torch.equal(first, cache.get(adapter, features, [0, 1])) and adapter.calls == 1
    # A final-block update is compatible; a frozen-block update is not.
    with torch.no_grad():
        adapter.model.encoder.layer[-1].weight.add_(1)
    reopened = DocumentPrefixCache(tmp_path / 'cache', adapter, 'frozen-chunks', 3)
    assert torch.equal(first, reopened.get(adapter, features, [0, 1]))
    with torch.no_grad():
        adapter.model.embeddings.weight.add_(1)
    with pytest.raises(ValueError, match='differs from frozen'):
        DocumentPrefixCache(tmp_path / 'cache', adapter, 'frozen-chunks', 3)
    cache.hidden[0, 0, 0] += 1
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        cache.get(adapter, features, [0, 1])


def test_saved_corpus_splits_are_disjoint_and_unchanged():
    from src.model_comparison import ROOT, cfg
    pairs = pd.read_parquet(ROOT / 'data/stage3/contrastive_pairs.parquet')
    assert pairs.groupby('split').size().to_dict() == cfg()['batterybert']['expected_splits']
    assert pairs.groupby('split_group_id')['split'].nunique().max() == 1


def test_resume_requires_both_identity_and_untampered_bytes(tmp_path):
    from run_model_comparison import verify_marker
    from src.stage3_common import digest, write_json
    artifact = tmp_path / 'result.npy'
    artifact.write_bytes(b'actual-result')
    marker = tmp_path / 'success.json'
    write_json(marker, dict(fingerprint='config-a', hashes={'result.npy': digest(artifact)}))
    verify_marker(marker, 'config-a', tmp_path)
    with pytest.raises(ValueError, match='code/config/inputs'):
        verify_marker(marker, 'config-b', tmp_path)
    artifact.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='integrity mismatch'):
        verify_marker(marker, 'config-a', tmp_path)


def test_atomic_checkpoint_retries_a_transient_sharing_violation(tmp_path, monkeypatch):
    from src import model_comparison_io as io
    target = tmp_path / 'checkpoint.json'
    io.write_json(target, {'completed': 32})
    original_replace = io.os.replace
    calls = []
    def temporarily_locked(source, destination):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError('Simulated Windows reader lock')
        return original_replace(source, destination)
    monkeypatch.setattr(io.os, 'replace', temporarily_locked)
    io.write_json(target, {'completed': 64})
    assert len(calls) == 3
    assert __import__('json').loads(target.read_text()) == {'completed': 64}


def test_atomic_replace_preserves_prior_checkpoint_on_permanent_denial(tmp_path, monkeypatch):
    from src import model_comparison_io as io
    target, pending = tmp_path / 'checkpoint.json', tmp_path / 'pending.tmp'
    target.write_text('previous-commit')
    pending.write_text('new-commit')
    def denied(*args):
        raise PermissionError('Permanent access denial')
    monkeypatch.setattr(io.os, 'replace', denied)
    with pytest.raises(PermissionError):
        io.replace_with_retry(pending, target, attempts=2, initial_delay=0)
    assert target.read_text() == 'previous-commit' and pending.read_text() == 'new-commit'


def test_atomic_checkpoint_survives_an_actual_open_reader(tmp_path):
    import json
    import threading
    import time
    from src.model_comparison_io import write_json
    target = tmp_path / 'live_checkpoint.json'
    write_json(target, {'completed': 32})
    opened = threading.Event()
    def hold_reader():
        with target.open('r', encoding='utf-8'):
            opened.set()
            time.sleep(.15)
    thread = threading.Thread(target=hold_reader)
    thread.start()
    assert opened.wait(timeout=2)
    try:
        write_json(target, {'completed': 64})
    finally:
        thread.join(timeout=2)
    assert json.loads(target.read_text()) == {'completed': 64}


def test_existing_lda_checkpoint_training_provenance_matches_frozen_inputs():
    import inspect
    import importlib.metadata
    import yaml
    from gensim import corpora
    from src.stage2 import _fit_lda, fingerprint
    from src.stage3_common import digest, read_json
    from src.model_comparison import ROOT
    config = yaml.safe_load((ROOT / 'configs/stage2.yaml').read_text())
    dictionary = corpora.Dictionary.load(str(ROOT / 'models/lda/dictionary.gensim'))
    inputs = fingerprint(dict(main=digest(ROOT / config['input_main']),
        tokens=digest(ROOT / config['input_tokens']), dictionary=dictionary.token2id))
    settings = {k: v for k, v in config['lda'].items()
                if k not in ('topics', 'base_seed', 'additional_seeds')}
    expected = fingerprint(dict(k=10, seed=42, settings=settings, input=inputs,
        gensim=importlib.metadata.version('gensim'), numpy=np.__version__,
        scipy=importlib.metadata.version('scipy'), trainer=inspect.getsource(_fit_lda)))
    assert expected == read_json(ROOT / 'models/lda/k10_seed42/training_success.json')['fingerprint']
