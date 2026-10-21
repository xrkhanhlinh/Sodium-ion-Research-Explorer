import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from src.stage3_common import ROOT,REPORT,TABLE,DATA,read_json,digest,verify_frozen,group_retrieval_metrics

def test_frozen_inputs_and_group_disjoint_splits():
    assert verify_frozen()>100
    p=pd.read_parquet(DATA/'contrastive_pairs.parquet')
    s=pd.read_parquet(DATA/'split_assignments.parquet')
    assert len(p)==13650 and len(s)==13702 and p.doc_id.is_unique
    assert p.groupby('split_group_id').split.nunique().max()==1
    assert s.groupby('split_group_id').split.nunique().max()==1
    assert s.eligible_finetune.sum()==13650 and (~s.eligible_finetune).sum()==52
    assert all(row['cross_split_exact_text_hashes']==0 for row in read_json(REPORT/'exact_text_leakage_audit.json'))
    assert all(row['cross_split_identical_token_sequences']==0 for row in read_json(REPORT/'token_sequence_leakage_audit.json'))

def test_real_pretrained_weights_updated_and_frozen_weights_unchanged():
    from safetensors import safe_open
    success=read_json(ROOT/'models/bert_finetuned/training_success.json')
    assert success['real_gradient_updates'] and success['selected_weight_delta_l2']>0
    assert success['frozen_hash_before']==success['frozen_hash_after']
    assert success['model_sha256']==digest(ROOT/'models/bert_finetuned/model.safetensors')
    assert 0<success['trainable_parameters']<success['total_parameters']
    with safe_open(str(ROOT/'models/bge_pretrained/model.safetensors'),framework='numpy') as base, safe_open(str(ROOT/'models/bert_finetuned/model.safetensors'),framework='numpy') as fine:
        # Transformers 5 does not persist the deterministic position_ids buffer.
        assert set(base.keys())-set(fine.keys()) <= {'embeddings.position_ids'}
        assert not (set(fine.keys())-set(base.keys()))
        unchanged=[k for k in fine.keys() if k.startswith('embeddings.') or k.startswith('encoder.layer.0.')]
        changed=[k for k in base.keys() if k.startswith('encoder.layer.11.')]
        assert unchanged and changed
        assert all(np.array_equal(base.get_tensor(k),fine.get_tensor(k)) for k in unchanged)
        assert any(not np.array_equal(base.get_tensor(k),fine.get_tensor(k)) for k in changed)

def test_full_epoch_and_validation_only_selection():
    s=read_json(ROOT/'models/bert_finetuned/training_success.json')
    history=pd.read_csv(TABLE/'training_loss.csv');validation=pd.read_csv(TABLE/'validation_metrics.csv')
    assert len(history)==s['planned_steps']==s['step']
    assert history.batch_pairs.sum()==s['rows_seen']==s['training_pairs']==10921
    assert np.isfinite(history.loss).all() and history.batch_pairs.min()>1
    assert not s['selection_uses_test'] and validation.split.eq('validation').all()
    best=validation.sort_values(['group_mrr_at_10','step'],ascending=[False,True]).iloc[0]
    assert s['best_step']==best.step and s['best_score']==pytest.approx(best.group_mrr_at_10)

def test_before_after_metrics_recomputed_on_identical_heldout_candidates():
    p=pd.read_parquet(DATA/'contrastive_pairs.parquet')
    for split in ['validation','test']:
        frame=p[p.split.eq(split)].reset_index(drop=True)
        for phase in ['pretrained','finetuned']:
            q=np.load(DATA/f'{phase}_{split}_titles.npy',mmap_mode='r');d=np.load(DATA/f'{phase}_{split}_abstracts.npy',mmap_mode='r')
            calculated,_=group_retrieval_metrics(q,d,frame.split_group_id.to_numpy(),frame.doc_id.to_numpy())
            saved=read_json(REPORT/f'{phase}_{split}_metrics.json')
            assert saved['queries']==saved['candidates']==len(frame)
            # Historical artifacts keep their original numerical values and
            # names. New Success keys describe those values without rewriting
            # old files; semantic metadata is deliberately non-numeric.
            historical_keys=['queries','candidates','group_mrr_at_10',
                'group_recall_at_1','group_recall_at_5','group_recall_at_10',
                'exact_mrr_at_10','exact_recall_at_10']
            for k in historical_keys:assert calculated[k]==pytest.approx(saved[k])
            for k in (1,5,10):
                assert calculated[f'group_success_at_{k}']==pytest.approx(saved[f'group_recall_at_{k}'])
                assert calculated['metric_semantics']['deprecated_aliases'][f'group_recall_at_{k}']==f'group_success_at_{k}'

def test_document_embeddings_ids_dimensions_and_real_model_inference():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    frame=pd.read_parquet(ROOT/'data/processed/bert_texts.parquet')
    stored=pd.read_parquet(DATA/'embeddings.parquet');arr=np.load(DATA/'embeddings.npy',mmap_mode='r')
    assert stored.doc_id.tolist()==frame.doc_id.tolist() and stored.doc_id.is_unique
    assert arr.shape==(13702,768) and np.isfinite(arr).all()
    assert np.allclose(np.linalg.norm(arr,axis=1),1,atol=1e-5)
    assert np.array_equal(stored.filter(regex='^emb_').to_numpy(),arr)
    model=SentenceTransformer(str(ROOT/'models/bert_finetuned'),device='cpu',local_files_only=True)
    assert model.max_seq_length==128
    indices=[0,5933,13701]
    fresh=model.encode(frame.text.iloc[indices].tolist(),normalize_embeddings=True,batch_size=3)
    assert np.allclose(fresh,arr[indices],atol=2e-5)
    import gc
    del model;gc.collect()

def test_bertopic_pipeline_assignments_metrics_and_saved_model():
    import gc
    from bertopic import BERTopic
    model=BERTopic.load(str(ROOT/'models/bertopic/seed42/model.pickle'))
    assert type(model.umap_model).__name__=='UMAP' and type(model.hdbscan_model).__name__=='HDBSCAN'
    assert model.nr_topics is None
    assignments=pd.read_parquet(DATA/'bertopic_assignments.parquet');main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    assert assignments.doc_id.tolist()==main.doc_id.tolist()
    assert np.array_equal(assignments.topic_id.to_numpy(),model.topics_)
    assert assignments.membership_strength.between(0,1).all()
    summary=pd.read_csv(TABLE/'bertopic_topic_summary.csv');metrics=read_json(REPORT/'bertopic_metrics.json')
    assert summary.documents.sum()==13702
    assert metrics['topics']==assignments.topic_id[assignments.topic_id.ne(-1)].nunique()
    assert metrics['outlier_ratio']==pytest.approx(assignments.topic_id.eq(-1).mean())
    tokens=pd.read_parquet(ROOT/'data/processed/lda_tokens.parquet')
    assert np.isfinite(model.c_tf_idf_.data).all() and np.isfinite(model.ctfidf_model._idf_diag.data).all()
    all_counts=model.vectorizer_model.transform([' '.join(t) for t in tokens.tokens])
    assert np.array_equal(np.asarray(all_counts.sum(axis=1)).ravel(),tokens.token_count.to_numpy())
    indices=[0,5933,13701]
    explicit=np.load(DATA/'embeddings.npy',mmap_mode='r')[indices]
    from joblib import parallel_backend
    with parallel_backend('sequential',n_jobs=1):
        predicted,strength=model.transform([' '.join(tokens.iloc[i]['tokens']) for i in indices],embeddings=explicit)
    assert len(predicted)==3 and set(predicted).issubset(set(model.topics_))
    assert np.asarray(strength).shape==(3,) and np.isfinite(strength).all()
    kw=pd.read_csv(TABLE/'bertopic_keywords.csv',keep_default_na=False)
    for topic in sorted(set(model.topics_)-{-1}):
        assert kw[kw.topic_id.eq(topic)].token.tolist()==[w for w,p in model.get_topic(topic)[:10]]
    assert 0<=metrics['c_v']<=1 and 0<metrics['topic_diversity']<=1
    assert read_json(REPORT/'bertopic_stability.json')
    del model;gc.collect()

def test_figures_source_tables_and_export_contract():
    figures=read_json(REPORT/'figure_manifest.json');assert len(figures)>=4
    for item in figures:
        with Image.open(ROOT/item['figure']) as im:
            assert im.width>=1000 and im.height>=800
            assert im.info['dpi'][0]>=299
        assert all((ROOT/p).exists() for p in item['tables'])
    reps=pd.read_csv(TABLE/'bertopic_representative_documents.csv')
    metrics=read_json(REPORT/'bertopic_metrics.json')
    assert len(reps)==3*metrics['topics']
    assert reps.groupby('topic_id').doc_id.nunique().eq(3).all()
    compare=pd.read_csv(TABLE/'lda_vs_bertopic.csv')
    assert len(compare)==2 and compare.iloc[0].topics==20
