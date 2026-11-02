"""Real saved run and exported BERT checks; no synthetic relevance labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/batterybert_full_document_20261024_v1'
pytestmark=pytest.mark.skipif(not (OUT/'embedding_success.json').exists(),reason='Full-document training and export must finish')


@pytest.fixture(scope='module')
def corpus():return pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')


@pytest.fixture(scope='module')
def encoder():
    from src.full_document_inference import FullDocumentEncoder
    return FullDocumentEncoder()


def test_actual_optimizer_coverage_and_frozen_training_protocol(corpus):
    from src.batterybert_full_document import sha
    rows=pd.read_csv(OUT/'training_document_coverage.csv')
    assert rows.doc_id.tolist()==corpus.doc_id.tolist()
    assert rows.loc[rows.split.eq('train'),'optimizer_appearances'].eq(1).all()
    assert rows.loc[~rows.split.eq('train'),'optimizer_appearances'].eq(0).all()
    assert rows.optimizer_appearances.sum()==10921
    lengths=pd.read_parquet(ROOT/'outputs/model_comparison_v2/batterybert_pretrained/document_chunk_counts.parquet')
    assert lengths.covered_abstract_tokens.eq(lengths.abstract_tokens).all()
    assert not lengths.title_truncated.any()
    audit=json.loads((OUT/'training_coverage.json').read_text())
    assert audit['status']=='PASS' and audit['actual_unique_abstract_tokens']==3420215
    assert audit['actual_chunks_backpropagated']==26147
    protocol=json.loads((OUT/'training_protocol.json').read_text())
    for relative,expected in protocol['source_hashes'].items():assert sha(ROOT/relative)==expected
    for relative,expected in protocol['inputs'].items():assert sha(ROOT/relative)==expected
    success=json.loads((OUT/'embedding_success.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    assert success['new_training_accepted']==selection['accepted']
    assert (success['selected_last_block_delta_l2']>0)==selection['accepted']
    assert success['article_full_text_available'] is False


def test_exported_full_document_encoder_reproduces_longest_and_title_only(corpus,encoder):
    counts=pd.read_parquet(ROOT/'outputs/model_comparison_v2/batterybert_pretrained/document_chunk_counts.parquet')
    matrix=np.load(OUT/'document_embeddings.npy',mmap_mode='r')
    indices=sorted(set([int(counts.abstract_tokens.argmax()),int(counts.chunk_count.argmax()),int(np.flatnonzero(counts.title_only)[0])]))
    checks=[]
    for i in indices:
        row=corpus.iloc[i];body='' if pd.isna(row.abstract) else str(row.abstract)
        vector,metadata=encoder.encode_document(row.title,body)
        np.testing.assert_allclose(vector,matrix[i],atol=2e-5,rtol=2e-5)
        assert metadata['covered_body_tokens']==metadata['body_tokens']==int(counts.iloc[i].abstract_tokens)
        assert metadata['body_truncated'] is False
        checks.append(dict(doc_id=row.doc_id,**metadata,max_error=float(np.max(np.abs(vector-matrix[i])))))
    (OUT/'inference_api_parity.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2),encoding='utf8')


def test_exported_query_encoder_matches_reused_validation_prefix(corpus,encoder):
    selection=json.loads((OUT/'selection.json').read_text())
    folder=(OUT/'experiments'/selection['name']/f"validation_step_{selection['best_step']}") if selection['best_step'] else OUT/'baseline/validation'
    expected=np.load(folder/'query_vectors.npy',mmap_mode='r')
    ranks=pd.read_csv(folder/'ranks.csv');meta=corpus.set_index('doc_id').loc[ranks.doc_id]
    probes=sorted(set([0,len(meta)-1,int(np.argmax(meta.title.map(len).to_numpy()))]))
    checks=[]
    for i in probes:
        actual,metadata=encoder.encode(meta.iloc[i].title)
        np.testing.assert_allclose(actual,expected[i],atol=2e-5,rtol=2e-5)
        checks.append(dict(doc_id=meta.index[i],max_error=float(np.max(np.abs(actual-expected[i]))),**metadata))
    alignment=json.loads((OUT/'validation_prefix_title_alignment.json').read_text())
    assert alignment['exact_title_matches']==alignment['validation_queries']==1361
    assert not alignment['title_mismatch_ids']
    (OUT/'validation_query_api_parity.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2),encoding='utf8')


def test_real_long_query_tail_changes_embedding_with_same_first_128_tokens(encoder):
    prefix='Hard carbon anodes for sodium ion batteries. '*80
    query_a=prefix+'Vanadium phosphate NASICON cathodes. '*50
    query_b=prefix+'Polymer electrolyte solvent additives. '*50
    tokens_a=encoder.tokenizer(query_a,add_special_tokens=False,truncation=False)['input_ids']
    tokens_b=encoder.tokenizer(query_b,add_special_tokens=False,truncation=False)['input_ids']
    assert tokens_a[:128]==tokens_b[:128] and len(tokens_a)>512 and len(tokens_b)>512
    a,meta_a=encoder.encode(query_a);b,meta_b=encoder.encode(query_b)
    assert meta_a['tokens']==len(tokens_a) and meta_b['tokens']==len(tokens_b)
    assert meta_a['coverage']==meta_b['coverage']==1.
    assert meta_a['chunks']>1 and meta_b['chunks']>1
    assert not meta_a['truncated'] and not meta_b['truncated']
    delta=float(np.linalg.norm(a-b));assert delta>1e-4
    (OUT/'long_query_tail_probe.json').write_text(json.dumps(dict(status='PASS',
        same_first_128_native_tokens=True,a=meta_a,b=meta_b,embedding_l2_difference=delta,
        independent_relevance_accuracy=False),indent=2),encoding='utf8')


def test_reloaded_topic_pipeline_only_fit_train_and_predicts_all(corpus):
    import joblib
    pipeline=joblib.load(OUT/'topics/topic_pipeline.joblib')
    splits=pd.read_parquet(ROOT/'data/stage3/split_assignments.parquet')
    assert set(pipeline.ordered_training_ids)==set(splits.loc[splits.split.eq('train'),'doc_id'])
    assigned=pd.read_csv(OUT/'topics/topic_assignments.csv')
    assert assigned.doc_id.tolist()==corpus.doc_id.tolist()
    matrix=np.load(OUT/'document_embeddings.npy',mmap_mode='r')
    for at in range(0,len(matrix),512):
        np.testing.assert_array_equal(pipeline.predict(matrix[at:at+512]),assigned.topic_id.to_numpy()[at:at+512])


def test_new_topic_keywords_have_exact_train_support():
    from src.batterybert_optimization import ContiguousScientificVectorizer
    tokens=pd.read_parquet(OUT/'topics/scientific_token_stream.parquet')
    split=pd.read_parquet(ROOT/'data/stage3/split_assignments.parquet').set_index('doc_id').loc[tokens.doc_id].split.to_numpy()
    text=tokens.tokens.tolist()
    keys=pd.read_csv(OUT/'topics/topic_keywords.csv');vocab={w:i for i,w in enumerate(sorted(set(keys.token)))}
    counts=ContiguousScientificVectorizer(vocabulary=vocab,binary=True,ngram_range=(1,2),lowercase=False).transform(
        [' '.join(t) for t,use in zip(text,split=='train') if use])
    assigned=pd.read_csv(OUT/'topics/topic_assignments.csv').topic_id.to_numpy()[split=='train']
    assert len(keys)==100 and keys.groupby('topic_id').size().eq(10).all()
    for row in keys.itertuples():
        assert int(counts[assigned==row.topic_id,vocab[row.token]].sum())==row.train_document_support>=5
