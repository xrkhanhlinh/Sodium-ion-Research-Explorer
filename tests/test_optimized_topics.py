"""Contracts on real frozen train/test artifacts and the mounted topic catalog."""
import json
import joblib
import numpy as np
import pandas as pd
from src import optimized_topics as topic
from src.batterybert_optimization import ContiguousScientificVectorizer
from src.batterybert_topic_tools import sha, repair_phrase_placeholders


def test_catalog_covers_every_real_document_and_labels():
    main = pd.read_parquet(topic.ROOT / 'data/processed/main_corpus.parquet')
    frame = topic.assignments(main.doc_id)
    assert frame.shape[0] == frame.doc_id.nunique() == 13702
    assert frame.topic_label.notna().all()
    data = topic.topic_data()
    assert data['summary'].documents.sum() == data['sources'].documents.sum() == 13702
    assert data['keywords'].groupby('topic_id').size().eq(10).all()
    assert data['metrics']['topics'] == 10


def test_reloaded_pipeline_predictions_and_training_groups():
    pipeline = joblib.load(topic.OUT / 'topic_pipeline.joblib')
    x = np.load(topic.ROOT / 'outputs/final_system/batterybert_finetuned/document_embeddings.npy', mmap_mode='r')
    labels = topic.assignments().topic_id.to_numpy()
    for start in range(0, len(x), 512):
        np.testing.assert_array_equal(pipeline.predict(x[start:start + 512]), labels[start:start + 512])
    splits = pd.read_parquet(topic.ROOT / 'data/stage3/split_assignments.parquet')
    assert set(pipeline.ordered_training_ids) == set(splits.loc[splits.split.eq('train'), 'doc_id'])
    assert not set(pipeline.ordered_training_ids) & set(splits.loc[splits.split.isin(['validation', 'test']), 'doc_id'])


def test_all_keywords_have_exact_training_document_support():
    import yaml
    tokens = pd.read_parquet(topic.ROOT / 'data/processed/lda_tokens.parquet')
    split = pd.read_parquet(topic.ROOT / 'data/stage3/split_assignments.parquet').set_index('doc_id').loc[tokens.doc_id].split.to_numpy()
    phrases = yaml.safe_load((topic.ROOT / 'configs/config.yaml').read_text(encoding='utf8'))['preprocessing']['domain_phrases']
    texts, _ = repair_phrase_placeholders(tokens.tokens.tolist(), phrases)
    keys = pd.read_csv(topic.OUT / 'topic_keywords.csv')
    vocab = {w: i for i, w in enumerate(sorted(set(keys.token)))}
    analyzer = ContiguousScientificVectorizer(vocabulary=vocab, binary=True, ngram_range=(1, 2), lowercase=False)
    x = analyzer.transform([' '.join(t) for t, use in zip(texts, split == 'train') if use])
    labels = topic.assignments().topic_id.to_numpy()[split == 'train']
    for row in keys.itertuples():
        assert int(x[labels == row.topic_id, vocab[row.token]].sum()) == row.train_document_support
        assert row.train_document_support >= 5


def test_frozen_data_and_vectors_still_match_original_run():
    manifest = json.loads((topic.OUT / 'inputs.json').read_text())
    checked = 0
    for path, expected in manifest.items():
        if path.replace('\\', '/').startswith(('data/', 'outputs/')):
            assert sha(topic.ROOT / path) == expected, path
            checked += 1
    assert checked == 9


def test_final_test_gate_and_probe_provenance():
    report = json.loads((topic.OUT / 'final_test.json').read_text())
    assert report['documents'] == 1368 and all(report['gates'].values())
    assert report['technical_gate_pass']
    assert report['selection']['selected_on'] == 'validation only'
    deployment = json.loads((topic.OUT / 'deployment.json').read_text())
    assert deployment['independent_validation'] is False


def test_search_uses_the_same_active_assignments_as_topics():
    from src.final_search import FinalSearchEngine
    engine = FinalSearchEngine.__new__(FinalSearchEngine)
    from src import full_document_topics
    expected = (full_document_topics if full_document_topics.ready() else topic).assignments()
    engine.ids = expected.doc_id.tolist(); engine.research_topic_map = None
    mapping = engine._research_topics()
    assert [mapping[r.doc_id]['topic_id'] for r in expected.itertuples()] == expected.topic_id.tolist()
    assert [mapping[r.doc_id]['label'] for r in expected.itertuples()] == expected.topic_label.tolist()


def test_topic_dashboard_defaults_to_the_optimized_catalog():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(topic.ROOT / 'app.py'), default_timeout=180).run()
    assert not app.exception
    app.sidebar.radio[0].set_value('Topics').run()
    assert not app.exception
    from src import full_document_topics
    assert app.radio(key='topic_model').value == (full_document_topics.NAME if full_document_topics.ready() else topic.NAME)
    assert any(m.label == 'Topics' and m.value == '10' for m in app.metric)
    assert any('ARI seed mới' in c.value for c in app.caption)
    app.sidebar.radio[0].set_value('Comparison').run()
    assert not app.exception
    assert any('test riêng' in h.value for h in app.subheader)
