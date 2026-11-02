"""Read-only catalog for the evaluated BatteryBERT topic pipeline."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/batterybert_optimization_20261023_v1'
NAME = 'BatteryBERT fine-tuned · optimized'


def ready():
    marker = OUT / 'deployment.json'
    return marker.exists() and json.loads(marker.read_text(encoding='utf8')).get('activate_as_main', False)


def assignments(expected_ids=None):
    frame = pd.read_csv(OUT / 'topic_assignments.csv')
    labels = pd.read_csv(OUT / 'topic_labels.csv').set_index('topic_id').topic_label.to_dict()
    if not frame.doc_id.is_unique or set(frame.topic_id) != set(range(10)):
        raise ValueError('Invalid optimized BatteryBERT assignments')
    if expected_ids is not None and frame.doc_id.tolist() != list(expected_ids):
        raise ValueError('Optimized topic order differs from search corpus')
    frame['topic_label'] = frame.topic_id.map(labels)
    if frame.topic_label.isna().any():
        raise ValueError('Missing optimized topic labels')
    return frame


def topic_data():
    assigned = assignments()
    keys = pd.read_csv(OUT / 'topic_keywords.csv')
    sources = assigned.groupby(['topic_id', 'source']).size().rename('documents').reset_index()
    years = assigned.dropna(subset=['year']).groupby(['topic_id', 'year']).size().rename('documents').reset_index()
    coherence = pd.read_csv(OUT / 'common_coherence.csv').set_index('model').loc['BatteryBERT_optimized']
    return dict(summary=pd.read_csv(OUT / 'topic_distribution.csv'), keywords=keys,
        representatives=pd.read_csv(OUT / 'representative_documents.csv').query("sample_role == 'central'"),
        assignments=assigned, sources=sources, years=years,
        labels=assigned.drop_duplicates('topic_id').set_index('topic_id').topic_label.to_dict(),
        metrics=dict(topics=10, c_v=float(coherence.c_v), topic_diversity=keys.token.nunique() / len(keys), outlier_ratio=0.))
