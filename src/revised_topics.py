"""Read-only revised catalog, activated only after independent-seed checks.

Original model outputs remain immutable. Semantic names are explicitly AI
interpretations and must not be confused with expert labels or accuracy.
"""
from pathlib import Path
import json
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research_revision_20261025_v1'
CATALOG=OUT/'final_topics'
NAME='BatteryBERT full-document · stable topics'


def _read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def available():
    required=[CATALOG/name for name in ['artifact_manifest.json','topic_assignments.csv','topic_keywords.csv','final_metrics.json']]
    required+=[OUT/'review/topic_labels.csv',OUT/'decisions/geometry.json']
    return all(path.is_file() for path in required) and _read(OUT/'decisions/geometry.json').get('accepted') is True


def ready():
    marker=OUT/'deployment.json'
    if not marker.exists() or not available():return False
    info=_read(marker)
    return (info.get('activate_revised_topics') is True and info.get('topic_geometry_gate_accepted') is True
        and info.get('topic_catalog_manifest_sha256')==_sha(CATALOG/'artifact_manifest.json')
        and info.get('topic_label_review_sha256')==_sha(OUT/'review/topic_labels.csv'))


def assignments(expected_ids=None):
    frame=pd.read_csv(CATALOG/'topic_assignments.csv')
    labels=pd.read_csv(OUT/'review/topic_labels.csv').set_index('topic_id').topic_label
    if not frame.doc_id.is_unique or set(frame.topic_id)!=set(range(10)):raise ValueError('Invalid revised topic assignments')
    if expected_ids is not None and frame.doc_id.tolist()!=list(expected_ids):raise ValueError('Revised topic order mismatch')
    frame['topic_label']=frame.topic_id.map(labels)
    if frame.topic_label.isna().any():raise ValueError('Missing revised topic name')
    return frame


def topic_data():
    frame=assignments();metrics=_read(CATALOG/'final_metrics.json')
    return dict(assignments=frame,summary=pd.read_csv(CATALOG/'topic_distribution.csv'),
       keywords=pd.read_csv(CATALOG/'topic_keywords.csv'),
       keyword_audit=pd.read_csv(CATALOG/'keyword_audit.csv'),
       representatives=pd.read_csv(CATALOG/'representative_documents.csv').query("sample_role == 'central'"),
       sources=frame.groupby(['topic_id','source']).size().rename('documents').reset_index(),
       years=frame.dropna(subset=['year']).groupby(['topic_id','year']).size().rename('documents').reset_index(),
       labels=frame.drop_duplicates('topic_id').set_index('topic_id').topic_label.to_dict(),
       metrics=dict(topics=10,c_v=metrics['c_v'],topic_diversity=metrics['topic_diversity'],outlier_ratio=0.),
       full_metrics=metrics)
