"""Read-only topic catalog for an explicitly accepted full-document model."""
import json
import pandas as pd
from .full_document_inference import OUT

NAME='BatteryBERT full-document · trained'


def available():
    selection=OUT/'selection.json'
    return (selection.exists() and json.loads(selection.read_text(encoding='utf8')).get('accepted',False)
        and all((OUT/'topics'/name).exists() for name in ['final_metrics.json','topic_assignments.csv','topic_keywords.csv','topic_labels.csv','representative_documents.csv']))


def ready():
    path=OUT/'deployment.json'
    return path.exists() and json.loads(path.read_text(encoding='utf8')).get('activate_topics',False)


def assignments(expected_ids=None):
    folder=OUT/'topics';frame=pd.read_csv(folder/'topic_assignments.csv')
    labels=pd.read_csv(folder/'topic_labels.csv').set_index('topic_id').topic_label.to_dict()
    if not frame.doc_id.is_unique or set(frame.topic_id)!=set(range(10)):raise ValueError('Invalid full-document topic membership')
    if expected_ids is not None and frame.doc_id.tolist()!=list(expected_ids):raise ValueError('Full-document topic ID order mismatch')
    frame['topic_label']=frame.topic_id.map(labels)
    if frame.topic_label.isna().any():raise ValueError('Missing full-document topic label')
    return frame


def topic_data():
    folder=OUT/'topics';frame=assignments();metrics=json.loads((folder/'final_metrics.json').read_text())
    return dict(assignments=frame,summary=pd.read_csv(folder/'topic_distribution.csv'),
        keywords=pd.read_csv(folder/'topic_keywords.csv'),
        representatives=pd.read_csv(folder/'representative_documents.csv').query("sample_role == 'central'"),
        sources=frame.groupby(['topic_id','source']).size().rename('documents').reset_index(),
        years=frame.dropna(subset=['year']).groupby(['topic_id','year']).size().rename('documents').reset_index(),
        labels=frame.drop_duplicates('topic_id').set_index('topic_id').topic_label.to_dict(),
        metrics=dict(topics=10,c_v=metrics['c_v'],topic_diversity=metrics['topic_diversity'],outlier_ratio=0.))
