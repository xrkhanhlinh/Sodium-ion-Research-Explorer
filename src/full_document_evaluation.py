"""Independent human-label evaluation for the expanded six-method pool."""
import hashlib
import io
import pandas as pd
from .full_document_inference import OUT,METHODS
from .final_search import RESEARCH_METHODS
from .research_completion_evaluation import evaluate_dataset
from .final_system_evaluation import validate_labels


def evaluate(labels_path=None):
    return evaluate_dataset(OUT,tuple(RESEARCH_METHODS)+tuple(METHODS),labels_path,
        rankings_filename='comparison_rankings_top10.csv',runs_filename='comparison_run_manifest.csv')


def import_labels(raw):
    frame=pd.read_csv(io.BytesIO(raw),keep_default_na=False)
    validate_labels(frame,pd.read_csv(OUT/'annotation_pool.csv',keep_default_na=False),max_grade=2)
    path=OUT/('human_labels_'+hashlib.sha256(raw).hexdigest()[:16]+'.csv')
    if not path.exists():path.write_bytes(raw)
    return evaluate(path)
