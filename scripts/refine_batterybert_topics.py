"""Second development round: tune topic words on fixed fitted memberships.

The first round exposed formula lists with low pairwise co-occurrence. Adjust
frequency/exclusivity exponents, while retaining exact chemistry and phrases.
No test split is scored; the original numerical acceptance gates are retained.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import json
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.batterybert_optimization import ContiguousScientificVectorizer, document_npmi
from src.batterybert_topic_tools import GENERIC, PROTECTED, repair_phrase_placeholders, write_json


def run(out=None):
    out = Path(out) if out is not None else ROOT / 'outputs/batterybert_optimization_20261023_v1'
    dest = out / 'representation_refinement'; dest.mkdir(exist_ok=True)
    plan = dict(frequency_powers=[.5, .75, 1.], exclusivity_powers=[0., .25, .5],
        candidates=['finetuned_source', 'finetuned_source_whiten64'],
        gate='Original gates, unchanged; development only, test remains unopened')
    write_json(dest / 'plan.json', plan)
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet')
    texts = pd.read_parquet(ROOT / 'data/processed/lda_tokens.parquet')
    splits = pd.read_parquet(ROOT / 'data/stage3/split_assignments.parquet').set_index('doc_id').loc[main.doc_id].split.to_numpy()
    phrases = yaml.safe_load((ROOT / 'configs/config.yaml').read_text(encoding='utf8'))['preprocessing']['domain_phrases']
    tokens, _ = repair_phrase_placeholders(texts.tokens.tolist(), phrases)
    docs = np.asarray([' '.join(t) for t in tokens], dtype=object)
    train, val = splits == 'train', splits == 'validation'
    vectors = np.load(ROOT / 'outputs/final_system/batterybert_finetuned/document_embeddings.npy', mmap_mode='r')
    stop = sorted((set(ENGLISH_STOP_WORDS) | set(GENERIC)) - PROTECTED)
    vectorizer = ContiguousScientificVectorizer(lowercase=False, ngram_range=(1, 2), min_df=5,
        max_df=.6, max_features=40000, stop_words=stop)
    counts = vectorizer.fit_transform(docs[train]).tocsr()
    binary = counts.copy(); binary.data[:] = 1
    corpus_df = np.asarray(binary.sum(0)).ravel()
    words = vectorizer.get_feature_names_out()
    rows = []
    for name in plan['candidates']:
        model = joblib.load(out / 'experiments' / name / 'pipeline.joblib')
        labels = model.predict(vectors[train])
        membership = sparse.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(10, len(labels)))
        tf = (membership @ counts).toarray(); tf /= tf.sum(1, keepdims=True)
        topic_df = (membership @ binary).toarray()
        exclusive = (topic_df + 1) / (corpus_df[None, :] + 10)
        for power in plan['frequency_powers']:
            for lift in plan['exclusivity_powers']:
                variant = f'{name}_tf{power}_ex{lift}'
                folder = dest / variant; folder.mkdir(exist_ok=True)
                weight = tf ** power * np.log1p(len(labels) / np.maximum(corpus_df, 1))[None, :] * exclusive ** lift
                keys = []
                for topic in range(10):
                    chosen = []
                    for j in np.argsort(-weight[topic], kind='stable'):
                        if topic_df[topic, j] < 5: continue
                        parts = set(words[j].split())
                        if any(parts <= p or p <= parts for p in chosen): continue
                        chosen.append(parts)
                        keys.append(dict(topic_id=topic, rank=len(chosen), token=words[j], weight=float(weight[topic, j]),
                            train_document_support=int(topic_df[topic, j])))
                        if len(chosen) == 10: break
                keys = pd.DataFrame(keys); keys.to_csv(folder / 'keywords.csv', index=False)
                metric = document_npmi(docs[val].tolist(), keys)
                metric.update(candidate=variant, base=name, frequency_power=power, exclusivity_power=lift,
                    keyword_diversity=keys.token.nunique() / len(keys))
                write_json(folder / 'metrics.json', metric)
                rows.append({k: v for k, v in metric.items() if not isinstance(v, list)})
                pd.DataFrame(rows).to_csv(dest / 'validation_comparison.csv', index=False)
                print(variant, round(metric['document_npmi'], 4), round(metric['keyword_diversity'], 2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='outputs/batterybert_optimization_20261023_v1')
    run(ROOT / parser.parse_args().output)
