"""Common-reference topic coherence for old and new saved keyword lists.

Run after selection: these full-corpus scores are descriptive, never a test gate.
No model inference or fitting is needed.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import argparse
import json
import numpy as np
import pandas as pd
import yaml
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from src.batterybert_topic_tools import repair_phrase_placeholders, phrase_reference, write_json


def run(out):
    paths = {
        'LDA_K10_historical': ROOT / 'outputs/model_comparison_v2/lda_k10/topic_keywords.csv',
        'BGE_BERTopic_historical': ROOT / 'outputs/model_comparison_v2/bertopic_bge_k10/topic_keywords.csv',
        'BatteryBERT_HDBSCAN_historical': ROOT / 'outputs/model_comparison_v2/bertopic_batterybert_k10/pretrained/topic_keywords.csv',
        'BatteryBERT_KMeans_web': ROOT / 'outputs/research_completion_v1/topics/topic_keywords.csv',
        'BatteryBERT_optimized': out / 'topic_keywords.csv',
    }
    paths.update({p.parent.name: p for p in (out / 'experiments').glob('*/keywords.csv')})
    keys = {k: pd.read_csv(p).query('topic_id >= 0 and rank <= 10') for k, p in paths.items()}
    words = set(w for frame in keys.values() for w in frame.token)
    frozen = pd.read_parquet(ROOT / 'data/processed/lda_tokens.parquet')
    phrases = yaml.safe_load((ROOT / 'configs/config.yaml').read_text(encoding='utf8'))['preprocessing']['domain_phrases']
    tokens, _ = repair_phrase_placeholders(frozen.tokens.tolist(), phrases)
    reference = list(phrase_reference(tokens, words)); dictionary = Dictionary(reference)
    bow = [dictionary.doc2bow(t) for t in reference]
    # The same dictionary, corpus, phrase occurrence rule and window for everyone.
    rows = []
    for name, frame in keys.items():
        topics, eligible = [], []
        for topic, group in frame.groupby('topic_id', sort=True):
            terms = [w.replace(' ', '\u2581') for w in group.sort_values('rank').token]
            if len(terms) == 10 and all(t in dictionary.token2id for t in terms):
                topics.append(terms); eligible.append(int(topic))
        if topics:
            cv = CoherenceModel(topics=topics, texts=reference, dictionary=dictionary,
                coherence='c_v', window_size=110, topn=10, processes=1).get_coherence_per_topic()
            um = CoherenceModel(topics=topics, corpus=bow, dictionary=dictionary,
                coherence='u_mass', topn=10, processes=1).get_coherence_per_topic()
            rows.append(dict(model=name, c_v=float(np.mean(cv)), u_mass=float(np.mean(um)),
                keyword_diversity=frame.token.nunique() / len(frame), eligible_topics=len(topics),
                total_topics=frame.topic_id.nunique(), eligible_topic_ids=json.dumps(eligible)))
            print(name, 'C_v', np.mean(cv), 'coverage', len(topics), flush=True)
            pd.DataFrame(rows).to_csv(out / 'common_coherence.csv', index=False)
    write_json(out / 'coherence_protocol.json', dict(scope='Descriptive full-corpus comparison after model selection, not independent accuracy',
        reference='Same repaired frozen tokens; exact contiguous phrases atomic with U+2581',
        documents=len(reference), dictionary=len(dictionary), window=110, topn=10,
        processes=1, partial_coverage='Only eligible topics averaged; do not rank partial vs complete as equivalent'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default='outputs/batterybert_optimization_20261023_v1')
    args = parser.parse_args(); run(ROOT / args.output)
