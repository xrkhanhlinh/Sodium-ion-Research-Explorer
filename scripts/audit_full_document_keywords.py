"""Audit historical keyword support against the actual raw-field adjacency."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import yaml
from src.scientific_keyword_stream import prepare_corpus_stream
from src.batterybert_optimization import ContiguousScientificVectorizer
from src.batterybert_topic_tools import repair_phrase_placeholders
from src.model_comparison_io import write_json


def main():
    out=ROOT/'outputs/batterybert_full_document_20261024_v1'
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    phrases=yaml.safe_load((ROOT/'configs/config.yaml').read_text(encoding='utf8'))['preprocessing']['domain_phrases']
    actual=prepare_corpus_stream(out,main,phrases)
    old=pd.read_parquet(ROOT/'data/processed/lda_tokens.parquet')
    if old.doc_id.tolist()!=main.doc_id.tolist():raise ValueError('Historical lexical IDs mismatch')
    legacy,_=repair_phrase_placeholders(old.tokens.tolist(),phrases)
    keys=pd.read_csv(ROOT/'outputs/batterybert_optimization_20261023_v1/topic_keywords.csv')
    words=sorted(set(keys.token));vocabulary={w:i for i,w in enumerate(words)}
    analyzer=ContiguousScientificVectorizer(vocabulary=vocabulary,binary=True,ngram_range=(1,2),lowercase=False)
    actual_df=analyzer.transform([' '.join(t) for t in actual]).sum(0).A1
    legacy_df=analyzer.transform([' '.join(t) for t in legacy]).sum(0).A1
    result=pd.DataFrame(dict(token=words,historical_stream_documents=legacy_df,source_adjacent_documents=actual_df))
    result.to_csv(out/'historical_keyword_adjacency_audit.csv',index=False)
    write_json(out/'keyword_adjacency_audit.json',dict(status='PASS',
        correction='Frozen LDA tokens cannot prove raw adjacency after stopword deletion; new source-based stream preserves boundaries',
        old_keywords=len(words),old_keywords_without_source_adjacent_support=int((actual_df==0).sum()),
        multiword_keywords=int(result.token.str.contains(' ').sum()),historical_inputs_modified=False,
        note='Document frequencies can also differ through legacy normalization; do not attribute every difference to a false phrase'))
    print(result[result.token.str.contains(' ')].to_string(index=False),flush=True)


if __name__=='__main__':main()
