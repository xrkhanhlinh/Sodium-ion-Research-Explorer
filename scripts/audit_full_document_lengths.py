"""Describe frozen text lengths and prove scope; never trains or rewrites data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.model_comparison_io import write_json


def main():
    out=ROOT/'outputs/batterybert_full_document_20261024_v1'
    legacy=ROOT/'outputs/model_comparison_v2/batterybert_pretrained'
    corpus=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    spans=pd.read_parquet(legacy/'document_chunk_counts.parquet')
    splits=pd.read_parquet(ROOT/'data/stage3/split_assignments.parquet')
    pairs=pd.read_parquet(ROOT/'data/stage3/contrastive_pairs.parquet')
    aligned=pairs[pairs.split.eq('validation')].merge(corpus[['doc_id','title']],on='doc_id',
        suffixes=('_cache_source','_main'),validate='one_to_one')
    mismatch=aligned.title_cache_source.ne(aligned.title_main)
    write_json(out/'validation_prefix_title_alignment.json',dict(validation_queries=len(aligned),
        exact_title_matches=int((~mismatch).sum()),title_mismatch_ids=aligned.loc[mismatch,'doc_id'].tolist()))
    if mismatch.any() or len(aligned)!=1361:raise ValueError('Reused validation title text differs from current corpus')
    if spans.doc_id.tolist()!=corpus.doc_id.tolist():raise ValueError('Length metadata IDs mismatch')
    frame=spans.merge(splits[['doc_id','split']],on='doc_id',how='left',validate='one_to_one')
    frame['split']=frame.split.fillna('title_only_inference')
    if not frame.covered_abstract_tokens.eq(frame.abstract_tokens).all():raise ValueError('Uncovered Abstract')
    summary=[]
    for name,g in frame.groupby('split'):
        summary.append(dict(split=name,documents=len(g),title_content_tokens=int(g.title_tokens.sum()),
            abstract_content_tokens=int(g.abstract_tokens.sum()),chunks=int(g.chunk_count.sum()),
            abstracts_over_128=int(g.abstract_tokens.gt(128).sum()),abstracts_over_126=int(g.abstract_tokens.gt(126).sum()),
            max_abstract_tokens=int(g.abstract_tokens.max()),max_chunks=int(g.chunk_count.max()),
            median_abstract_tokens=float(g.abstract_tokens.median()),p95_abstract_tokens=float(g.abstract_tokens.quantile(.95))))
    pd.DataFrame(summary).to_csv(out/'text_length_summary.csv',index=False)
    frame.to_csv(out/'text_length_by_document.csv',index=False)
    write_json(out/'text_length_audit.json',dict(documents=len(frame),raw_title_content_tokens=int(frame.title_tokens.sum()),
        query_native_tokens_including_cls_sep=int(frame.title_tokens.sum()+2*len(frame)),
        abstract_content_tokens=int(frame.abstract_tokens.sum()),abstract_coverage=1.,
        full_article_bodies_available=False,source_fields=['title','abstract'],
        title_context_truncated_documents=int(frame.title_truncated.sum()),
        overlap_weighting='Novel-token weights sum to content length; overlapped tokens remain contextual inside chunk means',
        chunk_count_quantiles={str(q):float(np.quantile(frame.chunk_count,q)) for q in [0.,.5,.9,.95,1.]}))
    longest=corpus.iloc[spans.abstract_tokens.nlargest(5).index].copy()
    longest['abstract_tokens']=spans.loc[longest.index,'abstract_tokens']
    longest['chunks']=spans.loc[longest.index,'chunk_count']
    longest.to_csv(out/'long_document_probes.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__':main()
