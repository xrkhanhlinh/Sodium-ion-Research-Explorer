"""Non-relevance diagnostics. Every flag is for review, never a human qrel."""
import re
import numpy as np
import pandas as pd


def potential_aqueous_contradiction(query,title):
    normalize=lambda text:re.sub('[‐‑–−]','-',str(text)).casefold()
    query,title=normalize(query),normalize(title)
    negative=r'\b(?:non[-\s]*(?:aqueous|water)|not\s+aqueous)\b'
    asks_aqueous=bool(re.search(r'\baqueous\b',query)) and not re.search(negative,query)
    return bool(asks_aqueous and re.search(negative,title))


def diagnose_rankings(rankings,metadata,queries):
    required={'query_id','method','doc_id','rank','title'}
    if not required<=set(rankings):raise ValueError('Missing diagnostic input columns')
    metadata=metadata.set_index('doc_id',drop=False) if 'doc_id' in metadata else metadata
    if not metadata.index.is_unique:raise ValueError('Duplicate metadata IDs')
    if not set(rankings.doc_id)<=set(metadata.index):raise ValueError('Unknown ranked document')
    q=queries.set_index('query_id')
    flags=[];rows=[]
    for (qid,method),group in rankings.groupby(['query_id','method']):
        group=group.sort_values('rank')
        meta=metadata.loc[group.doc_id]
        ids=meta.split_group_id.fillna('').astype(str)
        known=ids[ids.ne('')]
        rows.append(dict(query_id=qid,method=method,results=len(group),
            patent_results=int(meta.source.eq('Patent').sum()),publication_results=int(meta.source.eq('Publication').sum()),
            unique_known_split_groups=int(known.nunique()),repeated_known_group_slots=int(len(known)-known.nunique()),
            note='Source composition and candidate-group repetition are diagnostics, not relevance or verified patent-family duplication'))
        for row in group.itertuples():
            if potential_aqueous_contradiction(q.loc[qid,'query'],row.title):
                flags.append(dict(query_id=qid,method=method,rank=row.rank,doc_id=row.doc_id,title=row.title,
                    flag='POTENTIAL_AQUEOUS_TITLE_CONTRADICTION',label_status='UNVERIFIED_DIAGNOSTIC_ONLY',
                    reason='Query asks aqueous; Title explicitly says non-aqueous/non-water. Read full context before judging.'))
    return pd.DataFrame(rows),pd.DataFrame(flags,columns=['query_id','method','rank','doc_id','title','flag','label_status','reason'])


def topic_review_sample(assignments,representatives,summary,metadata,seed=42):
    metadata=metadata.set_index('doc_id',drop=False) if 'doc_id' in metadata else metadata
    if not metadata.index.is_unique or not assignments.doc_id.is_unique:raise ValueError('Duplicate document IDs')
    rng=np.random.default_rng(seed);rows=[]
    for topic in sorted(set(assignments.topic_id)-{-1}):
        members=assignments[assignments.topic_id.eq(topic)]
        top=representatives[representatives.topic_id.eq(topic)].sort_values('rank').doc_id.tolist()
        if len(top)!=3 or len(set(top))!=3 or not set(top)<=set(members.doc_id):
            raise ValueError('Need three distinct assigned representatives per topic')
        remainder=sorted(set(members.doc_id)-set(top))
        if len(remainder)<2:raise ValueError('Need two additional topic members')
        selected=top+rng.choice(remainder,size=2,replace=False).tolist()
        keyword=summary.set_index('topic_id').loc[topic,'top10_keywords']
        for rank,doc_id in enumerate(selected,1):
            document=metadata.loc[doc_id]
            rows.append(dict(topic_id=int(topic),review_position=rank,doc_id=doc_id,top10_keywords=keyword,
                title=document.title,abstract=document.abstract,source=document.source,
                selection_method='highest_membership_representative' if rank<=3 else 'random_additional_member_seed42',
                topic_fit_grade='',formula_issue='',suggested_topic_name='',reviewer_notes='',label_status='UNVERIFIED'))
    return pd.DataFrame(rows)
