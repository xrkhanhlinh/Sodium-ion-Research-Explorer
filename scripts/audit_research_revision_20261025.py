"""Read-only scientific/engineering audit of saved artifacts before revision."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/research_revision_20261025_v1'
RUN=ROOT/'outputs/batterybert_full_document_20261024_v1'
sys.path.insert(0,str(ROOT))
from src.scientific_keyword_stream import lexical_stream

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def doi(value):
    value=str(value or '').strip().lower()
    match=re.search(r'10\.\d{4,9}/[^\s]+',value)
    return match.group().rstrip('.,;') if match else ''

def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    cat=pd.read_csv(ROOT/'docs/TOP10_TOPICS_ALL_MODELS_20261024.csv')
    keys=pd.read_csv(RUN/'topics/topic_keywords.csv')
    assigned=pd.read_csv(RUN/'topics/topic_assignments.csv')
    reviews=pd.read_csv(RUN/'topics/scientific_review.csv')
    ranks=pd.read_csv(RUN/'comparison_rankings_top10.csv')
    corpus=main.set_index('doc_id')
    source_counts=assigned.groupby(['topic_id','source']).size().unstack(fill_value=0)
    source_counts['Patent_share']=source_counts.get('Patent',0)/source_counts.sum(axis=1)
    source_counts.to_csv(OUT/'baseline_topic_source.csv')
    ids=main.doc_id.tolist()
    identities={d:doi(corpus.loc[d,'identifier']) for d in ids}
    dup=[]
    for (q,m),group in ranks.groupby(['query_id','method'],sort=False):
        seen={}
        for row in group.sort_values('rank').itertuples():
            identity=identities[row.doc_id]
            if not identity:continue
            if identity in seen:
                dup.append({'query_id':q,'method':m,'doc_id':row.doc_id,'duplicate_of':seen[identity],
                            'doi':identity,'rank':int(row.rank),'identity_policy':'Exact normalized DOI only'})
            else:seen[identity]=row.doc_id
    pd.DataFrame(dup).to_csv(OUT/'baseline_search_exact_doi_duplicates.csv',index=False)
    examples=['Na3V2(PO4)3','Na3V2(PO4)(3)','Na(3)V(2)(PO(4))(3)','Fe(CN)(6)','Na3V2(PO4)2F3']
    formula=[{'raw':t,'current_tokens':lexical_stream(t)} for t in examples]
    context=[]
    for row in keys.itertuples():
        if not re.search(r'\d',row.token):continue
        candidate=main[main.doc_id.isin(assigned.loc[assigned.topic_id.eq(row.topic_id),'doc_id'])]
        hits=[]
        pattern=re.compile(re.escape(row.token)+r'\s*\(\s*\d+\s*\)')
        for doc in candidate.itertuples():
            raw=str(doc.title)+' '+str(doc.abstract)
            found=pattern.search(raw)
            if found:
                hits.append({'doc_id':doc.doc_id,'snippet':raw[max(0,found.start()-60):found.end()+60]})
        if hits:context.append({'topic_id':int(row.topic_id),'token':row.token,
                               'raw_continuation_documents':len(hits),'examples':hits[:3]})
    assignments=assigned.set_index('doc_id').topic_id
    corpus['current_topic']=main.doc_id.map(assignments).to_numpy()
    flags=[]
    for row in main.itertuples():
        raw=str(row.title)+' '+str(row.abstract)
        superconductivity=bool(re.search(r'\bsuperconduct|\bsuperconductiv|\bcuprate',str(row.title),re.I))
        missing_battery_context=not bool(re.search(r'batter|sodium.{0,20}stor|na.{0,10}ion.{0,10}stor|electrochem|intercalat',raw,re.I))
        if superconductivity:
            flags.append({'doc_id':row.doc_id,'title':row.title,'source':row.source,
                          'current_topic':int(assignments[row.doc_id]),'flag':'Superconductivity title requires scope review',
                          'no_obvious_battery_storage_context':missing_battery_context,'automatic_exclusion':False})
    pd.DataFrame(flags).to_csv(OUT/'corpus_scope_review_flags.csv',index=False)
    protected=[ROOT/'data/processed/main_corpus.parquet',ROOT/'data/stage3/split_assignments.parquet',
       ROOT/'docs/TOP10_TOPICS_ALL_MODELS_20261024.csv',ROOT/'outputs/batterybert_optimization_20261023_v1/topic_keywords.csv',
       RUN/'document_embeddings.npy',RUN/'ordered_doc_ids.json',RUN/'model/model.safetensors',
       RUN/'topics/topic_assignments.csv',RUN/'topics/topic_keywords.csv',RUN/'rankings_top10.csv',
       RUN/'comparison_rankings_top10.csv',RUN/'annotation_pool.csv',RUN/'annotation_template_0_2.csv',
       RUN/'evaluation_status.json',RUN/'selection.json',RUN/'deployment.json']
    manifest=[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in protected]
    (OUT/'protected_artifacts_before.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    counts=reviews.groupby(['sample_role','agent_status']).size().rename('documents').reset_index()
    counts.to_csv(OUT/'baseline_topic_review_counts.csv',index=False)
    summary={'corpus_documents':len(main),'topic_catalogs':int(cat.model.nunique()),
      'topic_keyword_slots':len(keys),'formula_parser_examples':formula,'formula_continuation_findings':context,
      'topic_review':{'scope':'Existing agent reading notes; biased central/random sample, not expert accuracy',
                      'counts':counts.to_dict('records')},
      'exact_doi_duplicate_top10_rows':len(dup),'affected_query_methods':len(set((r['query_id'],r['method']) for r in dup)),
      'scope_review_flags':len(flags),'automatic_corpus_changes':0,
      'frozen_relevance_status':read(RUN/'evaluation_status.json'),
      'editorial_findings':['LaTeX main snapshot19/10 differs from current Top10 full-document pipeline',
          'LDA selected K20 must be distinct from K10 catalog comparison',
          'Topic and retrieval encoders need not be identical',
          'Coherence, diversity, seed agreement, coverage and relevance are different outcomes',
          '0% KMeans outliers is an assignment property, not superiority',
          'Keyword noise correction does not by itself improve cluster membership'],
      'training_decision':'Investigate parser, representation and geometry first; low downstream diagnostic alone does not justify costly encoder training.'}
    (OUT/'baseline_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=True,indent=2))

if __name__=='__main__':audit()
