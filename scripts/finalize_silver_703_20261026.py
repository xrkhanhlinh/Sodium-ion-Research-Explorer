"""Freeze all explicit query-batch judgments before opening retrieval rankings."""
from pathlib import Path
import csv,hashlib,io,json,sys
import pandas as pd
from silver_labels_703 import ROOT,OUT,SOURCE,COLS,sha,put,js,load

def main():
    sys.stdout.reconfigure(encoding='utf8');pool=load();frames=[]
    for qid in pool.query_id.drop_duplicates():
        path=OUT/'batches'/f'{qid}_labels.csv';frame=pd.read_csv(path,keep_default_na=False,dtype=str)
        receipt=json.loads((OUT/'checkpoints'/f'{qid}.json').read_text(encoding='utf8'))
        assert receipt['status']=='completed' and sha(path)==receipt['output_sha256']
        expected=pool[pool.query_id.eq(qid)]
        assert list(zip(frame.query_id,frame.doc_id))==list(zip(expected.query_id,expected.doc_id))
        frames.append(frame)
    labels=pd.concat(frames,ignore_index=True)
    assert labels.columns.tolist()==COLS and len(labels)==703
    assert not labels.duplicated(['query_id','doc_id']).any() and labels.ai_label.isin(['0','1','2']).all()
    assert list(zip(labels.query_id,labels.doc_id))==list(zip(pool.query_id,pool.doc_id))
    amendments=[]
    # Textual QA correction only: the source mentions maintenance efficiency,
    # not maintenance safety. No grade changes and no retrieval results opened.
    idx=labels.query_id.eq('Q17')&labels.doc_id.eq('patent:202353720N')
    before=labels.loc[idx,'reasoning'].iloc[0]
    after='Hybrid power-pack mode control và maintenance efficiency, chưa có thermal control hoặc safety assessment.'
    labels.loc[idx,'reasoning']=after
    amendments.append({'query_id':'Q17','doc_id':'patent:202353720N','field':'reasoning','before':before,'after':after,'grade_changed':False})
    # Concise, exact quotes: no invented ellipsis inside source evidence.
    for i,row in labels.iterrows():
        pieces=row.evidence.split(' | ')
        labels.loc[i,'evidence']=' | '.join(piece if len(piece)<=230 else piece[:230].rsplit(' ',1)[0] for piece in pieces)
    # Critical full-intent evidence includes both full-cell configuration and
    # electrolyte additive, rather than relying on the shorter paper title.
    idx=labels.query_id.eq('Q20')&labels.doc_id.eq('publication:WOS:000439134600040')
    labels.loc[idx,'evidence']='using the ternary layered Na0.76Ni0.3Fe0.4Mn0.3O2 and the hard carbon as the cathode and anode materials, respectively. | ADN is selected as the electrolyte additive'
    # The only initially blank pilot judgment is now conservatively partial.
    amendments.append({'query_id':'Q01','doc_id':'patent:2022D5489X','field':'ai_label','before':'','after':'1',
       'reason':'Re-read definite biomass/hard-carbon/Na evidence; contradictory electrode role prevents2 but partial relation is defensible.',
       'pilot_original_unchanged':True,'review_required':True})
    for raw,judged in zip(pool.itertuples(),labels.itertuples()):
        assert all(piece and piece in raw.title+'\n'+raw.abstract for piece in judged.evidence.split(' | ')),judged.doc_id
        assert judged.confidence in ['high','medium','low'] and judged.review_required in ['TRUE','FALSE']
    put(OUT/'silver_labels_703.csv',labels.to_csv(index=False).encode('utf-8-sig'))
    put(OUT/'labels_with_source_703.csv',pool.merge(labels,on=['query_id','doc_id'],validate='one_to_one',sort=False).to_csv(index=False).encode('utf-8-sig'))
    js(OUT/'text_review_amendments.json',amendments)
    counts=labels.groupby(['query_id','ai_label']).size().unstack(fill_value=0).reindex(columns=['0','1','2'],fill_value=0)
    counts['processed']=counts.sum(axis=1);counts['review_required']=labels.assign(review=labels.review_required.eq('TRUE')).groupby('query_id').review.sum()
    put(OUT/'label_counts_per_query.csv',counts.reset_index().to_csv(index=False).encode('utf-8-sig'))
    put(OUT/'needs_review.csv',pool.merge(labels[labels.review_required.eq('TRUE')],on=['query_id','doc_id'],validate='one_to_one').to_csv(index=False).encode('utf-8-sig'))
    summary={'status':'completed','processed':703,'numeric_labels':703,'missing':0,'duplicates':0,'omitted_pairs':0,'queries':20,
      'label_counts':labels.ai_label.value_counts().sort_index().to_dict(),'confidence_counts':labels.confidence.value_counts().to_dict(),
      'review_required':int(labels.review_required.eq('TRUE').sum()),'empty_abstracts':int(pool.abstract.eq('').sum()),
      'source_sha256':sha(SOURCE),'labels_sha256':sha(OUT/'silver_labels_703.csv'),
      'label_method':'AI_ASSISTANT_SILVER_TEXT_REVIEW','human_verified':False,'used_for_training_or_model_selection':False,
      'label_freeze':'Labels committed and source/evidence validated before retrieval rankings are read for this evaluation.',
      'adjudication':'19 decided pilot labels reused; one blank explicitly re-reviewed as partial1 with low confidence/review flag; all old pilot files retained.',
      'evidence_policy':'Exact source quotes, shortened only at word boundaries; software validates quote existence, not semantic correctness.',
      'limitations':'One assistant, no independent second rater or human verification. Confidence is subjective, not calibrated.'}
    js(OUT/'labels_frozen_manifest.json',{'labels_sha256':summary['labels_sha256'],'input_sha256':sha(SOURCE),'batch_sha256':{p.name:sha(p) for p in sorted((OUT/'batches').glob('*.csv'))}})
    js(OUT/'label_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
