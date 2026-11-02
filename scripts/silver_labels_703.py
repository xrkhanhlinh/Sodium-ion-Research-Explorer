"""Commit explicit, blinded assistant judgments; never infer a grade from words/scores."""
from pathlib import Path
import argparse,csv,hashlib,io,json,sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/batterybert_full_document_20261024_v1/annotation_pool.csv'
OUT=ROOT/'outputs/ai_silver_labels_703_20261026_v1'
COLS=['query_id','doc_id','ai_label','confidence','evidence','reasoning','review_required']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,payload):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=payload:raise ValueError('Immutable result differs: '+str(p))
        return
    t=p.with_suffix(p.suffix+'.pending');t.write_bytes(payload);t.replace(p)
def js(p,v):put(p,json.dumps(v,ensure_ascii=False,indent=2).encode('utf8'))
def load():
    p=pd.read_csv(SOURCE,keep_default_na=False,dtype=str)
    assert len(p)==703 and not p.duplicated(['query_id','doc_id']).any()
    assert sha(SOURCE)=='6f6c6a7f22e29fc11b4ca0513016e8c182751e732b73f411a0cadf51f7aee513'
    OUT.mkdir(parents=True,exist_ok=True)
    js(OUT/'protocol.json',{'source':str(SOURCE),'source_sha256':sha(SOURCE),'pairs':703,
      'method':'Explicit assistant text review of full Query, Title and Abstract; no score or ranking signals',
      'silver_labels':True,'expert_verified':False,'used_for_training':False,
      'scale':{'0':'Off intent or explicit mandatory conflict','1':'Useful partial evidence; important criteria missing','2':'Directly satisfies query intent'},
      'uncertainty':'Keep blank only if no defensible relation can be judged. Flag missing/conflicting evidence. Never replace absent judgments with 0.',
      'evaluation':'Same frozen shared pool for all six methods. Threshold2; graded gain2^label-1; IDCG from entire judged query pool; recall is pooled only.',
      'pilot_policy':'Keep old pilot immutable; reuse19 decided judgments, explicitly adjudicate its one blank in a separate version.'})
    return p
def commit(qid):
    p=load();q=p[p.query_id.eq(qid)]
    decisions=json.loads((OUT/'decisions'/f'{qid}.json').read_text(encoding='utf8'))
    records=[]
    if qid=='Q01':
        pilot=pd.read_csv(ROOT/'outputs/ai_silver_labels_20261026_v1/batch_0001_labels_20.csv',keep_default_na=False,dtype=str)
        records=pilot.to_dict('records')
        assert list(pilot.doc_id)==list(q.iloc[:20].doc_id)
        # Explicit re-review: biomass/hard carbon/SIB evidence is definite, role
        # is contradictory, therefore partial relevance is defensible (not 2).
        records[5].update(ai_label=1,confidence='low',reasoning='Có bằng chứng biomass hard carbon cho pin sodium-ion, nhưng cùng Abstract gọi cathode và negative electrode. Chấm liên quan một phần vì vai trò chưa xác định; cần kiểm tra metadata. Bản pilot để trống được giữ nguyên.',review_required='TRUE')
        q=q.iloc[20:]
    assert len(q)==len(decisions),(qid,len(q),len(decisions))
    for row,d in zip(q.itertuples(),decisions):
        grade,reason,*options=d
        assert grade in [0,1,2,None] and reason.strip()
        opts=options[0] if options else {}
        evidence=opts.get('evidence',row.title)
        assert evidence and all(piece in row.title+'\n'+row.abstract for piece in evidence.split(' | ')),row.doc_id
        missing=not row.abstract.strip()
        review=opts.get('review',False) or missing or grade is None
        confidence=opts.get('confidence','medium' if grade==1 else 'high')
        if missing:confidence=opts.get('confidence','medium')
        records.append(dict(query_id=qid,doc_id=row.doc_id,ai_label='' if grade is None else grade,
          confidence=confidence,evidence=evidence,reasoning=reason,review_required='TRUE' if review else 'FALSE'))
    dest=OUT/'batches'/f'{qid}_labels.csv'
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=COLS);w.writeheader();w.writerows(records)
    put(dest,stream.getvalue().encode('utf-8-sig'))
    js(OUT/'checkpoints'/f'{qid}.json',{'status':'completed','pairs':len(records),'source_sha256':sha(SOURCE),'output_sha256':sha(dest),
      'label_counts':{str(g):sum(str(r['ai_label'])==str(g) for r in records) for g in [0,1,2]},'manual_decisions_sha256':sha(OUT/'decisions'/f'{qid}.json')})
    done=sorted((OUT/'checkpoints').glob('Q*.json'))
    state={'status':'completed' if len(done)==20 else 'running','completed_queries':[x.stem for x in done],
      'processed_pairs':sum(json.loads(x.read_text(encoding='utf8'))['pairs'] for x in done),'total_pairs':703}
    (OUT/'state.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    print(json.dumps(state))
def main():
    sys.stdout.reconfigure(encoding='utf8')
    a=argparse.ArgumentParser();a.add_argument('action',choices=['read','commit']);a.add_argument('query');a.add_argument('--start',type=int,default=1);a.add_argument('--end',type=int,default=1000);args=a.parse_args()
    if args.action=='commit':commit(args.query);return
    p=load();q=p[p.query_id.eq(args.query)]
    print('QUERY:',q.iloc[0]['query'])
    for n,row in enumerate(q.itertuples(),1):
        if n<args.start or n>args.end:continue
        print(f'\n[{n}] {row.doc_id}\nTITLE: {row.title}\nABSTRACT: {row.abstract}')
if __name__=='__main__':main()
