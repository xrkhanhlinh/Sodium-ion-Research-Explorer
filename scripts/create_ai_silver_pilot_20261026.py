"""Materialize the assistant's independent text-only review of the first20 pairs.

No retrieval scores, ranks, model names, inferred full text or human qrels are
used. Records below are explicit judgements after reading Query/Title/Abstract,
not a keyword-count labeller. The original input is immutable.
"""
from pathlib import Path
from datetime import datetime,timezone
import csv
import hashlib
import io
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/batterybert_full_document_20261024_v1/annotation_pool.csv'
OUT=ROOT/'outputs/ai_silver_labels_20261026_v1'
COLUMNS=['query_id','doc_id','ai_label','confidence','evidence','reasoning','review_required']

# (exact doc_id, grade or None, confidence, exact evidence pieces, reasoning, review)
JUDGEMENTS=[
('patent:201704291X',2,'high',
 ['Biomass hard carbon anode material is used for sodium ion battery (claimed).'],
 'Nêu trực tiếp hard carbon từ biomass, vai trò anode và pin sodium-ion; mô tả quy trình điều chế phù hợp truy vấn.',False),
('patent:201890217F',0,'medium',
 ['biomass-derived hard carbon/graphene sodium ion battery cathode material'],
 'Văn bản gọi vật liệu là cathode, khác vai trò anode được yêu cầu. Không tự sửa vai trò điện cực từ kiến thức thông thường. NEEDS_REVIEW do cách mô tả điện cực cần được kiểm tra lại.',True),
('patent:202192642S',0,'medium',
 ['Method for preparing biomass hard carbon cathode material of a sodium ion battery (claimed).'],
 'Đúng biomass, hard carbon và hệ pin, nhưng Title và Abstract đều nêu cathode thay vì anode. NEEDS_REVIEW về độ chính xác metadata; nhãn hiện dựa trên văn bản được cung cấp.',True),
('patent:2022B9059E',2,'high',
 ['Biomass-based hard carbon is used in negative electrode sheet for sodium-ion battery (all claimed).'],
 'Đáp ứng đồng thời tiền chất biomass, hard carbon, điện cực âm và pin sodium-ion; Abstract có quy trình carbon hóa.',False),
('patent:2022C19342',0,'medium',
 ['The hard carbon is useful in cathode pole piece of sodium-ion battery'],
 'Nghiên cứu đúng biomass-derived hard carbon nhưng mô tả sử dụng ở cathode, không đáp ứng vai trò anode của truy vấn. NEEDS_REVIEW metadata, không suy đoán sửa cathode thành anode.',True),
('patent:2022D5489X',None,'low',
 ['cathode electrode used for sodium ion battery',
  'polymer modified biomass hard carbon material negative electrode'],
 'NEEDS_REVIEW: cùng Abstract gọi vật liệu là cathode rồi là negative electrode. Hai mô tả vai trò mâu thuẫn nên chưa đủ căn cứ gán nhãn dứt khoát.',True),
('patent:2022E9483K',0,'medium',
 ['Method for preparing hetero atom-doped biomass-derived hard carbon material cathode in sodium ion battery electrode material.'],
 'Đúng nguồn biomass và hard carbon nhưng văn bản nêu cathode, khác anode trong truy vấn. NEEDS_REVIEW để xác minh cách ghi vai trò điện cực.',True),
('patent:202338303K',2,'high',
 ['Method for preparing biomass composite hard carbon negative electrode material in preparing sodium ion battery negative electrode (claimed).'],
 'Nêu rõ hard carbon composite từ biomass được dùng làm điện cực âm của sodium-ion; đáp ứng trực tiếp mục tiêu.',False),
('patent:202345397J',1,'medium',
 ['Hard carbon material for sodium-ion battery, prepared by mixing biomass material and asphalt and carbonizing'],
 'Có hard carbon, thành phần biomass và pin sodium-ion nhưng chưa nêu trực tiếp vai trò anode/negative electrode. NEEDS_REVIEW cho điều kiện vai trò điện cực; không tự suy ra từ tên vật liệu.',True),
('publication:WOS:000330041400022',1,'medium',
 ['Nitrogen-doped activated porous carbon fibres (ACFs) were prepared as anode materials for Na-ion batteries.'],
 'Có anode carbon cho Na-ion nhưng Title và Abstract không xác nhận nguồn biomass hoặc phân loại hard carbon. Liên quan nền, chưa đủ cho mức 2. NEEDS_REVIEW các điều kiện chưa được nêu.',True),
('publication:WOS:000340347700014',2,'high',
 ['Biomass derived hard carbon used as a high performance anode material for sodium ion batteries'],
 'Hard carbon từ vỏ bưởi, được dùng và thử nghiệm làm anode sodium-ion; đủ các yếu tố chính của truy vấn.',False),
('publication:WOS:000375356800024',1,'high',
 ['a low-cost, scalable waste tire-derived carbon as an anode for sodium-ion batteries (SIBs).'],
 'Có carbon anode cho sodium-ion và bối cảnh hard carbon, nhưng tiền chất được nêu là lốp thải, không được xác nhận là biomass. Hữu ích để tham khảo carbon tái chế, không đáp ứng đầy đủ nguồn tiền chất.',False),
('publication:WOS:000382110200003',2,'high',
 ['A waste biomass derived hard carbon as a high-performance anode material for sodium-ion batteries'],
 'Hard carbon từ lõi ngô và vai trò anode sodium-ion được nêu rõ; Abstract còn mô tả full cell sử dụng vật liệu này làm anode.',False),
('publication:WOS:000403934400017',2,'high',
 ['hard carbons derived from different sources-namely corncobs, peanut shells, and waste apples'],
 'Abstract xác định hard carbon là điện cực âm Na-ion và so sánh các tiền chất biomass cụ thể; đáp ứng trực tiếp truy vấn.',False),
('publication:WOS:000409187400002',1,'medium',
 ['Nitrogen-doped porous carbon is synthesised from mineral-rich egg yolks.'],
 'Có carbon từ nguồn sinh học và được dùng làm anode sodium-ion, nhưng văn bản chỉ gọi porous carbon, chưa xác nhận hard carbon. NEEDS_REVIEW phân loại vật liệu; không coi hai thuật ngữ là tương đương tự động.',True),
('publication:WOS:000410369600083',2,'high',
 ['The obtained hard carbons with an amazing high carbon yield of 60.73% were evaluated as anode for SIBs.'],
 'Abstract gọi tiền chất là biomass-derived hard carbon từ leonardite humic acid và nêu đánh giá làm anode SIBs; nhãn dựa trên mô tả có sẵn, không suy diễn nguồn ngoài văn bản.',False),
('publication:WOS:000417484000075',2,'high',
 ['Rice husk-derived hard carbons as high-performance anode materials for sodium-ion batteries'],
 'Nêu rõ hard carbon từ trấu, vai trò anode và pin sodium-ion; Abstract mô tả cả thí nghiệm full cell.',False),
('publication:WOS:000424885800081',2,'high',
 ['synthesize high-performance hard carbons from waste apricot shell.'],
 'Hard carbon từ vỏ mơ thải được nghiên cứu làm anode sodium-ion; Title và Abstract cùng hỗ trợ đầy đủ mục tiêu.',False),
('publication:WOS:000440661700018',2,'high',
 ['Hard carbons issued from date palm as efficient anode materials for sodium-ion batteries'],
 'Hard carbon từ biomass chà là được đánh giá làm anode sodium-ion, đáp ứng đúng nguồn, vật liệu và vai trò.',False),
('publication:WOS:000442603700004',2,'high',
 ['Biochars from various biomass types as precursors for hard carbon anodes in sodium-ion batteries'],
 'So sánh biochar từ biomass làm tiền chất hard carbon anode sodium-ion; có điều chế, đặc trưng và thử nghiệm điện cực.',False),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=data if isinstance(data,bytes) else data.encode('utf8')
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('Existing result differs; use a new reviewed version: '+str(path))
        return
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_bytes(payload);pending.replace(path)


def jwrite(path,value):immutable(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_hash=sha(SOURCE)
    data=pd.read_csv(SOURCE,keep_default_na=False,dtype=str)
    required=['query_id','query','doc_id','title','abstract']
    if any(column not in data for column in required):raise ValueError('Input lacks required columns')
    if data.duplicated(['query_id','doc_id']).any():raise ValueError('Duplicate input pairs require explicit resolution, never silent removal')
    pilot=data.iloc[:20].copy()
    if len(pilot)!=20 or pilot.doc_id.tolist()!=[item[0] for item in JUDGEMENTS]:raise ValueError('First20 input identity changed; review the new records independently')
    input_audit={'input':str(SOURCE),'sha256':source_hash,'rows':len(data),'columns':data.columns.tolist(),
       'duplicate_pairs':int(data.duplicated(['query_id','doc_id']).sum()),
       'missing_fields':{column:int(data[column].str.strip().eq('').sum()) for column in required},
       'pilot_start_row_one_based':1,'pilot_rows':20,'pilot_query_distribution':pilot.query_id.value_counts().to_dict(),
       'input_selection':'No new uploaded file was attached; existing project annotation_pool.csv used, disclosed to user',
       'judgement_inputs':required,'excluded_signals':['cosine','RRF','rank','search_method'],
       'label_method':'AI_ASSISTANT_SILVER_TEXT_REVIEW','human_verified':False}
    jwrite(OUT/'input_structure.json',input_audit)
    protocol={'label_scale':[0,1,2],'confidence':['high','medium','low'],'pilot_size':20,'batch_size':20,
       'missing_or_conflicting_evidence':'Blank ai_label and NEEDS_REVIEW when relevance cannot be decided',
       'explicit_wrong_electrode_role':'Label0 on supplied text, medium confidence and review flag when metadata may require correction; never silently change cathode/anode',
       'partial_evidence':'Label1 when useful partial relation is demonstrable but an important criterion is not established',
       'evidence':'Short exact substrings of supplied Title or Abstract; multiple pieces separated by |',
       'input_order_preserved':True,'human_verified':False,'expert_ground_truth':False,
       'used_for_training_or_model_selection':False,'pilot_is_representative_sample':False,
       'rubric_source':'User instruction dated26/10/2026 Asia/Bangkok'}
    jwrite(OUT/'protocol.json',protocol)
    blind=io.StringIO(newline='');pilot[required].to_csv(blind,index=False)
    immutable(OUT/'batch_0001_input_blinded.csv',blind.getvalue().encode('utf-8-sig'))
    records=[]
    for row,judgement in zip(pilot.itertuples(index=False),JUDGEMENTS):
        doc,label,confidence,pieces,reason,review=judgement
        raw=row.title+'\n'+row.abstract
        if not all(piece and piece in raw for piece in pieces):raise ValueError('Evidence not exact source substring: '+doc)
        if label is None and not review:raise ValueError('Unresolved record must be reviewed')
        records.append(dict(query_id=row.query_id,doc_id=doc,ai_label='' if label is None else label,
             confidence=confidence,evidence=' | '.join(pieces),reasoning=reason,review_required='TRUE' if review else 'FALSE'))
        if len(records)%5==0:
            jwrite(OUT/'checkpoints'/f'batch_0001_rows_{len(records):04d}.json',{
               'input_sha256':source_hash,'reviewed_rows':len(records),'judgements':records.copy(),'label_method':'AI_SILVER','status':'completed'})
    csv_output=io.StringIO(newline='');writer=csv.DictWriter(csv_output,fieldnames=COLUMNS);writer.writeheader();writer.writerows(records)
    output=OUT/'batch_0001_labels_20.csv';immutable(output,csv_output.getvalue().encode('utf-8-sig'))
    restored=pd.read_csv(output,keep_default_na=False,dtype=str)
    assert restored.columns.tolist()==COLUMNS
    assert len(restored)==len(pilot) and not restored.duplicated(['query_id','doc_id']).any()
    assert list(zip(restored.query_id,restored.doc_id))==list(zip(pilot.query_id,pilot.doc_id))
    assert restored.ai_label.isin(['','0','1','2']).all()
    summary={'status':'completed','batch_id':'batch_0001','processed':20,'input_total':len(data),'remaining_pending':len(data)-20,
       'label_counts':{str(grade):int(restored.ai_label.eq(str(grade)).sum()) for grade in [0,1,2]},
       'needs_review':int(restored.review_required.eq('TRUE').sum()),'blank_labels':int(restored.ai_label.eq('').sum()),
       'confidence_counts':restored.confidence.value_counts().to_dict(),'queries':restored.query_id.value_counts().to_dict(),
       'silver_labels_only':True,'human_verified_labels_created':0,'used_for_training_or_selection':False,
       'scope_note':'First20 in input order are all Q01; not a representative sample across20 queries. No overall Search accuracy estimated.',
       'input_sha256':source_hash,'output_sha256':sha(output),'output':str(output)}
    jwrite(OUT/'batch_0001_summary.json',summary)
    jwrite(OUT/'pipeline_status.json',{'input_validation':'completed','pilot_batch_0001':'completed',
       'remaining_batches':'pending_not_requested','processed_pairs':20,'remaining_pairs':len(data)-20,
       'resume_policy':'Validate exact input hash, pair order, source evidence and committed output before skipping; changed reviews require a new version'})
    if sha(SOURCE)!=source_hash:raise ValueError('Original input changed during materialization')
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
