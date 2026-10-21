"""Versioned Gemini research evaluation. Run as python -m scripts.run_retrieval_ai_evaluation.

No training, corpus edits, live app changes, human-label impersonation or paid API.
"""
import argparse
import atexit
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_judge import (ROOT, DEFAULT_OUTPUT, RUBRIC, GeminiFreeClient, PendingAPI,
                          blind_input, file_hash, fingerprint, judge, read_json, utc_now, write_json,
                          response_json, validate_judgment)
from src.ai_retrieval_evaluation import (METHODS, DEVELOPMENT, AI_STATUS, agreement,
                                        consensus_record, evaluate_ai, error_analysis, numeric)
from src.ai_synthetic_benchmark import (select_documents, generate_query, retrieve_synthetic, verify_legacy_parity)
from src.ai_evaluation_audit import api_usage,scan_active_credential,verify_synthetic_runs

SOURCE = ROOT/'outputs/search_upgrade_20261018'
PATHS = {'corpus':'data/processed/main_corpus.parquet', 'pool':str((SOURCE/'shared_pool_UNVERIFIED.csv').relative_to(ROOT)),
         'queries':str((SOURCE/'queries.csv').relative_to(ROOT)),
         'rankings':str((SOURCE/'evaluation_14_methods/top5_fourteen_methods.csv').relative_to(ROOT)),
         'runs':str((SOURCE/'evaluation_14_methods/run_manifest.csv').relative_to(ROOT))}
INTENTS = [
 'Sodium-ion hard-carbon anodes from biomass; connect biomass precursor and anode application.',
 'Hard-carbon anodes; initial Coulombic efficiency and sodium-storage mechanisms, including their relationship.',
 'Layered-oxide sodium cathodes in P2 and/or O3 structures; reversible anionic/oxygen redox. Either named structure may directly address the intent.',
 'Prussian-blue analogue sodium cathodes; low vacancies and low water content, including structural/electrochemical implications.',
 'Na3V2(PO4)3 (NASICON) sodium cathodes with carbon coating; not another NASICON formula alone.',
 'Low-cost iron-based phosphate and/or pyrophosphate sodium cathodes; cost motivation and electrochemical use.',
 'Fluoroethylene carbonate (FEC) electrolyte additive and stable SEI on hard-carbon sodium-ion anodes; relate additive to interface.',
 'NaFSI salt in ether electrolytes for low-temperature sodium-ion operation; all three aspects matter.',
 'Na3PS4 and/or Na3SbS4 sulfide solid electrolytes for sodium transport; either material directly relevant.',
 'NASICON Na3Zr2Si2PO12 ceramic solid electrolyte and ionic conductivity; preserve exact target stoichiometry or documented equivalent notation.',
 'Ti3C2Tx MXene sodium storage; interlayer engineering and its effect, rather than arbitrary MXene uses.',
 'MoS2 nanosheets for sodium storage with expanded interlayer spacing; connect spacing and storage.',
 'Sn4P3 sodium alloying anodes and volume-expansion control; different phosphides alone only partial or off intent.',
 'Organic quinone and/or covalent-framework sodium-ion electrodes; either class relevant, full relation required when both asserted.',
 'Sodium-ion electrode processing: slurry binders and/or current collectors; direct materials/manufacturing relevance, not generic cells alone.',
 'Roll pressing and/or electrode-coating equipment in sodium-ion electrode manufacturing; equipment/process relevance.',
 'Sodium-ion battery packs; thermal management and safety in pack/system context, not a materials safety mention alone.',
 'Recycling spent sodium-ion batteries; recovery of cathode metals, distinguish recycling from new cathode synthesis.',
 'Aqueous sodium-ion batteries using water-in-salt electrolytes; aqueous alone is partial, non-aqueous-only study conflicts with a mandatory condition.',
 'Full sodium-ion cells combining hard-carbon anode, layered-oxide cathode and electrolyte additives; relation in one full-cell system matters.',
 'Operando characterization of sodium intercalation and phase transitions; technique/process relationship, not ex situ-only characterization.',
 'Cellulose separators for sodium-ion batteries with improved electrolyte wettability; connect separator and wetting.',
 'Na2FeP2O7 cathode and electrochemical cycling stability; exact material and stability evidence matter.'
]


def csv(path):
    return pd.read_csv(path, keep_default_na=False)


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Frozen protocol/input drift: '+path.name)
    else:
        write_json(path, value)


def protected_paths():
    paths = set(PATHS.values())
    for folder in ('models','data/processed','data/stage2','data/stage3','data/stage4','data/search_v2','outputs/bertopic_chunked'):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*')
                     if p.is_file() and 'qdrant' not in p.parts and '__pycache__' not in p.parts)
    for pattern in ('src/stage*.py','src/search_v2*.py','configs/stage*.json','configs/*.yaml','configs/search_v2.json'):
        paths.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern) if p.is_file())
    paths.update(['src/text_processing.py','src/research_evaluation.py','app.py'])
    return sorted(paths)


def verify_inputs(output):
    expected=read_json(output/'protected_inputs.json')
    bad=[name for name,sha in expected.items() if not (ROOT/name).exists() or file_hash(ROOT/name)!=sha]
    write_json(output/'input_verification.json',{'checked_at':utc_now(),'files':len(expected),'status':'PASS' if not bad else 'FAIL',
                'changed':bad,'qdrant':'No client opened and no write requested; active DB files excluded from hashing'})
    if bad:raise ValueError('Frozen input changed: '+', '.join(bad[:5]))


def prepare(output, config):
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'protected_inputs.json'
    if not manifest.exists():
        write_json(manifest,{p:file_hash(ROOT/p) for p in protected_paths()})
    verify_inputs(output)
    corpus=pd.read_parquet(ROOT/PATHS['corpus']).fillna({'title':'','abstract':''})
    if len(corpus)!=13702 or not corpus.doc_id.is_unique:raise ValueError('Expected frozen 13,702 documents')
    pool,queries=csv(ROOT/PATHS['pool']),csv(ROOT/PATHS['queries'])
    if len(pool)!=942 or len(queries)!=24:raise ValueError('Unexpected pool/query size')
    canonical=corpus.set_index('doc_id')
    for row in pool.itertuples():
        if row.title!=canonical.loc[row.doc_id,'title'] or row.abstract!=canonical.loc[row.doc_id,'abstract']:
            raise ValueError('Pool differs from full frozen corpus text')
    intents={f'Q{i:02d}':text for i,text in enumerate(INTENTS,1)}
    immutable_json(output/'query_intents.json',intents)
    docs=[blind_input(row,intents[row['query_id']]) for row in pool.to_dict('records')]
    immutable_json(output/'blind_inputs.json',docs)
    if not (output/'blind_pool.csv').exists():pd.DataFrame(docs).to_csv(output/'blind_pool.csv',index=False,encoding='utf-8-sig')
    immutable_json(output/'protocol.json',{'config':config,'methods':list(METHODS),'development_queries':sorted(DEVELOPMENT),
        'paths':PATHS,'rubric':RUBRIC,'rubric_sha256':fingerprint(RUBRIC),'pool_sha256':file_hash(ROOT/PATHS['pool']),
        'query_intents_status':'AUTHOR_INTERPRETATION_FROZEN_BEFORE_GEMINI_CALLS_NOT_EXPERT_VALIDATED',
        'judge_design':'Same Gemini model, separate stateless calls; same-family dependence, no independent model claim',
        'limits':'AI consensus not expert truth. Query bootstrap conditional on fixed AI labels. Pool-complete common queries only.',
        'synthetic':'200 documents balanced by source; exact source-ID recovery, not general relevance or corpus recall',
        'oov_policy':'Q24 separate non-domain control, saved rankings only; not in primary qrels or relevance means'})
    selection=select_documents(corpus)
    splits=pd.read_parquet(ROOT/'data/stage3/split_assignments.parquet',columns=['doc_id','split','eligible_finetune'])
    overlap=selection[['synthetic_id','doc_id','source','split_group_id']].merge(splits,on='doc_id',validate='one_to_one')
    if len(overlap)!=200 or overlap['split'].isna().any():raise ValueError('Synthetic training-split audit incomplete')
    overlap.to_csv(output/'synthetic/selection_training_overlap.csv',index=False,encoding='utf-8-sig')
    records=selection.to_dict('records')
    immutable_json(output/'synthetic/selection.json',records)
    if not (output/'synthetic/selection.csv').exists():selection.to_csv(output/'synthetic/selection.csv',index=False,encoding='utf-8-sig')
    return docs,queries,pool,selection


def load_votes(output, documents, allow_other_queries=False):
    votes={}
    keys={(d['query_id'],d['doc_id']) for d in documents}
    for path in (output/'votes').glob('*.json') if (output/'votes').exists() else []:
        record=read_json(path)
        j=record['judgment'];key=(record['role'],j['query_id'],j['doc_id'])
        if key[1:] not in keys and allow_other_queries:continue
        if key[0] not in ('A','B','C') or key[1:] not in keys or key in votes:raise ValueError('Unexpected/duplicate vote file')
        cached=read_json(output/'api_cache'/(record['request_sha256']+'.json'))
        if cached['response_sha256']!=fingerprint(cached['response']) or cached['stage']!='JUDGE_'+key[0]:
            raise ValueError('Vote cache provenance mismatch')
        doc=next(d for d in documents if (d['query_id'],d['doc_id'])==key[1:])
        validated=validate_judgment(response_json(cached),doc,cached['requested_model'])
        if validated!=j or cached['server_model_version']!=record['model_version']:
            raise ValueError('Vote differs from actual API response')
        if key[0]!='C' and cached['input_sha256']!=fingerprint(doc):
            raise ValueError('Vote input drift')
        votes[key]=record
    return votes


def run_judges(client, output, docs, config, roles=('A','B','C')):
    votes=load_votes(output,docs,allow_other_queries=True)
    failures=read_json(output/'invalid_responses.json') if (output/'invalid_responses.json').exists() else []
    # Interleave A/B while calling separately, so quota exhaustion leaves paired
    # coverage rather than only 942 A votes. Each role has a different fixed order.
    permutations={role:np.random.default_rng(config['seed']+index).permutation(len(docs)).tolist()
                  for index,role in enumerate(('A','B'))}
    jobs=[]
    for i in range(len(docs)):
        for role in ('A','B'):
            if role in roles:jobs.append((role,docs[permutations[role][i]]))
    for role,doc in jobs:
        key=(role,doc['query_id'],doc['doc_id'])
        if key in votes:continue
        try:
            record=judge(client,doc,role,config)
        except (ValueError,KeyError,TypeError) as exc:
            failures.append({'role':role,'query_id':doc['query_id'],'doc_id':doc['doc_id'],
                             'status':'INVALID_RESPONSE_UNRESOLVED_NOT_ZERO','reason':str(exc)})
            write_json(output/'invalid_responses.json',failures)
            continue
        path=output/'votes'/(fingerprint(key)+'.json');write_json(path,record);votes[key]=record
        if len(votes)%10==0:print('Valid real Gemini judgments: '+str(len(votes)),flush=True)
    if 'C' not in roles:return
    for doc in docs:
        pair=(doc['query_id'],doc['doc_id']);a,b=votes.get(('A',*pair)),votes.get(('B',*pair))
        if a is None or b is None or ('C',*pair) in votes:continue
        if a['judgment']['grade']==b['judgment']['grade']:continue
        original=dict(doc)
        original['adjudication_task']='Resolve the two judgments against the original evidence; retain UNRESOLVED if insufficient.'
        original['prior_judgments']=[a['judgment'],b['judgment']]
        try:
            record=judge(client,original,'C',config)
        except (ValueError,KeyError,TypeError) as exc:
            failures.append({'role':'C','query_id':doc['query_id'],'doc_id':doc['doc_id'],
                             'status':'INVALID_ADJUDICATION_UNRESOLVED_NOT_ZERO','reason':str(exc)})
            write_json(output/'invalid_responses.json',failures);continue
        write_json(output/'votes'/(fingerprint(('C',*pair))+'.json'),record);votes[('C',*pair)]=record


def run_query_blocks(client, output, docs, config):
    # A fixed random order unrelated to grades; completes common query pools sooner
    # under daily quota. Existing votes are reused without selective rejudgment.
    query_ids=sorted({doc['query_id'] for doc in docs})
    order=np.random.default_rng(config['seed']).permutation(query_ids).tolist()
    immutable_json(output/'query_processing_order.json',{'order':order,'seed':config['seed'],
        'policy':'Fixed random query blocks; separate shuffled A/B calls then C per query; no result-based selection'})
    for qid in order:
        run_judges(client,output,[d for d in docs if d['query_id']==qid],config)
        queries,pool=csv(ROOT/PATHS['queries']),csv(ROOT/PATHS['pool'])
        labels,agree,results=export_metrics(output,docs,queries,pool,config)
        report(output,labels,agree,results,'RUNNING_PARTIAL_AI_EVALUATION')
        write_json(output/'run_state.json',{'updated_at':utc_now(),'phase':'all','status':'RUNNING',
            'last_query_block':qid,'blocking_status':None,'pool_pairs':len(labels),
            'ai_numeric_consensus':int(labels.label_status.eq(AI_STATUS).sum()),
            'complete_query_pools':results['scope']['complete_query_pools'],
            'synthetic_generations':len(synthetic_generations(output))})
        print('Completed judge block '+qid,flush=True)


def export_metrics(output, docs, queries, pool, config):
    votes=load_votes(output,docs)
    labels=pd.DataFrame([consensus_record(doc,*[votes.get((r,doc['query_id'],doc['doc_id'])) for r in ('A','B','C')]) for doc in docs])
    labels.to_csv(output/'ai_consensus_qrels.csv',index=False,encoding='utf-8-sig')
    for role in ('A','B','C'):
        raw=[{**r['judgment'],'request_sha256':r['request_sha256'],'model_version':r['model_version'],'role':role}
             for key,r in votes.items() if key[0]==role]
        write_json(output/('judge_'+role+'_raw.json'),raw)
    agree=agreement(docs,votes);agree['design']=config['agreement_scope']
    agree['structurally_repaired_selected_votes']={role:sum(r.get('validation_attempt',0)>0 for key,r in votes.items() if key[0]==role) for role in ('A','B','C')}
    agree['selection_policy']='First structurally/evidence-valid response; valid grades never retried for their value; raw rejected responses preserved'
    write_json(output/'agreement.json',agree)
    rankings=csv(ROOT/PATHS['rankings'])
    results=evaluate_ai(rankings,queries,pool,labels,csv(ROOT/PATHS['runs']),repetitions=config['bootstrap_repetitions'])
    for name,value in results.items():
        if isinstance(value,pd.DataFrame):value.to_csv(output/(name+'.csv'),index=False,encoding='utf-8-sig')
        else:write_json(output/(name+'.json'),value)
    errors=error_analysis(rankings,labels);errors.to_csv(output/'errors_by_query.csv',index=False,encoding='utf-8-sig')
    unique_errors=errors.drop_duplicates(['query_id','doc_id'])
    diagnosis=[{'axis':axis,'ai_diagnosis':value,'unique_low_grade_pairs':int(unique_errors[axis].eq(value).sum()),
                'scope':'Only AI-consensus grade<2 pairs; not corpus prevalence or expert error labels'}
               for axis in ('battery_system','material_formula','technical_condition','technical_relation')
               for value in ('MATCH','MISMATCH','PARTIAL','INSUFFICIENT','NOT_REQUIRED','JUDGES_DISAGREE')]
    pd.DataFrame(diagnosis).to_csv(output/'error_summary.csv',index=False,encoding='utf-8-sig')
    if len(errors):
        for axis in ('battery_system','material_formula','technical_condition','technical_relation'):
            errors.groupby(['query_id','method',axis]).size().rename('result_slots').reset_index().to_csv(output/('error_counts_'+axis+'.csv'),index=False,encoding='utf-8-sig')
    # Keep Q24 as a distinct diagnostic with no invented relevance labels.
    oov=rankings[rankings.query_id.eq('Q24')].groupby('method').size().reindex(METHODS,fill_value=0).rename('returned_top5').reset_index()
    oov['interpretation']='OOV control return count, not relevance or calibrated OOD accuracy'
    oov.to_csv(output/'oov_control.csv',index=False,encoding='utf-8-sig')
    return labels,agree,results


def synthetic_generations(output):
    folder=output/'synthetic/generations'
    return [read_json(p) for p in sorted(folder.glob('*.json'))] if folder.exists() else []


def freeze_synthetic_queries(output):
    selected={r['synthetic_id']:r for r in read_json(output/'synthetic/selection.json')}
    rows=[]
    for record in synthetic_generations(output):
        sid=record['synthetic_id'];source=selected[sid]
        cached=read_json(output/'api_cache'/(record['request_sha256']+'.json'))
        if (cached['input_sha256']!=fingerprint({'abstract':source['abstract']}) or
            cached['response_sha256']!=fingerprint(cached['response']) or
            record['query_sha256']!=fingerprint(record['query']) or
            record['abstract_sha256']!=fingerprint(source['abstract'])):
            raise ValueError('Synthetic generation provenance/input/query drift')
        if record['status']=='OK':
            raw=response_json(cached)
            if raw['status']!='OK' or raw['query'].strip()!=record['query']:
                raise ValueError('Frozen query differs from genuine generation response')
        rows.append({key:record[key] for key in ('synthetic_id','status','query','query_sha256','request_sha256','model_id','model_version')} |
                    {'doc_id':source['doc_id'],'source':source['source']})
    frame=pd.DataFrame(rows,columns=['synthetic_id','status','query','query_sha256','request_sha256','model_id','model_version','doc_id','source'])
    path=output/'synthetic/frozen_queries.csv'
    if path.exists():
        old=csv(path).set_index('synthetic_id')
        current=frame.set_index('synthetic_id')
        if not set(old.index)<=set(current.index) or any(old.loc[s,'status']=='OK' and old.loc[s,'query_sha256']!=current.loc[s,'query_sha256'] for s in old.index):
            raise ValueError('Previously frozen synthetic query changed or vanished')
    frame.to_csv(path,index=False,encoding='utf-8-sig')
    manifest={'selected_documents':len(selected),
        'queries':[{'synthetic_id':row['synthetic_id'],'query_sha256':row['query_sha256'],'request_sha256':row['request_sha256']} for row in rows if row['status']=='OK'],
        'policy':'Generated solely from Abstract; first structurally valid response frozen before that query is searched; no result-based edits'}
    path=output/'synthetic/frozen_queries_manifest.json'
    if path.exists():
        previous=read_json(path)
        prior=[r for r in previous['queries'] if r['query_sha256']!=fingerprint('')]
        by_id={r['synthetic_id']:r for r in manifest['queries']}
        if any(by_id.get(r['synthetic_id'])!=r for r in prior):raise ValueError('Frozen accepted synthetic manifest changed')
        if previous!=manifest:write_json(output/'synthetic/manifest_history'/(fingerprint(previous)+'.json'),previous)
    write_json(path,manifest)
    return frame


def report(output, labels, agree, results, pending=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts=labels.label_status.value_counts()
    fig,ax=plt.subplots(figsize=(9,4))
    bars=counts.sort_index();ax.bar(bars.index,bars.values)
    ax.set_ylabel('Query-document pairs');ax.set_title('Actual Gemini judgment coverage (not search quality)')
    ax.tick_params(axis='x',labelrotation=15);fig.tight_layout();fig.savefig(output/'judgment_coverage.png',dpi=160);plt.close(fig)
    scored=results['summary'].query("split == 'remaining' and queries_evaluated > 0")
    if len(scored):
        fig,ax=plt.subplots(figsize=(12,5))
        scored.set_index('method')[['Precision_at_5','nDCG_at_5','MRR_at_5']].plot.bar(ax=ax,ylim=(0,1))
        ax.legend(['P@5','nDCG@5','MRR@5'],loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
        ax.set_title(f'AI-judged remaining queries: n={int(scored.iloc[0].queries_evaluated)}/18; partial, not expert ground truth');fig.tight_layout()
        fig.savefig(output/'ai_relevance_metrics_remaining.png',dpi=160);plt.close(fig)
    gen=synthetic_generations(output)
    overlap_path=output/'synthetic/selection_training_overlap.csv'
    training_note=''
    if overlap_path.exists():
        counts=csv(overlap_path)['split'].value_counts().to_dict()
        training_note='Phân bố source documents theo split fine-tuning cũ: '+json.dumps(counts,ensure_ascii=False)+'.'
    if not (output/'synthetic/summary.csv').exists():
        pd.DataFrame([{'method':m,'Hit_at_1':None,'Hit_at_5':None,'Hit_at_10':None,'MRR_at_10':None,
                       'queries_evaluated':0,'queries_planned':200,'status':'PENDING_GENERATION_OR_RETRIEVAL'} for m in METHODS]).to_csv(output/'synthetic/summary.csv',index=False,encoding='utf-8-sig')
    summary=results['summary'][results['summary'].split.eq('all')]
    lines=['# Đánh giá tìm kiếm bằng Gemini — phiên bản v1','',
        '**Trạng thái:** '+(pending or ('AI_JUDGED_COMPLETE' if labels.label_status.eq(AI_STATUS).all() else 'PENDING_OR_PARTIAL')),
        '',f"Pool cố định: 942 cặp, 23 truy vấn miền, 14 phương pháp. Nhãn AI số đã đồng thuận: {int(labels.label_status.eq(AI_STATUS).sum())}/942.",
        f"A/B đã có cả hai lượt gọi hợp lệ: {agree['completed_both_calls']}/942; số cặp numeric dùng agreement: {agree['paired_numeric_ratings']}.",
        f"Exact agreement: {agree['exact_agreement']}; quadratic weighted kappa: {agree['quadratic_weighted_kappa']}.",
        'Đây là các lượt Gemini cùng model, không có lịch sử chấm chung; không phải chuyên gia hoặc hai họ mô hình độc lập.',
        f"Các lượt phân xử C hợp lệ: {int(labels.resolution.eq('C_ADJUDICATION').sum())}; đang chờ phân xử: {int(labels.label_status.eq('PENDING_ADJUDICATION').sum())}. Agreement chỉ mô tả các cặp A/B đã có response hợp lệ, có thể sau sửa cấu trúc; không đo độ đúng của nhãn.",
        '', '## Bảng chất lượng Top5', '', '| Method | P@5 | nDCG@5 | MRR@5 | Queries |', '|---|---:|---:|---:|---:|']
    for row in summary.itertuples():
        fmt=lambda v:'PENDING' if pd.isna(v) else f'{v:.4f}'
        lines.append(f'| {row.method} | {fmt(row.Precision_at_5)} | {fmt(row.nDCG_at_5)} | {fmt(row.MRR_at_5)} | {row.queries_evaluated}/23 |')
    lines+=['','UNRESOLVED và chưa gọi API giữ trống; không tính thành 0. Chỉ pool của một query đã chấm đủ mới được đưa vào trung bình cho tất cả methods. Nếu chỉ có một phần query đủ nhãn, mean bị giới hạn bởi selection/coverage bias.',
        'summary.csv tách all/development (5)/remaining (18); 18 query còn lại vẫn do tác giả soạn, chưa phải benchmark ngoài độc lập.',
        'nDCG dùng gain 2^grade−1, IDCG từ toàn pool chung của query. Pool có IDCG=0 vẫn được giữ với nDCG=0. P@5 và MRR@5 dùng grade≥2.',
        '', '## Synthetic benchmark','',f'Đã chọn trước 200 tài liệu: 100 Patent và 100 Publication; đã lưu {len(gen)} generation records, trong đó OK: {sum(g["status"]=="OK" for g in gen)}, chưa hợp lệ/UNRESOLVED: {sum(g["status"]!="OK" for g in gen)}.',
        'Generator chỉ nhận toàn Abstract, không nhận Title/doc_id. Query đóng băng trước retrieval, không sửa theo kết quả. Kiểm tra chống sao chép Title là heuristic, không chứng minh truy vấn giống người dùng thật.',
        'synthetic/summary.csv là Hit@1/5/10 và MRR@10 tìm lại đúng doc_id nguồn; không phải relevance, không phải Recall toàn corpus. Tài liệu trùng/nhóm và các tài liệu khác cùng đáp ứng query có thể làm metric này đánh giá thiếu chất lượng.',
        training_note,
        'Các source documents thuộc train có thể thuận lợi cho nhánh fine-tuned; synthetic benchmark này không phải held-out-document benchmark. by_training_split.csv tách chẩn đoán khi đã chạy retrieval. Nội dung query được AI sinh chưa được chuyên gia xác minh độ trung thành với Abstract; format/overlap checks không phát hiện được mọi hallucination.',
        '', '## Phân tích lỗi và giới hạn','',
        'errors_by_query.csv chỉ gồm kết quả có AI consensus grade<2, chia theo hệ pin, vật liệu/công thức, điều kiện và quan hệ kỹ thuật. Các nhãn trục lỗi là diễn giải AI, chưa phải nguyên nhân đã được chuyên gia xác minh. A/B cùng grade nhưng khác trục lỗi được ghi JUDGES_DISAGREE.',
        'Nếu chưa có nhãn thì bảng lỗi rỗng có header, không diễn giải là không có lỗi. Trích dẫn đúng substring chỉ chứng minh bằng chứng có trong đầu vào; không chứng minh suy luận relevance đúng. Confidence của LLM không được hiệu chỉnh.',
        'Hai lần gọi cùng Gemini có sai lệch tương quan; agreement cao không chứng minh đúng. Judge C vẫn có thiên lệch cùng họ và anchoring. Không dùng AI labels để ghi đè trạng thái nhãn chuyên gia của báo cáo cũ.',
        'CI dùng paired bootstrap theo query trên nhãn AI/checkpoints cố định, không bao gồm biến thiên judge, training seed hoặc domain shift. 91 cặp so sánh là exploratory, chưa chỉnh multiple comparisons; không tự tuyên bố phương pháp thắng.',
        'Pool942 lấy từ các phương pháp hiện tại nên có pooling bias, không bao phủ toàn corpus. Chưa kết luận Recall hoặc hiệu quả ngoài benchmark này.',
        '', '## Vận hành và bảo toàn','',
        'Quyền gửi dữ liệu và billing-off do người dùng xác nhận; không thể xác minh Cloud Billing từ API key. Chỉ model Standard có free tier trong allowlist được phép; không chuyển sang paid, grounding, Batch hoặc caching phía Google.',
        'Google có thể dùng nội dung Free Tier để cải thiện dịch vụ và cho người rà soát đọc: https://ai.google.dev/gemini-api/terms ; giá: https://ai.google.dev/gemini-api/docs/pricing .',
        'api_preflight.json/last_api_error.json ghi trạng thái HTTP thật, không chứa key. Khóa cũ bị Google báo leaked đã được người dùng thay. Không ghi giá trị key trong prompt, URL, log hay artifacts.',
        'protected_inputs.json + input_verification.json xác minh dữ liệu/model/source đã đóng băng. Qdrant không mở. api_cache/ lưu response thật kèm SHA-256, modelVersion, prompt/input hashes, usage metadata; votes/ lưu kết quả đã kiểm tra evidence.',
        'Quota/403/429/network lỗi dừng invocation và checkpoint; resume đọc cache, không tự kích hoạt billing. Chưa có kết quả thật thì giữ PENDING.',
        '', '## Chạy tiếp','', '```powershell',
        '.\\.venv\\Scripts\\python.exe -m scripts.run_retrieval_ai_evaluation --phase all',
        '.\\.venv\\Scripts\\python.exe -m scripts.run_retrieval_ai_evaluation --phase metrics',
        '.\\.venv\\Scripts\\python.exe -m scripts.run_retrieval_ai_evaluation --phase parity',
        '.\\.venv\\Scripts\\python.exe -m scripts.run_retrieval_ai_evaluation --phase synthetic-retrieve',
        '```','', 'Xem RUN_GUIDE.md cho phases và điều kiện resume. Không tái chấm chọn lọc để tăng điểm.']
    lines.extend(research_findings(output,results))
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def research_findings(output,results):
    lines=[]
    observed=results['summary'].query("split == 'remaining' and queries_evaluated > 0").sort_values(['nDCG_at_5','method'],ascending=[False,True])
    if len(observed):
        lines+=['','## Quan sát định lượng hiện có','',
            f"Có {int(observed.iloc[0].queries_evaluated)}/18 remaining queries đủ pool. Point estimates nDCG@5 cao nhất trong phần đã chấm:",'']
        for row in observed.head(3).itertuples():
            lines.append(f'- {row.method}: nDCG@5={row.nDCG_at_5:.4f}, P@5={row.Precision_at_5:.4f}, MRR@5={row.MRR_at_5:.4f}.')
        lines+=['','Đây là mô tả trên phần query đã chấm và nhãn AI hiện tại; không phải kết luận thắng trên toàn benchmark hoặc chuyên gia.']
        contrasts=[('DENSE_FINETUNED','DENSE_PRETRAINED'),('DENSE_CHUNKED','DENSE_FINETUNED'),
                   ('HYBRID_PRETRAINED','DENSE_PRETRAINED'),('RERANK_PRETRAINED','HYBRID_PRETRAINED')]
        pairs=results['paired_deltas']
        for left,right in contrasts:
            direct=pairs[(pairs['split']=='remaining')&(pairs.metric=='nDCG_at_5')&(pairs.left==left)&(pairs.right==right)]
            reverse=pairs[(pairs['split']=='remaining')&(pairs.metric=='nDCG_at_5')&(pairs.left==right)&(pairs.right==left)]
            if len(direct):
                row=direct.iloc[0];delta,low,high=row.mean_delta,row.ci95_low,row.ci95_high
            elif len(reverse):
                row=reverse.iloc[0];delta,low,high=-row.mean_delta,-row.ci95_high,-row.ci95_low
            else:continue
            lines.append(f'- nDCG@5 {left} − {right}: {delta:+.4f}, CI95% [{low:+.4f}, {high:+.4f}]; exploratory, chưa chỉnh multiple comparisons.')
    errors=csv(output/'errors_by_query.csv')
    if len(errors):
        unique=errors.drop_duplicates(['query_id','doc_id'])
        lines+=['','## Ví dụ và trục lỗi được AI ghi nhận','',f'{len(errors)} result slots grade<2 tương ứng {len(unique)} cặp query–document khác nhau.',
            'Không đếm lặp giữa methods khi thống kê cặp. Query thiếu toàn pool có thể có chẩn đoán lỗi nhưng chưa được dùng tính metrics.','']
        for axis in ('battery_system','material_formula','technical_condition','technical_relation'):
            lines.append(f'- {axis}: {int(unique[axis].eq("MISMATCH").sum())} cặp MISMATCH, {int(unique[axis].eq("PARTIAL").sum())} PARTIAL theo AI.')
        examples=unique.sort_values(['query_id','rank','doc_id']).groupby('query_id',sort=True).head(1).head(5)
        for row in examples.itertuples():
            lines+=['',f'**{row.query_id} — {row.doc_id}**: {row.title}',f'AI grade {row.relevance_grade}; lý do: {row.reason}']
        lines+=['','Các ví dụ minh họa diễn giải AI; đọc evidence gốc trong qrels/cache trước khi dùng để kết luận khoa học.']
    synthetic=csv(output/'synthetic/summary.csv')
    if synthetic.queries_evaluated.gt(0).any():
        lines+=['','## Kết quả tìm lại source document','',
            '| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Queries |','|---|---:|---:|---:|---:|---:|']
        for row in synthetic.itertuples():
            lines.append(f'| {row.method} | {row.Hit_at_1:.4f} | {row.Hit_at_5:.4f} | {row.Hit_at_10:.4f} | {row.MRR_at_10:.4f} | {row.queries_evaluated}/200 |')
        lines+=['','Đây là exact source-ID recovery trên AI-generated queries, không phải relevance. CI bootstrap theo split_group_id; xem synthetic/confidence_intervals.csv và paired_deltas.csv.']
        top=synthetic.sort_values(['Hit_at_10','method'],ascending=[False,True]).iloc[0]
        lines+=['',f"Point estimate Hit@10 cao nhất trong phần đã chạy: **{top['method']}**, Hit@10={top['Hit_at_10']:.4f}, MRR@10={top['MRR_at_10']:.4f}. Không dùng thứ hạng source recovery này thay cho chất lượng relevance trên truy vấn tự nhiên."]
        if (output/'synthetic/per_query.csv').exists():
            per=csv(output/'synthetic/per_query.csv')
            counts=per.drop_duplicates('synthetic_id').source.value_counts().to_dict()
            lines+=['','Số query đã chạy theo nguồn: '+json.dumps(counts,ensure_ascii=False)+'. Query bị loại vì định dạng/Title overlap không được tính thành miss; mẫu còn thiếu có thể gây selection bias.']
            splits=per.drop_duplicates('synthetic_id')['split'].value_counts().to_dict()
            lines+=['','Split fine-tuning của source documents đã chạy: '+json.dumps(splits,ensure_ascii=False)+'. Kết quả có điều kiện trên corpus/checkpoints và query do AI sinh; không phải đánh giá độc lập với toàn bộ dữ liệu training.']
        lines+=['','Các cấu hình chunked còn thay đổi Title context, cửa sổ và pooling so với baseline; chênh lệch không tự chứng minh tác dụng riêng của full coverage. Bootstrap theo query không bao gồm training-seed hoặc judge/generation-seed uncertainty.']
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,5))
        synthetic.set_index('method')[['Hit_at_1','Hit_at_10','MRR_at_10']].plot.bar(ax=ax,ylim=(0,1))
        ax.legend(['Hit@1','Hit@10','MRR@10'],loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
        ax.set_title(f'Synthetic exact source-ID recovery: n={int(synthetic.iloc[0].queries_evaluated)}/200; not relevance');fig.tight_layout()
        fig.savefig(output/'synthetic/source_recovery_metrics.png',dpi=160);plt.close(fig)
    if (output/'tests.xml').exists():
        import xml.etree.ElementTree as ET
        root=ET.parse(output/'tests.xml').getroot()
        suites=[root] if root.tag=='testsuite' else list(root.findall('testsuite'))
        totals={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        passed=totals['tests']-sum(totals[k] for k in ('failures','errors','skipped'))
        lines+=['','## Kiểm thử cục bộ','',f"JUnit tests.xml: {passed} passed, {totals['failures']} failures, {totals['errors']} errors, {totals['skipped']} skipped. Fixtures không phải nhãn nghiên cứu."]
    return lines


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['prepare','all','judge-a','judge-b','adjudicate','metrics','synthetic-generate','synthetic-retrieve','parity'],default='prepare')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--env-file',type=Path,default=ROOT/'docs/api.env')
    parser.add_argument('--runtime-rpm',type=int,default=None,help='Pacing only, 1..60; adaptive backoff on minute quota, no generation changes')
    parser.add_argument('--skip-synthetic-retrieval',action='store_true',help='Orchestration only: separate read-only CPU worker owns synthetic/runs')
    args=parser.parse_args();output=args.output.resolve()
    if output!=DEFAULT_OUTPUT.resolve():raise ValueError('v1 writes only to outputs/retrieval_ai_evaluation_v1')
    output.mkdir(parents=True,exist_ok=True)
    # A Windows file lock is released by the OS on interruption; prevents two
    # invocations from racing when they write cache, votes and summary files.
    if sys.platform=='win32':
        import msvcrt
        lock=(output/'.run.lock').open('a+b')
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('Evaluation already running; use one invocation at a time')
        atexit.register(lock.close)
    config=read_json(ROOT/'configs/ai_retrieval_evaluation_v1.json')
    docs,queries,pool,selection=prepare(output,config)
    previous=read_json(output/'run_state.json') if (output/'run_state.json').exists() else {}
    pending=previous.get('blocking_status') if args.phase in ('prepare','metrics','parity','synthetic-retrieve') else None
    try:
        if args.phase in ('all','judge-a','judge-b','adjudicate','synthetic-generate'):
            authorization=read_json(output/'authorization.json') if (output/'authorization.json').exists() else {}
            try:
                client=GeminiFreeClient(output,config,authorization,args.env_file)
                if args.runtime_rpm is not None:
                    if not 1<=args.runtime_rpm<=60:raise ValueError('RPM must be 1..60')
                    client.rpm=args.runtime_rpm
                client.preflight()
                if args.phase in ('all','synthetic-generate'):
                    for row in selection.to_dict('records'):
                        path=output/'synthetic/generations'/(row['synthetic_id']+'.json')
                        if path.exists():
                            prior=read_json(path)
                            if prior['status']!='INVALID_RESPONSE':continue
                            if list((output/'synthetic/runs').glob(row['synthetic_id']+'_*')):
                                raise ValueError('Cannot regenerate a query after its retrieval')
                        try:value=generate_query(client,row,config)
                        except ValueError as exc:
                            write_json(output/'synthetic/last_invalid_generation.json',{'synthetic_id':row['synthetic_id'],'reason':str(exc)})
                            continue
                        write_json(path,value)
                        print('Synthetic generation '+row['synthetic_id'],flush=True)
                if args.phase=='all':
                    run_query_blocks(client,output,docs,config)
                elif args.phase in ('judge-a','judge-b','adjudicate'):
                    roles={'judge-a':('A',),'judge-b':('B',),'adjudicate':('C',)}[args.phase]
                    run_judges(client,output,docs,config,roles)
            except PendingAPI as exc:
                pending=str(exc);print(pending,flush=True)
        if args.phase in ('all','parity'):
            if not verify_legacy_parity(output):pending='PENDING_LEGACY_PARITY'
        if args.phase in ('all','synthetic-retrieve') and not args.skip_synthetic_retrieval and synthetic_generations(output):
            retrieve_synthetic(output,selection,synthetic_generations(output))
    except PendingAPI as exc:
        pending=str(exc);print(pending,flush=True)
    finally:
        freeze_synthetic_queries(output)
        api_usage(output)
        verify_synthetic_runs(output)
        labels,agree,results=export_metrics(output,docs,queries,pool,config)
        verify_inputs(output)
        report(output,labels,agree,results,pending)
        write_json(output/'run_state.json',{'updated_at':utc_now(),'phase':args.phase,'blocking_status':pending,
            'ai_numeric_consensus':int(labels.label_status.eq(AI_STATUS).sum()),'pool_pairs':len(labels),
            'complete_query_pools':results['scope']['complete_query_pools'],'synthetic_generations':len(synthetic_generations(output))})
        scan_active_credential(output,args.env_file)
    print(json.dumps(read_json(output/'run_state.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
