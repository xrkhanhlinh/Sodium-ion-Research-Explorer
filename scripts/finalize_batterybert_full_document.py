"""Build/verify the new index, benchmark actual searches and report evidence."""
import os
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import argparse
import json
import hashlib
import time
import pandas as pd
import numpy as np
from src.model_comparison_io import write_json

OUT=ROOT/'outputs/batterybert_full_document_20261024_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def index():
    from src.final_system_vectors import VectorDatabase
    from src import full_document_topics,optimized_topics
    selection=read(OUT/'selection.json');coverage=read(OUT/'training_coverage.json');parity=read(OUT/'model_reload_parity.json')
    topics_metric=read(OUT/'topics/final_metrics.json')
    review_path=OUT/'topics/agent_review.json'
    review=read(review_path) if review_path.exists() else {}
    representatives=pd.read_csv(OUT/'topics/representative_documents.csv')
    reviewed=(review.get('reviewer_type')=='AGENT_NOT_EXPERT' and
        set(review.get('reviewed_document_ids',[]))==set(representatives.doc_id) and
        set(review.get('reviewed_topic_ids',[]))==set(range(10)) and
        review.get('independent_ground_truth') is False)
    activate_topics=bool(topics_metric['promotion_gate'] and reviewed and review.get('recommend_as_main_topic_pipeline') is True)
    assignments=full_document_topics.assignments() if activate_topics else optimized_topics.assignments()
    db=VectorDatabase(OUT/'qdrant',artifact_dir=OUT,topics=assignments)
    try:db.build('batterybert_fulldoc');db.verify()
    finally:db.close()
    activation=bool(selection['accepted'] and coverage['status']=='PASS' and parity['status']=='PASS')
    write_json(OUT/'deployment.json',dict(activate_search=activation,activate_topics=activate_topics,
        title_abstract_scope=True,article_full_text_available=False,independent_expert_validation=False,
        topic_agent_review=reviewed,reason='Validation-selected continuation and numerical/index gates; relevance still pending human labels'))
    baseline=read(OUT/'baseline/validation/metrics.json');test=read(OUT/'final_test/metrics.json')
    pd.DataFrame([dict(stage='baseline',split='validation',group_mrr_at_10=baseline['group_mrr_at_10'],protocol=baseline['protocol']),
        dict(stage='selected',split='validation',group_mrr_at_10=selection['best_score'],protocol=baseline['protocol']),
        dict(stage='selected',split='test',group_mrr_at_10=test['group_mrr_at_10'],protocol=test['protocol'])]).to_csv(OUT/'training_comparison.csv',index=False)
    print('INDEX/DEPLOYMENT',read(OUT/'deployment.json'),flush=True)


def benchmark():
    from src.final_search import FinalSearchEngine
    from src import full_document_inference as full
    if not full.ready():raise ValueError('Candidate is not enabled after validation; do not benchmark an assumed winner')
    if (OUT/'annotation_protocol.json').exists():
        from src.full_document_evaluation import evaluate
        evaluate();print('Retaining frozen benchmark and annotation pool',flush=True);return
    dest=OUT/'benchmark';dest.mkdir(exist_ok=True)
    queries=pd.read_csv(ROOT/'outputs/research_completion_v1/queries.csv')
    engine=FinalSearchEngine();rows=[];runs=[]
    try:
        for method in full.METHODS:
            for r in queries.itertuples():
                path=dest/f'{method}_{r.query_id}.json'
                if path.exists():response=read(path)
                else:
                    response=engine.search(r.query,method)
                    write_json(path,response)
                for result in response['results']:
                    rows.append(dict(query_id=r.query_id,query=r.query,method=method,**{k:v for k,v in result.items() if k!='topics'}))
                runs.append(dict(query_id=r.query_id,method=method,seconds=response['seconds'],results=len(response['results'])))
                print('BENCH',method,r.query_id,len(response['results']),flush=True)
    finally:engine.close()
    ranked=pd.DataFrame(rows);ranked.to_csv(OUT/'rankings_top10.csv',index=False)
    run_frame=pd.DataFrame(runs);run_frame.to_csv(OUT/'benchmark_runs.csv',index=False)
    warm=run_frame[run_frame.groupby('method').cumcount()>0]
    warm.groupby('method').seconds.agg(['count','mean','median','max']).reset_index().to_csv(OUT/'benchmark_latency.csv',index=False)
    old=pd.read_csv(ROOT/'outputs/research_completion_v1/rankings_top10.csv')
    combined=pd.concat([old,ranked],ignore_index=True);combined.to_csv(OUT/'comparison_rankings_top10.csv',index=False)
    canonical=queries.set_index('query_id')['query']
    if not combined['query'].eq(combined.query_id.map(canonical)).all():raise ValueError('Benchmark query texts differ across methods')
    queries.to_csv(OUT/'queries.csv',index=False)
    old_runs=pd.read_csv(ROOT/'outputs/research_completion_v1/run_manifest.csv')
    new_runs=run_frame.rename(columns={'results':'result_count'}).assign(status='OK')
    pd.concat([old_runs,new_runs],ignore_index=True).to_csv(OUT/'comparison_run_manifest.csv',index=False)
    pool=combined[['query_id','query','doc_id','title','abstract','source','year','identifier']].drop_duplicates(['query_id','doc_id'])
    pool=pool.sort_values(['query_id','doc_id'],kind='stable');pool.to_csv(OUT/'annotation_pool.csv',index=False)
    blind=pool.copy()
    for field in ['relevance_grade','annotator_id','verifier_id','verification_reference']:blind[field]=''
    blind['label_method']='HUMAN_INDEPENDENT';blind['label_status']=''
    blind.to_csv(OUT/'annotation_template_0_2.csv',index=False)
    from src.final_system_artifacts import sha
    write_json(OUT/'annotation_protocol.json',dict(scale=[0,1,2],precision_and_mrr_threshold=2,
        blank_is_zero=False,ranks_and_scores_hidden=True,independent_verifier_required=True,
        evaluation_role='Locked final evaluation; labels must not be used for tuning',
        pool_sha256=sha(OUT/'annotation_pool.csv'),queries_sha256=sha(OUT/'queries.csv'),
        rankings_sha256=sha(OUT/'comparison_rankings_top10.csv')))
    from src.full_document_evaluation import evaluate
    evaluate()
    print('NEW POOL',len(pool),'pairs; no invented grades',flush=True)


def report():
    from src.final_system_artifacts import sha
    cfg=read(OUT/'selection.json');coverage=read(OUT/'coverage_audit.json');training=read(OUT/'training_coverage.json')
    baseline=read(OUT/'baseline/validation/metrics.json');baseline_test=read(OUT/'baseline/test/metrics.json');test=read(OUT/'final_test/metrics.json')
    proof=read(OUT/'embedding_success.json');parity=read(OUT/'model_reload_parity.json')
    topic=read(OUT/'topics/final_metrics.json');deploy=read(OUT/'deployment.json')
    cv=pd.read_csv(OUT/'topics/common_coherence.csv')
    experiments=[]
    for p in (OUT/'experiments').glob('*/validation_history.json'):
        for m in read(p):experiments.append(dict(candidate=p.parent.name,step=m['step'],group_mrr_at_10=m['group_mrr_at_10']))
    pd.DataFrame(experiments).to_csv(OUT/'all_validation_checkpoints.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    for name,group in pd.DataFrame(experiments).groupby('candidate'):
        axes[0].plot(group.step,group.group_mrr_at_10,marker='o',label=name)
    axes[0].axhline(baseline['group_mrr_at_10'],linestyle='--',color='gray',label='Same-pooling baseline')
    axes[0].set(xlabel='Optimizer step',ylabel='Validation Group MRR@10',title='Title-conditioned full-document proxy')
    axes[0].legend(fontsize=7)
    history=pd.read_csv(OUT/'experiments'/cfg['name']/'training_history.csv')
    axes[1].plot(history.step,history.loss.rolling(20,min_periods=1).mean(),color='#157F83')
    axes[1].set(xlabel='Optimizer step',ylabel='Contrastive + distillation loss',title='Training loss: rolling mean, 20 batches')
    for axis in axes:axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'learning_curve.png',dpi=180);plt.close(fig)
    protected=read(OUT/'input_manifest.json');changes=[]
    for relative,expected in protected.items():
        if sha(ROOT/relative)!=expected:changes.append(relative)
    if changes:raise ValueError('Historical source data/checkpoint changed: '+str(changes))
    write_json(OUT/'preservation_check.json',dict(status='PASS',files=len(protected),changed=changes))
    report=f'''# BatteryBERT full-document: huấn luyện và kết quả ngày24/10/2026

## Phạm vi thực sự

Dữ liệu có13.702 bản ghi Title-Abstract, không có toàn văn bài báo. Pipeline mới đọc
toàn bộ văn bản sẵn có bằng chunk256/overlap64. Không gọi đây là đã train trên toàn
bộ bài báo. API `FullDocumentEncoder.encode_document(title, body)` có thể đọc toàn bộ
body dài do người dùng cung cấp; chất lượng trên toàn văn chưa có đánh giá độc lập.

## Thay đổi và bằng chứng thực thi

Đã tiếp tục train block cuối của checkpoint BatteryBERT fine-tuned bằng gradient
thật, frozen11 block đầu. Cache float32 cũ giữ read-only, bổ sung
{coverage['supplementary_chunks']} chunk còn thiếu và query prefixes không truncation.
Loss dùng batch32 cặp với gradient caching hai tầng, khác batch2/accumulation2 cũ.
Document pooling dùng các token Abstract và trọng số số token mới của mỗi đoạn;
overlap vẫn có mặt làm context. Dropout tắt để replay nhất quán, có warmup/decay,
gradient clipping và distillation nhẹ. Có title context trong các document chunks.

Một lượt train đã xử lý{training['train_documents']} tài liệu, backprop qua
{training['actual_chunks_backpropagated']} đoạn, bao phủ
{training['actual_unique_abstract_tokens']} token Abstract duy nhất theo tài liệu.
Validation/test xuất hiện trong optimizer: {training['validation_and_test_optimizer_appearances']}.
Checkpoint được chọn tại step{cfg['best_step']}, đã thấy
{cfg['selected_checkpoint_rows_seen']} bản ghi full-document trong vòng continuation.
Phân biệt coverage của toàn phiên train với coverage tới checkpoint được chọn.

## Kết quả encoder cùng giao thức

| Đo lường | Kết quả |
|---|---:|
| Validation baseline, cùng full-document pooling | {baseline['group_mrr_at_10']:.6f} |
| Validation checkpoint được chọn | {cfg['best_score']:.6f} |
| Gain validation | {cfg['best_score']-baseline['group_mrr_at_10']:.6f} |
| Test baseline, cùng full-document pooling | {baseline_test['group_mrr_at_10']:.6f} |
| Test checkpoint được chọn | {test['group_mrr_at_10']:.6f} |
| Chấp nhận trọng số continuation | {cfg['accepted']} |
| L2 thay đổi block cuối được export | {proof['selected_last_block_delta_l2']:.6f} |

Đây là paired Title→full document có Title context, có thể được lợi từ title;
không phải relevance search độc lập và không so trực tiếp với proxy Abstract128
cũ để quy mọi gain cho training. Learning rate/checkpoint chọn bằng validation;
test không dùng để chọn. Test là split lịch sử, không phải external test mới.

## Topic và cổng triển khai

Hai geometry(center/source), mỗi geometry ba trọng số exclusivity và hai mức
phrase boost (1 hoặc 1,5) được chọn bằng
validation. KMeans/vocabulary/keywords fit train; seed43 cho validation, seed137
cho cổng cuối. C_v được tính chung reference giữa model trước và mới.

{cv.to_markdown(index=False) if __import__('importlib').util.find_spec('tabulate') else cv.to_csv(index=False)}

- Source Cramér V toàn corpus: {topic['source_cramers_v']:.6f}.
- Top2 share: {topic['top2_topic_share']:.2%}.
- Topic test ARI: {topic['heldout']['stability_ari']:.6f}.
- Topic test NPMI: {topic['heldout']['document_npmi']:.6f}.
- Technical topic promotion gate: {topic['promotion_gate']}.
- Bật Search mới: {deploy['activate_search']}; bật topic mới: {deploy['activate_topics']}.

Nhãn và review của agent không phải expert ground truth. Nếu cổng topic không
đạt, catalog topic cũ tiếp tục được dùng; Search có quyết định triển khai
riêng. Giữ nguyên kết quả không đạt để đối chứng.

## Kiểm chứng

`model_reload_parity.json`: {parity['status']}; so prefix+last-block với forward
đầy đủ ở các tài liệu thật, gồm tài liệu dài nhất. `training_document_coverage.csv`
ghi ID và số lần tham gia optimizer. `input_manifest.json` và
`preservation_check.json` xác minh dữ liệu/checkpoint lịch sử không đổi.
`rankings_top10.csv`, `benchmark_runs.csv`, `benchmark_latency.csv` lưu requests
thật của hai method mới. Pool annotation mới bao gồm kết quả cũ và mới; chưa có
nhãn người thì không điền P@10/nDCG@10/MRR@10 relevance.

## Giới hạn và nhận xét

Đọc hết token qua các cửa sổ không tạo attention toàn cục giữa mọi đoạn.
Tổng hợp có thể làm mất chi tiết quan hệ dài; corpus không có full article,
không có nhãn khoa học độc lập và chưa đo concurrency. Không tuyên bố tối ưu
tuyệt đối. Trọng tâm cải thiện lần này là nhất quán độ dài train/inference,
số negatives thực, bảo toàn tail, kiểm chứng numerical và đối chứng minh bạch.

[Gao et al.(2021), gradient caching](https://arxiv.org/abs/2101.06983).
'''
    (OUT/'TRAINING_REPORT.md').write_text(report,encoding='utf8')
    (ROOT/'docs/BATTERYBERT_FULL_DOCUMENT_RESULTS_20261024.md').write_text(report,encoding='utf8')
    print('TRAINING REPORT WRITTEN',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['index','benchmark','report','evaluate'])
    parser.add_argument('--labels',type=Path)
    args=parser.parse_args()
    if args.phase=='evaluate':
        from src.full_document_evaluation import evaluate
        print(evaluate(args.labels).to_string(index=False),flush=True)
    else:
        if args.labels is not None:parser.error('--labels is only valid with evaluate')
        {'index':index,'benchmark':benchmark,'report':report}[args.phase]()
