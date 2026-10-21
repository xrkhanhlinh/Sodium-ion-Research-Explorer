"""Stage 2 final report and reproducibility manifest after fresh passing tests."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from .data_io import sha256
from .stage2 import checkpoint,write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'outputs/reports/stage2'
    tables=ROOT/'outputs/tables/stage2'
    tests=reports/'tests.xml'
    runtime_path=reports/'runtime_validation.json'
    if not tests.exists() or tests.stat().st_mtime<runtime_path.stat().st_mtime:
        raise RuntimeError('Run Stage 2 unit/integration tests after the latest successful export')
    tree=ET.parse(tests)
    cases=tree.findall('.//testcase')
    assert cases and not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
    runtime=json.loads(runtime_path.read_text(encoding='utf-8'))
    assert runtime['status']=='PASS'
    inputs=json.loads((reports/'input_manifest.json').read_text(encoding='utf-8'))
    assert all(sha256(ROOT/p)==digest for p,digest in inputs['protected_hashes'].items())
    selection=json.loads((reports/'selected_model.json').read_text(encoding='utf-8'))
    comparison=pd.read_csv(tables/'lda_comparison.csv')
    topics=pd.read_csv(tables/'topic_summary.csv',keep_default_na=False)
    figures=json.loads((reports/'figure_manifest.json').read_text(encoding='utf-8'))
    all_runs=pd.read_csv(tables/'lda_all_run_metrics.csv')
    uncertain=pd.read_parquet(ROOT/'data/stage2/uncertain_sensitivity_pool.parquet')
    seed_stats=all_runs.groupby('k')[['c_v','u_mass','topic_diversity']].agg(['mean','std','count'])
    seed_stats.columns=['_'.join(x) for x in seed_stats.columns]
    seed_stats.reset_index().to_csv(tables/'lda_seed_summary.csv',index=False,encoding='utf-8-sig')
    k=selection['selected_k']
    chosen=comparison[comparison.k.eq(k)].iloc[0]
    runner=comparison[comparison.k.isin(selection['promising']) & comparison.k.ne(k)].iloc[0]
    lines=['# STAGE 2 COMPLETION REPORT','', '**Trạng thái: COMPLETED — chỉ EDA & Gensim LDA (P3–P4).**','',
        '## 1. Corpus và phương pháp','',
        f"- Corpus đóng băng: **{inputs['documents']:,} tài liệu** ({runtime['source_counts']['Patent']:,} Patent, {runtime['source_counts']['Publication']:,} Publication), **{inputs['tokens']:,} token**, **{inputs['dictionary_size']:,} từ vựng**.",
        '- Đọc nguyên Stage 1 main_corpus và lda_tokens; không làm sạch, đổi nhãn hoặc lọc dictionary. Hash của toàn bộ dữ liệu Stage 1 và workbook gốc không đổi.',
        f"- Đã lưu riêng **{len(uncertain):,} UNCERTAIN** trong `data/stage2/uncertain_sensitivity_pool.parquet`, không dùng huấn luyện hoặc chọn K. Nhóm này còn {int(uncertain.retracted.sum())} cờ rút lại và {int(uncertain.duplicate_of.ne('').sum())} cờ trùng bản ghi; giữ nguyên cờ, không ghép thẳng vào training. Đây không phải test set độc lập; nhóm tài liệu liên quan có thể giao với corpus chính.",
        '- Gensim LdaModel: 10 passes, tối đa 100 iterations, chunksize 1024, online update_every=1, alpha asymmetric, eta symmetric; seed cơ sở 42. Thử đủ K=10,15,20,25,30 trên cùng dữ liệu.',
        '- Coherence topn=10: C_v window=110, U_Mass theo đồng xuất hiện tài liệu. Topic Diversity = số từ khác nhau trong toàn bộ Top-10 / (10K). Stability = Jaccard Top-10 ghép topic tối ưu bằng Hungarian, trung bình trên ba cặp seed 42/123/2026 của hai K có C_v seed42 cao nhất.', '',
        '## 2. So sánh cấu hình và chọn K','',
        '| K | C_v (seed42) | U_Mass (seed42) | Diversity | Stability ± SD | Điểm chọn |', '|---:|---:|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        stability='Chưa đo' if pd.isna(row.stability) else f'{row.stability:.6f} ± {row.stability_std:.6f}'
        score='—' if pd.isna(row.selection_score) else f'{row.selection_score:.6f}'
        lines.append(f'| {row.k} | {row.c_v:.6f} | {row.u_mass:.6f} | {row.topic_diversity:.6f} | {stability} | {score} |')
    lines += ['',f"**Chọn K={k}, seed=42.** Hai cấu hình triển vọng theo C_v là {selection['promising']}. K={k} đạt điểm {chosen.selection_score:.6f}, cao hơn K={int(runner.k)} ({runner.selection_score:.6f}) theo quy tắc đã ghi trước khi huấn luyện.",
        'Điểm = 0,50×C_v_minmax + 0,15×U_Mass_minmax + 0,15×Diversity + 0,20×Stability; min/max lấy trên năm cấu hình cơ sở. Chỉ hai cấu hình được đo Stability tham gia bước chọn cuối; K khác chưa được chứng minh kém ổn định. Giữ seed42 cho mô hình dùng tiếp, không chọn seed có kết quả thuận lợi nhất. Đây là lựa chọn vận hành theo các tiêu chí đã công bố, không khẳng định K tối ưu tuyệt đối.',
        f'Đã huấn luyện thật **{len(all_runs)} cấu hình K–seed**, lưu riêng model và metrics; các cấu hình thành công có fingerprint/checksum để tái sử dụng. Metrics từng seed và độ lệch giữa seed: `lda_all_run_metrics.csv`, `lda_seed_summary.csv`. SD ba cặp Stability có dùng chung seed nên không phải sai số chuẩn độc lập hoặc khoảng tin cậy.', '',
        '## 3. Top-10 keywords và số tài liệu/topic','',
        'Topic ID bắt đầu từ 0. Số tài liệu là số gán theo argmax, mỗi tài liệu được đếm một lần. Soft mass là tổng xác suất, được lưu riêng; không phải số tài liệu nhị phân. Tên chủ đề tự nhiên chưa được chuyên gia xác nhận.', '',
        '| Topic | Số tài liệu | Top-10 keywords |','|---:|---:|---|']
    for row in topics.itertuples():
        lines.append(f'| {row.topic_id} | {row.dominant_documents:,} | {row.top10_keywords} |')
    largest=topics.loc[topics.dominant_documents.idxmax()]
    lines += ['', f'Phân bố argmax rất lệch: topic {int(largest.topic_id)} có {int(largest.dominant_documents):,} tài liệu ({100*largest.dominant_documents/topics.dominant_documents.sum():.1f}%), trong khi topic nhỏ nhất chỉ có {int(topics.dominant_documents.min())} tài liệu. Không diễn giải 20 topic như 20 nhóm khoa học cân bằng.']
    lines += ['',f"Tổng gán topic: **{int(topics.dominant_documents.sum()):,}**. Có {int(topics.dominant_documents.eq(0).sum())} topic không có tài liệu argmax. Tài liệu đại diện Top-3/topic, Title/Abstract đầy đủ và xác suất nằm trong `topic_representative_documents.csv`; mọi fallback cho topic rỗng đều có cờ.", '',
        '## 4. EDA, trực quan hóa và kết quả lưu','',
        'EDA lưu thống kê theo nguồn/năm, số token/độ dài, tần suất từ và document frequency chuẩn hóa, token bigram/trigram, author keywords DE của Publication, classification và top organizations. Patent không có DE; không tạo keywords giả. N-grams là chuỗi token liên tiếp sau Stage 1 nên có thể bắc qua stopwords/câu/Title–Abstract đã xử lý.', '',
        'Tất cả biểu đồ PNG 300 DPI, đã có bảng nguồn CSV:', '', '| Hình | Bảng nguồn |','|---|---|']
    sensitivity=['### Độ nhạy theo seed của hai cấu hình triển vọng','', '| K | Seed | C_v trung bình ± SD | C_v min–max | U_Mass trung bình | Diversity trung bình |', '|---:|---|---:|---:|---:|---:|']
    for candidate,group in all_runs[all_runs.k.isin(selection['promising'])].groupby('k'):
        sensitivity.append(f"| {candidate} | {', '.join(map(str,group.seed.tolist()))} | {group.c_v.mean():.6f} ± {group.c_v.std(ddof=1):.6f} | {group.c_v.min():.6f}–{group.c_v.max():.6f} | {group.u_mass.mean():.6f} | {group.topic_diversity.mean():.6f} |")
    sensitivity += ['', 'Bảng này công khai biến động giữa seed; điểm lựa chọn đã đăng ký dùng metrics seed42 và Stability, không tối ưu trung bình C_v giữa seed. Ba seed chưa đủ chứng minh khác biệt có ý nghĩa thống kê.', '']
    insert_at=lines.index('## 3. Top-10 keywords và số tài liệu/topic')
    lines[insert_at:insert_at]=sensitivity
    eda_summary=pd.read_csv(tables/'eda_source_summary.csv')
    eda_lines=['', '### Kết quả EDA đáng chú ý','', '| Nguồn | Số tài liệu | Năm trong corpus | Token/tài liệu TB | Title ký tự TB | Thiếu Abstract |', '|---|---:|---|---:|---:|---:|']
    for row in eda_summary.itertuples():
        eda_lines.append(f'| {row.source} | {row.documents:,} | {row.year_min}–{row.year_max} | {row.mean_tokens:.2f} | {row.mean_title_chars:.2f} | {row.missing_abstract} |')
    grams=pd.read_csv(tables/'eda_ngrams.csv',keep_default_na=False)
    boiler=grams[(grams.source=='Patent')&(grams.ngram=='independent claim include')]
    boiler_share=float(boiler.document_share.iloc[0])*100 if len(boiler) else None
    if boiler_share is not None:
        eda_lines += ['',f'- Cụm token `independent claim include` xuất hiện trong **{boiler_share:.2f}%** Patent được chọn. Đây là dấu vết văn phong sáng chế có thể chi phối topic, không phải tự động một chủ đề vật liệu. Theo yêu cầu đóng băng, không loại thêm cụm này.']
    pat=eda_summary[eda_summary.source.eq('Patent')].iloc[0]
    pub=eda_summary[eda_summary.source.eq('Publication')].iloc[0]
    eda_lines += [f'- Patent chiếm khoảng {100*pat.documents/eda_summary.documents.sum():.1f}% số tài liệu; độ dài token trung bình của hai nguồn là {pat.mean_tokens:.2f} và {pub.mean_tokens:.2f}. Cần phân biệt topic theo nguồn/văn phong với topic khoa học; bảng tần suất chuẩn hóa và heatmap theo nguồn hỗ trợ kiểm tra.', f'- Mốc năm của corpus đã chọn khác dữ liệu thô: Patent trong corpus bắt đầu năm {pat.year_min}, Publication từ {pub.year_min}. Không diễn giải khoảng trống trước đó là không có sáng chế/công bố; đây có thể là hệ quả chọn corpus bằng heuristic.', '- Patent thiếu DE theo thiết kế nguồn; không suy thành thiếu nội dung NLP. Các tài liệu Publication chỉ có Title vẫn được giữ theo chính sách đã chốt.','']
    # Insert empirical EDA findings before the figure index, retaining the full table.
    insert_at=lines.index('Tất cả biểu đồ PNG 300 DPI, đã có bảng nguồn CSV:')
    lines[insert_at:insert_at]=eda_lines
    for item in figures:
        lines.append(f"| `{item['figure']}` | "+'; '.join(f'`{p}`' for p in item['tables'])+' |')
    lines += ['', 'Các đường dẫn chính:', '',
        f"- Model được chọn: `{selection['selected_model']}`; dictionary: `{selection['dictionary']}`.",
        '- Model/metrics mọi cấu hình: `models/lda/kXX_seedSEED/`; `training_success.json` có fingerprint, parameters, thời gian và checksum model.',
        '- BOW: `data/stage2/corpus_bow.mm` và `.index`; mapping từ: `outputs/tables/stage2/dictionary_tokens.csv`.',
        '- Vector: `data/stage2/lda_topic_vectors.parquet`, `lda_topic_vectors.npy`; CSV đầy đủ: `outputs/tables/stage2/document_topic_distributions.csv`. Một hàng/doc_id, vector K chiều theo topic 0..K-1.',
        '- Assignment: `data/stage2/document_topic_assignments.parquet`; bảng CSV tương ứng có dominant_topic, xác suất, entropy và provenance.',
        '- Topics: `outputs/tables/stage2/topic_summary.csv`, `topic_keywords.csv`, `topic_source_distribution.csv`, `topic_year_trends.csv`, `topic_representative_documents.csv`.',
        '- Metrics: `outputs/tables/stage2/lda_comparison.csv`, `lda_base_metrics.csv`, `lda_all_run_metrics.csv`, `lda_stability_pairs.csv`, `lda_stability_topic_matches.csv`.',
        '- Metadata vận hành: `outputs/reports/stage2/input_manifest.json`, `selected_model.json`, `retrieval_contract.json`, `runtime_validation.json`, `tests.xml`, `artifact_manifest.json`.',
        '- Config/dependencies/log: `configs/stage2.yaml`, `requirements-stage2.txt`, `outputs/logs/stage2.log`.', '',
        '## 5. Kiểm thử và hạn chế','',
        f'- **PASS: {len(cases)} unit/integration tests**, không FAIL hoặc SKIP. Kiểm tra permutation-invariant stability, checksum mô hình, tính lại lựa chọn K, toàn bộ dictionary/BOW, mapping IDs, vector xác suất, suy diễn từ model đã tải, số tài liệu/topic/năm, tệp biểu đồ/bảng nguồn và dữ liệu Stage 1 không đổi.',
        f"- Runtime PASS: vector shape={runtime['vector_shape']}; lỗi tổng xác suất tối đa={runtime['max_probability_sum_error']:.3g}; {runtime['successful_training_runs']} runs, {runtime['stability_pairs']} cặp Stability, {runtime['figures']} biểu đồ.",
        '- Corpus CORE/RELATED do heuristic, còn các trường hợp bị bỏ sót đã biết và quyết định nghiệm thu chưa áp dụng. Kết quả chỉ đại diện corpus tạm chốt, không đại diện mọi nghiên cứu Sodium-ion Battery.',
        '- Coherence/Diversity là mô tả trên corpus huấn luyện, không phải generalization/retrieval test. Stability chỉ phản ánh khởi tạo trên cùng corpus, ba seed và hai K; chưa đo robustness theo bootstrap, thời gian, nguồn hay mở rộng UNCERTAIN.',
        '- Không thay đổi Stage 1 nên giữ cả từ rất hiếm, thuật ngữ chung, khác biệt độ dài/văn phong Patent–Publication và một số tài liệu liên quan nhau. Những yếu tố này có thể chi phối topic; không nên xem topic tự động là nhãn chuyên gia.',
        '- Online LDA giữ thứ tự tài liệu Stage 1 theo các khối nguồn, chưa đối chiếu shuffled/source-balanced training. Stability chỉ so sánh Top-10 từ giữa seed trên cùng thứ tự, không đo độ ổn định assignment từng tài liệu hoặc loại trừ thiên lệch thứ tự/nguồn.',
        '- Mọi biểu đồ topic dùng một mô hình cố định. Năm Patent từ ngày PD công bố sớm nhất, Publication từ PY; năm đầu ít mẫu và năm cuối theo phạm vi export 2023 nên không suy diễn tốc độ phát triển của toàn lĩnh vực hoặc xu hướng hiện tại.',
        '- Keywords/metadata không đưa vào training. Từ/cụm n-gram thống kê không phải nhãn relevance mới. Chưa chạy sensitivity với UNCERTAIN hoặc đo chất lượng Search Engine.', '',
        'Tài liệu API dùng khi triển khai: [Gensim LdaModel](https://radimrehurek.com/gensim/models/ldamodel.html), [CoherenceModel](https://radimrehurek.com/gensim/models/coherencemodel.html), [so sánh topic](https://radimrehurek.com/gensim/auto_examples/howtos/run_compare_lda.html).', '',
    ]
    (ROOT/'outputs/reports/stage2/COMPLETION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(reports/'retrieval_contract.json',dict(model=selection['selected_model'],dictionary=selection['dictionary'],vectors='data/stage2/lda_topic_vectors.parquet',vector_numpy='data/stage2/lda_topic_vectors.npy',doc_id_order='data/stage2/lda_topic_vectors.parquet:doc_id',metadata='data/processed/main_corpus.parquet',vector_columns=[f'topic_{i:02d}' for i in range(k)],dimensions=k,topic_ids_zero_based=True,minimum_probability=0.0,preprocessing_function='src.text_processing.lda_tokens',preprocessing_config='configs/config.yaml',preprocessing_source_sha256=sha256(ROOT/'src/text_processing.py'),note='Stage 4 must use exactly this tokenizer/config, dictionary and model. No Qdrant collection or search implementation is created in Stage 2.'))
    write_json(reports/'stage2_summary.json',dict(status='COMPLETED',selected_k=k,selected_seed=42,main_documents=inputs['documents'],successful_models=len(all_runs),figures=len(figures),tests_passed=len(cases),failures=0,uncertain_pool_documents=6598,selected_metrics={key:float(chosen[key]) for key in ['c_v','u_mass','topic_diversity','stability','selection_score']},frozen_inputs_unchanged=True))
    checkpoint(f'Completed EDA and Gensim LDA: selected K={k}, seed42; {len(all_runs)} successful runs, {len(figures)} 300-DPI figures, {len(cases)} tests PASS. Stage 1 frozen at 13,702 documents; 6,598 UNCERTAIN kept separate. See outputs/reports/stage2/COMPLETION_REPORT.md. Stage-local report exported.',status='COMPLETED')
    artifacts=[]
    for folder in ['models/lda','data/stage2','outputs/reports/stage2','outputs/tables/stage2','outputs/figures/stage2']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and p.name!='artifact_manifest.json':
                artifacts.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)))
    for name in ['run_stage2.py','src/stage2.py','src/stage2_figures.py','src/stage2_complete.py','configs/stage2.yaml','requirements-stage2.txt','outputs/reports/stage2/COMPLETION_REPORT.md','docs/PROGRESS.md','docs/PROJECT_SPEC.md']:
        p=ROOT/name
        artifacts.append(dict(path=name,bytes=p.stat().st_size,sha256=sha256(p)))
    write_json(reports/'artifact_manifest.json',dict(status='COMPLETED',stage=2,artifacts=artifacts))
    print(json.dumps(json.loads((reports/'stage2_summary.json').read_text(encoding='utf-8')),indent=2))


if __name__=='__main__': main()
