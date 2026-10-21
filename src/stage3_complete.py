"""Finalize Stage 3 only after real training, exports and passing tests."""
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import importlib.metadata
import numpy as np
import pandas as pd
from .stage3_common import ROOT,REPORT,TABLE,DATA,FIGURE,config,read_json,write_json,csv,digest,progress,verify_frozen

def main():
    cfg=config();test_file=REPORT/'tests.xml'
    assert test_file.exists(),'Run Stage 3 tests before completion'
    for path in [REPORT/'figure_manifest.json',REPORT/'topic_success.json',ROOT/'models/bert_finetuned/training_success.json']:
        assert test_file.stat().st_mtime>=path.stat().st_mtime,'Tests predate latest successful outputs'
    tree=ET.parse(test_file);cases=tree.findall('.//testcase')
    assert cases and not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
    frozen=verify_frozen()
    trained=read_json(ROOT/'models/bert_finetuned/training_success.json')
    assert trained['real_gradient_updates'] and trained['selected_weight_delta_l2']>0
    metrics=read_json(REPORT/'bertopic_metrics.json');splits=read_json(REPORT/'split_audit.json')
    figures=read_json(REPORT/'figure_manifest.json');stability=read_json(REPORT/'bertopic_stability.json')
    before=read_json(REPORT/'pretrained_test_metrics.json');after=read_json(REPORT/'finetuned_test_metrics.json')
    comparison=pd.read_csv(TABLE/'lda_vs_bertopic.csv');topics=pd.read_csv(TABLE/'bertopic_topic_summary.csv',keep_default_na=False)
    retrieval=pd.read_csv(TABLE/'bert_before_after.csv');trunc=pd.read_csv(TABLE/'tokenizer_truncation.csv')
    history=pd.read_csv(TABLE/'training_loss.csv');validation=pd.read_csv(TABLE/'validation_metrics.csv')
    environment=read_json(REPORT/'neural_environment.json')
    physical=read_json(REPORT/'hardware_registry.json') if (REPORT/'hardware_registry.json').exists() else {}
    embeddings=read_json(REPORT/'embedding_manifest.json')
    log_lines=(ROOT/'outputs/logs/stage3.log').read_text(encoding='utf-8').splitlines()
    starts=[datetime.fromisoformat(line.split(' ',1)[0]) for line in log_lines if 'Real BGE CPU fine-tuning started/resumed:' in line]
    ends=[datetime.fromisoformat(line.split(' ',1)[0]) for line in log_lines if 'Real BGE fine-tuning complete:' in line]
    timings=dict(retained_optimizer_seconds=trained['training_seconds'],retained_validation_seconds=float(validation.seconds.sum()),wall_clock_start=min(starts).isoformat() if starts else None,wall_clock_end=max(ends).isoformat() if ends else None,wall_clock_seconds=(max(ends)-min(starts)).total_seconds() if starts and ends else None,wall_clock_includes='validation, cache creation, interruptions/recovery/repeated unsaved steps and model export; excludes download/initial benchmark')
    write_json(REPORT/'training_timing_summary.json',timings)
    delta=after['group_mrr_at_10']-before['group_mrr_at_10']
    direction='tăng' if delta>0 else 'giảm' if delta<0 else 'không đổi'
    lines=['# STAGE 3 COMPLETION REPORT — BERT Fine-tuning & BERTopic','', '**Trạng thái: COMPLETED — chỉ Stage 3 (P5–P6).**','',
      '## 1. Dữ liệu, chia tập và phần cứng','',
      '- Corpus giữ nguyên **13.702 tài liệu**: 5.933 Patent và 7.769 Publication. Có **13.650** cặp Title–Abstract; 52 tài liệu thiếu Abstract chỉ được embedding và gán topic, không tạo Abstract giả.',
      f"- Chia theo `split_group_id`: **{splits['pair_counts']['train']:,} train**, **{splits['pair_counts']['validation']:,} validation**, **{splits['pair_counts']['test']:,} test**; không giao nhóm. Chi tiết theo nguồn và nhóm trong `outputs/tables/stage3/split_summary.csv`, từng doc_id trong `split_assignments.csv`.",
      '- Split groups là các liên kết exact/candidate family/near-duplicate đã có ở Stage 1, không phải họ sáng chế đã được chuyên gia xác minh và không bảo đảm tìm đủ mọi quan hệ ngữ nghĩa.',
      '- Audit bổ sung: không có Title hoặc Abstract trùng nguyên văn qua các split sau casefold và chuẩn hóa khoảng trắng; chi tiết `exact_text_leakage_audit.json`. Không coi đây là bảo đảm không có semantic leakage.',
      '- Không có Title/Abstract với chuỗi token trùng nhau qua các split ở giới hạn128 token và có áp dụng query instruction cho Title; kiểm tra `token_sequence_leakage_audit.json`.',
      f"- Máy: 8 GB RAM, 4 nhân vật lý/8 luồng; RAM khả dụng đầu lượt neural khoảng {environment['ram']['available']/1024**3:.2f} GiB. PyTorch `{environment['torch_version']}`, CUDA khả dụng={environment['cuda_available']}; baseline chạy CPU với {environment['threads']} threads, training/inference dùng cấu hình sau benchmark ở mục2. Tên thiết bị được đọc từ registry vì truy vấn CIM không được cấp quyền.", '',
      f"- Registry Windows (read-only) bổ sung: CPU `{physical.get('cpu_name','not available')}`; display adapters: {', '.join(physical.get('display_adapters',[])) or 'not available'}. Chi tiết `hardware_registry.json`.", '',
      '## 2. Fine-tuning BGE thật','',
      f"- Pretrained: `BAAI/bge-base-en-v1.5`, revision `{read_json(REPORT/'pretrained_model.json')['revision']}`. Sentence-Transformers/PyTorch, CLS pooling, embeddings L2-normalized **768 chiều**.",
      f"- **{cfg['training']['epochs']} epoch đầy đủ**, **{trained['planned_steps']:,} optimizer steps**, **{trained['rows_seen']:,} cặp train đã dùng**. Đóng băng embeddings và 11 block đầu; cập nhật block encoder cuối: **{trained['trainable_parameters']:,}/{trained['total_parameters']:,} tham số** ({100*trained['trainable_parameters']/trained['total_parameters']:.2f}%). Đây là partial fine-tuning BGE, không phải huấn luyện toàn bộ network.",
      '- Loss: contrastive cross entropy đối xứng Title→Abstract và Abstract→Title; cosine/temperature=0,05. Batch=8, ghép batch cuối đơn lẻ vào batch trước. AdamW lr=2e-5, weight_decay=0,01; 10% warmup rồi linear decay; clip gradient norm=1. Frozen dropout tắt; dropout của block cuối bật trong training.',
      '- Để giảm CPU cho các lượt validation lặp lại, cache float32 đầu ra11 block frozen trên đĩa và tính mới block cuối cho từng checkpoint. Cache được đối chiếu với full BGE trên văn bản mẫu (atol2e-5); không cache lớp đang học hoặc lượng tử hóa trọng số.',
      f"- Negative trong batch: mask cặp cùng split_group_id hoặc title word-set Jaccard≥0,8; đã mask **{trained['masked_pairs']:,}** ô off-diagonal qua toàn epoch. Chưa loại được mọi false negative ngữ nghĩa trong miền Sodium-ion hẹp.",
      f"- Checkpoint chọn bằng **validation group MRR@10**, đánh giá mỗi250 bước và cuối epoch; checkpoint đã fine-tune tốt nhất là **step {trained['best_step']}**, validation MRR@10={trained['best_score']:.6f}. Không dùng test chọn checkpoint; giữ checkpoint fine-tuned kể cả khi thấp hơn pretrained.",
      f"- Loss thực tế: bước đầu={history.loss.iloc[0]:.6f}, bước cuối={history.loss.iloc[-1]:.6f}; trung bình100 bước đầu={history.loss.head(100).mean():.6f}, 100 bước cuối={history.loss.tail(100).mean():.6f}. Loss từng batch có độ khó khác nhau, không diễn giải chỉ hai giá trị đầu/cuối như bằng chứng hội tụ.",
      f"- Thời gian forward/backward/update của quỹ đạo được giữ: **{trained['training_seconds']:.1f}s** ({trained['training_seconds']/3600:.2f} giờ); validation được giữ={validation.seconds.sum():.1f}s. Wall-clock từ bắt đầu lượt train đầu đến xuất model, bao gồm validation/tạo cache/lỗi/phục hồi/chạy lại: **{timings['wall_clock_seconds']}s**. Không gồm download/benchmark ban đầu hoặc encoding sau huấn luyện. Chi tiết `training_timing_summary.json`, log timestamp tại `outputs/logs/stage3.log` và `stage3_execution.log`.",
      f"- CPU thread count: baseline={environment['threads']}; training/inference sau benchmark={trained['torch_threads']}. Chọn số threads bằng benchmark trên32 văn bản train, không dựa vào test; kết quả tốc độ và sai số số học được lưu trong `cpu_benchmark.json`.",
      f"- Bằng chứng trọng số thay đổi: L2 delta block được chọn so với pretrained = **{trained['selected_weight_delta_l2']:.6f} > 0**; SHA frozen tensors trước/sau trùng nhau. Model đã lưu đầy đủ: `models/bert_finetuned/`; model SHA-256=`{trained['model_sha256']}`. Resume optimizer/scheduler/RNG: `resume.pt`; trainable weights được chọn: `best_trainable.pt`; thông số/checksum: `training_success.json`.", '',
      '### Đánh giá trước/sau trên cùng dữ liệu giữ lại','',
      'Query là Title có instruction BGE; candidate là Abstract không thêm instruction trong chính split đó. Mọi candidate cùng split_group_id với query được tính liên quan. Xếp hạng cosine, xử lý hòa điểm theo hướng bi quan. MRR@10 và Success@k dưới đây là **proxy Title–Abstract**, chưa phải chất lượng SE1/SE2/SE3 hoặc nhãn truy vấn chuyên gia. Success@k đo tỷ lệ query có ít nhất một tài liệu cùng nhóm ở Top-k; các cột group_recall_at_k lịch sử là alias cũ, không phải document Recall@k.', '',
      '| Split | Model | Số query/candidates | Group MRR@10 | Success@1 | Success@5 | Success@10 | Exact MRR@10 |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in retrieval.itertuples():
        lines.append(f'| {row.split} | {row.phase} | {row.queries} | {row.group_mrr_at_10:.6f} | {row.group_recall_at_1:.6f} | {row.group_recall_at_5:.6f} | {row.group_recall_at_10:.6f} | {row.exact_mrr_at_10:.6f} |')
    lines += ['',f"Trên test độc lập, group MRR@10 **{direction} {abs(delta):.6f}** ({before['group_mrr_at_10']:.6f} → {after['group_mrr_at_10']:.6f}). Không tuyên bố cải thiện có ý nghĩa thống kê; chưa có nhiều seed fine-tuning hoặc confidence interval. Test chỉ dùng báo cáo sau chọn checkpoint.", '',
      '### Giới hạn độ dài','',
      'Giới hạn128 token để phù hợp CPU/RAM, áp dụng giống nhau cho baseline, fine-tuning và embedding toàn corpus. Không ghi đè văn bản Stage 1. Title bao gồm query instruction khi đo; dòng Abstract gồm cả52 chuỗi trống.', '',
      '| Trường | Số tài liệu | Vượt128 token | Tỷ lệ | Token tối đa |','|---|---:|---:|---:|---:|']
    for row in trunc.itertuples():lines.append(f'| {row.field} | {row.documents} | {row.truncated_at_128} | {100*row.truncation_share:.2f}% | {row.max_tokens} |')
    lines += ['', 'Cắt ngắn có thể bỏ nội dung hóa học hoặc phần cuối Abstract, nhất là Patent. Đây là giới hạn thực nghiệm ban đầu; chưa có chunking/long-document hoặc ablation độ dài.', '',
      '## 3. BERTopic — số topic thực tế','',
      '- Embeddings đầu vào chỉ từ checkpoint BGE đã fine-tune nêu trên; **13.702 × 768**, finite và L2-normalized, có doc_id theo đúng thứ tự corpus.',
      '- BERTopic fit toàn corpus, bao gồm tài liệu thuộc các split BGE, sau khi đánh giá test BGE đã được lưu. Đây là phân tích topic trên corpus quan sát được; coherence và phân bố topic là mô tả in-sample. Trọng số BGE không được cập nhật thêm trong giai đoạn BERTopic.',
      '- UMAP cosine: neighbors15, components5, min_dist0, seed42; HDBSCAN: min_cluster_size30, min_samples10, EOM; c-TF-IDF giảm ảnh hưởng từ phổ biến. **nr_topics=None**, không ép K=20.',
      f"- Phát hiện **{metrics['topics']} topics**; **{metrics['outliers']:,} outliers** (topic−1), tỷ lệ **{metrics['outlier_ratio']:.2%}**. Không ép phân bổ lại outliers. Số topic phụ thuộc các hyperparameters density đã công bố, không phải số chủ đề khoa học có ground truth.",
      '- Biểu diễn c-TF-IDF dùng nguyên token Stage 1 LDA, vocabulary giữ nguyên. Coherence dùng cùng full corpus/token reference, Top-10/window110 như LDA; loại riêng topic−1 khỏi coherence và diversity.', '',
      '- Dùng subclass importable `FrozenTokensBERTopic` (`src/stage3_bertopic_model.py`) để bỏ bước strip ký tự nội bộ của BERTopic, bảo toàn nguyên `sodium_ion`, `hard_carbon`, dấu gạch, điện tích và case từ Stage 1. Full pickle cần module này cùng phiên bản dependencies. Kiểm tra toàn bộ tần suất token/doc và c-TF-IDF hữu hạn; không thay Inf/NaN bằng số giả.', '',
      '- Embedding clustering quan sát tối đa128 WordPiece token/văn bản; c-TF-IDF và coherence vẫn dùng toàn bộ token Stage 1. Vì vậy keywords có thể phản ánh phần nội dung ngoài cửa sổ của encoder; coherence không đo trực tiếp thông tin bị mất do cắt ngắn.', '',
      '| Topic | Số tài liệu | Tỷ lệ | Top-10 keywords |','|---:|---:|---:|---|']
    for row in topics.itertuples():lines.append(f'| {row.topic_id} | {row.documents:,} | {row.document_share:.2%} | {row.top10_keywords} |')
    lines += ['', 'Tài liệu đại diện Top-3/topic (không chọn cho−1) có đầy đủ doc_id, nguồn, Title, Abstract, membership strength trong `bertopic_representative_documents.csv`. Đại diện được xếp theo HDBSCAN membership strength trong nhóm đã gán, khác với đại diện c-TF-IDF nội bộ của BERTopic; strength không phải xác suất relevance được hiệu chuẩn hoặc vector phân phối topic.', '',
      '## 4. So sánh ban đầu với LDA','',
      '| Model | Topics | C_v | U_Mass | Top-10 Diversity | Outlier ratio |','|---|---:|---:|---:|---:|---:|']
    for row in comparison.itertuples():lines.append(f'| {row.model} | {row.topics} | {row.c_v:.6f} | {row.u_mass:.6f} | {row.topic_diversity:.6f} | {row.outlier_ratio:.2%} |')
    lda_row=comparison.iloc[0];bt_row=comparison.iloc[1]
    lines += ['',f"So với LDA, BERTopic có chênh lệch C_v={bt_row.c_v-lda_row.c_v:+.6f} và Diversity={bt_row.topic_diversity-lda_row.topic_diversity:+.6f}; kết quả này đi kèm số topic {int(bt_row.topics)} so với20 và {metrics['outlier_ratio']:.2%} tài liệu ngoài topic. Đây là khác biệt mô tả của hai cấu hình thực nghiệm, không phải bằng chứng thống kê về ưu thế toàn diện."]
    lines += ['', 'Coherence/Diversity chung reference giúp đối chiếu mô tả; khác số topic, mục tiêu tối ưu, soft-vs-hard assignment và việc BERTopic để outliers ngoài topic vẫn làm so sánh có giới hạn. LDA gán argmax cho mọi tài liệu; điều đó không chứng minh LDA có chất lượng tốt hơn vì không có outliers. Chưa có đánh giá chuyên gia hay retrieval để kết luận mô hình nào tốt hơn cho Search Engine.', '',
      '### Stability của BERTopic','']
    for item in stability:
        lines += [f"- Seed42 so với seed{item['seed']}: {item['base_topics']} và {item['other_topics']} topics; matched={item['matched_topics']}. Hungarian Top-10 Jaccard (unmatched=0)={item['hungarian_top10_jaccard_unmatched_zero']:.6f}; matched-only={item['matched_topics_jaccard_mean']:.6f}; ARI gồm−1={item['ari_all_labels_including_outlier']:.6f}; common inliers={item['common_inlier_documents']:,}, ARI common inliers={item['ari_common_inliers']}."]
    lines += ['', 'Đây chỉ là hai lượt UMAP trên cùng embeddings/corpus; không đánh giá seed fine-tuning hoặc bootstrap. LDA Stage 2 dùng ba seed khởi tạo và ba cặp matched-keyword Jaccard với cùng K; **không so trực tiếp hai trị số stability như cùng một thước đo thực nghiệm**.', '',
      '## 5. Tệp và biểu đồ','',
      '- `models/bert_finetuned/`: checkpoint SentenceTransformer đầy đủ, tokenizer, pooling, best/resume state và bằng chứng update trọng số.',
      '- `models/bertopic/seed42/`: model.pickle đầy đủ UMAP/HDBSCAN/c-TF-IDF, compact safetensors, assignments và UMAP5D; `seed123/` cho stability. Khi transform phải cung cấp embeddings từ checkpoint fine-tuned; compact model không thay thế toàn bộ clustering pipeline.',
      '- `data/stage3/embeddings.parquet`: doc_id, source, emb_000..emb_767; `embeddings.npy` float32 với mapping cùng thứ tự. Bảng split và contrastive_pairs giữ provenance từ Stage 1. Các cache evaluation có fingerprint và progress để tiếp tục.',
      '- `data/stage3/bertopic_assignments.parquet`: doc_id/source/year/group/topic_id/membership_strength; `umap_2d.npy` chỉ để trực quan hóa.',
      '- `outputs/tables/stage3/`: split summary, losses/validation, before-after metrics/ranks, tokenizer truncation, topics/keywords/representatives, source/year, similarity/UMAP, LDA comparison và stability matches.',
      '- `outputs/reports/stage3/`: input/embedding/training/environment/split manifests, metrics, stability, tests, runtime validation, inference contract và artifact checksums. Dependencies: `requirements-stage3.txt`; config: `configs/stage3.yaml`; entry point: `run_stage3.py`.', '',
      '| Biểu đồ300 DPI | Bảng nguồn |','|---|---|']
    for item in figures:lines.append(f"| `{item['figure']}` | "+'; '.join(f'`{p}`' for p in item['tables'])+' |')
    lines += ['', 'Keyword figure hiển thị12 topic lớn nhất để đọc rõ; Top-10 của mọi topic nằm trong CSV/bảng ở trên. Similarity heatmap dùng centroid embeddings ngữ nghĩa, không phải c-TF-IDF. UMAP2D fit riêng để quan sát, không phải không gian5D dùng clustering; khoảng cách trên hình không phải bằng chứng relevance.', '',
      '## 6. Kiểm thử, hạn chế và bước dừng','',
      f'- **PASS {len(cases)} unit/integration tests**, không fail/error/skip. Đã kiểm tra split/group, metrics tính lại, masking, đầy đủ một epoch, validation-only selection, cập nhật trọng số thật và tensor frozen giữ nguyên, reload model và đối chiếu embedding mẫu, IDs/dimensions/norm, BERTopic pipeline/assignments/topics/outliers và DPI/bảng nguồn.',
      f'- **PASS {frozen} tệp Stage 1/2 được bảo vệ không đổi**; không làm sạch lại corpus, đổi relevance hoặc huấn luyện lại LDA. Manifest checksum các artifact mới được lưu riêng.',
      '- Sự cố đã khắc phục: API pooling của Sentence-Transformers6, UTF-8 console Windows và ánh xạ tên metric validation. Sự cố metric sau bước250 được phục hồi từ checkpoint200;50 bước chưa lưu đã chạy lại. Backup checkpoint gốc và bản ghi migration xác nhận giữ nguyên trọng số/optimizer/RNG tại điểm phục hồi. Thời gian tính toán của quỹ đạo được giữ và wall-clock được báo riêng; wall-clock đã gồm lỗi, phục hồi, chạy lại và tạo cache. Xem implementation_notes.md và execution log để truy vết.',
      '- P6 gặp chặn Windows named-pipe trong Joblib ThreadPool của PyNNDescent trước khi có model clustering thành công. Đã dùng Joblib sequential backend cho fit/transform để tránh IPC; tái sử dụng toàn bộ BGE checkpoint/embeddings, không đổi thuật toán hoặc dữ liệu. Môi trường sandbox có thể cần cùng context khi nạp pickle và transform.',
      '- Một lượt P6 khác dừng sau UMAP/HDBSCAN vì preprocessing tiếng Anh nội bộ làm mất token có dấu, dẫn tới IDF vô hạn trong vocabulary cố định. Đã sửa bằng bảo toàn token Stage 1 như trên, thêm regression tests và chạy lại P6; không dùng kết quả lỗi để báo metrics.',
      '- Corpus do heuristic, còn các trường hợp UNCERTAIN/ngoài main cần chuyên gia quyết định. Nhóm6.598 UNCERTAIN đã giữ riêng từ Stage 2 không được dùng để fine-tune/fit BERTopic hoặc chọn checkpoint.',
      '- Một seed fine-tuning, một epoch, chỉ block cuối và128 token là cấu hình hạn chế bởi tài nguyên. Không tối ưu toàn mạng, không loại được mọi false negative, không đảm bảo quan hệ giữa nhóm hoàn toàn độc lập. Corpus-only proxy metrics không chứng minh chất lượng tìm kiếm ngoài miền.',
      '- Source/style/length imbalance, patent boilerplate, công thức hóa học/tokenization, bias thời gian và phạm vi export đến2023 vẫn còn. Chủ đề tự động và đại diện cần chuyên gia xem xét trước diễn giải khoa học.', '',
      '- Tokenizer gốc của BGE có `do_lower_case=true` và dùng WordPiece; dù văn bản Stage 1 vẫn giữ nguyên công thức/ký hiệu, đầu vào neural có thể mất phân biệt case và tách nhỏ công thức. Encoder được yêu cầu không phải bộ phân tích hóa học chuyên biệt.', '',
      'API references: [BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5), [Sentence-Transformers](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html), [BERTopic](https://maartengr.github.io/BERTopic/api/bertopic.html), [serialization](https://maartengr.github.io/BERTopic/getting_started/serialization/serialization.html).', '',
      'Báo cáo này mô tả thực nghiệm fine-tuning và BERTopic baseline. Hệ thống hiện tại được tổng hợp tại docs/FINAL_RESEARCH_REPORT.md.']
    text='\n'.join(lines)+'\n'
    (ROOT/'outputs/reports/stage3/COMPLETION_REPORT.md').write_text(text,encoding='utf-8')
    write_json(REPORT/'runtime_validation.json',dict(status='PASS',frozen_files=frozen,embedding_shape=embeddings['shape'],real_weight_updates=True,weight_delta_l2=trained['selected_weight_delta_l2'],group_overlap=0,topic_count=metrics['topics'],outliers=metrics['outliers'],figures=len(figures),tests_passed=len(cases),stage2_lda_retrained=False))
    write_json(REPORT/'inference_contract.json',dict(embedding_model='models/bert_finetuned',model_sha256=trained['model_sha256'],dimensions=768,max_seq_length=cfg['max_seq_length'],normalize_embeddings=True,query_prefix=cfg['query_prefix'],document_prefix='',preprocessing='Use Stage 1 bert_texts.text unchanged; same Stage 1 normalization for future queries, no LDA tokens for dense encoding',embeddings='data/stage3/embeddings.parquet',topic_model='models/bertopic/seed42/model.pickle',topic_transform='Supply tokenized Stage 1 LDA representation plus fine-tuned BGE embeddings; never auto-download a different encoder',topic_membership='HDBSCAN strength, not probability distribution over topics',stop_stage=3))
    contract=read_json(REPORT/'inference_contract.json')
    contract.update(bertopic_class='src.stage3_bertopic_model.FrozenTokensBERTopic',umap_runtime_context='joblib.parallel_backend("sequential", n_jobs=1) in the managed Windows sandbox',representation='Join exact Stage 1 LDA tokens; preserve underscores/hyphens/charges/case; no second cleaning')
    write_json(REPORT/'inference_contract.json',contract)
    write_json(REPORT/'summary.json',dict(status='COMPLETED',documents=13702,training_pairs=13650,train_rows_seen=trained['rows_seen'],epochs=1,selected_step=trained['best_step'],real_weight_updates=True,pretrained_test=before,finetuned_test=after,bertopic=metrics,figures=len(figures),tests_passed=len(cases),frozen_files_unchanged=frozen))
    progress(f'Stage 3 completed: real BGE final-block fine-tuning, selected step {trained["best_step"]}; fixed held-out test group MRR@10 {before["group_mrr_at_10"]:.6f} → {after["group_mrr_at_10"]:.6f}; 13,702×768 embeddings. BERTopic {metrics["topics"]} topics, C_v={metrics["c_v"]:.6f}, {metrics["outlier_ratio"]:.2%} outliers; {len(cases)} tests PASS. See outputs/reports/stage3/COMPLETION_REPORT.md. Stage-local report exported.',status='COMPLETED')
    files=[]
    for folder in ['models/bert_finetuned','models/bertopic','data/stage3','outputs/tables/stage3','outputs/figures/stage3','outputs/reports/stage3']:
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.name!='artifact_manifest.json']
    files += [ROOT/n for n in ['run_stage3.py','scripts/stage3_smoke.py','configs/stage3.yaml','requirements-stage3.txt','outputs/reports/stage3/COMPLETION_REPORT.md','docs/PROGRESS.md','docs/EXPERIMENT_LOG.md','docs/PROJECT_SPEC.md']]
    files += list((ROOT/'src').glob('stage3*.py'))+list((ROOT/'tests').glob('test_stage3*.py'))
    write_json(REPORT/'artifact_manifest.json',dict(stage=3,status='COMPLETED',files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))]))
    print('STAGE 3 COMPLETED; report outputs/reports/stage3/COMPLETION_REPORT.md',flush=True)

if __name__=='__main__':main()
