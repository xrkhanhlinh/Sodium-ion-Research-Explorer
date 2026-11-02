# Trạng thái dự án — cập nhật 26/10/2026

## Kết quả yêu cầu gán nhãn toàn bộ 703 cặp

Đã đọc Query–Title–Abstract và gán nhãn đủ 703/703 cặp trong 20 batch: 143 nhãn 0, 302 nhãn 1, 258 nhãn 2; 70 trường hợp review_required=TRUE. Không thiếu/trùng ID. Giữ bản pilot cũ, re-review nhãn trống thành 1 với low confidence và review flag.

Đã khóa nhãn rồi tính đúng protocol P@10/MRR@10 threshold2 và nDCG gain 2^label−1 trên shared pool, đủ 20 query × 6 methods × Top10. Full-document hybrid có P@10=0,540; nDCG@10=0,769339; MRR@10=0,776667. Đối chiếu nDCG độc lập bằng sklearn cho 120 nhóm sai lệch tối đa 4,44e−16; P/MRR và ví dụ tính tay đều qua. Nhãn vẫn là AI Silver, không nhập nhãn người hoặc tuning model.

Điểm cần đọc đúng: Q08 không có nhãn 2 trong pool nhưng nDCG vẫn có thể đạt 1 với nhãn 1; Q20 có một bài đáp ứng trực tiếp nhưng full-document hybrid không tìm được vào Top10. Q18 chỉ 1/10 kết quả trực tiếp. CI/sensitivity/Recall trong pool được lưu riêng; không có Recall toàn corpus hoặc độ chính xác của AI judge.

[CSV đủ 703 cặp](../outputs/ai_silver_labels_703_20261026_v1/silver_labels_703.csv) · [Phụ lục PDF 6 trang](../output/pdf/DANH_GIA_SEARCH_SILVER_703_20261026_v2.pdf) · [Bảng 6 methods](../outputs/ai_silver_labels_703_20261026_v1/evaluation/metrics_summary.csv) · [Chi tiết 120 nhóm](../outputs/ai_silver_labels_703_20261026_v1/evaluation/metrics_per_query.csv).

Runner riêng `scripts/resume_silver_703_20261026.py` nhận lại 24 stage hợp lệ, lưu receipts/logs/thời gian và không chạy lại bước hoàn tất. PDF đã render và xem đủ 6 trang; bản draft có colorbar đè heatmap được giữ, v2 đã sửa và QA PASS. Pipeline/model/dữ liệu và metrics human cũ giữ nguyên.

## Snapshot hoàn tất pipeline và pilot trước yêu cầu mở rộng

Đã hoàn tất phần tiếp tục pipeline và pilot Silver Labels 20 cặp. Trạng thái có thể đọc bằng `.venv/Scripts/python.exe scripts/resume_research_revision.py --status`; chạy lại runner sẽ kiểm tra hash và bỏ qua output completed hợp lệ.

- Giữ nguyên 13.702 tài liệu, weights, embeddings và 16 artifact gốc được bảo vệ; không train/encode lại.
- Topic mới: KMeans K=10, n_init=50, source projection fit trên train; gate ARI 0,991808 và NPMI 0,110046 theo protocol geometry. Catalog gồm 10 topic, 100 keywords, audit và bằng chứng Title–Abstract; nhãn diễn giải AI chưa phải expert ground truth.
- Parser công thức v2 hoàn tất 54 shards, bảo toàn span chưa hỗ trợ và provenance; không đổi raw text.
- Đã so sánh model, xuất hình và báo cáo. Loại 5 recipe keyword không đạt ràng buộc; loại reranker vì gate MRR@10 giảm. Search hybrid giữ mặc định.
- 102 tests của phần bị ảnh hưởng PASS; routing smoke 10 kết quả đúng catalog. Không cộng dồn với test của các snapshot cũ.
- Silver pilot: 20/703 cặp, tất cả Q01 theo thứ tự đầu vào; 11 nhãn 2, 4 nhãn 1, 4 nhãn 0, 1 trống, 8 cần review. 683 cặp còn lại chưa xử lý theo phạm vi yêu cầu. Nhãn người được chấp nhận vẫn là 0; không dùng AI pilot làm human relevance.
- Báo cáo PDF v2: 22 trang, 8 hình kết quả, 22 bảng; đã render và kiểm tra toàn bộ trang. Bản cũ được giữ nguyên. QA cuối PASS; một lỗi kiểm tra live state ở lần trước đã được lưu log và sửa validator, không thay dữ liệu khoa học.

[Báo cáo PDF hiện hành](../output/pdf/BAO_CAO_NGHIEN_CUU_CAI_THIEN_SODIUM_ION_20261026_v2.pdf) · [Markdown](BAO_CAO_NGHIEN_CUU_CAI_THIEN_SODIUM_ION_20261026_v2.md) · [CSV pilot 20 cặp](../outputs/ai_silver_labels_20261026_v1/batch_0001_labels_20.csv) · [Nhật ký và quyết định](RESEARCH_REVISION_WORKLOG_20261025.md) · [QA cuối](../outputs/research_revision_20261025_v1/report_v2/validation_final.json).

## Snapshot ngày 24/10/2026

- Đã tiếp tục train thật BatteryBERT từ checkpoint cũ: 342 updates, toàn bộ 10.921 tài liệu train, 26.147 chunks và 3.420.215 token Abstract. Batch contrastive thực 32 cặp, gradient caching; 11 block đầu giữ nguyên, cập nhật block cuối.
- Train/inference đọc toàn bộ Title–Abstract qua các cửa sổ 256/64. Dữ liệu chưa có toàn văn bài báo; không tuyên bố đã train trên full article.
- Proxy cùng giao thức: validation 0,906174 → 0,933093; test 0,887839 → 0,924704. Document có Title context; đây chưa phải relevance Search độc lập.
- Topic chính mới: `BatteryBERT full-document · trained`, geometry học từ train + KMeans10. Qua cổng kỹ thuật đặt trước; top2 toàn corpus 24,59%. Bản 23/10 vẫn có ARI/diversity và coherence tốt hơn ở một số phép đo; giữ cả hai để đối chứng.
- Search mặc định mới: TF-IDF + BatteryBERT full-document/RRF60. Đã benchmark 20 truy vấn × 6 phương pháp, 1.200 dòng Top10, pool 703 cặp. **PENDING_HUMAN_ANNOTATION**, 0 nhãn người; chưa kết luận model thắng relevance.
- Đã review 60 đại diện topic; xem 200 title Search hybrid và đọc sâu 6 abstract Top1. Có lỗi gán topic và truy vấn nhiều ràng buộc; review agent không phải nhãn chuyên gia.
- Kiểm thử cuối: **112 trường hợp riêng biệt đã qua, 0 failures/errors/skips**, khử trùng các lượt chạy lại bằng `verification_summary.json`. Web 4/4 qua; HTTP smoke qua. Giữ XML timeout web ban đầu, không dùng kết quả tests làm bằng chứng relevance.
- Run và bằng chứng: `outputs/batterybert_full_document_20261024_v1/`.

[Báo cáo toàn pipeline mới](BAO_CAO_PIPELINE_BATTERYBERT_FULLDOC_20261024.md) · [PDF để in](../output/pdf/BAO_CAO_PIPELINE_BATTERYBERT_FULLDOC_20261024.pdf) · [Báo cáo training mới](BATTERYBERT_FULL_DOCUMENT_RESULTS_20261024.md) · [10 topic của cả 6 pipeline](TOP10_TOPICS_ALL_MODELS_20261024.md) · [Hướng dẫn chạy](RUN_DEMO.md).

## Snapshot lịch sử 18/10/2026

Các số dưới đây thuộc đợt cleanup trước; không phải cấu hình đang chạy hoặc tổng kiểm thử của bản 24/10.

- Search hiện tại: pretrained + lexical/dense hybrid + MiniLM reranker, cùng corpus 13.702 tài liệu.
- Đã chạy: 24 queries × 10 cấu hình; evaluator 14 methods / 322 tổ hợp / pool 942 cặp.
- Relevance: **PENDING_INDEPENDENT_LABELS**.
- Fine-tuning cũ và BERTopic baseline/chunked giữ làm đối chứng.
- Cleanup hoàn tất: **185 tests pass, 0 failures/errors/skips**. Bản trước sửa đã lưu, code/report cũ gỡ khỏi luồng hiện tại. Bằng chứng tại outputs/project_cleanup_20261018/.

[Báo cáo chính](FINAL_RESEARCH_REPORT.md) · [Hướng dẫn](RUN_DEMO.md). Log một Stage riêng không mô tả trạng thái toàn hệ thống.
