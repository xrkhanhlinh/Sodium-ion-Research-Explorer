# BatteryBERT full-document: huấn luyện và kết quả ngày24/10/2026

## Phạm vi thực sự

Dữ liệu có13.702 bản ghi Title-Abstract, không có toàn văn bài báo. Pipeline mới đọc
toàn bộ văn bản sẵn có bằng chunk256/overlap64. Không gọi đây là đã train trên toàn
bộ bài báo. API `FullDocumentEncoder.encode_document(title, body)` có thể đọc toàn bộ
body dài do người dùng cung cấp; chất lượng trên toàn văn chưa có đánh giá độc lập.

## Thay đổi và bằng chứng thực thi

Đã tiếp tục train block cuối của checkpoint BatteryBERT fine-tuned bằng gradient
thật, frozen11 block đầu. Cache float32 cũ giữ read-only, bổ sung
2881 chunk còn thiếu và query prefixes không truncation.
Loss dùng batch32 cặp với gradient caching hai tầng, khác batch2/accumulation2 cũ.
Document pooling dùng các token Abstract và trọng số số token mới của mỗi đoạn;
overlap vẫn có mặt làm context. Dropout tắt để replay nhất quán, có warmup/decay,
gradient clipping và distillation nhẹ. Có title context trong các document chunks.

Một lượt train đã xử lý10921 tài liệu, backprop qua
26147 đoạn, bao phủ
3420215 token Abstract duy nhất theo tài liệu.
Validation/test xuất hiện trong optimizer: 0.
Checkpoint được chọn tại step342, đã thấy
10921 bản ghi full-document trong vòng continuation.
Phân biệt coverage của toàn phiên train với coverage tới checkpoint được chọn.

## Kết quả encoder cùng giao thức

| Đo lường | Kết quả |
|---|---:|
| Validation baseline, cùng full-document pooling | 0.906174 |
| Validation checkpoint được chọn | 0.933093 |
| Gain validation | 0.026919 |
| Test baseline, cùng full-document pooling | 0.887839 |
| Test checkpoint được chọn | 0.924704 |
| Chấp nhận trọng số continuation | True |
| L2 thay đổi block cuối được export | 0.312042 |

Đây là paired Title→full document có Title context, có thể được lợi từ title;
không phải relevance search độc lập và không so trực tiếp với proxy Abstract128
cũ để quy mọi gain cho training. Learning rate/checkpoint chọn bằng validation;
test không dùng để chọn. Test là split lịch sử, không phải external test mới.

## Topic và cổng triển khai

Hai geometry(center/source), mỗi geometry ba trọng số exclusivity và hai mức
phrase boost (1 hoặc 1,5) được chọn bằng
validation. KMeans/vocabulary/keywords fit train; seed43 cho validation, seed137
cho cổng cuối. C_v được tính chung reference giữa model trước và mới.

model,c_v,topic_diversity,supported_keyword_fraction
previous,0.5602608195265611,0.8,1.0
full_document,0.5530220621268127,0.77,1.0


- Source Cramér V toàn corpus: 0.303641.
- Top2 share: 24.59%.
- Topic test ARI: 0.688833.
- Topic test NPMI: 0.079191.
- Technical topic promotion gate: True.
- Bật Search mới: True; bật topic mới: True.

Nhãn và review của agent không phải expert ground truth. Nếu cổng topic không
đạt, catalog topic cũ tiếp tục được dùng; Search có quyết định triển khai
riêng. Giữ nguyên kết quả không đạt để đối chứng.

## Kiểm chứng

`model_reload_parity.json`: PASS; so prefix+last-block với forward
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
