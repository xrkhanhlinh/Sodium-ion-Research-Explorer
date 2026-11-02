# Search Radar Pilot — báo cáo thực nghiệm 30/10/2026

Đã triển khai và chạy retrieval ba nhánh, parser, rule/evidence reranking, MiniLM control, pilot pooling và dashboard Streamlit. Production giữ nguyên. Chưa có bằng chứng cải thiện held-out và chưa chọn A7.

## Kết quả lịch sử (AI Silver, không phải nhãn người)

Bộ 703 cặp/20 queries ngày 26/10 giữ nguyên; full-document hybrid P@10=0.5400, nDCG@10=0.7693, MRR@10=0.7767. Không dùng bộ này để chọn trọng số. Q08 không có grade2: nDCG=1 không có nghĩa đủ điều kiện khoa học.

## Kết quả mới thực đo

- A0 exact matrix khớp thứ tự baseline 20/20 query.
- Candidate Recall@100 trong pool: A0=0.925342, A3=1.000000; 19 query có mẫu số. Paired delta=0.074658, CI95% [0.008097, 0.189224]. Không phải full-corpus recall.
- A0/A3 trên cùng 14/20 query đủ nhãn: P=0.542857/0.607143; nDCG=0.741564/0.810289; MRR=0.859524/0.952381.
- Đây là tập giao bị điều kiện hóa theo coverage, có thể thiên lệch; không so với mean trên 20 query hoặc tuyên bố độc lập. Precision_at_10: Δ=0.0643, CI95% [0.0071, 0.1214]; nDCG_at_10: Δ=0.0687, CI95% [0.0166, 0.1342]; MRR_at_10: Δ=0.0929, CI95% [0.0000, 0.2143].
- Có 119 cặp Top10 mới chưa chấm khi hợp mọi cấu hình/depth. Relevance metric của query có bất cứ ID chưa chấm được giữ missing.

## Ablation trên cùng query trong từng cặp với A0

| Cấu hình | n/20 | P@10 | ΔP | ΔnDCG | ΔMRR |
|---|---|---|---|---|---|
| A1 | 9 | 0,6778 | -0,0333 | -0,0046 | 0,0056 |
| A2 | 6 | 0,5500 | 0,0000 | 0,0328 | -0,0833 |
| A3 | 14 | 0,6071 | 0,0643 | 0,0687 | 0,0929 |
| A3 (1:1:2) | 16 | 0,5875 | 0,0437 | 0,0458 | 0,0604 |
| A4 | 7 | 0,6571 | 0,0857 | 0,0792 | 0,0714 |
| A6 N20 | 8 | 0,6250 | 0,0750 | 0,0794 | 0,0833 |
| A6 N30 | 6 | 0,6500 | 0,0833 | 0,0869 | 0,0833 |
| A6 N50 | 5 | 0,6400 | 0,0600 | 0,0744 | 0,1000 |
| A5 max | 16 | 0,6062 | 0,0437 | 0,0545 | 0,0604 |
| A5 mean | 16 | 0,6062 | 0,0312 | 0,0311 | -0,0125 |
| MiniLM N20 | 1 | 1,0000 | 0,1000 | 0,0694 | 0,0000 |

Các dòng có query set khác nhau; không dùng chúng để chọn model. Cột Δ lấy ứng viên trừ A0 trên đúng cùng query. Xem pairwise_complete_metrics.csv để biết ID và paired_bootstrap.csv/pairwise_bootstrap.csv để biết CI. Giao đầy đủ nhãn của toàn bộ methods tại K100 rỗng nên metrics_summary.csv không tạo điểm tổng hợp giả.

## Chunk, latency và memory

Index mới có 32,669 chunks, bao phủ 13,702 tài liệu. Chỉ chạy block cuối trên frozen-prefix cache đã kiểm hash, không huấn luyện và không encode lại toàn encoder. Sai lệch lớn nhất khi gộp lại thành document vector là 8.94e-08; kiểm trên toàn bộ corpus. Thời gian dựng 1639.0 giây; vector index 95.71 MiB. Context, offsets và checkpoint hash lưu trong records.parquet. Max chọn một chunk mạnh nhất; unique mean dùng số token mới để chống đếm overlap. Chưa có nhãn span độc lập để đo độ đúng đoạn bằng chứng.

Phạm vi latency: warm wall-clock search; separate benchmark after chunk build. A0 P50/P95=0.1063/0.1341s; A3=0.4522/0.4959s. RSS peak được lấy mẫu mỗi 20 ms, trong tiến trình mới cho từng phương pháp; gồm model và các index đã nạp, không phải chỉ riêng tensor encoder. P95 A3/A0=3.697 lần, so với mục tiêu cost gate ≤1.5 lần. CSV giữ từng query, P50/P95, RSS, index bytes và initialization. Hai lần lặp mỗi query ở benchmark tiến trình riêng chỉ mô tả query/repetition mix, không phải benchmark tải concurrent.

| Method | P50 seconds | P95 seconds | RSS MiB |
|---|---|---|---|
| A0 | 0,106 | 0,134 | 1238,9 |
| A1 | 0,114 | 0,166 | 1277,5 |
| A2 | 0,436 | 0,510 | 1693,6 |
| A3 | 0,452 | 0,496 | 1567,1 |
| A4 | 0,494 | 0,592 | 1574,2 |
| A5 | 0,753 | 1,145 | 1571,1 |
| A6 | 0,494 | 0,615 | 1584,5 |

Cost gate A3: **FAIL** theo ngưỡng P95/RSS ≤1.5× A0. RSS ratio=1.265; latency ratio=3.697. Quality gate độc lập chưa có nhãn; không tự chuyển cấu hình vào production. `gate_status.json` ghi quyết định.

## Development và held-out

12 query mới được hệ thống đề xuất: sáu dev families và sáu held-out families tách biệt. Đã chạy retrieval/pooling, chưa có expert query review hoặc human qrels. Không báo điểm dev/held-out và không gọi các query này là nhãn người. Biểu mẫu giữ trống reviewer/verifier/time/rationale/condition labels/agreement. Importer kiểm provenance và có thể tính exact agreement/Cohen kappa khi có người chấm thứ hai. Held-out yêu cầu config được khóa sau development gate và nhãn review sau thời điểm khóa.

## Kết quả âm và hạn chế

MiniLM lịch sử đã trượt gate 0.927083→0.850694; control mới không được bật vào production. A4/A6 chỉ là heuristic từ vựng, không phải entailment khoa học. Chưa có neural evidence model mới đã xác minh hoặc nhãn condition/span độc lập; chưa thể kết luận chunk, reranker hay A7 tốt hơn trên held-out. Mục tiêu 0.65/0.82/0.85 là mục tiêu, không phải kết quả đạt được. Giữ pipeline cũ.

Các trường hợp Q04–Q06 (điều kiện cathode/coating), Q08 (không có grade2), Q16 (particle/electrode coating), Q17 (pack thermal/voltage balancing), Q18 (spent SIB recycling) và Q20 (full-cell đồng thời nhiều điều kiện) nằm trong weak_queries.csv; không hard-code query_id vào thuật toán. Thiếu Abstract giữ missing evidence, không đổi metadata hoặc corpus để nâng điểm.

## Kiểm thử và bảo toàn

42/42 tests qua; automated_tests.xml ghi chi tiết. Baseline ranking parity 20/20, protection hashes trước/sau PASS. Không training, sửa checkpoint, encode lại toàn corpus qua full encoder hoặc ghi Qdrant. Chunk index mới chỉ replay block cuối từ prefix đã kiểm hash. App và report giữ riêng lịch sử, AI Silver, thí nghiệm mới, human/dev/held-out chưa có nhãn.

## Chạy lại

Từ gốc workspace trong PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\build_search_radar_chunks.py
.\.venv\Scripts\python.exe -X utf8 scripts\run_search_radar.py --phase historical --cross-encoder-control
.\.venv\Scripts\python.exe -X utf8 scripts\run_search_radar.py --phase pilot
.\.venv\Scripts\python.exe -X utf8 scripts\benchmark_search_radar.py
.\.venv\Scripts\python.exe -X utf8 scripts\report_search_radar.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Chọn **Search Radar Pilot**. Query vector cache/chunk progress có identity/hash và resume. Không xóa annotation_template.csv đã nhập; pool và annotation_template_latest.csv cập nhật riêng với đủ ứng viên mới. Sau khi chuyên gia chấm, chạy evaluate_search_radar_human.py --labels <CSV> --split dev; heldout cần thêm --frozen-config <JSON>. Run tự nhập trên UI dùng --run-dir <thư mục custom> --split unassigned, không coi là dev/held-out. Trước import điền query_review_reference thật, không điền giá trị giả. Nguồn LaTeX có chương mới, radar và ablation; PDF ngày 26/10 vẫn giữ nguyên.
