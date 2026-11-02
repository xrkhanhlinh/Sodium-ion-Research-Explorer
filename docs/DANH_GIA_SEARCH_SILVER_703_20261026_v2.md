## Đánh giá Search bằng 703 Silver Labels

Sodium-ion Research Explorer · 26/10/2026 · Phụ lục kết quả mới cho báo cáo nghiên cứu. Nhãn do AI đọc Query–Title–Abstract; chưa được chuyên gia hoặc người dùng xác nhận.

Đã xử lý đủ 703/703 cặp, 20 truy vấn, không thiếu hoặc trùng ID. Phân bố: 143 nhãn 0, 302 nhãn 1, 258 nhãn 2. Có 70 trường hợp cần review; 28 bản ghi thiếu Abstract. Tất cả đều có nhãn số theo bằng chứng hiện có, nhưng các trường hợp review vẫn là kết luận tạm thời.

| Phương pháp | P@10 | nDCG@10 | MRR@10 |
|---|---|---|---|
| TF-IDF | 0,4500 | 0,6719 | 0,6917 |
| BatteryBERT semantic | 0,2650 | 0,4457 | 0,5005 |
| BatteryBERT hybrid | 0,5050 | 0,7499 | 0,7238 |
| BGE semantic | 0,5300 | 0,7399 | 0,7383 |
| Full-document semantic | 0,4500 | 0,6786 | 0,7250 |
| Full-document hybrid | 0,5400 | 0,7693 | 0,7767 |

![Hình 1. Trung bình đều trên cùng 20 truy vấn; nhãn 2 là relevance trực tiếp.](../outputs/ai_silver_labels_703_20261026_v1/evaluation/figures_v2/method_comparison.png)
Hình 1. Trung bình đều trên cùng 20 truy vấn; nhãn 2 là relevance trực tiếp.

Full-document hybrid cao nhất về trung bình trên cả ba chỉ số chính trong lần chấm AI này. So với TF-IDF, mức tăng là +0,090 P@10; +0,0975 nDCG@10; +0,085 MRR@10. Chưa đủ căn cứ khẳng định thắng các phương pháp gần nhất trên người dùng mới.

## Giao thức, công thức và kiểm chứng

Nhãn được khóa trước khi ghép với thứ hạng: đọc đầy đủ Query, Title và Abstract từng cặp, không dùng cosine/RRF/rank/model để quyết định nhãn. 20 batch được checkpoint riêng. Giữ nguyên bản pilot; 19 nhãn đã quyết định được dùng lại, 1 nhãn trống được re-review thành nhãn 1, low confidence và cần review vì vai trò cathode/negative electrode mâu thuẫn.

| Chỉ số | Cách tính và ý nghĩa |
|---|---|
| P@10 | Số nhãn 2 trong Top 10 / 10. Đo tỷ lệ tài liệu đáp ứng trực tiếp; 0,540 tương đương trung bình 5,4 bài trực tiếp / 10. |
| MRR@10 | 1/r của nhãn 2 đầu tiên; bằng 0 nếu Top 10 không có nhãn 2. Đo mức xuất hiện sớm của kết quả trực tiếp. |
| nDCG@10 | DCG = tổng (2^label - 1)/log2(rank+1). Chia cho DCG lý tưởng của Top 10 sắp từ toàn pool chung của query. Nhãn 1 có gain=1; nhãn 2 có gain=3. |
| Recall_pool@10 | Phụ lục: số nhãn 2 tìm được / tổng nhãn 2 trong pool của query. Không phải Recall toàn corpus; Q08 không có nhãn 2 nên Recall undefined. |

Đánh giá đủ 120 nhóm query–method và 1.200 vị trí Top 10. Mỗi method có đúng 10 rank liên tục cho mỗi query; mọi doc_id đều thuộc pool và text khớp nguồn. Trung bình macro giữ đủ 20 truy vấn cho cả 6 methods, không bỏ query khó để nâng điểm.

Dùng lại hàm metrics_at_10 trong src/final_system_evaluation.py qua adapter Silver riêng, không đổi yêu cầu HUMAN_INDEPENDENT của evaluator chính. Kiểm tra nDCG độc lập bằng sklearn trên 120 nhóm: sai lệch tối đa 4,44×10^-16. P/MRR được đối chiếu trực tiếp; thêm hai ví dụ tính tay, gồm zero relevance. Hash của corpus pool, rankings, human labels và metrics cũ không đổi.

nDCG bằng 0 nếu IDCG=0; không đổi nhãn thiếu thành 0. Không tính accuracy/F1 của nhãn AI vì chưa có nhãn chuẩn để đối chiếu. Không tính kappa vì chỉ có một lượt chấm AI, không có người chấm thứ hai độc lập.

## Kết quả từng truy vấn của Full-document hybrid

| Query | P@10 | nDCG@10 | MRR@10 | Nhãn 2 trong pool |
|---|---|---|---|---|
| Q01 | 0,9000 | 0,8533 | 0,5000 | 25 |
| Q02 | 0,6000 | 0,7938 | 1,0000 | 14 |
| Q03 | 0,8000 | 0,8845 | 1,0000 | 17 |
| Q04 | 0,3000 | 0,6168 | 1,0000 | 13 |
| Q05 | 0,3000 | 0,5496 | 1,0000 | 12 |
| Q06 | 0,3000 | 0,4312 | 1,0000 | 13 |
| Q07 | 0,5000 | 0,9649 | 1,0000 | 5 |
| Q08 | 0,0000 | 1,0000 | 0,0000 | 0 |
| Q09 | 1,0000 | 1,0000 | 1,0000 | 13 |
| Q10 | 0,9000 | 0,9576 | 1,0000 | 21 |
| Q11 | 0,5000 | 0,6866 | 1,0000 | 12 |
| Q12 | 1,0000 | 1,0000 | 1,0000 | 20 |
| Q13 | 0,6000 | 0,7833 | 1,0000 | 13 |
| Q14 | 1,0000 | 1,0000 | 1,0000 | 23 |
| Q15 | 0,9000 | 0,9306 | 1,0000 | 26 |
| Q16 | 0,4000 | 0,5646 | 0,3333 | 14 |
| Q17 | 0,2000 | 0,6695 | 0,5000 | 4 |
| Q18 | 0,1000 | 0,3198 | 0,2000 | 4 |
| Q19 | 0,5000 | 0,7788 | 1,0000 | 8 |
| Q20 | 0,0000 | 0,6020 | 0,0000 | 1 |

Q08 có nDCG=1 nhưng P@10=MRR@10=0: pool không có bài nhãn 2, nên thứ tự lý tưởng cũng chỉ gồm bài nhãn 1. nDCG đo chất lượng xếp hạng tương đối trong pool, không chứng minh tìm được bài thỏa đủ NaFSI + ether + nhiệt độ thấp.

Q20 có một bài nhãn 2 trong pool nhưng full-document hybrid không lấy được vào Top 10. BGE semantic lấy bài này ở rank 10: publication:WOS:000439134600040, nghiên cứu full pouch cell HC/layered oxide với ADN electrolyte additive.

## Phân bố điểm và các truy vấn cần cải thiện

![Hình 2. Cùng nhãn và cùng pool: bên trái relevance trực tiếp, bên phải gain graded. Q08 là ví dụ cần đọc hai chỉ số cùng nhau.](../outputs/ai_silver_labels_703_20261026_v1/evaluation/figures_v2/query_heatmaps.png)
Hình 2. Cùng nhãn và cùng pool: bên trái relevance trực tiếp, bên phải gain graded. Q08 là ví dụ cần đọc hai chỉ số cùng nhau.

Q09, Q12, Q14: full-document hybrid trả 10/10 kết quả nhãn 2. Q18 chỉ 1/10 kết quả trực tiếp và kết quả đầu ở rank 5; Q17 đạt 2/10; Q04–Q06 đạt 3/10; Q16 đạt 4/10. Cần đọc rationale từng bản ghi trước khi quyết định lỗi nằm ở retrieval hay metadata.

## Độ nhạy với nhãn và bất định

| Phương pháp | P@10 gốc | Review→0 | Review→2 | P@10 nhãn ≥1 |
|---|---|---|---|---|
| TF-IDF | 0,4500 | 0,4300 | 0,5600 | 0,8600 |
| BatteryBERT semantic | 0,2650 | 0,2600 | 0,4050 | 0,6350 |
| BatteryBERT hybrid | 0,5050 | 0,4900 | 0,5600 | 0,9200 |
| BGE semantic | 0,5300 | 0,5150 | 0,5700 | 0,9250 |
| Full-document semantic | 0,4500 | 0,4450 | 0,4750 | 0,8900 |
| Full-document hybrid | 0,5400 | 0,5250 | 0,6000 | 0,9200 |

![Hình 3. Đếm cả liên quan một phần làm điểm tăng; đây là câu hỏi đánh giá khác, không thay thế ngưỡng chính.](../outputs/ai_silver_labels_703_20261026_v1/evaluation/figures_v2/threshold_sensitivity.png)
Hình 3. Đếm cả liên quan một phần làm điểm tăng; đây là câu hỏi đánh giá khác, không thay thế ngưỡng chính.

Hai cột Review→0/2 là kịch bản thay đồng thời 70 nhãn cần review, không phải nhãn mới, CI hay cận toán học bảo đảm cho nDCG. Full-document hybrid có P@10 0,525/0,600 và nDCG 0,7525/0,7734 trong hai kịch bản; kết luận phụ thuộc chất lượng nhãn.

Bootstrap 10.000 lần theo query, cùng resamples cho mọi method, seed 20261026. CI 95% của full-document hybrid: P@10 [0,395; 0,685], nDCG [0,6793; 0,8534], MRR [0,6117; 0,925]. Khoảng này chỉ phản ánh biến thiên giữa 20 query với nhãn cố định; không đo sai số của AI judge.

Chênh lệch với BGE và hybrid lịch sử có paired CI chứa 0 cho cả ba chỉ số. Các so sánh là thăm dò, 45 phép so sánh chưa hiệu chỉnh đa kiểm định; không suy ra chiến thắng chắc chắn hoặc ngoài tập query hiện tại.

## Diễn giải lỗi, giới hạn và file bàn giao

| Truy vấn | Vấn đề được phát hiện |
|---|---|
| Q04–Q06 | Có đúng họ vật liệu nhưng thiếu điều kiện water/vacancy/carbon coating/low cost; nhiều patent ghi ngược anode/cathode. Không tự sửa metadata khi chấm. |
| Q08 | Chưa có nhãn 2 trong pool. Phải mở rộng pooling từ dữ liệu hiện có và/hoặc người dùng cung cấp tài liệu, không kết luận cả corpus không có bài phù hợp. |
| Q16 | Nhầm electrode coating với particle coating, film wrapping hoặc công đoạn cutting; cần phân biệt intent trong bước phát triển sau. |
| Q17 | BMS/voltage balancing và material safety chỉ là nền; thermal management ở cấp pack cần bằng chứng riêng. |
| Q18 | Nhầm tái chế Li-ion/slag/masks để tạo electrode SIB với tái chế chính spent SIB và cathode metals. |
| Q20 | Các mảnh HC, layered oxide và additive phải cùng một full cell. Cathode sacrificial additive khác electrolyte additive. |

Các điểm này là chẩn đoán sau đánh giá, chưa được dùng để tuning/retrain hoặc chọn checkpoint. Nếu phát triển Search tiếp, cần tách tập phát triển mới và giữ đánh giá độc lập; không tối ưu trực tiếp trên 703 nhãn vừa dùng để báo cáo.

Silver Labels là kết quả của một AI assistant, không phải human ground truth hoặc thỏa thuận nhiều người chấm. Confidence high/medium/low chưa hiệu chuẩn. 70 trường hợp review có thể đổi sau xác minh; dữ liệu nguồn có 28 Abstract trống và các role labels đáng nghi. Những giới hạn này áp dụng cho toàn bộ kết luận xếp hạng.

| File | Nội dung |
|---|---|
| silver_labels_703.csv | Đúng 7 cột yêu cầu; giữ nguyên 703 query_id–doc_id theo thứ tự nguồn. |
| labels_with_source_703.csv / needs_review.csv | Bản kèm Query–Title–Abstract và danh sách 70 trường hợp review. |
| evaluation/metrics_summary.csv / metrics_per_query.csv | Ba chỉ số chính trên 6 methods và chi tiết 120 nhóm. |
| evaluation/confidence_intervals.csv / paired_deltas.csv | Bootstrap theo query; nhãn AI giữ cố định. |
| evaluation/sensitivity_summary.csv / pooled_recall_supplement.csv | Kiểm tra ngưỡng, review scenarios và Recall trong pool. |
| evaluation/validation.json / labels_frozen_manifest.json | Kiểm chứng số liệu, tính toàn vẹn và hash bộ nhãn. |

Thư mục gốc: outputs/ai_silver_labels_703_20261026_v1/. Báo cáo này bổ sung kết quả mới cho PDF nghiên cứu v2, thay phần pilot 20 cặp về mức bao phủ AI; trạng thái nhãn người vẫn chưa xác minh. Không ghi đè báo cáo cũ hoặc evaluator human.
