# Rà soát kết quả search ngày 19/10/2026

Hệ thống chạy được và các bảng kết quả nhất quán với nhau. **Chưa có bằng chứng đánh giá độc lập để công bố độ chính xác search hoặc chọn phương pháp tốt nhất.** Phiên rà soát này kiểm tra số liệu, chạy truy vấn thật và sửa phần hiển thị kết quả.

Phần kiểm chứng công thức, sửa lỗi biên/cache và thí nghiệm tối ưu CPU được ghi riêng tại [Kiểm chứng toán học và tối ưu code](MATH_AND_OPTIMIZATION_REVIEW.md).

## Số liệu đã đối chiếu

| Nội dung | Kết quả |
|---|---:|
| Tài liệu trong corpus | 13.702 |
| Truy vấn × cấu hình trong benchmark mới | 24 × 10 = 240 lượt |
| Dòng kết quả Top-10 | 2.330 |
| Dòng kết quả Top-5 | 1.165 |
| Phương pháp trong evaluator, gồm baseline lịch sử | 14 |
| Cặp query–document trong pool chấm chung | 942 |
| Ô điểm đã điền ở phiếu A / B | 0 / 0 |
| Phương pháp có Precision@5, nDCG@5, MRR@5 chính thức | 0 |
| Truy vấn miền có cùng thứ tự Top-5 pretrained/fine-tuned sau rerank | 17 / 23 |

Đã kiểm tra đủ tổ hợp trong protocol, số tài liệu từng lượt chạy, thứ hạng liên tiếp, doc_id thuộc corpus, không lặp tài liệu trong cùng lượt, Top-5 khớp phần đầu Top-10 và hai phiếu chấm có cùng pool. Không phát hiện sai lệch trong các kiểm tra này. Đây là tính nhất quán của artifact, không phải chứng minh tài liệu phù hợp nhu cầu.

Q24 là truy vấn OOV tổng hợp. Ba nhánh dense vẫn trả năm tài liệu; BM25 và sáu nhánh hybrid/rerank trả rỗng do thiếu lexical support. Một ca OOV không đủ để xác nhận khả năng phát hiện truy vấn ngoài miền. Cơ chế chặn này cũng có thể bỏ sót paraphrase hợp lệ.

17/23 là đồng thuận thứ hạng, không phải độ chính xác 73,9%. Các nhãn AI cũ chỉ thuộc development diagnostics, không thay nhãn độc lập và không dùng để chọn winner.

Nguồn: [protocol](../outputs/search_upgrade_20261018/experiment_protocol.json), [run manifest](../outputs/search_upgrade_20261018/run_manifest.csv), [Top-10](../outputs/search_upgrade_20261018/rankings_top10.csv), [Top-5](../outputs/search_upgrade_20261018/top5_comparison.csv), [summary chính thức](../outputs/search_upgrade_20261018/evaluation_14_methods/summary.csv), [phiếu A](../outputs/search_upgrade_20261018/annotator_A_NEW.csv), [phiếu B](../outputs/search_upgrade_20261018/annotator_B_NEW.csv). Kết quả kiểm tra lại: [artifact_audit.json](../outputs/search_review_20261019/artifact_audit.json).

## Kết quả tìm tài liệu bằng truy vấn mẫu

Chưa có từ khóa riêng từ người dùng nên dùng truy vấn development Q19: `Aqueous sodium-ion batteries with water-in-salt electrolytes`. Chạy thật `RERANK_PRETRAINED` hai lần; cả hai lần có cùng doc_id và thứ tự Top-5 như benchmark đã lưu.

| Hạng | Tài liệu | Năm | Identifier |
|---|---|---:|---|
| 1 | Fluorine-Free Water-in-Salt Electrolyte for Green and Low-Cost Aqueous Sodium-Ion Batteries | 2018 | 10.1002/cssc.201801930 |
| 2 | High-Energy Aqueous Sodium-Ion Batteries | 2021 | 10.1002/anie.202017167 |
| 3 | Halide-free water-in-salt electrolytes for stable aqueous sodium-ion batteries | 2020 | 10.1016/j.nanoen.2020.105176 |
| 4 | Synthesis and Application of an Aromatic Sulfonate Sodium Salt for Aqueous Sodium-Ion Battery Electrolytes | 2023 | 10.1002/ente.202201045 |
| 5 | A High-Voltage Aqueous Electrolyte for Sodium-Ion Batteries | 2017 | 10.1021/acsenergylett.7b00623 |

Đọc Abstract đã lưu: hạng 1–3 mô tả trực tiếp water-in-salt/WISE; hạng 5 mô tả NaFSI với tỉ lệ mol nước:muối dưới 2:1. Hạng 4 nói về dung dịch Na-TFP 0,5 m và so sánh với các điện giải loãng khác; Abstract chưa xác nhận điều kiện water-in-salt. Đây là nhận xét nội dung trên một query development, chưa phải chấm relevance độc lập hoặc bằng chứng P@5.

Q08 cũng cho thấy giới hạn nhiều ràng buộc: hạng 1 nói về ether và nhiệt độ thấp nhưng Abstract chưa nêu NaFSI; hạng 2 là màng polymer PFSA-Na; hạng 4 có NaFSI nhưng mô tả thử nghiệm ở nhiệt độ phòng. Không thể xác nhận đồng thời NaFSI + ether + nhiệt độ thấp chỉ từ các kết quả này. Bộ lọc từ bắt buộc chỉ xác nhận token presence, chưa xác nhận quan hệ giữa chúng.

Kết quả chạy thật và Abstract: [lượt 1](../outputs/search_review_20261019/live_q19.json), [lượt 2](../outputs/search_review_20261019/live_q19_repeat.json). Các nhận xét Q08 dựa trên [Top-5 đã lưu](../outputs/search_upgrade_20261018/top5_comparison.csv); không tạo hoặc sửa nhãn chấm.

## Thời gian xử lý

Sáu requests warm lịch sử trên ba query có mean **4,88 s**, median **4,59 s**, max **6,12 s**. Đây là mẫu nhỏ, không cache query/pair, không gồm model loading và không đo concurrency. Nguồn: [dữ liệu đo gốc](../outputs/search_upgrade_20261018/live_latency_uncached.csv), [phạm vi phép đo](../outputs/search_upgrade_20261018/live_latency_scope.json).

Hai lượt kiểm tra mới là query đầu tiên trong process mới nên không so trực tiếp với warm latency. Lượt 1 ghi thời gian bất thường 19.656,64 s trong cùng phiên AppTest timeout; chưa xác định nguyên nhân, giữ JSON gốc và không dùng làm số đo hiệu năng đại diện. Lượt 2 ghi 22,46 s, gồm tải model trong bước search. Các lượt này chỉ dùng để kiểm tra khả năng chạy và tái lập thứ hạng; chưa thực hiện benchmark latency mới.

## Giao diện đã sửa và cách xem

- **Search:** xóa kết quả cũ khi đổi nguồn/năm; hiện số tài liệu thực tế; phân biệt bộ lọc không còn tài liệu với trường hợp truy vấn thiếu lexical support. Kết quả rỗng không được diễn giải là corpus chắc chắn không có đáp án.
- **Comparison:** thêm tiêu đề và nguồn vào bảng thứ hạng để đọc được tài liệu đang so sánh; đổi bộ lọc sẽ xóa bảng kết quả cũ.
- **Evaluation:** mở **Tra cứu Top-5 benchmark đã lưu**, chọn query và phương pháp để đọc Title, Abstract, passage và tải CSV. Đây là snapshot benchmark, bộ lọc Search không áp dụng vào snapshot này.

Chạy app bằng `.\.venv\Scripts\python.exe run_demo.py`, mở `http://127.0.0.1:8501`. Search mặc định vẫn là RERANK_PRETRAINED. Phiên rà soát không thay mô hình, corpus, cấu hình xếp hạng hoặc kết quả benchmark cũ.

## Kiểm tra và phần còn thiếu

Kiểm tra search và giao diện bằng `tests/test_search_v2.py` cùng `tests/test_search_v2_app.py`: **20 tests passed, 0 failures/errors/skips**. Các luồng gồm bộ lọc, xóa kết quả cũ, truy vấn rỗng kết quả, bộ lọc không còn tài liệu, bảng so sánh có tiêu đề và tra cứu snapshot Q19/Q24. Kết quả cuối lưu tại [tests.xml](../outputs/search_review_20261019/tests.xml). Lần AppTest timeout đầu tiên được giữ riêng tại [tests_attempt1.xml](../outputs/search_review_20261019/tests_attempt1.xml); đã điều chỉnh thứ tự thao tác test để chọn BM25 sau lượt rerun bộ lọc.

Không có trình duyệt kết nối trong phiên này nên chưa kiểm tra trực quan bằng browser/screenshot. AppTest kiểm tra các widget và dữ liệu giao diện nhưng không thay kiểm tra bố cục bằng mắt.

Để có số liệu chất lượng chính thức: chốt intent/rubric, hai người chấm độc lập 942 cặp, đo agreement, phân xử thành consensus rồi chạy evaluator. Chưa thể kết luận pretrained, fine-tuned hay chunked tốt nhất từ các bằng chứng hiện có.
