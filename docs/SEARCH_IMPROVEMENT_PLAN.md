# Search improvement protocol — v1, 30/10/2026

Đóng băng thiết kế trước khi xem kết quả mới. Benchmark 703 nhãn chỉ là chẩn đoán lịch sử, không dùng chọn cấu hình.

1. Xác minh hash, ID order, shape/L2 của ba nhánh. Đối chiếu A0 exact matrix với thứ hạng baseline đã lưu; ghi sai lệch thay vì ngầm coi tương đương.
2. Candidate depth 50/100/200; RRF k=60. A0 TF-IDF+full-document; A1 TF-IDF+BGE; A2 BGE+full-document; A3 ba nhánh; thêm trọng số (1,1,2) cố định như ablation chưa chọn bằng nhãn.
3. A4: A3 + xếp theo số điều kiện khớp từ vựng trong Top 30, chỉ là rule-based đối chứng. A6: ưu tiên bằng chứng quan hệ trong cùng câu, thiếu bằng chứng khác phủ định; thử depth 20/30/50. Không suy diễn scientific entailment từ token presence.
4. A5: tái sử dụng frozen prefix theo hash, chỉ chạy block cuối checkpoint accepted, tạo chunk vectors trong namespace mới; giữ chunking/Abstract pooling cũ. Kiểm parity document aggregation. So max-chunk và unique-token-weighted mean; một ID tài liệu xuất hiện một lần.
5. MiniLM control riêng, tùy chọn CLI; không bật lại trong Search. Không có evidence model mới được xác minh thì ghi unavailable, không gọi rule matcher là neural entailment.
6. Query-family development/held-out mới có family disjoint, do hệ thống đề xuất và cần chuyên gia review. Xuất annotation template trống, provenance, reviewer, verifier, thời điểm, rationale; không tạo nhãn người. Pool hợp candidate cần chấm trước tính metric.
7. Gate chọn A7: chỉ dùng development đầy đủ human qrels; paired bootstrap delta P/nDCG/MRR có CI thấp không âm, ít nhất một CI thấp dương; latency P95 ≤ 1,5× baseline, peak RSS ≤ 1,5× baseline, không lỗi ID/filter; khóa cấu hình rồi mới chạy held-out. Nếu chưa đủ nhãn: NOT_SELECTED_PENDING_LABELS. Mục tiêu P≥0,65, nDCG≥0,82, MRR≥0,85 không phải kết quả.
8. Dashboard bảy tab, radar 100× metric trên cùng query set; không có CSR khi chưa có nhãn điều kiện kiểm chứng. Latency plot riêng. Lưu CSV/JSON/PNG/Markdown, tests và hashes; bổ sung kết quả thực vào báo cáo kết thúc học phần, giữ bản PDF ngày 26/10.
