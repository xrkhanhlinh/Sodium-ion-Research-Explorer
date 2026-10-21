# Nhật ký hiện tại

- 2026-10-18: Search v2 trên 13.702 tài liệu; benchmark 24 × 10, sáu requests live. Snapshot trước cleanup có 132 tests pass tại outputs/search_upgrade_20261018/.
- 2026-10-18: dọn code/report lỗi thời, gỡ backend Qdrant khỏi app, gom tài liệu về docs/FINAL_RESEARCH_REPORT.md. Nhật ký cũ đầy đủ được lưu trong ZIP lịch sử.
- 2026-10-18: full pytest sau cleanup **185 passed in 212.44s (0:03:32)**, không failures/errors/skips. 395/423 đầu vào lịch sử vẫn trùng file hiện có; 28 bản code/report đã thay thế được đối chiếu từ archive. Data/model không dùng archive fallback.

Kiểm tra cleanup: outputs/project_cleanup_20261018/. Ghi chép Stage mới được thêm bên dưới, không thay thế kết quả/timestamps đã lưu trong snapshots.
