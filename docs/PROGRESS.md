# Trạng thái dự án

- Search hiện tại: pretrained + lexical/dense hybrid + MiniLM reranker, cùng corpus 13.702 tài liệu.
- Đã chạy: 24 queries × 10 cấu hình; evaluator 14 methods / 322 tổ hợp / pool 942 cặp.
- Relevance: **PENDING_INDEPENDENT_LABELS**.
- Fine-tuning cũ và BERTopic baseline/chunked giữ làm đối chứng.
- Cleanup hoàn tất: **185 tests pass, 0 failures/errors/skips**. Bản trước sửa đã lưu, code/report cũ gỡ khỏi luồng hiện tại. Bằng chứng tại outputs/project_cleanup_20261018/.

[Báo cáo chính](FINAL_RESEARCH_REPORT.md) · [Hướng dẫn](RUN_DEMO.md). Log một Stage riêng không mô tả trạng thái toàn hệ thống.
