# Sodium-ion Research Explorer

## Search Radar Pilot · cập nhật 30/10/2026

Chạy `python -m streamlit run app.py`, chọn **Search Radar Pilot**. Nhánh TF-IDF + BGE + BatteryBERT, chunk và evidence reranking được tách khỏi Search mặc định. Kết quả, giới hạn nhãn và lệnh chạy lại nằm trong [báo cáo thực nghiệm](docs/SEARCH_FINAL_RESEARCH_REPORT.md). Báo cáo kết thúc học phần có chương bổ sung và PDF ngày 30/10 trong `output/pdf/`.

## Cập nhật ngày 26/10/2026

Đã tiếp tục từ checkpoint, giữ nguyên corpus, encoder và embeddings. Catalog hiện hành dùng KMeans K=10, n_init=50, parser bảo toàn công thức hóa học và nhãn diễn giải AI có bằng chứng. Search mặc định vẫn là hybrid TF-IDF + BatteryBERT/Qdrant + RRF60; reranker thử nghiệm bị loại sau gate.

- [Báo cáo nghiên cứu mới, 22 trang có kết quả và hình ảnh](output/pdf/BAO_CAO_NGHIEN_CUU_CAI_THIEN_SODIUM_ION_20261026_v2.pdf)
- [Bản Markdown](docs/BAO_CAO_NGHIEN_CUU_CAI_THIEN_SODIUM_ION_20261026_v2.md)
- [Đánh giá mới trên đủ 703 Silver Labels: PDF 6 trang](output/pdf/DANH_GIA_SEARCH_SILVER_703_20261026_v2.pdf) · [Markdown](docs/DANH_GIA_SEARCH_SILVER_703_20261026_v2.md)
- [CSV nhãn đủ 703 cặp](outputs/ai_silver_labels_703_20261026_v1/silver_labels_703.csv): 143 nhãn 0, 302 nhãn 1, 258 nhãn 2; 70 trường hợp cần review. Đã tính đủ 20 query × 6 methods. Full-document hybrid đạt P@10=0,540, nDCG@10=0,7693 và MRR@10=0,7767. Đây là Silver evaluation, chưa phải human ground truth.
- [CSV Silver Labels: thử 20 cặp đầu tiên](outputs/ai_silver_labels_20261026_v1/batch_0001_labels_20.csv): 11 nhãn 2, 4 nhãn 1, 4 nhãn 0, 1 nhãn trống; 8 trường hợp cần review. Đây là nhãn AI, không phải nhãn người xác nhận.
- [Nhật ký quyết định và giới hạn](docs/RESEARCH_REVISION_WORKLOG_20261025.md)

```powershell
.\.venv\Scripts\python.exe scripts/resume_research_revision.py --status
.\.venv\Scripts\python.exe scripts/resume_research_revision.py
```

Lệnh resume kiểm tra receipt/hash, bỏ qua bước completed hợp lệ, lưu checkpoint/log/lỗi và không ghi đè output cũ. Kết quả nghiên cứu ở `outputs/research_revision_20261025_v1/`; batch Silver pilot lưu riêng. Yêu cầu mới đã mở rộng và hoàn tất 703/703 cặp tại `outputs/ai_silver_labels_703_20261026_v1/`. Resume riêng nhánh nhãn/đánh giá bằng `.venv/Scripts/python.exe scripts/resume_silver_703_20261026.py`. Phần dưới ghi lại cấu hình và bằng chứng ngày 24/10; báo cáo mới và phụ lục Silver ở trên là tài liệu hiện hành.

## Snapshot ngày 24/10/2026

Tìm kiếm và khám phá topic trên **13.702 tài liệu đã làm sạch**. Phiên24/10 tiếp tục train BatteryBERT trên **toàn bộ Title–Abstract sẵn có**, chunk256/64, loss batch32; không còn cắt Abstract train ở128 token. Khi artifacts và cổng kiểm tra đầy đủ, web dùng **TF-IDF + BatteryBERT full-document/Qdrant + RRF k=60**, trả Top10. Các phương pháp cũ vẫn có để đối chứng; app không train khi khởi động.

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

- [Báo cáo hệ thống tích hợp](outputs/final_system/FINAL_SYSTEM_REPORT.md)
- [Kết quả train đầy đủ và đánh giá mới — 24/10/2026](docs/BATTERYBERT_FULL_DOCUMENT_RESULTS_20261024.md)
- [Báo cáo toàn pipeline sau training (PDF)](output/pdf/BAO_CAO_PIPELINE_BATTERYBERT_FULLDOC_20261024.pdf)
- [Protocol training và API đọc body dài](docs/BATTERYBERT_FULL_DOCUMENT_TRAINING_20261024.md)
- [Pipeline và tối ưu BatteryBERT trước training mới — 23/10/2026](docs/BATTERYBERT_OPTIMIZATION_20261023.md)
- [Mười topic của cả sáu pipeline, số tài liệu và top10 keywords](docs/TOP10_TOPICS_ALL_MODELS_20261024.md)
- [Kiến trúc và tái lập](outputs/final_system/SYSTEM_ARCHITECTURE.md)
- [Báo cáo nghiên cứu trước tích hợp](docs/FINAL_RESEARCH_REPORT.md)
- [Hướng dẫn chạy/đánh giá](docs/RUN_DEMO.md)
- [Audit và thử nghiệm BatteryBERT topic](docs/BATTERYBERT_TOPIC_AUDIT.md)
- [Hoàn thiện Search, topic và đánh giá 0–2](outputs/research_completion_v1/COMPLETION_REPORT.md)
- [Đặc tả](docs/PROJECT_SPEC.md)
- [Trạng thái](docs/PROGRESS.md)
- [Code/report lịch sử đã gỡ](archive/README.md)

Topic full-document mới dùng **BatteryBERT đã train đủ đoạn + train-fit source projection + KMeans10**. Catalog chỉ được kích hoạt sau cổng kỹ thuật và review agent; Search/Topics dùng chung catalog hiện hành. Bản23/10 vẫn được giữ và có ARI seed cao hơn. Bản mới cân hơn nhưng C_v/diversity/ARI thấp hơn; không tuyên bố thắng mọi chỉ số. Nhãn khoa học còn tạm thời.

Validation cùng full-document pooling tăng0,906174→0,933093; test tăng0,887839→0,924704. Đây là proxy Title→document có Title context, không phải relevance Search độc lập. Dữ liệu chưa có toàn văn bài báo; API có thể inference body dài được cung cấp, chất lượng trên toàn văn chưa được đánh giá.

Relevance chính thức còn chờ nhãn người0–2 cho pool20 truy vấn, mở rộng lên6 phương pháp. Topic coherence, AI pilot và tests kỹ thuật không phải accuracy. Artifacts mới: `outputs/batterybert_full_document_20261024_v1/`; model/vectors cũ được giữ nguyên. Xem thêm [đính chính về tính liền kề ngram](docs/KEYWORD_ADJACENCY_CORRECTION_20261024.md).

Dependencies bổ sung: `requirements-final-system.txt`. Qdrant Local chỉ có một process sở hữu storage: dừng app trước khi chạy CLI build/validation. Artifacts full-document nằm riêng trong `outputs/batterybert_full_document_20261024_v1/`; bản tích hợp cũ ở `outputs/final_system/` được giữ để đối chứng. Lệnh chuẩn bị và kiểm thử nằm trong báo cáo kiến trúc và protocol mới. Không chạy lại các script training để khởi động dashboard.
