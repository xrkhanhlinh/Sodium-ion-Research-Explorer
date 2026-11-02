# Chạy Sodium-ion Research Explorer

**Bản24/10/2026:** checkpoint BatteryBERT mới đã train đủ các đoạn Abstract,
không cắt train ở128 token. Khi `deployment.json` của run mới bật cờ, Search
mặc định là **BATTERYBERT_FULLDOC_HYBRID**, và catalog full-document là topic
chính. [Kết quả train và trade-off](BATTERYBERT_FULL_DOCUMENT_RESULTS_20261024.md).
Dữ liệu chỉ có Title–Abstract, chưa có toàn văn bài báo. Mô tả và lệnh phía
dưới về nhánh23/10/V2/V3 là lịch sử để đối chứng.

```powershell
.venv/Scripts/python.exe run_demo.py
# Tái lập offline trong thư mục MỚI, giữ nguyên run đã đóng băng:
.venv/Scripts/python.exe scripts/train_batterybert_full_document.py all --output outputs/batterybert_full_document_REPLAY
.venv/Scripts/python.exe scripts/evaluate_batterybert_full_document.py --output outputs/batterybert_full_document_REPLAY
# Nhập nhãn thật cho pool mở rộng6 phương pháp sau khi hoàn tất chấm:
.venv/Scripts/python.exe scripts/finalize_batterybert_full_document.py evaluate --labels PATH_TO_VERIFIED_LABELS.csv
```

Artifacts hiện hành: `outputs/batterybert_full_document_20261024_v1/`. Pool
mới có sáu phương pháp trên cùng20 query; Evaluation có phiếu riêng cho pool
này, không ghi đè pool bốn phương pháp cũ. Phiếu chưa chấm không tạo accuracy.
Main Search dùng Qdrant riêng tại thư mục run mới; dừng app trước khi CLI
truy cập cùng storage. App chỉ load checkpoint/index, không train lại.

Topic cũ vẫn chọn được và có seed ARI cao hơn. Ở topic full-document, nút tải
danh sách cần rà soát xuất các trường hợp xa centroid/gần hai nhóm; đây không
phải xác suất model sai. API đọc body dài được trình bày trong
[protocol](BATTERYBERT_FULL_DOCUMENT_TRAINING_20261024.md).

Topic chính từ 23/10/2026: **BatteryBERT fine-tuned · optimized**. Topics, nhãn kết quả và bộ lọc topic dùng chung catalog mới; Search vẫn mặc định dùng BatteryBERT pretrained/RRF. Xem [pipeline, 29 lượt thử nghiệm và kết quả test](BATTERYBERT_OPTIMIZATION_20261023.md). Các lệnh `run_research_completion.py topics/select` phía dưới tái lập nhánh topic lịch sử; không thay model mới khi khởi động app.

Tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

Mở http://127.0.0.1:8501; Ctrl+C để dừng. Mặc định web dùng **BATTERYBERT_HYBRID = TF-IDF + BatteryBERT pretrained/Qdrant → RRF k=60**, trả Top-10. Ba đối chứng trong cùng selector là TFIDF_COSINE, BATTERYBERT_SEMANTIC và BGE_SEMANTIC. Bật “Hiện phương pháp lịch sử V2/V3” để dùng các thuật toán cũ. Model load cục bộ; không train, không encode lại corpus khi query. Một worker sở hữu Qdrant Local persistent.

Kết quả hoàn thiện hiện tại: `outputs/research_completion_v1/`. [Báo cáo và cấu hình trước/sau](../outputs/research_completion_v1/COMPLETION_REPORT.md). Web có lọc nguồn/năm/topic, title/abstract/identifier, topic ID/label và điểm cosine hoặc RRF được ghi rõ. Chọn Topics để xem 10 cụm KMeans và đối chứng lịch sử. Nhãn topic còn tạm thời; 0% outlier là đặc tính KMeans.

Chuẩn bị hoặc kiểm tra từ **các model/vector đã có**, dừng app trước khi truy cập cùng Qdrant storage:

```powershell
.\.venv\Scripts\python.exe run_research_completion.py prepare
.\.venv\Scripts\python.exe run_research_completion.py topics
.\.venv\Scripts\python.exe run_research_completion.py select --representation ctfidf_bm25_1
.\.venv\Scripts\python.exe run_research_completion.py index
.\.venv\Scripts\python.exe run_research_completion.py benchmark
.\.venv\Scripts\python.exe run_research_completion.py report
```

Các bước topic chỉ cập nhật representation của model KMeans đã lưu. Index tái sử dụng collection hoàn tất sau khi kiểm tra provenance/count; benchmark đã hoàn tất được tái sử dụng theo checksum. Không gọi training. Máy mới cần có các checkpoint và vector gốc tương thích trước khi chạy các lệnh này.

Evaluation dùng `outputs/research_completion_v1/annotation_template_0_2.csv`: 0 không liên quan, 1 một phần, 2 trực tiếp. Pool đã ẩn rank/score/method. Không điền nhãn từ điểm ranking. Nhập phiếu đã được người chấm xác minh trên web hoặc:

```powershell
.\.venv\Scripts\python.exe run_research_completion.py evaluate --labels PATH_TO_REAL_HUMAN_LABELS.csv
```

P@10/MRR@10 dùng grade≥2; nDCG gain=2^grade−1 và IDCG từ pool chung được chấm đầy đủ. Nhãn trống không phải 0. Chưa có nhãn thì trạng thái là PENDING_HUMAN_ANNOTATION, metrics để trống. Không tính Recall khi chưa có tập relevant đầy đủ. Nhãn dùng đánh giá cuối, không dùng tuning ranking.

Kiểm thử đã dùng cho phiên bản này:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_research_completion.py tests/test_final_system.py tests/test_search_v2.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_search_v2_app.py -q
```

Artifacts mới: `outputs/final_system/`. Xem [SYSTEM_ARCHITECTURE.md](../outputs/final_system/SYSTEM_ARCHITECTURE.md) để chuẩn bị/reuse embeddings và collections, kiểm thử, benchmark và nhập nhãn độc lập. Dừng dashboard trước khi CLI truy cập cùng Qdrant Local storage. Thiếu model/vector sẽ báo lỗi, không fallback sang encoder khác.

```powershell
.\.venv\Scripts\python.exe -m src.search_v2 "Na2FeP2O7 cathode cycling stability"
.\.venv\Scripts\python.exe -m src.search_v2 "Hard carbon anodes" --method BM25_CHEMICAL
```

Required terms tách bằng `;`, ví dụ `NaFSI; ether`: chỉ kiểm tra token trong Title/Abstract, chưa xác nhận quan hệ/phủ định. Comparison hiện dùng 20 truy vấn chung và bốn phương pháp mới. Artifacts `outputs/final_system/` cùng các lệnh CLI dưới đây là đối chứng lịch sử; Search V3 vẫn dùng checkpoint fine-tuned thật.

Reranker CPU mất vài giây/lượt; lượt đầu có thêm model loading. TF-IDF là baseline trong app. Các lệnh CLI `src.search_v2` ở trên vẫn giữ Top5 gốc để đối chứng; dashboard xuất Top10 mà không đổi thuật toán V2.

Dependencies: `requirements-final-system.txt` (bao gồm requirements-stage5). Workspace phải có các checkpoint và embeddings đã xác minh. Không tự tải model khi tìm kiếm hoặc trộn checkpoint sai lineage.

Benchmark/latency Search V2 trước tích hợp (được giữ nguyên):

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_search_v2.py --output outputs/search_runs/my_benchmark
.\.venv\Scripts\python.exe scripts/measure_search_v2_live.py --output outputs/search_runs/my_latency
```

Chỉ sau khi có consensus labels thật:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_search_v2.py --labels PATH_TO_REAL_CONSENSUS.csv --output outputs/search_v2_verified_NEW
```

Kiểm tra nhanh:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_search_v2.py tests/test_search_v2_app.py tests/test_artifact_history.py -q
```

`run_stage4.py`, `search_stage4.py`, `src.stage4_benchmark`, `src.stage4_live_latency` là alias sang v2; không chạy backend Qdrant cũ. Qdrant mới chỉ dùng `outputs/final_system/qdrant/`. Báo cáo tích hợp: [FINAL_SYSTEM_REPORT.md](../outputs/final_system/FINAL_SYSTEM_REPORT.md). nDCG@10/P@10/MRR@10 giữ trống với trạng thái PENDING_INDEPENDENT_LABELS cho tới khi đủ nhãn được xác minh; Evaluation cho phép tải phiếu và import nhãn 0–3.
