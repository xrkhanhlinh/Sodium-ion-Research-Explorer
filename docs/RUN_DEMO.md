# Chạy Sodium-ion Research Explorer

Tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

Mở http://127.0.0.1:8501; Ctrl+C để dừng. Mặc định pretrained + hybrid + reranker trên 13.702 tài liệu. Không train hoặc mở Qdrant. BM25 cache có thể được tạo từ corpus; model load cục bộ.

```powershell
.\.venv\Scripts\python.exe -m src.search_v2 "Na2FeP2O7 cathode cycling stability"
.\.venv\Scripts\python.exe -m src.search_v2 "Hard carbon anodes" --method BM25_CHEMICAL
```

Search có bộ lọc nguồn/năm. Required terms tách bằng `;`, ví dụ `NaFSI; ether`: chỉ kiểm tra token trong Title/Abstract, chưa xác nhận quan hệ/phủ định. Comparison dùng các nhánh hiện tại. Topics có LDA/full-document/baseline. Evaluation dùng kết quả v2 và phiếu chấm 942 cặp.

Reranker CPU mất vài giây/lượt; lượt đầu có thêm model loading. HYBRID_PRETRAINED hoặc BM25_CHEMICAL giảm chi phí xử lý. Fine-tuned/chunked là đối chứng tùy chọn.

Dependencies: `requirements-stage5.txt`. Workspace đã có models. Nếu thiếu reranker, tải công khai bằng `scripts/download_search_reranker.py`; không tự tải model khi tìm kiếm hoặc trộn checkpoint sai lineage.

Benchmark/latency phiên bản mới:

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

`run_stage4.py`, `search_stage4.py`, `src.stage4_benchmark`, `src.stage4_live_latency` là alias sang v2; không chạy backend Qdrant cũ. Báo cáo chính: [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md).
