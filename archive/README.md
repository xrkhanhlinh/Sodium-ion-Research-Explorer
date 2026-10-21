# Code và báo cáo lịch sử

`cleanup_20261018/historical_files.zip` giữ nguyên bản trước cleanup: báo cáo trùng/lỗi thời, backend/report generator đã thay thế, slides trước Search v2 và bản trước sửa của các nguồn hiện tại. `cleanup_20261018/manifest.json` ghi SHA-256, kích thước và hành động mỗi file.

Đây không phải hướng dẫn hiện tại. Xem [báo cáo chính](../docs/FINAL_RESEARCH_REPORT.md). Dữ liệu/model/CSV/JSON kết quả giữ nguyên, gồm cả thực nghiệm có kết quả bất lợi.

Giải nén để xem vào thư mục riêng:

```powershell
.\.venv\Scripts\python.exe -m zipfile -e archive/cleanup_20261018/historical_files.zip outputs/restored_history
```

Không ghi đè code hiện tại. src/artifact_history.py xác minh bản code/report archived khi đối chiếu manifest cũ; data/model/metrics phải được kiểm tra từ file thật đang tồn tại.
