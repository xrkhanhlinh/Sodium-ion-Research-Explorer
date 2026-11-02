# Báo cáo kết thúc học phần: Sodium-ion Research Explorer

Bản tiếng Việt, khổ A4. Bản cập nhật **30/10/2026** bổ sung chương **Thực nghiệm nâng cấp Search và Search Radar Pilot**; PDF mới nằm ở `../output/pdf/BAO_CAO_KET_THUC_HOC_PHAN_SODIUM_ION_20261030_TNR13.pdf`. PDF ngày 26/10 được giữ nguyên. Bìa gốc vẫn giữ nguyên; ngày cập nhật và phạm vi kết quả mới ghi trong tóm tắt/chương bổ sung.

Bản chỉnh định dạng: đã bỏ toàn bộ mục **“Quy ước và phạm vi”**. Sau trang bìa, văn bản dùng **Times New Roman cỡ 13 TeX pt**, bao gồm tiêu đề, mục lục, bảng, chú thích hình, dòng nguồn và tài liệu tham khảo. Trang bìa đầu tiên giữ nguyên. Chữ nằm trong ảnh kết quả gốc được giữ theo ảnh; ký hiệu toán dùng font toán và cỡ chỉ số tương ứng. `typography_13pt.tex` áp dụng định dạng sau bìa; bộ kiểm tra xác minh font/cỡ chữ trong PDF và so ảnh bìa trước/sau.

## Nội dung

Chín chương trình bày bài toán, lý thuyết và công thức, dữ liệu và pipeline, huấn luyện, search engine chi tiết, giao thức đánh giá, 10 topic của từng model, Search Radar Pilot và kết luận. Phụ lục chứa các hình kết quả bổ sung và danh sách 20 query. Bảy catalog có **70 topic, 700 vị trí từ khóa**. Giữ kết quả **703 AI Silver Labels, 20 queries, 6 methods** ngày 26/10 như lịch sử; chương mới tách riêng phần đủ nhãn, missing labels và development/held-out chưa có human qrels.

Thông tin trường, học phần, nhóm/sinh viên, mã sinh viên, lớp và giảng viên chưa được cung cấp, nên để dòng trống ở trang bìa `main.tex`. Điền thông tin thật trước khi nộp.

## Biên dịch độc lập / Overleaf

Upload toàn bộ ZIP, chọn **XeLaTeX**, main document `main.tex`. Nguồn cần `typography_13pt.tex`, `chapters/`, `generated/`, `figures/` và `references.bib`. Máy biên dịch cần Times New Roman để giữ đúng định dạng yêu cầu. Phần bìa có cấu hình font riêng đã giữ nguyên.

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Từ thư mục `course_report_latex/` tại workspace gốc, có thể dùng Tectonic:

```powershell
..\.tools\tectonic\tectonic.exe --only-cached --keep-logs --keep-intermediates --outdir build main.tex
```

Gói TeX/font có thể cần tải lần đầu. Khi cache đầy đủ thêm `--only-cached` để chạy offline.

## Tạo lại bảng/hình từ artifacts trong dự án

Từ gốc dự án:

```powershell
.\.venv\Scripts\python.exe -X utf8 course_report_latex\build_report.py
.\.venv\Scripts\python.exe -X utf8 scripts\report_search_radar.py
```

Sau đó biên dịch, rồi kiểm tra:

```powershell
.\.venv\Scripts\python.exe -X utf8 course_report_latex\validate_report.py
```

`build_report.py` cần toàn bộ workspace để đọc các CSV/JSON/PNG gốc. **Biên dịch PDF từ nguồn đã đóng gói không cần workspace, models hoặc API.** Script không train, không encode, không mở Qdrant và không gọi AI. Các model/data gốc được giữ nguyên; `evidence_manifest.json` lưu hash các nguồn được dùng.

`figure_inventory.json` chỉ rõ hình đã nhúng, nguồn, hash và lý do bỏ bản trùng/công thức raster/contact sheet. `topic_catalog_70.csv` cho phép kiểm tra từng topic. `validation.json` ghi số trang, hình/bảng, kiểm tra tham chiếu, font, giới hạn trang và hash. Các PNG trong `qa/` chỉ dùng kiểm tra bố cục, không phải thí nghiệm mới.

Nhãn AI và tên topic chưa được chuyên gia xác minh. Full-document là toàn bộ Title–Abstract có sẵn, không phải full article. Proxy chọn checkpoint không phải search accuracy. Các kết luận và giới hạn này đã được ghi trong báo cáo.
