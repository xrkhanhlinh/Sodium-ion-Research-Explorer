# Báo cáo LaTeX Sodium-ion Research Explorer

Mở **`Sodium_ion_Research_Report.pdf`** để đọc bản PDF đã biên dịch; nguồn chính là `main.tex`.

Bản báo cáo chốt artifacts ngày **19-10-2026**, viết bằng tiếng Việt, khổ A4. Gồm bốn chương, kết luận, tài liệu tham khảo IEEE và bốn phụ lục. Tên trường, tác giả, mã sinh viên và giảng viên không có trong yêu cầu nên không được tự điền. Trước khi nộp, cập nhật trang bìa theo thông tin thật và mẫu của trường.

## Cấu trúc

- `main.tex`: thiết lập Unicode/font, bìa, tóm tắt, mục lục, danh mục hình/bảng.
- `chapters/`: nội dung phương pháp, triển khai, thực nghiệm, kết luận, phụ lục.
- `tables/`: 23 files tự sinh từ artifacts (bao gồm macros và phép tính Q19).
- `figures/`: 45 hình sao chép nguyên từ outputs; thêm một sơ đồ tổng thể TikZ. Hình chính theo từng stage, EDA/keywords và kết quả lịch sử trong phụ lục.
- `figure_catalog.json`, `figure_inventory.json`: danh mục hình, vị trí và lý do loại các ảnh trùng; 7 trang keywords bao phủ 84 topics chunked. Ảnh Stage4/Stage5 được ghi rõ là lịch sử.
- `references.bib`: nguồn học thuật và tài liệu chính thức có DOI/URL.
- `build_artifacts.py`: chỉ đọc corpus/kết quả/cấu hình, tạo bảng và manifest.
- `evidence_manifest.json`: SHA-256 nguồn, phép tính thực tế Q19 và phạm vi snapshot.
- `code_review_inventory.json`: danh sách source/config đã đọc đối chiếu.
- `validate_report.py`: kiểm tra source, số liệu snapshot, tham chiếu, PDF và render QA.
- `build_viewers.py`: cập nhật PDF tương thích, HTML offline và ZIP nguồn sau khi biên dịch.
- `build/`: PDF trung gian, log, BibTeX, TOC và AUX.
- `qa/`: PNG kiểm tra bố cục và contact sheets; không phải kết quả mô hình mới.
- `validation.json`: bằng chứng kiểm tra, giới hạn và cảnh báo còn lại.

## Biên dịch bằng XeLaTeX

Cần TeX Live hoặc MiKTeX với XeLaTeX, BibTeX, IEEEtran, babel-vietnamese, fontspec, TikZ, các gói có trong preamble. Font mặc định: **Times New Roman, Arial, Consolas**; các font này có sẵn trên máy đã biên dịch. Trên máy khác/Overleaf, thay ba lệnh font bằng `TeX Gyre Termes`, `TeX Gyre Heros`, `Latin Modern Mono` nếu thiếu font Microsoft, rồi kiểm tra lại bố cục.

Chạy từ `report_latex/`:

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Hoặc `latexmk -xelatex main.tex`. Không dùng pdfLaTeX với preamble fontspec.

## Biên dịch bằng Tectonic cục bộ

Chạy từ gốc `D:\daln`:

```powershell
.\.tools\tectonic\tectonic.exe --keep-logs --keep-intermediates --outdir report_latex/build report_latex/main.tex
```

Tectonic tự chạy BibTeX và các lượt cập nhật tham chiếu. Lần đầu có thể cần tải gói TeX công khai. Sau khi đã cache đủ, thêm `--only-cached` để biên dịch offline. Không có kết nối Gemini trong quá trình biên dịch. Sau khi biên dịch thành công, sao chép `build/main.pdf` thành `Sodium_ion_Research_Report.pdf` nếu thực hiện thủ công; script kiểm tra cũng thực hiện bước này khi hợp lệ.

## Tái dựng bảng và kiểm tra

```powershell
.\.venv\Scripts\python.exe report_latex/build_artifacts.py
.\.tools\tectonic\tectonic.exe --only-cached --keep-logs --keep-intermediates --outdir report_latex/build report_latex/main.tex
.\.venv\Scripts\python.exe report_latex/validate_report.py
.\.venv\Scripts\python.exe report_latex/build_viewers.py
```

`build_artifacts.py` không huấn luyện, không encode embeddings, không mở Qdrant, không gọi API, không đọc `api.env`. Nó từ chối nếu các điều kiện snapshot quan trọng đã đổi, bởi phần nhận xét là nội dung học thuật tĩnh cần được biên tập lại cùng số liệu mới. Không chỉ thay assertions để ép chạy qua khi quota đã được bổ sung và nhãn tăng.

## Phạm vi kết luận

- Relevance AI: 116/942 consensus; 2/23 query đủ pool (Q11/Q19), chưa có nhãn chuyên gia.
- Synthetic: 197/200 query, 14 methods, source-ID recovery; 163 source thuộc train cũ.
- Topic: LDA 20, BERTopic baseline 92 và chunked 84; noise/coherence đánh đổi.
- 237 tests là kết quả JUnit **đã lưu** của phiên trước, không được gọi là đã chạy lại trong phiên biên soạn.
- Qdrant/SE1/SE2/SE3 là lịch sử; app hiện tại dùng Search v2 pretrained + reranker.

Không có số liệu giả định hoặc API key trong nguồn báo cáo. Các bảng lỗi là chẩn đoán AI, chưa phải xác nhận nguyên nhân bởi chuyên gia. PDF có thể sử dụng làm bản thảo đồ án đầy đủ; việc nộp còn cần thông tin bìa, quy định trường và rà soát học thuật của tác giả/giảng viên.
