# Phần báo cáo riêng: Search engine cho patents và papers

Tài liệu được tách riêng từ phạm vi báo cáo Sodium-ion Research Explorer, dựa trên kế hoạch trong `docs/KE_HOACH_HOAN_THIEN_SEARCH_ENGINE_PATENTS_PAPERS.md`. Nội dung gồm bài toán, bốn hướng tiếp cận, pipeline, công thức, cách train encoder liên quan, đánh giá, kết quả, ba ví dụ Q01/Q16/Q20 và kế hoạch bổ sung các phần còn thiếu.

**Font chính: Times New Roman, cỡ 13 TeX pt; bảng và chú thích hình cũng cỡ 13.** Tiêu đề lớn hơn để phân cấp; sơ đồ và dòng nguồn dùng cỡ nhỏ hơn. Nguồn hình bitmap được giữ đúng dữ liệu đã lưu, không phải toàn bộ nhãn trong hình đều dùng cùng font với văn bản báo cáo. LaTeX quy đổi 13 TeX pt thành xấp xỉ 12,951 PDF points; đây vẫn là cỡ 13 trong tài liệu.

## File cần dùng

- `SEARCH_ENGINE_PATENTS_PAPERS.pdf`: bản PDF cuối đã biên dịch.
- `main.tex`: nguồn độc lập, thiết lập font, cỡ chữ và bố cục.
- `search_engine_section.tex`: phần nội dung để ghép vào bài lớn.
- `diagrams/`: ba sơ đồ TikZ có thể chỉnh sửa, gồm offline, online và evaluation.
- `figures/`: bảy hình kết quả; có hình train, so sánh phương pháp, query heatmap, sensitivity, reranker và Q01/Q16.
- `tables/`: bảng được sinh từ artifacts gốc; không nhập lại số bằng tay.
- `figure_sources.tex`, `evidence_manifest.json`: nguồn hình và hash bằng chứng.
- `references.bib`: tài liệu tham khảo.
- `validate_section.py`, `validation.json`, `qa/`: kiểm tra PDF, font, số liệu và bố cục.

## Biên dịch độc lập

Chọn **XeLaTeX**, main document `main.tex`. Máy cần cài Times New Roman, không tự đổi sang font tương tự. Trên Overleaf, nếu môi trường không có Times New Roman, cần bổ sung bộ font được phép sử dụng và cấu hình đường dẫn font phù hợp.

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Trong workspace hiện tại có thể chạy từ thư mục này:

```powershell
D:\daln\.tools\tectonic\tectonic.exe --only-cached --keep-logs --keep-intermediates --outdir build main.tex
```

Khi giải nén nguồn, tạo thư mục `build` nếu chưa có. Tectonic cần tải gói/font lần đầu nếu cache chưa đủ.

## Ghép vào báo cáo lớn

Sao chép thư mục `search_engine_patents_papers` vào dự án LaTeX, nạp các gói trong preamble của `main.tex` theo nhu cầu, rồi chèn:

```latex
\begingroup
\def\SearchAssetRoot{search_engine_patents_papers}
\input{search_engine_patents_papers/search_engine_section.tex}
\endgroup
```

Các label có tiền tố `se:` để hạn chế trùng với bài lớn. Gộp các BibTeX entries được dùng vào bibliography của bài; không chèn thêm toàn bộ `main.tex`. Font và cỡ chữ khi ghép được quản lý bởi tài liệu mẹ, nên đặt Times New Roman 13 ở preamble của bài lớn.

## Tái dựng và phạm vi kết luận

Từ gốc workspace:

```powershell
.\.venv\Scripts\python.exe -X utf8 output\search_engine_patents_papers\build_assets.py
```

Script cần corpus/output gốc trong workspace để tạo lại bảng và hình; biên dịch từ nguồn đã đóng gói không cần model, Qdrant hay API. Sau khi biên dịch:

```powershell
.\.venv\Scripts\python.exe -X utf8 output\search_engine_patents_papers\validate_section.py
```

Các metrics relevance dựa trên703 AI Silver Labels,20 queries; chưa là human ground truth. Đánh giá nguồn riêng, p95/concurrency và các thí nghiệm mới được ghi là phần cần bổ sung. Bản biên soạn này không train, encode, chấm nhãn hoặc chạy benchmark mới.
