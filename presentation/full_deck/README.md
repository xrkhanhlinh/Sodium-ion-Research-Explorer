# Sodium-ion Research Explorer: bộ slide LaTeX đầy đủ

Mở **main.pdf** để trình chiếu. Chỉnh sửa **main.tex** để sửa slide trực tiếp.

Bộ đầy đủ có **122 trang**, gồm **64 slide chính** và **58 trang phụ lục**, với **19 hình gốc từ outputs**. Phần phụ lục giữ toàn bộ **20 topic LDA, 92 topic BERTopic baseline và 84 topic BERTopic chunked**, cùng hai hàng noise riêng.

## Nội dung và cách sử dụng

| Trang | Nội dung |
|---|---|
| 1–14 | Bài toán, pipeline, dữ liệu và tiền xử lý |
| 15–20 | LDA, cập nhật phân phối và kết quả topics |
| 21–30 | BGE, contrastive loss, AdamW và kết quả fine-tuning |
| 31–40 | BERTopic, UMAP/HDBSCAN, chunking và đầu ra |
| 41–52 | Xây Search V2, mười methods, query và kết quả Q19 |
| 53–64 | Rubric, metrics, kết quả đánh giá, tối ưu và hạn chế |
| 65–78 | Công thức bổ sung và câu hỏi bảo vệ |
| 79–113 | Toàn bộ topic IDs, Top5 từ khóa và noise |
| 114–122 | Keyword heatmap, UMAP và Top10 từ khóa chunked |

- Slide chính đi theo thứ tự bài toán, dữ liệu, tiền xử lý, LDA, BGE, fine-tuning, BERTopic, chunking, Search V2, đánh giá, kết quả, hạn chế và hướng cải thiện.
- Các slide về trọng số có loss đối xứng, gradient clipping, AdamW, scheduler và bằng chứng cập nhật trọng số thật. LDA có phần diễn giải cập nhật variational inference.
- Kết quả mô hình, tần suất từ, topic examples, Top5 Q19 và passage lấy từ artifacts đã lưu. Ví dụ tính tay ở phụ lục được ghi rõ là minh họa.
- Phụ lục là tài liệu tra cứu, không cần trình bày tuần tự toàn bộ trong một buổi. Khi cần rút gọn, ưu tiên dữ liệu, pipeline, kết quả và search trước các bảng liệt kê mọi topic.
- `speaker_notes.md` giải thích thêm theo số slide, có giới hạn diễn giải và câu hỏi bảo vệ.
- `topic_catalog.csv` chứa đủ Top10 từ khóa, model, topic ID, số tài liệu và nguồn. Slides liệt kê toàn bộ IDs kèm Top5 để giữ cỡ chữ đọc được.
- `slide_inventory.json` chứa nội dung từng slide để tra cứu hoặc biên tập bằng script.
- `evidence_manifest.json` ghi SHA-256 của các nguồn sử dụng.

## Biên dịch trên máy hoặc Overleaf

Chọn **XeLaTeX**, không dùng pdfLaTeX. Các ảnh nằm trong `figures/`, đường dẫn tương đối nên không cần corpus hay model để biên dịch `main.tex`.

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Preamble ưu tiên Arial và Consolas trên Windows, tự dùng TeX Gyre Heros và Latin Modern Mono nếu không có các font đó. Khi đổi font, cần kiểm tra lại bố cục.

Từ gốc dự án `D:/daln`, dùng Tectonic đã có:

```powershell
.tools/tectonic/tectonic.exe --only-cached --keep-logs --keep-intermediates --outdir presentation/full_deck presentation/full_deck/main.tex
```

## Tái tạo từ artifacts dự án

Hai lệnh dưới chỉ cần khi muốn dựng lại nguồn và kiểm tra PDF từ repository đầy đủ. Việc biên tập `main.tex` thủ công không yêu cầu chạy generator. Chạy generator sẽ thay các file do script sinh trong thư mục này.

```powershell
.venv/Scripts/python.exe presentation/full_deck/build_deck.py
.venv/Scripts/python.exe presentation/full_deck/validate_deck.py
```

`build_deck.py` đọc outputs/code/config, tạo nguồn, sao chép hình và xuất catalog. Nó không train, không encode, không chạy query và không gọi API. Việc chạy validator phải sau bước biên dịch.

## Phạm vi kết luận

Search mặc định là BM25 + BGE pretrained, hợp nhất bằng RRF rồi MiniLM reranker. Không cộng trực tiếp BM25, cosine và reranker. Không có LLM sinh câu trả lời trong luồng search hiện tại.

Fine-tuning cải thiện proxy Title–Abstract đã đo. Relevance search độc lập vẫn chưa hoàn tất. AI relevance chỉ đủ pool ở 2/23 queries. Source recovery có 163/197 tài liệu nguồn thuộc train cũ, vì vậy không đại diện kiểm tra khái quát độc lập.

Ngày biên soạn 20/10/2026 không phải ngày huấn luyện lại. Số tests và timing trên slides là kết quả lịch sử có nguồn, không phải thí nghiệm mới của việc tạo slide.
