# Đính chính về tính liền kề của topic keywords - 24/10/2026

Trong lần rà soát huấn luyện full-document, phát hiện một giới hạn ở pipeline
representation ngày 23/10: `ContiguousScientificVectorizer` giữ đúng tính liền
kề của **chuỗi token đầu vào**, nhưng `lda_tokens.parquet` đã xóa stopword ở
bước trước. Vì vậy, kiểm thử vectorizer với chuỗi còn “and” chưa chứng minh
ngram thực sự liền kề trong Title–Abstract nguồn. Đây là thiếu sót của kết luận
trong báo cáo trước; không có nghĩa mọi keyword trước đều sai hoặc giả.

Nhánh mới tạo `scientific_token_stream.parquet` trực tiếp từ Title và Abstract
đã lưu, giữ ranh giới khi loại stopword, gặp dấu câu hoặc đổi trường. Công thức
và domain phrase có thật được bảo vệ, từ tiếng Anh được lemmatize. Tài liệu,
tokens LDA, vectors và model lịch sử không bị ghi đè. Vocabulary và trọng số
topic vẫn chỉ fit trên train; validation chọn cấu hình.

Kiểm thử nguồn có cả “hard and carbon”, “hard. Carbon”, “hard / carbon” và
Title=“hard”, Abstract=“carbon”: không trường hợp nào được tạo bigram
“hard carbon”. Phrase thật “hard carbon”, “hard-carbon” và công thức
`Na3V2(PO4)3` vẫn được giữ theo quy tắc đã cấu hình.

Audit `historical_keyword_adjacency_audit.csv` đối chiếu support của các
keywords cũ với representation mới. Chênh lệch document frequency còn có thể
do các bước chuẩn hóa lịch sử; không quy tất cả chênh lệch cho phrase giả.
Coherence model cũ/mới được tính cùng reference mới và báo coverage, không
so trực tiếp số C_v này với các reference khác trong lịch sử.

Audit thực tế: 100 vị trí top10 trước chứa 80 keywords khác nhau; cả 80 đều
có support trong nguồn. Không có keyword hai từ phân cách bằng khoảng trắng
trong danh sách cuối cũ; các phrase bảo vệ có dạng underscore. Vì vậy, đây là
sửa cơ chế sinh candidates và giới hạn của bằng chứng trước, không phải kết
luận danh sách top10 cũ đã bị xác nhận có phrase giả.

Đây là sửa representation, độc lập với thay đổi trọng số neural. Không sử dụng
nhãn test hoặc tự thêm từ khóa khoa học không có bằng chứng trong văn bản.
