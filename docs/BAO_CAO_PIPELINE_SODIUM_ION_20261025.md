# Báo cáo pipeline khám phá chủ đề và tìm kiếm tài liệu pin sodium ion

Sodium ion Research Explorer

Tổng hợp mã nguồn và kết quả thực nghiệm đã lưu trong D:\daln
Ngày biên soạn 25 tháng 10 năm 2026

Báo cáo trình bày toàn bộ đường đi từ dữ liệu Patent và Publication đến LDA, BERTopic, các biến thể BatteryBERT và kết quả Search Top 10. Mỗi phần liên kết phương pháp với bảng output, hình kết quả và giới hạn diễn giải.

| Dữ liệu hiện có | Đầu ra topic | Đầu ra Search |
|---|---|---|
| 13.702 Title–Abstract | 6 catalog và 60 topic; 600 vị trí từ khóa | 20 query benchmark; 703 cặp chờ nhãn người |

### Kết quả chính

LDA Stage2 chọn K = 20; bảng đối chứng dùng LDA K10. BERTopic–BatteryBERT pretrained có coherence lịch sử cao nhưng hai cụm chiếm 92,8% corpus. Nhánh full-document mới đọc đủ Abstract, cải thiện proxy ghép Title–document và tạo catalog KMeans10 đang được ứng dụng sử dụng.

Test proxy của continuation tăng từ 0,887839 lên 0,924704. Kết quả topic có đánh đổi: coherence, diversity và ARI giảm so với bản optimized trước. Chưa có nhãn relevance độc lập để tuyên bố model Search nào tốt nhất.

Phạm vi dữ liệu là Title và Abstract, chưa có toàn văn bài báo. Các số đo dưới đây được đọc từ output hiện có; báo cáo này không huấn luyện lại model.

## Nội dung báo cáo

Báo cáo đi theo đường xử lý dữ liệu, giải thích từng họ mô hình, trình bày output và đánh giá sau khi sinh kết quả. Phụ lục giữ đủ catalog để người đọc kiểm tra lại từng topic và danh sách từ khóa.

| Nội dung | Mục | Trang |
|---|---|---|
| Pipeline tổng thể | 1 | 3 |
| Dữ liệu và thiết kế thí nghiệm | 2 đến 3 | 4 đến 5 |
| LDA và kết quả chọn số topic | 4 đến 5 | 6 đến 7 |
| BGE và BatteryBERT | 6 | 8 |
| BERTopic và sáu catalog đối chứng | 7 đến 9 | 9 đến 11 |
| KMeans và huấn luyện full document | 10 đến 13 | 12 đến 15 |
| Output topic và đánh giá Top 10 keywords | 14 đến 17 | 16 đến 19 |
| Search Top 10 và đánh giá relevance | 18 đến 21 | 20 đến 23 |
| Thảo luận và đối chiếu nguồn output | 22 đến 23 | 24 đến 25 |
| Đủ 60 topic và 600 vị trí keywords | Phụ lục A đến F | 26 đến 31 |

### Cách đọc các kết quả

Các bảng có chữ lịch sử mô tả artifacts của lượt chạy trước. Các bảng full-document mô tả lượt continuation ngày 24/10. Những thước đo khác protocol được trình bày riêng để tránh so sánh sai phạm vi.

Ba loại Top 10 có ý nghĩa khác nhau: 10 topic là số nhóm của catalog; Top 10 keywords mô tả từng topic; Top 10 bài là kết quả truy hồi cho một query. Coherence đánh giá từ khóa, còn relevance có nhãn đánh giá kết quả tìm kiếm.

Biểu đồ trong báo cáo được dựng lại trực tiếp từ output CSV và JSON đã lưu. Các công thức được render thành hình để giữ đúng ký hiệu trong PDF; bản Markdown đi kèm giữ biểu thức và đường dẫn ảnh để có thể chỉnh sửa.

## 1 Bài toán và pipeline tổng thể

Mục tiêu thứ nhất là tổ chức tài liệu thành các nhóm chủ đề để người dùng khảo sát hướng nghiên cứu. Mục tiêu thứ hai là trả về các bài liên quan tới truy vấn. Hai mục tiêu chia sẻ dữ liệu và encoder nhưng sử dụng tiêu chí đánh giá khác nhau.

![Hình 1. Pipeline từ dữ liệu đến hai đầu ra topic và Search. Sơ đồ được dựng từ các module hiện có.](../outputs/report_pipeline_20261025/figures/01_pipeline.png)
Hình 1. Pipeline từ dữ liệu đến hai đầu ra topic và Search. Sơ đồ được dựng từ các module hiện có.
Nguồn output: src/pipeline.py; src/model_comparison.py; src/final_search.py

Huấn luyện, encode corpus và xây chỉ mục chạy offline. Khi khởi động, ứng dụng đọc artifacts đã chuẩn bị. Lúc tìm kiếm, hệ thống chỉ xử lý query, áp dụng bộ lọc, truy hồi và hợp nhất thứ hạng. Topic hỗ trợ duyệt và lọc, không được cộng như điểm thưởng vào ranking.

| Giai đoạn | Input | Output cụ thể |
|---|---|---|
| Chuẩn bị dữ liệu | Excel Patent và Publication | main_corpus.parquet; lda_tokens.parquet; bert_texts.parquet |
| Topic lexical | Tokens và BoW | LDA posterior; keywords; representatives |
| Topic neural | Embeddings BGE hoặc BatteryBERT | Assignments; catalog topic; diagnostics |
| Search | Query và bộ lọc | Top 10 bài; cosine; RRF; thời gian |

## 2 Dữ liệu và sàng lọc chất lượng

Hai tập nguồn có tổng cộng 20.576 bản ghi. Clean master giữ provenance của toàn bộ đầu vào; corpus chính dùng 13.702 tài liệu sau các quy tắc sàng lọc. Nhãn CORE và RELATED là heuristic, không phải đánh giá khoa học độc lập.

| Nguồn | Đầu vào | Corpus chính | Giữ lại |
|---|---|---|---|
| Patent | 8.657 | 5.933 | 68,53% |
| Publication | 11.919 | 7.769 | 65,18% |

![Hình 2. Số tài liệu trước và sau sàng lọc và phân chia dữ liệu cho nhánh neural mới.](../outputs/report_pipeline_20261025/figures/02_data.png)
Hình 2. Số tài liệu trước và sau sàng lọc và phân chia dữ liệu cho nhánh neural mới.
Nguồn output: outputs/reports/stage1_summary.json; outputs/batterybert_full_document_20261024_v1/text_length_summary.csv

| Disposition | Số bản ghi | Diễn giải |
|---|---|---|
| MAIN_CORPUS | 13.702 | Dùng trong corpus chính |
| RELEVANCE_REVIEW | 6.596 | Cần rà soát mức liên quan |
| OUT_OF_SCOPE | 254 | Ngoài phạm vi theo heuristic |
| LANGUAGE_REVIEW | 15 | Cần kiểm tra ngôn ngữ |
| RETRACTED / DUPLICATE / TEXT | 9 | 4 thu hồi; 3 trùng xác nhận; 2 chất lượng văn bản |

Quy trình gồm sửa cấu trúc bảng, chuẩn hóa metadata và văn bản, kiểm tra ngôn ngữ, thu hồi, trùng lặp và phạm vi. Giữ doc_id, nguồn, năm, identifier và split_group_id; không tạo Abstract giả cho bản ghi thiếu.

## 3 Biểu diễn văn bản và thiết kế thí nghiệm

Nhánh lexical dùng chuỗi token đã xử lý và Dictionary chung. LDA sử dụng số đếm Bag-of-Words. Nhánh neural giữ văn bản để tokenizer của encoder xử lý; không đưa chuỗi đã bỏ toàn bộ ngữ cảnh vào Transformer.

| Dữ liệu | Số tài liệu | Vai trò |
|---|---|---|
| Train | 10.921 | Optimizer; fit geometry, KMeans và vocabulary của nhánh inductive mới |
| Validation | 1.361 | Chọn learning rate, checkpoint và recipe topic |
| Test | 1.368 | Đánh giá sau khi chọn; không đưa vào optimizer |
| Title only | 52 | Chỉ inference; không tham gia contrastive training |

Grouped split giữ các tài liệu cùng split_group_id trong một tập nhằm hạn chế rò rỉ giữa train, validation và test. Việc này không chứng minh đã nhận diện mọi trường hợp trùng ý hoặc mọi patent family.

### Phân biệt hai phạm vi đánh giá

LDA và BERTopic đối chứng lịch sử là khám phá trên corpus đã đóng băng; coherence toàn corpus mô tả cấu trúc quan sát được. Nhánh optimized và full-document mới fit geometry, KMeans và từ vựng trên train, dùng validation để chọn và test để đánh giá. Hai giao thức không nên bị gọi chung là một phép thử held-out.

| Loại output | Ý nghĩa điểm | Không suy ra được |
|---|---|---|
| LDA posterior | Tỷ trọng topic theo mô hình xác suất | Độ đúng khoa học của tên topic |
| BERTopic affinity | Cosine BoW với c-TF-IDF, chuẩn hóa L1 | Posterior đã hiệu chuẩn |
| Dense cosine / RRF | Điểm dùng để xếp hạng tài liệu | Xác suất bài trả lời đúng query |
| Proxy Title → document | Khả năng tìm lại cặp hoặc nhóm tài liệu | Relevance query tự nhiên độc lập |

Nguồn output: configs/stage2.yaml; configs/stage3.yaml; configs/model_comparison.yaml; outputs/batterybert_full_document_20261024_v1/training_protocol.json

## 4 LDA và bản chất mô hình chủ đề xác suất

LDA giả định mỗi tài liệu là hỗn hợp nhiều topic và mỗi topic là một phân bố từ. Dữ liệu quan sát là các từ; biến ẩn là topic của từng vị trí từ và tỷ trọng topic trong tài liệu. Nền tảng phương pháp xem tài liệu tham khảo [1].

$$
\theta_d\sim Dir(\alpha),\quad\phi_k\sim Dir(\eta),\quad P(w\mid d)=\sum_k\theta_{dk}\phi_{kw}
$$

θ_dk là tỷ trọng topic k trong tài liệu d; φ_kw là xác suất từ w trong topic k. α và η là prior Dirichlet. Với từng vị trí từ, LDA giả định chọn topic từ θ_d rồi chọn từ từ φ của topic đó. Gensim suy luận các biến ẩn bằng phương pháp biến phân.

### Pipeline LDA trong dự án

Tokens Stage1 → Dictionary không lọc thêm → BoW → fit LDA → posterior tài liệu–topic → lấy argmax để hiển thị một nhãn chính → lấy 10 từ có P(w|topic) cao nhất → chọn tài liệu đại diện có posterior cao.

| Tham số | Giá trị | Tác động |
|---|---|---|
| Số topic K | 10, 15, 20, 25, 30 | Kiểm tra nhiều mức chi tiết |
| passes / iterations | 10 / 100 | Ngân sách cập nhật và suy luận |
| alpha / eta | asymmetric / symmetric | Prior tài liệu–topic và topic–word |
| Seeds | 42; thêm 123 và 2026 | Đo ổn định ở hai cấu hình triển vọng |
| Top keywords / C_v window | 10 / 110 token | Cố định đầu ra và protocol coherence |

BoW bỏ thứ tự từ, nên LDA khai thác đồng xuất hiện và phân bố từ thay vì quan hệ ngữ cảnh đầy đủ. Một tài liệu có thể thuộc nhiều chủ đề trong posterior, dù giao diện hiển thị topic trội nhất. Tên chủ đề là bước diễn giải sau mô hình.

Nguồn output: src/stage2.py; configs/stage2.yaml; src/model_comparison.py

## 5 Kết quả LDA và lựa chọn số topic

![Hình 3. Kết quả năm cấu hình LDA seed 42. Chấm cam đánh dấu K20 được chọn ở Stage2.](../outputs/report_pipeline_20261025/figures/03_lda_selection.png)
Hình 3. Kết quả năm cấu hình LDA seed 42. Chấm cam đánh dấu K20 được chọn ở Stage2.
Nguồn output: outputs/tables/stage2/lda_base_metrics.csv

| K | C_v | U_Mass | Diversity |
|---|---|---|---|
| 10 | 0,5379 | -1,5323 | 0,6600 |
| 15 | 0,5580 | -2,0147 | 0,7533 |
| 20 | 0,5716 | -1,9372 | 0,7400 |
| 25 | 0,5376 | -2,6434 | 0,7880 |
| 30 | 0,5302 | -3,0826 | 0,7967 |

Stage2 chọn K20, seed 42, với C_v = 0,5716, diversity = 0,7400 và stability Jaccard sau ghép Hungarian = 0,2620. Stability này khác định nghĩa ARI của KMeans và không được so trực tiếp như cùng một chỉ số.

Quy tắc chọn: sàng lọc hai K theo C_v ở seed cơ sở, sau đó kết hợp C_v chuẩn hóa 50%, U_Mass chuẩn hóa 15%, diversity 15% và stability 20%. Đây là lựa chọn trong lưới đã thử, không phải tối ưu toàn cục.

Catalog LDA K10 trong phần so sánh giữ K bằng các nhánh khác. K10 có C_v = 0,537876 ở protocol lịch sử; khác kết quả K20 Stage2. Trong K10, hai topic lớn nhất có 3.559 và 3.486 tài liệu, cho thấy các nhóm tương đối rộng.

Nguồn output: outputs/reports/stage2/selected_model.json; outputs/tables/stage2/lda_comparison.csv; docs/TOP10_TOPICS_ALL_MODELS_20261024.csv

## 6 BGE và BatteryBERT trong pipeline neural

BGE và BatteryBERT là encoder; BERTopic và KMeans là các cách khai thác embedding để tạo topic. BGE được dùng cho biểu diễn và truy hồi; BatteryBERT đưa tri thức ngôn ngữ miền pin vào encoder. Checkpoint gốc được ghi tại [3] và [4].

$$
Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V
$$

Attention tạo hidden state phụ thuộc ngữ cảnh. Q và K xác định trọng số tương tác; V cung cấp thông tin được tổng hợp. Vector tài liệu được tạo từ hidden state qua pooling và chuẩn hóa. Encoder tốt cho dự đoán ngôn ngữ miền pin chưa tự đảm bảo khoảng cách tài liệu phù hợp với mọi mục tiêu phân cụm.

| Nhánh | Đặc điểm thực thi | Output |
|---|---|---|
| BGE Stage3 | bge-base-en-v1.5; cập nhật block cuối; proxy độ dài 128 | Checkpoint và embeddings; BERTopic lịch sử |
| BatteryBERT pretrained | batterybert-cased; chưa cập nhật theo corpus dự án | Embeddings chunked và catalog đối chứng |
| BatteryBERT fine-tuned cũ | Title–Abstract contrastive; train tối đa 128 token | Checkpoint dùng cho optimization 23/10 |
| BatteryBERT full-document | Continuation block cuối; tất cả chunks Abstract | Checkpoint và 13.702 embeddings mới |

### Kết quả proxy BGE lịch sử

| Phép thử | Pretrained | Fine-tuned |
|---|---|---|
| Test Group MRR@10 | 0,893567 | 0,928083 |
| Queries / candidates | 1.368 / 1.368 | 1.368 / 1.368 |
| Độ dài proxy | 128 token | 128 token |

Đây là proxy Title → Abstract trong cùng split lịch sử. Không so trực tiếp điểm BGE 128 token với proxy full-document có Title context để kết luận encoder nào tốt hơn về Search.

Nguồn output: outputs/reports/stage3/pretrained_test_metrics.json; outputs/reports/stage3/finetuned_test_metrics.json

## 7 BERTopic từ embedding đến topic và từ khóa

BERTopic kết hợp biểu diễn ngữ nghĩa và mô tả từ vựng. Pipeline sử dụng embeddings đã encode trước, UMAP giảm chiều, HDBSCAN tìm cụm theo mật độ, rồi c-TF-IDF để mô tả từ khóa [2, 5].

| Bước | Cấu hình thực tế | Ý nghĩa |
|---|---|---|
| Embedding | BGE hoặc BatteryBERT; chunked khi tương ứng | Vị trí tài liệu trong không gian vector |
| UMAP | 15 neighbors; 5 chiều; min_dist 0; cosine; seed 42 | Tạo không gian để phân cụm |
| HDBSCAN | min_cluster_size 30; min_samples 10; eom | Tìm cụm và outlier −1 |
| c-TF-IDF | Frozen vocabulary; reduce_frequent_words=True | Xếp hạng từ khóa trong cụm |
| Gộp topic | Gộp về 10 nếu ban đầu có nhiều hơn | Giữ nguyên outlier để so sánh K10 |

$$
w_{c,t}=\sqrt{n_{c,t}/\sum_v n_{c,v}}\log(1+A/f_t)
$$

n_ct là số lần từ t xuất hiện trong cụm c; A là trung bình tổng token của các tài liệu cấp cụm; f_t là tổng tần suất từ qua các cụm. Căn bậc hai của TF giảm sự áp đảo của từ quá thường xuyên. Đây là công thức tương ứng nhánh c-TF-IDF mặc định có reduce_frequent_words, không áp dụng thay cho công thức keywords của nhánh optimized.

Trong bản xuất so sánh, ma trận document–topic của BERTopic là cosine affinity giữa BoW và c-TF-IDF, chuẩn hóa L1. Nó không phải posterior xác suất như LDA. Assignments và outlier lấy từ clustering đã lưu, không suy lại bằng argmax affinity.

Hình UMAP hai chiều phục vụ hiển thị. Clustering được fit trong không gian UMAP năm chiều; khoảng cách và độ tách cụm trên hình hai chiều chỉ có tính mô tả.

Nguồn output: src/model_comparison.py; src/stage3_bertopic_model.py; configs/model_comparison.yaml

## 8 Kết quả BERTopic đối chứng lịch sử

![Hình 4. So sánh lịch sử LDA K10 và hai nhánh BERTopic. C_v và diversity cao không đủ để xác nhận chất lượng khoa học.](../outputs/report_pipeline_20261025/figures/05_historical_comparison.png)
Hình 4. So sánh lịch sử LDA K10 và hai nhánh BERTopic. C_v và diversity cao không đủ để xác nhận chất lượng khoa học.
Nguồn output: outputs/final_system/topic_comparison.csv

| Model | C_v | Diversity | Outlier |
|---|---|---|---|
| LDA K10 | 0,5379 | 0,6600 | N/A |
| BERTopic–BGE | 0,6455 | 0,8600 | 22,77% |
| BERTopic–BatteryBERT pretrained | 0,7445 | 0,9200 | 0,80% |

BERTopic–BGE có các nhóm Prussian blue, điện ly, TiO2/titanate và MXene, nhưng 3.120 tài liệu là outlier. Tỷ lệ này mô tả phần corpus không được đưa vào topic, không tự chứng minh tất cả outlier là dữ liệu kém.

BERTopic–BatteryBERT pretrained có 109 outlier và C_v cao nhất trong ba nhánh lịch sử. Tuy nhiên, hai cụm lớn nhất gồm 7.261 và 5.449 tài liệu, tổng 92,76% corpus. Chúng được diễn giải là nhóm publication về hiệu năng và nhóm patent về phương pháp chế tạo. Đây là dấu hiệu các nhóm lớn có thể bị chi phối bởi nguồn hoặc cách viết.

Catalog cũng còn nhóm sodium cobaltate và siêu dẫn. Cần rà soát phạm vi corpus và tài liệu đại diện trước khi gọi đây là các hướng nghiên cứu pin sodium ion đã được xác nhận.

Nhánh BERTopic–BatteryBERT fine-tuned có trong code nhưng báo cáo model_comparison lịch sử ghi not_completed tại thời điểm xuất. Không dùng kết quả BatteryBERT KMeans để thay thế cho nhánh BERTopic chưa có kết quả hoàn tất.

Nguồn output: docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; outputs/model_comparison_v2/reports/MODEL_COMPARISON_REPORT.md

## 9 Sáu catalog và cấu trúc phân bố topic

![Hình 5. Số tài liệu theo topic ở sáu catalog. Cùng thang tung giúp thấy sự tập trung ở BERTopic–BatteryBERT.](../outputs/report_pipeline_20261025/figures/04_six_distributions.png)
Hình 5. Số tài liệu theo topic ở sáu catalog. Cùng thang tung giúp thấy sự tập trung ở BERTopic–BatteryBERT.
Nguồn output: docs/TOP10_TOPICS_ALL_MODELS_20261024.csv

| Catalog | Cụm lớn nhất | Hai cụm / corpus | Outlier |
|---|---|---|---|
| Full-document trained | 1.717 | 24,59% | 0 |
| Fine-tuned optimized | 2.063 | 27,49% | 0 |
| Pretrained KMeans | 2.309 | 29,09% | 0 |
| LDA K10 | 3.559 | 51,42% | 0 |
| BERTopic–BGE | 3.552 | 45,26% | 3.120 |
| BERTopic–BatteryBERT | 7.261 | 92,76% | 109 |

Topic ID chỉ có nghĩa trong từng catalog. Topic 0 của LDA không tương ứng topic 0 của BatteryBERT. Bảng 10 topic ở phụ lục liệt kê toàn bộ nhóm của mỗi catalog theo ID, không xếp hạng 10 chủ đề tốt nhất.

KMeans gán mọi tài liệu vào một trong K cụm, vì vậy outlier bằng 0 là đặc điểm thuật toán. Với LDA, tài liệu vẫn có posterior hỗn hợp dù bảng catalog chỉ đếm topic trội nhất.

## 10 Hiệu chỉnh hình học và BatteryBERT KMeans

Nhánh optimized kiểm tra giả thuyết embedding bị chi phối bởi nguồn. Phép source projection fit trên train loại thành phần nằm trên hướng chênh lệch trung bình hai nguồn rồi chuẩn hóa vector.

$$
u=(\mu_1-\mu_2)/\|\mu_1-\mu_2\|,\quad z=norm[(x-\mu)-((x-\mu)^Tu)u]
$$

μ là trung bình toàn bộ train; μ1 và μ2 là trung bình theo nguồn. Phép biến đổi chỉ loại một hướng tuyến tính; hướng này có thể chứa cả phong cách viết lẫn nội dung có ích. Do đó cần đánh giá tác động thay vì mặc định đây là khử toàn bộ thiên lệch.

$$
\min_{a,\nu}\sum_i\|z_i-\nu_{a_i}\|^2
$$

KMeans tìm 10 tâm cụm sao cho tổng bình phương khoảng cách từ mỗi tài liệu đến tâm được gán nhỏ nhất. K = 10 là cấu hình tổ chức catalog, không phải kết luận lĩnh vực có đúng 10 chủ đề tự nhiên.

![Hình 6. Các recipe center/source của nhánh full-document trên validation. Điểm NPMI cao vẫn có thể trượt gate lệch nguồn.](../outputs/report_pipeline_20261025/figures/12_validation_tradeoff.png)
Hình 6. Các recipe center/source của nhánh full-document trên validation. Điểm NPMI cao vẫn có thể trượt gate lệch nguồn.
Nguồn output: outputs/batterybert_full_document_20261024_v1/topics/validation_comparison.csv

Có 12 recipe: hai geometry × ba mức exclusivity × hai mức phrase boost. Cấu hình được chọn là source_ex0.25_pb1.0. Các cấu hình center có source Cramér V khoảng 0,988 trên validation; vì vậy không được chọn chỉ vì NPMI cao.

## 11 Huấn luyện BatteryBERT đọc đủ Abstract

Nhánh ngày 24/10 tiếp tục train từ checkpoint fine-tuned đã có. Mục tiêu là mọi chunk Abstract đóng góp vào biểu diễn tài liệu và gradient của block cuối, thay cho train chỉ trên phần đầu 128 token.

| Thành phần | Cấu hình |
|---|---|
| Cửa sổ / overlap | 256 / 64 token; ngân sách Abstract phụ thuộc Title và special tokens |
| Block cập nhật | Block cuối; 11 block đầu đóng băng, cache float32 |
| Batch loss / microbatch chunk | 32 cặp / 2 chunks; gradient caching hai tầng |
| Temperature / distillation | 0,05 / 0,02 |
| Learning rate được chọn | 5e-6; pilot thêm 2e-6 |
| Epoch / optimizer | Một lượt train; AdamW; warmup/decay; clipping norm 1 |

$$
c_j=norm(\sum_t m_{jt}h_{jt}/\sum_t m_{jt}),\quad d=norm(\sum_j w_jc_j/\sum_jw_j)
$$

m_jt chọn token Abstract và loại SEP; Title vẫn là context trong encoder nhưng không trực tiếp vào mean pooling của chunk Abstract. w_j là số token mới được bao phủ. Overlap vẫn ảnh hưởng hidden state, vì vậy weighting này không tương đương mỗi token chỉ ảnh hưởng đúng một lần.

$$
S_{ij}=q_i^Td_j/\tau,\quad L_{q\to d}=-(1/B)\sum_i\log[\exp(S_{ii})/\sum_{j\in A_i}\exp(S_{ij})]
$$

Loss lấy trung bình hai chiều Title → document và document → Title, có mask false negatives cùng group hoặc title rất giống; thêm khoảng cách bình phương tới embedding tham chiếu. Gradient caching tính loss chung của batch rồi replay từng nhóm chunk, giữ tập negatives lớn với RAM giới hạn [6]. Dropout tắt để replay nhất quán.

Nguồn output: configs/batterybert_full_document.yaml; src/batterybert_full_document.py

## 12 Output huấn luyện và lựa chọn checkpoint

![Hình 7. Validation Group MRR@10 theo optimizer updates của hai learning rate. Đây là proxy Title → document có Title context.](../outputs/report_pipeline_20261025/figures/06_learning_curve.png)
Hình 7. Validation Group MRR@10 theo optimizer updates của hai learning rate. Đây là proxy Title → document có Title context.
Nguồn output: outputs/batterybert_full_document_20261024_v1/all_validation_checkpoints.csv

| Output thực thi | Giá trị |
|---|---|
| Tài liệu train đã tham gia | 10.921 |
| Chunk đã backpropagate | 26.147 |
| Token Abstract duy nhất theo tài liệu | 3.420.215 |
| Số lần val và test vào optimizer | 0 |
| Checkpoint được chọn | Step 342; learning rate 5e-06 |
| Toàn corpus được encode lại | 13.702 vector tài liệu |

Checkpoint được chọn sau 342 optimizer updates và đã thấy toàn bộ 10.921 tài liệu của lượt continuation. Validation chọn learning rate và checkpoint; test không dùng để chọn. Test vẫn là split lịch sử đã được sử dụng trước đó, không phải external test mới.

Các output chứng minh độ bao phủ và thay đổi trọng số thật: training_coverage.json, training_document_coverage.csv, selection.json và embedding_success.json. Kiểm thử kỹ thuật kiểm tra gradient cache, ảnh hưởng phần cuối văn bản và parity giữa cache với full forward.

Nguồn output: outputs/batterybert_full_document_20261024_v1/training_coverage.json; outputs/batterybert_full_document_20261024_v1/selection.json; outputs/batterybert_full_document_20261024_v1/embedding_success.json

## 13 Hiệu quả continuation trên proxy cố định

![Hình 8. Baseline và checkpoint mới trên cùng full-document pooling; khoảng bootstrap dùng split_group_id.](../outputs/report_pipeline_20261025/figures/07_training_effect.png)
Hình 8. Baseline và checkpoint mới trên cùng full-document pooling; khoảng bootstrap dùng split_group_id.
Nguồn output: outputs/batterybert_full_document_20261024_v1/paired_training_effect.json

| Split | Baseline | Sau train | Gain | Khoảng 95% của gain |
|---|---|---|---|---|
| validation | 0,906174 | 0,933093 | 0,026919 | 0,021105 đến 0,033229 |
| test | 0,887839 | 0,924704 | 0,036865 | 0,030158 đến 0,044017 |

So sánh ghép cặp giữ cùng dữ liệu và pooling, vì vậy gain phù hợp để mô tả tác động của continuation trong protocol này. Bootstrap 10.000 lần theo split_group_id phản ánh biến động lấy mẫu có điều kiện trên model và tập đã dùng.

Khoảng validation chịu ảnh hưởng lựa chọn checkpoint trên chính validation. Khoảng test không bao hàm toàn bộ uncertainty do annotation, lựa chọn thiết kế trước đó hoặc training seed.

### Giới hạn ý nghĩa

Document chunks chứa Title context trong các block đóng băng. Do đó proxy Title → document có thể hưởng lợi từ thông tin Title. Điểm test 0,924704 là Group MRR@10 của phép ghép cặp, không phải Search chính xác 92,47%.

Không quy mọi chênh lệch giữa proxy 128 token cũ và proxy full-document mới cho training, vì độ dài và pooling khác nhau. Kết quả trong bảng này sử dụng baseline cùng protocol để tránh nhầm lẫn đó.

## 14 Output catalog topic full document

![Hình 9. Phân bố nguồn trong 10 topic đang kích hoạt. Topic 6 chủ yếu gồm Patent.](../outputs/report_pipeline_20261025/figures/08_topic_sources.png)
Hình 9. Phân bố nguồn trong 10 topic đang kích hoạt. Topic 6 chủ yếu gồm Patent.
Nguồn output: outputs/batterybert_full_document_20261024_v1/topics/topic_source_distribution.csv

| ID | Diễn giải topic | Tài liệu |
|---|---|---|
| 0 | Oxide Ti/Sn, titanate và composite điện cực | 1.408 |
| 1 | Prussian blue và các hợp chất điện cực chứa sắt | 1.483 |
| 2 | Oxide lớp P2/O3 và biến đổi cấu trúc điện cực | 1.717 |
| 3 | Điện ly, polymer, muối và phụ gia giao diện | 1.426 |
| 4 | Phosphate vanadium, NVP/NASICON và composite carbon | 1.192 |
| 5 | Anode hợp kim Sn/Sb, graphene và MXene | 1.266 |
| 6 | Phần cứng cell và thiết bị chế tạo, kiểm tra pin | 925 |
| 7 | Cấu trúc pha, trật tự vacancy và dẫn ion | 1.035 |
| 8 | Hard carbon, carbon xốp và tiền chất sinh khối | 1.653 |
| 9 | Sulfide/selenide, MoS2 và composite carbon | 1.597 |

Cụm lớn nhất chiếm 12,53%; hai cụm lớn nhất chiếm 24,59%; source Cramér V toàn corpus = 0,3036. Nhãn là diễn giải tạm thời từ keywords và tài liệu đại diện; agent review chưa phải expert ground truth.

## 15 Top 10 keywords được sinh và kiểm tra thế nào

$$
S(c,w)=TF(c,w)\log(1+N/df(w))[(df(c,w)+1)/(df(w)+10)]^{0.25}
$$

Công thức trên là recipe cuối của nhánh full-document: TF tương đối trong cụm, IDF theo số tài liệu train, và exclusivity để ưu tiên từ đặc trưng cho cụm. Phrase boost của cấu hình được chọn bằng 1. Mỗi keyword có ít nhất 5 tài liệu train trong cụm hỗ trợ; tránh các từ/cụm từ bao hàm nhau.

![Hình 10. Top 10 keywords và trọng số ở bốn topic minh họa. Thang điểm mô tả từ khóa, không phải xác suất khoa học.](../outputs/report_pipeline_20261025/figures/09_keywords.png)
Hình 10. Top 10 keywords và trọng số ở bốn topic minh họa. Thang điểm mô tả từ khóa, không phải xác suất khoa học.
Nguồn output: outputs/batterybert_full_document_20261024_v1/topics/topic_keywords.csv

Nhánh scientific keyword stream đọc raw fields và giữ ranh giới stopword, dấu câu và Title/Abstract trước khi tạo ngram. Điều này tránh tạo phrase giả do xóa từ ở giữa. Chuỗi token LDA lịch sử đã mất thông tin khoảng cách không thể dùng để xác nhận phrase liền kề trong văn bản gốc.

Nguồn output: src/scientific_keyword_stream.py; scripts/evaluate_batterybert_full_document.py; docs/KEYWORD_ADJACENCY_CORRECTION_20261024.md

## 16 Đánh giá sau khi đã sinh Top 10 từ khóa

Điểm dùng để lấy Top 10 keywords không đồng nghĩa với chất lượng topic. Sau khi sinh từ khóa, đánh giá cần dùng thống kê corpus và kiểm tra tài liệu độc lập với thứ tự trọng số đã chọn.

$$
NPMI(w_i,w_j)=\log[P(w_i,w_j)/(P(w_i)P(w_j))]/[-\log P(w_i,w_j)]
$$

| Chỉ số | Dữ liệu tham chiếu | Diễn giải và giới hạn |
|---|---|---|
| C_v | Top 10 keywords; cửa sổ 110 token | Coherence gián tiếp; cao không đảm bảo cụm có ý nghĩa khoa học |
| U_Mass | BoW và đồng xuất hiện cấp tài liệu | Cao hơn thường tốt hơn trong cùng protocol |
| Document NPMI | Có/không của keywords trong tài liệu | Dùng chọn recipe mới; khác sliding-window C_v |
| Diversity | Từ khóa khác nhau / 100 vị trí | Ít lặp không tự chứng minh relevance |
| ARI | Assignments qua các seed | Agreement của phân hoạch; không phải accuracy |
| Source Cramér V | Bảng topic × source | Chẩn đoán lệch nguồn; cần đọc nội dung |

Protocol coherence lịch sử dùng tokens Stage1. Bản common-reference dùng quy tắc phrase và dictionary tham chiếu đã sửa cho các danh sách từ khóa. C_v 0,537876 lịch sử của LDA và 0,529681 trong common-reference là hai phép tính khác protocol; không trộn thành một bảng xếp hạng duy nhất.

Đánh giá chuyên môn cần người đọc xem các tài liệu đại diện, độ đồng nhất nội dung, mức khớp keywords–label và các trường hợp nằm sát biên. Trong dự án hiện tại, nhãn khoa học chưa được chuyên gia xác nhận.

Nguồn output: scripts/evaluate_topic_candidates.py; outputs/batterybert_full_document_20261024_v1/topics/scientific_review_protocol.json

## 17 So sánh topic mới với bản optimized trước

![Hình 11. Các chỉ số có đánh đổi giữa bản optimized 23/10 và full-document 24/10. Không có chiến thắng trên mọi thước đo.](../outputs/report_pipeline_20261025/figures/10_topic_tradeoffs.png)
Hình 11. Các chỉ số có đánh đổi giữa bản optimized 23/10 và full-document 24/10. Không có chiến thắng trên mọi thước đo.
Nguồn output: outputs/batterybert_full_document_20261024_v1/topics/common_coherence.csv; outputs/batterybert_optimization_20261023_v1/final_test.json

| Thước đo | Optimized trước | Full-document |
|---|---|---|
| C_v toàn corpus cùng reference | 0,560261 | 0,553022 |
| Keyword diversity | 0,80 | 0,77 |
| Test stability ARI | 0,924481 | 0,688833 |
| Test document NPMI | 0,088294 | 0,079191 |
| Test source Cramér V | 0,337499 | 0,325436 |
| Test top 2 share | 27,05% | 23,10% |

Nhánh mới có phân bố cân hơn theo top2 share và source V trên test giảm nhẹ, trong khi ARI, NPMI, diversity và C_v giảm. Việc kích hoạt là quyết định theo mục tiêu đọc đầy đủ và cổng kỹ thuật đa tiêu chí, không phải chứng minh catalog mới tốt nhất về khoa học.

Các gate đặt giới hạn cho source, stability, coherence, độ lớn cụm, coverage và diversity. Promotion gate của output hiện tại đạt; vẫn cần đánh giá chuyên môn độc lập. Các catalog cũ được giữ để đối chứng.

## 18 Pipeline Search và tiêu chí xếp hạng Top 10

Query → lọc nguồn, năm, required terms và topic nếu có → encode query đầy đủ qua các cửa sổ → lấy tối đa 100 ứng viên dense từ Qdrant → lấy tối đa 100 ứng viên TF-IDF → hợp nhất bằng RRF → trả Top 10.

$$
s(q,d)=q^Td/(\|q\|\|d\|),\quad RRF(d)=\sum_{m:d\in L_m}1/(60+rank_m(d))
$$

| Nhánh | Bằng chứng dùng xếp hạng | Output điểm |
|---|---|---|
| TF-IDF | Từ ngữ trùng giữa query và Title–Abstract | Cosine lexical |
| Dense | Embedding query và document đúng checkpoint | Cosine semantic |
| Hybrid | Đồng thuận thứ hạng lexical và dense | RRF k = 60 |

Cosine của các vector L2-normalized bằng dot product. RRF cộng nghịch đảo thứ hạng, giúp kết hợp hai thang điểm khác nhau. Bài thiếu trong một danh sách không có đóng góp từ danh sách đó. Cả cosine lẫn RRF đều chưa được hiệu chuẩn thành xác suất relevance.

Ví dụ tính tay: A có lexical rank 1 và dense rank 10 thì RRF = 1/61 + 1/70 ≈ 0,03068. B có rank 3 ở cả hai nhánh thì RRF = 2/63 ≈ 0,03175. B đứng trước do đồng thuận hai nhánh cao hơn.

Required terms là bằng chứng từ vựng; có đủ token chưa xác nhận quan hệ hóa học, vai trò anode/cathode hay phủ định. Topic chỉ lọc phạm vi tài liệu khi người dùng chọn; không có topic bonus trong ranking.

Nguồn output: src/full_document_retrieval.py; src/full_document_inference.py; src/final_search.py

## 19 Output thực tế của một truy vấn Search

Query Q01: Hard carbon anodes derived from biomass for sodium-ion batteries

Method: BatteryBERT full-document hybrid. Bảng dưới lấy trực tiếp từ rankings_top10.csv; giữ nguyên thứ hạng và điểm. Đây là kết quả hệ thống đã sinh, chưa phải danh sách đã được người chấm xác nhận relevance.

| Hạng | Tiêu đề bài | Nguồn năm | RRF |
|---|---|---|---|
| 1 | Biomass-derived carbon anodes for sodium-ion batteries | Publication 2023 | 0,03028 |
| 2 | Biomass hard carbon of high initial coulombic efficiency for sodium-ion batteries: Preparation and application | Publication 2022 | 0,02878 |
| 3 | Biochars from various biomass types as precursors for hard carbon anodes in sodium-ion batteries | Publication 2018 | 0,02731 |
| 4 | Revealing the sodium storage behavior of biomass-derived hard carbon by using pure lignin and cellulose as model precursors | Publication 2022 | 0,02650 |
| 5 | A Stable Biomass-Derived Hard Carbon Anode for High-Performance Sodium-Ion Full Battery | Publication 2021 | 0,02513 |
| 6 | High Capacity and High Efficiency Maple Tree-Biomass-Derived Hard Carbon as an Anode Material for Sodium-Ion Batteries | Publication 2018 | 0,02492 |
| 7 | Hard carbon derived for lignin with robust and low-potential sodium ion storage | Publication 2022 | 0,02432 |
| 8 | Biomass derived hard carbon used as a high performance anode material for sodium ion batteries | Publication 2014 | 0,02312 |
| 9 | Pectin, Hemicellulose, or Lignin? Impact of the Biowaste Source on the Performance of Hard Carbons for SodiumIon Batteries | Publication 2017 | 0,02256 |
| 10 | A porous biomass-derived anode for high-performance sodium-ion batteries | Publication 2018 | 0,02158 |

Nguồn output: outputs/batterybert_full_document_20261024_v1/rankings_top10.csv; query_id Q01; method BATTERYBERT_FULLDOC_HYBRID

DOI/identifier, Abstract và đoạn trích evidence của từng kết quả nằm trong CSV. Đoạn evidence được chọn bằng lexical overlap nên không được gọi là lời giải thích đầy đủ cho quyết định của neural encoder.

## 20 Đánh giá relevance của Search Top 10

Sau khi hệ thống sinh Top 10, cần đánh giá từng cặp query–document theo intent người dùng. Protocol hiện tại dùng nhãn 0 không liên quan, 1 liên quan một phần, 2 liên quan trực tiếp; người chấm không thấy method, rank và score, đồng thời cần verifier độc lập.

$$
P@10=\sum_{r=1}^{10}1[g_r=2]/10,\quad DCG@10=\sum_{r=1}^{10}(2^{g_r}-1)/\log_2(r+1),\quad nDCG=DCG/IDCG
$$

MRR@10 lấy nghịch đảo hạng đầu tiên có grade 2, bằng 0 nếu không có trong Top 10, rồi trung bình theo query. nDCG dùng IDCG của pool chung giữa các method; đây là đánh giá theo pool đã thu thập, không phải qrels đầy đủ toàn corpus.

| Output đánh giá hiện tại | Giá trị |
|---|---|
| Số query benchmark | 20 |
| Cặp query–document trong pool | 703 |
| Nhãn người được chấp nhận | 0 |
| Query có pool nhãn hoàn chỉnh | 0 |
| Trạng thái | PENDING_HUMAN_ANNOTATION |
| P@10 / nDCG@10 / MRR@10 relevance | Chưa có kết quả hợp lệ để công bố |

Nhãn trống không được xem là 0. Không dùng coherence, cosine, proxy Title–document hoặc AI tự chấm thay cho ground truth người dùng. Không báo Recall@10 vì chưa biết toàn bộ tài liệu relevant trong corpus.

Các query hiện do tác giả soạn và đã dùng cho khảo sát hệ thống; chưa phải một benchmark chuyên gia bên ngoài. Tập đánh giá đã khóa không được tái dùng làm dữ liệu chọn cấu hình sau khi có nhãn.

Nguồn output: outputs/batterybert_full_document_20261024_v1/evaluation_status.json; outputs/batterybert_full_document_20261024_v1/annotation_protocol.json; src/research_completion_evaluation.py

## 21 Hiệu năng chạy và kiểm chứng kỹ thuật

![Hình 12. Độ trễ từ requests benchmark đã lưu. Request đầu tách riêng để không lẫn với thống kê warm.](../outputs/report_pipeline_20261025/figures/11_latency.png)
Hình 12. Độ trễ từ requests benchmark đã lưu. Request đầu tách riêng để không lẫn với thống kê warm.
Nguồn output: outputs/batterybert_full_document_20261024_v1/benchmark_runs.csv

| Method | Requests warm | Median giây | Mean giây | Max giây |
|---|---|---|---|---|
| Hybrid | 19 | 0,537 | 0,543 | 0,620 |
| Dense | 19 | 0,575 | 0,632 | 1,072 |

Thống kê warm bỏ request đầu mỗi method, còn 19 requests. Những số này mô tả một máy và workload đã chạy; chưa phải p95 production hay khả năng chịu nhiều người dùng đồng thời. Thứ tự chạy và cache cũng có thể ảnh hưởng chênh lệch giữa dense và hybrid.

| Kiểm chứng đã lưu | Kết quả | Phạm vi |
|---|---|---|
| Backend và web tests | 112 / 112 PASS | Kiểm tra các contract đã mã hóa trong tests |
| Model reload parity | PASS | Cache prefix + last block so với full forward |
| Coverage train | PASS | Mọi train ID và chunk của protocol được xử lý |
| Deployment Search / Topics | True / True | Trạng thái artifacts hiện có |

112 tests PASS là kết quả lần chạy được lưu trong verification_summary.json, không phải tuyên bố vừa chạy lại trong lúc biên soạn báo cáo. Các tests xác minh kỹ thuật, không chứng minh relevance hoặc độ đúng khoa học.

## 22 Thảo luận và hướng phát triển

Dự án đã xây được chuỗi xử lý có dữ liệu rõ nguồn, nhiều baseline topic, huấn luyện neural có kiểm tra độ bao phủ, catalog chủ đề và Search hybrid. Bằng chứng mạnh nhất hiện tại là khả năng tái lập output, continuation thật và cải thiện proxy trên cùng giao thức.

Hạn chế lớn nhất là thiếu đánh giá relevance và nhãn topic khoa học độc lập. Corpus chỉ có Title–Abstract; một số tài liệu có thể nằm ngoài mục tiêu pin sodium ion. Mean pooling qua nhiều chunk có thể làm mất chi tiết quan hệ, và các chunk không có attention toàn cục.

### Các thí nghiệm tiếp theo có thể kiểm chứng

| Ưu tiên | Thí nghiệm | Cách xác nhận |
|---|---|---|
| 1 | Hoàn thiện nhãn relevance 0–2 | Hai người chấm và verifier; P@10, nDCG@10, MRR@10 trên pool khóa |
| 1 | Đánh giá topic bởi người có chuyên môn | Đọc tài liệu đại diện và biên; ghi độ đồng nhất, nhãn sai và ngoài phạm vi |
| 2 | Tách tác động encoder và clustering | Giữ embedding để so KMeans/HDBSCAN; giữ clustering để so BGE/BatteryBERT |
| 2 | Ablation pooling và source projection | Cùng split, checkpoint và ngân sách; công bố cả kết quả giảm |
| 3 | Truy hồi nhiều vector theo chunk | Đo chi tiết bị bỏ bởi pooling và đánh đổi latency/index |
| 3 | Mở rộng toàn văn khi có dữ liệu | Version corpus mới; đánh giá riêng Methods, Results và quan hệ dài |

Báo cáo không chọn một model thắng mọi mục tiêu. LDA cung cấp baseline xác suất; BERTopic cung cấp cụm ngữ nghĩa có outlier; các nhánh KMeans tổ chức catalog cố định; Search cần được kết luận bằng relevance có nhãn. Những hướng trên là đề xuất từ chẩn đoán output, chưa phải kết quả đã thực nghiệm.

## 23 Đối chiếu output và tài liệu phương pháp

| Nội dung | Output chính trong workspace |
|---|---|
| Dữ liệu và kiểm tra | outputs/reports/stage1_summary.json |
| LDA chọn K | outputs/tables/stage2/lda_base_metrics.csv; outputs/reports/stage2/selected_model.json |
| LDA và BERTopic đối chứng | outputs/final_system/topic_comparison.csv; outputs/model_comparison_v2/ |
| Sáu catalog và 600 keywords | docs/TOP10_TOPICS_ALL_MODELS_20261024.csv |
| Training và proxy mới | outputs/batterybert_full_document_20261024_v1/training_coverage.json; selection.json; paired_training_effect.json |
| Topic mới | outputs/batterybert_full_document_20261024_v1/topics/final_metrics.json; topic_keywords.csv; topic_assignments.csv |
| Search và nhãn | outputs/batterybert_full_document_20261024_v1/rankings_top10.csv; evaluation_status.json; annotation_pool.csv |
| Hiệu năng và tests | outputs/batterybert_full_document_20261024_v1/benchmark_latency.csv; verification_summary.json |

### Tài liệu tham khảo phương pháp

[1] Blei, Ng và Jordan. Latent Dirichlet Allocation. JMLR 3, 2003. https://www.jmlr.org/papers/v3/blei03a.html

[2] Grootendorst. BERTopic Neural topic modeling with a class based TF IDF procedure. 2022. https://arxiv.org/abs/2203.05794

[3] BAAI. Model card bge base en v1.5. https://huggingface.co/BAAI/bge-base-en-v1.5

[4] BatteryData. Checkpoint BatteryBERT cased. https://huggingface.co/batterydata/batterybert-cased

[5] BERTopic documentation. Algorithm và c-TF-IDF implementation. https://maartengr.github.io/BERTopic/algorithm/algorithm.html ; https://maartengr.github.io/BERTopic/api/ctfidf.html

[6] Gao và cộng sự. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup. 2021. https://arxiv.org/abs/2101.06983

Nguồn phương pháp giải thích thuật toán. Các số đo thực nghiệm trong báo cáo lấy từ artifacts của dự án, không lấy từ các bài báo tham khảo. Manifest kèm theo ghi SHA-256 của những file đã dùng; hình mới được vẽ lại từ output đã lưu, không refit model.

## Phụ lục A BatteryBERT full-document trained

Output catalog gồm đầy đủ 10 topic và 10 keywords mỗi topic. Thứ tự dòng là ID topic; thứ tự từ khóa là rank trong chính model. Không ghép topic giữa các model bằng ID.

| ID | Diễn giải và số tài liệu | Top 10 keywords |
|---|---|---|
| 0 | Oxide Ti/Sn, titanate và composite điện cực; 1.408 tài liệu | TiO2, titanium, titanate, solution, dioxide, SnO2, oxide, anode, carbon, composite |
| 1 | Prussian blue và các hợp chất điện cực chứa sắt; 1.483 tài liệu | iron, prussian_blue, solution, sodium, ferrous, anode, prussian, cathode, ferrocyanide, FeS2 |
| 2 | Oxide lớp P2/O3 và biến đổi cấu trúc điện cực; 1.717 tài liệu | P2, layer, oxide, cathode, O3, layered_oxide, type, transition, phase, metal |
| 3 | Điện ly, polymer, muối và phụ gia giao diện; 1.426 tài liệu | electrolyte, polymer, additive, organic, solvent, salt, carbonate, sodium, electrode, non-aqueous |
| 4 | Phosphate vanadium, NVP/NASICON và composite carbon; 1.192 tài liệu | phosphate, Na3V2(PO4), vanadium, nvp, phosphorus, carbon, sodium, cathode, composite, fluorophosphate |
| 5 | Anode hợp kim Sn/Sb, graphene và MXene; 1.266 tài liệu | graphene, antimony, mxene, sb, anode, composite, sn, alloy, electrode, tin |
| 6 | Phần cứng cell và thiết bị chế tạo, kiểm tra pin; 925 tài liệu | connect, plate, device, end, pole, piece, body, electric, electrode, collector |
| 7 | Cấu trúc pha, trật tự vacancy và dẫn ion; 1.035 tài liệu | phase, na, NaxCoO2, sodium, compound, electrode, cathode, ion, solid_electrolyte, active |
| 8 | Hard carbon, carbon xốp và tiền chất sinh khối; 1.653 tài liệu | hard_carbon, carbon, biomass, porous, carbonize, precursor, carbonization, dry, asphalt, anode |
| 9 | Sulfide/selenide, MoS2 và composite carbon; 1.597 tài liệu | MoS2, sulfide, disulfide, molybdenum, carbon, composite, selenide, anode, solution, SnS2 |

Tài liệu ngoài 10 topic được gán outlier −1: 0. Nhãn là diễn giải từ khóa và tài liệu đại diện, chưa được chuyên gia xác nhận.

Nguồn output: docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; model = BatteryBERT full-document · trained

## Phụ lục B BatteryBERT fine-tuned optimized

Output catalog gồm đầy đủ 10 topic và 10 keywords mỗi topic. Thứ tự dòng là ID topic; thứ tự từ khóa là rank trong chính model. Không ghép topic giữa các model bằng ID.

| ID | Diễn giải và số tài liệu | Top 10 keywords |
|---|---|---|
| 0 | Anode MoS2, dichalcogenide lớp và MXene; 1.031 tài liệu | MoS2, graphene, mxene, molybdenum, disulfide, nanosheets, SnS2, MoSe2, anode, composite |
| 1 | Cathode phosphate vanadium và NASICON; 1.342 tài liệu | phosphate, vanadium, Na3V2(PO4), nvp, sodium, cathode, carbon, PO4, F-, fluorophosphate |
| 2 | Cathode Prussian blue và hợp chất sắt; 1.353 tài liệu | prussian_blue, iron, sodium, solution, prussian, cathode, ferrocyanide, ferrous, anode, iron-based |
| 3 | Cấu trúc cell, bộ góp dòng và lắp ráp pin; 1.084 tài liệu | connect, plate, device, end, electrode, collector, pole, electric, positive, piece |
| 4 | Điện ly, muối, dung môi và polymer; 1.532 tài liệu | electrolyte, polymer, additive, organic, salt, solvent, sodium, carbonate, electrode, solid_electrolyte |
| 5 | Anode sulfide/selenide và composite carbon; 1.235 tài liệu | sulfide, carbon, composite, selenide, CoSe2, cobalt, FeS2, anode, solution, nickel |
| 6 | Anode hợp kim Sn/Sb, phosphorus và chuyển hóa; 915 tài liệu | antimony, sb, tin, sn, anode, alloy, composite, carbon, phosphorus, Sb2S3 |
| 7 | Hard carbon, carbon xốp và tiền chất sinh khối; 1.703 tài liệu | hard_carbon, carbon, biomass, porous, carbonize, precursor, carbonization, dry, anode, asphalt |
| 8 | Anode oxide Ti/Sn và titanate; 1.444 tài liệu | TiO2, titanium, titanate, dioxide, SnO2, solution, oxide, anode, sodium, composite |
| 9 | Cathode oxide lớp P2/O3 và kim loại chuyển tiếp; 2.063 tài liệu | P2, layer, oxide, cathode, O3, layered_oxide, phase, transition, type, metal |

Tài liệu ngoài 10 topic được gán outlier −1: 0. Nhãn là diễn giải từ khóa và tài liệu đại diện, chưa được chuyên gia xác nhận.

Nguồn output: docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; model = BatteryBERT fine-tuned · optimized

## Phụ lục C BatteryBERT KMeans (pretrained)

Output catalog gồm đầy đủ 10 topic và 10 keywords mỗi topic. Thứ tự dòng là ID topic; thứ tự từ khóa là rank trong chính model. Không ghép topic giữa các model bằng ID.

| ID | Diễn giải và số tài liệu | Top 10 keywords |
|---|---|---|
| 0 | Wet-chemical synthesis of electrode materials; 2.309 tài liệu | solution, mix, dry, water, stir, hour, source, method, precursor, wash |
| 1 | Active-material formulations and electrode coatings; 1.677 tài liệu | electrode, active, active material, negative, battery, negative electrode, layer, positive, material, sodium_ion |
| 2 | Mixed electrode performance and electrolyte studies; 1.624 tài liệu | electrolyte, capacity, cycle, electrochemical, high, na-ion, performance, na, anode, result |
| 3 | Anode composites and cycling performance; 1.622 tài liệu | ma, capacity, mah, cycle, high, sib, capacity mah, rate, performance, anode |
| 4 | Sulfide/selenide-carbon composite anodes; 1.603 tài liệu | MoS2, storage, cycle, sib, nanosheets, capacity, anode, sodium storage, TiO2, high |
| 5 | Inorganic electrode compositions and chemical claims; 1.470 tài liệu | sodium, formula, material, metal, oxide, anode material, compound, anode, sodium_ion, sodium_ion battery |
| 6 | Layered-oxide and phosphate cathode materials; 1.351 tài liệu | P2, cathode, capacity, Na3V2(PO4), type, cycle, cathode material, O3, electrochemical, P2 type |
| 7 | Crystal structure, redox and intercalation studies; 1.291 tài liệu | na, phase, redox, structural, na-ion, property, Na+, structure, x-ray, diffusion |
| 8 | Cell hardware and assembly equipment; 480 tài liệu | connect, plate, device, end, box, rod, cover, fix, electric, groove |
| 9 | Reviews and research perspectives; 275 tài liệu | sib, application, energy_storage, advance, design, strategy, mechanism, focus, perspective, cost |

Tài liệu ngoài 10 topic được gán outlier −1: 0. Nhãn là diễn giải từ khóa và tài liệu đại diện, chưa được chuyên gia xác nhận.

Nguồn output: docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; model = BatteryBERT KMeans (pretrained)

## Phụ lục D LDA

Output catalog gồm đầy đủ 10 topic và 10 keywords mỗi topic. Thứ tự dòng là ID topic; thứ tự từ khóa là rank trong chính model. Không ghép topic giữa các model bằng ID.

| ID | Diễn giải và số tài liệu | Top 10 keywords |
|---|---|---|
| 0 | Sodium-storage anodes and carbon composites; 3.486 tài liệu | anode, cycle, capacity, carbon, high, sodium_ion, performance, storage, material, sib |
| 1 | Non-aqueous electrolytes and ionic transport; 997 tài liệu | battery, electrolyte, study, sodium_ion, deliver, sodium, use, conductivity, ionic, analysis |
| 2 | Cathode electrochemistry and vanadium phosphates; 3.559 tài liệu | capacity, cathode, cycle, electrochemical, material, na, performance, Na+, high, na-ion |
| 3 | Electrode-material inventions (mixed materials); 869 tài liệu | material, anode, sodium_ion, battery, sodium, cathode, advance, use, performance, high |
| 4 | Transition-metal and Prussian-white formulations; 78 tài liệu | nc, material, iron, prussian_blue, manganese, coat, layered_oxide, oxide, nickel, Ni1 |
| 5 | Layered oxide cathodes and P2/O3 structures; 84 tài liệu | layer, transition, P2, phase, metal, cathode, oxide, type, O3, material |
| 6 | Active electrode materials and characterization; 1.286 tài liệu | electrode, battery, material, use, sodium_ion, positive, layer, spectroscopy, active, negative |
| 7 | Material synthesis and preparation processes; 2.529 tài liệu | solution, obtain, material, prepare, development, mix, add, battery, sodium_ion, water |
| 8 | Battery devices, housings and assembly; 331 tài liệu | successfully, battery, connect, device, plate, sodium_ion, shell, provide, core, end |
| 9 | Hard-carbon preparation and related materials; 483 tài liệu | carbon, material, hard_carbon, precursor, mxene, obtain, prepare, method, use, temperature |

Tài liệu ngoài 10 topic được gán outlier −1: 0. Nhãn là diễn giải từ khóa và tài liệu đại diện, chưa được chuyên gia xác nhận.

Nguồn output: docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; model = LDA

## Phụ lục E BERTopic–BGE

Output catalog gồm đầy đủ 10 topic và 10 keywords mỗi topic. Thứ tự dòng là ID topic; thứ tự từ khóa là rank trong chính model. Không ghép topic giữa các model bằng ID.

| ID | Diễn giải và số tài liệu | Top 10 keywords |
|---|---|---|
| 0 | Electrode-material preparation patents; 3.552 tài liệu | obtain, material, prepare, solution, mix, source, use, add, precursor, dry |
| 1 | Layered-oxide and polyanion cathodes; 2.650 tài liệu | P2, cathode, capacity, cycle, electrochemical, type, na, phase, O3, Na3V2(PO4) |
| 2 | Alloy/conversion and carbon-based anodes; 1.261 tài liệu | sn, capacity, SnO2, ma, SnS2, anode, sb, cycle, storage, mah |
| 3 | Sulfide/selenide and carbon-composite anodes; 1.008 tài liệu | MoS2, FeS2, cycle, storage, sib, CoSe2, MoSe2, capacity, ma, mah |
| 4 | Prussian-blue analogue cathodes; 499 tài liệu | prussian_blue, ferrocyanide, prussian, hexacyanoferrate, analogue, white, Fe(CN), water, solution, pb |
| 5 | Titanate and TiO2 anodes; 425 tài liệu | TiO2, Na2Ti3O7, anatase, nto, ma, cycle, titanate, storage, capacity, anode |
| 6 | Electrolyte salts, solvents and additives; 397 tài liệu | electrolyte, additive, solvent, salt, non-aqueous, solid_electrolyte, organic, polymer, sodium, carbonate |
| 7 | Battery hardware and manufacturing devices; 382 tài liệu | connect, plate, device, end, fixedly, box, rod, fix, cover, frame |
| 8 | Sodium cobalt oxides and superconductivity; 219 tài liệu | NaxCoO2, NaCrO2, cuo, Co3O4, CoO2, superconducting, yh, superconductivity, phase, magnetic |
| 9 | MXene-based sodium-storage electrodes; 189 tài liệu | mxene, Ti3C2Tx, mxenes, Ti3C2, carbide, mxene-based, two-dimensional, titanium, etch, TiS2 |

Tài liệu ngoài 10 topic được gán outlier −1: 3.120. Nhãn là diễn giải từ khóa và tài liệu đại diện, chưa được chuyên gia xác nhận.

Nguồn output: docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; model = BERTopic–BGE

## Phụ lục F BERTopic–BatteryBERT (pretrained)

Output catalog gồm đầy đủ 10 topic và 10 keywords mỗi topic. Thứ tự dòng là ID topic; thứ tự từ khóa là rank trong chính model. Không ghép topic giữa các model bằng ID.

| ID | Diễn giải và số tài liệu | Top 10 keywords |
|---|---|---|
| 0 | Electrode performance (broad Publication cluster); 7.261 tài liệu | capacity, cycle, ma, mah, high, performance, electrochemical, sib, rate, storage |
| 1 | Material preparation methods (broad Patent cluster); 5.449 tài liệu | obtain, solution, prepare, material, mix, use, add, sodium, method, source |
| 2 | Reviews and research outlook; 218 tài liệu | review, recent, summarize, progress, research, discuss, future, challenge, development, sib |
| 3 | 2D anodes and first-principles calculations; 164 tài liệu | na, monolayer, adsorption, li, ev, barrier, atom, phosphorene, calculation, first-principles |
| 4 | Fixtures, protection and inspection equipment; 156 tài liệu | connect, plate, fixedly, utility, rod, model, end, box, groove, fix |
| 5 | Assembly, heating and mechanical equipment; 101 tài liệu | device, connect, plate, fix, frame, rotate, roller, rod, box, end |
| 6 | Battery packs, management and control; 80 tài liệu | module, unit, voltage, pack, data, battery, charge, power, management, detect |
| 7 | Sodium cobaltate superconductivity; 79 tài liệu | NaxCoO2, superconducting, yh, CoO2, superconductivity, magnetic, t-c, center, dot, superconductor |
| 8 | Cell housing and terminals; 48 tài liệu | portion, terminal, electric, monomer, assembly, toy, second, cover, end, direction |
| 9 | Mixed high-performance electrode materials; 37 tài liệu | high-performance, snsb-porous, vapor-redistribution, Se4P4, Na2C12H6O4, ultracycle-stable, photoassisted, krohnkite-type, na-ion, superior |

Tài liệu ngoài 10 topic được gán outlier −1: 109. Nhãn là diễn giải từ khóa và tài liệu đại diện, chưa được chuyên gia xác nhận.

Nguồn output: docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; model = BERTopic–BatteryBERT (pretrained)
