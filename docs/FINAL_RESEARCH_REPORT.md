# Báo cáo tổng quan nghiên cứu và đánh giá hệ thống Sodium-ion Research Explorer

**Ngày rà soát: 20/10/2026.** Báo cáo mô tả implementation và artifacts đang có trong dự án, không trình bày đề xuất tương lai như kết quả đã thực hiện. Các số liệu thực nghiệm được giữ theo phiên chạy gốc; ngày viết báo cáo không phải ngày chạy lại thí nghiệm.

**Đánh giá hiện tại:** hệ thống đã hình thành một pipeline nghiên cứu có khả năng truy vết, khám phá chủ đề và tìm tài liệu trên 13.702 bản ghi. Search phù hợp để hỗ trợ người nghiên cứu tìm ứng viên rồi đọc lại bằng chứng. Chưa đủ cơ sở khẳng định tìm kiếm đã đạt độ đúng chuyên môn mong muốn, cấu hình relevance tốt nhất hoặc sẵn sàng phục vụ nhiều người dùng. Tối ưu kỹ thuật đã có bằng chứng; tối ưu chất lượng tìm kiếm vẫn cần đánh giá độc lập.

Phụ lục [Kiểm chứng toán học và tối ưu code](MATH_AND_OPTIMIZATION_REVIEW.md) giữ chi tiết công thức, lỗi đã sửa, phép đo CPU và các giới hạn của kiểm chứng. [Rà soát kết quả search](SEARCH_RESULTS_REVIEW.md) cung cấp đối chiếu truy vấn cụ thể. Cấu hình sản phẩm hiện hành được chốt trong [PROJECT_SPEC](PROJECT_SPEC.md).

## 1. Tổng quan bài toán

### 1.1. Nhu cầu và đầu ra mong muốn

Người nghiên cứu pin sodium-ion phải đọc tài liệu thuộc nhiều nhóm: vật liệu điện cực, chất điện giải, cơ chế lưu trữ ion, điều kiện vận hành, công nghệ chế tạo và tái chế. Bài báo và bằng sáng chế có cách viết khác nhau; cùng một ý có thể xuất hiện dưới tên đầy đủ, viết tắt hoặc công thức hóa học. Một truy vấn thường đồng thời yêu cầu đúng vật liệu, đúng hệ pin và đúng điều kiện thí nghiệm.

Dự án giải quyết hai nhu cầu liên quan nhưng có tiêu chí khác nhau:

| Nhu cầu | Đầu ra | Tiêu chí đánh giá phù hợp |
|---|---|---|
| Khám phá corpus | Nhóm chủ đề, từ khóa, tài liệu đại diện, phân bố tài liệu | Coherence, diversity, stability, tỷ lệ outlier và review chuyên gia |
| Tìm tài liệu theo ý định | Danh sách Top5, metadata, Abstract và passage có thể kiểm tra | Relevance theo truy vấn, P@5/nDCG@5/MRR@5, khả năng lấy ứng viên và độ trễ |

Ví dụ, truy vấn “NaFSI ether electrolytes for low-temperature sodium-ion batteries” yêu cầu kiểm tra đồng thời muối NaFSI, dung môi ether, điều kiện nhiệt độ thấp và đúng hệ pin. Một tài liệu chỉ nhắc đến ba từ ở các ngữ cảnh tách biệt chưa chắc đáp ứng nhu cầu này. Vì vậy, bài toán khó hơn so khớp từ hoặc xếp hạng theo độ tương tự chung.

### 1.2. Phạm vi dữ liệu và giới hạn sản phẩm

Đơn vị truy hồi là **một bản ghi Title–Abstract**, kèm nguồn, năm, định danh và provenance. Hệ thống hiện không lập chỉ mục toàn văn bài báo, toàn bộ mô tả/claims của bằng sáng chế, bảng hay hình. Cụm từ “bao phủ đầy đủ Abstract” ở nhánh chunking chỉ nói đến Abstract sẵn có, không có nghĩa là full-text search.

Corpus được lưu từ nguồn dữ liệu đến năm 2023. Hệ thống không tự chứng minh đã bao quát toàn bộ tài liệu sodium-ion đến tháng 10/2026. Abstract cũng có thể không chứa điều kiện hoặc kết quả nằm trong phần thực nghiệm; thiếu bằng chứng ở Abstract phải được ghi nhận theo phạm vi đầu vào.

Đóng góp hiện tại là xây dựng và kiểm chứng một hệ thống tích hợp dữ liệu, topic modeling, biểu diễn và retrieval cho corpus cụ thể. Chưa có bằng chứng để tuyên bố một thuật toán truy hồi mới hoặc vượt mọi phương pháp hiện có.

## 2. Dữ liệu: đầu vào, làm sạch và khả năng truy vết

### 2.1. Quy mô và luồng quyết định

Nguồn số liệu: [stage1_summary.json](../outputs/reports/stage1_summary.json), [cleaning_summary.csv](../outputs/reports/cleaning_summary.csv).

| Hạng mục | Số bản ghi | Diễn giải |
|---|---:|---|
| Patent đầu vào | 8.657 | Dữ liệu nguồn được bảo toàn |
| Publication đầu vào | 11.919 | Dữ liệu nguồn được bảo toàn |
| Tổng đầu vào | 20.576 | Trước quyết định đưa vào corpus chính |
| Corpus chính | **13.702** | 5.933 Patent + 7.769 Publication |
| Nhãn relevance UNCERTAIN | 6.598 | Nhãn sàng lọc; không trộn vào thí nghiệm chính |
| Đủ Title và Abstract cho proxy fine-tuning | 13.650 | Chỉ những cặp có nội dung thật |
| Chỉ có Title | 52 | Không sinh Abstract giả để bù |

Phải phân biệt **nhãn relevance** với **disposition cuối cùng**. Bảng disposition loại trừ nhau gồm: MAIN_CORPUS 13.702; RELEVANCE_REVIEW 6.596; OUT_OF_SCOPE 254; LANGUAGE_REVIEW 15; CONFIRMED_DUPLICATE 3; RETRACTED 4; TEXT_QUALITY_REVIEW 2. Tổng bằng 20.576. Không lấy 20.576 trừ riêng 6.598 UNCERTAIN để suy ra corpus chính.

### 2.2. Các bước xử lý và lý do

1. **Đọc, chuẩn hóa schema và kiểm tra cấu trúc.** Hai nguồn được đưa về các trường chung để có thể so sánh bằng cùng bộ mã. Giữ nguồn gốc để truy ngược quyết định xử lý.
2. **Kiểm tra nội dung, ngôn ngữ và phạm vi.** CORE/RELATED có thể được đưa vào corpus sau các kiểm tra khác; UNCERTAIN được tách để review. Đây là sàng lọc heuristic, không phải nhãn chuyên gia đã xác nhận mọi tài liệu.
3. **Kiểm tra bản ghi trùng và liên kết ứng viên.** Dùng định danh/DOI, tiêu đề gần nhau và nhóm liên kết để hạn chế tài liệu tương tự rơi vào các split khác nhau. Cách này không phát hiện được mọi quan hệ semantic hoặc toàn bộ patent family.
4. **Tạo biểu diễn phục vụ từng nhiệm vụ.** Topic modeling dùng tokens; transformer nhận văn bản; Search V2 tạo lexical tokens riêng để xử lý từ thường và công thức. Giữ Title/Abstract gốc để người dùng kiểm tra.
5. **Đóng băng doc_id và thứ tự hàng.** Vector, metadata và kết quả phải nối theo cùng danh tính. Sai thứ tự embedding dù kích thước đúng có thể trả đúng điểm cho nhầm tài liệu.

Implementation: [pipeline.py](../src/pipeline.py), [quality.py](../src/quality.py), [text_processing.py](../src/text_processing.py), [data_io.py](../src/data_io.py). Các đầu ra chính là [main_corpus.parquet](../data/processed/main_corpus.parquet), [lda_tokens.parquet](../data/processed/lda_tokens.parquet), [bert_texts.parquet](../data/processed/bert_texts.parquet), [metadata_extended.parquet](../data/processed/metadata_extended.parquet).

### 2.3. Những giới hạn dữ liệu ảnh hưởng trực tiếp đến kết luận

Corpus chính chưa được chuyên gia sàng lọc toàn bộ; có thể bỏ sót tài liệu phù hợp trong UNCERTAIN hoặc giữ tài liệu chỉ liên quan gián tiếp. Mọi kết quả truy hồi có điều kiện trên corpus đã chọn, không đo coverage của toàn lĩnh vực. Patent và Publication còn khác nhau về độ dài, chất lượng Abstract và cách diễn đạt; cần báo cáo lỗi theo từng nguồn thay vì chỉ lấy một trung bình.

Audit trước phát hiện lỗi placeholder kế thừa ở tokens topic, ảnh hưởng 355 tài liệu/716 occurrences. Search V2 không dùng lại bộ tokens đó cho lexical index, nhưng kết quả topic lịch sử vẫn cần được hiểu trong giới hạn này. Sửa preprocessing phải tạo phiên bản dữ liệu và mô hình mới để so sánh, tránh ghi đè rồi gọi kết quả mới là cùng thí nghiệm cũ.

## 3. Pipeline tổng thể: phần nào chạy offline, phần nào chạy khi search?

| Giai đoạn | Đầu vào → xử lý → đầu ra | Vai trò trong sản phẩm |
|---|---|---|
| Stage 1 | Excel nguồn → audit/làm sạch/sàng lọc/nhóm trùng → corpus và representations | Dữ liệu nền dùng chung |
| Stage 2 | Tokens đóng băng → LDA và so sánh cấu hình → topics/keywords/assignments | Khám phá chủ đề; chạy offline |
| Stage 3 | Title–Abstract theo grouped split → BGE baseline/fine-tuning → embeddings và proxy metrics; UMAP/HDBSCAN/c-TF-IDF → BERTopic | Nhánh nghiên cứu biểu diễn và chủ đề |
| Chunked | Title + các cửa sổ Abstract → embedding từng chunk → pooling tài liệu → BERTopic đối chứng | Kiểm tra tác động của giới hạn độ dài |
| Xây chỉ mục Search V2 | Corpus + pretrained embeddings + tokenizer/config identity → lexical index và dense matrix | Chuẩn bị trước truy vấn |
| Truy vấn Search V2 | Query + filters → lexical/dense → RRF → passage selection → cross-encoder → Top5 | Luồng tương tác hiện hành |
| Đánh giá | Rankings đóng băng + rubric/labels → metrics, CI, error analysis | Tách khỏi chấm điểm trực tiếp trong UI |

Stage 1/2/3 và huấn luyện topic không chạy lại mỗi khi mở app. LDA/BERTopic phục vụ màn hình khám phá riêng; **không có topic bonus hoặc điểm LDA cộng vào ranking Search V2**. Các tên SE1–SE3/Qdrant thuộc đối chứng lịch sử, không mô tả backend mặc định hiện tại.

Hai câu hỏi phải được đánh giá riêng: “tài liệu được gom thành các nhóm dễ hiểu chưa?” và “tài liệu đáp ứng query có đứng đầu chưa?”. Topic coherence cao không chứng minh search tốt, và search tốt cũng không xác nhận ý nghĩa khoa học của mọi topic.

## 4. LDA: cách làm, lý do lựa chọn và giới hạn

### 4.1. Cơ chế

LDA biểu diễn mỗi tài liệu bằng phân phối trên K chủ đề, mỗi chủ đề bằng phân phối trên từ. Với từ w trong tài liệu d, xác suất được mô hình hóa theo tổng qua các chủ đề: `p(w|d) = Σ_k θ[d,k] × φ[k,w]`. Mô hình dùng bag-of-words nên bỏ thứ tự từ và phần lớn quan hệ cú pháp. Cơ sở phương pháp: [Blei, Ng & Jordan, 2003](https://www.jmlr.org/papers/v3/blei03a.html).

Dự án dùng dictionary từ toàn bộ tokens Stage 1, không lọc từ vựng ở bước này; huấn luyện K ∈ {10,15,20,25,30}, seed cơ sở 42. Mỗi cấu hình dùng 10 passes, 100 iterations, chunksize 1.024, alpha asymmetric và eta symmetric. Hai ứng viên có C_v tốt nhất được thử thêm seeds 123 và 2026. Chi tiết tại [stage2.yaml](../configs/stage2.yaml), [stage2.py](../src/stage2.py).

### 4.2. Vì sao dùng LDA?

LDA tạo baseline dễ giải thích bằng từ khóa và phân phối chủ đề, có thể chạy trên CPU và không phụ thuộc transformer. Một tài liệu có thể chứa nhiều chủ đề, phù hợp với nhu cầu xem sự giao thoa trong corpus. Baseline này cũng giúp xác định mô hình neural đem lại điều gì ngoài biểu diễn bag-of-words.

Đổi lại, LDA khó xử lý đồng nghĩa, quan hệ vật liệu–vai trò và công thức nhiều biến thể. Từ cùng xuất hiện không chứng minh một claim khoa học. Giữ toàn bộ từ vựng bảo toàn thuật ngữ hiếm nhưng cũng có thể giữ nhiễu; cần ablation có kiểm soát trước khi kết luận lựa chọn này tốt nhất.

### 4.3. Cách chọn K và mức độ tối ưu

Chỉ số tổng hợp dùng trọng số 0,50 cho C_v đã min–max; 0,15 cho U_Mass đã min–max; 0,15 cho diversity; 0,20 cho stability. Stability ghép topic giữa seeds bằng Hungarian matching, dựa trên Jaccard của Top10 từ. Cấu hình được chọn là **K=20 trong tập đã khảo sát**.

C_v phản ánh tính nhất quán đồng xuất hiện từ; U_Mass cũng dựa trên đồng xuất hiện với quy ước khác; diversity đo mức ít lặp từ giữa topics; stability đo độ giống của từ khóa giữa các lần chạy. Không chỉ số nào riêng lẻ xác nhận các nhóm có tên và ý nghĩa chuyên môn đúng.

Việc chỉ chạy nhiều seed cho Top2 khiến lượng bằng chứng giữa các K không cân bằng. Min–max còn phụ thuộc tập ứng viên: thêm một cấu hình có thể thay thang chuẩn hóa và đổi thứ tự điểm tổng hợp. Vì vậy, K=20 là lựa chọn theo protocol hiện tại, chưa là số chủ đề tối ưu toàn cục. Bước tốt hơn là đánh giá nhiều seed đồng đều, phân tích độ nhạy trọng số và review các topic bị trộn/tách.

## 5. BGE và fine-tuning Title → Abstract

### 5.1. Vì sao cần dense embeddings?

BM25 phụ thuộc từ xuất hiện; dense retrieval bổ sung khả năng nối các cách diễn đạt gần nghĩa. BGE mã hóa query và tài liệu riêng, vì vậy có thể tính trước vector tài liệu và chỉ mã hóa query khi tìm kiếm. Dự án dùng `BAAI/bge-base-en-v1.5`, vector 768 chiều; query thêm instruction tìm kiếm, tài liệu không thêm instruction. Cách dùng này phù hợp với [hướng dẫn BGE](https://bge-model.com/tutorial/1_Embedding/1.2.1.html).

Vector được L2-normalize. Khi hai vector khác 0 đều có norm bằng 1, `cos(q,d) = q·d`; do đó nhân ma trận có thể xếp hạng chính xác theo cosine. Sự bằng nhau về toán học không bảo đảm vector phân biệt đúng NaFSI/NaTFSI, aqueous/nonaqueous hoặc điều kiện nhiệt độ.

Giới hạn 128 tokens là cấu hình của nhánh baseline dự án, không nên diễn giải thành giới hạn kiến trúc chung của BGE. Embedding tài liệu ở nhánh này có thể bỏ nội dung ở cuối Abstract. Query neural quá 128 tokens bị từ chối để người dùng rút ngắn, tránh âm thầm thay đổi ý định bằng truncation.

### 5.2. Thiết kế fine-tuning

Tận dụng mỗi Title làm pseudo-query và Abstract tương ứng làm positive vì dự án chưa có đủ query–document relevance labels. Dữ liệu chia theo `split_group_id`, không chia ngẫu nhiên từng hàng để các liên kết đã biết không vượt split. [Split audit](../outputs/reports/stage3/split_audit.json) ghi 10.921 cặp train, 1.361 validation, 1.368 test; test có 1.359 nhóm; không có group overlap theo các nhóm đã biết.

Huấn luyện một epoch, chỉ block encoder cuối được mở: batch 8, learning rate 2×10⁻⁵, weight decay 0,01, warm-up 10%, temperature 0,05, gradient clipping 1. Đánh giá mỗi 250 steps và chọn checkpoint fine-tuned bằng validation group MRR@10. Cấu hình tại [stage3.yaml](../configs/stage3.yaml), logic tại [stage3_neural.py](../src/stage3_neural.py).

Với batch cặp `(q_i,d_i)`, đặt `s_ij = normalized(q_i)·normalized(d_j)/temperature`. Loss lấy trung bình cross-entropy theo hàng và theo cột; đường chéo là positive. Các cặp ngoài đường chéo cùng nhóm hoặc có Title Jaccard ≥0,8 được mask để hạn chế false negatives. Mask là heuristic; chưa bảo đảm mọi tài liệu liên quan đều được nhận diện.

Mở một block giảm chi phí CPU và giới hạn số tham số thay đổi. Tuy nhiên, cache validation tăng tốc hiện chỉ hỗ trợ đúng cấu hình một block trainable; code đã từ chối cấu hình khác. Chọn trong các checkpoint fine-tuned cũng chưa phải quy tắc tự động giữ pretrained nếu mọi checkpoint đều kém hơn nó.

### 5.3. Kết quả và cách đọc đúng

| Chỉ số proxy trên test | Pretrained | Fine-tuned | Chênh lệch tuyệt đối | CI 95% của chênh lệch |
|---|---:|---:|---:|---|
| Group MRR@10 | 0,893567 | 0,928083 | +0,034516 | [0,024765; 0,044217] |
| Group Success@1 | 0,859649 | 0,899123 | +0,039474 | [0,024945; 0,053249] |
| Group Success@5 | 0,937135 | 0,967105 | +0,029971 | [0,020482; 0,040205] |
| Group Success@10 | 0,957602 | 0,981725 | +0,024123 | [0,016093; 0,032967] |

Nguồn: các test metrics trong [thư mục Stage 3](../outputs/reports/stage3/), kiểm chứng diễn giải trong [phụ lục toán](MATH_AND_OPTIMIZATION_REVIEW.md). **Success@k** là có ít nhất một Abstract thuộc nhóm phù hợp trong Top-k, không phải tỷ lệ toàn bộ relevant documents được lấy về. Trường JSON lịch sử `group_recall_at_k` được giữ làm alias tương thích và đã có metadata giải thích.

CI được tính bằng paired cluster bootstrap theo split_group_id, với estimand trung bình theo tài liệu. Nó có điều kiện trên model, checkpoint, nhóm và dữ liệu đã quan sát; không bao gồm mọi biến thiên training seed, nhóm chưa phát hiện hoặc sai nhãn.

Kết quả hỗ trợ kết luận fine-tuning giúp nhiệm vụ Title→Abstract này. Nó chưa chứng minh cải thiện truy vấn tự nhiên trên toàn corpus: query xuất phát từ chính tài liệu, candidates chỉ trong split đánh giá, còn người dùng đặt query rộng hoặc nhiều điều kiện. Search mặc định vì thế vẫn dùng pretrained; fine-tuned được giữ như một đối chứng.

## 6. BERTopic và nhánh chunked

### 6.1. Cơ chế và vai trò từng thành phần

BERTopic trong dự án thực hiện: embeddings tài liệu → UMAP giảm còn 5 chiều → HDBSCAN phân cụm → c-TF-IDF lấy từ khóa. UMAP hỗ trợ biểu diễn cấu trúc lân cận cho bước clustering; HDBSCAN tìm cụm theo mật độ và cho phép một phần tài liệu là noise; c-TF-IDF tạo mô tả từ vựng cho từng cụm. Cấu trúc chung dựa trên [BERTopic, Grootendorst, 2022](https://arxiv.org/abs/2203.05794).

Các tham số chính: UMAP neighbors 15, min_dist 0; HDBSCAN min_cluster_size 30, min_samples 10, lựa chọn eom; không ép số topic bằng `nr_topics`. Khác LDA, số topic xuất hiện từ clustering. HDBSCAN membership là mức gắn với cụm theo mô hình, không phải xác suất tài liệu đúng khoa học hoặc đáp ứng query. Topic -1 là outlier, không nên đặt tên như một chủ đề chuyên môn thống nhất.

Vì sao dùng nhánh này? Embeddings giúp gom những cách diễn đạt gần nhau mà bag-of-words dễ tách; HDBSCAN không buộc mọi tài liệu vào cụm; từ khóa hỗ trợ người đọc kiểm tra. Giới hạn là phụ thuộc embedding, seed UMAP, mật độ và preprocessing. Baseline seed 42 có 92 topics nhưng seed 123 có 82 topics, cho thấy không nên coi mỗi cụm là một phân loại tự nhiên cố định.

### 6.2. Chunking giải quyết gì?

Nhánh chunked giữ Title làm ngữ cảnh, chia Abstract theo ngân sách 256 tokens với overlap 64, mã hóa các chunks, lấy mean của các vector chunk đã chuẩn hóa rồi L2-normalize lần nữa:

`v_doc = normalize((1/m) × Σ_j normalize(encoder(chunk_j)))`.

Đầu ra: **29.139 chunks**, ma trận **13.702 × 768**, bao phủ 100% Abstract WordPiece có sẵn; 52 tài liệu thiếu Abstract dùng Title thật. Xem [embedding_success.json](../outputs/bertopic_chunked/embedding_success.json), [bertopic_chunked.py](../src/bertopic_chunked.py).

Động cơ là tránh bỏ thông tin cuối Abstract. Đổi lại, mean chunk nén nhiều ý về một vector; Title và vùng overlap được biểu diễn lặp, còn chunk đuôi ngắn có trọng số bằng chunk dài. Đây là mean của các view qua encoder phi tuyến, không phải trung bình token không chệch. Chi phí tạo embedding ghi nhận khoảng 9.174 giây trong phiên batch 2, chỉ là thời gian của tác vụ embedding đó, không phải độ trễ mỗi truy vấn.

### 6.3. So sánh định lượng

| Chỉ số | BERTopic baseline 128 | BERTopic chunked 256 | Diễn giải |
|---|---:|---:|---|
| Topics, không gồm noise | 92 | 84 | Không có quy tắc ít topic hơn luôn tốt hơn |
| Outlier documents | 2.751 | 3.120 | Tăng 369 tài liệu |
| Outlier rate | 20,08% | 22,77% | Tăng khoảng 2,69 điểm phần trăm |
| C_v | 0,462821 | 0,477892 | Tăng trên evaluator hiện tại |
| U_Mass | −9,256268 | −9,080645 | Ít âm hơn theo quy ước chỉ số |
| Topic diversity | 0,858696 | 0,878571 | Từ khóa ít trùng hơn |

Nguồn: [comparison.csv](../outputs/bertopic_chunked/comparison.csv). Coherence/diversity tăng đồng thời tỷ lệ noise tăng. Vì thế, chưa thể kết luận chất lượng topic tổng thể tăng. Cũng chưa tách tác dụng riêng của coverage, vì Title context, độ dài cửa sổ và pooling cùng thay đổi. Cần ablation từng yếu tố, nhiều seed và review cả inliers lẫn outliers trước khi chọn nhánh tốt nhất.

## 7. Search V2: phương pháp chi tiết và lý do sử dụng

Implementation chính: [search_v2.py](../src/search_v2.py), [search_v2_constraints.py](../src/search_v2_constraints.py), [search_v2_service.py](../src/search_v2_service.py), [app.py](../app.py). Cấu hình: [search_v2.json](../configs/search_v2.json).

### 7.1. Chuẩn hóa query, công thức và bộ lọc

Lexical tokenizer chuẩn hóa Unicode, xử lý từ tiếng Anh bằng stemming và dành kênh `chem:` cho các token được nhận diện giống công thức. Các aliases hỗ trợ một số muối, gồm NaFSI/NaTFSI. Mục đích là giảm biến thể chữ thường/từ biến hình nhưng vẫn giữ tín hiệu công thức.

Đây chưa là chemical parser: không bảo đảm tương đương cấu trúc, đúng stoichiometry, mọi ký hiệu chỉ số hoặc mọi lỗi OCR. Truy vấn tiếng Việt cũng chưa được benchmark tương xứng vì encoder/tokenizer chính hướng đến tiếng Anh.

Source/year filters xác định tập tài liệu hợp lệ trước Top-K; tài liệu thiếu năm bị loại nếu có year range. Required terms phân tách bằng dấu chấm phẩy mang nghĩa AND, kiểm tra trên Title/Abstract đầy đủ theo token/aliases được hỗ trợ. “Có các token” không đồng nghĩa “các thành phần có đúng vai trò, quan hệ và điều kiện yêu cầu”.

### 7.2. BM25: giữ bằng chứng từ vựng

Với field F, thành phần theo một từ có dạng `IDF(t) × f(t,F)(k1+1) / [f(t,F)+k1(1−b+b×length(F)/avg_length)]`. Tần suất có bão hòa để lặp từ không tăng điểm tuyến tính; hiệu chỉnh độ dài giảm lợi thế chỉ do văn bản dài. Điểm hiện hành là:

`lexical(q,d) = 2 × max(BM25_Title(q,d),0) + max(BM25_Abstract(q,d),0)`.

Title được trọng số 2 vì thường cô đọng đối tượng nghiên cứu; Abstract bổ sung nội dung rộng hơn. `k1=1,5`, `b=0,75` là baseline thiết kế. Hai field chấm riêng rồi cộng có clipping, nên không gọi implementation này là BM25F đầy đủ. Định nghĩa IDF phải đối chiếu với [rank_bm25 source](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py), không thay tùy ý bằng một biến thể BM25 khác khi kiểm chứng.

BM25 hữu ích cho công thức và thuật ngữ cụ thể; điểm yếu là đồng nghĩa và không hiểu quan hệ. Clipping điểm âm còn tạo phản ví dụ ở corpus nhỏ: có token thật nhưng điểm bị đưa về 0. Chưa có đo lường cho thấy mức ảnh hưởng tương ứng trên corpus hiện tại; đây là lý do cần một thí nghiệm gate/IDF riêng, không phải lý do tự đổi ranking đã lưu.

### 7.3. Dense retrieval: bổ sung diễn đạt gần nghĩa

Query BGE được so với toàn bộ dense matrix bằng dot product, lấy Top100 trong tập hợp lệ. Ma trận float32 13.702×768 chiếm khoảng 42,1 MB thập phân, tức 40,1 MiB, chưa kể model, metadata và overhead. Với quy mô này, exact scan đơn giản và tránh thêm sai số truy hồi xấp xỉ; chưa có bằng chứng cần ANN/Qdrant để giải quyết nút thắt hiện tại.

Các nhánh pretrained, fine-tuned 128 và chunked 256 được giữ để so sánh. Hash checkpoint, vector, doc_id order, tính hữu hạn và norm là điều kiện để không dùng nhầm representation. Tương tự cao vẫn có thể là tài liệu cùng lĩnh vực nhưng sai công thức hoặc điều kiện.

### 7.4. RRF: hợp nhất lexical và dense

Mỗi nhánh lấy Top100; hợp có tối đa 200 ứng viên. Với hạng bắt đầu từ 1:

`RRF(d) = Σ_j 1 / (60 + rank_j(d))`, chỉ cộng ở danh sách chứa d.

RRF kết hợp thứ hạng nên tránh cộng trực tiếp BM25 với cosine vốn khác thang đo. Cơ sở phương pháp: [Cormack, Clarke & Büttcher, 2009](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf). Tham số 60 và trọng số bằng nhau hiện là lựa chọn thiết kế; bài báo không chứng minh chúng tối ưu cho corpus sodium-ion.

Hybrid/rerank hiện trả rỗng nếu không có lexical support. Đây là heuristic bảo thủ để giảm kết quả cho chuỗi vô nghĩa, có thể bỏ lỡ query paraphrase không trùng từ. Nó chưa phải bộ phát hiện ngoài miền được hiệu chỉnh. Các nhánh dense vẫn trả ứng viên với query vô nghĩa nên không thể dùng “luôn có 5 kết quả” làm bằng chứng chất lượng.

### 7.5. Passage selection và cross-encoder reranking

Reranker nhận tối đa **Top50 sau RRF**, chọn tối đa **2 passages/tài liệu**. Cửa sổ passage yêu cầu 320 tokens, overlap 64; Title tối đa 80, query tối đa 128, tổng cặp tối đa 512 gồm special tokens. Selector kiểm tra ngân sách thực tế và bước tiến cửa sổ để tránh lặp vô hạn khi đổi cấu hình.

Các cửa sổ Abstract được ưu tiên theo độ chồng lấp từ với query; offset hỗ trợ chỉ ra đoạn bằng chứng trong Abstract đã chuẩn hóa. Cross-encoder đọc đồng thời query và từng passage; điểm tài liệu là điểm passage cao nhất. Mô hình `cross-encoder/ms-marco-MiniLM-L6-v2` là model công khai pretrained trên MS MARCO, chạy cục bộ; không được fine-tune riêng cho hóa học trong dự án. Xem [model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

Thiết kế hai tầng tận dụng bi-encoder tìm ứng viên nhanh và cross-encoder so sánh chi tiết hơn; đây cũng là kiến trúc [retrieve & rerank của Sentence Transformers](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html). Nhưng reranker không cứu được tài liệu đã mất trước Top50, và chỉ đọc hai passages nên có thể bỏ qua bằng chứng ở đoạn khác. Phép max có thể ưu tiên đoạn rất hợp một phần query dù điều kiện còn lại không được đáp ứng.

Điểm reranker là điểm xếp hạng riêng của model, không phải xác suất tài liệu đúng. Token overlap trong passage cũng không kiểm chứng entailment, phủ định hoặc vai trò hóa học. Toàn Abstract được quét để chọn cửa sổ, nhưng cross-encoder không đọc toàn bộ mọi Abstract.

### 7.6. Hiển thị và vận hành

UI trả Top5 với Title, Abstract, nguồn/năm/định danh, score và passage để người đọc kiểm tra, có tải CSV. Mặc định là `RERANK_PRETRAINED`: lựa chọn vận hành giữ đường dùng ít phụ thuộc checkpoint dự án, **không phải phương pháp đã thắng một benchmark relevance độc lập**. Nhánh này không cần project fine-tuning hoặc Qdrant.

Service hiện dùng một worker để tuần tự hóa tác vụ và kiểm soát RAM/model state. So sánh nhiều phương pháp trong một request tái dùng query embedding, lexical scores và dense scores tương ứng; filters vẫn được áp riêng. Cache kết thúc cùng request. Thời gian hiển thị khi so sánh có tái dùng không đại diện benchmark các phương pháp chạy độc lập.

## 8. Bằng chứng chất lượng search: đã đo gì, chưa đo gì?

### 8.1. Phải tách bốn loại đánh giá

| Loại bằng chứng | Quy mô hiện có | Thực sự đo gì? | Có chứng minh search đáp ứng người dùng chưa? |
|---|---|---|---|
| Title→Abstract proxy | 1.368 test pairs, 1.359 nhóm | Tìm Abstract cùng tài liệu/nhóm từ Title | Chưa; thuận lợi hơn query người dùng và khác candidate pool |
| Rankings truy vấn miền | 24 queries × 10 methods | Kết quả thật trên corpus và khác biệt giữa methods | Chưa có độ đúng nếu thiếu labels |
| Relevance do AI chấm | 116/942 cặp có consensus; 2/23 query đủ pool | Relevance theo diễn giải Gemini trên phần đã hoàn tất | Chỉ chẩn đoán từng phần, chưa phải nhãn chuyên gia |
| AI-generated source recovery | 197/200 queries × 14 methods | Có tìm lại đúng doc_id đã sinh query không | Proxy bổ sung; không phải relevance hoặc Recall toàn corpus |

### 8.2. Bộ truy vấn miền và trạng thái nhãn độc lập

Benchmark Search V2 có 23 truy vấn miền và Q24 synthetic OOV, tổng 240 lượt của 10 methods. Artifacts có 1.165 hàng Top5 và 2.330 hàng Top10; không phải mọi lượt đều có đủ kết quả do các phương pháp có cơ chế abstain khác nhau. Nguồn: [REPORT](../outputs/search_upgrade_20261018/REPORT.md), [top5_comparison.csv](../outputs/search_upgrade_20261018/top5_comparison.csv), [rankings_top10.csv](../outputs/search_upgrade_20261018/rankings_top10.csv).

Evaluator mở rộng thêm bốn baseline lịch sử thành 14 methods, với pool chung **942 cặp query–document**. Hai biểu mẫu [annotator A](../outputs/search_upgrade_20261018/annotator_A_NEW.csv) và [annotator B](../outputs/search_upgrade_20261018/annotator_B_NEW.csv) hiện đều có **0/942 relevance_grade được điền**. Vì vậy, P@5/nDCG@5/MRR@5 dựa trên nhãn độc lập vẫn chưa được xác lập. Điểm AI ở thư mục khác không được dùng để ghi đè trạng thái này.

Q01/Q08/Q16/Q19/Q23 đã dùng trong development; 18 query còn lại vẫn do tác giả soạn và rankings đã được tạo. Nếu tiếp tục tune dựa trên những kết quả này, cần thêm query families thực sự mới cho đánh giá xác nhận. Số hàng ranking hay số lượt chạy không phải số quan sát relevance độc lập.

### 8.3. Kết quả AI relevance đang có

[Báo cáo AI v1](../outputs/retrieval_ai_evaluation_v1/REPORT.md) ghi trạng thái `PENDING_API_HTTP_429_RESOURCE_EXHAUSTED`. Đã có 118/942 cặp được cả A/B chấm hợp lệ; 116 cặp có consensus số, 2 cặp chờ phân xử. Chỉ **2/23 queries** đủ nhãn toàn pool để tính trung bình đồng nhất giữa các methods.

Trên đúng hai query này, RERANK_PRETRAINED có P@5=0,8000, nDCG@5=0,8812, MRR@5=1,0000; BM25_CHEMICAL tương ứng 0,6000/0,6377/0,6667. Đây là số quan sát từng phần, không phải kết quả cho cả 23 query và không đủ để chọn winner. Các query chưa hoàn tất có thể khác về độ khó; thiếu nhãn không được đổi thành grade 0.

Hai lượt chấm là cùng model Gemini với các lần gọi tách trạng thái. Exact agreement 0,9831 và quadratic weighted kappa 0,9938 chỉ mô tả nhất quán trên 118 cặp đã có response; cùng model có thể cùng mắc lỗi. Agreement cao không chứng minh relevance đúng hoặc tính độc lập của chuyên gia. Error analysis AI hiện gợi ý sai vật liệu/công thức, điều kiện và quan hệ kỹ thuật, nhưng cần đọc lại evidence trước khi quy thành nguyên nhân đã xác nhận.

Bản viết lại báo cáo này chỉ đọc các artifacts có sẵn; không gọi lại API, không phát sinh thêm nhãn hoặc tiếp tục job đang bị quota.

### 8.4. Kết quả source recovery với truy vấn do AI sinh

Generator nhận Abstract và sinh query trước khi retrieval; không nhận Title/doc_id. Dự kiến 200 source documents cân bằng hai nguồn; có 197 queries hợp lệ được chạy, gồm 100 Publication và 97 Patent. Ba query chưa hợp lệ không bị tính thành miss. Các kiểm tra format/Title overlap chỉ là heuristic, chưa xác minh mọi query trung thành với Abstract.

| Method | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25_CHEMICAL | 0,6345 | 0,8071 | 0,8376 | 0,7080 |
| DENSE_PRETRAINED | 0,6904 | 0,8528 | 0,8731 | 0,7600 |
| DENSE_FINETUNED | 0,7513 | 0,8782 | 0,8934 | 0,8066 |
| DENSE_CHUNKED | **0,8020** | **0,8934** | **0,9239** | **0,8436** |
| HYBRID_PRETRAINED | 0,7107 | 0,8528 | 0,8832 | 0,7741 |
| HYBRID_CHUNKED | 0,7462 | 0,8883 | 0,9137 | 0,8056 |
| RERANK_PRETRAINED | 0,6853 | 0,8579 | 0,9036 | 0,7651 |
| RERANK_CHUNKED | 0,6853 | 0,8629 | 0,8985 | 0,7655 |

Bảng trích các nhánh tiêu biểu; đủ 14 methods ở [synthetic/summary.csv](../outputs/retrieval_ai_evaluation_v1/synthetic/summary.csv). [Completion validation](../outputs/retrieval_ai_evaluation_v1/completion_validation.json) kiểm tra lại 2.758 run records, sai lệch metric tối đa khoảng 5,55×10⁻¹⁷; PASS xác nhận tính toán của phần đã chạy, không xác nhận relevance hay hoàn tất cả 200 queries.

DENSE_CHUNKED có point estimates cao nhất trong phép thử này. Reranking không cải thiện source recovery một cách đồng đều; ví dụ RERANK_PRETRAINED thấp hơn HYBRID_PRETRAINED ở Hit@1 nhưng cao hơn ở Hit@5/10. Điều này bác bỏ cách nói “thêm reranker luôn tốt hơn” trên mọi mục tiêu, nhưng chưa chứng minh phương pháp nào tốt hơn về relevance chuyên gia: một tài liệu khác đúng ý hơn source vẫn bị exact-ID metric tính là miss.

Trong 197 source documents đã đánh giá, **163 thuộc train**, 16 validation, 18 test của fine-tuning cũ. Vì vậy, phép so này không độc lập với tài liệu huấn luyện và có thể thuận lợi cho fine-tuned/chunked. Cần đọc [by_training_split.csv](../outputs/retrieval_ai_evaluation_v1/synthetic/by_training_split.csv) trước khi diễn giải khả năng khái quát. CI giữ cố định checkpoints và queries đã sinh; chưa bao gồm training/generation seed uncertainty.

### 8.5. Đối chiếu các truy vấn có ý nghĩa thực tế

**Q19: aqueous sodium-ion + water-in-salt.** Nhánh reranker đưa các bài water-in-salt cụ thể lên đầu; ba kết quả đầu trong lần rà soát có dấu hiệu phù hợp trực tiếp. Tuy nhiên, kết quả thứ tư về aromatic sulfonate có Abstract mô tả điện giải 0,5 m, chưa có bằng chứng rõ cho điều kiện water-in-salt. Đây là ví dụ tốt để thấy hệ thống tìm được ứng viên hữu ích nhưng vẫn cần kiểm tra điều kiện cốt lõi; không tự chuyển nhận xét này thành P@5 đã được chuyên gia chấm.

**Q08: NaFSI + ether + nhiệt độ thấp.** Một số tài liệu đáp ứng ether/nhiệt độ thấp nhưng chưa nêu NaFSI; tài liệu khác liên quan muối NaFSI nhưng chưa chứng minh đúng dung môi và điều kiện nhiệt độ. Required terms `NaFSI;ether` tăng độ chặt về từ vựng nhưng chưa chứng minh các điều kiện cùng thuộc hệ thí nghiệm cần tìm.

**Q23: Na2FeP2O7.** Token filter giúp tập trung vào công thức có xuất hiện, nhưng chưa xác nhận mọi kết quả đúng vai trò vật liệu và thuộc tính người dùng quan tâm. Hướng cải thiện phù hợp là đánh giá tách trục vật liệu–vai trò–điều kiện–quan hệ, không chỉ bổ sung thêm từ đồng nghĩa.

Chi tiết định danh và bằng chứng ở [SEARCH_RESULTS_REVIEW](SEARCH_RESULTS_REVIEW.md). Các ca này đã được quan sát trong development nên không dùng làm test độc lập cho thay đổi lấy cảm hứng từ chúng.

## 9. Code đã tối ưu chưa?

### 9.1. Những tối ưu đã có bằng chứng

| Phạm vi | Thay đổi/quan sát | Kết luận được phép |
|---|---|---|
| Correctness | Guard tham số, RRF, passage budget/stride, tên Success@k, frozen cache và artifact identity | Giảm lỗi biên và sai diễn giải đã xác định |
| Top-K kernel | Partition theo cutoff, giữ ties, sort theo score/doc_id như reference | Tập final: thời gian kernel mới/cũ 0,05300, giảm khoảng 94,7% |
| Required-term filter kernel | Membership hai dictionary thay tạo union keys | Tập final: tỷ lệ 0,15760, giảm khoảng 84,2% |
| CPU/batch | Lưới threads {1,2,4} × batch {4,8,16}; 3 dev queries, 2 lượt/query | 4 threads/batch 4 tốt nhất quan sát trong lưới trên máy đã đo |
| Compare request | Tái dùng embedding/scores cho các methods chung query | Hai query × ba methods: 6 danh sách giữ nguyên; BGE calls 2→1 và BM25 field calls 6→2 mỗi comparison |
| Cache index | Fingerprint theo tokenizer/config/corpus liên quan; live không nạp benchmark query cache | Tránh invalidation/nạp cache không cần thiết do thay đổi ngoài lexical logic |

Hai tỷ lệ kernel không được cộng/nhân hoặc quy thành phần trăm tăng tốc toàn search. Reranker có chi phí riêng và vẫn phải chạy các cặp query–passage; tối ưu Top-K nhỏ không tự loại bỏ nút thắt neural inference. Trong trường hợp quá nhiều cutoff ties, tối ưu Top-K vẫn có thể phải sort nhiều ứng viên.

### 9.2. Thời gian vận hành và giới hạn phép đo

Lưới CPU có 54 requests được đo và 9 warm-ups không tính. Baseline 4:4 có median **4,506 giây/request**; không ứng viên thay thế nào đạt gate giảm ít nhất 10% và không làm query riêng chậm quá 10%. Có một lượt 1:4 ghi 839,663 giây, được giữ trong raw data, chưa xác định nguyên nhân. Đây là phép đo nhỏ trên một máy, không phải p95 hoặc cam kết SLA.

Thử nghiệm runtime cache mới ghi hai comparison giảm 9,598→8,980 và 8,719→8,158 giây, cùng model đã warm và kết quả bằng nhau. Mẫu này chỉ minh họa tác vụ lặp được loại bỏ; chưa là benchmark latency đại diện hoặc chứng minh mức tăng tốc chung.

[verification.json](../outputs/search_runtime_optimization_20261019/verification.json) ghi **311 tests pass, 0 failures/errors/skips** tại commit `053cf32aabe45ed61fe432f65f53b89cc51c282f`, nhánh `arbor/runtime-cache`. Rà soát ngày 20/10 đối chiếu SHA-256 khớp cả bốn source/test files được liệt kê trong biên bản. Con số này là kết quả test đã lưu, không phải một lần chạy lại full suite khi viết báo cáo, cũng không là 311 ca relevance độc lập.

Nguồn: [protocol tối ưu](../outputs/math_optimization_20261019/PROTOCOL.md), [execution summary](../outputs/math_optimization_20261019/execution_dev/summary.json), [runtime RESULTS](../outputs/search_runtime_optimization_20261019/RESULTS.md). Phụ lục giữ raw/gate/final và kết quả âm để không chỉ trình bày ứng viên thuận lợi.

### 9.3. “Tham số tốt nhất” hiện có nghĩa gì?

| Nhóm tham số | Trạng thái |
|---|---|
| Threads=4, reranker batch=4 | Tốt nhất quan sát trong lưới CPU đã thử; không đảm bảo ở máy/workload khác |
| BM25 k1=1,5; b=0,75; Title weight=2 | Baseline thiết kế; chưa chọn bằng relevance labels độc lập |
| Candidate depth=100; rerank depth=50; RRF k=60 | Trade-off thiết kế; chưa có Pareto quality/latency được xác lập |
| Passage=320, overlap=64, tối đa 2 | Giới hạn chi phí; chưa chứng minh không bỏ bằng chứng cần thiết |
| LDA K=20 | Được chọn trong tập K và protocol đã khảo sát |
| UMAP/HDBSCAN và chunk pooling | Có kết quả thực nghiệm, chưa tối ưu đa seed/ablation |
| Fine-tuning lr/temperature/epochs | Một cấu hình nghiên cứu đã chạy; chưa có search không gian tham số đủ để gọi tối ưu |

Không thể chọn “bộ tham số tốt nhất” duy nhất khi mục tiêu còn gồm độ đúng, thời gian, RAM, độ ổn định và khả năng kiểm tra evidence. Trước tiên cần chốt mục tiêu chính và giới hạn vận hành, sau đó mới so sánh các cấu hình trên cùng protocol.

## 10. Có phương pháp tốt hơn không?

Có các hướng phù hợp để thử, nhưng chưa có phương pháp nào được chứng minh tốt hơn cho corpus này chỉ nhờ tên model, số tham số hoặc thành tích benchmark ngoài. Khác biệt giữa miền và nhiệm vụ là lý do cần đánh giá tại chỗ; [BEIR](https://arxiv.org/abs/2104.08663) cung cấp một khung tham khảo về đánh giá retrieval qua nhiều miền.

| Hướng | Vấn đề nhắm tới và lý do | Chi phí/rủi ro | Bằng chứng cần trước khi thay mặc định |
|---|---|---|---|
| Truy hồi từng passage rồi gộp tài liệu | Giữ đoạn chứa điều kiện thay vì mean toàn Abstract; đáng thử khi query chi tiết | Nhiều vectors hơn, bias tài liệu dài, phải gộp/de-duplicate | Candidate coverage và nDCG@5 trên queries nhiều điều kiện, cùng ngân sách |
| Chọn passage bằng hybrid thay chỉ lexical overlap | Giảm bỏ sót đoạn paraphrase phù hợp cho reranker | Thêm inference/index và thêm tham số | Passage evidence labels, lỗi bỏ bằng chứng, latency tổng |
| Multi-positive fine-tuning + hard negatives đã kiểm tra | Gần nhu cầu relevance hơn pseudo-query một positive | False negatives, chi phí nhãn và rò rỉ nếu chọn sai split | Dev có nhãn, grouped/temporal test, nhiều training seeds |
| Reranker phù hợp miền hơn hoặc fine-tune reranker | Phân biệt vật liệu, điều kiện và quan hệ kỹ thuật | Có thể tăng độ trễ; model lớn không bảo đảm đúng hóa học | Qrels độc lập, error axes và latency/RAM |
| Learned sparse retrieval, như [SPLADE](https://arxiv.org/abs/2107.05720) | Mở rộng từ vựng nhưng giữ biểu diễn sparse | Expansion có thể làm loãng công thức, cần index/model mới | So BM25 trên rare formulas/paraphrases, kiểm tra provenance |
| Late interaction, như [ColBERTv2](https://arxiv.org/abs/2112.01488) | Giữ biểu diễn mức token thay vì chỉ một vector tài liệu | Lưu nhiều vectors, thay retrieval/index, vận hành phức tạp hơn | Quality/latency/memory trên cùng query set; không chỉ cosine |
| Encoder đa ngôn ngữ, như [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Hỗ trợ nhu cầu query tiếng Việt/Anh và văn bản dài nếu thật sự cần | Re-encode corpus, khác resource budget; công thức vẫn cần kiểm tra | Bộ query song ngữ tương đương, kiểm soát dịch công thức |
| ONNX/OpenVINO hoặc quantization | Giảm chi phí inference CPU khi bottleneck được xác nhận | Sai lệch số có thể đổi hạng; hiệu quả phụ thuộc phần cứng | Benchmark độc lập có warm/cold, rank/score/relevance guards |
| ANN/vector service | Khi corpus/throughput vượt khả năng exact scan | Thêm dịch vụ, tham số, ANN recall loss và bảo trì | Scale test cho thấy exact scan là bottleneck, đo recall so exact |

SPLADE và ColBERT là các họ retrieval thay đổi cách biểu diễn/so khớp; không phải cờ có thể bật để tự cải thiện code hiện tại. BGE-M3 là một ứng viên cho nhu cầu đa ngôn ngữ, chưa phải đề nghị thay model ngay. Hướng tăng tốc ONNX/OpenVINO có tài liệu chính thức tại [Sentence Transformers efficiency](https://sbert.net/docs/sentence_transformer/usage/efficiency.html); cần benchmark trên máy đích thay vì dùng tốc độ GPU từ model card để dự báo CPU dự án.

Ưu tiên hợp lý hiện tại là sửa/đo khả năng đáp ứng **đồng thời các điều kiện**, rồi thử passage retrieval và passage selection. Chưa có lý do đủ mạnh để đầu tư hạ tầng vector service hoặc model lớn trước khi biết lỗi nằm ở candidate generation, representation hay reranking.

## 11. Thiết kế đánh giá và tối ưu tiếp theo

### 11.1. Tạo ground truth phù hợp câu hỏi sử dụng

Chốt intent và rubric cho từng query trước chấm, gồm hệ pin, vật liệu/công thức, vai trò, điều kiện và quan hệ kỹ thuật. Dùng thang grade 0–3; mô tả rõ khi nào là trực tiếp đáp ứng, một phần hoặc chỉ nền tảng. Hai người chấm độc lập trên danh sách trộn không lộ method/rank; ghi evidence và lý do. Tính agreement trước phân xử, sau đó lưu consensus có provenance. Dùng nhãn AI để gợi ý review khi cần, không coi nó tự tạo ra tính độc lập.

Pool hiện có hỗ trợ bắt đầu review nhưng còn pooling bias. Nếu thêm retriever mới, phải bổ sung nhãn cho các ứng viên mới theo cùng quy trình; không cho tài liệu chưa chấm mặc định bằng 0. Muốn đánh giá Recall toàn corpus cần thiết kế bổ sung để tìm relevant ngoài pool; pool942 hiện chưa cho phép kết luận đó.

### 11.2. Dùng metric đúng nghĩa

`P@5 = số tài liệu grade≥2 trong Top5 / 5`; danh sách thiếu kết quả vẫn chia 5 theo protocol. `MRR@5` dùng nghịch đảo hạng relevant đầu tiên, không có thì 0. `DCG@5 = Σ_r (2^grade_r−1)/log2(r+1)` và `nDCG@5 = DCG/IDCG`, với IDCG từ pool chung, quy ước IDCG=0 trả 0. Grade 1 vẫn đóng góp nDCG nhưng chưa qua ngưỡng P/MRR.

Ngoài Top5, đo candidate success/coverage ở Top100 và Top50 với relevant đã biết, passage evidence coverage, abstention và lỗi theo nguồn/điều kiện. Nếu relevant mất trước rerank thì tăng sức mạnh reranker không giải quyết nguyên nhân. Mỗi query hoặc query family là đơn vị so sánh thống kê; không coi từng hàng kết quả là một mẫu độc lập.

### 11.3. Quy trình chọn tham số có thể kiểm chứng

1. Định nghĩa workload và ngân sách máy: một người hay nhiều người, yêu cầu thời gian chờ, RAM và chi phí offline.
2. Tách dev, gate và test theo query family; test mới không dùng để lựa chọn sửa đổi. Dữ liệu sẵn có đã được xem phải ghi nhận mức exposure.
3. Chạy baseline cố định; ghi hashes corpus/model/query/config, seeds, phiên bản thư viện và raw timings.
4. Thử một giả thuyết mỗi lần: candidate depth, rerank depth, passage selection, pooling hoặc gate. Cố định các yếu tố còn lại để diễn giải được nguyên nhân.
5. Chọn theo nDCG@5 có nhãn dev cùng ràng buộc latency/RAM; đọc P@5, MRR và lỗi nghiêm trọng để tránh cải thiện trung bình nhưng làm hỏng nhóm query quan trọng.
6. Dùng paired bootstrap theo query family để mô tả uncertainty; số family ít cần công bố CI rộng và kết quả từng query. Không dò nhiều lần trên test rồi vẫn gọi test độc lập.
7. Khóa ứng viên, chạy gate/test đã chốt, công bố cả cải thiện và regression. Nếu thử nhiều cấu hình, ghi rõ phạm vi và kiểm soát diễn giải multiple comparisons.

Lưới relevance tiếp theo chỉ nên là tập ứng viên nhỏ, phù hợp ngân sách, chẳng hạn rerank depth {20,50,100} hoặc passages {1,2,4} trong các thí nghiệm riêng. Đây là **đề xuất**, chưa phải kết quả tối ưu và chưa có phiên chạy mới trong báo cáo này.

### 11.4. Thứ tự ưu tiên

| Ưu tiên | Công việc | Điều kiện hoàn thành |
|---|---|---|
| 1 | Hoàn thiện labels độc lập và query families mới | Báo cáo coverage, agreement, consensus và provenance; không còn trộn AI/human metrics |
| 1 | Định vị lỗi nhiều điều kiện và mất ứng viên | Có error breakdown cho candidate/passage/rerank và ví dụ evidence |
| 2 | Passage retrieval/selection, lexical support và rerank budget | Ablation trên cùng dev labels; quality/latency trade-off rõ |
| 2 | Topic preprocessing version mới, pooling và nhiều seeds | Không ghi đè baseline; review inlier/outlier và stability |
| 3 | Fine-tuning theo relevance và hard negatives | Grouped held-out evaluation, nhiều seeds và baseline pretrained |
| 3 | Inference backend/scale/concurrency | Đo cold/warm, queue, RAM, throughput và tail latency trên workload xác định |

## 12. Kết luận sử dụng và mức độ hoàn thành

**Search đã ổn để hỗ trợ đọc và tìm ứng viên trong corpus hiện tại.** Pipeline có lexical, dense, hybrid, reranker, filters và evidence để kiểm tra; nhiều lỗi biên đã được sửa và có kiểm chứng kỹ thuật. Tìm kiếm có các ví dụ hữu ích và các proxy cho thấy biểu diễn học được tín hiệu.

**Search chưa được chứng minh đủ tốt để tin mọi kết quả đầu hoặc chọn cấu hình relevance tối ưu.** Hai bảng chấm độc lập chưa có grade; AI relevance mới đủ 2/23 queries; source recovery có phần lớn source documents thuộc train và khác mục tiêu relevance. Query nhiều điều kiện vẫn có kết quả thiếu bằng chứng đồng thời. Cần giữ trạng thái `PENDING_INDEPENDENT_LABELS` và `DESIGN_DEFAULT_NOT_RELEVANCE_SELECTED` cho đến khi có bằng chứng phù hợp.

**Code đã tối ưu một số thành phần, chưa tối ưu toàn hệ thống và mọi tham số.** 4 threads/batch4 là lựa chọn có đo trong phạm vi nhỏ; các tối ưu kernel/cache giữ kết quả trong các phép đối chiếu đã lưu. Chưa có benchmark đủ rộng cho concurrency hoặc giới hạn thời gian phục vụ. Lựa chọn tiếp theo phải dựa trên nhãn relevance và bottleneck thực tế, thay vì mặc định thêm model, chunk hoặc dịch vụ là tốt hơn.

## 13. Tái lập, hồ sơ bằng chứng và nguồn phương pháp

Chạy giao diện hiện hành từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe run_demo.py
```

App dùng cấu hình [search_v2.json](../configs/search_v2.json). Không cần huấn luyện lại các Stage để mô tả những kết quả đã lưu. Các lệnh đo/huấn luyện hoặc tiếp tục AI job phải được coi là thí nghiệm riêng với output riêng, không là bước tự động của việc đọc báo cáo.

| Bằng chứng | Vị trí |
|---|---|
| Quy mô và quyết định dữ liệu | [Stage1 summary](../outputs/reports/stage1_summary.json), [cleaning summary](../outputs/reports/cleaning_summary.csv) |
| LDA và cấu hình được chọn | [Stage2 reports](../outputs/reports/stage2/), [stage2.yaml](../configs/stage2.yaml) |
| Split, fine-tuning và BERTopic baseline | [Stage3 reports](../outputs/reports/stage3/), [stage3.yaml](../configs/stage3.yaml) |
| Chunked representation và so sánh topics | [Chunked outputs](../outputs/bertopic_chunked/) |
| Rankings và biểu mẫu nhãn độc lập | [Search upgrade](../outputs/search_upgrade_20261018/) |
| AI relevance và source recovery | [AI evaluation report](../outputs/retrieval_ai_evaluation_v1/REPORT.md), [synthetic summary](../outputs/retrieval_ai_evaluation_v1/synthetic/summary.csv) |
| Công thức, audit và tối ưu thành phần | [Math optimization](../outputs/math_optimization_20261019/) |
| Runtime cache và 311 tests đã lưu | [Runtime verification](../outputs/search_runtime_optimization_20261019/verification.json) |
| Bản báo cáo trước lần viết lại | [source_before](../outputs/report_rewrite_20261020/source_before/) |

Nguồn học thuật và tài liệu chính thức đã được liên kết ngay tại phần giải thích tương ứng: LDA, BERTopic, BGE, RRF, MiniLM/MS MARCO, retrieve–rerank, BEIR, SPLADE, ColBERTv2 và các backend inference. Các nguồn đó giải thích phương pháp; số liệu của dự án phải tra artifacts của dự án, không lấy benchmark công bố của model thay cho đánh giá local.

Lần rà soát này áp dụng skill scientific-critical-thinking để phân biệt quan sát, suy luận và đề xuất. Ghi nhận nguồn phương pháp: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065v2](https://arxiv.org/abs/2609.00065v2). Skill hỗ trợ quy trình phản biện, không cung cấp thêm nhãn hoặc xác nhận hiệu quả của hệ thống.
