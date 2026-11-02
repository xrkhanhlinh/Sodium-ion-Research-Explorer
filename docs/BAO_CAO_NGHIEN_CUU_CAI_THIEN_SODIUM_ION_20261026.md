# Khám phá chủ đề và tìm kiếm tài liệu pin sodium ion sau rà soát và cải thiện

Báo cáo nghiên cứu sau rà soát và cải thiện

Sodium ion Research Explorer
Thí nghiệm phiên bản 25/10 được tiếp tục và tổng hợp ngày 26/10/2026

Nghiên cứu xây dựng hệ thống tổ chức chủ đề và truy hồi trên 13.702 tài liệu Patent và Publication. Bản này trình bày pipeline, cơ sở mô hình, kết quả đối chứng và các thay đổi đã được kiểm tra sau khi rà soát từ khóa, cụm chủ đề và Search Top 10.

| Kết quả được chấp nhận | Kết quả không được chấp nhận |
|---|---|
| KMeans50: gate ARI 0,6866 → 0,9918; NPMI 0,0792 → 0,1100 trên protocol gốc | Năm biến thể lọc từ khóa đều trượt ít nhất một tiêu chí |
| Parser mới bảo toàn công thức; NVP có đủ Na3V2(PO4)3 | Reranker Top30 tăng dev nhưng giảm gate: 0,9271 → 0,8507 |

Embeddings và checkpoint encoder hợp lệ được giữ nguyên; không huấn luyện lại hoặc encode lại corpus. Pipeline có trạng thái, logs và checkpoint để bỏ qua phần đã hoàn thành sau gián đoạn. Báo cáo giữ cả kết quả thất bại để giới hạn các kết luận.

Dữ liệu là toàn bộ Title–Abstract đang có, không phải toàn văn bài báo. Nhãn AI Silver pilot20 chưa phải nhãn con người, không dùng để chọn model và chưa đủ để công bố relevance tổng thể.

## 1 Câu hỏi nghiên cứu và các điểm được sửa

| Câu hỏi | Bằng chứng dùng trả lời | Phạm vi kết luận |
|---|---|---|
| RQ1 Topic có ổn định và diễn giải được không | Seed agreement, coherence, kích thước, nguồn và tài liệu đại diện | Cấu trúc thống kê; chưa có expert topic ground truth |
| RQ2 Đọc đủ Abstract và continuation cải thiện gì | Độ bao phủ và proxy Title–document cố định | Không đồng nhất với relevance query tự nhiên |
| RQ3 Search đưa bài phù hợp lên đầu không | Relevance0–2, benchmark giữ riêng, phân tích lỗi | Nhãn người còn thiếu; có pilot AI riêng |

Bản LaTeX trước chốt ngày 19/10 mô tả Search v2 Top5 và BERTopic84/92 topic. Bản hệ thống ngày 24/10 đã có Top10 và nhiều catalog K10. Bản này cập nhật luồng hiện hành; các số lịch sử chỉ được trình bày đúng snapshot và protocol của chúng.

LDA Stage2 chọn K20 được tách khỏi LDA K10 dùng đối chứng. BERTopic được tách khỏi encoder BGE/BatteryBERT và khỏi KMeans. C_v, ARI, coverage, cosine, RRF và relevance được định nghĩa riêng, tránh dùng một con số thay thế các mục tiêu khác.

Các phần API quota, AI consensus cũ, synthetic recovery và hàng chục phương pháp lịch sử được đưa về nguồn tham chiếu; không dùng chúng làm bằng chứng chính cho cấu hình hiện hành. Đóng góp được xác định là hệ thống ứng dụng và kiểm chứng so sánh, không tuyên bố một thuật toán học máy mới.

Từ khóa có vẻ chuyên biệt hơn chưa đủ để chứng minh topic tốt hơn. Vì vậy các thay đổi về clustering và keywords được thử riêng, giữ các yếu tố khác cố định khi có thể.

## 2 Pipeline toàn hệ thống và đầu ra

![Hình 1. Pipeline tổng thể giữ hai mục tiêu topic và Search riêng biệt.](../outputs/report_pipeline_20261025/figures/01_pipeline.png)
Hình 1. Pipeline tổng thể giữ hai mục tiêu topic và Search riêng biệt.
Nguồn output: src/pipeline.py; src/final_search.py; src/revised_topics.py

Excel → làm sạch và sàng lọc → corpus đóng băng. Nhánh BoW tạo LDA; nhánh neural tạo embedding rồi phân cụm bằng BERTopic hoặc KMeans. Search kết hợp TF-IDF và vector BatteryBERT trong Qdrant bằng RRF60. Topic chỉ hỗ trợ duyệt và lọc theo doc_id, không cộng điểm vào ranking.

| Giai đoạn | Output hiện hành |
|---|---|
| Dữ liệu | main_corpus.parquet; 13.702 doc_id duy nhất |
| Encoder | Checkpoint full-document đã chấp nhận; ma trận 13.702 × 768 |
| Clustering | Source projection train-fit; KMeans10 n_init50 |
| Keywords | 100 từ khóa, evidence và audit; parser v2 giữ công thức |
| Search | Top10 hybrid mặc định; reranker thử nghiệm bị loại |

## 3 Dữ liệu và giao thức thực nghiệm

| Nguồn | Đầu vào | Corpus chính |
|---|---|---|
| Patent | 8.657 | 5.933 |
| Publication | 11.919 | 7.769 |

| Split | Tài liệu | Vai trò |
|---|---|---|
| Train | 10.921 | Optimizer và fit geometry, vocabulary, KMeans |
| Validation | 1.361 | Phát triển và chọn cấu hình |
| Test | 1.368 | Đo sau lựa chọn; split đã dùng trong lịch sử |
| Title only | 52 | Chỉ inference |

Giữ nguyên dữ liệu gốc, các trường Title/Abstract, nguồn, năm, identifier và split_group_id. Các nhãn phạm vi CORE/RELATED là heuristic và có thể giữ tài liệu đa hệ pin, phần cứng hoặc sodium-metal. Không xóa tài liệu sau khi nhìn kết quả để làm tăng coherence.

Thí nghiệm geometry dùng seed 42/43/47 trên validation; coordinator kiểm tra một seed mới20261029 trên test lịch sử. Keyword experiments giữ một tập nhóm validation riêng cho gate, nhưng không ứng viên nào qua dev nên không mở gate ứng viên. Search dùng 16 title anchors phát triển và 16 anchors giữ riêng, không điều chỉnh query Q01–Q20 đã khóa.

Các tập vẫn lấy từ corpus và split lịch sử. Gate mới trong phiên sửa này không trở thành external test độc lập. Các proxy lexical-summary và Title-pair đều có giới hạn, không được gọi là accuracy khoa học hoặc relevance người dùng.

## 4 LDA và BERTopic trong bài toán chủ đề

![LDA mô hình hóa tài liệu bằng hỗn hợp topic và topic bằng phân bố từ](../outputs/report_pipeline_20261025/figures/eq_lda.png)
LDA mô hình hóa tài liệu bằng hỗn hợp topic và topic bằng phân bố từ


LDA nhận BoW, giả định mỗi tài liệu có tỷ trọng θ trên các topic và mỗi topic có phân bố từ φ. Top10 words lấy theo P(w|topic); tài liệu vẫn giữ posterior nhiều chủ đề dù giao diện hiển thị topic trội nhất. α và η là prior Dirichlet [1].

Stage2 thử K10/15/20/25/30, passes10, iterations100, alpha asymmetric, eta symmetric. Hai K có C_v tốt ở seed cơ sở được đánh giá thêm stability; điểm tổng hợp chọn K20. Đây là lựa chọn trong lưới đã thử, không phải tối ưu toàn cục.

| LDA Stage2 K20 | Giá trị |
|---|---|
| C_v | 0,5716 |
| Diversity | 0,7400 |
| Stability Jaccard sau ghép Hungarian | 0,2620 |

BERTopic dùng encoder → UMAP5 chiều → HDBSCAN → c-TF-IDF để mô tả từ. Cụm do mật độ quyết định và có outlier −1. Bảng đối chứng K10 gộp các topic sau phân cụm, không phải HDBSCAN được yêu cầu tạo đúng10 cụm [2].

![c-TF-IDF với reduce_frequent_words bật trong nhánh BERTopic](../outputs/report_pipeline_20261025/figures/eq_ctfidf.png)
c-TF-IDF với reduce_frequent_words bật trong nhánh BERTopic


UMAP15 neighbors, min_dist0, cosine; HDBSCAN min_cluster_size30, min_samples10, eom. Các từ khóa mô tả cụm sau khi clustering, nên thay representation không tự cải thiện membership. UMAP2D chỉ để minh họa.

## 5 BGE BatteryBERT và huấn luyện đã hoàn thành

![Attention tạo biểu diễn token phụ thuộc ngữ cảnh trong từng chunk](../outputs/report_pipeline_20261025/figures/eq_attention.png)
Attention tạo biểu diễn token phụ thuộc ngữ cảnh trong từng chunk


BGE và BatteryBERT là encoder; BERTopic và KMeans là cách sử dụng embedding để tạo topic. Các checkpoint BGE-base-en-v1.5 và BatteryBERT-cased được ghi trong cấu hình [3,4].

![Pooling các token Abstract rồi tổng hợp chunk có trọng số token mới](../outputs/report_pipeline_20261025/figures/eq_pool.png)
Pooling các token Abstract rồi tổng hợp chunk có trọng số token mới


BatteryBERT full-document dùng cửa sổ256, overlap64, Title làm context nhưng không trực tiếp pool vào chunk Abstract. Trọng số chunk là số token mới được bao phủ; overlap vẫn tác động ngữ cảnh. Đọc đủ các cửa sổ không tạo attention toàn cục giữa mọi đoạn.

![Contrastive loss theo batch; code lấy trung bình hai chiều và thêm distillation](../outputs/report_pipeline_20261025/figures/eq_loss.png)
Contrastive loss theo batch; code lấy trung bình hai chiều và thêm distillation


| Huấn luyện đã được kiểm tra | Giá trị |
|---|---|
| Block cập nhật | Block cuối; 11 block đóng băng |
| Loss batch / temperature | 32 cặp / 0,05; gradient caching hai tầng |
| Lượt train hoàn tất | 10.921 tài liệu; 26.147 chunks; 3.420.215 token Abstract |
| Checkpoint | Step342; learning rate5e-6 |
| Test proxy trước / sau | 0,887839 / 0,924704 |

Model weights, ordered IDs, vector hash, finite values và chuẩn L2 đã được xác nhận lại khi resume. Không lặp train hoặc embeddings. Điểm proxy có Title context không được diễn giải thành92,47% Search accuracy [5].

## 6 Đối chứng lịch sử và vấn đề cấu trúc topic

| Catalog K10 lịch sử | C_v | Diversity | Outlier |
|---|---|---|---|
| LDA | 0,5379 | 0,6600 | N/A |
| BERTopic–BGE | 0,6455 | 0,8600 | 22,77% |
| BERTopic–BatteryBERT (pretrained) | 0,7445 | 0,9200 | 0,80% |

BERTopic–BatteryBERT pretrained có C_v lịch sử cao nhưng hai cụm chứa7.261 và 5.449 tài liệu, tức92,76% corpus. Hai nhóm rộng phản ánh publication về hiệu năng và patent về phương pháp chế tạo. Điều này minh họa vì sao coherence cao chưa đủ chọn một model topic.

BERTopic–BGE tách được nhiều nhóm vật liệu cụ thể hơn nhưng có 3.120 outlier. KMeans có 0 outlier do bắt buộc gán mọi tài liệu; đây không phải bằng chứng vượt trội. LDA posterior, HDBSCAN membership và KMeans distance là các đại lượng khác nhau.

Bản optimized23/10 là đối chứng bổ sung. Đo lại bằng cùng recipe và seed phát triển cho ARI0,8455, NPMI0,08390, source V0,3103, diversity0,80 và silhouette0,1610. Model mới cải thiện một số mục tiêu, nhưng không thắng silhouette và diversity của đối chứng này.

C_v trong bảng là protocol lịch sử. Không đặt nó cùng C_v canonical mới để kết luận một thứ hạng tuyệt đối giữa mọi model. Các catalog gốc vẫn được lưu đầy đủ để tái lập.

## 7 Chạy lại clustering và kiểm tra giữ riêng

![Hình 2. Nhiều khởi tạo KMeans cải thiện stability, giữ cùng embeddings và keyword recipe gốc.](../outputs/research_revision_20261025_v1/visualizations/geometry_before_after.png)
Hình 2. Nhiều khởi tạo KMeans cải thiện stability, giữ cùng embeddings và keyword recipe gốc.
Nguồn output: comparison/geometry_comparison.csv

| Cấu hình dev | ARI nhiều seed | NPMI | Kết luận |
|---|---|---|---|
| baseline_source_n10 | 0,7435 | 0,0813 | Đạt |
| source_n50 | 0,9949 | 0,0949 | Đạt |
| source_whiten32_n50 | 0,9702 | 0,0556 | Không đạt |
| source_whiten64_n50 | 0,9183 | 0,0208 | Không đạt |

Source projection fit trên train loại một hướng chênh lệch trung bình nguồn. KMeans 10 n_init50 được chọn trên dev; source-whiten32/64 có stability cao nhưng giảm NPMI, và 64 chiều còn tạo cụm30,71% validation nên bị loại.

| Gate seed20261029 | Baseline n_init10 | n_init50 |
|---|---|---|
| ARI | 0,686587 | 0,991808 |
| Document NPMI | 0,079191 | 0,110046 |
| Source Cramér V | 0,3254 | 0,3291 |
| Top2 share | 23,10% | 27,49% |

Source V và top2 share trên gate tăng nhẹ, dù vẫn trong giới hạn đã chốt. Kết luận được chấp nhận là stability và coherence theo protocol gốc cải thiện, không phải mọi chỉ số đều tốt hơn.

## 8 Catalog mới và ý nghĩa của các nhóm

![Hình 3. Phân bố source sau clustering mới; phần cứng vẫn là một nội dung thật cần gọi tên đúng.](../outputs/research_revision_20261025_v1/visualizations/revised_topic_sources.png)
Hình 3. Phân bố source sau clustering mới; phần cứng vẫn là một nội dung thật cần gọi tên đúng.
Nguồn output: final_topics/topic_source_distribution.csv

| ID | Tên diễn giải sau khi đọc đại diện | Tài liệu |
|---|---|---|
| 0 | Anode oxide titan và titanate | 1.265 |
| 1 | Prussian blue và vật liệu điện cực chứa sắt | 1.511 |
| 2 | Cathode oxide lớp và biến đổi cấu trúc pha | 2.065 |
| 3 | Điện ly polymer và giao diện trong các hệ pin natri | 1.612 |
| 4 | Cathode phosphate vanadium và NASICON | 1.351 |
| 5 | Anode chứa Sn và Sb cùng composite chuyển hóa | 1.012 |
| 6 | Cấu trúc cell bộ góp dòng và thiết bị pin | 1.071 |
| 7 | Anode MoS2 và vật liệu hai chiều graphene MXene | 1.046 |
| 8 | Hard carbon và tiền chất sinh khối | 1.663 |
| 9 | Sulfide selenide kim loại chuyển tiếp và composite carbon | 1.106 |

Căn chỉnh ID bằng Hungarian trên overlap tài liệu train chỉ giữ tính tiện dụng hiển thị; không chứng minh ý nghĩa topic bất biến. ID7 chỉ giữ khoảng1,06% overlap theo phép ghép bắt buộc và nay là nhóm MoS2/MXene; tên pha/vacancy cũ đã được thay.

Tên nhóm là diễn giải AI từ 100 keywords và 10 Abstract đại diện trung tâm đã đọc đầy đủ.60 đại diện được xuất để rà soát tiếp; chưa có expert topic labels.

## 9 Bảo toàn công thức và kiểm chứng từ khóa

| Trường hợp gốc | Xử lý phiên bản mới |
|---|---|
| Na3V2(PO4)(3) | Na3V2(PO4)3; giữ raw span và phép biến đổi |
| Na(3)V(2)(PO(4))(3) | Chuẩn hóa ngoặc số không có khoảng trắng khi cú pháp hỗ trợ |
| TiO2 (101) hoặc TiO2(101) | Không nối thành TiO2101; giữ chú thích hoặc span mơ hồ |
| Na(1) / Fe[Fe(CN)(6)] | Giữ nguyên dạng site label/complex chưa hỗ trợ; không xuất fragment bên trong như toàn công thức |

Parser cũ có thể cắt Na3V2(PO4)(3) thành Na3V2(PO4). Parser v2 giữ complete spans, phân biệt case, ranh giới field, stopword và dấu câu; chỉ chuẩn hóa cách ghi số khi đủ điều kiện cú pháp. Unsupported không có nghĩa công thức sai hóa học.

Đã xử lý27.404 trường Title/Abstract qua 54 shard;41.002 span giống công thức,2.340 chuẩn hóa thận trọng và 15.702 span chưa hỗ trợ được giữ nguyên. Có 806 raw Na3V2(PO4)(3) equivalences;25 chú thích số cách trắng không bị ghép. Không thay đổi raw corpus.

![Hình 4. Cùng clustering và canonical reference để so danh sách trước/sau sửa ký hiệu.](../outputs/research_revision_20261025_v1/visualizations/canonical_correctness_effect.png)
Hình 4. Cùng clustering và canonical reference để so danh sách trước/sau sửa ký hiệu.
Nguồn output: final_topics/keyword_comparison.csv; common_coherence.csv

Test NPMI tăng 0,065294 →0,096332 và C_v cùng reference tăng 0,547907 →0,561193; diversity giảm 0,79 →0,75. Đây là chẩn đoán representation sau sửa cú pháp, khác số NPMI0,1100 của geometry gate trên stream gốc.

## 10 Thử lọc từ nhiễu và lý do không chọn

![Hình 5. Các danh sách ít từ chung có thể mất coverage và NPMI dù F1 tăng.](../outputs/research_revision_20261025_v1/visualizations/keyword_failed_hypotheses.png)
Hình 5. Các danh sách ít từ chung có thể mất coverage và NPMI dù F1 tăng.
Nguồn output: comparison/keyword_experiments.csv

| Recipe | F1 proxy | NPMI | Coverage | Quyết định |
|---|---|---|---|---|
| baseline_dev | 0,5371 | 0,0587 | 0,9706 | Giữ baseline |
| contrastive_df | 0,5733 | -0,4611 | 0,5164 | Loại |
| title_evidence | 0,5739 | -0,5165 | 0,5252 | Loại |
| contrastive_phrases | 0,5378 | -0,6019 | 0,4655 | Loại |
| conservative_tf1 | 0,5172 | 0,0462 | 0,9705 | Loại |
| conservative_sqrt_tf | 0,5812 | -0,2559 | 0,6918 | Loại |

Cycle1 thử ba cách ưu tiên specificity/cụm từ; cycle2 thử hai cách giữ TF và loại thủ tục/alias thận trọng. Cả năm đều trượt ít nhất một gate đã chốt. Không mở gate ứng viên rồi chỉnh tiếp để tìm số đẹp.

Các từ solution/dry/type thường yếu khi đặt tên, nhưng carbon/anode/oxide có thể kết nối nhiều phân họ đúng trong một cụm. Do đó model giữ weighting gốc, đồng thời xuất audit cho từ rộng, từ thao tác/cấu trúc, alias và ký hiệu chưa hỗ trợ. Không che từ bằng giao diện rồi gọi đó là cải thiện mô hình.

## 11 Top 10 keywords hiện hành và đọc đúng trọng số

![Hình 6. Bốn topic minh họa với đủ10 từ khóa và trọng số thống kê.](../outputs/research_revision_20261025_v1/visualizations/revised_top10_keywords.png)
Hình 6. Bốn topic minh họa với đủ10 từ khóa và trọng số thống kê.
Nguồn output: final_topics/topic_keywords.csv

![Weighting giữ TF tương đối × IDF × exclusivity mũ0,25](../outputs/report_pipeline_20261025/figures/eq_keywords.png)
Weighting giữ TF tương đối × IDF × exclusivity mũ0,25


Vocabulary chỉ fit train, unigram/bigram, min_df5, max_df0,6, tối đa40.000 features. Mỗi từ khóa có ít nhất5 tài liệu train trong topic hỗ trợ; các phrase phải có bằng chứng liền kề theo stream giữ ranh giới.

Điểm weight xếp hạng các từ mô tả cụm; không phải xác suất từ đúng khoa học hoặc xác suất tài liệu relevant. Bảng keyword_audit.csv và keyword_evidence.csv lưu support theo split và ngữ cảnh để người đọc rà soát.

## 12 Search Top 10 và thí nghiệm reranker

![Cosine so biểu diễn; RRF60 hợp nhất hai thứ hạng](../outputs/report_pipeline_20261025/figures/eq_rrf.png)
Cosine so biểu diễn; RRF60 hợp nhất hai thứ hạng


Query → lọc nguồn/năm/required terms/topic → lấy100 ứng viên TF-IDF và 100 dense → RRF60 → Top10. Topic mới được chuyển thành danh sách doc_id được phép trước retrieval; vector index không cần encode hoặc xây lại chỉ vì nhãn topic thay đổi.

![Hình 7. Reranker Top30 được chọn ở dev nhưng không chuyển giao sang gate; không kích hoạt.](../outputs/research_revision_20261025_v1/visualizations/search_gate_rejection.png)
Hình 7. Reranker Top30 được chọn ở dev nhưng không chuyển giao sang gate; không kích hoạt.
Nguồn output: comparison/search_proxy_comparison.csv

| Proxy | Baseline | Rerank30 |
|---|---|---|
| Dev16 Group MRR@10 | 0,832813 | 0,877083 |
| Gate16 Group MRR@10 | 0,927083 | 0,850694 |
| Gate Success@1 | 0,8750 | 0,8125 |
| Gate Success@10 | 1,0000 | 1,0000 |

MiniLM đọc Abstract-only passages và mọi query window, nhưng vẫn là proxy thiên lệch vì vector/TF-IDF chứa Title. Median rerank là chi phí bổ sung, không gồm encode query; thời gian baseline bằng0 trong bảng cache không chứng minh truy hồi miễn phí. Giữ hybrid mặc định.

## 13 Output Search thực tế và phân tích điều kiện

Q01: Hard carbon anodes derived from biomass for sodium-ion batteries. Bảng giữ output hybrid đã khóa; không sửa hạng sau khi đọc nhãn pilot.

| Hạng | Tiêu đề | RRF |
|---|---|---|
| 1 | Biomass-derived carbon anodes for sodium-ion batteries | 0,03028 |
| 2 | Biomass hard carbon of high initial coulombic efficiency for sodium-ion batteries: Preparation and application | 0,02878 |
| 3 | Biochars from various biomass types as precursors for hard carbon anodes in sodium-ion batteries | 0,02731 |
| 4 | Revealing the sodium storage behavior of biomass-derived hard carbon by using pure lignin and cellulose as model precursors | 0,02650 |
| 5 | A Stable Biomass-Derived Hard Carbon Anode for High-Performance Sodium-Ion Full Battery | 0,02513 |
| 6 | High Capacity and High Efficiency Maple Tree-Biomass-Derived Hard Carbon as an Anode Material for Sodium-Ion Batteries | 0,02492 |
| 7 | Hard carbon derived for lignin with robust and low-potential sodium ion storage | 0,02432 |
| 8 | Biomass derived hard carbon used as a high performance anode material for sodium ion batteries | 0,02312 |
| 9 | Pectin, Hemicellulose, or Lignin? Impact of the Biowaste Source on the Performance of Hard Carbons for SodiumIon Batteries | 0,02256 |
| 10 | A porous biomass-derived anode for high-performance sodium-ion batteries | 0,02158 |

Các lỗi đã ghi nhận ở query nhiều điều kiện: có NaFSI nhưng không đúng ether/nhiệt độ thấp; có quy trình cell nhưng sai công đoạn cần tìm; có voltage drop nhưng chưa chứng minh thermal management. Required terms chỉ xác nhận token, không xác nhận quan hệ hoặc phủ định.

Không dùng các lỗi benchmark đã xem để viết quy tắc riêng theo query_id. Điều cần thêm là đánh giá intent có nhãn và phân tích từng điều kiện trên tập phát triển độc lập, thay vì tự tăng score cho tên vật liệu.

## 14 Đánh giá Top 10 và Silver Labels thử nghiệm

![Precision và nDCG cần nhãn relevance độc lập theo đúng protocol](../outputs/report_pipeline_20261025/figures/eq_ndcg.png)
Precision và nDCG cần nhãn relevance độc lập theo đúng protocol


| Chỉ số | Ý nghĩa | Chưa chứng minh được |
|---|---|---|
| P@10 | Số bài grade2 trong Top10 chia10 | Không có qrels thì không suy từ cosine |
| nDCG@10 | Mức liên quan và vị trí; gain2^grade−1 | IDCG theo pool chung chưa là toàn corpus |
| MRR@10 | Nghịch đảo hạng bài grade2 đầu tiên | Khác proxy ghép Title–document |
| Recall@10 | Tỷ lệ toàn bộ relevant documents tìm được | Chưa có relevant set đầy đủ nên không công bố |

![Hình 8. Pilot AI theo yêu cầu mới, không dùng để chọn model và không thay human qrels.](../outputs/research_revision_20261025_v1/visualizations/silver_pilot20.png)
Hình 8. Pilot AI theo yêu cầu mới, không dùng để chọn model và không thay human qrels.
Nguồn output: outputs/ai_silver_labels_20261026_v1/batch_0001_summary.json

Đã đọc20 cặp đầu bảng annotation_pool.csv:11 nhãn2,4 nhãn1,4 nhãn0 và 1 trống;8 cặp review_required. Cả20 đều Q01 theo thứ tự gốc, nên không đại diện20 query hoặc toàn corpus.683 cặp còn lại chưa được chấm trong pilot này.

Một patent đồng thời gọi cathode và negative electrode được để trống, NEEDS_REVIEW. Các patent nêu cathode nhất quán được chấm0 theo văn bản, confidence medium và gắn cờ kiểm tra metadata. Không tự sửa vai trò bằng kiến thức về hard carbon. Pool người vẫn có 0 nhãn xác minh.

## 15 Resume checkpoint và phần đã bỏ qua

| Khối | Kết quả thực thi khi tiếp tục |
|---|---|
| Dữ liệu và model gốc | Kiểm hash; không thay raw corpus hoặc checkpoint |
| Embeddings13.702 ×768 | Xác nhận shape, norm, ordered IDs, model/vector hash;0 forward calls mới |
| Geometry dev và keywords5ứng viên | Adopt output có sẵn; không fit lại |
| Search baseline và rerank30 dev | Adopt bundle và query cache; không inference lại |
| Search rerank50 và comparator cũ | Chỉ chạy phần còn thiếu sau gián đoạn |
| Gate và export | Gate mỗi baseline/winner một lần; export chỉ predict |
| Scientific stream | Checkpoint256documents/shard;54 shard; merge một lần |
| Đánh giá và báo cáo | Manifest theo giai đoạn; logs và trạng thái riêng |

Lệnh tiếp tục: .venv/Scripts/python.exe scripts/resume_research_revision.py. Dùng --status để đọc trạng thái; --through tên_giai_đoạn để dừng ở mốc cụ thể. Mỗi stage có pending/running/failed/completed, log thời gian/lỗi, receipt và hash của output.

Receipt hợp lệ được bỏ qua. Output hoàn thành bị đổi hash sẽ báo lỗi và giữ nguyên để kiểm tra, không tự ghi đè. Process lock kiểm PID cùng creation time, khôi phục lock stale và chờ child còn chạy thay vì khởi động trùng. Test mô phỏng ngắt xác minh partial checkpoint được giữ.

Ngày thư mục25/10 định danh phiên nghiên cứu; các stage tiếp tục26/10 ghi timestamp UTC trong event logs. Không sửa ngày của kết quả cũ để làm chúng giống mới chạy.

## 16 Kiểm chứng và trạng thái sử dụng

| Kiểm chứng | Kết quả |
|---|---|
| Tests ảnh hưởng bởi thay đổi | 102 PASS;0 failures/errors/skips |
| Artifacts gốc bảo toàn | 16 file hash không đổi |
| Catalog mới | 13.702 assignments;10 topics;100 keywords;60 representatives |
| Bật catalog revised | True |
| Bật reranker thử nghiệm | False vì trượt gate |
| Human relevance labels | 0; AI Silver20 được tách riêng |

Ứng dụng đọc catalog mới với tên diễn giải đã rà soát ở mức AI, hiển thị các cờ từ khóa và giữ các catalog lịch sử để đối chiếu. Search vẫn dùng vector và hybrid cũ; topic chỉ đổi metadata và danh sách doc_id khi lọc.

Những tests xác minh luồng dữ liệu, ranh giới cú pháp, identity, skip/resume và công thức đã mã hóa. Chúng không thay chuyên gia đọc chủ đề hoặc đánh giá relevance. Gate đạt cũng không loại bỏ mọi lỗi trong corpus hoặc mọi tài liệu gán sai.

Không đưa reranker bị loại vào phương pháp mặc định. Code và output thử nghiệm nằm riêng để tái lập; không làm bảng đánh giá sáu method lịch sử tự phát sinh thêm method mới.

## 17 Thảo luận và công việc còn cần bằng chứng

Kết quả cho thấy instability của clustering không nhất thiết yêu cầu train lại encoder: tăng khởi tạo KMeans trong cùng không gian đã cải thiện rõ agreement trên dev và seed giữ riêng. Parser mới giải quyết lỗi cắt công thức thực sự và cải thiện coherence của danh sách trên cùng reference.

Các kết quả âm cũng làm rõ giới hạn: source whitening làm mất cấu trúc từ vựng hữu ích; keyword specificity quá mạnh làm mất coverage; cross-encoder tổng quát không cải thiện ổn định proxy của corpus này. Không có căn cứ để mặc định mô hình phức tạp hơn sẽ tốt hơn.

| Còn thiếu | Bước tiếp theo hợp lý |
|---|---|
| Relevance người dùng độc lập | Chấm pool theo intent với con người; xác minh và giữ query-family ngoài phát triển |
| Expert topic labels | Đọc đại diện và thành viên biên; kiểm tra vai trò điện cực, pha, hệ pin và phạm vi |
| Quan hệ nhiều điều kiện | Đánh giá evidence theo từng điều kiện trên bộ phát triển mới, có nhãn |
| Chi tiết dài bị pooling | Thử multi-vector/chunk retrieval sau khi có protocol và ngân sách mới |
| Toàn văn và khái quát | Ingest dữ liệu mới version riêng; không coi Title–Abstract là full article |

Kết luận được hỗ trợ là pipeline đã nhất quán hơn, có clustering ổn định hơn và xử lý ký hiệu có truy vết. Chất lượng khoa học và Search tổng quát còn cần nhãn độc lập. Silver pilot chỉ là bước chuẩn bị đánh giá, không phải xác nhận cuối.

## 18 Tài liệu phương pháp và truy vết output

[1] Blei, Ng và Jordan. Latent Dirichlet Allocation. JMLR3,2003. https://www.jmlr.org/papers/v3/blei03a.html

[2] Grootendorst. BERTopic Neural topic modeling with a class based TF IDF procedure.2022. https://arxiv.org/abs/2203.05794

[3] BAAI. BGE base en v1.5 model card. https://huggingface.co/BAAI/bge-base-en-v1.5

[4] BatteryData. BatteryBERT cased checkpoint. https://huggingface.co/batterydata/batterybert-cased

[5] Gao và cộng sự. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup.2021. https://arxiv.org/abs/2101.06983

[6] Sentence Transformers. Cross encoder usage. https://www.sbert.net/docs/cross_encoder/usage/usage.html

[7] Kassis, Agarwal, He, Patel và Brueckner. Scientific Agent Skills A Library of Procedural Knowledge for Research Agents.2026. https://doi.org/10.48550/arXiv.2609.00065

Arbor và kỹ năng phản biện khoa học trong [7] hỗ trợ ghi giả thuyết, giữ evaluator, kiểm tra gate và giới hạn kết luận. Chúng không cung cấp bằng chứng rằng model chính xác hơn. Phiên bản nguồn đã đối chiếu là arXiv v2; dẫn DOI không khóa hậu tố version.

| Nội dung | Đường dẫn trong outputs/research_revision_20261025_v1 |
|---|---|
| Baseline audit | baseline_audit.json; protected_artifacts_before.json |
| Clustering | geometry/evidence/geometry; decisions/geometry.json |
| Keywords thất bại | keywords/experiment_outputs; cycle2/comparison.json |
| Cú pháp và catalog cuối | scientific_stream_v2; final_topics; review |
| Search dev/gate | search/experiment_outputs; decisions/search_gate.json |
| So sánh và hình | comparison; visualizations |
| Resume | execution/pipeline_state.json; checkpoints; logs; events.jsonl |

Các bảng/hình dựa trên số đo thực có manifest. Bản báo cáo31 trang ngày 25/10 và nguồn LaTeX19/10 được giữ nguyên, bản này là phiên bản tiếp tục với kết quả mới.

## Phụ lục A Toàn bộ Top 10 keywords của catalog mới

| ID | Tên diễn giải | Top10 keywords |
|---|---|---|
| 0 | Anode oxide titan và titanate | TiO2, titanium, titanate, dioxide, solution, anode, sodium, oxide, vanadium, Na2Ti3O7 |
| 1 | Prussian blue và vật liệu điện cực chứa sắt | iron, prussian_blue, sodium, ferrous, solution, anode, cathode, prussian, fe, capacity |
| 2 | Cathode oxide lớp và biến đổi cấu trúc pha | layer, oxide, cathode, layered_oxide, phase, transition, metal, manganese, P2-type, anode |
| 3 | Điện ly polymer và giao diện trong các hệ pin natri | electrolyte, polymer, additive, organic, salt, solvent, sodium, solid_electrolyte, carbonate, electrode |
| 4 | Cathode phosphate vanadium và NASICON | phosphate, vanadium, nvp, Na3V2(PO4)3, phosphorus, cathode, sodium, carbon, composite, fluorophosphate |
| 5 | Anode chứa Sn và Sb cùng composite chuyển hóa | tin, antimony, sn, sb, composite, alloy, anode, carbon, graphene, SnO2 |
| 6 | Cấu trúc cell bộ góp dòng và thiết bị pin | connect, plate, device, end, draw, electrode, pole, collector, electric, piece |
| 7 | Anode MoS2 và vật liệu hai chiều graphene MXene | MoS2, graphene, mxene, molybdenum, disulfide, nanosheets, anode, composite, electrode, mxenes |
| 8 | Hard carbon và tiền chất sinh khối | hard_carbon, carbon, biomass, porous, carbonize, precursor, carbonization, anode, dry, asphalt |
| 9 | Sulfide selenide kim loại chuyển tiếp và composite carbon | sulfide, cobalt, carbon, composite, selenide, nickel, solution, hollow, metal, anode |

Danh sách giữ đúng rank và trọng số của recipe đã xuất. Các từ rộng và từ thao tác còn tồn tại được đánh dấu trong keyword_audit.csv, không bị xóa khỏi bảng sau khi xem điểm. Tên diễn giải không phải expert ground truth.

## Phụ lục B File giao nộp và cách tiếp tục

Bản PDF và Markdown của báo cáo mới nằm tại output/pdf và docs với tên BAO_CAO_NGHIEN_CUU_CAI_THIEN_SODIUM_ION_20261026. Để chỉnh sửa số liệu, sửa nguồn hoặc recipe trong một phiên bản mới và tạo output riêng; không ghi đè model, pool hoặc metric đã hoàn thành.

| Mục đích | Lệnh hoặc file |
|---|---|
| Resume toàn bộ |  .venv/Scripts/python.exe scripts/resume_research_revision.py |
| Đọc trạng thái |  .venv/Scripts/python.exe scripts/resume_research_revision.py --status |
| Chạy ứng dụng |  .venv/Scripts/python.exe run_demo.py |
| Pilot AI20 | outputs/ai_silver_labels_20261026_v1/batch_0001_labels_20.csv |
| Checkpoint từng5nhãn | outputs/ai_silver_labels_20261026_v1/checkpoints |
| Nhãn người | Giữ nguyên annotation_pool và accepted_human_labels; không nhập AI vào đó |

Sau khi có nhãn mới do con người xác minh, tính relevance theo protocol đã khóa. Nếu tiếp tục Silver Labels, tạo batch tiếp theo đúng cặp đầu vào và lưu output độc lập; không suy ra nhãn của683 cặp còn lại từ kết quả20 cặp đầu.
