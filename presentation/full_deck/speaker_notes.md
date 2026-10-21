# Ghi chú thuyết trình

Nguồn số liệu là các artifacts đã lưu. Không có huấn luyện hoặc benchmark mới khi biên soạn.

Phần chính có thể chia thành hai buổi: dữ liệu/mô hình và search/đánh giá. Phụ lục dành cho tra cứu topics và câu hỏi bảo vệ.


## Slide 2: Bài toán và phạm vi

Mở đầu bằng nhu cầu của người nghiên cứu: tìm đúng hệ pin, vật liệu và điều kiện trong nhiều tài liệu. Dự án có hai nhiệm vụ: khám phá corpus và truy hồi theo query. Hệ thống không lập chỉ mục toàn văn, bảng, hình hay toàn bộ claims của patent. Dữ liệu nguồn đến năm 2023; không tuyên bố bao quát tài liệu đến năm 2026.

Nguồn: docs/FINAL_RESEARCH_REPORT.md, mục 1


## Slide 3: Một truy vấn có nhiều điều kiện

Đây là ví dụ có trong báo cáo. Tài liệu chỉ nói ether và nhiệt độ thấp nhưng thiếu NaFSI chưa đủ đáp ứng. Tài liệu có NaFSI trong phần so sánh cũng chưa chắc dùng nó trong hệ thí nghiệm được hỏi. Khó khăn cốt lõi là quan hệ giữa các điều kiện, không chỉ có từng từ khóa.

Nguồn: docs/FINAL_RESEARCH_REPORT.md, mục 1.1 và 8.5


## Slide 4: Pipeline toàn bộ dự án

Đây là luồng tổng quan, không phải một chuỗi bắt buộc đưa mọi model vào search. Sau corpus, dự án phân nhánh. LDA và BERTopic phục vụ khám phá. Search V2 mặc định dùng BGE pretrained và reranker công khai. Huấn luyện và tạo embedding tài liệu là offline; khi search chỉ xử lý query, truy hồi và rerank.

Nguồn: src/pipeline.py; src/stage2.py; src/stage3_neural.py; src/search_v2.py


## Slide 5: Offline và online trong cùng một hệ thống

Tách build-time và query-time. BM25 index được kiểm tra identity và có thể tái dùng từ cache. Models và dense matrix được nạp khi cần. Không có train hoặc gọi Gemini trong search mặc định.

Nguồn: src/search_v2.py; src/search_v2_service.py; configs/search_v2.json


## Slide 6: Dữ liệu trước và sau tiền xử lý

Không diễn giải 6.874 bản ghi ngoài corpus như toàn bộ là trùng hay ngoài miền. Các disposition còn gồm chờ review relevance, ngôn ngữ, chất lượng và retracted. CORE/RELATED là nhãn sàng lọc theo quy tắc, chưa phải nhãn relevance của một query người dùng.

Nguồn: outputs/reports/stage1_summary.json; outputs/reports/stage3/split_audit.json


## Slide 7: Quyết định sử dụng toàn bộ 20.576 bản ghi

Các hàng loại trừ nhau và cộng đúng 20.576. RELEVANCE_REVIEW là 6.596, khác tổng nhãn UNCERTAIN 6.598 vì một bản ghi còn có các điều kiện chất lượng khác trước quyết định cuối.

Nguồn: outputs/reports/cleaning_summary.csv


## Slide 8: Thay đổi dữ liệu qua từng quyết định xử lý

Bảng quyết định theo nguồn. Các nhóm review vẫn tồn tại trong clean master.

Nguồn: outputs/figures/data_preparation/figure_05_cleaning_summary.png


## Slide 9: Tiền xử lý: mỗi bước phục vụ mục đích gì?

normalize_text không lowercase toàn bộ văn bản gốc: code bảo toàn case để giữ công thức. Ký tự chỉ số dưới ₀…₉ chuyển thành 0…9. Phần tokens mới xử lý từ thường theo nhánh. Ngôn ngữ dùng langdetect trên tối đa 1.400 ký tự; điều kiện chính là en và confidence ít nhất 0,90. Confidence này không được coi là độ chính xác phân loại đã hiệu chỉnh.

Nguồn: src/data_io.py; src/text_processing.py; src/quality.py; configs/config.yaml


## Slide 10: Phát hiện gần trùng và chống rò rỉ split

Ngoài near-duplicate còn có các liên kết exact ID/DOI/text và các số patent/priority chung. Đây là candidate relationships, không xác minh mọi patent family. Tài liệu cùng nhóm đi cùng split.

Nguồn: src/quality.py::duplicate_audit; configs/config.yaml


## Slide 11: Ba biểu diễn văn bản khác nhau

Không lấy danh sách tokens LDA để thay thẳng đầu vào transformer. BGE cần văn bản còn ngữ cảnh; lexical tokenizer phục vụ đối sánh từ nên có stemming và kênh chem:. Bảo vệ công thức là heuristic nhận dạng chuỗi, không phải chemical parser. Audit còn ghi nhận placeholder trong artifacts topic lịch sử, nên không khẳng định mọi công thức đều được bảo toàn hoàn hảo.

Nguồn: src/text_processing.py::lda_tokens; src/search_v2.py::lexical_tokens


## Slide 12: Sau tiền xử lý thu được gì?

Sau tiền xử lý chưa có mô hình đã học hay điểm relevance. Đây là dữ liệu sẵn sàng cho từng nhiệm vụ. Cùng 13.702 hàng ở các biểu diễn chính phải có cùng thứ tự ID. Nếu ma trận vector bị đảo hàng, phép dot product vẫn tính được nhưng ánh xạ điểm sang sai tài liệu. metadata_extended chứa thông tin cho dữ liệu tích hợp, không nên giả định chỉ có corpus chính.

Nguồn: src/pipeline.py::run, validate


## Slide 13: Các từ xuất hiện nhiều theo từng nguồn

Thống kê tokens Stage 1. Tần suất cao thể hiện phổ biến, chưa thể hiện độ phân biệt của một topic.

Nguồn: outputs/figures/stage2/eda_02_top_terms.png


## Slide 14: Top5 tokens: số lần và số tài liệu chứa từ

TF là tổng số lần xuất hiện. DF là số tài liệu chứa token. Một từ xuất hiện nhiều lần trong một tài liệu tăng TF nhưng chỉ tăng DF một lần. Đây là token của nhánh topic, khác lexical tokens Search V2.

Nguồn: outputs/tables/stage2/eda_term_frequencies.csv


## Slide 15: LDA: định nghĩa và lý do dùng

LDA không phải mô hình phân loại có nhãn chuyên gia. Tài liệu là hỗn hợp topic, topic là phân phối từ. Chọn LDA vì tính giải thích và vai trò baseline bag-of-words. Từ khóa xuất hiện trong topic không xác nhận một claim hóa học. Các phân phối được suy ra bằng Gensim LdaModel, không tự viết bộ tối ưu LDA.

Nguồn: src/stage2.py::_fit_lda; configs/stage2.yaml


## Slide 16: LDA: quy trình và sản phẩm sau huấn luyện

Input của LDA là BOW: mỗi tài liệu là các cặp ID từ và số lần xuất hiện. Sau fit có ma trận topic–word, và sau inference có document–topic. Chỉ hai K tốt nhất theo C_v cơ sở được chạy thêm seed 123 và 2026. Đây là screened search nên chưa so nhiều seed đồng đều cho mọi K. Không gọi K=20 là số chủ đề tối ưu toàn cục.

Nguồn: src/stage2.py::export_topics; outputs/reports/stage2/selected_model.json


## Slide 17: LDA: các phân phối thay đổi trong quá trình fit

Đây là mô tả toán học dạng rút gọn, không phải tự viết optimizer trong dự án. Trong BOW, thay tổng từng vị trí từ bằng tổng có trọng số count. K=20 vẫn cố định trong một lần fit. Thuật toán điều chỉnh phân phối, không tạo thêm topic. Gensim online variational Bayes kết hợp sufficient statistics theo learning-rate rho.

Nguồn: src/stage2.py::_fit_lda; gensim.models.ldamodel.LdaModel


## Slide 18: So sánh các cấu hình LDA đã chạy

Chọn K=20 theo protocol hiện tại. Chỉ hai K qua sàng lọc được thử thêm seed.

Nguồn: outputs/figures/stage2/lda_01_model_comparison.png


## Slide 19: Output LDA: ba topic trội lớn nhất

Ba topic có nhiều tài liệu nhất. Tên chủ đề không được tự gán thành nhãn chuyên gia. Số tài liệu LDA là dominant assignments, BERTopic là assigned clusters. Phụ lục có toàn bộ IDs và Top5, topic_catalog.csv giữ đủ Top10.

Nguồn: outputs/tables/stage2/topic_summary.csv


## Slide 20: Phân bố tài liệu trên 20 topic LDA

Dominant assignment đếm mỗi tài liệu vào một topic. Soft mass cộng xác suất topic qua tài liệu.

Nguồn: outputs/figures/stage2/lda_02_topic_distribution.png


## Slide 21: BGE: bi-encoder tạo vector ngữ nghĩa

Bi-encoder dùng cùng encoder để biểu diễn hai phía độc lập. Instruction thật là Represent this sentence for searching relevant passages: . Norm là chia vector khác 0 cho độ dài L2. Vì cả hai norm bằng 1, dot product bằng cosine. Đặc điểm này làm dense search thành nhân ma trận; không chứng minh embedding hiểu đúng mọi công thức hay phủ định.

Nguồn: src/stage3_neural.py; models/bge_pretrained/1_Pooling/config.json


## Slide 22: BGE: token tương tác qua self-attention

Công thức giải thích cơ chế transformer đã dùng trong checkpoint, không phải module attention tự viết. BGE base có 12 encoder layers. Dự án chỉ mở layer.11 khi fine-tuning. Q và K trong công thức attention là token projections, khác query/document trong retrieval.

Nguồn: models/bge_pretrained/config.json; src/stage3_neural.py


## Slide 23: Fine-tuning: học từ Title và Abstract

Title–Abstract là tín hiệu tự giám sát gần với retrieval, nhưng Title thường sát nội dung của chính tài liệu hơn query thực tế. Group split giảm rò rỉ từ các bản ghi liên kết đã phát hiện, không bảo đảm phát hiện mọi tài liệu gần nghĩa hoặc patent family. Có 1.359 nhóm trong 1.368 cặp test. Candidate pool của test chỉ gồm Abstract trong split test.

Nguồn: outputs/reports/stage3/split_audit.json; configs/stage3.yaml


## Slide 24: Fine-tuning: contrastive loss đối xứng

Công thức là trung bình cross-entropy hàng và cột, đúng phép tính trong train. Trong công thức hiểu s là logits sau mask cho các cặp được loại. Code dùng -1e4 chứ không phải xóa hàng/cột; số mũ của logit này gần 0. Mask giúp giảm false negatives, nhưng vẫn có thể bỏ sót Abstract liên quan khác. Evaluator chấp nhận nhiều positives cùng nhóm trong khi objective chỉ dùng positive đường chéo.

Nguồn: src/stage3_neural.py::false_negative_mask, train


## Slide 25: False negatives: cơ chế mask và kết quả thực tế

Phân biệt cơ chế triển khai và tác động đã quan sát. Mặc dù có logic mask, các batch của run này không gặp cặp thỏa điều kiện theo counter đã lưu. Chưa có ablation mask on/off để suy ra lợi ích.

Nguồn: models/bert_finetuned/training_success.json; src/stage3_neural.py


## Slide 26: Fine-tuning: cập nhật gì, lưu lại gì?

Mở một block nhằm hạn chế chi phí CPU và số tham số thay đổi. Training gồm forward, tính similarity matrix, mask, loss, backward, clip gradient, optimizer và scheduler step. Test không dùng để chọn checkpoint. Cấu hình chỉ chọn trong fine-tuned checkpoints, chưa có quy tắc tự động quay về pretrained nếu mọi checkpoint đều kém. Frozen validation cache hiện chỉ hỗ trợ đúng một trainable block.

Nguồn: configs/stage3.yaml; src/stage3_neural.py::train, embed


## Slide 27: AdamW cập nhật trọng số block cuối

Tính gradient đối với các tham số requires_grad=True, rồi clip global norm tối đa 1. AdamW dùng moment bậc một và hai, weight decay tách riêng. Các beta và epsilon theo mặc định dependency đã cài, không tự nhận dự án đã tune chúng. Scheduler.step chạy sau optimizer.step.

Nguồn: src/stage3_neural.py::train; torch.optim.AdamW


## Slide 28: Bằng chứng cập nhật trọng số của phiên huấn luyện

Tổng 109.482.240, trainable 7.087.872, khoảng 6,47%. Layer cuối có các ma trận attention, FFN và layernorm được cập nhật. L2 delta khác 0 và frozen hash bằng nhau xác nhận có thay trọng số đúng phạm vi, chưa chứng minh lợi ích search.

Nguồn: models/bert_finetuned/training_success.json


## Slide 29: Loss huấn luyện và chọn checkpoint bằng validation

Loss giảm trong run đã lưu. Chọn step 1.365 bằng group MRR@10 trên validation, không dùng test.

Nguồn: outputs/figures/stage3/bert_01_training.png


## Slide 30: Kết quả fine-tuning trên proxy test

Các trường lịch sử có chữ recall thực chất đo success: có ít nhất một positive. Slide đổi nhãn sang Success nhưng không sửa artifact gốc. Group MRR tăng khoảng 0,0345; báo cáo ghi CI 95% của delta [0,024765; 0,044217], theo paired cluster bootstrap. Không diễn giải thành độ chính xác 92,8% cho mọi query. Checkpoint và query pool được giữ cố định trong phép suy luận này.

Nguồn: outputs/reports/stage3/{pretrained,finetuned}_test_metrics.json


## Slide 31: BERTopic: định nghĩa và chuỗi thành phần

Trong dự án, BERTopic nhận embeddings đã tính sẵn bằng BGE fine-tuned, không train transformer thêm ở đây. Từ khóa lấy từ tokens Stage 1 đóng băng, dùng vocabulary LDA. Cluster được nhận diện trong không gian UMAP 5 chiều; hình UMAP 2 chiều phục vụ hiển thị là phép chiếu riêng, không phải không gian dùng để fit cụm.

Nguồn: src/stage3_topics.py::fit; src/stage3_bertopic_model.py


## Slide 32: BERTopic: tham số, đầu ra và ý nghĩa

Lý do chọn: gom các diễn đạt gần nhau dựa trên embeddings, không ép mọi tài liệu vào một cụm. Các cụm phụ thuộc biểu diễn và mật độ nên phải kiểm tra độ ổn định. Seed 123 cho 82 topics, khác seed 42. Tài liệu đại diện được chọn theo membership strength cao nhất trong cụm, không phải xác nhận bởi chuyên gia. nr_topics=None, không ép số cụm.

Nguồn: configs/stage3.yaml; outputs/reports/stage3/bertopic_metrics.json


## Slide 33: Output BERTopic baseline: ba cụm lớn nhất

Ba topic có nhiều tài liệu nhất. Tên chủ đề không được tự gán thành nhãn chuyên gia. Số tài liệu LDA là dominant assignments, BERTopic là assigned clusters. Phụ lục có toàn bộ IDs và Top5, topic_catalog.csv giữ đủ Top10.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 34: Phân bố cụm BERTopic baseline và noise

92 topics và 2.751 outliers. Topic -1 gom noise, không đại diện một chủ đề chuyên môn thống nhất.

Nguồn: outputs/figures/stage3/bertopic_01_distribution.png


## Slide 35: Biểu diễn 2D của BERTopic baseline

Hình để quan sát cấu trúc. Clustering thực tế dùng UMAP 5D, không dùng trực tiếp tọa độ trên hình.

Nguồn: outputs/figures/stage3/bertopic_04_umap_2d.png


## Slide 36: Chunking: bao phủ phần cuối Abstract

Chunking dùng encoder fine-tuned đã có, không fine-tune lại. Capacity Abstract thực tế bằng 256 trừ special tokens và Title được giữ. Code giữ ít nhất 65 tokens Abstract để capacity lớn hơn overlap. Mean ở đây là mean của CLS chunk embeddings, không phải mean token pooling của BGE. Tài liệu thiếu Abstract sử dụng Title thật; không tạo dữ liệu bù.

Nguồn: src/bertopic_chunked.py::chunk_document, encode_features, embed


## Slide 37: Output BERTopic chunked: ba cụm lớn nhất

Ba topic có nhiều tài liệu nhất. Tên chủ đề không được tự gán thành nhãn chuyên gia. Số tài liệu LDA là dominant assignments, BERTopic là assigned clusters. Phụ lục có toàn bộ IDs và Top5, topic_catalog.csv giữ đủ Top10.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 38: Phân bố 84 topics của nhánh chunked

Nhánh chunked có 3.120 outliers. Topic ID được tạo lại, không đồng nhất trực tiếp với baseline.

Nguồn: outputs/bertopic_chunked/figures/topic_distribution.png


## Slide 39: Kết quả BERTopic: có sự đánh đổi

Không kết luận chunked tốt hơn toàn diện hay ít topics hơn là tốt hơn. Metric loại topic -1 nên tăng coherence của inliers phải đọc cùng tỷ lệ outliers. Mean pooling có thể làm mờ một điều kiện hiếm, lặp Title và vùng overlap, đồng thời cho chunk đuôi ngắn trọng số ngang chunk dài. Cần ablation và review chuyên gia.

Nguồn: outputs/bertopic_chunked/comparison.csv


## Slide 40: Từ khóa và trọng số của các topic BERTopic

Các từ khóa theo c-TF-IDF của baseline. Phụ lục có đủ 92 IDs và bảng 84 IDs chunked.

Nguồn: outputs/figures/stage3/bertopic_02_keywords.png


## Slide 41: Search liên hệ với các mô hình trước ra sao?

Đây là điểm nối quan trọng để bảo vệ bài. Search mặc định RERANK_PRETRAINED không dùng checkpoint fine-tuning của dự án. DENSE_FINETUNED và DENSE_CHUNKED là nhánh so sánh có sử dụng nó. LDA và BERTopic được nạp cho khám phá chủ đề riêng. BM25 index và ánh xạ vector–metadata là lớp dữ liệu truy hồi mới; không đồng nghĩa train encoder từ đầu.

Nguồn: src/search_v2.py::_matrix, encode; configs/search_v2.json


## Slide 42: Mỗi mô hình tạo bao nhiêu topic?

768 là số chiều embedding, không phải 768 topic. Fine-tuning thay encoder chứ không tạo taxonomy. LDA gán xác suất cho 20 topic đặt trước. BERTopic suy ra số cụm từ dữ liệu và các tham số mật độ.

Nguồn: outputs/reports/stage3/bertopic_metrics.json; outputs/bertopic_chunked/metrics.json


## Slide 43: Xây chỉ mục Search V2 trước khi truy vấn

Index BM25 là cấu trúc thống kê từ corpus; không cần nhãn query–document. Dense matrix là vector đã encode offline; không tính lại mọi tài liệu cho mỗi query. Code có cache/fingerprint để tránh dùng index sai tokenizer hay cấu hình. Nạp file và kiểm tra identity khác với huấn luyện. Bộ dữ liệu này chưa có bằng chứng exact scan là nút thắt cần vector service.

Nguồn: src/search_v2.py::_load_lexical, _matrix; configs/search_v2.json


## Slide 44: Mười phương pháp Search V2 có thể chọn

Tên đầy đủ gồm DENSE_PRETRAINED, DENSE_FINETUNED, DENSE_CHUNKED, HYBRID_PRETRAINED, HYBRID_FINETUNED, HYBRID_CHUNKED, RERANK_PRETRAINED, RERANK_FINETUNED, RERANK_CHUNKED và BM25_CHEMICAL. CHUNKED dùng query encoder fine-tuned với query max128, khác document chunk pair256.

Nguồn: src/search_v2.py::METHODS; configs/search_v2.json


## Slide 45: BM25: định nghĩa và công thức thực tế

Mô hình sparse/BM25 dựa trên tần suất và độ hiếm từ. Đây là tổng hai field chấm riêng, clipping từng field rồi cộng, không phải suy dẫn BM25F đầy đủ. Query token lặp được dependency cộng lại theo vòng lặp. IDF là biến thể rank_bm25 đang cài; phụ lục ghi chính xác. Trọng số Title=2 là lựa chọn thiết kế, chưa tối ưu bằng relevance độc lập.

Nguồn: src/search_v2.py::lexical; rank_bm25.py::BM25Okapi


## Slide 46: Query được chuẩn hóa và lọc thế nào?

Ví dụ nhập NaFSI;ether yêu cầu token của cả hai nhóm xuất hiện trong Title hoặc Abstract đầy đủ. Bộ lọc không kiểm tra chúng có cùng vai trò hay bị phủ định. Tài liệu thiếu năm bị loại khi có khoảng năm. Query và corpus dùng lexical tokenizer tương ứng; semantic encoder nhận văn bản query kèm instruction, không nhận danh sách stem tokens. Truy vấn tiếng Việt chưa có benchmark tương xứng.

Nguồn: src/search_v2.py::normalize, lexical_tokens, allowed, encode; src/search_v2_constraints.py


## Slide 47: Dense retrieval và RRF

Kết quả sau RRF là danh sách hợp nhất, chưa phải score cross-encoder. Không dùng công thức alpha BM25 + beta cosine + gamma reranker trong dự án. Một tài liệu có trong cả hai nhánh cộng hai phần, một tài liệu có một nhánh chỉ cộng một phần. Gate lexical có thể từ chối một paraphrase hợp lệ không trùng token nên chưa coi là phát hiện query ngoài miền được hiệu chỉnh.

Nguồn: src/search_v2.py::dense, reciprocal_rank_fusion, rank


## Slide 48: MiniLM reranker: cross-encoder

Bi-encoder cho phép encode tài liệu độc lập nên nhanh ở bước tìm ứng viên. Cross-encoder đọc hai phía cùng lúc, tương tác token sâu hơn nhưng phải chạy theo từng cặp. Reranker nhận tối đa 100 cặp khi 50 tài liệu mỗi tài liệu có 2 passages. Trong code predict cho scalar score; đây không phải xác suất relevance. Không có bằng chứng thêm reranker luôn tăng mọi metric.

Nguồn: src/search_v2.py::_load_reranker, rerank


## Slide 49: Chọn passage: ngân sách và giới hạn

320 là độ rộng yêu cầu, thực tế phải trừ ngân sách query, Title và special tokens theo tokenizer. Code kiểm tra bước tiến để tránh overlap làm vòng lặp không tiến. Không nhầm cửa sổ reranker 320 với chunk embedding 256: hai tác vụ khác nhau. Selector quét Abstract nhưng cross-encoder chỉ đọc các passages đã chọn. Max score có thể ưu tiên một đoạn khớp tốt một phần query.

Nguồn: src/search_v2.py::select_passages; configs/search_v2.json


## Slide 50: Luồng online và kết quả trả về

Trong bài, câu trả lời tìm kiếm thực tế là danh sách tài liệu và evidence passage, không phải văn bản tổng hợp bằng LLM. AI/Gemini xuất hiện trong nhánh đánh giá/sinh query, không ở đường search mặc định. Không gán BLEU/ROUGE hay factuality của câu trả lời sinh cho pipeline hiện hành. Service một worker tuần tự hóa để kiểm soát model state và RAM.

Nguồn: src/search_v2.py::search; src/search_v2_service.py; app.py


## Slide 51: Query Q19: Top5 thực tế của RERANK_PRETRAINED

Hạng 1: publication:WOS:000449952200002 — Fluorine-Free Water-in-Salt Electrolyte for Green and Low-Cost Aqueous Sodium-Ion Batteries — score 9.818456649780273
Hạng 2: publication:WOS:000642311300001 — High-Energy Aqueous Sodium-Ion Batteries — score 9.336146354675293
Hạng 3: publication:WOS:000581738300066 — Halide-free water-in-salt electrolytes for stable aqueous sodium-ion batteries — score 9.304271697998047
Hạng 4: publication:WOS:000878318300001 — Synthesis and Application of an Aromatic Sulfonate Sodium Salt for Aqueous Sodium-Ion Battery Electrolytes — score 9.203011512756348
Hạng 5: publication:WOS:000410600500010 — A High-Voltage Aqueous Electrolyte for Sodium-Ion Batteries — score 9.13907241821289

Nguồn: outputs/search_upgrade_20261018/query_runs/Q19.json


## Slide 52: Q19: passage để người dùng kiểm tra bằng chứng

Toàn bộ passage gốc:
Fluorine-Free Water-in-Salt Electrolyte for Green and Low-Cost Aqueous Sodium-Ion Batteries
The highly concentrated "water-in-salt" electrolyte (WiSE) containing sodium acetate and potassium acetate demonstrates surprising performance (specific capacity of 37 mA h(-1) g(-1) at the 5th cycle and average discharge voltage of 0.82 V) in aqueous sodium-ion batteries (SIBs) based on Na2MnFe(CN)(6) and NaTi2(PO4)(3). The fluorine-free electrolyte offers a wide electrochemical stability window and compatibility with Al current collector. The electrolyte, current collector, and the electrode materials based on abundant elements make the proposed battery chemistry safe, low-cost, and environmentally friendly.

Tài liệu publication:WOS:000449952200002. Hình thức trích dẫn ở slide không tạo ra nhận định khoa học mới. Search không có LLM tổng hợp câu trả lời.

Nguồn: outputs/search_upgrade_20261018/query_runs/Q19.json


## Slide 53: Đánh giá phải đúng với từng đầu ra

Mỗi loại bằng chứng trả lời câu hỏi khác nhau. Topic coherence cao chỉ nói về thống kê từ khóa, không đảm bảo thứ hạng theo query. Test kỹ thuật cho biết điều kiện code đã mã hóa được đáp ứng. Source recovery xem tài liệu nguồn có được tìm lại không, trong khi relevance cho phép nhiều tài liệu khác cùng đúng. Phải giữ các bảng này tách biệt.

Nguồn: docs/FINAL_RESEARCH_REPORT.md, mục 8; src/research_evaluation.py


## Slide 54: Công thức đánh giá Top5

Precision đo tỷ lệ tài liệu vượt ngưỡng relevance 2 trong 5 vị trí. MRR nhấn mạnh tài liệu đúng đầu tiên. nDCG dùng grade phân bậc: grade 1 vẫn có gain dù chưa qua ngưỡng P/MRR. Kết quả dưới 5 phần tử được tính DCG trên phần có kết quả, tương đương gain 0 ở vị trí trống; điều này khác tài liệu chưa được chấm nhãn.

Nguồn: src/research_evaluation.py::metrics_at_5, evaluate_frames


## Slide 55: Rubric relevance và bằng chứng cần chấm

Rubric hiện được mã hóa trong nhánh AI. Kế hoạch nhãn độc lập cần chốt intent/rubric trước khi chấm, ẩn method/rank, hai người chấm và phân xử. UNRESOLVED không được tự đổi thành 0. Grade 2/3 tính relevant cho P/MRR. Evidence substring chỉ xác nhận đoạn có tồn tại, không xác nhận diễn giải hoặc tính đúng khoa học của đoạn đó.

Nguồn: src/ai_judge.py::RUBRIC, validate_judgment; src/research_evaluation.py


## Slide 56: Nhãn relevance độc lập: hiện chưa hoàn tất

Đây là hạn chế trọng yếu và cần nói rõ khi bảo vệ. Code evaluator đã có không đồng nghĩa đã có điểm đúng. Pooling giúp giảm công chấm nhưng không tìm mọi relevant trong toàn corpus. Nếu thêm method mới phải bổ sung labels cho các tài liệu mới. Không lấy số hàng Top5 hoặc 240 lượt chạy làm cỡ mẫu relevance độc lập.

Nguồn: docs/FINAL_RESEARCH_REPORT.md, mục 8.2; outputs/search_upgrade_20261018/


## Slide 57: Mức hoàn thành của bộ nhãn AI

AI consensus số: 116/942 cặp. Chỉ 2/23 query đủ pool. Hai biểu mẫu nhãn độc lập vẫn trống.

Nguồn: outputs/retrieval_ai_evaluation_v1/judgment_coverage.png


## Slide 58: AI relevance: kết quả từng phần

Có 118 cặp A/B hợp lệ, 116 có consensus và 2 chờ phân xử. Exact agreement và kappa cao chỉ phản ánh nhất quán của cùng model trên phần phản hồi đã có; hai lượt có thể cùng sai. Không ngoại suy trung bình 2 queries sang toàn 23 queries. Quá trình làm slide chỉ đọc artifacts, không tiếp tục gọi API hoặc tạo nhãn mới.

Nguồn: outputs/retrieval_ai_evaluation_v1/REPORT.md; docs/FINAL_RESEARCH_REPORT.md, mục 8.3


## Slide 59: Source recovery: một proxy khác

Các điểm đọc trực tiếp từ CSV đã lưu. DENSE_CHUNKED có point estimate cao nhất trong phép thử, nhưng một tài liệu khác cũng đúng ý vẫn bị exact-ID metric tính là miss. 16 source thuộc validation và 18 test. Reranker không cải thiện đồng đều source recovery; so với hybrid pretrained, Hit@1 giảm nhưng Hit@5/10 tăng. Không có một winner chung cho mọi mục tiêu.

Nguồn: outputs/retrieval_ai_evaluation_v1/synthetic/summary.csv; docs/FINAL_RESEARCH_REPORT.md, mục 8.4


## Slide 60: Phân tích lỗi từ truy vấn thực tế

Các ca này là đối chiếu định tính đã được xem trong development, không tự chuyển thành P@5 chuyên gia. Nếu lấy chúng làm cảm hứng sửa model thì cần queries mới để test độc lập. Có token là bằng chứng từ vựng; xác nhận quan hệ cần đọc ngữ cảnh và nhãn. Abstract không nói một điều kiện chưa đủ kết luận toàn văn không có điều kiện đó.

Nguồn: docs/SEARCH_RESULTS_REVIEW.md; docs/FINAL_RESEARCH_REPORT.md, mục 8.5


## Slide 61: Tối ưu kỹ thuật đã có bằng chứng

Không cộng hay nhân hai tỷ lệ kernel để suy ra tốc độ cả search. Không tuyên bố giảm 94,7% thời gian end-to-end. Lưới threads 1,2,4 và batch 4,8,16 gồm 54 requests đo cùng 9 warmups. Baseline 4:4 được giữ vì không có cấu hình khác đạt gate. Các phép đo nhỏ trên một máy không phải p95/SLA hoặc throughput production. Tests là evidence lịch sử, chưa chạy lại trong phiên soạn slide.

Nguồn: docs/MATH_AND_OPTIMIZATION_REVIEW.md; outputs/search_runtime_optimization_20261019/verification.json


## Slide 62: Độ trễ đo được và giới hạn vận hành

Không so cold startup với warm query latency. Kết quả kernel Top-K/filter không cộng hoặc nhân thành mức tăng tốc end-to-end. Compare request có reuse nên thời gian của method phía sau không tương đương benchmark chạy độc lập. Threads/batch tốt nhất trong lưới nhỏ trên máy đã đo.

Nguồn: outputs/math_optimization_20261019/execution_dev/summary.json; src/search_v2_service.py


## Slide 63: Cải thiện tiếp theo và cách kiểm chứng

Ưu tiên dữ liệu đánh giá trước khi đổi model lớn hơn. Mỗi thí nghiệm giữ baseline, thay một giả thuyết, chọn trên dev có nhãn rồi khóa cấu hình trước test mới. Tách kết quả theo Patent/Publication, công thức hiếm và query nhiều điều kiện. Thay preprocessing cần version dữ liệu mới, không ghi đè artifacts và gọi như cùng thí nghiệm.

Nguồn: docs/FINAL_RESEARCH_REPORT.md, mục 10--11


## Slide 64: Kết quả và giới hạn của bài

Kết bài bằng năng lực đã có cùng giới hạn ảnh hưởng kết luận. Không tuyên bố thuật toán mới, tối ưu toàn cục hay thay người nghiên cứu xác minh khoa học. Phụ lục tiếp theo có công thức và câu hỏi bảo vệ; có thể dừng phần thuyết trình chính ở đây.

Nguồn: README.md; docs/FINAL_RESEARCH_REPORT.md, mục 12


## Slide 65: Bản đồ mô hình, input và output

RRF là quy tắc fusion, không có bước training. Chunking là chiến lược tạo biểu diễn và pooling, không phải model mới. BERTopic là pipeline phối hợp các thành phần trên embeddings. BM25 fit index thống kê khác với fine-tune neural weights.

Nguồn: src/stage2.py; src/stage3_topics.py; src/search_v2.py


## Slide 66: LDA: quá trình sinh và suy luận

Đây là định nghĩa mô hình LDA tương ứng với API đang dùng; không nói code trực tiếp sinh corpus theo quá trình này. Dirichlet là prior cho vector xác suất, categorical là lấy một hạng mục theo vector xác suất. Entropy cao gợi ý phân phối topic trải rộng, không phải độ không chắc chắn relevance. Code dùng max(theta, 1e-30) trong log để tránh log 0.

Nguồn: src/stage2.py::_fit_lda, export_topics


## Slide 67: Chọn LDA K và đo topic stability

D là diversity. T trên công thức là stability của một cặp seed; code lấy trung bình qua các cặp seed có chạy. C_v và U_Mass tính qua Gensim CoherenceModel, topn 10, C_v window110. Không gọi C_v là cosine của hai embeddings tài liệu. C_v có pipeline thống kê đồng xuất hiện riêng; đây không phải tỷ lệ chủ đề đúng chuyên môn.

Nguồn: src/stage2.py::match_stability, main; configs/stage2.yaml


## Slide 68: c-TF-IDF đúng với cấu hình đang dùng

Công thức đối chiếu dependency cài trong venv của dự án. bm25_weighting mặc định False, không có seed multiplier trong khởi tạo này. A là int của độ dài trung bình các class documents. IDF nhân TF đã normalize rồi sqrt. Keywords mô tả cụm; không tạo thêm evidence cho query và không đưa c-TF-IDF vào search ranking.

Nguồn: src/stage3_topics.py::fit; bertopic/vectorizers/_ctfidf.py::fit, transform


## Slide 69: UMAP và HDBSCAN: khái niệm toán học

core_k mô tả khoảng cách lõi theo tham số mật độ min_samples, không gán tùy ý nó thành xác suất. Công thức mutual reachability là diễn giải cơ chế thư viện, không phải một hàm loss được code dự án tối ưu trực tiếp. Không so cosine 768 chiều với Euclidean 5 chiều như cùng thang điểm. Hình UMAP 2D chỉ để minh họa và có min_dist riêng.

Nguồn: src/stage3_topics.py::fit; configs/stage3.yaml


## Slide 70: BM25 IDF: biến thể thư viện đang cài

Giá trị epsilon mặc định là 0,25. Nếu average_idf âm trên corpus nhỏ, floor vẫn có thể âm; vì vậy không khẳng định rank_bm25 luôn cho điểm dương. Code dự án còn np.maximum từng field. Audit có phản ví dụ toy corpus nhưng chưa đo ảnh hưởng lên corpus thật. Muốn đổi IDF hay lexical gate cần thí nghiệm phiên bản riêng.

Nguồn: rank_bm25.py::BM25Okapi._calc_idf; src/search_v2.py::lexical


## Slide 71: RRF: ví dụ tính tay minh họa

Bảng chỉ dùng để giải thích số học, không giả làm kết quả thực nghiệm. Sau RRF, nếu chọn method rerank, thứ tự cuối được thay bằng cross-encoder scores. Không cộng RRF score vào scalar reranker. Các danh sách đầu vào phải có ID duy nhất trong mỗi danh sách; code đã guard ID trùng và k không hợp lệ.

Nguồn: src/search_v2.py::reciprocal_rank_fusion


## Slide 72: Đánh giá Top5: ví dụ tính tay minh họa

Grade 3 và 2 được tính relevant, grade 1 chỉ đóng góp gain trong DCG. RR=1 vì relevant đứng ngay đầu. IDCG phải từ pool chung chứ không riêng Top5 một method; ở ví dụ cố ý chọn pool có đúng năm grade để tính tay đơn giản. Nếu pool có tài liệu grade cao hơn ở ngoài Top5 thì IDCG sẽ thay đổi.

Nguồn: src/research_evaluation.py::metrics_at_5


## Slide 73: Success, Recall và source recovery khác gì?

Đây là khác biệt đã sửa trong tên báo cáo. Alias group_recall_at_k trong JSON lịch sử được giữ tương thích, nhưng slide gọi đúng Success. Success@k trên nhiều query là trung bình các indicator. Recall cần biết số relevant tài liệu, điều mà nhãn pool chưa bảo đảm cho toàn corpus.

Nguồn: docs/MATH_AND_OPTIMIZATION_REVIEW.md; src/research_evaluation.py


## Slide 74: Bootstrap và độ tin cậy của kết quả

Khi resample nhóm, tổng delta của các tài liệu resample chia tổng số tài liệu resample. Nhóm lớn có nhiều tài liệu đóng góp hơn; không lấy mean mỗi nhóm rồi mean ngang nhau. Query families mới cần thiết nếu các câu hỏi cùng họ có phụ thuộc. Không dùng hàng nghìn hàng ranking như các query độc lập. CI trên 2 AI queries đặc biệt chưa cho kết luận tổng quát.

Nguồn: src/research_evaluation.py::paired_bootstrap, paired_cluster_bootstrap


## Slide 75: Câu hỏi bảo vệ: vì sao không dùng mọi model?

Các lý do này là giải thích thiết kế phù hợp code, không phải tuyên bố thắng mọi baseline. Pretrained là mặc định vận hành có ít phụ thuộc checkpoint dự án; notes config ghi rõ chưa chọn theo relevance. MiniLM pretrained trên dữ liệu khác miền có thể sắp xếp lại sai; cần nhãn miền để xác minh.

Nguồn: configs/search_v2.json; docs/FINAL_RESEARCH_REPORT.md, mục 5 và 7


## Slide 76: Câu hỏi bảo vệ: hỏi gì về kết quả trả về?

Chỉ số score BM25, cosine, RRF và cross-encoder không so trực tiếp nhau như xác suất. Không có câu trả lời sinh nên yêu cầu đánh giá câu trả lời cần chuyển thành mức đáp ứng của Top5 và passage. Nếu sau này thêm RAG thì phải đánh giá faithfulness và citation riêng; đó là tính năng tương lai, không đưa vào kết quả bài này.

Nguồn: src/search_v2.py::rank, search; src/search_v2_constraints.py; app.py


## Slide 77: Hạn chế dữ liệu và tái lập

Search V2 lexical không dùng lại tokens topic có placeholder, nhưng BERTopic/LDA lịch sử có thể chịu ảnh hưởng. Không âm thầm sửa tokens rồi gọi lại là cùng thí nghiệm. Tái lập gồm corpus, model, tokenizer, configs, environment và seed; cùng seed cũng không bảo đảm kết quả giống trên mọi backend. Các số liệu slide là artifacts đã lưu, không phải chạy lại training.

Nguồn: docs/FINAL_RESEARCH_REPORT.md, mục 2.3; docs/MATH_AND_OPTIMIZATION_REVIEW.md


## Slide 78: Đối chiếu slide với code

Dùng bảng này để mở code khi được hỏi sâu. Bản notes.md đi kèm ghi lời giải thích cho từng slide và nguồn. evidence_manifest.json lưu SHA-256 của các tài liệu/cấu hình/artifacts đã dùng. Phụ lục không đưa mô hình đề xuất như SPLADE, ColBERT vào danh sách mô hình đã triển khai.

Nguồn: Đường dẫn tương đối từ D:/daln; file Python ở src/ nếu không ghi khác


## Slide 79: LDA K20: topic 0–5

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage2/topic_summary.csv


## Slide 80: LDA K20: topic 6–11

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage2/topic_summary.csv


## Slide 81: LDA K20: topic 12–17

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage2/topic_summary.csv


## Slide 82: LDA K20: topic 18–19

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage2/topic_summary.csv


## Slide 83: BERTopic baseline: topic 0–5

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 84: BERTopic baseline: topic 6–11

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 85: BERTopic baseline: topic 12–17

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 86: BERTopic baseline: topic 18–23

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 87: BERTopic baseline: topic 24–29

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 88: BERTopic baseline: topic 30–35

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 89: BERTopic baseline: topic 36–41

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 90: BERTopic baseline: topic 42–47

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 91: BERTopic baseline: topic 48–53

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 92: BERTopic baseline: topic 54–59

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 93: BERTopic baseline: topic 60–65

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 94: BERTopic baseline: topic 66–71

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 95: BERTopic baseline: topic 72–77

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 96: BERTopic baseline: topic 78–83

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 97: BERTopic baseline: topic 84–89

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 98: BERTopic baseline: topic 90–91

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv


## Slide 99: BERTopic chunked: topic 0–5

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 100: BERTopic chunked: topic 6–11

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 101: BERTopic chunked: topic 12–17

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 102: BERTopic chunked: topic 18–23

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 103: BERTopic chunked: topic 24–29

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 104: BERTopic chunked: topic 30–35

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 105: BERTopic chunked: topic 36–41

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 106: BERTopic chunked: topic 42–47

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 107: BERTopic chunked: topic 48–53

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 108: BERTopic chunked: topic 54–59

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 109: BERTopic chunked: topic 60–65

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 110: BERTopic chunked: topic 66–71

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 111: BERTopic chunked: topic 72–77

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 112: BERTopic chunked: topic 78–83

Bảng gốc được phân trang, không đặt thêm tên chủ đề. Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.

Nguồn: outputs/bertopic_chunked/topic_summary.csv


## Slide 113: Noise trong hai nhánh BERTopic

Noise không được đếm vào 92/84. Các tài liệu noise vẫn có thể là tài liệu phù hợp với query search.

Nguồn: outputs/tables/stage3/bertopic_topic_summary.csv; outputs/bertopic_chunked/topic_summary.csv


## Slide 114: Keyword heatmap của 20 topic LDA

Trọng số topic--word theo model đã chọn. Đây là phân phối từ, khác tần suất từ toàn corpus.

Nguồn: outputs/figures/stage2/lda_06_keyword_heatmap.png


## Slide 115: UMAP 2D của biểu diễn chunked

Projection 2D để quan sát. Phân cụm vẫn chạy trên 5 chiều.

Nguồn: outputs/bertopic_chunked/figures/umap_2d.png


## Slide 116: Top10 từ khóa chunked: trang 1/7

Hình gốc từ outputs. Bảy trang hình bao phủ 84 topic, mỗi topic có trọng số c-TF-IDF.

Nguồn: outputs/bertopic_chunked/figures/top10_keywords_01.png


## Slide 117: Top10 từ khóa chunked: trang 2/7

Hình gốc từ outputs. Bảy trang hình bao phủ 84 topic, mỗi topic có trọng số c-TF-IDF.

Nguồn: outputs/bertopic_chunked/figures/top10_keywords_02.png


## Slide 118: Top10 từ khóa chunked: trang 3/7

Hình gốc từ outputs. Bảy trang hình bao phủ 84 topic, mỗi topic có trọng số c-TF-IDF.

Nguồn: outputs/bertopic_chunked/figures/top10_keywords_03.png


## Slide 119: Top10 từ khóa chunked: trang 4/7

Hình gốc từ outputs. Bảy trang hình bao phủ 84 topic, mỗi topic có trọng số c-TF-IDF.

Nguồn: outputs/bertopic_chunked/figures/top10_keywords_04.png


## Slide 120: Top10 từ khóa chunked: trang 5/7

Hình gốc từ outputs. Bảy trang hình bao phủ 84 topic, mỗi topic có trọng số c-TF-IDF.

Nguồn: outputs/bertopic_chunked/figures/top10_keywords_05.png


## Slide 121: Top10 từ khóa chunked: trang 6/7

Hình gốc từ outputs. Bảy trang hình bao phủ 84 topic, mỗi topic có trọng số c-TF-IDF.

Nguồn: outputs/bertopic_chunked/figures/top10_keywords_06.png


## Slide 122: Top10 từ khóa chunked: trang 7/7

Hình gốc từ outputs. Bảy trang hình bao phủ 84 topic, mỗi topic có trọng số c-TF-IDF.

Nguồn: outputs/bertopic_chunked/figures/top10_keywords_07.png
