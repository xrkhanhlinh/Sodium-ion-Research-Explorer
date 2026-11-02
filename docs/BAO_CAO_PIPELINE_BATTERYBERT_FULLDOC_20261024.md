# Báo cáo toàn bộ pipeline và huấn luyện BatteryBERT full-document - 24/10/2026

Sodium-ion Research Explorer

Kết quả thực thi, đánh giá, sửa lỗi và giới hạn - 24/10/2026

Bản cập nhật sau huấn luyện thật, thay cho việc chỉ đề xuất cách train dài hơn. Các số đo được lấy từ artifacts của lượt chạy mới; model và dữ liệu lịch sử được giữ để đối chứng.

| Corpus | Lượt train đầy đủ | Văn bản train được bao phủ |
| --- | --- | --- |
| 13.702 Title-Abstract | 10.921 tài liệu | 3.420.215 token Abstract |

### Kết quả chính

Validation proxy cùng giao thức: 0,906174 → 0,933093; test checkpoint được chọn: 0,924704. Checkpoint continuation được chấp nhận: True.

Search full-document được bật: True. Topic chính đang dùng: BatteryBERT full-document trained. Model mới chỉ được kích hoạt theo cổng đã kiểm tra; không gọi một encoder là tốt nhất ở mọi chỉ số.

> Dữ liệu hiện có Title và Abstract, chưa có toàn văn bài báo. “Đọc toàn bộ” trong thí nghiệm nghĩa là đọc hết văn bản sẵn có qua nhiều cửa sổ, không phải đã thu thập và học phần Methods/Results của mọi bài báo.

Nhãn chuyên gia và relevance người dùng độc lập còn thiếu. MRR proxy, coherence, seed agreement và tests kỹ thuật không được gọi là accuracy khoa học.

## 1. Bài toán và pipeline toàn chương trình

Hai đầu ra chính: (1) 10 cụm topic, mỗi cụm có top 10 keywords và đại diện; (2) Top 10 tài liệu cho truy vấn. Hai đầu ra có protocol đánh giá riêng.

Excel → sửa cấu trúc bảng và metadata → sàng lọc/ngôn ngữ/trùng lặp/thu hồi → corpus. Nhánh lexical tạo LDA, TF-IDF và keywords; nhánh neural tạo BGE/BatteryBERT embeddings. Topic được fit trên train; Search lọc tập doc_id trước retrieval và hợp nhất thứ hạng.

Các bước train, encode corpus và xây index chạy offline. Web đọc artifacts; mỗi truy vấn chỉ encode query, lọc, tìm kiếm và hiển thị. Topic hỗ trợ duyệt/lọc, không cộng điểm vào ranking.

| Khối | Implementation chính |
| --- | --- |
| Dữ liệu | src/pipeline.py; text_processing.py; quality.py |
| LDA/BGE/BatteryBERT lịch sử | stage2.py; stage3_neural.py; model_comparison.py |
| Training đầy đủ mới | batterybert_full_document.py; train_batterybert_full_document.py |
| Keywords/topic mới | scientific_keyword_stream.py; evaluate_batterybert_full_document.py |
| Serving | final_search.py; full_document_inference.py; full_document_retrieval.py |
| Giao diện/đánh giá | final_dashboard.py; full_document_evaluation.py |

## 2. Dữ liệu, chất lượng và split

| Nguồn/phạm vi | Bản ghi | Nguồn/phạm vi | Bản ghi |
| --- | --- | --- | --- |
| Patent Excel | 8.657 | Publication Excel | 11.919 |
| Clean master bảo toàn nguồn | 20.576 | Corpus chính | 13.702 |
| Cặp Title-Abstract | 13.650 | Title-only inference | 52 |

Corpus chính gồm 5.933 Patent và 7.769 Publication.

Làm sạch giữ doc_id, Title/Abstract, source/year/identifier, provenance và split_group_id. Sàng lọc CORE/RELATED là heuristic, chưa phải nhãn chuyên gia. Không tạo Abstract còn thiếu, không thêm full article giả và không thay corpus để làm đẹp điểm.

| Disposition | Bản ghi | Disposition | Bản ghi |
| --- | --- | --- | --- |
| LANGUAGE_REVIEW | 15 | MAIN_CORPUS | 13.702 |
| OUT_OF_SCOPE | 254 | RELEVANCE_REVIEW | 6.596 |
| CONFIRMED_DUPLICATE | 3 | RETRACTED | 4 |
| TEXT_QUALITY_REVIEW | 2 | TỔNG | 20.576 |

Kiểm tra cấu trúc Excel, ngôn ngữ, thu hồi và trùng lặp theo thứ tự ưu tiên. Tổng disposition bằng 20.576; số nhãn heuristic và disposition có thể khác do quy tắc ưu tiên.

| Split | Tài liệu | Token Abstract | Đoạn |
| --- | --- | --- | --- |
| embedding_only | 52 | 0 | 52 |
| test | 1.368 | 425.179 | 3.242 |
| train | 10.921 | 3.420.215 | 26.147 |
| validation | 1.361 | 423.952 | 3.228 |

Train cập nhật optimizer và fit geometry/KMeans/vocabulary; validation chọn learning rate/checkpoint/topic representation; test chỉ đo sau khi chốt. 52 title-only không tham gia optimizer. Các split là split lịch sử, không phải external test mới.

Grouped split hạn chế trùng nhóm giữa tập nhưng không chứng minh đã nhận diện mọi patent family hay semantic duplicate. Cần review ngoài phạm vi, khác biệt vai trò anode/cathode và trường hợp nhiều chủ đề trong cùng tài liệu.

Catalog đối chứng còn có nhóm sodium cobaltate/siêu dẫn. Đây là dấu hiệu cần rà soát phạm vi khoa học của corpus; quy tắc liên quan ở Stage1 không chứng minh mọi tài liệu đều trực tiếp nghiên cứu pin sodium-ion. Không xóa các nhóm này sau khi xem test để làm đẹp kết quả.

## 3. BatteryBERT cũ chưa tốt ở đâu và đã sửa gì

| Vấn đề | Thay đổi đã chạy |
| --- | --- |
| Train cắt 128 token | Train dùng toàn bộ các đoạn 256/64; mỗi đoạn tham gia pooling và backward |
| Batch 2, accumulation không tăng negatives trong loss | Gradient caching với batch thực 32 cặp, tối đa 31 negatives/cặp trước masking |
| Train/inference lệch cách đọc | Query đầy đủ; document pooling nhất quán lúc train, export và serving |
| Nguồn Patent/Publication chi phối topic | Tiếp tục kiểm tra train-fit source projection; không coi full-text loss tự xóa lệch nguồn |
| Ngram trên tokens đã xóa stopword | Đọc lại raw fields; giữ ranh giới từ bị loại, dấu câu và Title/Abstract |
| Báo cáo/model/encoder dễ bị trộn | Hash, ordered IDs, frozen protocol, replay parity và version riêng |

Giữ checkpoint fine-tuned lịch sử làm điểm khởi đầu; khóa 11 block đầu và train block cuối. Cache float32 của 11 block được tái sử dụng đúng hash, đọc read-only; chỉ tính bổ sung các prefix còn thiếu. Đây là cập nhật neural thật, không thay pretrained vector rồi gọi là fine-tuned.

Không xóa hàng loạt các từ như solution/dry/connect để tạo danh sách đẹp. Chúng có thể là nội dung quy trình hoặc phần cứng thật; lỗi gán cụm cần được đánh giá qua tài liệu, không chỉ tên topic.

## 4. Cơ chế đọc dài và huấn luyện

Lượt chạy dùng CPU, 4 threads, torch 2.14.1+cpu, transformers 5.20.1. Chỉ cache 11 block cố định; block cuối được replay với graph thật để giảm RAM mà giữ gradient. Không dùng GPU trong lượt này.

> Độ dài thực: 4.269.346 token Abstract trên corpus. Tối đa 1.673 token Abstract; số đoạn tối đa của một tài liệu là 14 (không phải cùng tài liệu, vì độ dài Title thay đổi payload). Không có Title trong corpus bị cắt context. Native query prefixes có 560.311 token kể cả CLS/SEP; raw Title content có 532.907 token.

Mỗi document chunk dùng template native BERT với Title làm context. Pool chỉ token Abstract, loại SEP; title-only có fallback pool nội dung Title. Chuẩn hóa vector từng chunk, gộp với trọng số số token mới, rồi chuẩn hóa tài liệu.

> chunk = L2[ masked mean(hidden states của nội dung) ]
document = L2[ Σj wj × chunkj / Σj wj ]
wj = số token nội dung mới được bao phủ ở đoạn j

Trọng số không cộng lặp độ dài overlap; token overlap vẫn ảnh hưởng ngữ cảnh và chunk mean. Toàn bộ văn bản được bao phủ nhưng không có attention toàn cục giữa mọi đoạn.

### Gradient caching hai tầng

Forward không lưu graph cho toàn batch → tính contrastive loss đối xứng với cả batch → lấy gradient query/document embeddings → truyền gradient qua phép pooling → replay từng nhóm query/chunk để cập nhật BERT. Dropout tắt để replay khớp; tests so sánh với full-batch gradient, gồm BERT attention thật.

| Thông số | Giá trị |
| --- | --- |
| Optimizer | AdamW; weight decay 0,01; clip norm 1 |
| Learning rates pilot | 2e-6 và 5e-6, mỗi cấu hình 24 updates |
| Learning rate chọn | 5e-06 |
| Epoch / số update lượt thắng | 1 / 342 |
| Batch / microbatch chunk | 32 cặp / 2 đoạn; batch cuối có 9 cặp |
| Temperature / distillation | 0,05 / 0,02 |
| Lịch LR | Warmup rồi decay; giữ best validation kể cả baseline |

Batch phân tầng nguồn, mỗi ID train xuất hiện đúng một lần trong lượt hoàn chỉnh. Mask same-group và title gần trùng; chưa có nhãn khoa học để khẳng định đã loại mọi false negative.

## 5. Kết quả encoder và đánh giá thống kê

| Phép đo cùng full-document protocol | MRR@10 |
| --- | --- |
| Validation baseline | 0,906174 |
| Validation checkpoint được chọn | 0,933093 |
| Test baseline, cùng pooling | 0,887839 |
| Test checkpoint được chọn | 0,924704 |

Checkpoint được chọn tại step 342, đã thấy 10.921 tài liệu trong continuation. Toàn lượt train đã backward qua 26.147 đoạn của 10.921 tài liệu. Hai phạm vi này được ghi riêng.

L2 thay đổi block cuối được export: 0,312042. Validation/test xuất hiện trong optimizer: 0. Model reload parity: PASS.

validation: gain paired=0,026919, khoảng bootstrap percentile 95% [0,021105; 0,033229], resample theo 1358 split_group_id, 10.000 lượt.

test: gain paired=0,036865, khoảng bootstrap percentile 95% [0,030158; 0,044017], resample theo 1359 split_group_id, 10.000 lượt.

![Learning curve](../outputs/batterybert_full_document_20261024_v1/learning_curve.png)

> Proxy là Title → document có Title context. Nó có thể được lợi từ Title; không phải relevance query độc lập. Khoảng validation còn chịu selection bias. Bootstrap trên test đánh giá proxy cho checkpoint cố định, không xác nhận accuracy khoa học hoặc thắng BGE trên Search.

Các số MRR lịch sử 128-token/BGE dùng protocol khác được giữ làm bối cảnh; không ghép validation của model này với test của model khác để kết luận encoder thắng tuyệt đối.

## 6. Tối ưu 10 topic và từ khóa

Fit hai geometry center/source, KMeans10 seed 42 và seed 43 trên train. Với mỗi geometry, thử exclusivity 0/0,25/0,5 và phrase boost 1/1,5: tổng 12 cấu hình representation. Vocabulary chỉ fit train, min_df 5, max_df 0,6, tối đa 40.000 features; mỗi keyword có ≥5 tài liệu train hỗ trợ.

| Ứng viên validation | V nguồn | ARI | NPMI | Div. | Đạt |
| --- | --- | --- | --- | --- | --- |
| center_ex0.0_pb1.0 | 0,988 | 0,983 | 0,128 | 0,68 | False |
| center_ex0.0_pb1.5 | 0,988 | 0,983 | 0,124 | 0,68 | False |
| center_ex0.25_pb1.0 | 0,988 | 0,983 | 0,135 | 0,76 | False |
| center_ex0.25_pb1.5 | 0,988 | 0,983 | 0,122 | 0,77 | False |
| center_ex0.5_pb1.0 | 0,988 | 0,983 | 0,100 | 0,89 | False |
| center_ex0.5_pb1.5 | 0,988 | 0,983 | 0,088 | 0,89 | False |
| source_ex0.0_pb1.0 | 0,316 | 0,699 | 0,081 | 0,64 | False |
| source_ex0.0_pb1.5 | 0,316 | 0,699 | 0,076 | 0,67 | False |
| source_ex0.25_pb1.0 | 0,316 | 0,699 | 0,081 | 0,77 | True |
| source_ex0.25_pb1.5 | 0,316 | 0,699 | 0,067 | 0,80 | True |
| source_ex0.5_pb1.0 | 0,316 | 0,699 | 0,039 | 0,89 | True |
| source_ex0.5_pb1.5 | 0,316 | 0,699 | -0,032 | 0,91 | False |

Ứng viên được chọn: source_ex0.25_pb1.0. Cổng đặt trước: lớn nhất≤30%, top 2≤45%, nhỏ nhất≥0,5%, V≤0,5, ARI≥0,6, diversity≥0,75, support coverage≥95%, NPMI≥0. Seed 137 kiểm tra test sau chọn; không refit toàn corpus.

Cổng topic test: True; gate triển khai tổng: True. Topic mới chỉ thay catalog chính khi qua gate và có review agent theo đủ 10 nhóm. Gate kỹ thuật không phải chuẩn accuracy khoa học.

Có 1.185 tài liệu được đánh dấu ưu tiên rà soát vì xa centroid hoặc gần hai topic; ngưỡng lấy từ train. Danh sách không thay assignments và không được hiểu là xác suất model sai.

## 7. So sánh topic và chọn model chính

| Cùng reference mới | C_v | Diversity | Keyword coverage |
| --- | --- | --- | --- |
| previous | 0,5603 | 0,80 | 100,00% |
| full_document | 0,5530 | 0,77 | 100,00% |

Reference lần này dùng tokens nguồn có ranh giới, phrase thật và cùng window 110/top 10. Chỉ so hai dòng trong bảng trực tiếp với nhau; C_v cũ tính trên reference khác được ghi dưới dạng lịch sử.

So sánh topic là so pipeline: encoder, document pooling, geometry và trọng số keywords có thể cùng khác. Không quy toàn bộ chênh lệch topic cho neural fine-tuning. Phép so encoder baseline/continuation ở mục 5 mới giữ cùng full-document pooling.

ARI ở đây đo agreement giữa các lần fit KMeans với seed khác; không phải tỷ lệ phân loại đúng. Neural training mới chỉ dùng một seed, nên chưa đo biến thiên qua nhiều lần train BERT độc lập.

| Pipeline lịch sử | C_v lịch sử | Giới hạn |
| --- | --- | --- |
| BatteryBERT pretrained/HDBSCAN | 0,7436 | Hai cụm 92,76%; source V≈0,9986 |
| BatteryBERT pretrained/KMeans | 0,6941 | Kích thước tốt hơn, vẫn lệch nguồn rất mạnh |
| BatteryBERT fine-tuned/projection 23/10 | 0,5601 | Test ARI 0,9245; còn tài liệu lệch nhóm |
| LDA K10 | 0,5297 | Đối chứng lexical, không có encoder |
| BERTopic BGE K10 | 0,6455 | Outlier 22,77% trong nhánh lịch sử |

Catalog chính hiện hành: BatteryBERT full-document trained. Trên toàn corpus: source V=0,3036, top 2=24,59%. KMeans luôn gán cụm; 0% outlier không chứng minh mọi tài liệu phù hợp.

Model tốt nhất phụ thuộc đầu ra: baseline C_v cao có thể collapse theo nguồn; encoder thắng proxy chưa chắc thắng relevance. Chọn BatteryBERT làm model chính theo yêu cầu đề tài, đồng thời giữ BGE/LDA và checkpoint cũ làm đối chứng trung thực.

Stage2 chọn LDA K=20 bằng protocol riêng. LDA K10 trong bảng trên là đối chứng ép cùng 10 topic; không thay kết quả Stage2 bằng K10. Các pilot AI-judge và synthetic source-ID recovery lịch sử không được gộp thành relevance có nhãn người.

| Proxy test lịch sử, protocol riêng | Group MRR@10 |
| --- | --- |
| BGE pretrained | 0,8936 |
| BGE fine-tuned | 0,9281 |

BGE fine-tuned dẫn trong phép so BGE lịch sử này. Các số trên không phải benchmark Search có nhãn người và không so trực tiếp với FULLDOC có Title context để kết luận encoder nào tốt nhất tuyệt đối.

### 10 topic đối chứng: BatteryBERT fine-tuned · optimized

Tên diễn giải lấy từ catalog hiện hữu. ID không tương ứng giữa các model; chưa có chuyên gia xác nhận. Bảng đầy đủ top 10 keywords được xuất riêng tại docs/TOP 10_TOPICS_ALL_MODELS_20261024.md.

| ID | Nhãn trong catalog | Tài liệu | Ba keywords đầu |
| --- | --- | --- | --- |
| 0 | Anode MoS2, dichalcogenide lớp và MXene | 1.031 | MoS2; graphene; mxene |
| 1 | Cathode phosphate vanadium và NASICON | 1.342 | phosphate; vanadium; Na3V2(PO4) |
| 2 | Cathode Prussian blue và hợp chất sắt | 1.353 | prussian_blue; iron; sodium |
| 3 | Cấu trúc cell, bộ góp dòng và lắp ráp pin | 1.084 | connect; plate; device |
| 4 | Điện ly, muối, dung môi và polymer | 1.532 | electrolyte; polymer; additive |
| 5 | Anode sulfide/selenide và composite carbon | 1.235 | sulfide; carbon; composite |
| 6 | Anode hợp kim Sn/Sb, phosphorus và chuyển hóa | 915 | antimony; sb; tin |
| 7 | Hard carbon, carbon xốp và tiền chất sinh khối | 1.703 | hard_carbon; carbon; biomass |
| 8 | Anode oxide Ti/Sn và titanate | 1.444 | TiO2; titanium; titanate |
| 9 | Cathode oxide lớp P2/O3 và kim loại chuyển tiếp | 2.063 | P2; layer; oxide |

### 10 topic đối chứng: BatteryBERT KMeans (pretrained)

Tên diễn giải lấy từ catalog hiện hữu. ID không tương ứng giữa các model; chưa có chuyên gia xác nhận. Bảng đầy đủ top 10 keywords được xuất riêng tại docs/TOP 10_TOPICS_ALL_MODELS_20261024.md.

| ID | Nhãn trong catalog | Tài liệu | Ba keywords đầu |
| --- | --- | --- | --- |
| 0 | Wet-chemical synthesis of electrode materials | 2.309 | solution; mix; dry |
| 1 | Active-material formulations and electrode coatings | 1.677 | electrode; active; active material |
| 2 | Mixed electrode performance and electrolyte studies | 1.624 | electrolyte; capacity; cycle |
| 3 | Anode composites and cycling performance | 1.622 | ma; capacity; mah |
| 4 | Sulfide/selenide-carbon composite anodes | 1.603 | MoS2; storage; cycle |
| 5 | Inorganic electrode compositions and chemical claims | 1.470 | sodium; formula; material |
| 6 | Layered-oxide and phosphate cathode materials | 1.351 | P2; cathode; capacity |
| 7 | Crystal structure, redox and intercalation studies | 1.291 | na; phase; redox |
| 8 | Cell hardware and assembly equipment | 480 | connect; plate; device |
| 9 | Reviews and research perspectives | 275 | sib; application; energy_storage |

### 10 topic đối chứng: LDA

Tên diễn giải lấy từ catalog hiện hữu. ID không tương ứng giữa các model; chưa có chuyên gia xác nhận. Bảng đầy đủ top 10 keywords được xuất riêng tại docs/TOP 10_TOPICS_ALL_MODELS_20261024.md.

| ID | Nhãn trong catalog | Tài liệu | Ba keywords đầu |
| --- | --- | --- | --- |
| 0 | Sodium-storage anodes and carbon composites | 3.486 | anode; cycle; capacity |
| 1 | Non-aqueous electrolytes and ionic transport | 997 | battery; electrolyte; study |
| 2 | Cathode electrochemistry and vanadium phosphates | 3.559 | capacity; cathode; cycle |
| 3 | Electrode-material inventions (mixed materials) | 869 | material; anode; sodium_ion |
| 4 | Transition-metal and Prussian-white formulations | 78 | nc; material; iron |
| 5 | Layered oxide cathodes and P2/O3 structures | 84 | layer; transition; P2 |
| 6 | Active electrode materials and characterization | 1.286 | electrode; battery; material |
| 7 | Material synthesis and preparation processes | 2.529 | solution; obtain; material |
| 8 | Battery devices, housings and assembly | 331 | successfully; battery; connect |
| 9 | Hard-carbon preparation and related materials | 483 | carbon; material; hard_carbon |

### 10 topic đối chứng: BERTopic–BGE

Tên diễn giải lấy từ catalog hiện hữu. ID không tương ứng giữa các model; chưa có chuyên gia xác nhận. Bảng đầy đủ top 10 keywords được xuất riêng tại docs/TOP 10_TOPICS_ALL_MODELS_20261024.md.

| ID | Nhãn trong catalog | Tài liệu | Ba keywords đầu |
| --- | --- | --- | --- |
| 0 | Electrode-material preparation patents | 3.552 | obtain; material; prepare |
| 1 | Layered-oxide and polyanion cathodes | 2.650 | P2; cathode; capacity |
| 2 | Alloy/conversion and carbon-based anodes | 1.261 | sn; capacity; SnO2 |
| 3 | Sulfide/selenide and carbon-composite anodes | 1.008 | MoS2; FeS2; cycle |
| 4 | Prussian-blue analogue cathodes | 499 | prussian_blue; ferrocyanide; prussian |
| 5 | Titanate and TiO2 anodes | 425 | TiO2; Na2Ti3O7; anatase |
| 6 | Electrolyte salts, solvents and additives | 397 | electrolyte; additive; solvent |
| 7 | Battery hardware and manufacturing devices | 382 | connect; plate; device |
| 8 | Sodium cobalt oxides and superconductivity | 219 | NaxCoO2; NaCrO2; cuo |
| 9 | MXene-based sodium-storage electrodes | 189 | mxene; Ti3C2Tx; mxenes |

### 10 topic đối chứng: BERTopic–BatteryBERT (pretrained)

Tên diễn giải lấy từ catalog hiện hữu. ID không tương ứng giữa các model; chưa có chuyên gia xác nhận. Bảng đầy đủ top 10 keywords được xuất riêng tại docs/TOP 10_TOPICS_ALL_MODELS_20261024.md.

| ID | Nhãn trong catalog | Tài liệu | Ba keywords đầu |
| --- | --- | --- | --- |
| 0 | Electrode performance (broad Publication cluster) | 7.261 | capacity; cycle; ma |
| 1 | Material preparation methods (broad Patent cluster) | 5.449 | obtain; solution; prepare |
| 2 | Reviews and research outlook | 218 | review; recent; summarize |
| 3 | 2D anodes and first-principles calculations | 164 | na; monolayer; adsorption |
| 4 | Fixtures, protection and inspection equipment | 156 | connect; plate; fixedly |
| 5 | Assembly, heating and mechanical equipment | 101 | device; connect; plate |
| 6 | Battery packs, management and control | 80 | module; unit; voltage |
| 7 | Sodium cobaltate superconductivity | 79 | NaxCoO2; superconducting; yh |
| 8 | Cell housing and terminals | 48 | portion; terminal; electric |
| 9 | Mixed high-performance electrode materials | 37 | high-performance; snsb-porous; vapor-redistribution |

## 8. 10 topic hiện hành: nhóm 0-4

ID chỉ có ý nghĩa trong catalog đang hiển thị. Nhãn là diễn giải tạm thời, chưa phải nhãn chuyên gia. Số lượng lấy từ toàn corpus.

### Topic 0: Oxide Ti/Sn, titanate và composite điện cực

1.408 tài liệu (10,28%).

Top 10: TiO2; titanium; titanate; solution; dioxide; SnO2; oxide; anode; carbon; composite.

### Topic 1: Prussian blue và các hợp chất điện cực chứa sắt

1.483 tài liệu (10,82%).

Top 10: iron; prussian_blue; solution; sodium; ferrous; anode; prussian; cathode; ferrocyanide; FeS2.

### Topic 2: Oxide lớp P2/O3 và biến đổi cấu trúc điện cực

1.717 tài liệu (12,53%).

Top 10: P2; layer; oxide; cathode; O3; layered_oxide; type; transition; phase; metal.

### Topic 3: Điện ly, polymer, muối và phụ gia giao diện

1.426 tài liệu (10,41%).

Top 10: electrolyte; polymer; additive; organic; solvent; salt; carbonate; sodium; electrode; non-aqueous.

### Topic 4: Phosphate vanadium, NVP/NASICON và composite carbon

1.192 tài liệu (8,70%).

Top 10: phosphate; Na3V2(PO4); vanadium; nvp; phosphorus; carbon; sodium; cathode; composite; fluorophosphate.

## 9. 10 topic hiện hành: nhóm 5-9

ID chỉ có ý nghĩa trong catalog đang hiển thị. Nhãn là diễn giải tạm thời, chưa phải nhãn chuyên gia. Số lượng lấy từ toàn corpus.

### Topic 5: Anode hợp kim Sn/Sb, graphene và MXene

1.266 tài liệu (9,24%).

Top 10: graphene; antimony; mxene; sb; anode; composite; sn; alloy; electrode; tin.

### Topic 6: Phần cứng cell và thiết bị chế tạo, kiểm tra pin

925 tài liệu (6,75%).

Top 10: connect; plate; device; end; pole; piece; body; electric; electrode; collector.

### Topic 7: Cấu trúc pha, trật tự vacancy và dẫn ion

1.035 tài liệu (7,55%).

Top 10: phase; na; NaxCoO2; sodium; compound; electrode; cathode; ion; solid_electrolyte; active.

### Topic 8: Hard carbon, carbon xốp và tiền chất sinh khối

1.653 tài liệu (12,06%).

Top 10: hard_carbon; carbon; biomass; porous; carbonize; precursor; carbonization; dry; asphalt; anode.

### Topic 9: Sulfide/selenide, MoS2 và composite carbon

1.597 tài liệu (11,66%).

Top 10: MoS2; sulfide; disulfide; molybdenum; carbon; composite; selenide; anode; solution; SnS2.

### Phân bố nguồn trong 10 topic hiện hành

Các nhóm không bị ép có kích thước bằng nhau. Phân bố theo nguồn được kiểm tra riêng vì topic có thể học văn phong Patent/Publication.

| ID | Patent | Publication | Tổng |
| --- | --- | --- | --- |
| 0 | 554 | 854 | 1.408 |
| 1 | 607 | 876 | 1.483 |
| 2 | 685 | 1.032 | 1.717 |
| 3 | 742 | 684 | 1.426 |
| 4 | 424 | 768 | 1.192 |
| 5 | 421 | 845 | 1.266 |
| 6 | 860 | 65 | 925 |
| 7 | 293 | 742 | 1.035 |
| 8 | 818 | 835 | 1.653 |
| 9 | 529 | 1.068 | 1.597 |

## 10. Search Top 10, bộ lọc và relevance

| Nhánh | Cơ chế |
| --- | --- |
| FULLDOC semantic | Query nhiều cửa sổ → checkpoint mới → Qdrant cosine |
| FULLDOC hybrid | TF-IDF Top 100 + FULLDOC dense Top 100 → RRF k=60 → Top 10 |
| 4 phương pháp nghiên cứu cũ | TF-IDF, BatteryBERT pretrained semantic/hybrid, BGE semantic |
| V2/V3 lịch sử | BM25 + BGE/BatteryBERT fine-tuned + MiniLM |

> RRF(d) = Σnhánh 1 / [60 + ranknhánh(d)]. Cosine, TF-IDF, RRF và MiniLM khác thang đo; chúng không phải xác suất relevance.

Nguồn/năm/từ bắt buộc/topic tạo tập doc_id hợp lệ trước khi rank ở cả lexical/dense. Gắn nhãn topic hiện hành sau retrieval bằng doc_id. Encoder query phải đúng checkpoint/pooling với corpus vectors; không thay pretrained cùng 768 chiều rồi coi là tương đương.

Benchmark chung: 20 query, 6 phương pháp, 1.200 hàng Top 10; pool có 703 cặp. Nhãn người được chấp nhận: 0; query đủ pool: 0/20. Trạng thái: PENDING_HUMAN_ANNOTATION.

| FULLDOC warm CPU | Mean giây | Median | Requests |
| --- | --- | --- | --- |
| BATTERYBERT_FULLDOC_HYBRID | 0,543 | 0,537 | 19 |
| BATTERYBERT_FULLDOC_SEMANTIC | 0,632 | 0,575 | 19 |

Warm loại request đầu mỗi phương pháp. Đo tuần tự, chưa đo concurrency. Phiếu mới ẩn method/rank/score; grade0–2, P@10/MRR@10 dùng grade≥2, nDCG gain 2^grade−1. Chỉ tính trên query có toàn pool được người chấm và người xác minh khác nhau chấm đủ. Nhãn trống không phải 0; chưa báo Recall toàn corpus.

### Tự đọc kết quả Search và kiểm tra hiệu năng

Agent đã xem 200 title Top 10 của 20 truy vấn FULLDOC hybrid, đọc đủ Title–Abstract của 6 kết quả Top 1 được kiểm tra sâu. Không sinh grade relevance và không dùng để đổi model sau benchmark.

Top 1 abstract xác nhận anionic redox trong P2 oxide dù title không nêu redox; ví dụ title-only có thể thiếu bằng chứng.

Top 1 là review NASICON NVP có carbon coating trong abstract; không phải nghiên cứu thực nghiệm coating duy nhất.

Top 1 là NaFSI/PEO polymer electrolyte, abstract nêu phép đo ở nhiệt độ cao; chưa thỏa rõ yêu cầu ether liquid và low-temperature. Truy vấn nhiều ràng buộc còn yếu.

Top 1 là thấm electrolyte bằng hot/cold rolling của cell; chưa phải chính xác thiết bị phủ/ép điện cực. Có lexical overlap nhưng sai công đoạn.

Top 1 là giảm voltage drop của pack; abstract không chứng minh thermal management. Tài liệu thermal cụ thể ở hạng thấp hơn.

Top 1 mô tả tái chế vật liệu cathode; nhưng các title Top 10 còn cathode review và tái chế khẩu trang, cần review relevance đầy đủ.

Các title về hard carbon, MoS2, Sn4P3 và NASICON nhìn chung cùng miền truy vấn; chưa kiểm định mọi ràng buộc trên toàn bộ abstract.

Q20 có hai kết quả cùng title Guidelines for Air-Stable Lithium/Sodium Layered Oxide Cathodes; khác doc_id cần kiểm tra DOI/family trước gộp.

Không dùng các truy vấn benchmark này để chỉnh model sau khi đọc kết quả; giữ nguyên lỗi và thứ hạng làm bằng chứng.

Tối ưu phục vụ sau benchmark: đổi tập doc_id hợp lệ thành point IDs đã xác minh, vẫn lọc trước ranking. Ba lượt trên collection thật (693 tài liệu hợp lệ) giữ nguyên Top 100 và scores; median riêng bước retrieval 3,690 → 2,535 giây, nhanh 1,46 lần. Đây không phải gain relevance và không thay 1.200 thứ hạng benchmark đã lưu.

Lượt web đầu có một timeout 240 giây ở truy vấn dài/lọc kết hợp; XML thất bại được giữ tại web_initial_tests.xml. Chạy riêng đã qua; profiler xác định cold import/model loading và bộ lọc Qdrant Local có chi phí đáng kể. Kết quả kiểm thử cuối được ghi riêng sau tối ưu. Chưa quy toàn bộ timeout đầu cho một nguyên nhân duy nhất.

## 11. Tự kiểm tra, lỗi còn lại và nhận xét

Đã review 60 tài liệu đại diện ở 10 topic mới: 3 gần centroid và 3 random có seed mỗi topic. Đây là nhận xét agent, không phải expert ground truth hay ước lượng precision. Phạm vi đọc và các ca lỗi được lưu cùng review.

30 mẫu central cho thấy trọng tâm nội dung của 10 nhóm; mẫu random còn giao thoa và lệch nhóm. Đây là nhận xét agent, không phải ước lượng precision.

Nhóm 7 có đại diện NASICON, trật tự pha/vacancy và dẫn ion; đây là chủ đề cơ chế/cấu trúc, không phải một họ vật liệu duy nhất. Từ khóa na/compound/active vẫn chưa đặc trưng.

Nhóm 6 thực sự có cấu trúc cell, thiết bị sấy và kiểm tra. connect/plate/end là ngôn ngữ phần cứng; blacklist không sửa được thành viên bị gán lệch.

Các ca lệch đã thấy: Fe-phosphate trong nhóm trọng tâm PB, P2 oxide trong nhóm NVP, sodium-nickel oxide trong nhóm Sn/Sb, polymer điện cực hoạt tính trong nhóm điện ly.

Một số patent gọi anode/cathode chưa nhất quán với mô tả vật liệu; không tự sửa cực tính của nguồn bằng suy đoán.

Top 2 toàn corpus giảm 27,49%→24,59%; nhưng ARI test giảm 0,9245→0,6888, diversity0,80→0,77 và C_v chung reference0,560261→0,553022. Không tuyên bố topic mới thắng mọi chỉ số.

Tích hợp tạm thời vì encoder đã học các đoạn đầy đủ và qua cổng kỹ thuật/review định trước. Giữ topic cũ làm đối chứng ổn định seed tốt hơn; còn chờ nhãn chuyên gia độc lập.

Audit keywords cũ: 80 từ khác nhau đều có support nguồn; không xác nhận danh sách top 10 cũ có phrase giả. Sửa mới bảo toàn ranh giới để ngram tương lai không nối qua stopword/field. Số file dữ liệu/checkpoint được xác minh giữ nguyên: 5.

| Kiểm chứng kỹ thuật mới | Kết quả |
| --- | --- |
| Tests riêng biệt | 112 qua / 0 lỗi / 0 errors |
| Numerical parity | Prefix/last-block, full BERT export, query và document API |
| Full-text contracts | Coverage, gradient tail, same-first128/different-tail query |
| Data/topic/search | Hash/ID train-only, keyword support, predict reload, filters |
| Web/HTTP | AppTest và server smoke thực tế |

> Điều kiện dừng là hoàn thành protocol đã định trước và kiểm tra các gate; không tuyên bố đã đạt tối ưu tuyệt đối. Chưa có toàn văn bài báo, nhãn topic/relevance độc lập, kiểm chứng ngoài corpus hoặc đo production đa người dùng.

## 12. Tái lập và hướng nghiên cứu tiếp

.venv/Scripts/python.exe run_demo.py

Chạy train trong thư mục MỚI:
.venv/Scripts/python.exe scripts/train_batterybert_full_document.py all
  --output outputs/batterybert_full_document_REPLAY
.venv/Scripts/python.exe scripts/evaluate_batterybert_full_document.py
  --output outputs/batterybert_full_document_REPLAY

Giữ đủ frozen corpus/split, checkpoint khởi đầu, cache 11 block và versions. Resume có fingerprint, không sửa source/config rồi tiếp tục dùng tên run cũ. Artifact index/catalog đang phục vụ nằm trong run v1, không tự tải hoặc train khi mở web.

| Artifacts | Nội dung |
| --- | --- |
| training_protocol / code_snapshot | Config, version, SHA nguồn/input và protocol đã đóng băng |
| training_document_coverage / training_coverage | ID optimizer, số đoạn/token thực sự backward |
| selection / paired_training_effect | Lựa chọn validation và so sánh paired sau chọn |
| model / document_embeddings / ordered_doc_ids | Checkpoint export thật và vectors đúng thứ tự |
| topics / annotation_pool / rankings | Topic candidates, đại diện, pool 6 phương pháp và kết quả thật |
| verification_summary / preservation_check | Kiểm thử riêng biệt và bảo toàn dữ liệu |

### Các bước tiếp theo cần thêm bằng chứng

Ưu tiên tập scientific-topic độc lập có nhãn chuyên gia và các ca giao thoa/ngoài phạm vi. Tạo positives cùng chủ đề giữa Patent/Publication và hard negatives tránh false negatives; chọn bằng validation khoa học. Với full article, cần ingest/version thân bài và đánh giá mới, không tái dùng cache Abstract cho văn bản khác.

Có thể so sánh hierarchical pooling/adapters/nhiều block và batch lớn hơn khi có tài nguyên; chỉ gọi là cải thiện sau phép so cùng protocol. Giữ baseline và lỗi quan sát được, không sửa test để bắt BatteryBERT thắng.

### Phương pháp và nguồn

Gao et al. (2021), Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/abs/2101.06983. Model card: https://huggingface.co/batterydata/batterybert-cased. BERTopic clustering: https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html

Các số đo nội bộ được đọc từ artifacts; evidence manifest đi kèm ghi SHA-256. Review agent và kế hoạch nghiên cứu được tách khỏi kết quả có nhãn. Báo cáo cũ ngày 23/10 được giữ như snapshot lịch sử, có đính chính về tính liền kề tokens.
