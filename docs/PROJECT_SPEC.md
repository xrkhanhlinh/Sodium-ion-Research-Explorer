# Đặc tả hiện tại

**Hiện hành24/10/2026:** BatteryBERT full-document là nhánh mới: train đủ các
đoạn Title–Abstract, native chunk256/64, loss batch32 bằng gradient caching,
khóa11 block và cập nhật block cuối. Chỉ train10.921 IDs; validation1.361 chọn
LR/checkpoint; test1.368 đo sau lựa chọn; 52 title-only chỉ inference.

Search full-document dùng checkpoint mới và vectors đúng hash/doc_id trong
Qdrant riêng, hợp nhất TF-IDF/dense bằng RRF60. Query có windowing đầy đủ;
không áp dụng giới hạn128-token của V2 lịch sử. Bộ lọc áp dụng trước ranking.
Topic dùng train-fit source projection/KMeans10 và keywords từ raw fields có
giữ ranh giới stopword/dấu câu/field; không dùng topic cộng điểm ranking.

Model/index/catalog chỉ kích hoạt theo `outputs/batterybert_full_document_20261024_v1/deployment.json`.
Các model cũ giữ làm đối chứng. Corpus chưa có toàn văn bài báo; không gọi
phép thử Title→document có Title context là relevance Search độc lập. ARI
là agreement KMeans, không phải accuracy. Không có nhãn người thì metrics
relevance0–2 của pool6 phương pháp vẫn để trống.

Implementation mới: `src/batterybert_full_document.py`,
`src/scientific_keyword_stream.py`, `src/full_document_inference.py`,
`src/full_document_retrieval.py`, `src/full_document_topics.py`,
`src/full_document_evaluation.py`; serving chung qua `src/final_search.py`
và `src/final_dashboard.py`. Phần dưới mô tả các nhánh lịch sử.

**Cập nhật 23/10/2026:** Search mặc định hiện là TF-IDF + BatteryBERT pretrained/Qdrant/RRF, topic chính là BatteryBERT fine-tuned/projection/KMeans10. [Pipeline và bằng chứng hiện hành](BATTERYBERT_OPTIMIZATION_20261023.md). Mô tả BGE/MiniLM phía dưới là cấu hình V2 lịch sử, vẫn có thể chọn để đối chứng.

- Corpus: data/processed/main_corpus.parquet, đúng 13.702 doc_id duy nhất. Không trộn UNCERTAIN vào thí nghiệm hiện tại.
- Giữ Title/Abstract, provenance/source/year/identifier/split_group_id. Không tạo nội dung thiếu hoặc sửa corpus để tăng điểm.
- BGE pretrained + MiniLM reranker là đường mặc định. Fine-tuned/chunked là ablation tùy chọn.
- LDA/BERTopic phục vụ khám phá topic riêng, không cộng topic bonus vào score v2.

| Thành phần | Nguồn hiện tại |
|---|---|
| UI | app.py |
| Worker/compare/rank matrix | src/search_v2_service.py |
| Lexical/dense/hybrid/rerank | src/search_v2.py |
| Required terms | src/search_v2_constraints.py |
| Config | configs/search_v2.json |
| Benchmark | scripts/benchmark_search_v2.py |
| Latency không cache | scripts/measure_search_v2_live.py |
| Relevance 14 methods | scripts/evaluate_search_v2.py, src/research_evaluation.py |
| Xác minh hash lịch sử | src/artifact_history.py |

Stage4 search còn tên file chỉ là alias, không chứa implementation song song. Backend Qdrant, generator báo cáo Stage4/5 và slides trước v2 đã đưa vào archive. Dữ liệu/model/bảng kết quả baseline không xóa.

Dense matrix phải đúng thứ tự doc_id/checkpoint, hữu hạn và L2-normalized. Query dùng instruction BGE; neural query quá 128 tokens bị từ chối để rút ngắn. Reranker đọc tối đa hai passage của Top50, không phải full text. Required terms và công thức chỉ là lexical evidence, không phải kiểm chứng hóa học/ngữ nghĩa.

Không có nhãn độc lập thì P@5/nDCG@5/MRR@5 để trống; không dùng AI grades/cosine/coherence/membership làm qrels. Proxy Title→Abstract ghi đúng tên và phạm vi. Phân biệt development queries, query tác giả soạn và benchmark độc lập.

Stage1/2/3 và chunking vẫn có giá trị tái lập nghiên cứu, không chạy khi app khởi động. Progress/report phạm vi từng Stage ghi vào outputs/reports/stageN/, không ghi đè tài liệu tổng hợp hiện hành.

Hash lịch sử của code/report được tra từ ZIP có manifest. Data/model phải luôn đúng hash từ file thật; không dùng archive để che thay đổi dữ liệu hoặc weights.
