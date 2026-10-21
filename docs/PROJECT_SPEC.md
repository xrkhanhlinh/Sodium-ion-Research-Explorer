# Đặc tả hiện tại

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
