# Search baseline audit — 30/10/2026

Audit mã nguồn trước khi triển khai Search Radar Pilot. Corpus giữ nguyên; mọi kết quả mới nằm trong `outputs/search_radar_20261030_v1`.

## Bản đồ hệ thống

- `app.py` → `src/final_dashboard.py` → worker `FinalSearchService` → `src/final_search.py`.
- Search mặc định: `BATTERYBERT_FULLDOC_HYBRID`; `src/full_document_retrieval.py` áp dụng source/year/required terms/topic trước truy hồi, lấy TF-IDF 100 và BatteryBERT 100, RRF k=60, trả tối đa 10 tài liệu duy nhất.
- TF-IDF: `outputs/final_system/tfidf/{vectorizer.pkl,matrix.npz,ordered_doc_ids.json,manifest.json}`. Tokenizer có fingerprint; vectorizer và matrix có SHA-256. Cosine TF-IDF không cộng với cosine dense.
- BGE: `models/bge_pretrained`, CLS pooling (`1_Pooling/config.json`); `data/stage4/pretrained_embeddings.npy`, 13.702 × 768, L2. Thứ tự trong `pretrained_embedding_ids.parquet`; SHA trong `outputs/reports/stage4/pretrained_embedding_manifest.json`. Query dùng prefix `Represent this sentence for searching relevant passages: ` và giới hạn 128 tokens, đúng hợp đồng lịch sử.
- BatteryBERT: `outputs/batterybert_full_document_20261024_v1/model`, vector cùng namespace, 13.702 × 768, L2. `embedding_success.json` lưu SHA weights/tokenizer/vector. Document pooling chỉ đọc Abstract (fallback Title khi thiếu Abstract), chuẩn hóa mỗi chunk rồi trung bình theo số token duy nhất và L2. Query dùng attention-masked mean, cửa sổ 256 và overlap 64, gộp trọng số token duy nhất. Hai protocol khác nhau có chủ đích, không dùng BGE encode cho BatteryBERT.
- Qdrant collection được kiểm model/vector hash trong `full_document_retrieval.py`. Pilot có thể dùng exact matrix search trên chính các vector đã kiểm để tránh khóa local collection và giữ nguyên Qdrant hiện hành.
- `src/final_system_evaluation.py` từ chối nhãn không có VERIFIED HUMAN_INDEPENDENT và người xác minh khác người chấm. Không sửa evaluator này.

## Artifacts và rủi ro xác nhận được

Checkpoint full-document đã deploy. Có frozen-prefix cache và 32.669 chunk records, nhưng chưa có chunk embeddings của checkpoint full-document trong namespace của nó. `outputs/bertopic_chunked/chunk_embeddings.npy` thuộc mô hình khác: không được dùng thay thế. Có thể tái sử dụng prefix đã kiểm hash và chỉ chạy block cuối được chấp nhận để tạo index pilot; kiểm parity gộp lại với document vectors trước khi dùng.

Đường dẫn `selected_checkpoint` trong selection lịch sử còn trỏ `D:\daln`; resolver mới phải giải theo workspace hiện tại, không sửa manifest gốc. Không huấn luyện hoặc encode lại toàn bộ encoder.

703 AI Silver Labels chỉ bao phủ hợp Top 10 sáu phương pháp cũ. Top 100/200 và fusion mới có thể chứa ID chưa chấm. Chỉ tính P/nDCG/MRR khi Top 10 đầy đủ nhãn; giữ query chưa đủ nhãn là missing, không gán 0. So sánh macro phải dùng cùng tập query đầy đủ cho mọi cấu hình. Candidate Recall chỉ là recall trong pool, Q08 không có grade 2 nên undefined.

MiniLM từng trượt gate 0,927083 → 0,850694; chỉ được chạy đối chứng tách biệt, không tự bật production. Chưa có qrels độc lập cho query-family mới nên chưa thể lựa chọn trọng số, A7 hoặc tuyên bố cải thiện held-out.

## Điểm tích hợp và đo đạc

Thêm module/parser/evaluator/page riêng; chỉ bổ sung lựa chọn điều hướng. Giữ nguyên default, corpus, split, checkpoint, nhãn và collection. Đo initialization, warm branch latency, candidate overlap, RAM RSS, index bytes, query-level bootstrap và missing candidates trong runner có manifest. Kết quả số được ghi sau khi thực thi vào `SEARCH_IMPROVEMENT_AUDIT.md` và báo cáo cuối.
