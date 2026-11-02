# Huấn luyện BatteryBERT trên toàn bộ văn bản sẵn có - 24/10/2026

## Phạm vi và mục tiêu

Đây là vòng huấn luyện mới, tách khỏi kết quả đã đóng băng ngày 23/10.
Corpus hiện có Title và Abstract, không có toàn văn bài báo. Không gọi
full-document trong vòng này là full article. Nếu cung cấp toàn văn, cần
ingest/version dữ liệu mới và tạo features tương ứng, không tái sử dụng cache
Abstract cho văn bản khác.

Mục tiêu chính: chấm dứt việc train chỉ trên 128 token đầu Abstract. Mọi chunk
của Abstract phải đóng góp vào biểu diễn tài liệu và nhận gradient ở block BERT
được cập nhật. Truy vấn cũng có windowing đầy đủ, không âm thầm cắt ở 128 token.

## Phương pháp mới

- Khởi đầu từ checkpoint BatteryBERT fine-tuned thật đã có.
- Đóng băng 11 block đầu; cập nhật block cuối bằng gradient thật.
- Giữ nguyên cache cũ dưới dạng read-only. Chỉ tính prefix của query và các
  chunk còn thiếu, lưu float32 kèm SHA-256. Không lượng tử hóa cache.
- Chunk 256 token, overlap 64. Pool token Abstract, không pool trực tiếp Title
  lặp lại và token SEP. Tổng hợp normalized chunk vectors với trọng số số
  token Abstract mới được bao phủ; các đoạn overlap không bị cộng lặp trọng số.
- Gradient caching hai tầng: loss contrastive có cả batch 32 cặp; replay query
  và từng nhóm chunk để backprop với RAM giới hạn. Đây là 32 cặp trong loss,
  khác gradient accumulation của các loss batch2 riêng biệt.
- Dropout tắt trong replay để gradient cache khớp chính xác; clipping,
  AdamW, warmup/decay và một hạng distillation nhẹ hạn chế drift.
- Chia batch phân tầng theo nguồn, không oversample hoặc bỏ tài liệu train.
  Mask negatives cùng group hoặc title rất giống, không tạo nhãn chuyên gia giả.

## Protocol đã chốt trước khi train

Train: 10.921 tài liệu; validation: 1.361; test: 1.368. 52 title-only chỉ được
inference. Chỉ IDs train tham gia optimizer. Pilot hai learning rates 2e-6 và
5e-6, mỗi cấu hình 24 updates; chọn bằng validation. Ứng viên tốt nhất tiếp tục
đến hết một lượt dữ liệu train, không bắt đầu lại để cộng trùng một epoch.
Lưu checkpoint và chọn tốt nhất trong cả baseline; gain validation tối thiểu
0,002. Test không dùng để chọn learning rate/checkpoint. Test là split lịch sử
đã được sử dụng trước đây, không phải external test mới.

Chỉ số chọn là paired title -> full-document Group MRR@10. **Document chunks
vẫn có Title làm context ở các block đóng băng**; đây là proxy có thể được lợi
từ Title, không phải relevance theo query độc lập. Không dùng proxy này để
tuyên bố chất lượng Search thắng BGE hoặc đạt accuracy khoa học. Topic và các
trường hợp lỗi cần được kiểm tra thêm sau khi encoder được chọn.

## Kiểm chứng và triển khai

Nhánh từ khóa mới đọc lại Title–Abstract gốc và giữ ranh giới stopword, dấu câu
và ranh giới hai trường trước khi tạo ngram. Chuỗi token LDA lịch sử đã mất
thông tin khoảng cách này nên không được dùng làm bằng chứng phrase liền kề.
Xem [đính chính riêng](KEYWORD_ADJACENCY_CORRECTION_20261024.md).

Kiểm thử cần chứng minh gradient cache khớp full-batch gradient, phần cuối văn
bản thay đổi embedding và nhận gradient, mọi token được bao phủ, sampler giữ
mỗi ID train đúng một lần, cache khớp forward BERT thật. Checkpoint/model/index
mới nằm riêng trong `outputs/batterybert_full_document_20261024_v1/`.
Các model và báo cáo cũ được giữ để đối chứng. Không tự thay model đang phục vụ
khi thiếu artifacts hoặc khi cổng đánh giá chưa đạt.

Lệnh chạy theo giai đoạn:

```powershell
.venv/Scripts/python.exe scripts/train_batterybert_full_document.py prepare
.venv/Scripts/python.exe scripts/train_batterybert_full_document.py pilot
.venv/Scripts/python.exe scripts/train_batterybert_full_document.py train
.venv/Scripts/python.exe scripts/train_batterybert_full_document.py embed
```

## Trạng thái thực tế

Đã hoàn tất continuation thật: 342 updates, 10.921 tài liệu train, 26.147
đoạn và 3.420.215 token Abstract duy nhất theo tài liệu. Checkpoint cuối được
chọn; toàn bộ corpus 13.702 tài liệu đã được encode lại. Validation proxy cùng
pooling tăng0,906174→0,933093; test tăng0,887839→0,924704.

Topic: 12 cấu hình được chọn bằng validation; tất cả cổng test của cấu hình
được chọn đều đạt. ARI test0,6888 thấp hơn0,9245 của bản trước; C_v chung
reference giảm0,560261→0,553022 và diversity0,80→0,77. Đây là triển khai tạm
thời theo mục tiêu đọc đầy đủ và cổng đa tiêu chí, không phải thắng mọi chỉ số.
Xem [kết quả và giới hạn](BATTERYBERT_FULL_DOCUMENT_RESULTS_20261024.md) và
[báo cáo toàn pipeline](../output/pdf/BAO_CAO_PIPELINE_BATTERYBERT_FULLDOC_20261024.pdf).

## Nguồn phương pháp

Khi có thân bài dạng văn bản, API inference đọc toàn bộ body qua nhiều đoạn:

```python
from pathlib import Path
from src.full_document_inference import FullDocumentEncoder

body = Path("article.txt").read_text(encoding="utf-8")
vector, coverage = FullDocumentEncoder().encode_document("Original article title", body)
print(coverage)
```

Đây là khả năng inference văn bản dài; dữ liệu training hiện vẫn là
Title–Abstract. Không có đánh giá độc lập trên Methods/Results/toàn văn, và
các đoạn không có attention toàn cục. Không dùng cache Abstract cũ cho body mới.

[Gao et al. (2021), Scaling Deep Contrastive Learning Batch Size under Memory
Limited Setup](https://arxiv.org/abs/2101.06983): cơ sở gradient caching.
Implementation hai tầng và toàn-document pooling trong dự án được kiểm chứng
bằng kiểm thử gradient riêng, không coi việc trích dẫn paper là bằng chứng code đúng.
