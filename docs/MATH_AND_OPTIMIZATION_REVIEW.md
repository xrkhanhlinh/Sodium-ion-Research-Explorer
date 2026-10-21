# Kiểm chứng toán học và tối ưu code — 19/10/2026

## Kết luận và phạm vi

Các công thức retrieval/evaluation được kiểm tra bằng đối chiếu SymPy, ví dụ tính tay, phản ví dụ có giới hạn và regression tests. Có lỗi biên và lỗi diễn giải chỉ số cần sửa; chưa tìm thấy bằng chứng rằng các điểm relevance chính thức hiện có bị tính sai, vì dự án chưa có nhãn relevance độc lập để tính các điểm đó.

**Tham số vận hành giữ lại: `threads=4`, `reranker_batch_size=4`.** Đây là lựa chọn tốt nhất quan sát được trong lưới chín cấu hình CPU đã đo trên máy này, với ba query development và hai lượt/query. Không phải nghiệm tối ưu toàn cục hoặc cấu hình tốt nhất về relevance. Hai tối ưu hàm thành phần được xét riêng bằng gate giữ nguyên kết quả.

Không huấn luyện lại transformer, LDA, BERTopic, UMAP hoặc HDBSCAN. Không viết lại corpus, weights, nhãn hay kết quả thực nghiệm lịch sử. Kiểm tra nguồn của các thư viện tối ưu bên trong không đồng nghĩa chứng minh toán học toàn bộ các thư viện đó.

## Các hàm toán học đã kiểm tra

| Nhóm | Định nghĩa đối chiếu | Kết quả và giới hạn |
|---|---|---|
| RRF | `score(d) = sum(1/(k + rank_j(d)))`, rank bắt đầu từ 1 | SymPy xác nhận giảm theo rank khi `k>=0`; fixture hai danh sách khớp chính xác. Thêm kiểm tra miền `k` và ID lặp trong cùng một danh sách. |
| BM25 | `IDF * f*(k1+1)/(f + k1*(1-b+b*L/avgL))` | Khớp dependency đã cài trên corpus nhỏ tính tay. Code dùng tổng hai field riêng, có trọng số Title và clipping; không gọi đây là suy dẫn BM25F đầy đủ. |
| Dense similarity | `cos(u,v) = u·v/(||u||||v||)` | Dot product bằng cosine khi cả hai vector khác 0 đã chuẩn hóa. Kiểm tra query arrays thực tế 24×768; code có kiểm tra chuẩn hóa document arrays. Không chứng minh embedding mang đúng ý nghĩa khoa học. |
| Precision@5 | số grade≥2 trong Top-5, chia cố định cho 5 | Đối chiếu exact, danh sách thiếu kết quả vẫn dùng mẫu số 5 theo protocol. Nhãn thiếu/NaN/boolean/phân số/ngoài miền bị từ chối. |
| MRR@5 | nghịch đảo hạng đầu tiên có grade≥2, hoặc 0 | Khớp reference độc lập. |
| nDCG@5 | gain `2^grade-1`, discount `log2(rank+1)`, IDCG từ pool chung | 150 trường hợp ngẫu nhiên có giới hạn khớp SymPy. IDCG=0 được quy ước trả 0. Grade 1 vẫn đóng góp nDCG dù chưa đạt ngưỡng Precision/MRR. |
| Bootstrap theo query | cùng chỉ số resample cho cả hai phương pháp | Hai phương pháp giống nhau có delta và CI bằng 0. Không xác minh độ phủ CI cho mọi quần thể query. |
| Bootstrap theo nhóm | tổng delta resample / tổng số tài liệu resample | Ví dụ nhóm kích thước 1 và 2 cho mean `2/3`, đúng estimand trung bình theo tài liệu; khác với cân bằng trọng số từng nhóm. |
| Weighted kappa | `1 - observed_disagreement/expected_disagreement` | Fixture bốn cặp cho kết quả exact `4/5` ở hai đường evaluator. Kappa không xác minh chuyên môn hoặc tính độc lập của người chấm. |
| Contrastive loss | trung bình CE theo hàng và cột của normalized dot product / temperature | Đã rà soát source và mask; kiểm tra đại số softmax/CE hai logits. Chưa chứng minh gradient toàn model, hội tụ hoặc lợi ích downstream. |
| Topic/chunk | mean các chunk rồi L2-normalize; diversity; Hungarian Jaccard | Đã rà soát công thức, mẫu số và guard vector 0; không chạy lại optimizer/coherence internals. Mean các view có overlap không phải pooling token không chệch. |

Nguồn: [audit toán baseline](../outputs/math_optimization_20261019/math_audit/REPORT.md), [bằng chứng exact](../outputs/math_optimization_20261019/math_audit/evidence.json), [audit khoa học](../outputs/math_optimization_20261019/science_audit/SCIENTIFIC_AUDIT.md). Source hashes và snapshot của baseline được giữ riêng; các phản ví dụ baseline không được gọi là lỗi còn tồn tại sau khi sửa.

## Các vấn đề đã sửa

1. **Vòng lặp passage không tiến khi thay tham số.** Ví dụ Title=80 tokens, query=128, width yêu cầu 400, overlap=301: ngân sách BERT 512 trừ 3 special tokens chỉ còn width=301; bước kế tiếp lại bắt đầu ở `301-301=0`. Đã kiểm tra width hiệu dụng trước khi tạo cửa sổ. Title-only/no-offset fallback chỉ cần đủ ngân sách cặp, không bị áp điều kiện overlap không cần thiết.
2. **Ngân sách passage hardcode 512.** Trước đây thay `reranker_max_length` xuống 256/384 có thể khiến text hiển thị dài hơn phần model được phép đọc. Đã truyền cấu hình vào selector và tính pair overhead từ tokenizer. Các bảo đảm này có giả định tokenizer/template tương ứng; không suy thành chứng minh cho mọi tokenizer bất kỳ.
3. **Miền tham số chưa được kiểm tra sớm.** RRF có thể chia 0 với `k=-1`; cấu hình BM25, thread/batch và kích thước cửa sổ sai miền chưa được từ chối rõ ràng. Đã thêm validation trước khi đọc corpus và kiểm tra ID lặp trong từng danh sách RRF. Caller chuẩn vốn tạo ID duy nhất nên kiểm tra lặp là hardening, không phải bằng chứng kết quả hiện tại chứa trùng.
4. **Success bị gọi là Recall.** `mean(best_relevant_rank<=k)` đo Success@k, không đo tỷ lệ toàn bộ relevant documents được lấy về. Ví dụ ba tài liệu ở nhóm A,A,B có Success@1=1 nhưng document Recall@1=`2/3`. Đã thêm `group_success_at_1/5/10`, giữ alias cũ có metadata để tương thích, sửa nhãn báo cáo hiện hành và nhãn generator tương lai. CSV/JSON/figures lịch sử giữ nguyên.
5. **Frozen validation chỉ hỗ trợ một block trainable.** Cache tăng tốc hardcode prefix 11 block. Đã từ chối các giá trị khác số nguyên 1 trước cache/training; model loading và embedding thông thường vẫn dùng được. Muốn tune số block phải triển khai validation tổng quát hoặc bỏ đường cache này trước.
6. **Cache topic/projection thiếu identity.** Metrics trước đây tái dùng theo sự tồn tại của file. Đã gắn fingerprint với model, representation, dictionary/BOW, evaluator và phiên bản; projection gắn ordered IDs, embedding hash và tham số UMAP. Cache lịch sử thiếu fingerprint bị từ chối trước fit/exports, không được tự đóng dấu xác nhận. Muốn đánh giá lại phải giữ bản cũ và dùng thư mục thực nghiệm tách riêng; module hiện chưa có CLI `--output` mới cho toàn Stage 3.
7. **Diagnostic bỏ sót `nonaqueous`.** Đã nhận cả cách viết liền; vẫn chỉ là cờ cần đọc lại, không phải nhãn relevance.

## Tham số vận hành đã đo

Protocol được chốt trước chọn ứng viên: threads `{1,2,4}` × batch `{4,8,16}`, query Q01/Q16/Q19, hai repetitions, warm-up không tính, không cache query/pair. Thứ tự đo được shuffle bằng seed cố định. Các tham số ảnh hưởng retrieval, passage và model giữ nguyên.

| Threads | Batch | Median giây/request | Median tỉ lệ paired so với 4:4 |
|---:|---:|---:|---:|
| **4** | **4** | **4,506** | **1,000** |
| 4 | 8 | 4,672 | 1,060 |
| 4 | 16 | 4,779 | 1,070 |
| 2 | 8 | 5,474 | 1,248 |
| 2 | 4 | 5,415 | 1,262 |
| 2 | 16 | 6,106 | 1,364 |
| 1 | 4 | 8,689 | 1,850 |
| 1 | 8 | 8,561 | 1,902 |
| 1 | 16 | 9,097 | 2,091 |

Tỉ lệ paired là median các tỉ lệ từng query/repetition, không phải thương của hai median ở cột thời gian. Có một lượt 1:4 ghi **839,663 s**; giữ nguyên trong dữ liệu và phép tính, chưa xác định nguyên nhân. Tất cả 54 lượt đo đều giữ Top-5 và score guards. Không ứng viên thay thế nào đạt quy tắc giảm ít nhất 10% và không làm query riêng chậm quá 10%; không dò tiếp trên gate để tìm ứng viên thuận lợi.

Do đó giữ nguyên `configs/search_v2.json`: threads=4, batch=4, candidate_depth=100, rerank_depth=50, RRF k=60, Title weight=2, BM25 k1=1.5/b=.75, passage=320/overlap64/tối đa2, pair max512. **Chỉ thread/batch được so sánh trong lưới này**; các giá trị retrieval còn lại là baseline thiết kế, chưa được chứng minh tốt nhất.

Nguồn: [protocol](../outputs/math_optimization_20261019/PROTOCOL.md), [clarification](../outputs/math_optimization_20261019/PROTOCOL_ADDENDUM.md), [summary đủ chín cấu hình](../outputs/math_optimization_20261019/execution_dev/summary.json), [raw timings và kết quả](../outputs/math_optimization_20261019/execution_dev/raw.json).

## Tối ưu hàm thành phần

Hai giả thuyết được thử riêng trong Git worktree. Top-K dùng partition tìm cutoff trước khi sort, giữ toàn bộ cutoff ties rồi dùng đúng quy tắc score/doc_id cũ. Cutoff được chuyển sang float64 cho khớp `float(scores[i])` của baseline, tránh sai với int64 quanh `2**53`. Evidence filter dùng membership trong hai dictionary thay cho tạo union keys cho từng tài liệu và từng nhóm từ.

| Hàm | Dev: tỉ lệ mới/cũ | Gate riêng | Tập seed cuối | Phép đối chiếu tính đúng dev/gate/final |
|---|---:|---:|---:|---|
| Top-K | 0,07145 | 0,06753 | **0,05300** | 9.600 / 5.760 / 5.760 trường hợp exact |
| Evidence filter | 0,17078 | 0,16184 | **0,15760** | 1.233.260 / 739.988 / 739.988 phép so sánh mask/validation |

Cả hai đạt tiêu chí giảm ít nhất 10%, không sai kết quả ở gate và tập seed cuối; đã tích hợp vào code chính. Trên tập cuối, thời gian kernel giảm khoảng **94,7%** và **84,2%**. Phép so sánh mask không phải số query độc lập hay số nhãn người; không dùng hàng triệu phép so sánh này làm cỡ mẫu thống kê relevance. Trường hợp int64 quanh `2**53` được giữ thành regression test riêng; longdouble trên Windows này không có mantissa rộng hơn float64 nên trường hợp extra-precision chỉ có fixture điều kiện, chưa được thực thi trên nền tảng precision rộng.

Các tỉ lệ kernel không được cộng, nhân hoặc chuyển thành mức tăng tốc toàn bộ search; chưa đo mức tăng tốc end-to-end của bản tích hợp so với code cũ. Các lượt chậm vẫn được giữ nguyên trong raw records. Tập cuối được dùng một lần sau chọn ứng viên và không dùng để đổi thuật toán/tham số.

Bằng chứng: [Top-K gate](../outputs/math_optimization_20261019/topk_gate/summary.json), [Top-K final](../outputs/math_optimization_20261019/topk_final/summary.json), [filter gate](../outputs/math_optimization_20261019/filter_gate.json), [filter final](../outputs/math_optimization_20261019/filter_final.json). Các file tương ứng chứa hoặc liên kết raw timings, seed và hashes evaluator/candidate.

## Hạn chế còn lại và ưu tiên cải thiện

| Ưu tiên | Hạn chế còn lại | Cải thiện có thể kiểm chứng |
|---|---|---|
| 1 | Chưa có relevance labels độc lập; query do tác giả soạn và đã đọc nhiều kết quả | Hai người chấm theo intent/rubric, phân xử; giữ query-family mới chưa dùng để tune làm test. Tối ưu nDCG@5 trên dev có nhãn rồi khóa cấu hình trước test. |
| 1 | Truy vấn nhiều điều kiện: token presence chưa xác nhận quan hệ/phủ định | Phân tích lỗi theo intent; đánh giá bộ lọc/claim extraction trên tập có nhãn riêng. Chưa coi diagnostic hoặc LLM tự chấm là ground truth. |
| 2 | Lexical gate có thể từ chối paraphrase; clipping BM25 có phản ví dụ corpus nhỏ | Tách khái niệm token support khỏi dấu của BM25; thử variant IDF/gate có version, chỉ chọn bằng dev labels. Không đổi ranking production dựa vào toy example. |
| 2 | Mean chunk lặp Title/overlap, tail ngắn có trọng số như chunk dài | Ablation từng yếu tố, so coverage-aware pooling ở cùng cửa sổ/ngân sách. Trọng số token không bảo đảm semantics không chệch vì encoder phi tuyến. |
| 2 | Title dài có thể làm stride chunk còn 1 token | Giữ min stride hoặc giới hạn overlap theo payload trong thí nghiệm mới. Ca cực đoan chưa xuất hiện ở corpus hiện tại (max 13 chunks/document). |
| 2 | LDA chọn top-2 C_v rồi mới nhiều seed; composite minmax nhạy tập ứng viên | Đo nhiều seed cho mọi cấu hình hoặc ghi rõ screened search; kiểm tra độ nhạy trọng số/Pareto. Thêm ứng viên bị trội vẫn có thể đảo thứ tự sau minmax. |
| 2 | Contrastive diagonal positives khác với evaluator đa positive cùng nhóm | So multi-positive objective/hard negatives đã xác minh trong grouped splits; không suy chất lượng truy vấn tự nhiên từ proxy Title→Abstract. |
| 3 | Một máy, ít query/repetition, có timing spikes; một worker | Đo cold/warm, RAM, queue/concurrency và workload lớn hơn theo yêu cầu triển khai. Chưa có p95 quần thể hoặc throughput production. |
| 3 | CI bootstrap cố định model, labels và nhóm đã biết | Bổ sung training/topic seeds và query-family clusters; công bố uncertainty riêng cho annotation, training và sampling. |

Không có dữ liệu để công bố bộ tham số tối ưu toàn cục cho LDA/BERTopic/fine-tuning/relevance. Ưu tiên sửa correctness và đo đúng mục tiêu trước khi chạy thêm grid lớn.

## Bằng chứng chạy và tái lập

Bản code trước sửa được giữ tại [archive manifest](../archive/math_optimization_20261019_review/manifest.json). Các audit chỉ đọc lại dữ liệu đã có, không tạo nhãn người hoặc thay điểm còn thiếu bằng 0. **10 nhóm đối chiếu toán học trên code hiện hành đã qua**, bao gồm 150 fixture nDCG/Precision/MRR exact: [current_math_evidence.json](../outputs/math_optimization_20261019/current_math_evidence.json).

**Toàn bộ 303 tests passed, 0 failures/errors/skips**: [full_tests.xml](../outputs/math_optimization_20261019/full_tests.xml). Kết quả này kiểm tra các điều kiện kỹ thuật đã mã hóa trong tests, không chứng minh relevance hoặc toàn bộ gradient/optimizer toán học của thư viện. Biên bản cuối liên kết hashes source, gate/final, archive và truy vấn thật: [final_verification.json](../outputs/math_optimization_20261019/final_verification.json).

Chạy thật Q19 trên bản tích hợp cho **cùng thứ tự Top-5, sai lệch score tối đa 0** so với snapshot cũ: [integrated_q19.json](../outputs/math_optimization_20261019/integrated_q19.json). Lượt đầu trong process mới ghi 80,23 s, gồm tải model và chi phí cold; đây là correctness smoke, không so với median warm 4,506 s và không dùng để khẳng định tăng tốc.

Lệnh tái lập kiểm chứng hiện hành, dùng tên output mới để không ghi đè bằng chứng:

```powershell
.\.venv\Scripts\python.exe scripts/verify_math_contracts.py --output outputs/math_checks/new_run.json
.\.venv\Scripts\python.exe -m pytest tests -q
```

Cây Arbor toàn cục giữ baseline thời gian end-to-end, vì không cấu hình CPU mới nào vượt tiêu chí. Hai kernel dùng cây admission riêng để không so những tỉ lệ có mẫu số khác nhau: [cây chung](../outputs/math_optimization_20261019/.arbor/tree.json), [Top-K](../outputs/math_optimization_20261019/topk_tree/.arbor/tree.json), [filter](../outputs/math_optimization_20261019/filter_tree/.arbor/tree.json). Đã dùng hết ba chu kỳ đã chốt, không mở rộng lưới để tìm kết quả thuận lợi.

Bản source tích hợp được giữ trên nhánh **`arbor/best`** trong Git repository thử nghiệm `outputs/math_optimization_20261019/material`; commit thực tế nằm trong biên bản cuối. Code tương ứng đã áp dụng vào workspace chính. Để checkout snapshot source:

```powershell
git -C outputs/math_optimization_20261019/material checkout arbor/best
```

Repository này lưu source/evidence, không nhân bản model và corpus. Evaluator nhận `--project-root D:/daln` để đọc dữ liệu canonical; ví dụ các lần chạy trong protocol. Nhánh execution giữ baseline 4:4; nhánh Top-K/filter đã qua gate riêng và được tích hợp bằng Git, không gọi đây là một lần đo tốc độ toàn hệ thống mới.

Audit trail ngắn: ba hướng đã xét, hai kernel được nhận qua gate/final, không có profile CPU mới được nhận. Mỗi kernel dùng một gate và một tập final riêng; không chọn lại ứng viên từ phản hồi final. Giả thuyết Top-K ban đầu có phản ví dụ do coercion int64; đã giữ commit sai và sửa để đáp ứng nguyên giả thuyết exactness trước các phép đo tiếp theo. Tối ưu dictionary được nhận vì giữ đúng định nghĩa token presence, không vì tạo điểm relevance mới. [Trạng thái cây cuối](../outputs/math_optimization_20261019/ARBOR_STATUS.txt) và các nhánh/commit thành phần giữ cả kết quả thuận lợi lẫn kết quả âm.

Ảnh/PDF/LaTeX xuất trước phiên này chưa tái xuất. Chúng có thể còn nhãn Recall cũ; khi dùng cho bản báo cáo mới cần cập nhật nhãn Success tương ứng. Các con số, checkpoint và CI lịch sử không được ghi lại như một thí nghiệm mới.

## Nguồn phương pháp

Bốn skill được dùng cho: đề xuất giả thuyết có thể bác bỏ, đối chiếu đại số exact, phê bình độ mạnh bằng chứng và ghi cây thử nghiệm có gate. Chúng không cung cấp bằng chứng rằng hệ thống này chính xác hơn. Đã kiểm tra bản ghi nguồn và ghi nhận phần mềm phương pháp: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv/DOI](https://doi.org/10.48550/arXiv.2609.00065), bản ghi hiện tại được đọc là v2.

Đối chiếu API/implementation chính thức: [SymPy assumptions](https://docs.sympy.org/latest/guides/assumptions.html), [rank_bm25 source](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py), [SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html), [CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html). Giới hạn/chứng minh của kiểm tra local được nêu riêng ở trên; tài liệu thư viện không thay thử nghiệm trên dữ liệu dự án.
