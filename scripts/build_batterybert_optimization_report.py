"""Build an evidence-based Vietnamese report and the read-only topic catalog."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import json
import importlib.metadata
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from src.batterybert_topic_tools import sha, write_json

OUT = ROOT / 'outputs/batterybert_optimization_20261023_v1'
LABELS = [
    'Anode MoS2, dichalcogenide lớp và MXene',
    'Cathode phosphate vanadium và NASICON',
    'Cathode Prussian blue và hợp chất sắt',
    'Cấu trúc cell, bộ góp dòng và lắp ráp pin',
    'Điện ly, muối, dung môi và polymer',
    'Anode sulfide/selenide và composite carbon',
    'Anode hợp kim Sn/Sb, phosphorus và chuyển hóa',
    'Hard carbon, carbon xốp và tiền chất sinh khối',
    'Anode oxide Ti/Sn và titanate',
    'Cathode oxide lớp P2/O3 và kim loại chuyển tiếp',
]
FLAGS = {
    'patent:202235650E': ('neighboring_theme', 'Sodium iron phosphate; phosphate chemistry rather than Prussian blue. Topic2 is broader than its central examples.'),
    'patent:202386521F': ('misplaced', 'Homogenization of anode material; synthesis workflow rather than cell assembly.'),
    'publication:WOS:000989869100001': ('misplaced', 'PDI-bolaamphiphile organic anode; not an electrolyte, despite organic-chemistry vocabulary.'),
    'patent:2017766545': ('misplaced', 'Porous carbon coating on cathode particles; related carbon processing but not a hard-carbon anode.'),
    'publication:WOS:000718314900003': ('misplaced', 'NaCrO2/C cathode; should be inspected against layered-oxide cathodes rather than Ti/Sn oxide anodes.'),
    'patent:201957071G': ('ambiguous_polarity', 'Abstract calls the material an anode and only says transition metal oxide; insufficient evidence for a layered cathode label.'),
    'publication:WOS:000378510700016': ('overlap', 'SnSe alloying/conversion anode overlaps the metal-selenide group.'),
    'publication:WOS:001000356600002': ('overlap', 'Bi2Se3 conversion-alloying anode overlaps the metal-selenide group.'),
}


def build():
    final = json.loads((OUT / 'final_test.json').read_text())
    full = json.loads((OUT / 'full_corpus_diagnostics.json').read_text())
    if not final['technical_gate_pass']:
        raise ValueError('Cannot activate a candidate that failed the frozen engineering gates')
    preserved = json.loads((OUT / 'inputs.json').read_text())
    immutable = {p: digest for p, digest in preserved.items() if p.replace('\\', '/').startswith(('data/', 'outputs/'))}
    if len(immutable) != 9:
        raise ValueError('Expected nine frozen corpus/split/vector/order files in the provenance manifest')
    changes = [p for p, digest in immutable.items() if sha(ROOT / p) != digest]
    if changes: raise ValueError(f'Frozen data/vectors changed: {changes}')
    write_json(OUT / 'preservation_check.json', dict(status='PASS', immutable_files=len(immutable), changed=changes,
        mutable_source_code='Authorized changes are recorded separately; old run code is in code_snapshot/'))
    embedding = json.loads((ROOT / 'outputs/final_system/batterybert_finetuned/embedding_success.json').read_text())
    if embedding['sha256'] != sha(ROOT / 'outputs/final_system/batterybert_finetuned/document_embeddings.npy'):
        raise ValueError('Fine-tuned embedding provenance mismatch')
    for filename, expected in embedding['identity']['weights'].items():
        if sha(ROOT / 'outputs/model_comparison_v2/batterybert_finetuned/model' / filename) != expected:
            raise ValueError('Fine-tuned weights no longer match the reused embeddings')
    test_runs = []
    for filename in ('backend_tests.xml', 'web_tests.xml'):
        if (OUT / filename).exists():
            suites = ET.parse(OUT / filename).getroot().iter('testsuite')
            for suite in suites:
                test_runs.append(dict(file=filename, **{k: int(suite.get(k, '0'))
                    for k in ('tests', 'failures', 'errors', 'skipped')}))
    tested = sum(r['tests'] for r in test_runs)
    failed = sum(r['failures'] + r['errors'] for r in test_runs)
    http = json.loads((OUT / 'dashboard_server_smoke.json').read_text()) if (OUT / 'dashboard_server_smoke.json').exists() else {}
    write_json(OUT / 'validation_summary.json', dict(tests=test_runs, distinct_tests=tested,
        failures_and_errors=failed, http_status=http.get('status', 'NOT_RUN'),
        full_corpus_reload_predictions_checked_by_pytest=13702 if tested else None))
    summary = pd.read_csv(OUT / 'topic_distribution.csv')
    keywords = pd.read_csv(OUT / 'topic_keywords.csv')
    representatives = pd.read_csv(OUT / 'representative_documents.csv').fillna('')
    labels = pd.DataFrame(dict(topic_id=range(10), topic_label=LABELS,
        label_status='TENTATIVE_AGENT_INTERPRETATION', independent_expert_validation=False))
    labels.to_csv(OUT / 'topic_labels.csv', index=False)
    reviews = []
    for row in representatives.itertuples():
        status, note = FLAGS.get(row.doc_id, ('consistent_in_inspected_text',
            'Title and inspected abstract excerpt support the broad topic interpretation; not an exhaustive full-paper assessment.'))
        reviews.append(dict(topic_id=row.topic_id, doc_id=row.doc_id, sample_role=row.sample_role,
            title=row.title, status=status, evidence=note, reviewer='Codex agent',
            independent_expert=False, inspected_fields='title + abstract excerpt (central:450 chars; random:800 chars)',
            sample_design='3 nearest-centroid + 3 seeded-random per topic; not a population accuracy estimate'))
    pd.DataFrame(reviews).to_csv(OUT / 'scientific_review.csv', index=False)
    cv = pd.read_csv(OUT / 'common_coherence.csv').set_index('model')
    table = '\n'.join(f'| {t} | {LABELS[t]} | {int(summary.set_index("topic_id").loc[t,"documents"]):,} | '
        + ', '.join(keywords[keywords.topic_id.eq(t)].sort_values('rank').token.head(5)) + ' |' for t in range(10))
    report = f'''# Pipeline, chẩn đoán và tối ưu BatteryBERT — 23/10/2026

## Kết luận theo từng câu hỏi

BatteryBERT fine-tuned + phép chiếu giảm hướng Patent/Publication + KMeans10 là model topic chính mới. Bản này đạt toàn bộ **tiêu chí kỹ thuật định trước**, cho các nhóm vật liệu rõ hơn trong tài liệu được kiểm tra. Đây là lựa chọn triển khai tạm thời theo mục tiêu cân bằng, bớt lệch nguồn và ổn định; chưa phải chiến thắng khoa học có nhãn chuyên gia.

Không có một model tốt nhất cho mọi việc. BatteryBERT HDBSCAN cũ dẫn đầu C_v nhưng hai cụm chứa 92,76% corpus, hầu như tách theo nguồn. BGE có proxy Title→Abstract tốt hơn trong các kết quả test lịch sử đã hoàn tất (pretrained MRR@10=0,8936; fine-tuned=0,9281). BatteryBERT fine-tuned đạt validation proxy=0,8232, so với pretrained=0,4953; chưa có test proxy tương đương của checkpoint này trong báo cáo cũ. Các phép đo khác pooling/chunking/ngân sách nên không diễn giải thành encoder thắng tuyệt đối. Search Top10 vẫn chưa có nhãn người độc lập để kết luận model nào trả tài liệu liên quan tốt nhất.

"10 topic" ở đây là 10 cụm của corpus; mỗi cụm có top10 keywords. "Search Top10" là 10 tài liệu được xếp hạng cho truy vấn, là một bài toán khác.

## Pipeline toàn chương trình

```mermaid
flowchart TD
    A[patent.xlsx + publication.xlsx] --> B[Stage1: đọc và sửa cấu trúc bảng, chuẩn hóa metadata]
    B --> C[Ngôn ngữ, trùng lặp, thu hồi, quy tắc liên quan]
    C --> D[20.576 bản ghi clean master; 13.702 tài liệu corpus chính]
    D --> E[Nhánh lexical: giữ công thức, phrase, lemmatize]
    E --> L[LDA: dictionary + BoW, đối chứng topic]
    D --> F[Chia theo split_group_id: train / validation / test]
    F --> G[BGE và BatteryBERT: pretrained hoặc contrastive fine-tuning]
    G --> H[Embedding đầy đủ Title–Abstract; masked mean, chunk 256/64 cho BatteryBERT]
    H --> I[Topic cũ: UMAP + HDBSCAN + c-TF-IDF]
    H --> J[Topic chính mới: fine-tuned BatteryBERT + train-only projection + KMeans10]
    E --> K[Từ khóa: TF × IDF × exclusivity; phrase thật; vocabulary fit train]
    J --> K
    K --> T[10 topic + keywords + đại diện + xu hướng nguồn/năm]
    D --> X[TF-IDF index]
    H --> Q[Vector index / Qdrant hoặc BGE memmap]
    U[Query + nguồn/năm/từ bắt buộc/topic] --> V[Lọc doc_id trước khi xếp hạng]
    V --> X
    V --> Q
    X --> R[Hybrid chính: Top100 lexical + Top100 BatteryBERT pretrained → RRF k=60]
    Q --> R
    R --> Z[Top10 tài liệu + điểm đúng thang + metadata]
    T --> UI[Streamlit: Topics / Comparison]
    Z --> UI
    Z --> N[Pool gán nhãn 0–2 → P@10, MRR@10, nDCG@10 khi đủ nhãn]
```

Code tương ứng: `src/pipeline.py`, `src/text_processing.py`, `src/quality.py`; `src/stage2.py`; `src/batterybert.py`, `src/stage3_neural.py`, `src/model_comparison.py`; `src/batterybert_optimization.py`; `src/final_search.py`, `src/final_system_vectors.py`, `src/final_dashboard.py`. App không train khi khởi động. Search mặc định vẫn là TF-IDF + BatteryBERT pretrained/RRF; V2 BGE/BM25/MiniLM và V3 fine-tuned BatteryBERT/BM25/MiniLM được giữ để đối chứng. Topic chính mới dùng encoder **fine-tuned**; tên encoder Search được ghi riêng. Topic không cộng điểm vào xếp hạng.

## Chương trình chưa tốt ở đâu và đã sửa gì

1. **Sai mục tiêu phân cụm:** embedding pretrained BERT và HDBSCAN EOM ưu tiên cấu trúc văn phong/nguồn. Mất cân bằng không có nghĩa phải ép mỗi cụm bằng nhau. KMeans10 cho số cụm yêu cầu, còn phép chiếu học từ train làm giảm hướng phân biệt nguồn. Mỗi tài liệu inference đều dùng cùng phép chiếu, không cần biết nguồn. Phép chiếu có thể bỏ một phần khác biệt khoa học thực sự tương quan với nguồn; vì vậy luôn lưu bản gốc và đọc đại diện.
2. **Fine-tune không tối ưu trực tiếp topic:** positive Title–Abstract, batch2, một epoch, chỉ cập nhật block cuối. Gradient accumulation2 không tạo tập negatives lớn hơn bên trong loss. Khoảng 97,66% abstract train bị cắt ở ngân sách128 tokens, trong khi inference dùng đủ abstract với chunk256/64. Fine-tune giúp proxy nhưng không tự giải quyết nguồn; thử nghiệm `finetuned_raw` vẫn có source V≈0,99.
3. **Từ khóa thiếu tính đại diện:** chỉ tăng diversity hoặc phạt tần suất mạnh làm nổi danh sách công thức hiếm, ít đồng xuất hiện. Đã thử 18 biến thể frequency/exclusivity trên hai phân cụm cố định. Chọn TF power1, IDF=log(1+N/df), exclusivity power0,25; vocabulary chỉ fit train, min_df5, max_df0,6, max_features40.000. Mỗi keyword có ≥5 tài liệu train hỗ trợ. Không tự chế keyword kỹ thuật.
4. **Phrase giả và placeholder:** vectorizer mới chỉ tạo ngram liền kề thực trong tài liệu gốc, không nối qua stopword bị xóa hoặc ranh giới tài liệu. Sửa `lda_tokens` để khôi phục phrase nằm trong từ ghép như `all-solid-state`; không chạy lại hay sửa corpus frozen. Công thức, chữ hoa và token kỹ thuật được giữ theo dữ liệu đã có; không khẳng định phục hồi toàn bộ thông tin đã mất ở tiền xử lý lịch sử.
5. **Báo cáo không đồng bộ:** checkpoint fine-tuned đã train thành công (2.730 optimizer updates lịch sử) nhưng báo cáo model comparison vẫn ghi chưa hoàn tất do bước kiểm tra serialization tokenizer. Đã dùng đúng vector fine-tuned đã được tích hợp sau đó, kiểm tra hash checkpoint và thứ tự doc_id. Không lấy pretrained thay cho fine-tuned.
6. **Thiếu thẩm định relevance và topic:** chưa có nhãn người độc lập cho Search, còn lỗi nhóm trong mẫu đọc. Báo cáo và giao diện ghi rõ; coherence và seed agreement không phải accuracy.

## Thiết kế thử nghiệm và điều kiện dừng

Vòng1 chạy 11 cấu hình embedding/geometry, mỗi cấu hình fit KMeans seed42 và seed43; fit trên10.921 tài liệu train, chọn bằng1.361 validation. Vòng2 chạy18 cấu hình từ khóa trên hai phân cụm fine-tuned có lệch nguồn thấp (bao gồm2 lần tái kiểm tra cấu hình từ khóa ban đầu). Tổng29 lượt cấu hình phát triển; không phải29 encoder neural được train. Không thực hiện gradient update mới và không tính lại transformer embeddings trong lượt này.

Tiêu chí chấp nhận khai báo trước vòng1: đúng10 topic, cụm lớn nhất≤30%, top2≤45%, cụm nhỏ nhất≥0,5%, source Cramér V≤0,5, ARI≥0,6, diversity≥0,75, keyword coverage≥0,95, NPMI validation≥0. **Đây là ngưỡng kỹ thuật do thử nghiệm đặt ra, không phải chuẩn khoa học hay accuracy.** Giữ nguyên các ngưỡng ở vòng2. Chọn cấu hình đạt ngưỡng có NPMI validation cao nhất; cấu hình NPMI cao hơn nhưng diversity thấp bị loại.

Sau khi đóng băng lựa chọn mới chạy1.368 test và seed137 chưa dùng trong lựa chọn. Test split đã tồn tại trong nghiên cứu lịch sử; "held-out" ở đây nghĩa là không dùng để fit/chọn cấu hình trong lượt tối ưu này, không tuyên bố bộ dữ liệu hoàn toàn mới đối với mọi nghiên cứu trước.52 tài liệu chỉ có title được inference, không tham gia fit/selection/test. Không refit toàn corpus sau test. Dừng thay đổi model sau khi vượt cổng cuối; các kiểm tra coherence toàn corpus và đọc đại diện sau đó là mô tả, không dùng để sửa model theo test.

| Chẩn đoán trên test | Kết quả | Ngưỡng |
|---|---:|---:|
| Cụm lớn nhất | {final['largest_topic_share']:.2%} | ≤30% |
| Hai cụm lớn nhất | {final['top2_topic_share']:.2%} | ≤45% |
| Cụm nhỏ nhất | {final['smallest_topic_share']:.2%} | ≥0,5% |
| Source Cramér V | {final['source_cramers_v']:.4f} | ≤0,5 |
| ARI với seed137 | {final['stability_ari']:.4f} | ≥0,6 |
| Keyword diversity@10 | {final['keyword_diversity']:.2f} | ≥0,75 |
| Keyword xuất hiện trong test | {final['supported_keyword_fraction']:.0%} | ≥95% |
| Document NPMI | {final['document_npmi']:.4f} | ≥0 |

Trên toàn13.702 tài liệu: top2={full['top2_topic_share']:.2%}, cụm lớn nhất={full['largest_topic_share']:.2%}, source V={full['source_cramers_v']:.4f}. Baseline HDBSCAN cũ: top2=92,76%, source V=0,9986. Đây là so sánh các pipeline có cả khác biệt encoder, clustering và biểu diễn; không quy mọi cải thiện cho fine-tune. KMeans luôn gán cụm nên0% outlier không chứng minh mọi tài liệu phù hợp.

## Model nào cho 10 topic tốt hơn

Coherence tính lại trên **cùng một reference** của toàn corpus, cùng window110, top10, phrase chính xác,10/10 topic đều đủ coverage:

| Model/pipeline | C_v | Diversity@10 | Nhận định |
|---|---:|---:|---|
| LDA K10 lịch sử | {cv.loc['LDA_K10_historical','c_v']:.4f} | 0,66 | Đối chứng lexical, không có encoder |
| BERTopic BGE lịch sử | {cv.loc['BGE_BERTopic_historical','c_v']:.4f} | 0,86 | 22,77% outlier trong kết quả lịch sử |
| BatteryBERT pretrained/HDBSCAN cũ | {cv.loc['BatteryBERT_HDBSCAN_historical','c_v']:.4f} | 0,92 | C_v cao nhất; top2 chiếm92,76%, lệch nguồn rất mạnh |
| BatteryBERT pretrained/KMeans bản web trước | {cv.loc['BatteryBERT_KMeans_web','c_v']:.4f} | 0,81 | Kích thước tốt hơn, vẫn gần như tách theo nguồn |
| BatteryBERT fine-tuned/projection/KMeans mới | {cv.loc['BatteryBERT_optimized','c_v']:.4f} | 0,80 | Chọn làm topic chính theo cổng đa tiêu chí, không thắng C_v |

C_v của bản mới giảm0,6941→0,5601 so với web trước; không che giấu đánh đổi. NPMI test của bản mới0,0883, so với pretrained/raw−0,0485, fine-tuned/raw−0,2010 và BGE/raw−0,1987 trong đối chứng KMeans train-only cùng cách trích từ khóa ban đầu. So sánh bản cuối bao gồm cả thay đổi trọng số keywords; không phải encoder-only ablation.

## Mười topic đang hiển thị

Nhãn là diễn giải tạm thời của agent từ keywords và tài liệu đại diện, chưa có chuyên gia xác nhận. ID không tương ứng với ID của model cũ.

| ID | Diễn giải | Tài liệu | Năm từ khóa đầu |
|---|---|---:|---|
{table}

## Tự kiểm tra nội dung và giới hạn còn lại

Đã xem title cùng đoạn đầu abstract của60 tài liệu:3 gần centroid và3 lấy ngẫu nhiên có seed cho mỗi topic. Lưu đầy đủ abstract gốc và chú thích tại `scientific_review.csv`; đây là đánh giá agent, không phải expert ground truth hay ước lượng precision. Ví dụ đúng chủ đề: MoS2/CNT (`publication:WOS:000627423800016`), NASICON phosphate (`publication:WOS:000600805400001`), Prussian blue (`publication:WOS:000792137800013`), hard carbon từ lõi thân ngô (`publication:WOS:000428070800013`).

Các vấn đề đã ghi nhận: sodium iron phosphate nằm gần nhóm Prussian blue (`patent:202235650E`); organic anode PDI nằm trong nhóm điện ly (`publication:WOS:000989869100001`); cathode NaCrO2/C nằm trong nhóm anode oxide (`publication:WOS:000718314900003`); carbon coating cathode ở nhóm hard carbon (`patent:2017766545`). Các nhóm0/5/6 có vùng chồng lấn chuyển hóa–hợp kim–chalcogenide. Nhóm3 là phần cứng pin thực sự có trong corpus và có nhiều patent; không đổi tên nó thành một họ vật liệu giả để đủ10 chủ đề vật liệu. Một số keywords như solution, dry, connect còn xuất hiện vì nội dung tổng hợp/lắp ráp; xóa toàn bộ bằng blacklist sẽ làm đẹp từ khóa mà không sửa tài liệu bị gán lệch.

## Làm BatteryBERT học tốt hơn ở vòng nghiên cứu sau

Ưu tiên gán nhãn một tập scientific topic độc lập theo nhóm tài liệu, gồm cả trường hợp giao thoa và ngoài phạm vi; tách topic label khỏi relevance Search. Dùng positive cùng chủ đề giữa Patent và Publication để chống học văn phong, đồng thời giữ nhóm phần cứng hợp lệ. Dùng hard negatives khác họ vật liệu, tránh coi hai mô tả cùng họ là negative. Khi có GPU, so sánh batch thực16–32 hoặc cached contrastive loss, không coi accumulation là số negatives. So sánh chunk pooling/hierarchical contrastive phù hợp abstract đầy đủ; không mặc định128 tokens đủ. Fine-tune từng1–2 block hoặc adapter, warmup, clipping, checkpoint chọn theo validation khoa học và kiểm tra drift với checkpoint gốc. Giữ BGE và LDA làm đối chứng, không ép BatteryBERT thắng bằng cách đổi dữ liệu hoặc nhãn test.

Chưa train lại neural trong lượt này vì vấn đề nguồn và trọng số keywords đã được kiểm định bằng các thay đổi nhẹ; mục tiêu proxy cũ không bảo đảm tối ưu topic. Khuyến nghị training trên đây là thiết kế cho nghiên cứu tiếp, chưa phải kết quả đã chạy.

## Chạy và tái lập

```powershell
.venv/Scripts/python.exe run_demo.py
# Mở Topics: BatteryBERT fine-tuned · optimized
# Chạy lại thí nghiệm trong thư mục MỚI, tránh sửa phiên bản đã đóng băng:
.venv/Scripts/python.exe scripts/optimize_batterybert.py --output outputs/batterybert_optimization_REPLAY --phase all
.venv/Scripts/python.exe scripts/evaluate_topic_candidates.py --output outputs/batterybert_optimization_REPLAY
# Kiểm thử bản đang tích hợp:
.venv/Scripts/python.exe -m pytest tests/test_unit.py tests/test_batterybert_optimization.py tests/test_optimized_topics.py tests/test_research_completion.py tests/test_final_system.py tests/test_search_v2_app.py -q
```

Đã chạy **{tested} kiểm thử riêng biệt, {failed} lỗi/failure** trong hai XML backend/web; lượt unit ban đầu được chạy lại trong backend nên không cộng trùng. Bao gồm reload/predict đủ13.702 tài liệu, chỉ fit đúng ID train, kiểm tra support thực của100 keywords, dữ liệu/vectors không đổi, đồng nhất topic giữa Search và Topics, bộ lọc trước ranking, truy vấn V3/hybrid thật và giao diện. HTTP smoke: **{http.get('status', 'NOT_RUN')}**; server kiểm tra được dừng cùng các process con do nó tạo. Cảnh báo Qdrant Local về exact search không phải test failure. Không dùng tests kỹ thuật làm bằng chứng relevance.

`topic_pipeline.joblib` là pipeline học thật; truyền vector **fine-tuned BatteryBERT đúng checkpoint/pooling**, không dùng pretrained vector hoặc vector query Search để thay thế. Predict không cần source, không tự đổi encoder. Bản chạy dùng source snapshot trong `code_snapshot/`; các hardening sau chạy (không sửa mảng float64 đầu vào, hoàn thiện CLI/resume) được kiểm thử riêng. Fingerprint cũ cố ý từ chối replay vào thư mục cũ khi code đổi. Reproduction mới có thể khác phần rất nhỏ do package/hardware; dùng package versions đã lưu.

`validation_comparison.csv`, `refined_validation_comparison.csv`, `inductive_test_comparison.csv`, `common_coherence.csv` giữ số đo thật. `final_test.json` là cổng cuối đóng băng; `preservation_check.json` xác minh dữ liệu/vectors giữ nguyên; `artifact_manifest.json` và `package_versions.json` ghi provenance. `deployment.json` chỉ kích hoạt catalog và bộ lọc topic mới. Qdrant hiện hữu không được build/upsert lại: tìm kiếm lọc bằng doc_id, sau đó hiển thị topic mới từ catalog; payload topic cũ trong storage không được xem là nhãn hiện hành.

## Nguồn phương pháp

- [BERTopic — clustering](https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html): cho phép thay HDBSCAN bằng KMeans; việc có10 cụm không tự bảo đảm chất lượng khoa học.
- [BERTopic — parameter tuning](https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html): tách clustering và representation.
- [WhiteningBERT, EMNLP2021](https://aclanthology.org/2021.findings-emnlp.23/): cơ sở thử thay đổi hình học embedding; không bảo đảm whitening thắng trên corpus này.
- [Is anisotropy really the cause..., EMNLP2022](https://aclanthology.org/2022.findings-emnlp.314/): anisotropy không đủ để kết luận chất lượng ngữ nghĩa.

Đánh giá sử dụng hướng dẫn `scientific-critical-thinking` để tách kết quả kỹ thuật, lựa chọn trên development và kết luận khoa học; không áp dụng checklist lâm sàng cho topic modeling.
'''
    (OUT / 'BAO_CAO_TOI_UU.md').write_text(report, encoding='utf8')
    (ROOT / 'docs/BATTERYBERT_OPTIMIZATION_20261023.md').write_text(report, encoding='utf8')
    code_files = ['src/batterybert_optimization.py', 'src/optimized_topics.py', 'src/text_processing.py',
        'src/final_dashboard.py', 'src/final_search.py', 'scripts/optimize_batterybert.py',
        'scripts/refine_batterybert_topics.py', 'scripts/finalize_batterybert_optimization.py',
        'scripts/evaluate_topic_candidates.py', 'scripts/build_batterybert_optimization_report.py',
        'scripts/smoke_final_dashboard_server.py', 'configs/batterybert_optimization.json']
    write_json(OUT / 'current_code_manifest.json', {p: sha(ROOT / p) for p in code_files})
    write_json(OUT / 'package_versions.json', {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'gensim', 'streamlit']})
    write_json(OUT / 'deployment.json', dict(activate_as_main=True, technical_gate_pass=True,
        model='BatteryBERT fine-tuned + train-only source projection + KMeans10',
        scientific_status='PROVISIONAL_AGENT_REVIEW', independent_validation=False,
        retrieval_encoder_changed=False, qdrant_payloads_rewritten=False))
    plot(summary, pd.read_csv(OUT / 'topic_source_distribution.csv'))
    write_json(OUT / 'artifact_manifest.json', {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.name != 'artifact_manifest.json'})
    print('Report/catalog ready; preserved frozen inputs; checkpoint provenance PASS', flush=True)


def plot(summary, sources):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sources.set_index('topic_id')[['Patent', 'Publication']].plot.bar(stacked=True, ax=axes[0], color=['#de9856', '#357c9e'])
    axes[0].set(title='Optimized BatteryBERT: actual source composition', xlabel='Topic ID', ylabel='Documents')
    axes[1].bar(['Historical\nHDBSCAN', 'Optimized\nKMeans'], [92.7602, 27.4850], color=['#ba5c53', '#357c9e'])
    axes[1].set(title='Share of two largest topics (all 13,702 documents)', ylabel='Corpus share (%)', ylim=(0, 100))
    axes[1].text(.98, .95, 'Different encoder + geometry + clustering\nDescriptive pipeline comparison', transform=axes[1].transAxes, ha='right', va='top', fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'topic_balance.png', dpi=180); plt.close(fig)


if __name__ == '__main__': build()
