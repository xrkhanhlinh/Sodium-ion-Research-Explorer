"""Run fixed v2 ablations on all 13,702 cleaned documents; no training.

Five previously AI-reviewed queries are development diagnostics. The remaining
18 domain queries are reserved for future independent judgments, not auto-score
optimization. Saved runs support interruption/resume with exact lineage checks.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.search_v2 import ROOT, METHODS, SearchEngineV2, digest, write_json
from src.artifact_history import verify_snapshot

DEFAULT = ROOT / 'outputs/search_runs' / ('benchmark_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
DEV_QUERIES = ['Q01', 'Q08', 'Q16', 'Q19', 'Q23']


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    queries = pd.read_parquet(ROOT / 'data/stage4/queries.parquet')
    identity = {p: digest(ROOT / p) for p in [
        'configs/search_v2.json', 'src/search_v2.py', 'data/processed/main_corpus.parquet',
        'data/stage4/queries.parquet', 'models/reranker_minilm_l6/model.safetensors',
        'models/bge_pretrained/model.safetensors', 'models/bert_finetuned/model.safetensors',
        'data/stage4/pretrained_embeddings.npy', 'data/stage3/embeddings.npy',
        'outputs/bertopic_chunked/document_embeddings.npy',
    ]}
    marker = output / 'experiment_protocol.json'
    protocol = dict(
        identity=identity, methods=METHODS, candidate_documents=13702, new_training=False,
        query_ids=queries.query_id.tolist(), development_queries=DEV_QUERIES,
        reserved_queries=[x for x in queries.query_id if x not in DEV_QUERIES and x != 'Q24'],
        external_corpus_documents_added=0, query_control='Q24 synthetic OOV; not primary mean',
        selection='One fixed configuration, no relevance-driven parameter search. Pretrained default chosen for avoiding project fine-tuning dependency, not as an effectiveness winner.',
        cached_embeddings='Reuse exact checkpoint-matched saved corpus and query embeddings; reranker performs new real inference.',
        latency='Per-method times in this offline ablation can reuse query and reranker pair caches and include cold loads. Not a live end-to-end latency benchmark.',
        model_source='https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2',
        model_revision=json.loads((ROOT / 'models/reranker_minilm_l6/download_provenance.json').read_text(encoding='utf-8'))['revision'],
        judgment_policy='No new human labels; no official retrieval metric calculated. Missing judgments remain missing, never grade 0.',
    )
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8'))['identity'] != identity:
            raise ValueError('Experiment changed. Use a new output version.')
    else:
        write_json(marker, protocol)
    queries.to_csv(output / 'queries.csv', index=False)
    saved = output / 'query_runs'
    saved.mkdir(exist_ok=True)
    engine = SearchEngineV2(use_query_cache=True)
    started = time.perf_counter()
    try:
        for q in queries.itertuples():
            path = saved / f'{q.query_id}.json'
            if path.exists():
                existing = json.loads(path.read_text(encoding='utf-8'))
                if existing['query'] != q.query or set(existing['responses']) != set(METHODS):
                    raise ValueError('Invalid saved query checkpoint')
                print('Reused', q.query_id, flush=True)
                continue
            responses = {}
            for method in METHODS:
                response = engine.search(q.query, method, top_k=10)
                response['query_id'] = q.query_id
                response['is_edge_case'] = bool(q.is_edge_case)
                responses[method] = response
            write_json(path, dict(query=q.query, responses=responses))
            print('Completed', q.query_id, '10 methods;', round(time.perf_counter() - started, 1), 'seconds elapsed', flush=True)
            write_json(output / 'run_status.json', dict(status='RUNNING', completed_query=q.query_id,
                       quality_status='PENDING_INDEPENDENT_LABELS'))
    finally:
        engine.close()
    aggregate(output, queries)


def aggregate(output, queries):
    rows, runs, timings = [], [], []
    for q in queries.itertuples():
        state = json.loads((output / 'query_runs' / f'{q.query_id}.json').read_text(encoding='utf-8'))
        for method, response in state['responses'].items():
            runs.append(dict(query_id=q.query_id, method=method, results_count=min(5, len(response['results'])),
                             is_edge_case=q.is_edge_case, status=response['details']['status']))
            timings.append(dict(query_id=q.query_id, method=method, seconds=response['seconds'],
                                timing_scope='OFFLINE_INCREMENTAL_CACHED_NOT_LIVE_LATENCY'))
            for r in response['results']:
                rows.append(dict(query_id=q.query_id, query=q.query, method=method,
                                 is_edge_case=q.is_edge_case, **r))
    ranked = pd.DataFrame(rows)
    ranked.to_csv(output / 'rankings_top10.csv', index=False)
    top5 = ranked[ranked['rank'].le(5)]
    top5.to_csv(output / 'top5_comparison.csv', index=False)
    pd.DataFrame(runs).to_csv(output / 'run_manifest.csv', index=False)
    pd.DataFrame(timings).to_csv(output / 'offline_times_NOT_LIVE_LATENCY.csv', index=False)
    old = pd.read_csv(ROOT / 'outputs/research_validation/retrieval/blind_relevance_pool_with_chunked.csv', keep_default_na=False)
    pool = pd.concat([old[['query_id', 'query', 'doc_id', 'title', 'abstract', 'source']],
                      ranked[['query_id', 'query', 'doc_id', 'title', 'abstract', 'source']]], ignore_index=True)
    pool = pool[pool.query_id.ne('Q24')].drop_duplicates(['query_id', 'doc_id']).sort_values(['query_id', 'doc_id'])
    pool['relevance_grade'] = ''
    for field in ['annotator_id', 'verifier_id', 'verification_reference', 'reviewer_notes', 'label_method']:
        pool[field] = ''
    pool['label_status'] = 'UNVERIFIED'
    # Shared pool includes all old and new candidates; never overwrite old forms.
    for annotator, seed in [('A', 20261018), ('B', 20261019)]:
        path = output / f'annotator_{annotator}_NEW.csv'
        if not path.exists():
            pool.sample(frac=1, random_state=seed).to_csv(path, index=False)
    pool.to_csv(output / 'shared_pool_UNVERIFIED.csv', index=False)
    pd.DataFrame([dict(method=m, Precision_at_5=None, nDCG_at_5=None, MRR_at_5=None,
                       verified_queries=0, status='PENDING_INDEPENDENT_LABELS') for m in METHODS]).to_csv(
                           output / 'official_metrics_PENDING.csv', index=False)
    known = pd.read_csv(ROOT / 'outputs/search_quality_pilot_20261018/ai_assessments.csv')
    reviewed = top5.merge(known[['query_id', 'doc_id', 'ai_grade', 'label_status']],
                         on=['query_id', 'doc_id'], how='left', validate='many_to_one')
    reviewed.to_csv(output / 'top5_old_AI_judgment_coverage.csv', index=False)
    coverage = []
    for (qid, method), g in reviewed.groupby(['query_id', 'method']):
        if qid not in DEV_QUERIES:
            continue
        n = int(g.ai_grade.notna().sum())
        positives = int(g.ai_grade.ge(2).sum())
        coverage.append(dict(query_id=qid, method=method, known_ai_pairs=n,
                             unjudged_pairs=5-n, ai_P_at_5_if_complete=positives/5 if n == 5 else None,
                             ai_P_lower_bound=positives/5, ai_P_upper_bound=(positives+5-n)/5,
                             status='INCOMPLETE_OLD_AI_LABELS' if n < 5 else 'OLD_AI_ONLY_UNVERIFIED',
                             warning='Bounds from missing labels, not a confidence interval or expert score'))
    pd.DataFrame(coverage).to_csv(output / 'development_AI_coverage_bounds.csv', index=False)
    # Exact lineage check includes all old scientific inputs. Authorized UI changes
    # are reported explicitly rather than declaring historical source hashes match.
    protected = json.loads((ROOT / 'outputs/research_validation/protected_inputs.json').read_text(encoding='utf-8'))
    preservation = verify_snapshot(protected)
    preservation['corpus_sha256'] = digest(ROOT / 'data/processed/main_corpus.parquet')
    write_json(output / 'preservation_check.json', preservation)
    write_report(output, len(pool), pd.DataFrame(coverage), top5)
    write_json(output / 'run_status.json', dict(status='COMPLETED_FIXED_ABLATIONS', methods=len(METHODS),
               queries=len(queries), corpus_documents=13702, human_labels=0,
               quality_status='PENDING_INDEPENDENT_LABELS', top5_slots=len(top5), shared_judgment_pool=len(pool)))
    print('Completed all ablations; new pool:', len(pool), 'pairs. Independent relevance still pending.', flush=True)


def write_report(output, pool_size, coverage, top5):
    q19 = top5[(top5.query_id == 'Q19') & (top5.method == 'RERANK_PRETRAINED')].sort_values('rank')
    lines = [
        '# Nâng cấp Search Engine trên đúng 13.702 tài liệu', '',
        'Đã chạy đủ 24 query (23 miền + 1 OOV) × 10 cấu hình cố định. Không thêm tài liệu và không fine-tune mới. Dữ liệu/model/kết quả cũ được giữ để đối chứng; các thay đổi giao diện được ghi riêng trong preservation_check.json.', '',
        '## Thay đổi đã triển khai', '',
        '- BM25_CHEMICAL đọc toàn bộ Title/Abstract đã làm sạch, tách field Title/Abstract với trọng số Title 2. Không dùng LDA tokens chứa placeholder. Chỉ tạo lexical view mới, không sửa corpus hoặc thực nghiệm topic cũ.',
        '- Lexical channel giữ case của chuỗi dạng công thức; có chuẩn hóa Unicode và một số cách viết chỉ số, alias NaFSI/NaTFSI. Đây là heuristic tokenization, không phải bộ phân tích hóa học đầy đủ và không giải quyết mọi dạng OCR/formula.',
        '- HYBRID_* dùng RRF k=60 giữa lexical Top100 và dense Top100. Không trộn LDA hoặc topic bonus vào xếp hạng v2. DENSE_PRETRAINED/DENSE_FINETUNED dùng vectors 128 tokens đã lưu; DENSE_CHUNKED dùng mean chunked vectors 256 đã lưu từ checkpoint fine-tuned.',
        '- RERANK_* lấy Top50 hybrid và dùng cross-encoder/ms-marco-MiniLM-L6-v2 (22,7M parameters) suy luận cục bộ. Mỗi tài liệu tối đa 2 passage được chọn theo lexical overlap từ các cửa sổ Abstract; max 320 tokens, overlap 64; Title tối đa 80 tokens. Đọc offsets của toàn Abstract để chọn passage không có nghĩa cross-encoder đọc toàn văn.',
        '- RERANK_PRETRAINED là lựa chọn thiết kế mặc định để vận hành không phụ thuộc bước fine-tune của dự án. Nó dùng BGE pretrained và reranker pretrained, không dùng checkpoint dự án fine-tuned. RERANK_FINETUNED/CHUNKED được chạy đối chứng cùng cấu hình. Chưa chọn winner bằng relevance.',
        '- Hybrid/rerank trả rỗng nếu không có lexical support. Đây là heuristic có thể bỏ sót paraphrase hợp lệ; không phải bộ phát hiện OOD đã hiệu chỉnh. Dense thuần vẫn được giữ để quan sát hành vi đối chứng.', '',
        '- Giao diện mới có trường từ/công thức bắt buộc, phân tách bằng dấu ;. Bộ lọc tùy chọn trong src/search_v2_constraints.py áp dụng trước khi lấy Top-K và đọc toàn lexical view Title/Abstract. Presence không chứng minh các từ cùng mô tả một hệ điện giải hay vật liệu; phủ định/quan hệ vẫn cần đọc nội dung. Các lượt filter riêng là development probes, không trộn vào 10 cấu hình benchmark cố định.', '',
        '## Kết quả có thể kết luận', '',
        'Các phương pháp đã chạy thật trên toàn bộ corpus 13.702 ứng viên, cùng bộ query và các tham số đã ghi trước khi benchmark. Có Top-10/Top-5, đoạn passage, model revision/hash và các checkpoint query để tái lập. Chưa có điểm Precision/nDCG/MRR chính thức vì chưa có nhãn độc lập.', '',
        'Ví dụ Q19, Top-5 mới của RERANK_PRETRAINED:', '',
    ]
    for r in q19.itertuples():
        lines.append(f'{r.rank}. {r.title}')
    lines += ['', 'Đây là minh họa đã dùng để phát triển hệ thống, không phải test độc lập chứng minh tăng relevance. Một Title nghe phù hợp vẫn cần đối chiếu Abstract/nguồn và chấm theo intent.', '',
              '## Đánh giá và chống chọn kết quả thuận lợi', '',
              'Q01/Q08/Q16/Q19/Q23 đã được đọc trong AI pilot nên được coi là development diagnostics. 18 query miền còn lại dành cho đánh giá độc lập tương lai; chúng cũng do tác giả soạn và chưa phải benchmark ngoài dự án. Không dùng các query này để chọn tham số theo điểm tự sinh.', '',
              'development_AI_coverage_bounds.csv chỉ nối lại các nhãn AI cũ nếu cặp đã được đọc. Tài liệu mới không được chấm 0. P@5 chỉ có khi đủ 5 nhãn cũ; nếu thiếu thì để null và ghi bounds về thiếu nhãn, không phải CI. Không xếp hạng winner từ các hàng có coverage khác nhau.', '',
              f'Pool chung mới gồm {pool_size} cặp của 23 queries, hợp nhất các ứng viên cũ và mới. annotator_A_NEW.csv/B_NEW.csv độc lập thứ tự, không có method/rank/score, đều chưa chấm. Không dùng nhãn AI cũ làm nhãn người.', '',
              '## Các việc còn cần làm', '',
              '1. Chốt intent/rubric, hai người chấm độc lập, phân xử, rồi tính relevance với shared pool và paired CI. Đây là việc cần con người; code không tự chứng minh chuyên môn người chấm.',
              '2. Kiểm tra chất lượng riêng theo query/intent/source và đọc các kết quả trái ràng buộc. So sánh reranker có/không, pretrained/fine-tuned, dense/lexical/hybrid cùng labels. Không suy ưu thế từ cosine hoặc sự thay đổi thứ hạng.',
              '3. Nếu sau đánh giá mới cần fine-tune: lấy query–positive–hard-negative có xác minh từ train, split theo family/group và giữ test độc lập. Không fine-tune trên 23 query rồi báo lại như test; không bỏ checkpoint cũ chỉ vì một bảng thuận lợi hơn.',
              '4. Nếu mục tiêu còn gồm topic modeling: sửa preprocessing và chạy topic trong phiên bản mới, ablation Title/context/pooling và nhiều seed. Chưa chạy lại LDA/BERTopic vì phiên này tập trung Search Engine; số topic/coherence cũ không phải bằng chứng search mới tốt.', '',
              '## Tái lập và sử dụng', '',
              'CLI: `.\\.venv\\Scripts\\python.exe -m src.search_v2 "Aqueous sodium-ion batteries with water-in-salt electrolytes"`.', '',
              'Benchmark: `.\\.venv\\Scripts\\python.exe scripts/benchmark_search_v2.py` (resume query checkpoints khi identity không đổi).', '',
              'Model nguồn: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 ; kiến trúc retrieve/rerank: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html . Điểm trên model card là benchmark ngoài miền, không chuyển sang điểm của bài này.', '',
              'offline_times_NOT_LIVE_LATENCY.csv có cache query/pair và cold loads nên không dùng làm latency live. Những lần đo riêng live không cache và kiểm thử được lưu riêng nếu có. Score cross-encoder không phải xác suất relevance.', '']
    if (output / 'pretrained_vs_finetuned_rerank_overlap.csv').exists():
        overlap = pd.read_csv(output / 'pretrained_vs_finetuned_rerank_overlap.csv')
        lines += ['## Đối chứng pretrained/fine-tuned sau rerank', '',
                  f'Top-5 có cùng thứ tự ở {int(overlap.same_order.sum())}/23 truy vấn miền. Đây là đồng thuận thứ hạng, không phải kết quả relevance hoặc kiểm định tương đương. Kết quả proxy fine-tuning cũ vẫn được giữ nguyên; không xóa kết quả bất lợi hoặc thay tên proxy thành Search Engine accuracy.', '']
    if (output / 'live_latency_scope.json').exists():
        timing = json.loads((output / 'live_latency_scope.json').read_text(encoding='utf-8'))
        lines += ['## Đo live riêng', '',
                  f'{timing["requests"]} requests trên 3 development queries, 2 lượt/query: mean {timing["mean_seconds"]:.3f} giây, median {timing["median_seconds"]:.3f}, max {timing["max_seconds"]:.3f}. Có chạy thật BGE query encoding và reranker, không cache query/pair. Loại engine/model loading sau warm-up; không đo queue/concurrency và không gọi max này là p95.', '']
    if (output / 'test_command.json').exists():
        check = json.loads((output / 'test_command.json').read_text(encoding='utf-8'))
        lines += ['## Kiểm thử', '',
                  f'JUnit: {check["tests"]} tests, {check["failures"]} failures, {check["errors"]} errors, {check["skipped"]} skipped. Actual pytest stdout: `{check["observed_stdout"]}`. Có AppTest chạy giao diện mới và bộ lọc. Không chạy integration Qdrant cũ. Tests kỹ thuật không xác minh relevance của tài liệu.', '']
    if (output / 'required_terms_checks.json').exists():
        lines += ['## Lưu ý ca Q08', '',
                  'Reranker chung chưa bảo đảm đúng đồng thời NaFSI + ether + nhiệt độ thấp. Bộ lọc explicit chỉ kiểm tra token/alias trong Title/Abstract: các tài liệu tổng hợp muối có nhắc ether cũng có thể lọt vì quan hệ khác mục tiêu. Chưa kết luận corpus không có đáp án. required_terms_checks.json chỉ xác nhận token presence, không cho Precision@5.', '']
    if (output / 'evaluation_14_methods/scope.json').exists():
        lines += ['## Evaluator cho phiên bản mới', '',
                  'evaluation_14_methods gộp 10 nhánh mới và 4 baseline cũ SE1/SE2/SE3/BM25, đủ 322 tổ hợp query–method trên 23 queries. Ba baseline dense cũ và mới có cùng Top-5 ở 24/24 queries nên chỉ tính một tên mỗi baseline. Có explicit empty runs cho control OOV. Toàn bộ scores chính thức vẫn null khi chưa có labels.', '',
                  'Sau khi có consensus labels thật: `.\\.venv\\Scripts\\python.exe scripts/evaluate_search_v2.py --labels PATH_TO_REAL_CONSENSUS.csv --output outputs/search_v2_verified_NEW`. Wrapper dùng evaluator nghiêm về provenance của nghiên cứu, pool chung 942 cặp; không chấp nhận file nhãn AI của pilot.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    run(args.output)
