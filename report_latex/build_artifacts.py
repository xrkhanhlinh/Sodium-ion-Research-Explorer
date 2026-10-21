"""Read-only scientific evidence extraction. No models, APIs or original writes.

Run from any cwd: .venv/Scripts/python.exe report_latex/build_artifacts.py
The narrative is a fixed snapshot; fail closed if its key scope changes.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
AI = 'outputs/retrieval_ai_evaluation_v1/'
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def register(rel):
    p = ROOT / rel
    assert p.is_file(), rel
    assert p.suffix.lower() not in {'.env'}, 'Never read environment credentials'
    SOURCES[rel.replace('\\', '/')] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    return p


def frame(rel):
    return pd.read_csv(register(rel), encoding='utf-8-sig')


def obj(rel):
    return json.loads(register(rel).read_text(encoding='utf-8-sig'))


def esc(value):
    if pd.isna(value):
        return 'Chưa tính'
    tr = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
          '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(tr.get(c, c) for c in str(value)).replace('−', '-').replace('–', '--').replace('—', '--')


def num(x, digits=4):
    return f'{float(x):.{digits}f}'.replace('.', ',') if pd.notna(x) else 'Chưa tính'


def integer(x):
    return f'{int(x):,}'.replace(',', r'\,')


def method(x):
    return r'{\footnotesize\texttt{' + esc(x) + '}}'


def table(name, caption, label, columns, headers, rows, source):
    # Every row is already LaTeX-escaped or explicitly mathematical.
    head = ' & '.join(headers) + r' \\'
    content = '\n'.join(' & '.join(row) + r' \\' for row in rows)
    if name in {'worked_example', 'relevance_all', 'synthetic'}:
        text = (r'\begin{table}[H]\centering\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.16}' '\n'
                + r'\caption{' + caption + r'}\label{' + label + '}' '\n'
                + r'\begin{tabular}{' + columns + '}\n' + r'\toprule ' + head + '\n'
                + r'\midrule' '\n' + content + '\n' + r'\bottomrule\end{tabular}' '\n'
                + r'\source{' + source + '}' '\n' + r'\end{table}' '\n')
        (OUT / 'tables' / (name + '.tex')).write_text(text, encoding='utf-8')
        return
    setup = r'\begingroup\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.16}'
    if name in {'synthetic_split', 'evidence'}:
        setup = r'\begingroup\setstretch{1.0}\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}'
    text = (setup + '\n'
            + r'\begin{longtable}{' + columns + '}\n'
            + r'\caption{' + caption + r'}\label{' + label + r'}\\' '\n'
            + r'\toprule ' + head + '\n' + r'\midrule\endfirsthead' '\n'
            + r'\multicolumn{' + str(len(headers)) + r'}{l}{\textit{Bảng tiếp theo}}\\' '\n'
            + r'\toprule ' + head + '\n' + r'\midrule\endhead' '\n'
            + r'\bottomrule\endfoot' '\n' + content + '\n' + r'\end{longtable}\endgroup' '\n'
            + r'\source{' + source + '}\n')
    (OUT / 'tables' / (name + '.tex')).write_text(text, encoding='utf-8')


def main():
    for d in ['tables', 'figures', 'build', 'qa']:
        (OUT / d).mkdir(exist_ok=True)
    state = obj(AI + 'run_state.json')
    qrels = frame(AI + 'ai_consensus_qrels.csv')
    coverage = frame(AI + 'query_coverage.csv')
    agreement = obj(AI + 'agreement.json')
    completion = obj(AI + 'completion_validation.json')
    status_counts = qrels.label_status.value_counts().to_dict()
    assert len(qrels) == 942 and qrels.relevance_grade.notna().sum() == 116
    assert status_counts == {'PENDING_JUDGES': 824, 'AI_CONSENSUS_UNVERIFIED': 116, 'PENDING_ADJUDICATION': 2}
    assert set(coverage.loc[coverage.included_all_methods, 'query_id']) == {'Q11', 'Q19'}
    assert completion['synthetic_queries_evaluated'] == 197
    assert state['blocking_status'] == 'PENDING_API_HTTP_429_RESOURCE_EXHAUSTED'
    (OUT / 'tables/snapshot_macros.tex').write_text(
        '% Generated from checked snapshot\n\\newcommand{\\ConsensusPairs}{116}\n'
        '\\newcommand{\\CompleteQueries}{2}\n', encoding='utf-8')

    clean = frame('outputs/reports/cleaning_summary.csv')
    assert clean['count'].sum() == 20576
    table('cleaning', 'Phân bố bản ghi theo disposition sau xử lý.', 'tab:cleaning', 'llr',
          ['Nguồn', 'Disposition', 'Bản ghi'],
          [[esc(r.source), esc(r.disposition), integer(r.count)] for r in clean.itertuples()],
          'outputs/reports/cleaning_summary.csv')
    splits = frame('outputs/tables/stage3/split_summary.csv')
    table('splits', 'Cặp Title--Abstract theo split và nguồn.', 'tab:splits', 'llrr',
          ['Split', 'Nguồn', 'Cặp', 'Groups trong nguồn'],
          [[esc(r.split), esc(r.source), integer(r.documents), integer(r.groups)] for r in splits.itertuples()],
          'outputs/tables/stage3/split_summary.csv')
    lda = frame('outputs/tables/stage2/lda_comparison.csv')
    table('lda_grid', 'Khảo sát LDA ở seed 42; stability chỉ được tính cho shortlist.',
          'tab:lda-grid', 'rrrrrr', ['$K$', '$C_v$', 'U\_Mass', 'Diversity', 'Stability', 'Selection'],
          [[str(r.k), num(r.c_v), num(r.u_mass), num(r.topic_diversity), num(r.stability), num(r.selection_score)] for r in lda.itertuples()],
          'outputs/tables/stage2/lda_comparison.csv')
    baseline = frame('outputs/tables/stage3/lda_vs_bertopic.csv')
    chunk = obj('outputs/bertopic_chunked/metrics.json')
    rows = []
    for r in baseline.itertuples():
        rows.append([esc('LDA K=20' if r.topics == 20 else 'BERTopic 128'), str(r.topics), num(r.c_v), num(r.u_mass), num(r.topic_diversity), num(r.outlier_ratio * 100, 2) + r'\%'])
    rows.append(['BERTopic chunked', '84', num(chunk['c_v']), num(chunk['u_mass']), num(chunk['topic_diversity']), num(chunk['outlier_ratio'] * 100, 2) + r'\%'])
    table('topic_comparison', 'Chỉ số topic trên tokens tham chiếu đóng băng; khác độ hạt và policy noise.',
          'tab:topics', 'lrrrrr', ['Mô hình', 'Topics', '$C_v$', 'U\_Mass', 'Diversity', 'Outlier'], rows,
          'outputs/tables/stage3/lda_vs_bertopic.csv; outputs/bertopic_chunked/metrics.json')
    topics = frame('outputs/bertopic_chunked/topic_summary.csv')
    selected = topics[topics.topic_id >= 0].nlargest(5, 'documents')
    table('topic_examples', 'Năm cụm lớn nhất BERTopic chunked và keywords nguyên artifact.',
          'tab:topic-examples', r'rrp{.70\linewidth}', ['Topic', 'Documents', 'Top10 keywords'],
          [[str(r.topic_id), integer(r.documents), esc(r.top10_keywords)] for r in selected.itertuples()],
          'outputs/bertopic_chunked/topic_summary.csv')
    table('judgment_status', 'Trạng thái 942 cặp tại snapshot; chưa có C hợp lệ.', 'tab:label-status',
          r'p{.70\linewidth}r', ['Trạng thái', 'Cặp'],
          [[esc(k), str(v)] for k, v in status_counts.items()], AI + 'ai_consensus_qrels.csv')
    table('agreement', 'Agreement raw A/B trước phân xử, cùng Gemini model.', 'tab:agreement',
          r'p{.68\linewidth}r', ['Đại lượng', 'Giá trị'],
          [['Cặp A/B numeric', str(agreement['paired_numeric_ratings'])], ['Đồng grade', '116'],
           ['Exact agreement', num(agreement['exact_agreement'], 6)], ['Quadratic weighted kappa', num(agreement['quadratic_weighted_kappa'], 6)],
           ['Selected votes qua sửa cấu trúc A / B', '10 / 11'], ['C hợp lệ', '0']], AI + 'agreement.json')

    rel = frame(AI + 'summary.csv')
    all_rel = rel[rel.split == 'all']
    assert len(all_rel) == 14 and (all_rel.queries_evaluated == 2).all()
    table('relevance_all', 'Relevance AI một phần: cùng Q11/Q19, 2/23 queries; chưa có chuyên gia.',
          'tab:relevance', r'p{.39\linewidth}rrr', ['Method', 'P@5', 'nDCG@5', 'MRR@5'],
          [[method(r.method), num(r.Precision_at_5), num(r.nDCG_at_5), num(r.MRR_at_5)] for r in all_rel.itertuples()], AI + 'summary.csv')
    table('relevance_splits', 'Mẫu query đủ pool theo split; chưa phải external test.', 'tab:rel-splits',
          'llrr', ['Split', 'Query đủ pool', 'Đã tính', 'Kế hoạch'],
          [['Development', 'Q19', '1', '5'], ['Remaining authored', 'Q11', '1', '18'], ['All', 'Q11, Q19', '2', '23']], AI + 'query_coverage.csv')

    ranks = frame('outputs/search_upgrade_20261018/evaluation_14_methods/top5_fourteen_methods.csv')
    example = ranks[(ranks.query_id == 'Q19') & (ranks.method == 'RERANK_PRETRAINED')].sort_values('rank')
    example = example.merge(qrels[['query_id', 'doc_id', 'relevance_grade']], on=['query_id', 'doc_id'], validate='one_to_one')
    grades = example.relevance_grade.astype(int).tolist()
    assert grades == [3, 3, 3, 1, 3]
    gains = [2**g - 1 for g in grades]
    dcg_terms = [gain / math.log2(i + 2) for i, gain in enumerate(gains)]
    dcg = sum(dcg_terms)
    ideal = sorted(qrels.loc[qrels.query_id == 'Q19', 'relevance_grade'].astype(int), reverse=True)[:5]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    table('worked_example', 'Top5 Q19 của RERANK\_PRETRAINED và nhãn AI thực tế.', 'tab:worked-example',
          r'rp{.59\linewidth}rr', ['Hạng', 'Title', 'Grade', 'DCG term'],
          [[str(r['rank']), esc(r['title']), str(int(r['relevance_grade'])), num(dcg_terms[i], 6)] for i, r in enumerate(example.to_dict('records'))],
          'outputs/search_upgrade_20261018/evaluation_14_methods/top5_fourteen_methods.csv; ' + AI + 'ai_consensus_qrels.csv')
    calc = (r'\begin{align}' '\n'
            + r'P@5(Q19)&=4/5=0{,}8,\qquad RR@5(Q19)=1,\\' '\n'
            + r'DCG@5(Q19)&=7+\frac7{\log_2 3}+\frac7{\log_2 4}+\frac1{\log_2 5}+\frac7{\log_2 6}\\' '\n'
            + r'&\approx ' + num(dcg, 6).replace(',', '{,}') + r',\\' '\n'
            + r'IDCG@5(Q19)&\approx ' + num(idcg, 6).replace(',', '{,}') + r',\\' '\n'
            + r'nDCG@5(Q19)&\approx ' + num(dcg / idcg, 6).replace(',', '{,}') + r'.' '\n'
            + r'\end{align}' '\n'
            + 'Pool Q19 có ' + str(len(qrels[qrels.query_id == 'Q19'])) + ' cặp và ít nhất năm grades 3, nên ideal Top5 là [3,3,3,3,3]. Các biến được định nghĩa tại Mục~\\ref{sec:metrics}.\n')
    (OUT / 'tables/worked_calculation.tex').write_text(calc, encoding='utf-8')
    per = frame(AI + 'per_query.csv')
    expected = per[(per.query_id == 'Q19') & (per.method == 'RERANK_PRETRAINED')].iloc[0]
    assert abs(dcg / idcg - expected.nDCG_at_5) < 1e-12

    proxy = frame('outputs/research_validation/proxy_test_metrics_ci.csv')
    table('proxy', 'Title--Abstract proxy trên 1.368 held-out candidates; CI paired cluster bootstrap.',
          'tab:proxy', r'p{.29\linewidth}rrp{.25\linewidth}', ['Metric group', 'Pretrained', 'Fine-tuned', 'CI 95\% delta'],
          [[esc(r.metric.replace('group_', '').replace('_at_', '@')), num(r.before, 6), num(r.after, 6), '[' + num(r.ci95_low, 6) + '; ' + num(r.ci95_high, 6) + ']'] for r in proxy.itertuples()],
          'outputs/research_validation/proxy_test_metrics_ci.csv')
    synth = frame(AI + 'synthetic/summary.csv')
    assert len(synth) == 14 and (synth.queries_evaluated == 197).all()
    table('synthetic', 'Exact source-ID recovery trên 197/200 query; không phải relevance.', 'tab:synthetic',
          r'p{.37\linewidth}rrrr', ['Method', 'Hit@1', 'Hit@5', 'Hit@10', 'MRR@10'],
          [[method(r.method), num(r.Hit_at_1), num(r.Hit_at_5), num(r.Hit_at_10), num(r.MRR_at_10)] for r in synth.itertuples()], AI + 'synthetic/summary.csv')
    bysplit = frame(AI + 'synthetic/by_training_split.csv')
    splitcol = 'training_split' if 'training_split' in bysplit else 'split'
    test = bysplit[(bysplit[splitcol] == 'test') & bysplit.method.isin(['DENSE_PRETRAINED', 'DENSE_FINETUNED', 'DENSE_CHUNKED'])]
    table('synthetic_test', 'Chẩn đoán 18 synthetic queries từ source thuộc test split cũ.', 'tab:synthetic-test',
          r'p{.37\linewidth}rrrr', ['Method', 'Hit@1', 'Hit@5', 'Hit@10', 'MRR@10'],
          [[method(r.method), num(r.Hit_at_1), num(r.Hit_at_5), num(r.Hit_at_10), num(r.MRR_at_10)] for r in test.itertuples()], AI + 'synthetic/by_training_split.csv')
    for fname, groupcol, label, cap in [('synthetic/by_source.csv', 'source', 'source', 'Source recovery theo nguồn; 97 Patent và 100 Publication.'),
                                      ('synthetic/by_training_split.csv', splitcol, 'split', 'Source recovery theo split của source document; không phải relevance.')]:
        df = frame(AI + fname)
        ncol = 'queries' if 'queries' in df else 'queries_evaluated'
        table('synthetic_' + label, cap, 'tab:synthetic-' + label, r'p{.34\linewidth}lrrr',
              ['Method', 'Nhóm', '$n$', 'Hit@10', 'MRR@10'],
              [[method(r['method']), esc(r[groupcol]), str(int(r[ncol])), num(r['Hit_at_10']), num(r['MRR_at_10'])] for r in df.to_dict('records')], AI + fname)

    errors = frame(AI + 'errors_by_query.csv')
    # Use exact accepted reasons from qrels rather than inventing error narratives.
    ids = [('Q11', 'publication:WOS:000671795300006'), ('Q18', 'patent:201903393M'), ('Q19', 'patent:202290422V')]
    erows = []
    for q, d in ids:
        r = qrels[(qrels.query_id == q) & (qrels.doc_id == d)].iloc[0]
        erows.append([esc(q), str(int(r.relevance_grade)), esc(r.reason)])
    table('error_examples', 'Ví dụ lỗi từ AI consensus; chưa xác minh chuyên gia.', 'tab:errors',
          r'lrp{.77\linewidth}', ['Query', 'Grade', 'Diễn giải AI'], erows, AI + 'ai_consensus_qrels.csv')
    # Recompute per-axis counts for unique query/document pairs from qrels.
    low = errors.drop_duplicates(['query_id', 'doc_id'])
    assert len(errors) == 86 and len(low) == 39
    axes = ['battery_system', 'material_formula', 'technical_condition', 'technical_relation']
    axisrows = []
    for axis in axes:
        vals = low[axis].tolist()
        axisrows.append([esc(axis), str(vals.count('MISMATCH')), str(vals.count('PARTIAL'))])
    table('error_axes', 'Các trục lỗi AI trên cặp grade dưới 2; trục có thể chồng lấp.', 'tab:error-axes',
          r'p{.57\linewidth}rr', ['Trục', 'Mismatch', 'Partial'], axisrows, AI + 'errors_by_query.csv (khử trùng query/doc_id)')

    query = frame('outputs/tables/stage4/queries.csv')
    dev = set(obj(AI + 'protocol.json')['development_queries'])
    table('queries', 'Truy vấn miền đã lưu; nội dung tiếng Anh giữ nguyên.', 'tab:queries',
          r'lp{.67\linewidth}l', ['ID', 'Query', 'Vai trò'],
          [[esc(r.query_id), esc(r.query), 'Dev' if r.query_id in dev else 'Authored'] for r in query.itertuples() if r.query_id != 'Q24'],
          'outputs/tables/stage4/queries.csv; ' + AI + 'protocol.json')
    table('coverage', 'Coverage nhãn từng query; chỉ query đủ pool vào mean chung.', 'tab:coverage',
          'lrrl', ['Query', 'Pool', 'Resolved', 'Đưa vào metrics'],
          [[esc(r.query_id), str(r.pool_pairs), str(r.resolved), 'Có' if r.included_all_methods else 'Chưa'] for r in coverage.itertuples()], AI + 'query_coverage.csv')

    config_rows = [
        ['Stage 1', 'Language confidence', '0,90'], ['Stage 1', 'Title / Abstract near-duplicate thresholds', '0,88 / 0,80'],
        ['LDA', 'K khảo sát / passes / iterations', '10,15,20,25,30 / 10 / 100'], ['LDA', 'alpha / eta / seed cơ sở', 'asymmetric / symmetric / 42'],
        ['BGE', 'Dimension / train max tokens', '768 / 128'], ['Fine-tuning', 'Epoch / batch / learning rate', r'1 / 8 / $2\times10^{-5}$'],
        ['Fine-tuning', 'Block trainable / temperature', '1 block cuối / 0,05'], ['Chunking', 'Max pair / overlap / pooling', '256 / 64 / normalized mean CLS'],
        ['UMAP', 'Neighbors / components / min-dist', '15 / 5 / 0'], ['HDBSCAN', 'Min cluster / min samples / selection', '30 / 10 / EOM'],
        ['Search v2', 'Candidate / rerank depth / RRF k', '100 / 50 / 60'], ['Search v2', 'Title weight / BM25 k1 / b', '2 / 1,5 / 0,75'],
        ['Reranker', 'Passage / overlap / max passages', '320 / 64 / 2'], ['Reranker', 'Max pair / Title budget / batch', '512 / 80 / 4'],
        ['Gemini', 'Model A/B/C/generator', 'gemini-3.5-flash-lite'], ['Evaluation', 'Bootstrap repeats / seed', '2.000 / 42']]
    table('configuration', 'Cấu hình thực nghiệm và app hiện tại; phạm vi được phân biệt theo thành phần.',
          'tab:configuration', r'p{.19\linewidth}p{.40\linewidth}p{.31\linewidth}', ['Thành phần', 'Tham số', 'Giá trị'],
          [[esc(a), esc(b), c if '$' in c else esc(c)] for a, b, c in config_rows],
          'configs/config.yaml; configs/stage2.yaml; configs/stage3.yaml; configs/search_v2.json; ' + AI + 'protocol.json')

    images = {
        'data_composition.png': 'outputs/figures/data_preparation/figure_01_dataset_composition.png',
        'lda_comparison.png': 'outputs/figures/stage2/lda_01_model_comparison.png',
        'training.png': 'outputs/figures/stage3/bert_01_training.png',
        'chunked_umap.png': 'outputs/bertopic_chunked/figures/umap_2d.png',
        'source_recovery.png': AI + 'synthetic/source_recovery_metrics.png'}
    catalog = json.loads((OUT / 'figure_catalog.json').read_text(encoding='utf-8'))
    captions = json.loads((OUT / 'figure_captions.json').read_text(encoding='utf-8'))
    caption_indices = {}
    groups = {}
    for record in catalog['items']:
        index = caption_indices.get(record['group'], 0)
        record['caption'] = captions[record['group']][index]
        assert '?' not in record['caption'], 'Invalid caption encoding'
        caption_indices[record['group']] = index + 1
        images[record['dest']] = record['source']
        if record['group'] != 'existing':
            groups.setdefault(record['group'], []).append(record)
    (OUT / 'figure_catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding='utf-8')
    for group, records in groups.items():
        fragment = []
        for record in records:
            height = '.83' if record['tall'] else '.43'
            fragment.append(r'\evidencefigure{' + record['dest'] + '}{' + record['caption'] + '}{'
                            + record['label'] + '}{' + record['source'] + '}{' + height + '}')
        fragment.append(r'\FloatBarrier')
        (OUT / 'chapters' / ('figures_' + group + '.tex')).write_text('\n'.join(fragment) + '\n', encoding='utf-8')
    for dest, relpath in images.items():
        shutil.copy2(register(relpath), OUT / 'figures' / dest)
    available = sorted({p.as_posix() for folder in ['outputs/figures', 'outputs/bertopic_chunked/figures']
                        for p in (ROOT / folder).rglob('*.png')})
    available += [p.as_posix() for p in (ROOT / AI).glob('*.png')]
    available += [p.as_posix() for p in (ROOT / AI / 'synthetic').glob('*.png')]
    selected = set(images.values())
    audit = []
    for absolute in available:
        relative = Path(absolute).relative_to(ROOT).as_posix()
        audit.append({'source': relative, 'included': relative in selected,
                      'reason': 'Included with version/scope caption' if relative in selected else
                      ('Duplicate overview of the seven full keyword pages' if 'keyword_contact_sheet' in relative else
                       'Redundant historical UI screenshot; three representative historical screens included')})
    (OUT / 'figure_inventory.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

    evidence = [
        ('Corpus và disposition', 'data/processed/main_corpus.parquet'),
        ('Số liệu xử lý', 'outputs/reports/cleaning_summary.csv'),
        ('LDA grid / lựa chọn', 'outputs/tables/stage2/lda_comparison.csv'),
        ('Proxy trước/sau và CI', 'outputs/research_validation/proxy_test_metrics_ci.csv'),
        ('Coverage chunking', 'outputs/bertopic_chunked/chunking_success.json'),
        ('Topic metrics', 'outputs/bertopic_chunked/metrics.json'),
        ('Chuyển topic/noise và ARI', 'outputs/research_validation/between_versions_agreement.json'),
        ('Rankings 14 methods', 'outputs/search_upgrade_20261018/evaluation_14_methods/top5_fourteen_methods.csv'),
        ('Qrels AI và provenance', AI + 'ai_consensus_qrels.csv'),
        ('Agreement và coverage', AI + 'agreement.json'),
        ('Relevance AI một phần', AI + 'summary.csv'),
        ('Synthetic summary', AI + 'synthetic/summary.csv'),
        ('Tái tính 2.758 runs', AI + 'completion_validation.json'),
        ('Latency warm uncached', 'outputs/search_upgrade_20261018/live_latency_scope.json'),
        ('237 tests đã thực hiện', AI + 'tests.xml'),
        ('Hash inputs AI task', AI + 'input_verification.json'),
        ('Parity baseline lịch sử', AI + 'legacy_parity.json'),
        ('Implementation lịch sử', 'archive/cleanup_20261018/historical_files.zip')]
    table('evidence', 'Bản đồ bằng chứng dùng để kiểm tra nội dung.', 'tab:evidence',
          r'p{.29\linewidth}p{.62\linewidth}', ['Nội dung', 'Artifact'],
          [[esc(desc), r'\artifact{' + path + '}'] for desc, path in evidence], 'Các đường dẫn tương đối từ workspace root')
    for _, path in evidence:
        register(path)
    # Explicitly read all current Python source and configs, excluding caches/env.
    inventory = []
    for folder in ['src', 'scripts', 'configs']:
        for p in sorted((ROOT / folder).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix not in {'.py', '.yaml', '.json'}:
                continue
            text = p.read_text(encoding='utf-8-sig')
            rec = {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'lines': len(text.splitlines())}
            if p.suffix == '.py':
                tree = ast.parse(text)
                rec['definitions'] = [n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
                rec['docstring'] = ast.get_docstring(tree)
            inventory.append(rec)
    for relpath in ['app.py', 'docs/PROJECT_SPEC.md', 'docs/FINAL_RESEARCH_REPORT.md', 'docs/PROGRESS.md',
                    'outputs/reports/stage3/summary.json', 'outputs/reports/stage3/inference_contract.json',
                    'outputs/reports/stage4/inference_contract.json', 'outputs/reports/stage2/retrieval_contract.json',
                    AI + 'REPORT.md', AI + 'scope.json',
                    AI + 'synthetic/confidence_intervals.csv', AI + 'synthetic/metric_scope.json']:
        register(relpath).read_text(encoding='utf-8-sig')
    drift = frame('outputs/research_validation/document_embedding_drift.csv')
    flow = drift.transition.value_counts().to_dict()
    assert flow['inlier_to_outlier'] == 1599 and flow['outlier_to_inlier'] == 1230
    placeholders = frame('outputs/research_validation/stage1_placeholder_occurrences.csv')
    assert placeholders.doc_id.nunique() == 355 and placeholders.occurrences.sum() == 716
    review = frame('outputs/research_validation/topic_review_420_documents.csv')
    assert len(review) == 420 and review.topic_id.nunique() == 84
    assert review.groupby('topic_id').size().eq(5).all()
    assert review.topic_fit_grade.isna().all()
    outlier_review = frame('outputs/research_validation/outlier_review_stratified.csv')
    assert len(outlier_review) == 40
    ci = frame(AI + 'synthetic/confidence_intervals.csv')
    hit = ci[(ci.method == 'DENSE_CHUNKED') & (ci.metric == 'Hit_at_10')].iloc[0]
    mrr = ci[(ci.method == 'DENSE_CHUNKED') & (ci.metric == 'MRR_at_10')].iloc[0]
    assert round(hit.ci95_low, 4) == 0.8832 and round(hit.ci95_high, 4) == 0.9594
    assert round(mrr.ci95_low, 4) == 0.7959 and round(mrr.ci95_high, 4) == 0.8879
    # Read corpus metadata only, no embedding/model load.
    corpus = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet', columns=['doc_id', 'source', 'title', 'abstract'])
    assert len(corpus) == 13702 and corpus.doc_id.is_unique
    assert corpus.source.value_counts().to_dict() == {'Publication': 7769, 'Patent': 5933}
    assert corpus.abstract.fillna('').str.strip().eq('').sum() == 52
    tests = ET.parse(ROOT / (AI + 'tests.xml')).getroot()
    suites = [tests] if tests.tag == 'testsuite' else list(tests.iter('testsuite'))
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    assert totals == {'tests': 237, 'failures': 0, 'errors': 0, 'skipped': 0}
    for relpath, rec in SOURCES.items():
        assert sha(ROOT / relpath) == rec['sha256'], f'Input changed: {relpath}'
    (OUT / 'code_review_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'snapshot_date': '2026-10-19', 'operation': 'Read-only report extraction; no API/training/index writes',
                'sources': SOURCES, 'tables_generated': sorted(p.name for p in (OUT / 'tables').glob('*.tex')),
                'figures_copied': images, 'corpus_rows': len(corpus), 'qrels_status_counts': status_counts,
                'worked_example': {'query': 'Q19', 'method': 'RERANK_PRETRAINED', 'grades': grades, 'DCG': dcg, 'IDCG': idcg,
                                   'P_at_5': 0.8, 'RR_at_5': 1.0, 'nDCG_at_5': dcg / idcg},
                'recorded_test_results_not_rerun': totals, 'source_hashes_unchanged_after_read': True}
    (OUT / 'evidence_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'tables': len(manifest['tables_generated']), 'figures': len(images),
                      'evidence_sources': len(SOURCES), 'python_config_files_read': len(inventory),
                      'snapshot_checks': 'PASS', 'Q19_nDCG': dcg / idcg}, ensure_ascii=False))


if __name__ == '__main__':
    main()
