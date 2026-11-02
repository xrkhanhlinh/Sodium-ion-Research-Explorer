"""Build a standalone course report from frozen, local research evidence.

Does not import project inference modules, retrain, open Qdrant or use APIs.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEN = HERE / 'generated'
FIG = HERE / 'figures'
FULL = ROOT / 'outputs/batterybert_full_document_20261024_v1'
REV = ROOT / 'outputs/research_revision_20261025_v1'
SILVER = ROOT / 'outputs/ai_silver_labels_703_20261026_v1'
for folder in (GEN, FIG, HERE / 'build', HERE / 'qa'):
    folder.mkdir(parents=True, exist_ok=True)
SOURCES = {}
INVENTORY = []
USED = set()
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 175})
METHOD_LABELS = {
    'TFIDF_COSINE': 'TF-IDF',
    'BATTERYBERT_SEMANTIC': 'BatteryBERT semantic',
    'BATTERYBERT_HYBRID': 'BatteryBERT hybrid',
    'BGE_SEMANTIC': 'BGE semantic',
    'BATTERYBERT_FULLDOC_SEMANTIC': 'Full-document semantic',
    'BATTERYBERT_FULLDOC_HYBRID': 'Full-document hybrid',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    SOURCES[path.relative_to(ROOT).as_posix()] = sha(path)
    return path


def csv(path):
    return pd.read_csv(record(path), encoding='utf-8-sig')


def js(path):
    return json.loads(record(path).read_text(encoding='utf-8-sig'))


def esc(value):
    if pd.isna(value):
        return '--'
    s = str(value).replace('−', '-').replace('–', '--').replace('—', '---')
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in s)


def num(value, digits=4):
    return f'{float(value):.{digits}f}'.replace('.', ',')


def write(name, text):
    (GEN / name).write_text(text + '\n', encoding='utf8')


def table(caption, headers, rows, spec, source=None, long=False, size='small'):
    lines = ['\\begingroup\\' + size + r'\setstretch{1.05}\setlength{\tabcolsep}{4pt}']
    if long:
        lines += [r'\begin{longtable}{' + spec + '}', r'\caption{' + esc(caption) + r'}\\',
                  r'\toprule ' + ' & '.join(map(esc, headers)) + r'\\\midrule\endfirsthead',
                  r'\toprule ' + ' & '.join(map(esc, headers)) + r'\\\midrule\endhead',
                  r'\bottomrule\endfoot']
    else:
        lines += [r'\begin{table}[htbp]\centering\caption{' + esc(caption) + '}',
                  r'\begin{tabularx}{\textwidth}{' + spec + '}',
                  r'\toprule ' + ' & '.join(map(esc, headers)) + r'\\\midrule']
    lines += [' & '.join(map(esc, row)) + r'\\' for row in rows]
    lines += [r'\end{longtable}' if long else r'\bottomrule\end{tabularx}\end{table}']
    if source:
        lines += [r'\source{' + source + '}']
    lines += [r'\endgroup']
    return '\n'.join(lines)


def fig(path, caption, *, height=.58, placement='htbp', generated=False, sources=None):
    path = Path(path)
    if not generated:
        record(path)
        USED.add(path.resolve())
        name = path.relative_to(ROOT / 'outputs').as_posix().replace('/', '__')
        shutil.copy2(path, FIG / name)
        origin = path.relative_to(ROOT).as_posix()
    else:
        name = path.name
        origin = '; '.join(sources or [])
    ident = f'fig:course{len(INVENTORY)+1:03d}'
    INVENTORY.append(dict(file=name, caption=caption, source=origin, label=ident,
                          generated_from_frozen_data=generated, sha256=sha(FIG / name)))
    return '\n'.join([
        r'\begin{figure}[' + placement + r']\centering',
        r'\includegraphics[width=\linewidth,height=' + str(height) + r'\textheight,keepaspectratio]{figures/' + name + '}',
        r'\caption[' + esc(caption) + ']{' + esc(caption) + r'}\label{' + ident + '}',
        r'\source{' + origin + '}', r'\end{figure}',
    ])


def bundle(name, items):
    write(name, '\n'.join(fig(ROOT / p, cap, height=h) for p, cap, h in items) + r'\FloatBarrier')


def build_theory():
    original = record(ROOT / 'report_latex/chapters/02_methods.tex').read_text(encoding='utf8')
    start = original[:original.index('Fine-tuning sử dụng')]
    # The new report documents current training in its own chapter.
    start = start.replace('Search v2', 'nhánh search lịch sử v2')
    start = start.replace('app nhánh search lịch sử v2 không dùng phép này', 'search hiện hành không dùng phép này')
    middle = original[original.index(r'\subsection{Giảm chiều'):original.index(r'\section{Phương pháp truy xuất')]
    metrics = original[original.index(r'\section{Phương pháp đánh giá}'):original.index(r'\subsection{Relevance Top5')]
    metrics = metrics.replace('Phép lấy trung bình qua segmentation và topics do Gensim thực hiện, với window 110 và Top10. Theo biểu diễn cosine ở công thức~\\eqref{eq:cv}, điểm thuộc $[-1,1]$ khi các vector khác zero; các kết quả thực nghiệm ở đây dương. Điểm không phải xác suất. Trường hợp denominator zero/không đủ thống kê cần xử lý, không tự thay bằng một relevance grade.', 'Gensim tổng hợp segmentation/topics với window 110 và Top10. Điểm cosine ở công thức~\\eqref{eq:cv} không phải xác suất; reference và smoothing phải giữ cố định khi so sánh.')
    metrics = metrics[:metrics.index('$\\pi$ là phép hoán vị')] + '$\\pi$ là hoán vị tối ưu. Stability LDA và BERTopic dùng protocol khác nhau; ARI/NMI giữa hai phiên bản biểu diễn không thay stability nhiều seed.\n'
    extra = r'''
\subsection{BatteryBERT và lựa chọn pooling}
BatteryBERT là encoder BERT tiếp tục pretraining trên tài liệu pin bằng MLM; checkpoint cased phân biệt chữ hoa/chữ thường \cite{batterybert}. Dự án không pretrain lại từ đầu. Khác BGE dùng CLS, BatteryBERT sử dụng trung bình hidden states của các token được mask cho phép rồi chuẩn hóa L2:
\begin{equation}
h(x)=\frac{\sum_{i=1}^{L}m_iH_i}{\sum_{i=1}^{L}m_i},\qquad e(x)=h(x)/\|h(x)\|_2.
\end{equation}
$m_i$ chỉ chọn token nội dung theo từng phiên bản. Bản pretrained lịch sử dùng attention-mask mean; bản continuation full-document loại Title/SEP khỏi pooling nội dung Abstract. Title vẫn làm ngữ cảnh attention. Việc checkpoint chuyên miền dùng MLM không đồng nghĩa cosine mặc nhiên là relevance tốt; cần contrastive adaptation và đối chứng.

\section{KMeans và hiệu chỉnh hướng nguồn}
KMeans nhận các vector train đã biến đổi, tối thiểu hóa tổng bình phương khoảng cách:
\begin{equation}
\min_{\mu_1,\ldots,\mu_K,z}\sum_{d\in\mathcal D_{train}}\|\widetilde e_d-\mu_{z_d}\|_2^2,\qquad K=10.
\end{equation}
Sau fit, validation/test được predict bằng centroid đã học. KMeans không có reject/outlier. n\_init là số khởi tạo để chọn nghiệm objective tốt nhất trong một lần fit, khác số epoch train encoder.

Source projection dùng hướng $u=\operatorname{norm}_2(\overline e_P-\overline e_U)$ từ trung bình Patent/Publication trên train. Sau centering theo train, một dạng biểu diễn là:
\begin{equation}
\widetilde e_d=\operatorname{norm}_2\left[(I-uu^\top)(e_d-\overline e_{train})\right].
\end{equation}
Mục đích là giảm một hướng phân biệt nguồn, nhưng có thể loại cả tín hiệu nội dung thật. Vì vậy so cả coherence, nguồn, độ tập trung cụm và stability; không mặc định source independence hoàn toàn là tốt nhất.

\section{Từ khóa source-corrected}
Nhánh KMeans dùng vocabulary unigram/bigram fit train, min\_df5, max\_df0,6, tối đa40.000 features. Điểm từ khóa tỷ lệ với TF tương đối, IDF và exclusivity lũy thừa0,25:
\begin{equation}
w_{kt}=\frac{n_{kt}}{\sum_vn_{kv}}\log\left(1+\frac{N_{train}}{\max(df_t,1)}\right)\left(\frac{df_{kt}+1}{df_t+K}\right)^{0,25}.
\end{equation}
Ở đây $n_{kt}$ là tổng tần suất, $df_{kt}$ là số tài liệu train của topic chứa từ và $df_t$ là document frequency trên toàn train. Recipe giữ frequency power1 và phrase boost1; từ phải có ít nhất5 tài liệu train trong topic hỗ trợ. Đây là weighting khác c-TF-IDF BERTopic; điểm không phải xác suất đúng khoa học. Audit lưu bằng chứng support và ngữ cảnh, đặc biệt với công thức và bigram cần liền kề trong stream.
'''
    # Keep BERTopic immediately after its encoder discussion, before KMeans.
    text = start + extra[:extra.index(r'\section{KMeans')] + middle + extra[extra.index(r'\section{KMeans'):] + metrics
    text = text.replace(r'\chapter{Cơ sở lý thuyết và phương pháp}', r'\chapter{Cơ sở lý thuyết các mô hình và công thức}')
    (HERE / 'chapters/02_theory.tex').write_text(text, encoding='utf8')
    bib = record(ROOT / 'report_latex/references.bib').read_text(encoding='utf8')
    bib = bib.replace('Truy cập ngày 19-10-2026', 'Nguồn tham khảo từ snapshot 19-10-2026')
    bib += r'''
@misc{batterybert,
 author={{BatteryData}}, title={BatteryBERT-cased: Model Card},
 url={https://huggingface.co/batterydata/batterybert-cased},
 note={Đối chiếu ngày 26-10-2026; thông số thực nghiệm theo checkpoint cục bộ}}
@inproceedings{gradcache,
 author={Gao, Luyu and Zhang, Yunyi and Han, Jiawei and Callan, Jamie},
 title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
 booktitle={Proceedings of RepL4NLP}, year={2021},
 url={https://arxiv.org/abs/2101.06983}, doi={10.48550/arXiv.2101.06983}}
'''
    (HERE / 'references.bib').write_text(bib, encoding='utf8')


def main_figures():
    bundle('problem_figures.tex', [('outputs/figures/data_preparation/figure_01_dataset_composition.png', 'Thành phần dữ liệu ở giai đoạn chuẩn bị; đọc cùng bảng corpus đã sàng lọc.', .35)])
    bundle('pipeline_figure.tex', [('outputs/report_pipeline_20261025/figures/01_pipeline.png', 'Pipeline từ dữ liệu đến topic và search; topic và xếp hạng là hai nhánh riêng.', .45)])
    bundle('data_figures.tex', [
        ('outputs/figures/data_preparation/figure_05_cleaning_summary.png', 'Kết quả làm sạch và sàng lọc dữ liệu.', .36),
        ('outputs/figures/stage2/eda_05_length_distributions.png', 'Phân bố độ dài văn bản; số từ khác số WordPiece tokens.', .4)])
    bundle('lda_figures.tex', [('outputs/figures/stage2/lda_01_model_comparison.png', 'Lưới lựa chọn số topic LDA Stage2; kết quả lịch sử K20.', .42)])
    bundle('initial_training_figures.tex', [
        ('outputs/figures/stage3/bert_01_training.png', 'Đường huấn luyện encoder BGE ở Stage3 lịch sử; không phải continuation BatteryBERT.', .42),
        ('outputs/figures/stage3/bert_02_before_after.png', 'Đối chứng biểu diễn trước/sau fine-tuning Stage3 lịch sử.', .4)])
    bundle('full_training_figures.tex', [
        ('outputs/batterybert_full_document_20261024_v1/learning_curve.png', 'Loss và validation của continuation BatteryBERT full-document; nguồn log train thật.', .48),
        ('outputs/report_pipeline_20261025/figures/07_training_effect.png', 'Đối chứng cùng full-document pooling trước/sau continuation; proxy có Title context.', .4)])
    bundle('revision_figures.tex', [
        ('outputs/research_revision_20261025_v1/visualizations/geometry_before_after.png', 'Ảnh hưởng số khởi tạo KMeans; embeddings được giữ nguyên.', .4),
        ('outputs/research_revision_20261025_v1/visualizations/canonical_correctness_effect.png', 'Ảnh hưởng sửa công thức khi so keyword trên cùng canonical reference.', .4),
        ('outputs/research_revision_20261025_v1/visualizations_v2/keyword_failed_hypotheses.png', 'Các recipe lọc keyword bị loại do đánh đổi coherence và coverage.', .48)])
    bundle('rerank_figure.tex', [('outputs/research_revision_20261025_v1/visualizations/search_gate_rejection.png', 'Reranker tăng dev nhưng giảm gate; không được kích hoạt.', .4)])
    bundle('comparison_figures.tex', [
        ('outputs/report_pipeline_20261025/figures/04_six_distributions.png', 'Phân bố10 topic của sáu catalog lịch sử; chưa bao gồm stable topics mới nhất.', .46),
        ('outputs/report_pipeline_20261025/figures/05_historical_comparison.png', 'So sánh các catalog theo protocol lịch sử; không xếp hạng chung với canonical reference mới.', .4)])
    bundle('silver_figures.tex', [
        ('outputs/ai_silver_labels_703_20261026_v1/evaluation/figures_v2/method_comparison.png', 'P@10, nDCG@10 và MRR@10 trên703 Silver Labels, cùng20 queries và sáu methods.', .43),
        ('outputs/ai_silver_labels_703_20261026_v1/evaluation/figures_v2/query_heatmaps.png', 'Điểm từng query; P@10 và nDCG cần được đọc cùng nhau.', .57),
        ('outputs/ai_silver_labels_703_20261026_v1/evaluation/figures_v2/threshold_sensitivity.png', 'Độ nhạy với ngưỡng relevance; nhãn1 và nhãn2 trả lời mục tiêu khác nhau.', .42)])


def main_tables():
    selection = js(FULL / 'selection.json')
    effect = js(FULL / 'paired_training_effect.json')
    # Explicit protocol values are also checked against source JSON recursively.
    all_numbers = []
    def visit(obj):
        if isinstance(obj, dict):
            for v in obj.values(): visit(v)
        elif isinstance(obj, list):
            for v in obj: visit(v)
        elif isinstance(obj, (int, float)): all_numbers.append(obj)
    visit(effect)
    assert any(abs(v - .887839) < .000001 for v in all_numbers)
    train = [('Validation baseline', num(selection['baseline_score'], 6)),
             ('Validation selected', num(selection['best_score'], 6)),
             ('Gain validation', num(selection['best_score'] - selection['baseline_score'], 6)),
             ('Test baseline', '0,887839'), ('Test selected', '0,924704'),
             ('Checkpoint / learning rate', 'Step342 / 5e-6'),
             ('Tài liệu / chunks / token Abstract train', '10.921 / 26.147 / 3.420.215')]
    write('training_results.tex', table('Kết quả continuation trên cùng full-document proxy.', ['Đo lường', 'Giá trị'], train, 'Xr', source='outputs/batterybert_full_document_20261024_v1/selection.json; paired_training_effect.json'))
    metrics = csv(SILVER / 'evaluation/metrics_summary.csv')
    label_summary=js(SILVER/'label_summary.json')
    assert label_summary['processed']==703 and label_summary['queries']==20
    assert label_summary['label_counts']=={'0':143,'1':302,'2':258}
    assert label_summary['review_required']==70 and label_summary['empty_abstracts']==28
    assert len(metrics) == 6 and metrics.queries.eq(20).all()
    rows = [(METHOD_LABELS[r.method], num(r.Precision_at_10), num(r.nDCG_at_10), num(r.MRR_at_10)) for r in metrics.itertuples()]
    write('search_metrics.tex', table('So sánh search trên703 nhãn AI Silver; chưa xác minh bởi con người.', ['Phương pháp', 'P@10', 'nDCG@10', 'MRR@10'], rows, 'Xrrr', source='outputs/ai_silver_labels_703_20261026_v1/evaluation/metrics_summary.csv'))
    comp = csv(ROOT / 'outputs/model_comparison_v2/evaluation/model_comparison.csv')
    rows = [(r.model, num(r.c_v), num(r.u_mass), num(r.diversity), '--' if pd.isna(r.outlier_percent) else num(r.outlier_percent, 2)+'%') for r in comp.query("status == 'completed'").itertuples()]
    write('topic_metrics.tex', table('Đối chứng K10 cùng protocol lịch sử; nhánh fine-tuned BERTopic chưa hoàn thành không có điểm.', ['Catalog', 'C_v', 'U_Mass', 'Diversity', 'Outlier'], rows, 'Xrrrr', source='outputs/model_comparison_v2/evaluation/model_comparison.csv'))
    per = csv(SILVER / 'evaluation/metrics_per_query.csv')
    per = per.query("method == 'BATTERYBERT_FULLDOC_HYBRID'").sort_values('query_id')
    rows = [(r.query_id, num(r.Precision_at_10), num(r.nDCG_at_10), num(r.MRR_at_10), int(r.relevant_in_judged_pool)) for r in per.itertuples()]
    write('per_query.tex', table('Full-document hybrid theo từng truy vấn; cùng pool Silver.', ['Query', 'P@10', 'nDCG@10', 'RR@10', 'Nhãn2 trong pool'], rows, 'lrrrr', long=True, source='outputs/ai_silver_labels_703_20261026_v1/evaluation/metrics_per_query.csv'))
    latency = pd.concat([csv(ROOT/'outputs/research_completion_v1/latency.csv'), csv(FULL/'benchmark_latency.csv')], ignore_index=True)
    rows = [(METHOD_LABELS[r.method], int(r.count), num(r.mean, 3), num(r.median, 3), num(r.max, 3)) for r in latency.itertuples()]
    write('latency_table.tex', table('Độ trễ requests đã lưu, đơn vị giây;19 warm requests/method, khác phiên chạy.', ['Phương pháp', 'n', 'Mean', 'Median', 'Max'], rows, 'Xrrrr', source='outputs/research_completion_v1/latency.csv; outputs/batterybert_full_document_20261024_v1/benchmark_latency.csv'))
    queries = csv(FULL/'queries.csv')
    write('queries.tex', table('Toàn bộ20 query tiếng Anh đã khóa trước chấm.', ['ID', 'Nhóm', 'Truy vấn'], [(r.query_id, r.category, r.query) for r in queries.itertuples()], 'p{.08\\textwidth}p{.19\\textwidth}p{.66\\textwidth}', long=True, source='outputs/batterybert_full_document_20261024_v1/queries.csv'))
    initial = csv(ROOT/'outputs/model_comparison_v2/batterybert_finetuned/training_history.csv')
    success = js(ROOT/'outputs/model_comparison_v2/batterybert_finetuned/training_success.json')
    assert success['optimizer_updates']==2730 and success['rows_seen']==10921
    f,ax=plt.subplots(figsize=(11,3.8))
    ax.plot(initial.step, initial.loss, alpha=.22, linewidth=.5, color='#167d8d',label='Batch loss')
    ax.plot(initial.step, initial.loss.rolling(100,min_periods=1).mean(),color='#183b56',linewidth=1.5,label='Rolling mean 100 batches')
    ax.set(xlabel='Microbatch step',ylabel='Contrastive loss',title='BatteryBERT initial fine-tuning · 128 tokens · real saved training log')
    ax.legend();f.tight_layout();f.savefig(FIG/'battery128_training.png',bbox_inches='tight');plt.close(f)
    write('battery128_training.tex',fig(FIG/'battery128_training.png','Loss BatteryBERT128 từ training_history.csv; accumulation không mở rộng tập negatives.',generated=True,sources=['outputs/model_comparison_v2/batterybert_finetuned/training_history.csv'],height=.35))


def search_examples():
    ranks = csv(SILVER/'evaluation/rankings_with_silver_labels.csv')
    q = ranks.query("query_id == 'Q01' and method == 'BATTERYBERT_FULLDOC_HYBRID'").sort_values('rank')
    assert q['rank'].tolist() == list(range(1, 11))
    rows = [(r.rank, r.title, num(r.score, 5), int(r.ai_label)) for r in q.itertuples()]
    write('q01_results.tex', table('Top10 thực tế của full-document hybrid cho Q01 và nhãn Silver ghép sau retrieval.', ['Hạng', 'Tiêu đề', 'RRF', 'Nhãn'], rows, 'p{.08\\textwidth}p{.64\\textwidth}p{.1\\textwidth}p{.08\\textwidth}', long=True, source='outputs/ai_silver_labels_703_20261026_v1/evaluation/rankings_with_silver_labels.csv'))
    labels = csv(SILVER/'silver_labels_703.csv')
    grades = q.ai_label.to_numpy(dtype=int)
    pool = labels.query("query_id == 'Q01'").ai_label.to_numpy(dtype=int)
    discounts = np.log2(np.arange(1, 11)+1)
    gains = (2.**grades-1)/discounts
    ideal = np.sort(pool)[::-1][:10]
    idcg = float(((2.**ideal-1)/discounts).sum())
    precision = float((grades == 2).mean())
    rr = 1/(int(np.flatnonzero(grades == 2)[0])+1)
    ndcg = float(gains.sum()/idcg)
    assert abs(precision-.9)<1e-9 and abs(rr-.5)<1e-9
    rows = [(i+1, int(g), int(2**g-1), num(discounts[i], 4), num(gains[i], 5)) for i,g in enumerate(grades)]
    tex = table('Tính DCG từ output Q01, full-document hybrid.', ['Rank', 'Grade', 'Gain', 'log2(rank+1)', 'DCG thành phần'], rows, 'rrrrr', long=True)
    tex += '\n' + r'\noindent $DCG=' + num(gains.sum(),6) + r'$, $IDCG=' + num(idcg,6) + r'$, $nDCG=' + num(ndcg,6) + r'$; $P@10=0,9$, $RR@10=0,5$.'
    write('worked_metrics.tex', tex)
    # A faithful result figure, deliberately labelled as a reconstruction.
    figobj, ax = plt.subplots(figsize=(8.8, 10.2))
    ax.axis('off')
    cellrows = [[str(r.rank), '\n'.join(textwrap.wrap(r.title, 61)), f'{r.score:.5f}', str(int(r.ai_label))] for r in q.itertuples()]
    t = ax.table(cellText=cellrows, colLabels=['Rank', 'Title', 'RRF', 'Silver'], colWidths=[.06,.76,.1,.08], loc='center', cellLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(11)
    for (i,j), cell in t.get_celld().items():
        cell.set_height(.078 if i else .055)
        cell.set_edgecolor('#d5dde5')
        cell.set_facecolor('#183b56' if i==0 else ('#f0f5f8' if i%2 else 'white'))
        if i==0: cell.get_text().set_color('white')
    ax.set_title('Q01 · Full-document hybrid · frozen Top 10\nReconstructed from saved output · AI Silver labels', fontsize=13, pad=8)
    figobj.tight_layout(); figobj.savefig(FIG/'q01_search_results.png', bbox_inches='tight'); plt.close(figobj)
    write('search_result_figure.tex', fig(FIG/'q01_search_results.png', 'Hình Top10 dựng lại từ output đã lưu của Q01; không phải ảnh chụp phiên giao diện mới.', generated=True, sources=['outputs/ai_silver_labels_703_20261026_v1/evaluation/rankings_with_silver_labels.csv'], height=.73))
    # Compact architecture diagram, exact current method.
    f, ax = plt.subplots(figsize=(12, 4.7)); ax.axis('off')
    boxes = [(.02,.4,.21,.26,'Query + bộ lọc\nsource / year / terms / topic'),
             (.29,.66,.25,.22,'TF-IDF cosine\nTop 100 lexical'),
             (.29,.12,.25,.22,'Full-document BatteryBERT\nQdrant · Top 100 dense'),
             (.62,.4,.15,.26,'RRF\nk = 60'),
             (.83,.4,.15,.26,'Top 10\nmetadata + excerpt')]
    from matplotlib.patches import FancyBboxPatch
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor='#eaf3f8',edgecolor='#167d8d',linewidth=1.4))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for a,b in [((.23,.54),(.29,.77)),((.23,.5),(.29,.23)),((.54,.77),(.62,.57)),((.54,.23),(.62,.48)),((.77,.53),(.83,.53))]:
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#183b56',lw=1.6))
    ax.text(.5,.96,'Search hiện hành: topic lọc doc_id trước retrieval; không cộng vào score',ha='center',fontsize=13,weight='bold')
    ax.text(.5,.02,'Encoder offline → 13.702 × 768 vectors; query dùng cùng checkpoint/pooling',ha='center',fontsize=10)
    f.tight_layout();f.savefig(FIG/'search_pipeline.png',bbox_inches='tight');plt.close(f)
    record(ROOT/'src/full_document_retrieval.py');record(ROOT/'src/final_search.py')
    write('search_pipeline.tex', fig(FIG/'search_pipeline.png', 'Chi tiết luồng search full-document hybrid theo mã nguồn hiện hành.', generated=True, sources=['src/full_document_retrieval.py; src/final_search.py'], height=.36))


def topic_catalogs():
    old = csv(ROOT/'docs/TOP10_TOPICS_ALL_MODELS_20261024.csv')
    paths = [FULL/'topics', ROOT/'outputs/batterybert_optimization_20261023_v1', ROOT/'outputs/research_completion_v1/topics', ROOT/'outputs/model_comparison_v2/lda_k10', ROOT/'outputs/model_comparison_v2/bertopic_bge_k10', ROOT/'outputs/model_comparison_v2/bertopic_batterybert_k10/pretrained']
    specs = []
    for (name, group), folder in zip(old.groupby('model', sort=False), paths):
        assigned = csv(folder/'topic_assignments.csv')
        keys = csv(folder/'topic_keywords.csv')
        assert len(assigned)==13702 and assigned.doc_id.is_unique
        for r in group.itertuples():
            assert int((assigned.topic_id==r.topic_id).sum()) == r.documents
            actual = keys.query('topic_id == @r.topic_id').sort_values('rank').head(10).token.tolist()
            assert actual == r.top10_keywords.split('; '), (name,r.topic_id)
        specs.append((name,group.copy(),keys,assigned,folder))
    folder = REV/'final_topics'
    assigned = csv(folder/'topic_assignments.csv'); keys=csv(folder/'topic_keywords.csv')
    labels = csv(REV/'review/topic_labels.csv').set_index('topic_id')
    rows = [dict(model='BatteryBERT full-document · stable topics', topic_id=i, topic_label=labels.loc[i,'topic_label'], documents=int((assigned.topic_id==i).sum()), top10_keywords='; '.join(keys.query('topic_id == @i').sort_values('rank').head(10).token)) for i in range(10)]
    group=pd.DataFrame(rows); specs.append((rows[0]['model'],group,keys,assigned,folder))
    text=[];allrows=[]
    for index,(name,group,keys,assigned,folder) in enumerate(specs):
        assert len(group)==10 and set(group.topic_id)==set(range(10))
        noise=int((assigned.topic_id==-1).sum())
        allrows.extend(group.to_dict('records'))
        text += [r'\clearpage\section{' + esc(name) + '}',
                 f'Catalog có10 topic; {13702-noise:,} tài liệu được gán topic và {noise:,} outliers ngoài catalog.'.replace(',', '.'),
                 table('Đủ10 topic và10 keywords/topic: '+name, ['ID / Số bài', 'Tên diễn giải và Top10 keywords'],
                       [(f'{r.topic_id} / {r.documents:,}'.replace(',','.'), r.topic_label+'\n'+r.top10_keywords.replace('; ', ', ')) for r in group.itertuples()],
                       'p{.12\\textwidth}p{.81\\textwidth}', long=True, source=folder.relative_to(ROOT).as_posix()+'/topic_keywords.csv; topic_assignments.csv', size='small')]
        f,ax=plt.subplots(figsize=(8,3.1))
        counts=group.sort_values('topic_id').documents.to_numpy()
        ax.bar(np.arange(10),counts,color='#167d8d')
        for i,v in enumerate(counts):ax.text(i,v+max(counts)*.015,f'{v:,}',ha='center',fontsize=11)
        ax.set(xticks=np.arange(10),xlabel='Topic ID (local to this catalog)',ylabel='Documents',title=f'{name}\nN=13,702 · outliers={noise:,}')
        ax.set_ylim(0,max(counts)*1.2);f.tight_layout();path=FIG/f'catalog_{index+1:02d}_distribution.png';f.savefig(path,bbox_inches='tight');plt.close(f)
        text += [fig(path, 'Phân bố đầy đủ10 topic: '+name, generated=True, sources=[folder.relative_to(ROOT).as_posix()+'/topic_assignments.csv'], height=.27), r'\FloatBarrier']
        for part in range(2):
            f,axes=plt.subplots(3,2,figsize=(9,10.4));axes=axes.ravel()
            for j in range(5):
                topic=part*5+j;ax=axes[j]
                k=keys.query('topic_id == @topic').sort_values('rank').head(10).iloc[::-1]
                assert len(k)==10
                ax.barh(k.token,k.weight,color='#167d8d');ax.tick_params(axis='y',labelsize=12)
                title=group.set_index('topic_id').loc[topic,'topic_label']
                documents=int(group.set_index('topic_id').loc[topic,'documents'])
                ax.set_title(f'Topic {topic} · n = {documents:,}',fontsize=12,loc='left')
                ax.set_xlabel('Keyword weight',fontsize=10);ax.tick_params(axis='x',labelsize=9)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[-1].axis('off')
            f.suptitle(f'{name}\nTopics {part*5}–{part*5+4} · all 10 keywords per topic',fontsize=13)
            f.tight_layout(rect=(0,0,1,.94));path=FIG/f'catalog_{index+1:02d}_keywords_{part+1}.png';f.savefig(path,bbox_inches='tight');plt.close(f)
            text += [r'\clearpage',fig(path, f'Đủ10 từ khóa và trọng số cho topic{part*5}--{part*5+4}: '+name, generated=True, sources=[folder.relative_to(ROOT).as_posix()+'/topic_keywords.csv'],height=.74,placement='p'),r'\clearpage']
    write('all_topics.tex','\n'.join(text))
    out=pd.DataFrame(allrows);out.to_csv(HERE/'topic_catalog_70.csv',index=False,encoding='utf-8-sig')
    assert len(out)==70 and out.top10_keywords.str.split('; ').str.len().eq(10).all()


CAPTIONS = {
    'figure_02_missing_values': 'Mức thiếu trường dữ liệu trong giai đoạn chuẩn bị.',
    'figure_03_publication_trends': 'Phân bố tài liệu theo năm và nguồn khi chuẩn bị dữ liệu.',
    'figure_04_document_lengths': 'Phân bố độ dài Title và Abstract ban đầu.',
    'figure_06_domain_relevance': 'Kết quả heuristic sàng lọc phạm vi sodium-ion.',
    'eda_01_publication_trends': 'Xu hướng tài liệu theo năm của corpus đã chuẩn hóa.',
    'eda_02_top_terms': 'Các term phổ biến trong corpus.',
    'eda_03_ngrams': 'Các n-gram phổ biến; cần đọc cùng quy tắc ranh giới token.',
    'eda_04_author_keywords': 'Author keywords xuất hiện trong dữ liệu có sẵn.',
    'eda_06_domain_term_trends': 'Xu hướng các term miền theo năm.',
    'lda_02_topic_distribution': 'Phân bố topic LDA Stage2 K20 lịch sử.',
    'lda_03_topic_source_heatmap': 'Topic LDA theo nguồn Patent/Publication, Stage2 lịch sử.',
    'lda_04_topic_year_heatmap': 'Topic LDA theo năm, Stage2 lịch sử.',
    'lda_05_largest_topic_trends': 'Xu hướng các topic LDA lớn, Stage2 lịch sử.',
    'lda_06_keyword_heatmap': 'Từ khóa các topic LDA, Stage2 K20 lịch sử.',
    'bertopic_01_distribution': 'Phân bố BERTopic Stage3 lịch sử, trước catalog K10.',
    'bertopic_02_keywords': 'Từ khóa BERTopic Stage3 lịch sử.',
    'bertopic_03_similarity_heatmap': 'Độ tương tự giữa các topic BERTopic Stage3 lịch sử.',
    'bertopic_04_umap_2d': 'UMAP2D BERTopic Stage3; chỉ là trực quan hóa.',
    'comparison_01_topics': 'Đối chiếu topic LDA/BERTopic theo protocol Stage3 lịch sử.',
    'latency_components': 'Các thành phần độ trễ SE1/SE2/SE3, Stage4 Top5 lịch sử.',
    'latency_distribution': 'Phân bố độ trễ SE1/SE2/SE3, Stage4 Top5 lịch sử.',
    'result_composition': 'Thành phần nguồn trong kết quả, Stage4 Top5 lịch sử.',
    'se2_reranking_changes': 'Thay đổi thứ hạng do topic bonus SE2 lịch sử.',
    'top5_overlap_heatmap': 'Độ giao Top5 giữa các search engine Stage4 lịch sử.',
    'bertopic_dashboard': 'Ảnh dashboard BERTopic đã lưu ở Stage5 lịch sử.',
    'lda_dashboard': 'Ảnh dashboard LDA đã lưu ở Stage5 lịch sử.',
    'comparison': 'Ảnh giao diện so sánh mô hình đã lưu ở Stage5 lịch sử.',
    'evaluation': 'Ảnh giao diện đánh giá ở Stage5 lịch sử; không phải kết quả703 Silver Labels.',
    'initial_search': 'Ảnh giao diện và kết quả search đã lưu ở Stage5 lịch sử.',
    'initial_search_form': 'Form search đã lưu ở Stage5 lịch sử.',
    'restart_search': 'Kết quả kiểm tra search sau khởi động lại ở Stage5 lịch sử.',
    'restart_search_form': 'Form search sau khởi động lại ở Stage5 lịch sử.',
    'topic_representatives': 'Ảnh giao diện các tài liệu đại diện topic, Stage5 lịch sử.',
    'umap_2d': 'UMAP2D BERTopic chunked lịch sử; clustering dùng không gian khác.',
    'topic_distribution': 'Phân bố84 topic BERTopic chunked lịch sử, trước bước gộp K10.',
    'topic_balance': 'Độ tập trung topic trước/sau optimization BatteryBERT ngày23/10.',
    'bertopic_final_10_topics': 'Kết quả10 topic tại phiên topic audit ngày22/10.',
    'experiment_comparison': 'Các cấu hình clustering được so tại phiên topic audit ngày22/10.',
    'topic_keywords_10_topics': 'Đủ10 topic KMeans BatteryBERT pretrained, ảnh output gốc.',
    'batterybert_10_topics': 'Đủ10 topic BERTopic--BatteryBERT pretrained, ảnh output gốc.',
    'bge_10_topics': 'Đủ10 topic BERTopic--BGE, ảnh output gốc.',
    'lda_10_topics': 'Đủ10 topic LDA K10, ảnh output gốc.',
    'three_models_10_topics': 'So sánh ba catalog K10, ảnh output gốc.',
    'revised_top10_keywords': 'Một số topic stable với Top10 keywords, ảnh revision gốc.',
    'revised_topic_sources': 'Phân bố nguồn của catalog stable topics mới nhất.',
    'silver_pilot20': 'Pilot20 cặp AI lịch sử; đã được mở rộng thành703 nhãn ở phần chính.',
    'source_recovery_metrics': 'Source-ID recovery synthetic lịch sử; khác đánh giá relevance Top10.',
    'judgment_coverage': 'Coverage AI consensus Top5 lịch sử, trước bộ Silver703.',
    'ai_relevance_metrics_remaining': 'Điểm trên phần AI consensus đủ pool Top5 lịch sử; không dùng thay Silver703.',
    '02_data': 'Quy mô dữ liệu và split trong báo cáo pipeline25/10.',
    '03_lda_selection': 'Lựa chọn K của LDA Stage2, hình tổng hợp25/10.',
    '06_learning_curve': 'Đường train BatteryBERT full-document, hình tổng hợp25/10.',
    '08_topic_sources': 'Phân bố nguồn của catalog full-document24/10 trước stable revision.',
    '09_keywords': 'Từ khóa catalog full-document24/10 trước stable revision.',
    '10_topic_tradeoffs': 'Đánh đổi giữa các chỉ số topic, hình tổng hợp25/10.',
    '11_latency': 'Độ trễ search theo benchmark đã lưu, hình tổng hợp25/10.',
    '12_validation_tradeoff': 'Các ứng viên topic validation, hình tổng hợp25/10.',
}


def appendix():
    # Scope: every research-result raster in these output roots, never PDF QA pages.
    roots=[ROOT/'outputs/figures',ROOT/'outputs/bertopic_chunked/figures',
           ROOT/'outputs/final_system/topic_figures',ROOT/'outputs/topic_results/figures',
           ROOT/'outputs/report_pipeline_20261025/figures',REV/'visualizations',REV/'visualizations_v2',
           ROOT/'outputs/retrieval_ai_evaluation_v1',ROOT/'outputs/batterybert_topic_audit_20261022_v1',
           ROOT/'outputs/batterybert_optimization_20261023_v1',ROOT/'outputs/research_completion_v1/topics',
           FULL,SILVER/'evaluation/figures',SILVER/'evaluation/figures_v2',ROOT/'outputs/reports/stage5']
    candidates=sorted({p for folder in roots for p in folder.glob('*.png')} | set((ROOT/'outputs/figures').rglob('*.png')))
    seen={sha(p) for p in USED}; omitted=[]; text=[];count=0
    for path in candidates:
        if path.resolve() in USED: continue
        reason=None
        if path.stem.startswith('eq_'): reason='Công thức được trình bày lại bằng LaTeX; không là hình kết quả.'
        elif path.name=='keyword_contact_sheet.png': reason='Contact sheet lặp các ảnh keywords gốc đã nhúng đủ.'
        elif path.parent == SILVER/'evaluation/figures':reason='Thay bằng cùng hình figures_v2 với caption/bố cục đã sửa.'
        elif path.parent==REV/'visualizations' and (REV/'visualizations_v2'/path.name).exists():reason='Thay bằng bản visualizations_v2.'
        elif sha(path) in seen: reason='Trùng SHA-256 với hình kết quả đã nhúng.'
        if reason:
            omitted.append(dict(path=path.relative_to(ROOT).as_posix(),reason=reason));continue
        seen.add(sha(path))
        if count%2==0:text.append(r'\clearpage')
        cap=CAPTIONS.get(path.stem)
        if path.stem.startswith('top10_keywords_'):
            cap='BERTopic chunked lịch sử: Top10 keywords của các topic, trang '+path.stem[-2:]+'. Cả7 trang được giữ đủ.'
        if cap is None:
            raise ValueError('Write a meaningful caption for '+str(path))
        text.append(fig(path,cap,height=.34,placement='H'));count+=1
    write('appendix_figures.tex','\n'.join(text)+r'\clearpage')
    (HERE/'figure_inventory.json').write_text(json.dumps(dict(scope=[p.relative_to(ROOT).as_posix() for p in roots],embedded=INVENTORY,excluded=omitted,qa_images_excluded=True),ensure_ascii=False,indent=2),encoding='utf8')


def evidence():
    for rel in ['configs/batterybert_full_document.yaml','configs/model_comparison.yaml','src/batterybert_full_document.py','src/full_document_inference.py','src/revised_topics.py','src/final_system_evaluation.py','scripts/evaluate_silver_703_20261026.py','scripts/refine_batterybert_topics.py','outputs/research_revision_20261025_v1/geometry/scripts/finalize_revision_topics.py','outputs/research_revision_20261025_v1/deployment.json','outputs/research_revision_20261025_v1/final_topics/final_metrics.json','outputs/ai_silver_labels_703_20261026_v1/label_summary.json','outputs/ai_silver_labels_703_20261026_v1/evaluation/confidence_intervals.csv','outputs/ai_silver_labels_703_20261026_v1/evaluation/paired_deltas.csv','outputs/ai_silver_labels_703_20261026_v1/evaluation/sensitivity_summary.csv','outputs/batterybert_full_document_20261024_v1/training_coverage.json','outputs/ai_silver_labels_703_20261026_v1/evaluation/validation.json','data/processed/main_corpus.parquet','data/stage3/split_assignments.parquet']:
        record(ROOT/rel)
    rows=[('Corpus và split','data/processed/main_corpus.parquet; data/stage3/split_assignments.parquet'),
          ('Train và checkpoint','outputs/batterybert_full_document_20261024_v1/training_protocol.json; selection.json'),
          ('Topic K10 lịch sử','docs/TOP10_TOPICS_ALL_MODELS_20261024.csv'),
          ('Catalog stable','outputs/research_revision_20261025_v1/final_topics; review/topic_labels.csv; deployment.json'),
          ('Rank và nhãn','outputs/ai_silver_labels_703_20261026_v1/evaluation/rankings_with_silver_labels.csv'),
          ('Metrics và bất định','outputs/ai_silver_labels_703_20261026_v1/evaluation/metrics_summary.csv; confidence_intervals.csv; paired_deltas.csv'),
          ('Toàn bộ hình','figure_inventory.json trong bộ nguồn báo cáo'),
          ('Hash nguồn đã đọc','evidence_manifest.json trong bộ nguồn báo cáo')]
    # Use path-aware wrapping instead of ordinary words for the provenance table.
    tx=table('Các nguồn bằng chứng chính.', ['Nội dung','Nguồn'],rows,'p{.22\\textwidth}p{.70\\textwidth}',long=True,size='footnotesize')
    for _,p in rows:tx=tx.replace(esc(p),r'\artifact{'+p+'}')
    write('evidence_table.tex',tx)
    manifest=dict(snapshot='2026-10-26 Asia/Bangkok',source_sha256=SOURCES,
                  topic_catalogs=7,topics=70,keyword_slots=700,
                  figures=len(INVENTORY),training_executed=False,encoder_inference_executed=False,
                  external_api_called=False,models_modified=False,
                  method='Read existing outputs; generate report tables and scientific figures only')
    (HERE/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')


def normalize_prose():
    """Separate counts from adjacent prose while protecting math, paths and IDs."""
    words = ['corpus', 'với', 'và', 'trên', 'từ', 'lên', 'xuống', 'giảm', 'tăng',
             'chứa', 'có', 'đủ', 'đúng', 'batch', 'microbatch', 'accumulation', 'Step',
             'step', 'pilot', 'proxy', 'khoảng', 'Điểm', 'mỗi', 'kỳ', 'thiểu', 'decay',
             'clipping', 'là', 'đạt', 'topic', 'rank', 'Top', 'nhãn', 'query', 'seed',
             'NPMI', 'ARI', 'diversity', 'RRF', 'KMeans', 'overlap', 'cửa sổ', 'error',
             'Bootstrap', 'thành', 'Review', 'threshold', 'cho', 'outlier', 'outliers',
             'cụm', 'chỉ', 'còn', 'gồm', 'dùng', 'giữ', 'âm', 'dương', 'giá', 'ngày',
             'ngoài', 'Trên', 'Nhưng', 'ngưỡng', 'chiều', 'thứ', 'phút']
    protected = r'\\(?:artifact|source|input|label|ref|eqref|cite|includegraphics)\{[^}]*\}|\$[^$]*\$|\\begin\{(?:equation|align)\}[\s\S]*?\\end\{(?:equation|align)\}'
    pattern = re.compile('(?P<protected>'+protected+')|(?P<word>\\b(?:'+'|'.join(map(re.escape,words))+'))(?P<digit>[0-9])')
    for path in [HERE/'main.tex', *sorted((HERE/'chapters').glob('*.tex')), *sorted(GEN.glob('*.tex'))]:
        s=path.read_text(encoding='utf8')
        s=pattern.sub(lambda m:m.group(0) if m.group('protected') else m.group('word')+' '+m.group('digit'),s)
        s=s.replace('n\\_init50','n\\_init 50').replace('n\\_init10','n\\_init 10')
        path.write_text(s,encoding='utf8')


if __name__=='__main__':
    build_theory();main_figures();main_tables();search_examples();topic_catalogs();appendix();evidence();normalize_prose()
    print(json.dumps(dict(status='BUILT',source_files=len(SOURCES),figures=len(INVENTORY),catalogs=7,topics=70),ensure_ascii=False))
