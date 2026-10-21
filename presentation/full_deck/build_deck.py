"""Create a portable Vietnamese Beamer deck from existing project evidence.

Reads artifacts only. Does not train, encode, access credentials or call APIs.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import zipfile

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCES = {}
SLIDES = []


def register(rel):
    path = ROOT / rel
    SOURCES[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(rel):
    return json.loads(register(rel).read_text(encoding='utf-8'))


def rows(rel):
    with register(rel).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def esc(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(text))


def items(*lines):
    return '\\begin{itemize}\n' + '\n'.join('\\item '+x for x in lines) + '\n\\end{itemize}'


def table(headers, data, spec=None, size=''):
    spec = spec or ('l' + 'r'*(len(headers)-1))
    spec = spec.replace('p{', r'>{\raggedright\arraybackslash}p{')
    return ('{'+size+'\n\\begin{center}\\begin{tabular}{'+spec+'}\n\\toprule\n'
            + ' & '.join(headers) + r' \\ \midrule' + '\n'
            + '\n'.join(' & '.join(map(str, row))+r' \\' for row in data)
            + '\n\\bottomrule\n\\end{tabular}\\end{center}}')


def add(title, body, source, notes='', appendix=False):
    SLIDES.append(dict(title=title, body=body, source=source, notes=notes, appendix=appendix))


def fig(title, rel, caption, notes='', appendix=False, height='4.85cm'):
    source = register(rel)
    dest = OUT / 'figures' / source.name
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(source, dest)
    body = (r'\begin{center}\includegraphics[width=\linewidth,height='+height
            +r',keepaspectratio]{figures/'+dest.name+r'}\end{center}'
            +r'\vspace{-0.12cm}{\small '+caption+'}')
    add(title, body, rel, notes or caption, appendix)


def topic_sample(title, data, count_field, rel):
    selected = sorted([r for r in data if int(r['topic_id']) >= 0],
                      key=lambda r: -int(r[count_field]))[:3]
    body = ''
    for r in selected:
        body += (r'\textbf{Topic '+r['topic_id']+' ('+f"{int(r[count_field]):,}".replace(',', '.')
                 +' tài liệu)}'+r'\par{\small '+esc(r['top10_keywords'])+r'}\par\medskip')
    body += r'{\small ID chỉ có ý nghĩa trong cùng model. Các từ bên dưới là output gốc.}'
    add(title, body, rel,
        'Ba topic có nhiều tài liệu nhất. Tên chủ đề không được tự gán thành nhãn chuyên gia. '
        'Số tài liệu LDA là dominant assignments, BERTopic là assigned clusters. '
        'Phụ lục có toàn bộ IDs và Top5, topic_catalog.csv giữ đủ Top10.')


core = json.loads((OUT / 'core_slides.json').read_text(encoding='utf-8'))
lda = rows('outputs/tables/stage2/topic_summary.csv')
bert = rows('outputs/tables/stage3/bertopic_topic_summary.csv')
chunk = rows('outputs/bertopic_chunked/topic_summary.csv')
training = read_json('models/bert_finetuned/training_success.json')
q19 = read_json('outputs/search_upgrade_20261018/query_runs/Q19.json')
terms = rows('outputs/tables/stage2/eda_term_frequencies.csv')

# Attach important current source files to the evidence manifest.
for path in ['docs/FINAL_RESEARCH_REPORT.md', 'docs/MATH_AND_OPTIMIZATION_REVIEW.md',
             'docs/SEARCH_RESULTS_REVIEW.md', 'src/pipeline.py', 'src/data_io.py',
             'src/text_processing.py', 'src/quality.py', 'src/stage2.py',
             'src/stage3_neural.py', 'src/stage3_common.py', 'src/stage3_topics.py',
             'src/bertopic_chunked.py', 'src/search_v2.py', 'src/search_v2_constraints.py',
             'src/search_v2_service.py', 'src/research_evaluation.py', 'src/ai_judge.py', 'configs/config.yaml',
             'configs/stage2.yaml', 'configs/stage3.yaml', 'configs/search_v2.json',
             'outputs/reports/stage1_summary.json', 'outputs/reports/stage2/selected_model.json',
             'outputs/reports/stage3/pretrained_test_metrics.json',
             'outputs/reports/stage3/finetuned_test_metrics.json',
             'outputs/reports/stage3/bertopic_metrics.json', 'outputs/bertopic_chunked/metrics.json',
             'outputs/retrieval_ai_evaluation_v1/REPORT.md',
             'outputs/retrieval_ai_evaluation_v1/synthetic/summary.csv',
             'outputs/search_runtime_optimization_20261019/verification.json']:
    register(path)


for original in core:
    if original['appendix']:
        continue
    s = dict(original)
    s.pop('number', None)
    # Keep one role per frame, avoid overclaiming the supervision and search scores.
    if s['title'] == 'Fine-tuning: học từ Title và Abstract':
        s['body'] = s['body'].replace(
            r'Không cần tự tạo relevance labels cho truy vấn tự nhiên để chạy proxy này.',
            r'Title--Abstract là supervision thay thế, chưa có nhãn query tự nhiên.')
    if s['title'] == 'Kết quả BERTopic: có sự đánh đổi':
        s['body'] = s['body'].replace(r'$C\_v$', r'$C_v$')
    add(**s)
    title = s['title']

    if title == 'Pipeline toàn bộ dự án':
        add('Offline và online trong cùng một hệ thống',
            table(['Chế độ', 'Công việc', 'Đầu ra'], [
                ['Offline', 'Làm sạch, fit LDA / BERTopic', 'Corpus và topics'],
                ['Offline', 'Fine-tune BGE, encode corpus', 'Weights và vectors'],
                ['Chuẩn bị', 'Nạp / xây BM25, kiểm tra artifacts', 'Search engine'],
                ['Online', 'Encode query, truy hồi, RRF, rerank', 'Top5 + passage']],
                r'lp{6.0cm}p{3.5cm}', r'\small') + items(
                r'Huấn luyện không lặp lại khi người dùng gửi query.',
                r'Nhánh mặc định dùng BGE pretrained, không cần checkpoint fine-tuned.'),
            'src/search_v2.py; src/search_v2_service.py; configs/search_v2.json',
            'Tách build-time và query-time. BM25 index được kiểm tra identity và có thể tái dùng từ cache. '
            'Models và dense matrix được nạp khi cần. Không có train hoặc gọi Gemini trong search mặc định.')

    elif title == 'Dữ liệu trước và sau tiền xử lý':
        disposition = rows('outputs/reports/cleaning_summary.csv')
        sums = {}
        for r in disposition:
            sums[r['disposition']] = sums.get(r['disposition'], 0)+int(r['count'])
        add('Quyết định sử dụng toàn bộ 20.576 bản ghi',
            table(['Disposition', 'Số bản ghi'],
                  [[esc(k), f'{v:,}'.replace(',', '.')] for k, v in sorted(sums.items(), key=lambda p: -p[1])],
                  'lr', r'\small')
            + r'{\small Relevance label và disposition là hai trường khác nhau. Clean master giữ mọi bản ghi.}',
            'outputs/reports/cleaning_summary.csv',
            'Các hàng loại trừ nhau và cộng đúng 20.576. RELEVANCE_REVIEW là 6.596, khác tổng nhãn '
            'UNCERTAIN 6.598 vì một bản ghi còn có các điều kiện chất lượng khác trước quyết định cuối.')
        fig('Thay đổi dữ liệu qua từng quyết định xử lý',
            'outputs/figures/data_preparation/figure_05_cleaning_summary.png',
            'Bảng quyết định theo nguồn. Các nhóm review vẫn tồn tại trong clean master.', height='4.8cm')

    elif title == 'Tiền xử lý: mỗi bước phục vụ mục đích gì?':
        add('Phát hiện gần trùng và chống rò rỉ split',
            r'\[\cos(x,y)=\frac{x^\top y}{\|x\|\|y\|},\qquad J(A,B)=\frac{|A\cap B|}{|A\cup B|}\]'
            + items(r'Title TF-IDF unigram/bigram: lấy cặp có cosine $\geq0{,}88$.',
                    r'Kiểm tra tiếp Abstract word Jaccard $\geq0{,}80$.',
                    r'Union-Find gộp liên kết thành \texttt{split\_group\_id}.',
                    r'Chỉ hợp nhất bản ghi trùng đã xác nhận. Patent khác ID có thể được giữ.'),
            'src/quality.py::duplicate_audit; configs/config.yaml',
            'Ngoài near-duplicate còn có các liên kết exact ID/DOI/text và các số patent/priority chung. '
            'Đây là candidate relationships, không xác minh mọi patent family. Tài liệu cùng nhóm đi cùng split.')

    elif title == 'Sau tiền xử lý thu được gì?':
        fig('Các từ xuất hiện nhiều theo từng nguồn',
            'outputs/figures/stage2/eda_02_top_terms.png',
            'Thống kê tokens Stage 1. Tần suất cao thể hiện phổ biến, chưa thể hiện độ phân biệt của một topic.')
        frequent = []
        for src in ['Patent', 'Publication']:
            top = sorted([r for r in terms if r['source']==src], key=lambda r: -int(r['term_frequency']))[:5]
            frequent.extend([[src, esc(r['term']), f"{int(r['term_frequency']):,}".replace(',', '.'),
                              f"{int(r['document_frequency']):,}".replace(',', '.')] for r in top])
        add('Top5 tokens: số lần và số tài liệu chứa từ',
            table(['Nguồn', 'Token', 'TF', 'DF'], frequent, 'llrr', r'\small'),
            'outputs/tables/stage2/eda_term_frequencies.csv',
            'TF là tổng số lần xuất hiện. DF là số tài liệu chứa token. Một từ xuất hiện nhiều lần trong '
            'một tài liệu tăng TF nhưng chỉ tăng DF một lần. Đây là token của nhánh topic, khác lexical tokens Search V2.')

    elif title == 'LDA: quy trình và sản phẩm sau huấn luyện':
        add('LDA: các phân phối thay đổi trong quá trình fit',
            r'\[r_{dnk}\propto\exp\{\mathbb E[\log\theta_{dk}]+\mathbb E[\log\phi_{k,w_{dn}}]\}\]'
            r'\[\gamma_{dk}=\alpha_k+\sum_n r_{dnk},\quad'
            r'\widehat\lambda_{kw}=\eta_w+\frac{D}{|\mathcal B|}\sum_{d\in\mathcal B,n:w_{dn}=w}r_{dnk}\]'
            r'\[\lambda^{new}=(1-\rho_t)\lambda^{old}+\rho_t\widehat\lambda\]'
            +r'{\small Diễn giải variational inference của Gensim. $r$: trách nhiệm topic, $\gamma$: trạng thái tài liệu, '
            +r'$\lambda$: trạng thái topic--word. Dự án gọi thư viện để thực hiện cập nhật.}',
            'src/stage2.py::_fit_lda; gensim.models.ldamodel.LdaModel',
            'Đây là mô tả toán học dạng rút gọn, không phải tự viết optimizer trong dự án. '
            'Trong BOW, thay tổng từng vị trí từ bằng tổng có trọng số count. '
            'K=20 vẫn cố định trong một lần fit. Thuật toán điều chỉnh phân phối, không tạo thêm topic. '
            'Gensim online variational Bayes kết hợp sufficient statistics theo learning-rate rho.')
        fig('So sánh các cấu hình LDA đã chạy',
            'outputs/figures/stage2/lda_01_model_comparison.png',
            'Chọn K=20 theo protocol hiện tại. Chỉ hai K qua sàng lọc được thử thêm seed.')
        topic_sample('Output LDA: ba topic trội lớn nhất', lda, 'dominant_documents',
                     'outputs/tables/stage2/topic_summary.csv')
        fig('Phân bố tài liệu trên 20 topic LDA',
            'outputs/figures/stage2/lda_02_topic_distribution.png',
            'Dominant assignment đếm mỗi tài liệu vào một topic. Soft mass cộng xác suất topic qua tài liệu.')

    elif title == 'BGE: bi-encoder tạo vector ngữ nghĩa':
        add('BGE: token tương tác qua self-attention',
            r'\[Q=HW_Q,\quad K=HW_K,\quad V=HW_V\]'
            r'\[\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V\]'
            + items(r'$H$: biểu diễn token. $M$: attention mask cho padding.',
                    r'Các lớp attention và feed-forward tạo biểu diễn có ngữ cảnh.',
                    r'BGE lấy hidden state \texttt{[CLS]} làm vector, sau đó L2-normalize.'),
            'models/bge_pretrained/config.json; src/stage3_neural.py',
            'Công thức giải thích cơ chế transformer đã dùng trong checkpoint, không phải module attention tự viết. '
            'BGE base có 12 encoder layers. Dự án chỉ mở layer.11 khi fine-tuning. '
            'Q và K trong công thức attention là token projections, khác query/document trong retrieval.')

    elif title == 'Fine-tuning: contrastive loss đối xứng':
        add('False negatives: cơ chế mask và kết quả thực tế',
            items(r'In-batch negatives chỉ lấy các Abstract khác trong cùng batch.',
                  r'Mask nếu cùng nhóm hoặc Title Jaccard $\geq0{,}8$.',
                  r'Artifact run ghi \texttt{masked\_pairs = 0}.',
                  r'Vì vậy, kết quả hiện có chưa đo được lợi ích thực nghiệm của mask.',
                  r'Evaluator chấp nhận nhiều positives cùng nhóm, loss dùng positive đường chéo.'),
            'models/bert_finetuned/training_success.json; src/stage3_neural.py',
            'Phân biệt cơ chế triển khai và tác động đã quan sát. Mặc dù có logic mask, các batch của run này '
            'không gặp cặp thỏa điều kiện theo counter đã lưu. Chưa có ablation mask on/off để suy ra lợi ích.')

    elif title == 'Fine-tuning: cập nhật gì, lưu lại gì?':
        add('AdamW cập nhật trọng số block cuối',
            r'\[g_t=\nabla_\theta\mathcal L,\qquad '
            r'g_t\leftarrow g_t\min\left(1,\frac{1}{\|g_t\|_2+\varepsilon}\right)\]'
            r'\[m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad '
            r'v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2\]'
            r'\[\theta_t=(1-\eta_t\lambda)\theta_{t-1}-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}\]'
            +r'{\small $\widehat m,\widehat v$: bias correction. $\lambda=0{,}01$, learning rate cực đại '
            +r'$2\times10^{-5}$. Warm-up 10\% rồi giảm tuyến tính. Các tham số frozen giữ nguyên.}',
            'src/stage3_neural.py::train; torch.optim.AdamW',
            'Tính gradient đối với các tham số requires_grad=True, rồi clip global norm tối đa 1. '
            'AdamW dùng moment bậc một và hai, weight decay tách riêng. Các beta và epsilon theo mặc định '
            'dependency đã cài, không tự nhận dự án đã tune chúng. Scheduler.step chạy sau optimizer.step.')
        pct = 100*training['trainable_parameters']/training['total_parameters']
        add('Bằng chứng cập nhật trọng số của phiên huấn luyện',
            table(['Hạng mục', 'Giá trị đã lưu'], [
                ['Tổng tham số', f"{training['total_parameters']:,}".replace(',', '.')],
                ['Tham số trainable', f"{training['trainable_parameters']:,}".replace(',', '.')+f' ({pct:.2f}\\%)'],
                ['Optimizer steps / best checkpoint', '1.365 / 1.365'],
                [r'$\|\theta_{selected}-\theta_{initial}\|_2$', f"{training['selected_weight_delta_l2']:.6f}"],
                ['Hash tham số frozen trước / sau', 'Giống nhau'],
                ['Test tham gia chọn checkpoint', 'False']], 'lr', r'\small')
            +r'{\small Các giá trị lấy từ training\_success.json, không chạy lại huấn luyện.}',
            'models/bert_finetuned/training_success.json',
            'Tổng 109.482.240, trainable 7.087.872, khoảng 6,47%. Layer cuối có các ma trận attention, '
            'FFN và layernorm được cập nhật. L2 delta khác 0 và frozen hash bằng nhau xác nhận '
            'có thay trọng số đúng phạm vi, chưa chứng minh lợi ích search.')
        fig('Loss huấn luyện và chọn checkpoint bằng validation',
            'outputs/figures/stage3/bert_01_training.png',
            'Loss giảm trong run đã lưu. Chọn step 1.365 bằng group MRR@10 trên validation, không dùng test.')

    elif title == 'BERTopic: tham số, đầu ra và ý nghĩa':
        topic_sample('Output BERTopic baseline: ba cụm lớn nhất', bert, 'documents',
                     'outputs/tables/stage3/bertopic_topic_summary.csv')
        fig('Phân bố cụm BERTopic baseline và noise',
            'outputs/figures/stage3/bertopic_01_distribution.png',
            '92 topics và 2.751 outliers. Topic -1 gom noise, không đại diện một chủ đề chuyên môn thống nhất.')
        fig('Biểu diễn 2D của BERTopic baseline',
            'outputs/figures/stage3/bertopic_04_umap_2d.png',
            'Hình để quan sát cấu trúc. Clustering thực tế dùng UMAP 5D, không dùng trực tiếp tọa độ trên hình.')

    elif title == 'Chunking: bao phủ phần cuối Abstract':
        topic_sample('Output BERTopic chunked: ba cụm lớn nhất', chunk, 'documents',
                     'outputs/bertopic_chunked/topic_summary.csv')
        fig('Phân bố 84 topics của nhánh chunked',
            'outputs/bertopic_chunked/figures/topic_distribution.png',
            'Nhánh chunked có 3.120 outliers. Topic ID được tạo lại, không đồng nhất trực tiếp với baseline.')

    elif title == 'Kết quả BERTopic: có sự đánh đổi':
        fig('Từ khóa và trọng số của các topic BERTopic',
            'outputs/figures/stage3/bertopic_02_keywords.png',
            'Các từ khóa theo c-TF-IDF của baseline. Phụ lục có đủ 92 IDs và bảng 84 IDs chunked.')

    elif title == 'Search liên hệ với các mô hình trước ra sao?':
        add('Mỗi mô hình tạo bao nhiêu topic?',
            table(['Thành phần', 'Đầu ra', 'Số topic'], [
                ['LDA K=20', 'Document--topic + topic--word', '20'],
                ['BERTopic baseline, seed 42', 'Cluster ID + keywords', '92'],
                ['BERTopic chunked, seed 42', 'Cluster ID + keywords', '84'],
                ['BGE pretrained / fine-tuned', 'Vector 768 chiều', 'Không tạo topic'],
                ['BM25 / RRF / MiniLM', 'Điểm và danh sách xếp hạng', 'Không tạo topic']],
                r'lp{6.2cm}l', r'\small')
            +r'{\small Noise -1 không nằm trong 92/84 topics. Seed BERTopic 123 của baseline cho 82 topics.}',
            'outputs/reports/stage3/bertopic_metrics.json; outputs/bertopic_chunked/metrics.json',
            '768 là số chiều embedding, không phải 768 topic. Fine-tuning thay encoder chứ không tạo taxonomy. '
            'LDA gán xác suất cho 20 topic đặt trước. BERTopic suy ra số cụm từ dữ liệu và các tham số mật độ.')

    elif title == 'Xây chỉ mục Search V2 trước khi truy vấn':
        add('Mười phương pháp Search V2 có thể chọn',
            table(['Nhóm', 'Các phương pháp'], [
                ['Lexical', r'\texttt{BM25\_CHEMICAL}'],
                ['Dense', 'PRETRAINED, FINETUNED, CHUNKED'],
                ['Hybrid (RRF)', 'PRETRAINED, FINETUNED, CHUNKED'],
                ['Rerank', 'PRETRAINED, FINETUNED, CHUNKED']], 'll')
            + items(r'Các tên dense/hybrid/rerank là kết hợp nhóm với hậu tố biểu diễn.',
                    r'Mặc định: \texttt{RERANK\_PRETRAINED}.',
                    r'14 methods ở evaluator gồm thêm bốn đối chứng lịch sử, không phải 14 backend mới.'),
            'src/search_v2.py::METHODS; configs/search_v2.json',
            'Tên đầy đủ gồm DENSE_PRETRAINED, DENSE_FINETUNED, DENSE_CHUNKED, '
            'HYBRID_PRETRAINED, HYBRID_FINETUNED, HYBRID_CHUNKED, '
            'RERANK_PRETRAINED, RERANK_FINETUNED, RERANK_CHUNKED và BM25_CHEMICAL. '
            'CHUNKED dùng query encoder fine-tuned với query max128, khác document chunk pair256.')

    elif title == 'Luồng online và kết quả trả về':
        result = q19['responses']['RERANK_PRETRAINED']['results'][:5]
        add('Query Q19: Top5 thực tế của RERANK_PRETRAINED',
            r'{\small Aqueous sodium-ion batteries with water-in-salt electrolytes}'
            +table(['Hạng', 'Title (rút gọn để hiển thị)', 'Score'],
                   [[str(r['rank']), esc(r['title'][:96]+('...' if len(r['title'])>96 else '')),
                     f"{r['score']:.3f}"] for r in result], r'rp{10.7cm}r', r'\small')
            +r'{\small Score chỉ dùng xếp hạng trong method. Đây là kết quả đã lưu, chưa có grade độc lập.}',
            'outputs/search_upgrade_20261018/query_runs/Q19.json',
            '\n'.join(f"Hạng {r['rank']}: {r['doc_id']} — {r['title']} — score {r['score']}" for r in result))
        top = result[0]
        passage = top['evidence_passage']
        # Quote an actual contiguous span in the stored evidence, never generate a paraphrase as a quote.
        body = passage.split('\n', 1)[-1]
        excerpt = body[:620]
        if len(body)>620:
            excerpt = excerpt.rsplit(' ', 1)[0]+'...'
        add('Q19: passage để người dùng kiểm tra bằng chứng',
            r'{\small\textbf{'+esc(top['title'])+r'}}\par\medskip'
            +r'\begin{quote}\small '+esc(excerpt)+r'\end{quote}'
            +r'{\small Trích liên tục từ passage của hạng 1, rút gọn phần cuối. '
            +r'Offset ánh xạ về Abstract đã chuẩn hóa. Người dùng cần đọc toàn Abstract và tài liệu gốc.}',
            'outputs/search_upgrade_20261018/query_runs/Q19.json',
            'Toàn bộ passage gốc:\n'+passage+'\n\nTài liệu '+top['doc_id']
            +'. Hình thức trích dẫn ở slide không tạo ra nhận định khoa học mới. Search không có LLM tổng hợp câu trả lời.')

    elif title == 'Công thức đánh giá Top5':
        add('Rubric relevance và bằng chứng cần chấm',
            table(['Grade', 'Ý nghĩa theo rubric đang dùng'], [
                ['0', 'Ngoài intent hoặc mâu thuẫn một yêu cầu bắt buộc'],
                ['1', 'Liên quan một phần / nền tảng, thiếu đặc điểm chính'],
                ['2', 'Đáp ứng intent chính có bằng chứng, còn thiếu chi tiết'],
                ['3', 'Đáp ứng đầy đủ intent và các quan hệ được yêu cầu'],
                ['UNRESOLVED', 'Title/Abstract chưa đủ để đánh giá có cơ sở']],
                r'lp{10.7cm}', r'\small')
            +items(r'Các trục: hệ pin, vật liệu/công thức, điều kiện và quan hệ kỹ thuật.',
                   r'Code kiểm tra trích dẫn có thật trong văn bản. Việc diễn giải vẫn cần review.'),
            'src/ai_judge.py::RUBRIC, validate_judgment; src/research_evaluation.py',
            'Rubric hiện được mã hóa trong nhánh AI. Kế hoạch nhãn độc lập cần chốt intent/rubric trước '
            'khi chấm, ẩn method/rank, hai người chấm và phân xử. UNRESOLVED không được tự đổi thành 0. '
            'Grade 2/3 tính relevant cho P/MRR. Evidence substring chỉ xác nhận đoạn có tồn tại, '
            'không xác nhận diễn giải hoặc tính đúng khoa học của đoạn đó.')

    elif title == 'Nhãn relevance độc lập: hiện chưa hoàn tất':
        fig('Mức hoàn thành của bộ nhãn AI',
            'outputs/retrieval_ai_evaluation_v1/judgment_coverage.png',
            'AI consensus số: 116/942 cặp. Chỉ 2/23 query đủ pool. Hai biểu mẫu nhãn độc lập vẫn trống.')

    elif title == 'Tối ưu kỹ thuật đã có bằng chứng':
        add('Độ trễ đo được và giới hạn vận hành',
            items(r'Lưới CPU: threads $\{1,2,4\}$ và batch $\{4,8,16\}$.',
                  r'3 development queries, 2 repetitions, tổng 54 requests được đo.',
                  r'Giữ 4 threads / batch 4, median warm 4,506 giây/request.',
                  r'Có spike 839,663 giây ở một cấu hình khác, vẫn giữ trong raw data.',
                  r'Một worker tuần tự hóa tác vụ. Chưa có benchmark concurrency/p95 đại diện.'),
            'outputs/math_optimization_20261019/execution_dev/summary.json; src/search_v2_service.py',
            'Không so cold startup với warm query latency. Kết quả kernel Top-K/filter không cộng hoặc nhân '
            'thành mức tăng tốc end-to-end. Compare request có reuse nên thời gian của method phía sau '
            'không tương đương benchmark chạy độc lập. Threads/batch tốt nhất trong lưới nhỏ trên máy đã đo.')


# Existing conceptual appendices remain available after the presentation route.
for original in core:
    if original['appendix']:
        s = dict(original)
        s.pop('number', None)
        add(**s)

# Complete topic inventory: show every topic ID, count and its first five stored keywords.
# The accompanying catalog retains all ten keywords and the original fields.
catalog = []
for name, data, count_field, rel in [
        ('LDA K20', lda, 'dominant_documents', 'outputs/tables/stage2/topic_summary.csv'),
        ('BERTopic baseline', bert, 'documents', 'outputs/tables/stage3/bertopic_topic_summary.csv'),
        ('BERTopic chunked', chunk, 'documents', 'outputs/bertopic_chunked/topic_summary.csv')]:
    clean = sorted([r for r in data if int(r['topic_id'])>=0], key=lambda r: int(r['topic_id']))
    for r in data:
        catalog.append(dict(model=name, topic_id=r['topic_id'], documents=r[count_field],
                            is_outlier=int(r['topic_id'])<0, top10_keywords=r['top10_keywords'], source=rel))
    per_page=6
    for start in range(0, len(clean), per_page):
        batch=clean[start:start+per_page]
        add(f'{name}: topic {batch[0]["topic_id"]}–{batch[-1]["topic_id"]}',
            table(['ID', 'Tài liệu', 'Top5 từ khóa xuất trực tiếp'],
                  [[r['topic_id'], f"{int(r[count_field]):,}".replace(',', '.'),
                    esc('; '.join(r['top10_keywords'].split('; ')[:5]))] for r in batch],
                  r'rrp{10.6cm}', r'\small')
            +r'{\small Thứ tự ID gốc. Top10 đầy đủ nằm trong \texttt{topic\_catalog.csv}. '
            +r'Số tài liệu dùng dominant assignments cho LDA, cluster assignments cho BERTopic.}',
            rel, 'Bảng gốc được phân trang, không đặt thêm tên chủ đề. '
            'Các ký hiệu như O-, P2 và các token bất thường giữ đúng artifact để người đọc nhận diện giới hạn preprocessing.', True)

add('Noise trong hai nhánh BERTopic',
    table(['Nhánh', 'ID', 'Tài liệu', 'Top5 từ khóa trong noise'], [
        [name, '-1', f"{int(next(r for r in data if r['topic_id']=='-1')['documents']):,}".replace(',', '.'),
         esc('; '.join(next(r for r in data if r['topic_id']=='-1')['top10_keywords'].split('; ')[:5]))]
        for name,data in [('Baseline',bert),('Chunked',chunk)]], r'lrrp{8.8cm}', r'\small')
    + items(r'Noise chứa tài liệu có mật độ không phù hợp các cụm được chọn.',
            r'Từ khóa của noise chưa tạo thành một chủ đề thống nhất.',
            r'Tăng coherence của inlier đồng thời tăng noise là một đánh đổi cần review.'),
    'outputs/tables/stage3/bertopic_topic_summary.csv; outputs/bertopic_chunked/topic_summary.csv',
    'Noise không được đếm vào 92/84. Các tài liệu noise vẫn có thể là tài liệu phù hợp với query search.', True)

# One figure per page, preserve the original evidence image and aspect ratio.
fig('Keyword heatmap của 20 topic LDA', 'outputs/figures/stage2/lda_06_keyword_heatmap.png',
    'Trọng số topic--word theo model đã chọn. Đây là phân phối từ, khác tần suất từ toàn corpus.', appendix=True)
fig('UMAP 2D của biểu diễn chunked', 'outputs/bertopic_chunked/figures/umap_2d.png',
    'Projection 2D để quan sát. Phân cụm vẫn chạy trên 5 chiều.', appendix=True)
for number in range(1,8):
    fig(f'Top10 từ khóa chunked: trang {number}/7',
        f'outputs/bertopic_chunked/figures/top10_keywords_{number:02d}.png',
        'Hình gốc từ outputs. Bảy trang hình bao phủ 84 topic, mỗi topic có trọng số c-TF-IDF.', appendix=True,
        height='5.5cm')

with (OUT/'topic_catalog.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(catalog[0]))
    writer.writeheader();writer.writerows(catalog)

PREAMBLE=r'''\documentclass[11pt,aspectratio=169]{beamer}
\usepackage{fontspec}
\IfFontExistsTF{Arial}{\setsansfont{Arial}\setmainfont{Arial}}{\setsansfont{TeX Gyre Heros}\setmainfont{TeX Gyre Heros}}
\IfFontExistsTF{Consolas}{\setmonofont{Consolas}[Scale=0.85]}{\setmonofont{Latin Modern Mono}[Scale=0.85]}
\usepackage{amsmath,amssymb,booktabs,array,graphicx}
\definecolor{Navy}{HTML}{12304A}
\definecolor{Teal}{HTML}{087F8C}
\definecolor{Ink}{HTML}{213449}
\definecolor{Muted}{HTML}{62788A}
\setbeamercolor{normal text}{fg=Ink,bg=white}
\setbeamercolor{frametitle}{fg=Navy}
\setbeamercolor{structure}{fg=Teal}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize item}{\small$\bullet$}
\setbeamersize{text margin left=0.7cm,text margin right=0.7cm}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}
\setbeamertemplate{frametitle}{\vspace{0.26cm}\insertframetitle\par\vspace{0.1cm}}
\setbeamertemplate{footline}{\hspace{0.7cm}\textcolor{Muted}{\scriptsize Sodium-ion Research Explorer}\hfill\textcolor{Muted}{\scriptsize\insertframenumber/\inserttotalframenumber}\hspace{0.7cm}\vspace{0.18cm}}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.17}
\setlength{\emergencystretch}{1em}
\AtBeginEnvironment{itemize}{\setlength{\itemsep}{0.12cm}}
\newcommand{\src}[1]{\vfill{\color{Muted}\fontsize{5.8}{7}\selectfont Nguồn: #1\par}}
\newcommand{\norm}{\operatorname{norm}}
\hypersetup{pdftitle={Sodium-ion Research Explorer: pipeline, models, retrieval and evaluation},pdfauthor={},colorlinks=true,linkcolor=Teal,urlcolor=Teal}
\begin{document}
\begin{frame}[plain]
\vspace{0.45cm}
{\color{Teal}\large PHƯƠNG PHÁP VÀ KẾT QUẢ NGHIÊN CỨU}\par
\vspace{0.55cm}
{\color{Navy}\LARGE\bfseries Sodium-ion\\[0.12cm]Research Explorer}\par
\vspace{0.45cm}
{\large Dữ liệu, mô hình chủ đề, học biểu diễn\\và hệ thống tìm kiếm tài liệu}\par
\vfill
{\color{Muted}\small Bằng chứng từ code và outputs của dự án\\Rà soát ngày 20/10/2026}
\end{frame}
'''

tex=[PREAMBLE]
notes=['# Ghi chú thuyết trình\n',
       'Nguồn số liệu là các artifacts đã lưu. Không có huấn luyện hoặc benchmark mới khi biên soạn.\n',
       'Phần chính có thể chia thành hai buổi: dữ liệu/mô hình và search/đánh giá. '
       'Phụ lục dành cho tra cứu topics và câu hỏi bảo vệ.\n']
for i,s in enumerate(SLIDES,2):
    s['number']=i
    if s['appendix'] and not SLIDES[i-3]['appendix']:
        tex.append('\\appendix\n')
    # Very long titles need a line break rather than a reduced body font.
    title=esc(s['title'])
    if len(s['title'])>74:
        split=s['title'].rfind(' ',0,65)
        title=esc(s['title'][:split])+r'\\'+esc(s['title'][split+1:])
    # A leading {...} can otherwise be parsed as Beamer's optional subtitle.
    tex.append('\\begin{frame}[t]{'+title+'}\n\\relax\n'+s['body']+'\n\\src{'+esc(s['source'])+'}\n\\end{frame}\n')
    notes.append(f"\n## Slide {i}: {s['title']}\n\n{s['notes']}\n\nNguồn: {s['source']}\n")
tex.append('\\end{document}\n')
(OUT/'main.tex').write_text(''.join(tex),encoding='utf-8')
(OUT/'speaker_notes.md').write_text('\n'.join(notes),encoding='utf-8')
(OUT/'slide_inventory.json').write_text(json.dumps(SLIDES,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=dict(date='2026-10-20',source_root=str(ROOT),
              main_slides=1+sum(not s['appendix'] for s in SLIDES),
              appendix_slides=sum(s['appendix'] for s in SLIDES),
              figures=len(list((OUT/'figures').glob('*.png'))),
              topic_counts={'LDA':20,'BERTopic_baseline':92,'BERTopic_chunked':84},
              sources=SOURCES,
              scope='Read-only synthesis of stored artifacts; no retraining, query execution, APIs or credentials.')
(OUT/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},ensure_ascii=False))
