"""Read saved evidence and produce the Vietnamese whole-system print report.

No training, embedding inference, Qdrant access or source data mutation.
ReportLab is installed in .tools/pdf_report_deps, outside project dependencies.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/pdf_report_deps'))
sys.path.insert(0, str(ROOT))
import hashlib
import json
from html import escape
import re
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

AUDIT = ROOT / 'outputs/batterybert_optimization_20261023_v1'
OUT = ROOT / 'output/pdf'
NAME = 'BAO_CAO_TOAN_BO_PIPELINE_20261024'
NAVY = colors.HexColor('#16344B')
TEAL = colors.HexColor('#157F83')
MUTED = colors.HexColor('#526776')
LIGHT = colors.HexColor('#EDF4F7')
ORANGE = colors.HexColor('#A55D28')
WIDTH = A4[0] - 96
SOURCES = []


def read_json(relative):
    p = ROOT / relative; SOURCES.append(p)
    return json.loads(p.read_text(encoding='utf8'))


def read_csv(relative):
    p = ROOT / relative; SOURCES.append(p)
    return pd.read_csv(p)


def fmt(value, digits=4):
    return f'{value:.{digits}f}'.replace('.', ',')


def pct(value):
    return f'{value * 100:.2f}%'.replace('.', ',')


def count(value):
    return f'{int(value):,}'.replace(',', '.')


for name, file in [('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf'), ('Arial-Italic', 'ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + file))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-Bold')
STYLES = {
    'body': ParagraphStyle('body', fontName='Arial', fontSize=10.2, leading=15.1, textColor=NAVY, spaceAfter=8),
    'small': ParagraphStyle('small', fontName='Arial', fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=6),
    'h1': ParagraphStyle('h1', fontName='Arial-Bold', fontSize=20, leading=25, textColor=NAVY, spaceAfter=14),
    'h2': ParagraphStyle('h2', fontName='Arial-Bold', fontSize=11.5, leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=6),
    'cell': ParagraphStyle('cell', fontName='Arial', fontSize=8.5, leading=12, textColor=NAVY),
    'headcell': ParagraphStyle('headcell', fontName='Arial-Bold', fontSize=8.5, leading=12, textColor=colors.white),
}
story, markdown = [], []


def clean(text):
    text = str(text)
    words = 'vòng|bước|topic|nhóm|cụm|seed|batch|block|power|pool|top|train|test|corpus|split|tập|từ|trên|dưới|xuống|còn|đạt|có|của|ngày|bằng|với|tới|đến|khoảng|ít nhất|tối đa|window|overlap|min_df|max_df|n_init|max_iter|pooling|JMLR|và|bỏ|chọn|thử|qua|bị|ở|bộ|catalog|chia|chiếm|chunk|chỉ|cùng|exclusivity|gain|giá|giảm|hiện|là|lên|lưu|mục|nhãn|nhất|nhật|nên|phải|ra|soạn|stability|sánh|số|thực|tiên|tra|trong|trả|tăng|tạo|tổng|đúng|đủ|whitening|ARI|Coverage|RRF|Search'
    text = re.sub(r'(?i)\b(' + words + r')(?=\d)', r'\1 ', text)
    return text


def p(text, kind='body'):
    # Only explicit **bold** markup is admitted; evidence text remains escaped.
    text = clean(text)
    safe = escape(text).replace('\n', '<br/>')
    safe = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe)
    story.append(Paragraph(safe, STYLES[kind]))
    markdown.append(str(text) + '\n')


def h(text):
    text = clean(text)
    story.append(Paragraph(escape(text), STYLES['h2'])); markdown.append('### ' + text + '\n')


def section(number, title, subtitle=None):
    title = clean(title)
    if story: story.append(PageBreak())
    story.append(Paragraph(f'{number:02d} / {escape(title)}', STYLES['h1']))
    markdown.append(f'## {number}. {title}\n')
    if subtitle: p(subtitle, 'small')


def table(headers, rows, widths=None):
    headers = [clean(c) for c in headers]
    rows = [[clean(c) for c in row] for row in rows]
    data = [[Paragraph(escape(str(c)), STYLES['headcell']) for c in headers]]
    data += [[Paragraph(escape(str(c)).replace('\n', '<br/>'), STYLES['cell']) for c in row] for row in rows]
    weights = widths or [1] * len(headers)
    t = Table(data, colWidths=[WIDTH * x / sum(weights) for x in weights], repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('LINEBELOW', (0, 0), (-1, 0), .8, TEAL),
        ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#D5E0E6')),
    ]))
    story.extend([t, Spacer(1, 9)])
    markdown.extend(['| ' + ' | '.join(map(str, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
    markdown.extend('| ' + ' | '.join(str(c).replace('\n', ' ') for c in row) + ' |' for row in rows)
    markdown.append('')


def note(text):
    text = clean(text)
    q = Paragraph(re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escape(text)).replace('\n', '<br/>'), STYLES['body'])
    box = Table([[q]], colWidths=[WIDTH])
    box.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), LIGHT),
        ('BOX', (0, 0), (-1, -1), .5, TEAL), ('LEFTPADDING', (0, 0), (-1, -1), 11),
        ('RIGHTPADDING', (0, 0), (-1, -1), 11), ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    story.extend([box, Spacer(1, 10)]); markdown.append('> ' + text + '\n')


def pipeline_drawing():
    d = Drawing(WIDTH, 285)
    def box(x, y, w, lines, fill=LIGHT):
        d.add(Rect(x, y, w, 47, rx=5, ry=5, fillColor=fill, strokeColor=TEAL, strokeWidth=.7))
        for j, text in enumerate(lines):
            d.add(String(x + w / 2, y + 30 - j * 13, text, fontName='Arial-Bold' if j == 0 else 'Arial', fontSize=8.3, textAnchor='middle', fillColor=NAVY))
    def arrow(x, y, xx, yy):
        d.add(Line(x, y, xx, yy, strokeColor=MUTED, strokeWidth=.8))
        if abs(xx-x) < 1:
            d.add(Polygon([xx, yy, xx-3, yy+5, xx+3, yy+5], fillColor=MUTED, strokeColor=None))
        else:
            sign=1 if xx>x else -1
            d.add(Polygon([xx, yy, xx-sign*5, yy+3, xx-sign*5, yy-3], fillColor=MUTED, strokeColor=None))
    box(0, 228, WIDTH, ['1. Excel nguồn → Làm sạch và sàng lọc', '20.576 bản ghi → 13.702 tài liệu chính'])
    box(0, 151, 151, ['2A. Lexical', 'Tokens / BoW / TF-IDF'])
    box(174, 151, WIDTH-174, ['2B. Transformer', 'BGE / BatteryBERT; split theo nhóm'])
    arrow(75,228,75,198); arrow(337,228,337,198)
    box(0, 75, 151, ['3A. Topic đối chứng', 'LDA; BERTopic lịch sử'])
    box(174, 75, 151, ['3B. Topic chính', 'BatteryBERT + projection', 'KMeans10 + keywords'])
    box(348, 75, WIDTH-348, ['3C. Search', 'Lexical + dense', 'Filters → RRF → Top10'])
    arrow(75,151,75,122); arrow(250,151,250,122); arrow(421,151,421,122)
    box(0, 0, WIDTH, ['4. Streamlit và đánh giá độc lập', 'Topics / Search / Comparison / Evaluation; không train khi mở app'])
    for x in (75,250,421): arrow(x,75,x,47)
    return d


def source_chart(source):
    d=Drawing(WIDTH, 190)
    left,bottom,chart_w,chart_h=40,29,WIDTH-53,132
    ceiling=2500
    for tick in range(0,2501,500):
        y=bottom+tick/ceiling*chart_h
        d.add(Line(left,y,left+chart_w,y,strokeColor=colors.HexColor('#D8E1E8'),strokeWidth=.4))
        d.add(String(left-6,y-2,str(tick),fontName='Arial',fontSize=7.5,textAnchor='end',fillColor=MUTED))
    shades=[colors.HexColor('#D69A61'),colors.HexColor('#347E9F')]
    for t in range(10):
        x=left+(t+.18)*chart_w/10; y=bottom
        for key,shade in zip(['Patent','Publication'],shades):
            height=float(source.loc[t,key])/ceiling*chart_h
            d.add(Rect(x,y,chart_w/10*.60,height,strokeColor=None,fillColor=shade));y+=height
        d.add(String(x+chart_w/10*.30,bottom-14,str(t),fontName='Arial',fontSize=8,textAnchor='middle',fillColor=NAVY))
    for x,label,shade in [(left,'Patent',shades[0]),(left+90,'Publication',shades[1])]:
        d.add(Rect(x,176,9,9,strokeColor=None,fillColor=shade))
        d.add(String(x+14,177,label,fontName='Arial',fontSize=8.5,fillColor=NAVY))
    d.add(String(WIDTH-5,177,'Số tài liệu theo topic',fontName='Arial',fontSize=8.5,textAnchor='end',fillColor=MUTED))
    return d


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.states=[]
    def showPage(self):
        self.states.append(dict(self.__dict__)); self._startPage()
    def save(self):
        total=len(self.states)
        for state in self.states:
            self.__dict__.update(state)
            self.setStrokeColor(colors.HexColor('#C5D5DF')); self.setLineWidth(.4)
            self.line(48, 39, A4[0]-48, 39)
            self.setFont('Arial', 8); self.setFillColor(MUTED)
            self.drawString(48, 25, 'SODIUM-ION RESEARCH EXPLORER | Báo cáo ngày 24/10/2026')
            self.drawRightString(A4[0]-48, 25, f'{self._pageNumber} / {total}')
            if self._pageNumber > 1:
                self.setFont('Arial', 8)
                self.drawString(48, A4[1]-31, 'PIPELINE, KẾT QUẢ VÀ ĐÁNH GIÁ HỆ THỐNG')
            super().showPage()
        super().save()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    stage1=read_json('outputs/reports/stage1_summary.json')
    final=read_json('outputs/batterybert_optimization_20261023_v1/final_test.json')
    full=read_json('outputs/batterybert_optimization_20261023_v1/full_corpus_diagnostics.json')
    checks=read_json('outputs/batterybert_optimization_20261023_v1/validation_summary.json')
    preservation=read_json('outputs/batterybert_optimization_20261023_v1/preservation_check.json')
    deploy=read_json('outputs/batterybert_optimization_20261023_v1/deployment.json')
    cv=read_csv('outputs/batterybert_optimization_20261023_v1/common_coherence.csv').set_index('model')
    val=read_csv('outputs/batterybert_optimization_20261023_v1/validation_comparison.csv')
    refined=read_csv('outputs/batterybert_optimization_20261023_v1/refined_validation_comparison.csv')
    test=read_csv('outputs/batterybert_optimization_20261023_v1/inductive_test_comparison.csv')
    sizes=read_csv('outputs/batterybert_optimization_20261023_v1/topic_distribution.csv').set_index('topic_id')
    keys=read_csv('outputs/batterybert_optimization_20261023_v1/topic_keywords.csv')
    labels=read_csv('outputs/batterybert_optimization_20261023_v1/topic_labels.csv').set_index('topic_id')
    source=read_csv('outputs/batterybert_optimization_20261023_v1/topic_source_distribution.csv').set_index('topic_id')
    review=read_csv('outputs/batterybert_optimization_20261023_v1/scientific_review.csv')
    status=read_json('outputs/research_completion_v1/evaluation_status.json')
    latency=read_csv('outputs/research_completion_v1/latency.csv')
    battery=read_json('outputs/model_comparison_v2/batterybert_pretrained/proxy_metrics.json')
    training=read_json('outputs/model_comparison_v2/batterybert_finetuned/training_success.json')
    bge_pre_val=read_json('outputs/reports/stage3/pretrained_validation_metrics.json')
    bge_pre_test=read_json('outputs/reports/stage3/pretrained_test_metrics.json')
    bge_fine_val=read_json('outputs/reports/stage3/finetuned_validation_metrics.json')
    bge_fine_test=read_json('outputs/reports/stage3/finetuned_test_metrics.json')
    selected_lda=read_json('outputs/reports/stage2/selected_model.json')
    lda_history=read_json('outputs/reports/stage2/stage2_summary.json')
    bge_history=read_json('outputs/reports/stage3/summary.json')
    chunk_history=read_json('outputs/bertopic_chunked/metrics.json')
    ai_scope=read_json('outputs/retrieval_ai_evaluation_v1/scope.json')
    ai_validation=read_json('outputs/retrieval_ai_evaluation_v1/completion_validation.json')
    synthetic=read_csv('outputs/retrieval_ai_evaluation_v1/synthetic/summary.csv')
    assert len(val)==11 and len(refined)==18 and checks['distinct_tests']==80
    assert final['technical_gate_pass'] and preservation['immutable_files']==9
    assert sizes.documents.sum()==13702 and len(keys)==100 and len(review)==60

    # 1 - Cover and executive summary.
    story.append(Spacer(1, 20))
    story.append(Paragraph('BÁO CÁO TOÀN BỘ<br/>PIPELINE CHƯƠNG TRÌNH', ParagraphStyle('cover', fontName='Arial-Bold', fontSize=27, leading=35, textColor=NAVY, spaceAfter=16)))
    p('Sodium-ion Research Explorer', 'h1')
    p('Dữ liệu - Biểu diễn - Topic modeling - Tìm kiếm - Đánh giá', 'h2')
    p('Ngày biên soạn: 24/10/2026 | Kết quả tối ưu gần nhất: 23/10/2026', 'small')
    p('Báo cáo tổng hợp implementation và artifacts đã lưu trong dự án D:/daln. Không có lần huấn luyện hay benchmark model mới trong phiên biên soạn này. Các phiên nghiên cứu lịch sử được ghi rõ phạm vi, không cộng dồn thành một thí nghiệm duy nhất.')
    table(['Corpus chính', 'Thử nghiệm phát triển', 'Kiểm thử kỹ thuật đã lưu'],
        [['13.702 tài liệu', '11 geometry + 18 keywords', '80 qua / 0 lỗi']], [1,1.2,1.2])
    h('Kết luận chính')
    p('**BatteryBERT fine-tuned + phép chiếu giảm ảnh hưởng nguồn + KMeans10** là pipeline topic chính đang được tích hợp. Lựa chọn này cải thiện sự ổn định và giảm tách nhóm theo Patent/Publication, nhưng không dẫn đầu coherence C_v.')
    p('Trên toàn corpus, hai cụm lớn nhất giảm từ92,76% ở HDBSCAN cũ xuống27,49%; source Cramér V giảm từ0,9986 xuống0,2975. Trên1.368 tài liệu test của vòng tối ưu, ARI với seed kiểm tra mới đạt0,9245. Đây là chẩn đoán kỹ thuật, không phải tỷ lệ tài liệu được phân loại đúng.')
    note('Search mặc định: TF-IDF + BatteryBERT pretrained/Qdrant → RRF k=60 → Top10. Topic chính dùng fine-tuned BatteryBERT; hai encoder này được ghi riêng. Chưa có nhãn người độc lập cho pool20 truy vấn, nên chưa xác nhận model Search tốt nhất.')
    p('Phạm vi sử dụng: báo cáo kỹ thuật/nghiên cứu và bản thảo trình bày cho giảng viên. Tên tác giả, trường, mã sinh viên và người hướng dẫn không được tự điền.', 'small')
    markdown.insert(0, '# Báo cáo toàn bộ pipeline chương trình - 24/10/2026\n')

    # 2 - Scope and architecture.
    section(1, 'Bài toán và kiến trúc tổng thể')
    p('Chương trình hỗ trợ người nghiên cứu khám phá chủ đề và tìm tài liệu về pin sodium-ion trong hai nguồn: bài báo và bằng sáng chế. Đơn vị xử lý là một bản ghi Title-Abstract kèm metadata; hệ thống không lập chỉ mục toàn văn, bảng thí nghiệm hay hình của tài liệu.')
    story.append(pipeline_drawing()); story.append(Spacer(1,10))
    markdown.append('```mermaid\nflowchart TD\n A[Excel nguồn] --> B[Làm sạch và sàng lọc]\n B --> C[13.702 tài liệu]\n C --> D[Tokens / BoW / TF-IDF]\n C --> E[BGE và BatteryBERT]\n D --> F[LDA đối chứng]\n E --> G[Topic chính: projection + KMeans10]\n D --> H[Search lexical]\n E --> I[Search dense]\n H --> J[Filters + RRF + Top10]\n I --> J\n F --> K[Streamlit]\n G --> K\n J --> K\n```\n')
    h('Hai bài toán cần đánh giá riêng')
    p('Topic modeling tạo10 cụm, mỗi cụm có top10 keywords, nhãn diễn giải và tài liệu đại diện. Retrieval trả10 tài liệu cho một truy vấn. Topic coherence cao không chứng minh kết quả tìm kiếm liên quan, còn kết quả tìm kiếm tốt không xác nhận mọi topic đều có ý nghĩa khoa học.')
    p('Các bước làm sạch, train, encoding corpus và xây index chạy offline. Khi mở web, app đọc artifacts; mỗi query chỉ xử lý bộ lọc, query embedding, retrieval và xếp hạng. Topic dùng để duyệt/lọc; không cộng topic bonus vào điểm Search.')

    # 3 - Data.
    section(2, 'Dữ liệu đầu vào và quy trình làm sạch')
    table(['Hạng mục', 'Số lượng', 'Diễn giải'], [
        ['Patent nguồn','8.657','Giữ nguyên file và provenance'],
        ['Publication nguồn','11.919','Giữ Title, Abstract, metadata'],
        ['Tổng clean master','20.576','Bảo toàn toàn bộ bản ghi nguồn'],
        ['Corpus chính','13.702','5.933 Patent + 7.769 Publication'],
        ['Đủ cặp Title-Abstract','13.650','Dùng cho split và proxy fine-tuning'],
        ['Chỉ có Title','52','Inference; không tạo Abstract giả'],
    ], [1.35,.7,2.1])
    h('Luồng quyết định')
    p('Đọc hai Excel, chuẩn hóa schema, sửa hàng bị sụp cấu trúc khi có đủ bằng chứng; chuẩn hóa Unicode/khoảng trắng và metadata; kiểm tra ngôn ngữ, thu hồi, chất lượng văn bản và trùng lặp; sàng lọc mức liên quan; sau đó tạo corpus chính. CORE/RELATED là nhãn heuristic sàng lọc, chưa phải ground truth của chuyên gia.')
    dispositions={}
    for row in stage1['dispositions']:
        dispositions[row['disposition']]=dispositions.get(row['disposition'],0)+row['count']
    table(['Disposition cuối cùng (loại trừ nhau)', 'Bản ghi'], [[k,count(v)] for k,v in dispositions.items()], [3,1])
    p('Tổng các disposition bằng20.576. Nhãn UNCERTAIN có6.598 bản ghi, nhưng RELEVANCE_REVIEW cuối cùng là6.596 vì còn các quyết định ưu tiên khác. Không lấy số nhãn heuristic trừ trực tiếp để suy ra corpus chính.', 'small')
    note('Giới hạn dữ liệu: phạm vi chính là Title-Abstract đã thu thập, không phải toàn bộ tài liệu sodium-ion đến ngày viết báo cáo. Sàng lọc tự động có thể giữ tài liệu liên quan gián tiếp hoặc bỏ sót tài liệu hữu ích cần review.')

    # 4 - Representations and splits.
    section(3, 'Tiền xử lý và chia dữ liệu')
    table(['Nhánh', 'Biểu diễn đầu vào', 'Lý do tách riêng'], [
        ['LDA/topic words','Tokens, công thức, phrase, lemmatization','Từ khóa kiểm tra được, BoW dễ diễn giải'],
        ['Transformer','Title + Abstract sẵn có','Giữ ngữ cảnh và chữ hoa theo tokenizer'],
        ['Lexical Search','Tokenizer riêng cho từ và công thức','Khớp truy vấn, TF-IDF/BM25 và required terms'],
    ], [1,1.6,1.9])
    h('Bảo toàn thông tin và lỗi đã xử lý')
    p('Giữ doc_id duy nhất, nguồn, năm, identifier, provenance và thứ tự hàng. Vector phải nối đúng doc_id của corpus. Kiểm tra shape/norm vẫn chưa đủ nếu thứ tự hàng bị đảo; vì vậy luôn lưu ordered_doc_ids và hash artifacts.')
    p('Lỗi placeholder kế thừa xuất hiện716 lần trong355 tài liệu. Nhánh representation phục hồi các marker đã biết; hàm tiền xử lý được sửa để phục hồi phrase trong từ ghép như all-solid-state. Corpus frozen không bị ghi đè. Không khẳng định đã phục hồi mọi ký hiệu hóa học bị mất trong xử lý lịch sử.')
    p('Vectorizer mới chỉ lấy ngram thực sự liền kề. Ví dụ, “hard and carbon” không được biến thành phrase “hard carbon” bằng cách bỏ “and”; cũng không tạo ngram xuyên qua ranh giới hai tài liệu.')
    table(['Tập', 'Tài liệu', 'Vai trò trong vòng tối ưu mới'], [
        ['Train','10.921','Fit phép chiếu, KMeans, vocabulary và keywords'],
        ['Validation','1.361','Chọn geometry và trọng số keywords'],
        ['Test','1.368','Cổng kiểm tra cuối sau khi đóng băng lựa chọn'],
        ['Embedding-only','52','Chỉ có Title; gán topic khi inference'],
    ], [1,.8,2.7])
    p('Split theo split_group_id bằng grouped split, nhằm hạn chế nhóm tài liệu liên quan đi vào nhiều tập. Cách nhóm không chứng minh loại bỏ mọi semantic near-duplicate hoặc toàn bộ patent family. Test đã tồn tại trong các nghiên cứu lịch sử; held-out chỉ nói đến việc không dùng để fit/chọn cấu hình trong vòng tối ưu này.')

    # 5 - Models.
    section(4, 'Các model và cơ chế học')
    table(['Thành phần', 'Vai trò', 'Đặc điểm trong dự án'], [
        ['LDA','Đối chứng topic lexical','BoW; phân phối từ-topic và topic-document'],
        ['BGE base en v1.5','Encoder đối chứng','Pretrained và fine-tuned; có query instruction riêng'],
        ['BatteryBERT cased','Encoder chuyên miền pin','MLM pretrained; native cased tokenizer; masked mean pooling'],
        ['MiniLM cross-encoder','Reranker của V2/V3','Đọc query + passage; không phải nhánh hybrid mặc định mới'],
    ], [1,1.2,2.3])
    h('LDA')
    p('LDA mô hình hóa p(w|d)=Σk θ(d,k)φ(k,w). Nhánh Stage2 thử K=10,15,20,25,30,10 passes và100 iterations; hai ứng viên tốt được thử thêm seed. Model Stage2 được chọn và bộ so sánh ép cùng K=10 là hai mục tiêu khác nhau. Bảng coherence trong báo cáo này dùng LDA K10 để so sánh10 topic; không thay kết quả Stage2 gốc bằng kết quả K10.')
    h('Contrastive fine-tuning BGE và BatteryBERT')
    p('Mỗi title là query và abstract cùng bản ghi là positive; các bản ghi khác trong microbatch tạo negatives, có mask với cặp cùng nhóm hoặc tiêu đề rất giống. Dùng loss đối xứng query→document và document→query; validation Group MRR@10 chọn checkpoint, không chọn bằng test.')
    table(['Thông số lịch sử','BGE','BatteryBERT'], [
        ['Epoch / block cập nhật','1 / block cuối','1 / block cuối'],
        ['Microbatch / accumulation','8 / 1','2 / 2'],
        ['Learning rate / temperature','2e-5 / 0,05','2e-5 / 0,05'],
        ['Ngân sách training/proxy','128 tokens','128 tokens'],
    ], [1.5,1.5,1.5])
    note('BatteryBERT đã có2.730 optimizer updates ở phiên train lịch sử; lượt tối ưu ngày23/10 không cập nhật lại trọng số transformer. Report cũ ghi chưa hoàn tất do bước kiểm tra tokenizer serialization; checkpoint và embeddings fine-tuned đã được tích hợp sau đó.')

    # 6 - Mathematics of current topic pipeline.
    section(5, 'Pipeline topic chính và công thức')
    h('Bước1. Embedding đầy đủ Abstract sẵn có')
    p('BatteryBERT tạo hidden states cho từng chunk tối đa256 tokens, overlap64. Pooling dùng trung bình có attention mask, chuẩn hóa L2 từng chunk, lấy trung bình các vector chunk rồi chuẩn hóa tài liệu. Vectors cuối có768 chiều. Bao phủ Abstract không có nghĩa đọc toàn văn bài báo.')
    note('chunk = L2[ Σt mask(t)h(t) / Σt mask(t) ]\ndocument = L2[ mean(chunk1, ..., chunkm) ]')
    h('Bước2. Giảm hướng phân biệt nguồn bằng train')
    p('Tính mean chung μ và chênh lệch mean Patent/Publication trên train; chuẩn hóa hướng chênh lệch thành u. Với mỗi embedding x, trừ mean và bỏ thành phần dọc theo u. Sau đó chuẩn hóa L2. Phép chiếu là một hướng toàn cục, không cần source của tài liệu mới khi predict; không ép hai nguồn có tỷ lệ bằng nhau trong từng cụm.')
    note('z = x - μ\nu = L2[ meanPatent - meanPublication ]\nx_projected = L2[ z - (z·u)u ]')
    h('Bước3. KMeans10')
    p('Fit KMeans trên vectors đã chiếu của train: K=10, seed42, n_init10, max_iter300. Validation và test được predict bằng model đã fit; không refit toàn corpus sau test. KMeans gán mọi tài liệu nên0% outlier là thuộc tính thuật toán, không chứng minh corpus sạch hoàn toàn.')
    h('Bước4. Từ khóa đặc trưng có bằng chứng')
    p('Vocabulary fit trên train: unigram/bigram, min_df5, max_df0,6, tối đa40.000 features. Cấu hình cuối dùng TF power1, IDF và exclusivity power0,25. df(c,w) là số tài liệu của topic c chứa w; df(w) là số tài liệu train chứa w.')
    note('weight(c,w) = TF(c,w) × log[1 + N / df(w)] × [(df(c,w)+1)/(df(w)+10)]^0,25')
    p('Mỗi keyword có ít nhất5 tài liệu train hỗ trợ. Các bản sao phrase chứa nhau bị hạn chế để top10 không chỉ là lặp một cụm từ. Đây là trọng số topic words tự thiết kế, không phải xác suất relevance hay mô hình BERTopic chuẩn được giữ nguyên toàn bộ.', 'small')

    # 7 - Experimental search.
    section(6, 'Thiết kế và diễn biến tối ưu')
    p('Vòng1 kiểm tra11 cấu hình geometry/encoder; mỗi cấu hình fit seed42 và43 trên cùng train. Mục tiêu là phân biệt tác động của pretrained/fine-tuned encoder, centering, bỏ3 principal components, projection nguồn và whitening64 chiều.')
    rows=[]
    for r in val.itertuples():
        rows.append([r.candidate.replace('pretrained','pre').replace('finetuned','fine'),fmt(r.source_cramers_v,3),fmt(r.stability_ari,3),fmt(r.document_npmi,3)])
    table(['Cấu hình validation','Source V','ARI42/43','NPMI'], rows, [2.6,.8,.85,.8])
    p('Centering giúp ổn định nhưng vẫn lệch nguồn; whitening đơn thuần không giải quyết nguồn. Fine-tuned/raw cũng còn source V≈0,99. Fine-tuned/source giảm V còn0,310 và ARI đạt0,792, nhưng cách lấy từ khóa ban đầu cho NPMI−0,303.')
    h('Vòng2: representation trên các cụm cố định')
    p('Thử18 biến thể: TF power∈{0,5;0,75;1} và exclusivity power∈{0;0,25;0,5} trên hai geometry fine-tuned có lệch nguồn thấp. Có2 cấu hình là tái kiểm tra cách trích từ khóa vòng1; tổng29 lượt cấu hình, không phải29 encoder neural.')
    p('Chọn fine-tuned/source, TF power1 và exclusivity0,25. Validation NPMI=0,0839, diversity=0,80; memberships không thay trong bước chọn từ khóa. Cấu hình NPMI cao hơn nhưng diversity dưới ngưỡng bị loại. Không sửa nhãn tài liệu để tạo ra10 cụm đẹp.')

    # 8 - Heldout results.
    section(7, 'Điều kiện dừng và kết quả test')
    p('Các ngưỡng được định trước vòng1 và giữ nguyên vòng2. Chúng phục vụ chấp nhận kỹ thuật cho ứng dụng cụ thể, không phải chuẩn đánh giá khoa học phổ quát. Sau khi chọn cấu hình bằng validation mới chạy test và seed137.')
    table(['Chẩn đoán trên1.368 test','Kết quả','Ngưỡng'], [
        ['Số topic',str(final['topics']),'10'],
        ['Cụm lớn nhất',pct(final['largest_topic_share']),'≤30%'],
        ['Hai cụm lớn nhất',pct(final['top2_topic_share']),'≤45%'],
        ['Cụm nhỏ nhất',pct(final['smallest_topic_share']),'≥0,5%'],
        ['Source Cramér V',fmt(final['source_cramers_v']),'≤0,5'],
        ['ARI với seed137',fmt(final['stability_ari']),'≥0,6'],
        ['Keyword diversity@10',fmt(final['keyword_diversity'],2),'≥0,75'],
        ['Keyword support coverage','100%','≥95%'],
        ['Document NPMI',fmt(final['document_npmi']),'≥0'],
    ], [2.3,1,1])
    h('Ý nghĩa chính xác của các chỉ số')
    p('Cramér V đo liên hệ giữa topic và nguồn; giảm liên hệ giúp phát hiện bớt cụm văn phong, nhưng liên hệ bằng0 không tự chứng minh tốt vì hai nguồn có thể khác nội dung thật. ARI đo agreement giữa các lần fit với seed khác, không phải tỷ lệ đúng. Diversity là số keywords khác nhau chia100 slots. Coverage100% nghĩa là mỗi keyword xuất hiện ít nhất một lần trong tập test, không phải mọi test document đều có keyword.')
    p('NPMI đo đồng xuất hiện các cặp từ theo tài liệu; cặp không đồng xuất hiện được chấm−1. Báo cáo lấy trung bình qua topic. Công thức vật liệu thay thế nhau có thể ít đồng xuất hiện, nên NPMI cần được đọc cùng khoa học và đại diện tài liệu.')
    table(['Đối chứng KMeans train-only trên test','NPMI','Source V'], [[r.model,fmt(r.document_npmi),fmt(r.source_cramers_v)] for r in test.itertuples()], [2.7,1,1])
    p('Bản cuối thay cả geometry và cách trích từ khóa; bảng trên là so sánh pipeline, không phải encoder-only ablation. Không có confidence interval hay chuyên gia-label để kết luận chênh lệch này là độ chính xác khoa học.', 'small')

    # 9 - Coherence and balance.
    section(8, 'So sánh10 topic và đánh đổi chất lượng')
    ids=['LDA_K10_historical','BGE_BERTopic_historical','BatteryBERT_HDBSCAN_historical','BatteryBERT_KMeans_web','BatteryBERT_optimized']
    names=['LDA K10','BERTopic + BGE lịch sử','BatteryBERT/HDBSCAN cũ','BatteryBERT/KMeans web trước','BatteryBERT topic tối ưu mới']
    table(['Pipeline','C_v','U_Mass','Diversity'], [[name,fmt(cv.loc[i,'c_v']),fmt(cv.loc[i,'u_mass']),fmt(cv.loc[i,'keyword_diversity'],2)] for i,name in zip(ids,names)], [2.45,.7,.85,.8])
    p('Cùng reference toàn corpus, repaired tokens/phrase thật, window110, top10, một process; mỗi model đều đủ10/10 topic có support. Đây là số đo mô tả sau lựa chọn; không dùng để chỉnh lại model theo test.')
    p('BatteryBERT/HDBSCAN cũ có C_v cao nhất nhưng top2 chiếm92,76% và source V=0,9986. Bản web KMeans trước đã giảm mất cân bằng kích thước nhưng vẫn gần như tách theo nguồn. Bản mới giảm source V toàn corpus xuống0,2975; topic lớn nhất15,06%, top2=27,49%.')
    story.append(source_chart(source)); story.append(Spacer(1,8))
    markdown.append('![Phân bố topic và so sánh top2](../outputs/batterybert_optimization_20261023_v1/topic_balance.png)\n')
    note('Đánh đổi cần báo cáo: C_v giảm0,6941→0,5601 so với web trước, diversity giảm0,81→0,80. Lựa chọn bản mới dựa trên cổng đa tiêu chí, không dựa trên việc thắng mọi chỉ số. Giảm kích thước cụm và giảm lệch nguồn chưa đủ để gọi là phân loại chủ đề khoa học chính xác.')

    # 10,11 - All ten topics with actual top10 words.
    for start in (0,5):
        section(9 if start==0 else 10, f'Danh mục10 topic: nhóm{start}-{start+4}')
        p('Nhãn là diễn giải tạm thời từ keywords và tài liệu đại diện. Topic ID chỉ có ý nghĩa trong pipeline này; không ghép ID với model cũ. Số lượng là toàn bộ corpus13.702 tài liệu.', 'small')
        for t in range(start,start+5):
            h(f'Topic {t}. {labels.loc[t,"topic_label"]}')
            p(f'{count(sizes.loc[t,"documents"])} tài liệu ({pct(sizes.loc[t,"corpus_share"])} corpus) | Patent: {count(source.loc[t,"Patent"])} | Publication: {count(source.loc[t,"Publication"])}', 'small')
            p('Top10: ' + '; '.join(keys[keys.topic_id.eq(t)].sort_values('rank').token) + '.')
        if start==0:
            note('Topic3 là nhóm phần cứng/lắp ráp pin có thật trong corpus. Tỷ lệ Patent cao không tự động là lỗi; cần phân biệt khác biệt nội dung thật với ảnh hưởng văn phong. Không đổi tên nhóm này thành một họ vật liệu giả để đủ10 nhóm vật liệu.')
        else:
            note('Các nhóm0/5/6 có phần giao thoa giữa chalcogenide, chuyển hóa và hợp kim. Nhóm8 còn có cathode oxide bị gán lệch; nhóm7 đôi lúc chứa carbon coating cathode thay vì hard-carbon anode. Vì vậy nhãn mô tả trọng tâm, không bao phủ chính xác mọi thành viên.')

    # 12 - Search pipeline.
    section(11, 'Pipeline tìm kiếm và luồng giao diện')
    p('Mỗi request nhận query tiếng Anh, source, năm, required terms và topic tùy chọn. Bộ lọc tạo tập doc_id đủ điều kiện trước khi rank ở cả hai nhánh. Required terms chỉ kiểm tra bằng chứng từ/công thức, không xác minh quan hệ hóa học hoặc claim khoa học.')
    table(['Phương pháp','Luồng xử lý','Vai trò'], [
        ['TFIDF_COSINE','Full Title-Abstract TF-IDF → cosine','Baseline lexical'],
        ['BATTERYBERT_SEMANTIC','Query pretrained BatteryBERT → Qdrant cosine','Đối chứng semantic'],
        ['BATTERYBERT_HYBRID','TF-IDF Top100 + dense Top100 → RRF60 → Top10','Mặc định hiện tại'],
        ['BGE_SEMANTIC','BGE query protocol → vectors pretrained có sẵn','Đối chứng encoder'],
        ['SEARCH_V2 lịch sử','BM25 + BGE → RRF → MiniLM rerank','Giữ để so sánh'],
        ['SEARCH_V3 lịch sử','BM25 + fine-tuned BatteryBERT/Qdrant → RRF → MiniLM','Giữ để so sánh'],
    ], [1.4,2.1,1])
    h('Công thức hybrid')
    note('RRF(d) = Σnhánh 1 / [60 + ranknhánh(d)]\nTài liệu không có trong Top100 của một nhánh không được cộng điểm của nhánh đó.')
    p('RRF hợp nhất thứ hạng, không cộng trực tiếp cosine và TF-IDF. Điểm RRF/cosine/MiniLM khác thang đo và không phải xác suất phù hợp. Một kết quả lexical ngoài dense Top100 vẫn có thể được hiển thị cosine thật, nhưng không được coi như có dense candidate rank.')
    h('Tích hợp topic mới')
    p('Topics và bộ lọc Search dùng cùng catalog mới. Sau retrieval, nhãn hiện hành được gắn từ doc_id; payload topic cũ trong Qdrant không được coi là nhãn chính. Không rebuild/upsert vector database để đổi nhãn catalog. Query encoder của Search mặc định vẫn pretrained, còn encoder topic mới là fine-tuned; không trộn hai không gian vector.')
    p('Trang Search hiển thị metadata/Abstract/điểm và xuất CSV; Topics hiển thị keywords, phân bố nguồn/năm, đại diện; Comparison đặt các phương pháp cạnh nhau; Evaluation quản lý nhãn và độ phủ. Passage trích theo lexical overlap là đoạn minh họa, không phải lời giải thích neural về relevance.', 'small')

    # 13 - Retrieval evidence.
    section(12, 'Kết quả retrieval và giới hạn đánh giá')
    h('Proxy Title→Abstract lịch sử')
    table(['Encoder','Validation MRR@10','Test MRR@10'], [
        ['BGE pretrained',fmt(bge_pre_val['group_mrr_at_10']),fmt(bge_pre_test['group_mrr_at_10'])],
        ['BGE fine-tuned',fmt(bge_fine_val['group_mrr_at_10']),fmt(bge_fine_test['group_mrr_at_10'])],
        ['BatteryBERT pretrained',fmt(battery['validation']['group_mrr_at_10']),fmt(battery['test']['group_mrr_at_10'])],
        ['BatteryBERT fine-tuned',fmt(training['best_score']),'Chưa có test proxy tương đương'],
    ], [1.65,1.2,1.65])
    p('Proxy tìm abstract cùng split/cùng group bằng title; không phải bài toán người dùng gõ query trên toàn corpus. Các nhánh khác instruction, pooling và chunking/ngân sách, nên đây không phải encoder-only benchmark đồng nhất. Không so validation của model này với test của model khác để kết luận thắng tuyệt đối.')
    h('Đánh giá Search Top10 theo query')
    p(f'Đã lưu20 truy vấn,80 requests của4 phương pháp và800 hàng xếp hạng. Pool chung có{status["pool_pairs"]} cặp query-document. Nhãn người được chấp nhận: {status["accepted_human_labels"]}; query đủ pool: {status["eligible_queries"]}/20. Trạng thái: PENDING_HUMAN_ANNOTATION.')
    p('Thang nhãn0: không liên quan;1: một phần;2: trực tiếp. P@10 và MRR@10 dùng grade≥2; nDCG dùng gain2^grade−1 và IDCG từ pool đầy đủ. Query chỉ đủ điều kiện khi toàn pool được chấm và xác minh. Nhãn trống không phải0; không báo Recall toàn corpus vì chưa biết hết tài liệu relevant.')
    h('Độ trễ warm CPU đã lưu')
    table(['Phương pháp','Mean (giây)','Median','Số request'], [[r.method,fmt(r.mean,3),fmt(r.median,3),str(r.count)] for r in latency.itertuples()], [2.3,.8,.8,.8])
    p('Bỏ lượt đầu mỗi phương pháp; chạy tuần tự theo phương pháp, chưa random hóa thứ tự và chưa đo đa người dùng. Bảng là số đo lịch sử trước khi thay catalog topic; không phải benchmark chạy lại ngày24/10 hay bằng chứng latency production.', 'small')

    # 14 - Scientific error analysis.
    section(13, 'Tự đánh giá nội dung và phân tích lỗi')
    p('Agent đã xem title và đoạn đầu Abstract của60 tài liệu:3 gần centroid và3 random có seed mỗi topic. Đoạn central được xem tới450 ký tự, random tới800; Abstract đầy đủ vẫn được lưu. Đây không phải đọc toàn văn, không có chuyên gia độc lập và không phải mẫu đủ để ước lượng accuracy toàn corpus.')
    table(['Ca thực tế','Quan sát','Nhận xét'], [
        ['WOS:000627423800016\nTopic0','MoS2 trên carbon nanotubes, sodium storage','Phù hợp trọng tâm dichalcogenide anode'],
        ['WOS:000600805400001\nTopic1','Na3V2(PO4)3 cathode, đồng doping K-Cl','Phù hợp phosphate vanadium/NASICON'],
        ['patent:202235650E\nTopic2','Carbon-coated sodium iron phosphate','Nhóm lân cận; không phải Prussian blue'],
        ['WOS:000989869100001\nTopic4','PDI-bolaamphiphile organic anode','Lệch khỏi trọng tâm electrolyte'],
        ['patent:2017766545\nTopic7','Porous carbon coating cathode particles','Carbon processing gần nhau nhưng khác vai trò điện cực'],
        ['WOS:000718314900003\nTopic8','NaCrO2/C cathode','Cần review với nhóm oxide lớp'],
        ['patent:202386521F\nTopic3','Quy trình homogenizing anode material','Workflow tổng hợp, không phải cấu trúc cell'],
    ], [1.45,1.6,1.6])
    h('Vì sao vẫn còn từ chung chung?')
    p('Các từ solution, dry, connect có thể thật sự phổ biến trong quy trình tổng hợp hoặc thiết bị. Xóa bằng blacklist chỉ làm đẹp tên nhóm mà không sửa lỗi gán thành viên. Từ khóa cần vừa có support, vừa đặc trưng, vừa được đọc trong ngữ cảnh; nhãn tự đặt không được dùng như ground truth cho chính model.')
    note('Kết luận khoa học hiện tại: có bằng chứng định tính rằng trọng tâm nhiều nhóm rõ hơn, nhưng vẫn có misplacement, polarity ambiguity và overlap. Chưa có expert-labeled precision/recall/F1 cho topic. Không chuyển ARI0,9245 thành “đúng92,45%”.')

    # 15 - Technical verification.
    section(14, 'Kiểm thử, tái lập và truy vết')
    table(['Nhóm kiểm tra đã lưu','Kết quả','Nội dung'], [
        ['Backend / dữ liệu / UI topic','77 qua','Predict đủ corpus; ID train;100 keywords; filters; semantics'],
        ['Web với truy vấn thật','3 qua','TF-IDF, V3, hybrid và topic filter'],
        ['Tổng riêng biệt','80 qua /0 lỗi','Không cộng trùng lượt unit đã chạy lại'],
        ['HTTP smoke','PASS /200','Khởi động Streamlit; dừng process kiểm tra'],
        ['Frozen inputs','9 file /0 đổi','Corpus, split, vectors và ordered IDs'],
    ], [1.5,.95,2.1])
    p('Đây là kết quả kiểm thử đã lưu từ phiên tối ưu trước. Trong phiên biên soạn24/10, chỉ đối chiếu artifacts và kiểm tra bố cục PDF; không gọi các số80 là vừa chạy lại hôm nay. Cảnh báo exact-search của Qdrant Local không phải failure; một process sở hữu local storage.')
    h('Artifacts cần giữ')
    table(['Đầu ra','Ý nghĩa'], [
        ['topic_pipeline.joblib','Geometry + KMeans đã fit; predict từ vectors đúng checkpoint'],
        ['topic_assignments / keywords / labels','Catalog hiện hành; nhãn còn tạm thời'],
        ['validation_comparison / refined_validation_comparison','11 geometry và18 lượt representation'],
        ['final_test.json / common_coherence.csv','Cổng cuối và so sánh mô tả chung reference'],
        ['scientific_review.csv','Ví dụ đúng/lệch nhóm; review agent'],
        ['inputs / current_code_manifest / code_snapshot','Fingerprint, phiên code và nguồn thực nghiệm'],
    ], [2.1,2.5])
    h('Cách chạy và tái lập')
    p('.venv/Scripts/python.exe run_demo.py\nMở Topics → BatteryBERT fine-tuned · optimized.\n\nThí nghiệm mới cần output mới:\n.venv/Scripts/python.exe scripts/optimize_batterybert.py\n  --output outputs/batterybert_optimization_REPLAY --phase all\n.venv/Scripts/python.exe scripts/evaluate_topic_candidates.py\n  --output outputs/batterybert_optimization_REPLAY', 'small')
    p('Predict phải dùng embedding BatteryBERT fine-tuned đúng checkpoint và pooling256/64. Không thay bằng pretrained/BGE vectors dù cùng768 chiều. Fingerprint cố ý từ chối resume trên code/input khác trong thư mục frozen.', 'small')

    # 16 - Assessment and recommendations.
    section(15, 'Đánh giá tổng hợp và hướng cải thiện')
    table(['Khía cạnh','Đánh giá hiện tại','Việc cần tiếp theo'], [
        ['Dữ liệu và truy vết','Có pipeline và hash/ID rõ ràng; scope vẫn heuristic','Rà soát ngoài phạm vi, family trùng và polarity'],
        ['Topic chính','Cân hơn, ít lệch nguồn, seed ổn định','Chấm nhãn khoa học và phân tích overlap'],
        ['Keywords','Có support và phrase thật; còn từ quy trình','Review tính đặc trưng theo từng chủ đề'],
        ['Search Top10','Luồng hoạt động và filters đã kiểm thử','Hoàn thành pool20 query bằng nhãn người'],
        ['Tốc độ/vận hành','Có CPU measurements và local app','Đo cold/warm, concurrency, lỗi khóa storage'],
    ], [1,2,2])
    h('Model nào tốt nhất?')
    p('Nếu chỉ xét C_v, BatteryBERT/HDBSCAN cũ dẫn đầu nhưng collapse theo nguồn nên không thích hợp để gọi là10 chủ đề khoa học tốt nhất. Nếu ưu tiên10 nhóm bớt lệch nguồn và ổn định theo cổng hiện có, pipeline BatteryBERT mới là lựa chọn triển khai tạm thời. Nếu xét proxy test Title→Abstract đã hoàn tất, BGE có kết quả cao hơn; Search query thật chưa có nhãn để kết luận.')
    h('Lộ trình cải thiện theo thứ tự ưu tiên')
    p('**1. Xây tập đánh giá độc lập.** Chuyên gia chấm topic/ngoài phạm vi, gồm trường hợp giao thoa; chấm relevance theo query riêng. Khóa một tập mới trước các vòng tuning tiếp để tránh tối ưu dần vào test cũ.')
    p('**2. Điều chỉnh mục tiêu học.** Tạo positive cùng chủ đề giữa Patent và Publication, negatives khác họ vật liệu, mask false negatives. Giữ phần cứng thật trong dữ liệu thay vì buộc mọi topic phải là vật liệu.')
    p('**3. Fine-tune có kiểm soát.** Khi có tài nguyên phù hợp, so sánh batch thực16-32 hoặc cached contrastive loss; gradient accumulation không tăng số negatives trong cùng loss. So sánh objective bao phủ Abstract/chunk pooling với128 tokens, cập nhật1-2 block hoặc adapter và chọn checkpoint bằng validation đúng mục tiêu.')
    p('**4. Đánh giá lại đồng nhất.** Giữ LDA/BGE làm đối chứng, kiểm soát pooling/context khi muốn kết luận encoder-only, báo lỗi theo nguồn và nhóm vật liệu. Không ép BatteryBERT thắng bằng thay dữ liệu, nhãn hoặc reference coherence.')

    # 17 - Historical studies, deliberately separated from current results.
    section(16, 'Phụ lục: các kết quả nghiên cứu lịch sử')
    p('Các thí nghiệm sau thuộc giai đoạn nghiên cứu trước khi chốt catalog10 topic hiện hành. Chúng giúp hiểu tiến trình phát triển, không phải phép so đồng nhất trên cùng K/reference và không được trộn vào bảng đánh giá Search0-2 hiện tại.')
    table(['Nhánh lịch sử','Topic','C_v gốc','Outlier / giới hạn'], [
        ['LDA được Stage2 chọn',str(selected_lda['selected_k']),fmt(lda_history['selected_metrics']['c_v']),'Stability keyword matching='+fmt(lda_history['selected_metrics']['stability'],3)],
        ['BERTopic BGE /128 tokens',str(bge_history['bertopic']['topics']),fmt(bge_history['bertopic']['c_v']),pct(bge_history['bertopic']['outlier_ratio'])],
        ['BERTopic BGE chunk256/64',str(chunk_history['topics']),fmt(chunk_history['c_v']),pct(chunk_history['outlier_ratio'])],
    ], [2, .5, .75, 1.7])
    p('Stage2 chọn LDA K20 bằng chính sách kết hợp coherence, diversity và stability trên các ứng viên; stability0,262 là phép matching từ khóa giữa seeds, không phải ARI của model mới. Chunking BGE tăng C_v từ0,4628 lên0,4779 nhưng outlier tăng20,08%→22,77%. Vì nhiều yếu tố context/pooling thay đổi, không kết luận chunking thắng tổng thể. Các C_v gốc này khác reference của bảng so sánh10 topic đã tính lại.')
    h('AI judge pilot lịch sử')
    p(f'Pilot có{ai_scope["domain_queries"]} truy vấn theo protocol riêng; chỉ{len(ai_scope["complete_query_pools"])} pool đủ: '+', '.join(ai_scope['complete_query_pools'])+'. Nhãn là GEMINI_LLM_JUDGE, expert_verified=false. Các pool chưa đủ bị loại chung cho tất cả phương pháp; không xem nhãn thiếu là0. Không gộp pilot này với pool20 query cần nhãn người hiện tại.')
    h('Synthetic source-ID recovery lịch sử')
    wanted=['DENSE_PRETRAINED','DENSE_FINETUNED','DENSE_CHUNKED']
    table(['Nhánh dense lịch sử','Hit@10','MRR@10'], [[r.method,fmt(r.Hit_at_10),fmt(r.MRR_at_10)] for r in synthetic[synthetic.method.isin(wanted)].itertuples()], [2.4,1,1])
    p(f'Đánh giá{ai_validation["synthetic_queries_evaluated"]}/200 query và{ai_validation["synthetic_methods"]} methods. Mục tiêu là lấy lại đúng source doc_id đã dùng sinh query; đây không phải corpus Recall@10 hay relevance theo người dùng. Có163 source thuộc train cũ,16 validation,18 test; chỉ18 source test nên chưa đủ để khẳng định khái quát ra query mới. Không coi synthetic Hit@10 cao là xác nhận encoder tốt nhất.')

    # 18 - Conclusion and sources.
    section(17, 'Kết luận và nguồn bằng chứng')
    p('Chương trình đã có pipeline end-to-end cho dữ liệu, topic modeling, tìm kiếm và trình bày kết quả trên13.702 tài liệu. Tối ưu vừa qua giải quyết đáng kể mất cân bằng và ảnh hưởng nguồn trong topic BatteryBERT, đồng thời sửa lỗi representation và đồng nhất catalog giữa Search/Topics.')
    p('Giá trị đã chứng minh là tính hoạt động, khả năng truy vết, độ ổn định và cải thiện các chẩn đoán phát triển. Chưa chứng minh được độ đúng khoa học của toàn bộ10 topic, chất lượng relevance Search vượt các model khác hoặc hiệu năng production. Hệ thống phù hợp làm công cụ hỗ trợ khám phá và tạo danh sách tài liệu để đọc lại bằng chứng.')
    note('Đề xuất trình bày với giảng viên: “BatteryBERT là encoder chính theo yêu cầu; lựa chọn cuối là pipeline đa thành phần đã kiểm định. Kết quả tốt hơn về lệch nguồn và độ ổn định nhưng giảm coherence, còn cần chuyên gia đánh giá chủ đề và relevance.”')
    h('Nguồn nội bộ chính')
    p('outputs/reports/stage1_summary.json; outputs/reports/stage2/selected_model.json; outputs/reports/stage3/*_metrics.json; outputs/model_comparison_v2/batterybert_pretrained/proxy_metrics.json và batterybert_finetuned/training_success.json; outputs/research_completion_v1/evaluation_status.json, latency.csv; outputs/batterybert_optimization_20261023_v1/*.', 'small')
    p('Tệp evidence_manifest_20261024.json đi kèm ghi danh sách nguồn đã dùng và SHA-256. Các số liệu trong bản PDF được đọc từ artifacts; những đoạn đánh giá và lộ trình tương lai là nhận xét, không phải thí nghiệm mới.', 'small')
    h('Tài liệu phương pháp đã đối chiếu')
    references=[
        ('[1] Blei, Ng & Jordan (2003). Latent Dirichlet Allocation. JMLR3,993-1022.', 'https://www.jmlr.org/papers/v3/blei03a.html'),
        ('[2] BERTopic documentation. Clustering: HDBSCAN, KMeans và mô hình thay thế.', 'https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html'),
        ('[3] Huang et al. (2021). WhiteningBERT. Findings of EMNLP.', 'https://aclanthology.org/2021.findings-emnlp.23/'),
        ('[4] BatteryData. Model card batterybert-cased; checkpoint dự án được pin riêng.', 'https://huggingface.co/batterydata/batterybert-cased'),
    ]
    for title,url in references:
        p(title,'small')
        story.append(Paragraph(f'<link href="{url}" color="#157F83">{escape(url)}</link>',STYLES['small']))
        markdown.append(url+'\n')
    p('Ngày đối chiếu phương pháp:24/10/2026. Trang model card có cách đặt tiêu đề cased/uncased chưa đồng nhất; identity thực nghiệm phải dựa vào model_id, revision và hash đã lưu, không dựa riêng vào tên hiển thị của website.', 'small')

    # Export.
    target=OUT/(NAME+'.pdf')
    doc=SimpleDocTemplate(str(target), pagesize=A4, leftMargin=48, rightMargin=48,
        topMargin=54, bottomMargin=51, title='Báo cáo toàn bộ pipeline Sodium-ion Research Explorer', author='Sodium-ion Research Explorer',
        subject='Pipeline, kết quả thực nghiệm, đánh giá và giới hạn; snapshot23/10/2026')
    doc.build(story,canvasmaker=NumberedCanvas)
    # Add a real outline so long reports can be navigated in a PDF reader.
    import fitz
    pdf=fitz.open(target)
    outline=[[1,'Tóm tắt điều hành',1]]
    for i,page in enumerate(pdf):
        first=page.get_text().splitlines()[0]
        if re.match(r'^\d{2} / ',first):outline.append([1,first,i+1])
    pdf.set_toc(outline)
    temporary=target.with_suffix('.outline.pdf')
    pdf.save(temporary,garbage=4,deflate=True);pdf.close();temporary.replace(target)
    md=ROOT/'docs'/(NAME+'.md'); md.write_text('\n'.join(markdown),encoding='utf8')
    provenance={}
    for file in sorted(set(SOURCES)):
        provenance[file.relative_to(ROOT).as_posix()]={
            'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
    (OUT/'evidence_manifest_20261024.json').write_text(json.dumps(dict(report_date='2026-10-24',
        result_snapshot='2026-10-23 and identified earlier runs', model_runs_performed_during_authoring=False,
        pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), sources=provenance),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(pdf=str(target),markdown=str(md),sources=len(provenance)),ensure_ascii=False))


if __name__=='__main__': build()
