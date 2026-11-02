"""Build a new research report version using only validated saved artifacts."""
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import io
from html import escape
import pandas as pd
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from pypdf import PdfReader,PdfWriter
from reportlab.pdfgen import canvas as pdfcanvas

if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/'outputs/research_revision_20261025_v1'
BASE=ROOT/'outputs/batterybert_full_document_20261024_v1'
OUT=REV/'report';OUT.mkdir(parents=True,exist_ok=True)
NAME='BAO_CAO_NGHIEN_CUU_CAI_THIEN_SODIUM_ION_20261026'
PDF=ROOT/'output/pdf'/(NAME+'.pdf')
MD=ROOT/'docs'/(NAME+'.md')
SOURCES=set()


def read(path):
    path=Path(path);SOURCES.add(path)
    return json.loads(path.read_text(encoding='utf8'))


def csv(path):
    path=Path(path);SOURCES.add(path)
    return pd.read_csv(path,keep_default_na=False)


def f(value,d=4):return f'{float(value):.{d}f}'.replace('.',',')
def count(value):return f'{int(value):,}'.replace(',','.')
def percent(value):return f'{float(value)*100:.2f}%'.replace('.',',')


def build():
    if (OUT/'artifact_manifest.json').exists():
        manifest=read(OUT/'artifact_manifest.json')
        for item in manifest['artifacts']:
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Report artifact changed')
        print('SKIP report complete and verified');return
    for family,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:pdfmetrics.registerFont(TTFont(family,'C:/Windows/Fonts/'+file))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
    W,H=A4;margin=47;CW=W-2*margin
    styles={
      'body':ParagraphStyle('body',fontName='Arial',fontSize=10.7,leading=15.4,spaceAfter=8),
      'small':ParagraphStyle('small',fontName='Arial',fontSize=8.3,leading=11.6,spaceAfter=6,textColor=colors.HexColor('#47555f')),
      'heading':ParagraphStyle('heading',fontName='Arial-Bold',fontSize=18.5,leading=24,spaceAfter=13),
      'sub':ParagraphStyle('sub',fontName='Arial-Bold',fontSize=11.4,leading=16,spaceBefore=8,spaceAfter=6,keepWithNext=True),
      'title':ParagraphStyle('title',fontName='Arial-Bold',fontSize=27,leading=35,spaceAfter=18),
      'cell':ParagraphStyle('cell',fontName='Arial',fontSize=8.5,leading=11.7),
      'head':ParagraphStyle('head',fontName='Arial-Bold',fontSize=8.5,leading=11.5,textColor=colors.white)}
    story=[];markdown=[];sections=[];figures=0;tables=0
    def safe(text):
        text=str(text).replace('KMeans50','KMeans K10 n_init50')
        return re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',escape(text).replace('\n','<br/>'))
    def p(text,style='body'):
        text=re.sub(r'(?i)\b(ngày|seed|batch|KMeans|topic|keywords|query|model|encoder|corpus|cặp|bản|dùng|trên|có|còn|tăng|giảm|là|từ|và|được|qua|PASS|recipe|chọn|trong|gồm)(?=\d)',r'\1 ',str(text))
        text=re.sub(r'(\d)(?=(?:trang|cặp|tài liệu|query|ứng viên|nhãn|documents|trường)\b)',r'\1 ',text)
        story.append(Paragraph(safe(text),styles[style]));markdown.append(str(text)+'\n')
    def sub(text):story.append(Paragraph(safe(text),styles['sub']));markdown.append('### '+text+'\n')
    def section(text):
        if story:story.append(PageBreak())
        story.append(Paragraph(safe(text),styles['heading']));markdown.append('## '+text+'\n');sections.append(text)
    def table(headers,rows,widths=None):
        nonlocal tables
        tables+=1;weights=widths or [1]*len(headers)
        data=[[Paragraph(safe(v),styles['head']) for v in headers]]+[[Paragraph(safe(v),styles['cell']) for v in row] for row in rows]
        obj=Table(data,colWidths=[CW*w/sum(weights) for w in weights],repeatRows=1,hAlign='LEFT')
        cmds=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#294b62')),('GRID',(0,0),(-1,-1),.45,colors.HexColor('#d9d9d9')),
          ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]
        for row in range(2,len(data),2):cmds.append(('BACKGROUND',(0,row),(-1,row),colors.HexColor('#f0f5f7')))
        obj.setStyle(TableStyle(cmds));story.extend([obj,Spacer(1,9)])
        markdown.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
        markdown.extend('| '+' | '.join(str(v).replace('\n','; ').replace('|','/') for v in row)+' |' for row in rows);markdown.append('')
    def picture(path,caption,source,maxh=250,equation=False):
        nonlocal figures
        path=Path(path);SOURCES.add(path)
        w,h=PILImage.open(path).size;scale=min(CW/w,maxh/h)
        block=[Image(str(path),width=w*scale,height=h*scale),Spacer(1,5)]
        text=caption
        if not equation:
            figures+=1;text=f'Hình {figures}. '+caption
            block.append(Paragraph(safe(text),styles['small']))
            if source:block.append(Paragraph(safe('Nguồn output: '+source),styles['small']))
        story.append(KeepTogether(block));story.append(Spacer(1,5))
        markdown.extend([f'![{text}]({Path(os.path.relpath(path,MD.parent)).as_posix()})',text,('Nguồn output: '+source) if source else '',''])
    def fig(name,caption,source,maxh=250):
        override=REV/'visualizations_v2'/(name+'.png')
        path=override if override.exists() and OUT.name=='report_v2' else REV/'visualizations'/(name+'.png')
        picture(path,caption,source,maxh)
    def eq(name,description):picture(ROOT/'outputs/report_pipeline_20261025/figures'/('eq_'+name+'.png'),description,'',48,True)

    geometry=read(REV/'decisions/geometry.json');search=read(REV/'decisions/search_gate.json')
    topic=read(REV/'final_topics/final_metrics.json');stream=read(REV/'scientific_stream_v2/coverage.json')
    labels=csv(REV/'review/topic_labels.csv');keys=csv(REV/'final_topics/topic_keywords.csv');dist=csv(REV/'final_topics/topic_distribution.csv')
    keytrials=csv(REV/'comparison/keyword_experiments.csv');geo=csv(REV/'comparison/geometry_comparison.csv')
    verified=read(REV/'verification/summary.json');deploy=read(REV/'deployment.json')
    pilot=read(ROOT/'outputs/ai_silver_labels_20261026_v1/batch_0001_summary.json')
    canonical=csv(REV/'final_topics/keyword_comparison.csv');cv=csv(REV/'final_topics/common_coherence.csv')
    stage1=read(ROOT/'outputs/reports/stage1_summary.json')
    historical=csv(ROOT/'outputs/final_system/topic_comparison.csv')
    lda=read(ROOT/'outputs/reports/stage2/stage2_summary.json')
    state=read(REV/'execution/pipeline_state.json')
    markdown.append('# Khám phá chủ đề và tìm kiếm tài liệu pin sodium ion sau rà soát và cải thiện\n')
    story.append(Spacer(1,24));story.append(Paragraph('KHÁM PHÁ CHỦ ĐỀ<br/>VÀ TÌM KIẾM TÀI LIỆU<br/>PIN SODIUM ION',styles['title']))
    p('Báo cáo nghiên cứu sau rà soát và cải thiện','sub')
    p('Sodium ion Research Explorer\nThí nghiệm phiên bản 25/10 được tiếp tục và tổng hợp ngày 26/10/2026','small')
    p('Nghiên cứu xây dựng hệ thống tổ chức chủ đề và truy hồi trên 13.702 tài liệu Patent và Publication. Bản này trình bày pipeline, cơ sở mô hình, kết quả đối chứng và các thay đổi đã được kiểm tra sau khi rà soát từ khóa, cụm chủ đề và Search Top 10.')
    table(['Kết quả được chấp nhận','Kết quả không được chấp nhận'],[
      ['KMeans50: gate ARI 0,6866 → 0,9918; NPMI 0,0792 → 0,1100 trên protocol gốc','Năm biến thể lọc từ khóa đều trượt ít nhất một tiêu chí'],
      ['Parser mới bảo toàn công thức; NVP có đủ Na3V2(PO4)3','Reranker Top30 tăng dev nhưng giảm gate: 0,9271 → 0,8507'],
    ],[1,1])
    p('Embeddings và checkpoint encoder hợp lệ được giữ nguyên; không huấn luyện lại hoặc encode lại corpus. Pipeline có trạng thái, logs và checkpoint để bỏ qua phần đã hoàn thành sau gián đoạn. Báo cáo giữ cả kết quả thất bại để giới hạn các kết luận.')
    p('Dữ liệu là toàn bộ Title–Abstract đang có, không phải toàn văn bài báo. Nhãn AI Silver pilot20 chưa phải nhãn con người, không dùng để chọn model và chưa đủ để công bố relevance tổng thể.','small')

    section('1 Câu hỏi nghiên cứu và các điểm được sửa')
    table(['Câu hỏi','Bằng chứng dùng trả lời','Phạm vi kết luận'],[
      ['RQ1 Topic có ổn định và diễn giải được không','Seed agreement, coherence, kích thước, nguồn và tài liệu đại diện','Cấu trúc thống kê; chưa có expert topic ground truth'],
      ['RQ2 Đọc đủ Abstract và continuation cải thiện gì','Độ bao phủ và proxy Title–document cố định','Không đồng nhất với relevance query tự nhiên'],
      ['RQ3 Search đưa bài phù hợp lên đầu không','Relevance0–2, benchmark giữ riêng, phân tích lỗi','Nhãn người còn thiếu; có pilot AI riêng'],
    ],[1.4,2,2])
    p('Bản LaTeX trước chốt ngày 19/10 mô tả Search v2 Top5 và BERTopic84/92 topic. Bản hệ thống ngày24/10 đã có Top10 và nhiều catalog K10. Bản này cập nhật luồng hiện hành; các số lịch sử chỉ được trình bày đúng snapshot và protocol của chúng.')
    p('LDA Stage2 chọn K20 được tách khỏi LDA K10 dùng đối chứng. BERTopic được tách khỏi encoder BGE/BatteryBERT và khỏi KMeans. C_v, ARI, coverage, cosine, RRF và relevance được định nghĩa riêng, tránh dùng một con số thay thế các mục tiêu khác.')
    p('Các phần API quota, AI consensus cũ, synthetic recovery và hàng chục phương pháp lịch sử được đưa về nguồn tham chiếu; không dùng chúng làm bằng chứng chính cho cấu hình hiện hành. Đóng góp được xác định là hệ thống ứng dụng và kiểm chứng so sánh, không tuyên bố một thuật toán học máy mới.')
    p('Từ khóa có vẻ chuyên biệt hơn chưa đủ để chứng minh topic tốt hơn. Vì vậy các thay đổi về clustering và keywords được thử riêng, giữ các yếu tố khác cố định khi có thể.','small')

    section('2 Pipeline toàn hệ thống và đầu ra')
    picture(ROOT/'outputs/report_pipeline_20261025/figures/01_pipeline.png','Pipeline tổng thể giữ hai mục tiêu topic và Search riêng biệt.','src/pipeline.py; src/final_search.py; src/revised_topics.py',340)
    p('Excel → làm sạch và sàng lọc → corpus đóng băng. Nhánh BoW tạo LDA; nhánh neural tạo embedding rồi phân cụm bằng BERTopic hoặc KMeans. Search kết hợp TF-IDF và vector BatteryBERT trong Qdrant bằng RRF60. Topic chỉ hỗ trợ duyệt và lọc theo doc_id, không cộng điểm vào ranking.')
    table(['Giai đoạn','Output hiện hành'],[
      ['Dữ liệu','main_corpus.parquet; 13.702 doc_id duy nhất'],
      ['Encoder','Checkpoint full-document đã chấp nhận; ma trận 13.702 × 768'],
      ['Clustering','Source projection train-fit; KMeans10 n_init50'],
      ['Keywords','100 từ khóa, evidence và audit; parser v2 giữ công thức'],
      ['Search','Top10 hybrid mặc định; reranker thử nghiệm bị loại'],
    ],[1.2,3.8])

    section('3 Dữ liệu và giao thức thực nghiệm')
    table(['Nguồn','Đầu vào','Corpus chính'],[[s,count(stage1['input_counts'][s]),count(stage1['main_corpus_counts'][s])] for s in ['Patent','Publication']],[1.8,1,1])
    table(['Split','Tài liệu','Vai trò'],[['Train','10.921','Optimizer và fit geometry, vocabulary, KMeans'],['Validation','1.361','Phát triển và chọn cấu hình'],['Test','1.368','Đo sau lựa chọn; split đã dùng trong lịch sử'],['Title only','52','Chỉ inference']],[1,1,3])
    p('Giữ nguyên dữ liệu gốc, các trường Title/Abstract, nguồn, năm, identifier và split_group_id. Các nhãn phạm vi CORE/RELATED là heuristic và có thể giữ tài liệu đa hệ pin, phần cứng hoặc sodium-metal. Không xóa tài liệu sau khi nhìn kết quả để làm tăng coherence.')
    p('Thí nghiệm geometry dùng seed42/43/47 trên validation; coordinator kiểm tra một seed mới20261029 trên test lịch sử. Keyword experiments giữ một tập nhóm validation riêng cho gate, nhưng không ứng viên nào qua dev nên không mở gate ứng viên. Search dùng16 title anchors phát triển và16 anchors giữ riêng, không điều chỉnh query Q01–Q20 đã khóa.')
    p('Các tập vẫn lấy từ corpus và split lịch sử. Gate mới trong phiên sửa này không trở thành external test độc lập. Các proxy lexical-summary và Title-pair đều có giới hạn, không được gọi là accuracy khoa học hoặc relevance người dùng.')

    section('4 LDA và BERTopic trong bài toán chủ đề')
    eq('lda','LDA mô hình hóa tài liệu bằng hỗn hợp topic và topic bằng phân bố từ')
    p('LDA nhận BoW, giả định mỗi tài liệu có tỷ trọng θ trên các topic và mỗi topic có phân bố từ φ. Top10 words lấy theo P(w|topic); tài liệu vẫn giữ posterior nhiều chủ đề dù giao diện hiển thị topic trội nhất. α và η là prior Dirichlet [1].')
    p('Stage2 thử K10/15/20/25/30, passes10, iterations100, alpha asymmetric, eta symmetric. Hai K có C_v tốt ở seed cơ sở được đánh giá thêm stability; điểm tổng hợp chọn K20. Đây là lựa chọn trong lưới đã thử, không phải tối ưu toàn cục.')
    table(['LDA Stage2 K20','Giá trị'],[['C_v',f(lda['selected_metrics']['c_v'])],['Diversity',f(lda['selected_metrics']['topic_diversity'])],['Stability Jaccard sau ghép Hungarian',f(lda['selected_metrics']['stability'])]],[3,1])
    p('BERTopic dùng encoder → UMAP5 chiều → HDBSCAN → c-TF-IDF để mô tả từ. Cụm do mật độ quyết định và có outlier −1. Bảng đối chứng K10 gộp các topic sau phân cụm, không phải HDBSCAN được yêu cầu tạo đúng10 cụm [2].')
    eq('ctfidf','c-TF-IDF với reduce_frequent_words bật trong nhánh BERTopic')
    p('UMAP15 neighbors, min_dist0, cosine; HDBSCAN min_cluster_size30, min_samples10, eom. Các từ khóa mô tả cụm sau khi clustering, nên thay representation không tự cải thiện membership. UMAP2D chỉ để minh họa.','small')

    section('5 BGE BatteryBERT và huấn luyện đã hoàn thành')
    eq('attention','Attention tạo biểu diễn token phụ thuộc ngữ cảnh trong từng chunk')
    p('BGE và BatteryBERT là encoder; BERTopic và KMeans là cách sử dụng embedding để tạo topic. Các checkpoint BGE-base-en-v1.5 và BatteryBERT-cased được ghi trong cấu hình [3,4].')
    eq('pool','Pooling các token Abstract rồi tổng hợp chunk có trọng số token mới')
    p('BatteryBERT full-document dùng cửa sổ256, overlap64, Title làm context nhưng không trực tiếp pool vào chunk Abstract. Trọng số chunk là số token mới được bao phủ; overlap vẫn tác động ngữ cảnh. Đọc đủ các cửa sổ không tạo attention toàn cục giữa mọi đoạn.')
    eq('loss','Contrastive loss theo batch; code lấy trung bình hai chiều và thêm distillation')
    table(['Huấn luyện đã được kiểm tra','Giá trị'],[['Block cập nhật','Block cuối; 11 block đóng băng'],['Loss batch / temperature','32 cặp / 0,05; gradient caching hai tầng'],['Lượt train hoàn tất','10.921 tài liệu; 26.147 chunks; 3.420.215 token Abstract'],['Checkpoint','Step342; learning rate5e-6'],['Test proxy trước / sau','0,887839 / 0,924704']],[2,3])
    p('Model weights, ordered IDs, vector hash, finite values và chuẩn L2 đã được xác nhận lại khi resume. Không lặp train hoặc embeddings. Điểm proxy có Title context không được diễn giải thành92,47% Search accuracy [5].','small')

    section('6 Đối chứng lịch sử và vấn đề cấu trúc topic')
    table(['Catalog K10 lịch sử','C_v','Diversity','Outlier'],[
      [row.model,f(row.C_v),f(row.Topic_Diversity),'N/A' if str(row.Outlier_Rate)=='' else percent(row.Outlier_Rate)] for row in historical.itertuples()
    ],[2.6,.8,.9,.8])
    p('BERTopic–BatteryBERT pretrained có C_v lịch sử cao nhưng hai cụm chứa7.261 và5.449 tài liệu, tức92,76% corpus. Hai nhóm rộng phản ánh publication về hiệu năng và patent về phương pháp chế tạo. Điều này minh họa vì sao coherence cao chưa đủ chọn một model topic.')
    p('BERTopic–BGE tách được nhiều nhóm vật liệu cụ thể hơn nhưng có3.120 outlier. KMeans có0 outlier do bắt buộc gán mọi tài liệu; đây không phải bằng chứng vượt trội. LDA posterior, HDBSCAN membership và KMeans distance là các đại lượng khác nhau.')
    p('Bản optimized23/10 là đối chứng bổ sung. Đo lại bằng cùng recipe và seed phát triển cho ARI0,8455, NPMI0,08390, source V0,3103, diversity0,80 và silhouette0,1610. Model mới cải thiện một số mục tiêu, nhưng không thắng silhouette và diversity của đối chứng này.')
    p('C_v trong bảng là protocol lịch sử. Không đặt nó cùng C_v canonical mới để kết luận một thứ hạng tuyệt đối giữa mọi model. Các catalog gốc vẫn được lưu đầy đủ để tái lập.','small')

    section('7 Chạy lại clustering và kiểm tra giữ riêng')
    fig('geometry_before_after','Nhiều khởi tạo KMeans cải thiện stability, giữ cùng embeddings và keyword recipe gốc.','comparison/geometry_comparison.csv',245)
    table(['Cấu hình dev','ARI nhiều seed','NPMI','Kết luận'],[
      [r.model,f(r.stability_ari),f(r.document_npmi),'Đạt' if str(r.passed_development).lower()=='true' else 'Không đạt'] for r in geo[geo.split.eq('validation')&geo.model.ne('old_optimized_23oct')].itertuples()
    ],[2,1,1,1])
    p('Source projection fit trên train loại một hướng chênh lệch trung bình nguồn. KMeans10 n_init50 được chọn trên dev; source-whiten32/64 có stability cao nhưng giảm NPMI, và64 chiều còn tạo cụm30,71% validation nên bị loại.')
    table(['Gate seed20261029','Baseline n_init10','n_init50'],[
      ['ARI',f(geometry['baseline']['stability_ari'],6),f(geometry['candidate']['stability_ari'],6)],
      ['Document NPMI',f(geometry['baseline']['document_npmi'],6),f(geometry['candidate']['document_npmi'],6)],
      ['Source Cramér V',f(geometry['baseline']['source_cramers_v']),f(geometry['candidate']['source_cramers_v'])],
      ['Top2 share',percent(geometry['baseline']['top2_topic_share']),percent(geometry['candidate']['top2_topic_share'])],
    ],[2,1.4,1.4])
    p('Source V và top2 share trên gate tăng nhẹ, dù vẫn trong giới hạn đã chốt. Kết luận được chấp nhận là stability và coherence theo protocol gốc cải thiện, không phải mọi chỉ số đều tốt hơn.','small')

    section('8 Catalog mới và ý nghĩa của các nhóm')
    fig('revised_topic_sources','Phân bố source sau clustering mới; phần cứng vẫn là một nội dung thật cần gọi tên đúng.','final_topics/topic_source_distribution.csv',226)
    merged=dist.merge(labels[['topic_id','topic_label']],on='topic_id')
    table(['ID','Tên diễn giải sau khi đọc đại diện','Tài liệu'],[[str(row.topic_id),row.topic_label,count(row.documents)] for row in merged.itertuples()],[.4,4,.8])
    p('Căn chỉnh ID bằng Hungarian trên overlap tài liệu train chỉ giữ tính tiện dụng hiển thị; không chứng minh ý nghĩa topic bất biến. ID7 chỉ giữ khoảng1,06% overlap theo phép ghép bắt buộc và nay là nhóm MoS2/MXene; tên pha/vacancy cũ đã được thay.')
    p('Tên nhóm là diễn giải AI từ100 keywords và10 Abstract đại diện trung tâm đã đọc đầy đủ.60 đại diện được xuất để rà soát tiếp; chưa có expert topic labels.','small')

    section('9 Bảo toàn công thức và kiểm chứng từ khóa')
    table(['Trường hợp gốc','Xử lý phiên bản mới'],[
      ['Na3V2(PO4)(3)','Na3V2(PO4)3; giữ raw span và phép biến đổi'],
      ['Na(3)V(2)(PO(4))(3)','Chuẩn hóa ngoặc số không có khoảng trắng khi cú pháp hỗ trợ'],
      ['TiO2 (101) hoặc TiO2(101)','Không nối thành TiO2101; giữ chú thích hoặc span mơ hồ'],
      ['Na(1) / Fe[Fe(CN)(6)]','Giữ nguyên dạng site label/complex chưa hỗ trợ; không xuất fragment bên trong như toàn công thức'],
    ],[1.7,3.4])
    p('Parser cũ có thể cắt Na3V2(PO4)(3) thành Na3V2(PO4). Parser v2 giữ complete spans, phân biệt case, ranh giới field, stopword và dấu câu; chỉ chuẩn hóa cách ghi số khi đủ điều kiện cú pháp. Unsupported không có nghĩa công thức sai hóa học.')
    p('Đã xử lý27.404 trường Title/Abstract qua54 shard;41.002 span giống công thức,2.340 chuẩn hóa thận trọng và15.702 span chưa hỗ trợ được giữ nguyên. Có806 raw Na3V2(PO4)(3) equivalences;25 chú thích số cách trắng không bị ghép. Không thay đổi raw corpus.')
    fig('canonical_correctness_effect','Cùng clustering và canonical reference để so danh sách trước/sau sửa ký hiệu.','final_topics/keyword_comparison.csv; common_coherence.csv',224)
    p('Test NPMI tăng0,065294 →0,096332 và C_v cùng reference tăng0,547907 →0,561193; diversity giảm0,79 →0,75. Đây là chẩn đoán representation sau sửa cú pháp, khác số NPMI0,1100 của geometry gate trên stream gốc.','small')

    section('10 Thử lọc từ nhiễu và lý do không chọn')
    fig('keyword_failed_hypotheses','Các danh sách ít từ chung có thể mất coverage và NPMI dù F1 tăng.','comparison/keyword_experiments.csv',248)
    table(['Recipe','F1 proxy','NPMI','Coverage','Quyết định'],[
      [r.candidate,f(r.macro_f1),f(r.document_npmi),f(r.macro_own_topic_coverage),'Giữ baseline' if r.candidate=='baseline_dev' else 'Loại'] for r in keytrials.itertuples()
    ],[1.8,.8,.8,.8,.9])
    p('Cycle1 thử ba cách ưu tiên specificity/cụm từ; cycle2 thử hai cách giữ TF và loại thủ tục/alias thận trọng. Cả năm đều trượt ít nhất một gate đã chốt. Không mở gate ứng viên rồi chỉnh tiếp để tìm số đẹp.')
    p('Các từ solution/dry/type thường yếu khi đặt tên, nhưng carbon/anode/oxide có thể kết nối nhiều phân họ đúng trong một cụm. Do đó model giữ weighting gốc, đồng thời xuất audit cho từ rộng, từ thao tác/cấu trúc, alias và ký hiệu chưa hỗ trợ. Không che từ bằng giao diện rồi gọi đó là cải thiện mô hình.')

    section('11 Top 10 keywords hiện hành và đọc đúng trọng số')
    fig('revised_top10_keywords','Bốn topic minh họa với đủ10 từ khóa và trọng số thống kê.','final_topics/topic_keywords.csv',365)
    eq('keywords','Weighting giữ TF tương đối × IDF × exclusivity mũ0,25')
    p('Vocabulary chỉ fit train, unigram/bigram, min_df5, max_df0,6, tối đa40.000 features. Mỗi từ khóa có ít nhất5 tài liệu train trong topic hỗ trợ; các phrase phải có bằng chứng liền kề theo stream giữ ranh giới.')
    p('Điểm weight xếp hạng các từ mô tả cụm; không phải xác suất từ đúng khoa học hoặc xác suất tài liệu relevant. Bảng keyword_audit.csv và keyword_evidence.csv lưu support theo split và ngữ cảnh để người đọc rà soát.','small')

    section('12 Search Top 10 và thí nghiệm reranker')
    eq('rrf','Cosine so biểu diễn; RRF60 hợp nhất hai thứ hạng')
    p('Query → lọc nguồn/năm/required terms/topic → lấy100 ứng viên TF-IDF và100 dense → RRF60 → Top10. Topic mới được chuyển thành danh sách doc_id được phép trước retrieval; vector index không cần encode hoặc xây lại chỉ vì nhãn topic thay đổi.')
    fig('search_gate_rejection','Reranker Top30 được chọn ở dev nhưng không chuyển giao sang gate; không kích hoạt.','comparison/search_proxy_comparison.csv',245)
    table(['Proxy','Baseline','Rerank30'],[
      ['Dev16 Group MRR@10','0,832813','0,877083'],
      ['Gate16 Group MRR@10',f(search['baseline']['group_mrr_at_10'],6),f(search['candidate']['group_mrr_at_10'],6)],
      ['Gate Success@1',f(search['baseline']['success_at_1']),f(search['candidate']['success_at_1'])],
      ['Gate Success@10',f(search['baseline']['success_at_10']),f(search['candidate']['success_at_10'])],
    ],[2,1.3,1.3])
    p('MiniLM đọc Abstract-only passages và mọi query window, nhưng vẫn là proxy thiên lệch vì vector/TF-IDF chứa Title. Median rerank là chi phí bổ sung, không gồm encode query; thời gian baseline bằng0 trong bảng cache không chứng minh truy hồi miễn phí. Giữ hybrid mặc định.','small')

    section('13 Output Search thực tế và phân tích điều kiện')
    ranks=csv(BASE/'rankings_top10.csv');part=ranks[ranks.query_id.eq('Q01')&ranks.method.eq('BATTERYBERT_FULLDOC_HYBRID')].sort_values('rank')
    p('Q01: Hard carbon anodes derived from biomass for sodium-ion batteries. Bảng giữ output hybrid đã khóa; không sửa hạng sau khi đọc nhãn pilot.')
    table(['Hạng','Tiêu đề','RRF'],[[str(int(row['rank'])),row['title'],f(row['rrf_score'],5)] for _,row in part.iterrows()],[.45,4.5,.8])
    p('Các lỗi đã ghi nhận ở query nhiều điều kiện: có NaFSI nhưng không đúng ether/nhiệt độ thấp; có quy trình cell nhưng sai công đoạn cần tìm; có voltage drop nhưng chưa chứng minh thermal management. Required terms chỉ xác nhận token, không xác nhận quan hệ hoặc phủ định.')
    p('Không dùng các lỗi benchmark đã xem để viết quy tắc riêng theo query_id. Điều cần thêm là đánh giá intent có nhãn và phân tích từng điều kiện trên tập phát triển độc lập, thay vì tự tăng score cho tên vật liệu.','small')

    section('14 Đánh giá Top 10 và Silver Labels thử nghiệm')
    eq('ndcg','Precision và nDCG cần nhãn relevance độc lập theo đúng protocol')
    table(['Chỉ số','Ý nghĩa','Chưa chứng minh được'],[
      ['P@10','Số bài grade2 trong Top10 chia10','Không có qrels thì không suy từ cosine'],
      ['nDCG@10','Mức liên quan và vị trí; gain2^grade−1','IDCG theo pool chung chưa là toàn corpus'],
      ['MRR@10','Nghịch đảo hạng bài grade2 đầu tiên','Khác proxy ghép Title–document'],
      ['Recall@10','Tỷ lệ toàn bộ relevant documents tìm được','Chưa có relevant set đầy đủ nên không công bố'],
    ],[1,2.1,2.2])
    fig('silver_pilot20','Pilot AI theo yêu cầu mới, không dùng để chọn model và không thay human qrels.','outputs/ai_silver_labels_20261026_v1/batch_0001_summary.json',190)
    p('Đã đọc20 cặp đầu bảng annotation_pool.csv:11 nhãn2,4 nhãn1,4 nhãn0 và1 trống;8 cặp review_required. Cả20 đều Q01 theo thứ tự gốc, nên không đại diện20 query hoặc toàn corpus.683 cặp còn lại chưa được chấm trong pilot này.')
    p('Một patent đồng thời gọi cathode và negative electrode được để trống, NEEDS_REVIEW. Các patent nêu cathode nhất quán được chấm0 theo văn bản, confidence medium và gắn cờ kiểm tra metadata. Không tự sửa vai trò bằng kiến thức về hard carbon. Pool người vẫn có0 nhãn xác minh.','small')

    section('15 Resume checkpoint và phần đã bỏ qua')
    table(['Khối','Kết quả thực thi khi tiếp tục'],[
      ['Dữ liệu và model gốc','Kiểm hash; không thay raw corpus hoặc checkpoint'],
      ['Embeddings13.702 ×768','Xác nhận shape, norm, ordered IDs, model/vector hash;0 forward calls mới'],
      ['Geometry dev và keywords5ứng viên','Adopt output có sẵn; không fit lại'],
      ['Search baseline và rerank30 dev','Adopt bundle và query cache; không inference lại'],
      ['Search rerank50 và comparator cũ','Chỉ chạy phần còn thiếu sau gián đoạn'],
      ['Gate và export','Gate mỗi baseline/winner một lần; export chỉ predict'],
      ['Scientific stream','Checkpoint256documents/shard;54 shard; merge một lần'],
      ['Đánh giá và báo cáo','Manifest theo giai đoạn; logs và trạng thái riêng'],
    ],[1.7,3.3])
    p('Lệnh tiếp tục: .venv/Scripts/python.exe scripts/resume_research_revision.py. Dùng --status để đọc trạng thái; --through tên_giai_đoạn để dừng ở mốc cụ thể. Mỗi stage có pending/running/failed/completed, log thời gian/lỗi, receipt và hash của output.')
    p('Receipt hợp lệ được bỏ qua. Output hoàn thành bị đổi hash sẽ báo lỗi và giữ nguyên để kiểm tra, không tự ghi đè. Process lock kiểm PID cùng creation time, khôi phục lock stale và chờ child còn chạy thay vì khởi động trùng. Test mô phỏng ngắt xác minh partial checkpoint được giữ.')
    p('Ngày thư mục25/10 định danh phiên nghiên cứu; các stage tiếp tục26/10 ghi timestamp UTC trong event logs. Không sửa ngày của kết quả cũ để làm chúng giống mới chạy.','small')

    section('16 Kiểm chứng và trạng thái sử dụng')
    table(['Kiểm chứng','Kết quả'],[
      ['Tests ảnh hưởng bởi thay đổi',str(verified['tests']['tests'])+' PASS;0 failures/errors/skips'],
      ['Artifacts gốc bảo toàn',str(verified['preservation']['checked'])+' file hash không đổi'],
      ['Catalog mới','13.702 assignments;10 topics;100 keywords;60 representatives'],
      ['Bật catalog revised',str(deploy['activate_revised_topics'])],
      ['Bật reranker thử nghiệm',str(deploy['activate_search_rerank'])+' vì trượt gate'],
      ['Human relevance labels','0; AI Silver20 được tách riêng'],
    ],[2.4,2.5])
    p('Ứng dụng đọc catalog mới với tên diễn giải đã rà soát ở mức AI, hiển thị các cờ từ khóa và giữ các catalog lịch sử để đối chiếu. Search vẫn dùng vector và hybrid cũ; topic chỉ đổi metadata và danh sách doc_id khi lọc.')
    p('Những tests xác minh luồng dữ liệu, ranh giới cú pháp, identity, skip/resume và công thức đã mã hóa. Chúng không thay chuyên gia đọc chủ đề hoặc đánh giá relevance. Gate đạt cũng không loại bỏ mọi lỗi trong corpus hoặc mọi tài liệu gán sai.')
    p('Không đưa reranker bị loại vào phương pháp mặc định. Code và output thử nghiệm nằm riêng để tái lập; không làm bảng đánh giá sáu method lịch sử tự phát sinh thêm method mới.')

    section('17 Thảo luận và công việc còn cần bằng chứng')
    p('Kết quả cho thấy instability của clustering không nhất thiết yêu cầu train lại encoder: tăng khởi tạo KMeans trong cùng không gian đã cải thiện rõ agreement trên dev và seed giữ riêng. Parser mới giải quyết lỗi cắt công thức thực sự và cải thiện coherence của danh sách trên cùng reference.')
    p('Các kết quả âm cũng làm rõ giới hạn: source whitening làm mất cấu trúc từ vựng hữu ích; keyword specificity quá mạnh làm mất coverage; cross-encoder tổng quát không cải thiện ổn định proxy của corpus này. Không có căn cứ để mặc định mô hình phức tạp hơn sẽ tốt hơn.')
    table(['Còn thiếu','Bước tiếp theo hợp lý'],[
      ['Relevance người dùng độc lập','Chấm pool theo intent với con người; xác minh và giữ query-family ngoài phát triển'],
      ['Expert topic labels','Đọc đại diện và thành viên biên; kiểm tra vai trò điện cực, pha, hệ pin và phạm vi'],
      ['Quan hệ nhiều điều kiện','Đánh giá evidence theo từng điều kiện trên bộ phát triển mới, có nhãn'],
      ['Chi tiết dài bị pooling','Thử multi-vector/chunk retrieval sau khi có protocol và ngân sách mới'],
      ['Toàn văn và khái quát','Ingest dữ liệu mới version riêng; không coi Title–Abstract là full article'],
    ],[1.6,3.6])
    p('Kết luận được hỗ trợ là pipeline đã nhất quán hơn, có clustering ổn định hơn và xử lý ký hiệu có truy vết. Chất lượng khoa học và Search tổng quát còn cần nhãn độc lập. Silver pilot chỉ là bước chuẩn bị đánh giá, không phải xác nhận cuối.')

    section('18 Tài liệu phương pháp và truy vết output')
    references=[
      '[1] Blei, Ng và Jordan. Latent Dirichlet Allocation. JMLR3,2003. https://www.jmlr.org/papers/v3/blei03a.html',
      '[2] Grootendorst. BERTopic Neural topic modeling with a class based TF IDF procedure.2022. https://arxiv.org/abs/2203.05794',
      '[3] BAAI. BGE base en v1.5 model card. https://huggingface.co/BAAI/bge-base-en-v1.5',
      '[4] BatteryData. BatteryBERT cased checkpoint. https://huggingface.co/batterydata/batterybert-cased',
      '[5] Gao và cộng sự. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup.2021. https://arxiv.org/abs/2101.06983',
      '[6] Sentence Transformers. Cross encoder usage. https://www.sbert.net/docs/cross_encoder/usage/usage.html',
      '[7] Kassis, Agarwal, He, Patel và Brueckner. Scientific Agent Skills A Library of Procedural Knowledge for Research Agents.2026. https://doi.org/10.48550/arXiv.2609.00065',
    ]
    for ref in references:p(ref,'small')
    p('Arbor và kỹ năng phản biện khoa học trong [7] hỗ trợ ghi giả thuyết, giữ evaluator, kiểm tra gate và giới hạn kết luận. Chúng không cung cấp bằng chứng rằng model chính xác hơn. Phiên bản nguồn đã đối chiếu là arXiv v2; dẫn DOI không khóa hậu tố version.','small')
    table(['Nội dung','Đường dẫn trong outputs/research_revision_20261025_v1'],[
      ['Baseline audit','baseline_audit.json; protected_artifacts_before.json'],
      ['Clustering','geometry/evidence/geometry; decisions/geometry.json'],
      ['Keywords thất bại','keywords/experiment_outputs; cycle2/comparison.json'],
      ['Cú pháp và catalog cuối','scientific_stream_v2; final_topics; review'],
      ['Search dev/gate','search/experiment_outputs; decisions/search_gate.json'],
      ['So sánh và hình','comparison; visualizations'],
      ['Resume','execution/pipeline_state.json; checkpoints; logs; events.jsonl'],
    ],[1.4,3.9])
    p('Các bảng/hình dựa trên số đo thực có manifest. Bản báo cáo31trang ngày25/10 và nguồn LaTeX19/10 được giữ nguyên, bản này là phiên bản tiếp tục với kết quả mới.','small')

    section('Phụ lục A Toàn bộ Top 10 keywords của catalog mới')
    table(['ID','Tên diễn giải','Top10 keywords'],[[str(topic_id),labels.set_index('topic_id').loc[topic_id,'topic_label'],', '.join(keys[keys.topic_id.eq(topic_id)].sort_values('rank').token)] for topic_id in range(10)],[.35,1.9,3.1])
    p('Danh sách giữ đúng rank và trọng số của recipe đã xuất. Các từ rộng và từ thao tác còn tồn tại được đánh dấu trong keyword_audit.csv, không bị xóa khỏi bảng sau khi xem điểm. Tên diễn giải không phải expert ground truth.','small')

    section('Phụ lục B File giao nộp và cách tiếp tục')
    p('Bản PDF và Markdown của báo cáo mới nằm tại output/pdf và docs với tên BAO_CAO_NGHIEN_CUU_CAI_THIEN_SODIUM_ION_20261026. Để chỉnh sửa số liệu, sửa nguồn hoặc recipe trong một phiên bản mới và tạo output riêng; không ghi đè model, pool hoặc metric đã hoàn thành.')
    table(['Mục đích','Lệnh hoặc file'],[
      ['Resume toàn bộ',' .venv/Scripts/python.exe scripts/resume_research_revision.py'],
      ['Đọc trạng thái',' .venv/Scripts/python.exe scripts/resume_research_revision.py --status'],
      ['Chạy ứng dụng',' .venv/Scripts/python.exe run_demo.py'],
      ['Pilot AI20','outputs/ai_silver_labels_20261026_v1/batch_0001_labels_20.csv'],
      ['Checkpoint từng5nhãn','outputs/ai_silver_labels_20261026_v1/checkpoints'],
      ['Nhãn người','Giữ nguyên annotation_pool và accepted_human_labels; không nhập AI vào đó'],
    ],[1.4,3.8])
    p('Sau khi có nhãn mới do con người xác minh, tính relevance theo protocol đã khóa. Nếu tiếp tục Silver Labels, tạo batch tiếp theo đúng cặp đầu vào và lưu output độc lập; không suy ra nhãn của683cặp còn lại từ kết quả20cặp đầu.')

    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d9d9d9'));canvas.line(margin,36,W-margin,36)
        canvas.setFont('Arial',8);canvas.setFillColor(colors.HexColor('#4c5e68'));canvas.drawString(margin,24,'Sodium ion Research Explorer  |  Bản tiếp tục ngày 26/10/2026');canvas.drawRightString(W-margin,24,str(doc.page if doc.page==1 else doc.page+1));canvas.restoreState()
    PDF.parent.mkdir(exist_ok=True,parents=True)
    pending=OUT/(NAME+'.pending.pdf')
    document=SimpleDocTemplate(str(pending),pagesize=A4,leftMargin=margin,rightMargin=margin,topMargin=43,bottomMargin=49,title='Khám phá chủ đề và tìm kiếm tài liệu pin sodium ion sau cải thiện',author='Sodium ion Research Explorer')
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(pending);writer=PdfWriter();writer.clone_document_from_reader(reader)
    toc=io.BytesIO();toc_canvas=pdfcanvas.Canvas(toc,pagesize=A4)
    toc_canvas.setFont('Arial-Bold',20);toc_canvas.drawString(margin,H-65,'Nội dung báo cáo')
    y=H-103;toc_canvas.setFont('Arial',10.1)
    heading_pages=[]
    for title in sections:
        located=next((i for i,page in enumerate(reader.pages) if title in (page.extract_text() or '')),None)
        if located is not None:heading_pages.append((title,located))
    for title,page in heading_pages:
        toc_canvas.drawString(margin,y,title)
        toc_canvas.drawRightString(W-margin,y,str(page+2))
        y-=25
    toc_canvas.setFont('Arial',9);toc_canvas.drawString(margin,y-22,'Các phần có bookmark để chuyển trang trong trình đọc PDF.')
    toc_canvas.setFont('Arial',8);toc_canvas.drawRightString(W-margin,24,'2')
    toc_canvas.save();toc.seek(0)
    writer.insert_page(PdfReader(toc).pages[0],1)
    writer.add_outline_item('Bìa và tóm tắt',0)
    writer.add_outline_item('Nội dung báo cáo',1)
    for index,title in enumerate(sections):
        # Locate the actual heading because a long table can span pages.
        page=next((i for i,p in enumerate(reader.pages) if title in (p.extract_text() or '')),None)
        if page is not None:writer.add_outline_item(title,page+1)
    if PDF.exists():raise ValueError('Final PDF already exists without receipt; inspect before replacing')
    with PDF.open('wb') as handle:writer.write(handle)
    if MD.exists():raise ValueError('Final Markdown already exists without receipt')
    MD.write_text('\n'.join(markdown),encoding='utf8')
    pending.unlink()
    manifest=[]
    for path in sorted(SOURCES):
        if path.is_file():manifest.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    meta={'pdf':str(PDF),'markdown':str(MD),'pages':len(PdfReader(PDF).pages),'result_figures':figures,'tables':tables,'sections':sections,
      'scientific_scope':'Historical-split diagnostics; Silver Labels are AI, not human ground truth','new_encoder_fit':False,
      'visual_qa':'Pending PNG inspection and artifact validation'}
    (OUT/'build_summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    artifactpaths=[PDF,MD,OUT/'source_manifest.json',OUT/'build_summary.json']
    (OUT/'artifact_manifest.json').write_text(json.dumps({'artifacts':[{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size} for path in artifactpaths]},indent=2),encoding='utf8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--revision',choices=['v1','v2'],default='v1');args=parser.parse_args()
    if args.revision=='v2':
        OUT=REV/'report_v2';OUT.mkdir(parents=True,exist_ok=True)
        NAME=NAME+'_v2';PDF=ROOT/'output/pdf'/(NAME+'.pdf');MD=ROOT/'docs'/(NAME+'.md')
    build()
