"""Export actual Silver-label retrieval measurements as figures, Markdown and PDF."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/ai_silver_labels_703_20261026_v1';E=OUT/'evaluation';FIG=E/'figures'
NAME='DANH_GIA_SEARCH_SILVER_703_20261026'
PDF=ROOT/'output/pdf'/(NAME+'.pdf');MD=ROOT/'docs'/(NAME+'.md')
REPORT_MANIFEST=E/'report_manifest.json'
RUNTIME=Path('C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
ALIASES={'TFIDF_COSINE':'TF-IDF','BATTERYBERT_SEMANTIC':'BatteryBERT semantic','BATTERYBERT_HYBRID':'BatteryBERT hybrid','BGE_SEMANTIC':'BGE semantic',
 'BATTERYBERT_FULLDOC_SEMANTIC':'Full-document semantic','BATTERYBERT_FULLDOC_HYBRID':'Full-document hybrid'}
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def figures():
    import pandas as pd
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(E/'metrics_summary.csv');cols=['Precision_at_10','nDCG_at_10','MRR_at_10'];x=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(11,4.5))
    for j,(c,color,name) in enumerate(zip(cols,['#29566d','#279d8f','#c18b3c'],['P@10','nDCG@10','MRR@10'])):
        bars=ax.bar(x+(j-1)*.25,d[c],width=.24,label=name,color=color)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.set_xticks(x,[ALIASES[m].replace(' ','\n',1) for m in d.method]);ax.set_ylim(0,1.05);ax.legend(ncol=3);ax.set_ylabel('Macro mean across 20 queries')
    ax.set_title('AI Silver Labels | 703 pairs | direct relevance threshold = 2');ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(FIG/'method_comparison.png',dpi=180);plt.close(fig)
    p=pd.read_csv(E/'metrics_per_query.csv');methods=list(d.method);qids=sorted(p.query_id.unique())
    fig,axes=plt.subplots(1,2,figsize=(12,9))
    for ax,c,title in zip(axes,['Precision_at_10','nDCG_at_10'],['P@10: only label 2 counts','nDCG@10: labels 1 and 2 receive gain']):
        v=p.pivot(index='query_id',columns='method',values=c).loc[qids,methods].to_numpy();im=ax.imshow(v,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(6),['TF-IDF','BB sem.','BB hybrid','BGE','Full sem.','Full hybrid'],rotation=50,ha='right');ax.set_yticks(range(20),qids);ax.set_title(title,fontsize=11)
        for i in range(20):
            for j in range(6):ax.text(j,i,f'{v[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if v[i,j]>.65 else 'black')
    fig.suptitle('Shared frozen pool | AI labels, not expert ground truth')
    fig.subplots_adjust(left=.06,right=.9,bottom=.12,top=.93,wspace=.22)
    cax=fig.add_axes([.93,.22,.015,.58]);fig.colorbar(im,cax=cax,label='Score')
    fig.savefig(FIG/'query_heatmaps.png',dpi=180);plt.close(fig)
    s=pd.read_csv(E/'sensitivity_summary.csv');rel=s[s.scenario.eq('partial_or_direct_threshold1')].set_index('method').loc[methods]
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(x-.18,d.Precision_at_10,width=.36,label='Direct only (label2)',color='#29566d');ax.bar(x+.18,rel.Precision_at_10,width=.36,label='Partial or direct (label1/2)',color='#8abeb6')
    ax.set_xticks(x,[ALIASES[m].replace(' ','\n',1) for m in methods]);ax.set_ylim(0,1.05);ax.legend();ax.set_title('Changing relevance threshold changes the question being measured');ax.set_ylabel('P@10, macro mean');fig.tight_layout();fig.savefig(FIG/'threshold_sensitivity.png',dpi=180);plt.close(fig)

def document():
    import pandas as pd
    from html import escape
    from PIL import Image as PILImage
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak,Image,KeepTogether
    from pypdf import PdfReader
    if REPORT_MANIFEST.exists():
        m=json.loads(REPORT_MANIFEST.read_text(encoding='utf8'));assert all(h(ROOT/x['path'])==x['sha256'] for x in m['artifacts']);print('SKIP verified Silver report');return
    for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
    styles={n:ParagraphStyle(n,fontName='Arial-Bold' if n in ['heading','head'] else 'Arial',fontSize=size,leading=lead,spaceAfter=space,
        textColor=colors.white if n=='head' else colors.HexColor('#20313d')) for n,size,lead,space in [('body',10,14,8),('small',8,11,6),('heading',20,26,14),('cell',8,10,0),('head',8,10,0)]}
    cw=A4[0]-88;story=[];md=[];sources=set();pages=[]
    def read(name):
        p=E/name;sources.add(p);return pd.read_csv(p)
    def p(t,style='body'):story.append(Paragraph(escape(str(t)),styles[style]));md.append(str(t)+'\n')
    def section(t):
        if story:story.append(PageBreak())
        story.append(Paragraph(escape(t),styles['heading']));md.append('## '+t+'\n');pages.append(t)
    def table(headers,rows,widths):
        cells=[[Paragraph(escape(str(v)),styles['head']) for v in headers]]+[[Paragraph(escape(str(v)),styles['cell']) for v in r] for r in rows]
        obj=Table(cells,colWidths=[cw*w/sum(widths) for w in widths],repeatRows=1,hAlign='LEFT')
        obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#29566d')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f5f6')]),
          ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d4dce0')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([obj,Spacer(1,9)])
        md.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v).replace('|','/') for v in r)+' |' for r in rows]+[''])
    def pic(name,caption,maxheight):
        path=FIG/name;sources.add(path);w,hh=PILImage.open(path).size;scale=min(cw/w,maxheight/hh)
        story.append(KeepTogether([Image(str(path),w*scale,hh*scale),Spacer(1,4),Paragraph(escape(caption),styles['small'])]));story.append(Spacer(1,6))
        md.extend([f'![{caption}]({Path(os.path.relpath(path,MD.parent)).as_posix()})',caption,''])
    def f(v):return f'{v:.4f}'.replace('.',',')
    summary=read('metrics_summary.csv');per=read('metrics_per_query.csv');sensitivity=read('sensitivity_summary.csv');cis=read('confidence_intervals.csv');deltas=read('paired_deltas.csv');recall=read('pooled_recall_supplement.csv')
    labels=json.loads((OUT/'label_summary.json').read_text(encoding='utf8'));sources.add(OUT/'label_summary.json');sources.add(OUT/'silver_labels_703.csv');sources.add(E/'protocol.json');sources.add(E/'validation.json')
    section('Đánh giá Search bằng 703 Silver Labels')
    p('Sodium-ion Research Explorer · 26/10/2026 · Phụ lục kết quả mới cho báo cáo nghiên cứu. Nhãn do AI đọc Query–Title–Abstract; chưa được chuyên gia hoặc người dùng xác nhận.')
    p('Đã xử lý đủ 703/703 cặp, 20 truy vấn, không thiếu hoặc trùng ID. Phân bố: 143 nhãn 0, 302 nhãn 1, 258 nhãn 2. Có 70 trường hợp cần review; 28 bản ghi thiếu Abstract. Tất cả đều có nhãn số theo bằng chứng hiện có, nhưng các trường hợp review vẫn là kết luận tạm thời.')
    table(['Phương pháp','P@10','nDCG@10','MRR@10'],[[ALIASES[r.method],f(r.Precision_at_10),f(r.nDCG_at_10),f(r.MRR_at_10)] for r in summary.itertuples()],[3.3,1,1.2,1.1])
    pic('method_comparison.png','Hình 1. Trung bình đều trên cùng 20 truy vấn; nhãn 2 là relevance trực tiếp.',230)
    p('Full-document hybrid cao nhất về trung bình trên cả ba chỉ số chính trong lần chấm AI này. So với TF-IDF, mức tăng là +0,090 P@10; +0,0975 nDCG@10; +0,085 MRR@10. Chưa đủ căn cứ khẳng định thắng các phương pháp gần nhất trên người dùng mới.',style='small')
    section('Giao thức, công thức và kiểm chứng')
    p('Nhãn được khóa trước khi ghép với thứ hạng: đọc đầy đủ Query, Title và Abstract từng cặp, không dùng cosine/RRF/rank/model để quyết định nhãn. 20 batch được checkpoint riêng. Giữ nguyên bản pilot; 19 nhãn đã quyết định được dùng lại, 1 nhãn trống được re-review thành nhãn 1, low confidence và cần review vì vai trò cathode/negative electrode mâu thuẫn.')
    table(['Chỉ số','Cách tính và ý nghĩa'],[
      ['P@10','Số nhãn 2 trong Top 10 / 10. Đo tỷ lệ tài liệu đáp ứng trực tiếp; 0,540 tương đương trung bình 5,4 bài trực tiếp / 10.'],
      ['MRR@10','1/r của nhãn 2 đầu tiên; bằng 0 nếu Top 10 không có nhãn 2. Đo mức xuất hiện sớm của kết quả trực tiếp.'],
      ['nDCG@10','DCG = tổng (2^label - 1)/log2(rank+1). Chia cho DCG lý tưởng của Top 10 sắp từ toàn pool chung của query. Nhãn 1 có gain=1; nhãn 2 có gain=3.'],
      ['Recall_pool@10','Phụ lục: số nhãn 2 tìm được / tổng nhãn 2 trong pool của query. Không phải Recall toàn corpus; Q08 không có nhãn 2 nên Recall undefined.']],[1.15,4.85])
    p('Đánh giá đủ 120 nhóm query–method và 1.200 vị trí Top 10. Mỗi method có đúng 10 rank liên tục cho mỗi query; mọi doc_id đều thuộc pool và text khớp nguồn. Trung bình macro giữ đủ 20 truy vấn cho cả 6 methods, không bỏ query khó để nâng điểm.')
    p('Dùng lại hàm metrics_at_10 trong src/final_system_evaluation.py qua adapter Silver riêng, không đổi yêu cầu HUMAN_INDEPENDENT của evaluator chính. Kiểm tra nDCG độc lập bằng sklearn trên 120 nhóm: sai lệch tối đa 4,44×10^-16. P/MRR được đối chiếu trực tiếp; thêm hai ví dụ tính tay, gồm zero relevance. Hash của corpus pool, rankings, human labels và metrics cũ không đổi.')
    p('nDCG bằng 0 nếu IDCG=0; không đổi nhãn thiếu thành 0. Không tính accuracy/F1 của nhãn AI vì chưa có nhãn chuẩn để đối chiếu. Không tính kappa vì chỉ có một lượt chấm AI, không có người chấm thứ hai độc lập.')
    section('Kết quả từng truy vấn của Full-document hybrid')
    current=per[per.method.eq('BATTERYBERT_FULLDOC_HYBRID')].sort_values('query_id')
    table(['Query','P@10','nDCG@10','MRR@10','Nhãn 2 trong pool'],[[r.query_id,f(r.Precision_at_10),f(r.nDCG_at_10),f(r.MRR_at_10),r.relevant_in_judged_pool] for r in current.itertuples()],[.8,1,1.25,1.15,1.5])
    p('Q08 có nDCG=1 nhưng P@10=MRR@10=0: pool không có bài nhãn 2, nên thứ tự lý tưởng cũng chỉ gồm bài nhãn 1. nDCG đo chất lượng xếp hạng tương đối trong pool, không chứng minh tìm được bài thỏa đủ NaFSI + ether + nhiệt độ thấp.',style='small')
    p('Q20 có một bài nhãn 2 trong pool nhưng full-document hybrid không lấy được vào Top 10. BGE semantic lấy bài này ở rank 10: publication:WOS:000439134600040, nghiên cứu full pouch cell HC/layered oxide với ADN electrolyte additive.',style='small')
    section('Phân bố điểm và các truy vấn cần cải thiện')
    pic('query_heatmaps.png','Hình 2. Cùng nhãn và cùng pool: bên trái relevance trực tiếp, bên phải gain graded. Q08 là ví dụ cần đọc hai chỉ số cùng nhau.',490)
    p('Q09, Q12, Q14: full-document hybrid trả 10/10 kết quả nhãn 2. Q18 chỉ 1/10 kết quả trực tiếp và kết quả đầu ở rank 5; Q17 đạt 2/10; Q04–Q06 đạt 3/10; Q16 đạt 4/10. Cần đọc rationale từng bản ghi trước khi quyết định lỗi nằm ở retrieval hay metadata.',style='small')
    section('Độ nhạy với nhãn và bất định')
    table(['Phương pháp','P@10 gốc','Review→0','Review→2','P@10 nhãn ≥1'],[[ALIASES[r.method],f(r.Precision_at_10),f(sensitivity[sensitivity.method.eq(r.method)&sensitivity.scenario.eq('review_flags_as_0')].iloc[0].Precision_at_10),f(sensitivity[sensitivity.method.eq(r.method)&sensitivity.scenario.eq('review_flags_as_2')].iloc[0].Precision_at_10),f(sensitivity[sensitivity.method.eq(r.method)&sensitivity.scenario.eq('partial_or_direct_threshold1')].iloc[0].Precision_at_10)] for r in summary.itertuples()],[2.7,1,1,1,1.2])
    pic('threshold_sensitivity.png','Hình 3. Đếm cả liên quan một phần làm điểm tăng; đây là câu hỏi đánh giá khác, không thay thế ngưỡng chính.',210)
    p('Hai cột Review→0/2 là kịch bản thay đồng thời 70 nhãn cần review, không phải nhãn mới, CI hay cận toán học bảo đảm cho nDCG. Full-document hybrid có P@10 0,525/0,600 và nDCG 0,7525/0,7734 trong hai kịch bản; kết luận phụ thuộc chất lượng nhãn.')
    p('Bootstrap 10.000 lần theo query, cùng resamples cho mọi method, seed 20261026. CI 95% của full-document hybrid: P@10 [0,395; 0,685], nDCG [0,6793; 0,8534], MRR [0,6117; 0,925]. Khoảng này chỉ phản ánh biến thiên giữa 20 query với nhãn cố định; không đo sai số của AI judge.')
    p('Chênh lệch với BGE và hybrid lịch sử có paired CI chứa 0 cho cả ba chỉ số. Các so sánh là thăm dò, 45 phép so sánh chưa hiệu chỉnh đa kiểm định; không suy ra chiến thắng chắc chắn hoặc ngoài tập query hiện tại.',style='small')
    section('Diễn giải lỗi, giới hạn và file bàn giao')
    table(['Truy vấn','Vấn đề được phát hiện'],[
      ['Q04–Q06','Có đúng họ vật liệu nhưng thiếu điều kiện water/vacancy/carbon coating/low cost; nhiều patent ghi ngược anode/cathode. Không tự sửa metadata khi chấm.'],
      ['Q08','Chưa có nhãn 2 trong pool. Phải mở rộng pooling từ dữ liệu hiện có và/hoặc người dùng cung cấp tài liệu, không kết luận cả corpus không có bài phù hợp.'],
      ['Q16','Nhầm electrode coating với particle coating, film wrapping hoặc công đoạn cutting; cần phân biệt intent trong bước phát triển sau.'],
      ['Q17','BMS/voltage balancing và material safety chỉ là nền; thermal management ở cấp pack cần bằng chứng riêng.'],
      ['Q18','Nhầm tái chế Li-ion/slag/masks để tạo electrode SIB với tái chế chính spent SIB và cathode metals.'],
      ['Q20','Các mảnh HC, layered oxide và additive phải cùng một full cell. Cathode sacrificial additive khác electrolyte additive.']],[1,5])
    p('Các điểm này là chẩn đoán sau đánh giá, chưa được dùng để tuning/retrain hoặc chọn checkpoint. Nếu phát triển Search tiếp, cần tách tập phát triển mới và giữ đánh giá độc lập; không tối ưu trực tiếp trên 703 nhãn vừa dùng để báo cáo.')
    p('Silver Labels là kết quả của một AI assistant, không phải human ground truth hoặc thỏa thuận nhiều người chấm. Confidence high/medium/low chưa hiệu chuẩn. 70 trường hợp review có thể đổi sau xác minh; dữ liệu nguồn có 28 Abstract trống và các role labels đáng nghi. Những giới hạn này áp dụng cho toàn bộ kết luận xếp hạng.')
    table(['File','Nội dung'],[
      ['silver_labels_703.csv','Đúng 7 cột yêu cầu; giữ nguyên 703 query_id–doc_id theo thứ tự nguồn.'],
      ['labels_with_source_703.csv / needs_review.csv','Bản kèm Query–Title–Abstract và danh sách 70 trường hợp review.'],
      ['evaluation/metrics_summary.csv / metrics_per_query.csv','Ba chỉ số chính trên 6 methods và chi tiết 120 nhóm.'],
      ['evaluation/confidence_intervals.csv / paired_deltas.csv','Bootstrap theo query; nhãn AI giữ cố định.'],
      ['evaluation/sensitivity_summary.csv / pooled_recall_supplement.csv','Kiểm tra ngưỡng, review scenarios và Recall trong pool.'],
      ['evaluation/validation.json / labels_frozen_manifest.json','Kiểm chứng số liệu, tính toàn vẹn và hash bộ nhãn.']],[2.5,3.5])
    p('Thư mục gốc: outputs/ai_silver_labels_703_20261026_v1/. Báo cáo này bổ sung kết quả mới cho PDF nghiên cứu v2, thay phần pilot 20 cặp về mức bao phủ AI; trạng thái nhãn người vẫn chưa xác minh. Không ghi đè báo cáo cũ hoặc evaluator human.',style='small')
    def footer(c,doc):
        c.setFont('Arial',8);c.setFillColor(colors.HexColor('#566773'));c.drawString(44,24,'Sodium-ion Research Explorer | AI Silver Labels | 26/10/2026');c.drawRightString(A4[0]-44,24,str(doc.page))
    if PDF.exists() or MD.exists():raise ValueError('Existing unreceipted report; preserve and inspect')
    SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=43,bottomMargin=43).build(story,onFirstPage=footer,onLaterPages=footer)
    MD.write_text('\n'.join(md),encoding='utf8')
    count=len(PdfReader(PDF).pages)
    artifacts=[PDF,MD,*sorted(sources)]
    manifest={'status':'completed','pages':count,'sections':pages,'artifacts':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':h(p)} for p in artifacts]}
    REPORT_MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({'pages':count,'pdf':str(PDF)},ensure_ascii=False))
def main():
    global FIG,NAME,PDF,MD,REPORT_MANIFEST
    sys.stdout.reconfigure(encoding='utf8');p=argparse.ArgumentParser();p.add_argument('--pdf-only',action='store_true');p.add_argument('--revision',choices=['v1','v2'],default='v1');a=p.parse_args()
    if a.revision=='v2':
        FIG=E/'figures_v2';NAME=NAME+'_v2';PDF=ROOT/'output/pdf'/(NAME+'.pdf');MD=ROOT/'docs'/(NAME+'.md');REPORT_MANIFEST=E/'report_manifest_v2.json'
    if a.pdf_only:document()
    else:
        if not REPORT_MANIFEST.exists():figures()
        raise SystemExit(subprocess.call([str(RUNTIME),str(Path(__file__).resolve()),'--pdf-only','--revision',a.revision],cwd=ROOT))
if __name__=='__main__':main()
