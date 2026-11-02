"""Vietnamese research report from saved outputs only; never train or change data.

Figures: project Python with matplotlib. PDF: bundled Codex Python/reportlab.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
from html import escape
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
RUN = 'outputs/batterybert_full_document_20261024_v1'
OLD = 'outputs/batterybert_optimization_20261023_v1'
WORK = ROOT / 'outputs/report_pipeline_20261025'
FIG = WORK / 'figures'
NAME = 'BAO_CAO_PIPELINE_SODIUM_ION_20261025'
PDF = ROOT / 'output/pdf' / (NAME + '.pdf')
MD = ROOT / 'docs' / (NAME + '.md')
SOURCES = set()


def read_json(path):
    p = ROOT / path
    SOURCES.add(p)
    return json.loads(p.read_text(encoding='utf-8'))


def read_csv(path):
    import pandas as pd
    p = ROOT / path
    SOURCES.add(p)
    return pd.read_csv(p)


def n(value):
    return f'{int(value):,}'.replace(',', '.')


def f(value, digits=4):
    return f'{float(value):.{digits}f}'.replace('.', ',')


def pc(value):
    return f'{100*float(value):.2f}%'.replace('.', ',')


def figures():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white'})
    navy, teal, orange = '#244962', '#188477', '#c78035'

    def save(fig, name):
        fig.savefig(FIG / (name + '.png'), dpi=200, bbox_inches='tight', pad_inches=.17)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set(xlim=(0, 12), ylim=(0, 8)); ax.axis('off')
    def box(x, y, w, h, text, color='#edf3f7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',
                     facecolor=color,edgecolor='#97aabd',linewidth=1.1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='#172f40')
    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
                     mutation_scale=13,color='#536d7d',linewidth=1.4))
    box(3.7,6.95,4.6,.7,'20.576 bản ghi Patent và Publication')
    box(3.7,5.7,4.6,.8,'Làm sạch và sàng lọc\n13.702 tài liệu Title–Abstract')
    arrow(6,6.95,6,6.52)
    box(.25,3.9,3.25,1.05,'Từ vựng\nBoW → LDA / TF-IDF')
    box(4.3,3.9,3.4,1.05,'Encoder\nBGE / BatteryBERT + chunks')
    box(8.5,3.9,3.2,1.05,'Split cho neural mới\nTrain / Validation / Test')
    arrow(4.4,5.7,2,5.0); arrow(6,5.7,6,5.0); arrow(7.6,5.7,10,5.0)
    box(.25,1.9,3.25,1.05,'LDA\nPhân bố topic và từ khóa')
    box(4.3,1.9,3.4,1.05,'Topic từ embedding\nBERTopic hoặc KMeans10')
    box(8.5,1.9,3.2,1.05,'Search\nTF-IDF + Qdrant + RRF')
    arrow(1.9,3.9,1.9,3.0); arrow(6,3.9,6,3.0); arrow(7.7,4.1,9.7,3.0)
    arrow(3.5,4.1,9.5,3.0)
    box(.5,.25,11,1.0,'OUTPUT  •  Catalog chủ đề và Top 10 keywords  •  Top 10 bài tìm kiếm\nĐÁNH GIÁ  •  Coherence / stability / source  •  Relevance người chấm / latency', '#e8f3ed')
    arrow(1.9,1.9,2.2,1.3); arrow(6,1.9,6,1.3); arrow(10.1,1.9,9.8,1.3)
    save(fig,'01_pipeline')

    stage1=read_json('outputs/reports/stage1_summary.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    labels=['Patent','Publication']
    x=np.arange(2); w=.32
    axes[0].bar(x-w/2,[stage1['input_counts'][s] for s in labels],w,label='Đầu vào',color=navy)
    axes[0].bar(x+w/2,[stage1['main_corpus_counts'][s] for s in labels],w,label='Corpus chính',color=teal)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Số tài liệu',title='Dữ liệu trước và sau sàng lọc');axes[0].legend(frameon=False)
    splits=read_csv(RUN+'/text_length_summary.csv')
    axes[1].bar(splits['split'],splits['documents'],color=[orange,navy,teal,'#9fb6c1'])
    axes[1].set(title='Phân chia dữ liệu neural',ylabel='Số tài liệu')
    for i,v in enumerate(splits.documents):axes[1].text(i,v+160,n(v),ha='center',fontsize=10)
    axes[1].tick_params(axis='x',rotation=15)
    fig.tight_layout();save(fig,'02_data')

    lda=read_csv('outputs/tables/stage2/lda_base_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(lda.k,lda.c_v,'o-',color=teal);axes[0].scatter([20],[float(lda.loc[lda.k.eq(20),'c_v'].iloc[0])],s=100,color=orange,zorder=3)
    axes[0].set(xlabel='Số topic K',ylabel='C_v',title='LDA seed 42 trên 5 giá trị K',xticks=lda.k)
    axes[1].plot(lda.k,lda.topic_diversity,'o-',color=navy)
    axes[1].set(xlabel='Số topic K',ylabel='Diversity Top 10 keywords',title='Đánh đổi giữa coherence và diversity',xticks=lda.k,ylim=(.60,.85))
    fig.tight_layout();save(fig,'03_lda_selection')

    catalogs=read_csv('docs/TOP10_TOPICS_ALL_MODELS_20261024.csv')
    names=list(catalogs.model.unique())
    short=['Full-document trained','Fine-tuned optimized','Pretrained KMeans','LDA K10','BERTopic–BGE','BERTopic–BatteryBERT']
    fig,axes=plt.subplots(2,3,figsize=(12,6.4))
    for ax,name,title in zip(axes.flat,names,short):
        part=catalogs[catalogs.model.eq(name)].sort_values('topic_id')
        ax.bar(part.topic_id,part.documents,color=teal if 'Full-document' in title else navy)
        ax.set(title=title,xticks=[0,3,6,9],xlabel='ID topic',ylabel='Số tài liệu',ylim=(0,8000))
    fig.tight_layout();save(fig,'04_six_distributions')

    historical=read_csv('outputs/final_system/topic_comparison.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4.0))
    labels=['LDA K10','BERTopic\nBGE','BERTopic\nBatteryBERT']
    for ax,column,title in zip(axes,['C_v','Topic_Diversity','Outlier_Rate'],['Coherence C_v','Diversity','Outlier (%)']):
        vals=historical[column].to_numpy(dtype=float)
        if column=='Outlier_Rate':vals=vals*100
        ax.bar(range(3),np.nan_to_num(vals),color=[navy,teal,orange])
        ax.set(title=title,xticks=range(3),xticklabels=labels)
        for i,val in enumerate(vals):
            ax.text(i,(0 if np.isnan(val) else val)+(0.6 if column=='Outlier_Rate' else .015),
                    'N/A' if np.isnan(val) else f'{val:.2f}',ha='center',fontsize=10)
        ax.set_ylim(0,30 if column=='Outlier_Rate' else 1.03)
    fig.tight_layout();save(fig,'05_historical_comparison')

    checkpoints=read_csv(RUN+'/all_validation_checkpoints.csv')
    base=read_json(RUN+'/baseline/validation/metrics.json')['group_mrr_at_10']
    fig,ax=plt.subplots(figsize=(10.5,4.0))
    for key,part in checkpoints.groupby('candidate'):
        ax.plot(part.step,part.group_mrr_at_10,'o-',label=key)
    ax.axhline(base,color='#72808b',linestyle='--',label='Baseline cùng pooling')
    ax.set(xlabel='Optimizer updates',ylabel='Validation Group MRR@10',title='Lựa chọn checkpoint bằng proxy Title → document')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'06_learning_curve')

    effects=read_json(RUN+'/paired_training_effect.json')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for i,row in enumerate(effects):
        axes[0].bar(i-.16,row['baseline_mrr_at_10'],.32,color=navy,label='Baseline' if i==0 else None)
        axes[0].bar(i+.16,row['selected_mrr_at_10'],.32,color=teal,label='Continuation' if i==0 else None)
        low,high=row['cluster_bootstrap_percentile_95_interval'];gain=row['paired_gain']
        axes[1].errorbar(gain,i,xerr=[[gain-low],[high-gain]],fmt='o',capsize=5,color=teal)
    axes[0].set(xticks=[0,1],xticklabels=['Validation','Test'],ylim=(.84,.96),ylabel='Group MRR@10',title='Proxy trên cùng giao thức');axes[0].legend(frameon=False)
    axes[1].set(yticks=[0,1],yticklabels=['Validation','Test'],xlabel='Chênh lệch MRR và khoảng bootstrap 95%',title='Paired gain theo split_group_id',xlim=(0,.05));axes[1].axvline(0,color='gray',linestyle='--')
    fig.tight_layout();save(fig,'07_training_effect')

    dist=read_csv(RUN+'/topics/topic_source_distribution.csv')
    fig,ax=plt.subplots(figsize=(10.5,4.2))
    ax.bar(dist.topic_id,dist.Patent,label='Patent',color=orange)
    ax.bar(dist.topic_id,dist.Publication,bottom=dist.Patent,label='Publication',color=teal)
    ax.set(xticks=range(10),xlabel='ID topic full-document',ylabel='Số tài liệu',title='Phân bố topic theo nguồn trên toàn corpus')
    ax.legend(frameon=False,ncol=2);fig.tight_layout();save(fig,'08_topic_sources')

    keys=read_csv(RUN+'/topics/topic_keywords.csv')
    fig,axes=plt.subplots(2,2,figsize=(11.3,7))
    for ax,topic,title in zip(axes.flat,[2,3,8,9],['Topic 2  Oxide lớp','Topic 3  Điện ly','Topic 8  Hard carbon','Topic 9  Sulfide và selenide']):
        part=keys[keys.topic_id.eq(topic)].sort_values('rank').iloc[::-1]
        ax.barh(part.token,part.weight,color=teal)
        ax.set(title=title,xlabel='Trọng số từ khóa');ax.tick_params(axis='y',labelsize=9)
    fig.tight_layout(w_pad=2.5,h_pad=2);save(fig,'09_keywords')

    oldtest=read_json(OLD+'/final_test.json');newtest=read_json(RUN+'/topics/final_metrics.json')['heldout']
    cv=read_csv(RUN+'/topics/common_coherence.csv')
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.6))
    pairs=[(cv.c_v.tolist(),'C_v chung reference'),([.8,.77],'Keyword diversity'),([oldtest['stability_ari'],newtest['stability_ari']],'ARI trên test')]
    for ax,(vals,title) in zip(axes,pairs):
        ax.bar([0,1],vals,color=[navy,teal]);ax.set(title=title,xticks=[0,1],xticklabels=['23/10','Full-document'],ylim=(0,1.03))
        for i,val in enumerate(vals):ax.text(i,val+.02,f'{val:.3f}',ha='center',fontsize=10)
    fig.tight_layout();save(fig,'10_topic_tradeoffs')

    runs=read_csv(RUN+'/benchmark_runs.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    labels=['Dense full-document','Hybrid full-document']
    methods=['BATTERYBERT_FULLDOC_SEMANTIC','BATTERYBERT_FULLDOC_HYBRID']
    warm=[];cold=[]
    for method in methods:
        part=runs[runs.method.eq(method)];cold.append(float(part.seconds.iloc[0]));warm.append(part.seconds.iloc[1:].tolist())
    axes[0].boxplot(warm,tick_labels=['Dense','Hybrid'],patch_artist=True,boxprops={'facecolor':'#dceee9'})
    axes[0].set(ylabel='Giây / request',title='19 request sau request đầu mỗi method')
    axes[1].bar(['Dense','Hybrid'],cold,color=[navy,teal]);axes[1].set(ylabel='Giây / request',title='Request đầu mỗi method')
    for i,val in enumerate(cold):axes[1].text(i,val+max(cold)*.02,f'{val:.2f}s',ha='center')
    fig.tight_layout();save(fig,'11_latency')

    val=read_csv(RUN+'/topics/validation_comparison.csv')
    fig,ax=plt.subplots(figsize=(10.5,4.1))
    for mode,color in [('center',orange),('source',teal)]:
        part=val[val.geometry.eq(mode)]
        ax.scatter(part.source_cramers_v,part.document_npmi,s=80,c=color,label=mode,alpha=.75)
    sel=read_json(RUN+'/topics/selection.json')
    ax.scatter([sel['source_cramers_v']],[sel['document_npmi']],s=220,marker='*',color=navy,label='Được chọn',zorder=4)
    ax.axvline(.5,color='gray',linestyle='--',label='Gate source V ≤ 0,5')
    ax.set(xlabel='Source Cramér V trên validation',ylabel='Document NPMI trên validation',title='Chọn geometry và từ khóa bằng nhiều tiêu chí')
    ax.legend(frameon=False);fig.tight_layout();save(fig,'12_validation_tradeoff')

    formulas={
      'lda':r'$\theta_d\sim\mathrm{Dir}(\alpha),\quad\phi_k\sim\mathrm{Dir}(\eta),\quad P(w\mid d)=\sum_{k=1}^{K}\theta_{dk}\phi_{kw}$',
      'attention':r'$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$',
      'pool':r'$c_j=\mathrm{norm}\left(\frac{\sum_t m_{jt}h_{jt}}{\sum_t m_{jt}}\right),\quad d=\mathrm{norm}\left(\frac{\sum_j w_jc_j}{\sum_j w_j}\right)$',
      'loss':r'$S_{ij}=q_i^Td_j/\tau,\quad\mathcal{L}_{q\to d}=-\frac{1}{B}\sum_i\log\frac{\exp(S_{ii})}{\sum_{j\in A_i}\exp(S_{ij})}$',
      'ctfidf':r'$w_{c,t}=\sqrt{\frac{n_{c,t}}{\sum_v n_{c,v}}}\,\log\left(1+\frac{A}{f_t}\right)$',
      'geometry':r'$u=\frac{\mu_1-\mu_2}{\|\mu_1-\mu_2\|},\quad z=\mathrm{norm}\left[(x-\mu)-((x-\mu)^Tu)u\right]$',
      'kmeans':r'$\min_{a,\nu}\sum_i\|z_i-\nu_{a_i}\|^2$',
      'keywords':r'$S(c,w)=TF(c,w)\log\left(1+\frac{N}{df(w)}\right)\left(\frac{df(c,w)+1}{df(w)+10}\right)^{0.25}$',
      'npmi':r'$\mathrm{NPMI}(w_i,w_j)=\frac{\log[P(w_i,w_j)/(P(w_i)P(w_j))]}{-\log P(w_i,w_j)}$',
      'rrf':r'$s(q,d)=\frac{q^Td}{\|q\|\|d\|},\quad\mathrm{RRF}(d)=\sum_{m:d\in L_m}\frac{1}{60+\mathrm{rank}_m(d)}$',
      'ndcg':r'$P@10=\frac{\sum_{r=1}^{10}\mathbf{1}[g_r=2]}{10},\quad DCG@10=\sum_{r=1}^{10}\frac{2^{g_r}-1}{\log_2(r+1)},\quad nDCG=\frac{DCG}{IDCG}$'
    }
    for name,formula in formulas.items():
        fig,ax=plt.subplots(figsize=(11,.85));ax.axis('off');ax.text(.5,.5,formula,ha='center',va='center',fontsize=17)
        save(fig,'eq_'+name)
    print(json.dumps({'figures':len(list(FIG.glob('*.png'))),'directory':str(FIG)},ensure_ascii=False))


def build():
    import pandas as pd
    from PIL import Image as PILImage
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    for family,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
        pdfmetrics.registerFont(TTFont(family,'C:/Windows/Fonts/'+file))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
    W,H=A4;left=47;right=47;content=W-left-right
    styles={
      'body':ParagraphStyle('body',fontName='Arial',fontSize=10.3,leading=15.1,spaceAfter=8,textColor=colors.black),
      'small':ParagraphStyle('small',fontName='Arial',fontSize=8.2,leading=11.1,spaceAfter=6,textColor=colors.HexColor('#485562')),
      'h1':ParagraphStyle('h1',fontName='Arial-Bold',fontSize=19,leading=24,spaceAfter=13,textColor=colors.black),
      'h2':ParagraphStyle('h2',fontName='Arial-Bold',fontSize=11.4,leading=15,spaceBefore=9,spaceAfter=6,textColor=colors.black,keepWithNext=True),
      'title':ParagraphStyle('title',fontName='Arial-Bold',fontSize=27,leading=35,spaceAfter=20,textColor=colors.black),
      'cell':ParagraphStyle('cell',fontName='Arial',fontSize=8.5,leading=11.9),
      'cellsmall':ParagraphStyle('cellsmall',fontName='Arial',fontSize=8.05,leading=11.1),
      'head':ParagraphStyle('head',fontName='Arial-Bold',fontSize=8.4,leading=11.5,textColor=colors.white),
    }
    story=[];md=[];page_info=[];figure_count=0;table_count=0
    def safe(text):
        t=escape(str(text)).replace('\n','<br/>')
        return re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',t)
    def p(text,style='body'):
        story.append(Paragraph(safe(text),styles[style]));md.append(str(text)+'\n')
    def sub(text):
        story.append(Paragraph(safe(text),styles['h2']));md.append('### '+text+'\n')
    def section(title):
        if story:story.append(PageBreak())
        story.append(Paragraph(safe(title),styles['h1']));md.append('## '+title+'\n');page_info.append(title)
    def table(headers,rows,widths=None,small=False):
        nonlocal table_count
        table_count+=1
        weights=widths or [1]*len(headers)
        data=[[Paragraph(safe(v),styles['head']) for v in headers]]
        data += [[Paragraph(safe(v),styles['cellsmall' if small else 'cell']) for v in row] for row in rows]
        obj=Table(data,colWidths=[content*x/sum(weights) for x in weights],repeatRows=1,hAlign='LEFT')
        commands=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#294b62')),
                  ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('GRID',(0,0),(-1,-1),.45,colors.HexColor('#d9d9d9')),
                  ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                  ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]
        for row in range(1,len(data)):
            if row%2==0:commands.append(('BACKGROUND',(0,row),(-1,row),colors.HexColor('#f1f5f7')))
        obj.setStyle(TableStyle(commands));story.extend([obj,Spacer(1,9)])
        md.append('| '+' | '.join(map(str,headers))+' |\n|'+'|'.join(['---']*len(headers))+'|')
        md.extend('| '+' | '.join(str(v).replace('\n','; ').replace('|','/') for v in row)+' |' for row in rows);md.append('')
    def fig(name,caption,source=None,maxh=250):
        nonlocal figure_count
        path=FIG/(name+'.png') if not str(name).endswith('.png') else ROOT/name
        SOURCES.add(path)
        w,h=PILImage.open(path).size;scale=min(content/w,maxh/h)
        image=Image(str(path),width=w*scale,height=h*scale);image.hAlign='CENTER'
        figure_count+=1
        text=f'Hình {figure_count}. {caption}'
        elements=[image,Spacer(1,4),Paragraph(safe(text),styles['small'])]
        if source:elements.append(Paragraph(safe('Nguồn output: '+source),styles['small']))
        story.append(KeepTogether(elements));story.append(Spacer(1,5))
        relative=Path(__import__('os').path.relpath(path,MD.parent)).as_posix()
        md.extend([f'![{text}]({relative})',text,('Nguồn output: '+source) if source else '', ''])
    def eq(name,latex_description):
        path=FIG/('eq_'+name+'.png');w,h=PILImage.open(path).size
        scale=min(content/w,48/h)
        story.append(Image(str(path),width=w*scale,height=h*scale));story.append(Spacer(1,6))
        md.extend(['$$',latex_description,'$$',''])
    def source(text):p('Nguồn output: '+text,'small')

    stage1=read_json('outputs/reports/stage1_summary.json')
    lda_summary=read_json('outputs/reports/stage2/stage2_summary.json')
    lda_base=read_csv('outputs/tables/stage2/lda_base_metrics.csv')
    lda_sel=read_json('outputs/reports/stage2/selected_model.json')
    historical=read_csv('outputs/final_system/topic_comparison.csv')
    catalogs=read_csv('docs/TOP10_TOPICS_ALL_MODELS_20261024.csv')
    train=read_json(RUN+'/training_coverage.json');sel=read_json(RUN+'/selection.json')
    topic=read_json(RUN+'/topics/final_metrics.json');topic_sel=read_json(RUN+'/topics/selection.json')
    effects=read_json(RUN+'/paired_training_effect.json')
    check=read_json(RUN+'/verification_summary.json');status=read_json(RUN+'/evaluation_status.json')
    deployment=read_json(RUN+'/deployment.json');latency=read_csv(RUN+'/benchmark_latency.csv')
    sources=read_csv(RUN+'/topics/topic_source_distribution.csv')
    reps=read_csv(RUN+'/topics/representative_documents.csv')
    rankings=read_csv(RUN+'/rankings_top10.csv')
    oldtest=read_json(OLD+'/final_test.json');common=read_csv(RUN+'/topics/common_coherence.csv')
    bgepre=read_json('outputs/reports/stage3/pretrained_test_metrics.json')
    bgeft=read_json('outputs/reports/stage3/finetuned_test_metrics.json')
    tests=read_json(RUN+'/model_reload_parity.json')

    md.append('# Báo cáo pipeline khám phá chủ đề và tìm kiếm tài liệu pin sodium ion\n')
    story.append(Spacer(1,28))
    story.append(Paragraph('BÁO CÁO PIPELINE<br/>KHÁM PHÁ CHỦ ĐỀ<br/>VÀ TÌM KIẾM TÀI LIỆU<br/>PIN SODIUM ION',styles['title']))
    p('Sodium ion Research Explorer','h2')
    p('Tổng hợp mã nguồn và kết quả thực nghiệm đã lưu trong D:\\daln\nNgày biên soạn 25 tháng 10 năm 2026','small')
    p('Báo cáo trình bày toàn bộ đường đi từ dữ liệu Patent và Publication đến LDA, BERTopic, các biến thể BatteryBERT và kết quả Search Top 10. Mỗi phần liên kết phương pháp với bảng output, hình kết quả và giới hạn diễn giải.')
    table(['Dữ liệu hiện có','Đầu ra topic','Đầu ra Search'],[['13.702 Title–Abstract','6 catalog và 60 topic\n600 vị trí từ khóa','20 query benchmark\n703 cặp chờ nhãn người']],[1,1,1])
    sub('Kết quả chính')
    p('LDA Stage2 chọn K = 20; bảng đối chứng dùng LDA K10. BERTopic–BatteryBERT pretrained có coherence lịch sử cao nhưng hai cụm chiếm 92,8% corpus. Nhánh full-document mới đọc đủ Abstract, cải thiện proxy ghép Title–document và tạo catalog KMeans10 đang được ứng dụng sử dụng.')
    p('Test proxy của continuation tăng từ 0,887839 lên 0,924704. Kết quả topic có đánh đổi: coherence, diversity và ARI giảm so với bản optimized trước. Chưa có nhãn relevance độc lập để tuyên bố model Search nào tốt nhất.')
    p('Phạm vi dữ liệu là Title và Abstract, chưa có toàn văn bài báo. Các số đo dưới đây được đọc từ output hiện có; báo cáo này không huấn luyện lại model.','small')

    section('Nội dung báo cáo')
    p('Báo cáo đi theo đường xử lý dữ liệu, giải thích từng họ mô hình, trình bày output và đánh giá sau khi sinh kết quả. Phụ lục giữ đủ catalog để người đọc kiểm tra lại từng topic và danh sách từ khóa.')
    table(['Nội dung','Mục','Trang'],[
      ['Pipeline tổng thể','1','3'],
      ['Dữ liệu và thiết kế thí nghiệm','2 đến 3','4 đến 5'],
      ['LDA và kết quả chọn số topic','4 đến 5','6 đến 7'],
      ['BGE và BatteryBERT','6','8'],
      ['BERTopic và sáu catalog đối chứng','7 đến 9','9 đến 11'],
      ['KMeans và huấn luyện full document','10 đến 13','12 đến 15'],
      ['Output topic và đánh giá Top 10 keywords','14 đến 17','16 đến 19'],
      ['Search Top 10 và đánh giá relevance','18 đến 21','20 đến 23'],
      ['Thảo luận và đối chiếu nguồn output','22 đến 23','24 đến 25'],
      ['Đủ 60 topic và 600 vị trí keywords','Phụ lục A đến F','26 đến 31'],
    ],[3.6,1.1,1])
    sub('Cách đọc các kết quả')
    p('Các bảng có chữ lịch sử mô tả artifacts của lượt chạy trước. Các bảng full-document mô tả lượt continuation ngày 24/10. Những thước đo khác protocol được trình bày riêng để tránh so sánh sai phạm vi.')
    p('Ba loại Top 10 có ý nghĩa khác nhau: 10 topic là số nhóm của catalog; Top 10 keywords mô tả từng topic; Top 10 bài là kết quả truy hồi cho một query. Coherence đánh giá từ khóa, còn relevance có nhãn đánh giá kết quả tìm kiếm.')
    p('Biểu đồ trong báo cáo được dựng lại trực tiếp từ output CSV và JSON đã lưu. Các công thức được render thành hình để giữ đúng ký hiệu trong PDF; bản Markdown đi kèm giữ biểu thức và đường dẫn ảnh để có thể chỉnh sửa.','small')

    section('1 Bài toán và pipeline tổng thể')
    p('Mục tiêu thứ nhất là tổ chức tài liệu thành các nhóm chủ đề để người dùng khảo sát hướng nghiên cứu. Mục tiêu thứ hai là trả về các bài liên quan tới truy vấn. Hai mục tiêu chia sẻ dữ liệu và encoder nhưng sử dụng tiêu chí đánh giá khác nhau.')
    fig('01_pipeline','Pipeline từ dữ liệu đến hai đầu ra topic và Search. Sơ đồ được dựng từ các module hiện có.','src/pipeline.py; src/model_comparison.py; src/final_search.py',maxh=332)
    p('Huấn luyện, encode corpus và xây chỉ mục chạy offline. Khi khởi động, ứng dụng đọc artifacts đã chuẩn bị. Lúc tìm kiếm, hệ thống chỉ xử lý query, áp dụng bộ lọc, truy hồi và hợp nhất thứ hạng. Topic hỗ trợ duyệt và lọc, không được cộng như điểm thưởng vào ranking.')
    table(['Giai đoạn','Input','Output cụ thể'],[
      ['Chuẩn bị dữ liệu','Excel Patent và Publication','main_corpus.parquet; lda_tokens.parquet; bert_texts.parquet'],
      ['Topic lexical','Tokens và BoW','LDA posterior; keywords; representatives'],
      ['Topic neural','Embeddings BGE hoặc BatteryBERT','Assignments; catalog topic; diagnostics'],
      ['Search','Query và bộ lọc','Top 10 bài; cosine; RRF; thời gian'],
    ],[1.1,1.25,2.2])

    section('2 Dữ liệu và sàng lọc chất lượng')
    p('Hai tập nguồn có tổng cộng 20.576 bản ghi. Clean master giữ provenance của toàn bộ đầu vào; corpus chính dùng 13.702 tài liệu sau các quy tắc sàng lọc. Nhãn CORE và RELATED là heuristic, không phải đánh giá khoa học độc lập.')
    table(['Nguồn','Đầu vào','Corpus chính','Giữ lại'],[
      [s,n(stage1['input_counts'][s]),n(stage1['main_corpus_counts'][s]),pc(stage1['main_corpus_counts'][s]/stage1['input_counts'][s])] for s in ['Patent','Publication']
    ],[1.4,1,1,1])
    fig('02_data','Số tài liệu trước và sau sàng lọc và phân chia dữ liệu cho nhánh neural mới.','outputs/reports/stage1_summary.json; '+RUN+'/text_length_summary.csv',maxh=228)
    dispositions={}
    for row in stage1['dispositions']:dispositions[row['disposition']]=dispositions.get(row['disposition'],0)+row['count']
    table(['Disposition','Số bản ghi','Diễn giải'],[
      ['MAIN_CORPUS',n(dispositions['MAIN_CORPUS']),'Dùng trong corpus chính'],
      ['RELEVANCE_REVIEW',n(dispositions['RELEVANCE_REVIEW']),'Cần rà soát mức liên quan'],
      ['OUT_OF_SCOPE',n(dispositions['OUT_OF_SCOPE']),'Ngoài phạm vi theo heuristic'],
      ['LANGUAGE_REVIEW',n(dispositions['LANGUAGE_REVIEW']),'Cần kiểm tra ngôn ngữ'],
      ['RETRACTED / DUPLICATE / TEXT',n(sum(dispositions.get(k,0) for k in ['RETRACTED','CONFIRMED_DUPLICATE','TEXT_QUALITY_REVIEW'])),'4 thu hồi; 3 trùng xác nhận; 2 chất lượng văn bản'],
    ],[1.6,.8,2.2])
    p('Quy trình gồm sửa cấu trúc bảng, chuẩn hóa metadata và văn bản, kiểm tra ngôn ngữ, thu hồi, trùng lặp và phạm vi. Giữ doc_id, nguồn, năm, identifier và split_group_id; không tạo Abstract giả cho bản ghi thiếu.','small')

    section('3 Biểu diễn văn bản và thiết kế thí nghiệm')
    p('Nhánh lexical dùng chuỗi token đã xử lý và Dictionary chung. LDA sử dụng số đếm Bag-of-Words. Nhánh neural giữ văn bản để tokenizer của encoder xử lý; không đưa chuỗi đã bỏ toàn bộ ngữ cảnh vào Transformer.')
    table(['Dữ liệu','Số tài liệu','Vai trò'],[
      ['Train','10.921','Optimizer; fit geometry, KMeans và vocabulary của nhánh inductive mới'],
      ['Validation','1.361','Chọn learning rate, checkpoint và recipe topic'],
      ['Test','1.368','Đánh giá sau khi chọn; không đưa vào optimizer'],
      ['Title only','52','Chỉ inference; không tham gia contrastive training'],
    ],[1,1,3])
    p('Grouped split giữ các tài liệu cùng split_group_id trong một tập nhằm hạn chế rò rỉ giữa train, validation và test. Việc này không chứng minh đã nhận diện mọi trường hợp trùng ý hoặc mọi patent family.')
    sub('Phân biệt hai phạm vi đánh giá')
    p('LDA và BERTopic đối chứng lịch sử là khám phá trên corpus đã đóng băng; coherence toàn corpus mô tả cấu trúc quan sát được. Nhánh optimized và full-document mới fit geometry, KMeans và từ vựng trên train, dùng validation để chọn và test để đánh giá. Hai giao thức không nên bị gọi chung là một phép thử held-out.')
    table(['Loại output','Ý nghĩa điểm','Không suy ra được'],[
      ['LDA posterior','Tỷ trọng topic theo mô hình xác suất','Độ đúng khoa học của tên topic'],
      ['BERTopic affinity','Cosine BoW với c-TF-IDF, chuẩn hóa L1','Posterior đã hiệu chuẩn'],
      ['Dense cosine / RRF','Điểm dùng để xếp hạng tài liệu','Xác suất bài trả lời đúng query'],
      ['Proxy Title → document','Khả năng tìm lại cặp hoặc nhóm tài liệu','Relevance query tự nhiên độc lập'],
    ],[1.2,1.8,1.8])
    source('configs/stage2.yaml; configs/stage3.yaml; configs/model_comparison.yaml; '+RUN+'/training_protocol.json')

    section('4 LDA và bản chất mô hình chủ đề xác suất')
    p('LDA giả định mỗi tài liệu là hỗn hợp nhiều topic và mỗi topic là một phân bố từ. Dữ liệu quan sát là các từ; biến ẩn là topic của từng vị trí từ và tỷ trọng topic trong tài liệu. Nền tảng phương pháp xem tài liệu tham khảo [1].')
    eq('lda',r'\theta_d\sim Dir(\alpha),\quad\phi_k\sim Dir(\eta),\quad P(w\mid d)=\sum_k\theta_{dk}\phi_{kw}')
    p('θ_dk là tỷ trọng topic k trong tài liệu d; φ_kw là xác suất từ w trong topic k. α và η là prior Dirichlet. Với từng vị trí từ, LDA giả định chọn topic từ θ_d rồi chọn từ từ φ của topic đó. Gensim suy luận các biến ẩn bằng phương pháp biến phân.')
    sub('Pipeline LDA trong dự án')
    p('Tokens Stage1 → Dictionary không lọc thêm → BoW → fit LDA → posterior tài liệu–topic → lấy argmax để hiển thị một nhãn chính → lấy 10 từ có P(w|topic) cao nhất → chọn tài liệu đại diện có posterior cao.')
    table(['Tham số','Giá trị','Tác động'],[
      ['Số topic K','10, 15, 20, 25, 30','Kiểm tra nhiều mức chi tiết'],
      ['passes / iterations','10 / 100','Ngân sách cập nhật và suy luận'],
      ['alpha / eta','asymmetric / symmetric','Prior tài liệu–topic và topic–word'],
      ['Seeds','42; thêm 123 và 2026','Đo ổn định ở hai cấu hình triển vọng'],
      ['Top keywords / C_v window','10 / 110 token','Cố định đầu ra và protocol coherence'],
    ],[1.5,1.35,2])
    p('BoW bỏ thứ tự từ, nên LDA khai thác đồng xuất hiện và phân bố từ thay vì quan hệ ngữ cảnh đầy đủ. Một tài liệu có thể thuộc nhiều chủ đề trong posterior, dù giao diện hiển thị topic trội nhất. Tên chủ đề là bước diễn giải sau mô hình.')
    source('src/stage2.py; configs/stage2.yaml; src/model_comparison.py')

    section('5 Kết quả LDA và lựa chọn số topic')
    fig('03_lda_selection','Kết quả năm cấu hình LDA seed 42. Chấm cam đánh dấu K20 được chọn ở Stage2.','outputs/tables/stage2/lda_base_metrics.csv',maxh=235)
    table(['K','C_v','U_Mass','Diversity'],[[str(int(r.k)),f(r.c_v),f(r.u_mass),f(r.topic_diversity)] for r in lda_base.itertuples()],[.5,1,1,1])
    p('Stage2 chọn K20, seed 42, với C_v = '+f(lda_summary['selected_metrics']['c_v'])+', diversity = '+f(lda_summary['selected_metrics']['topic_diversity'])+' và stability Jaccard sau ghép Hungarian = '+f(lda_summary['selected_metrics']['stability'])+'. Stability này khác định nghĩa ARI của KMeans và không được so trực tiếp như cùng một chỉ số.')
    p('Quy tắc chọn: sàng lọc hai K theo C_v ở seed cơ sở, sau đó kết hợp C_v chuẩn hóa 50%, U_Mass chuẩn hóa 15%, diversity 15% và stability 20%. Đây là lựa chọn trong lưới đã thử, không phải tối ưu toàn cục.')
    p('Catalog LDA K10 trong phần so sánh giữ K bằng các nhánh khác. K10 có C_v = 0,537876 ở protocol lịch sử; khác kết quả K20 Stage2. Trong K10, hai topic lớn nhất có 3.559 và 3.486 tài liệu, cho thấy các nhóm tương đối rộng.')
    source('outputs/reports/stage2/selected_model.json; outputs/tables/stage2/lda_comparison.csv; docs/TOP10_TOPICS_ALL_MODELS_20261024.csv')

    section('6 BGE và BatteryBERT trong pipeline neural')
    p('BGE và BatteryBERT là encoder; BERTopic và KMeans là các cách khai thác embedding để tạo topic. BGE được dùng cho biểu diễn và truy hồi; BatteryBERT đưa tri thức ngôn ngữ miền pin vào encoder. Checkpoint gốc được ghi tại [3] và [4].')
    eq('attention',r'Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V')
    p('Attention tạo hidden state phụ thuộc ngữ cảnh. Q và K xác định trọng số tương tác; V cung cấp thông tin được tổng hợp. Vector tài liệu được tạo từ hidden state qua pooling và chuẩn hóa. Encoder tốt cho dự đoán ngôn ngữ miền pin chưa tự đảm bảo khoảng cách tài liệu phù hợp với mọi mục tiêu phân cụm.')
    table(['Nhánh','Đặc điểm thực thi','Output'],[
      ['BGE Stage3','bge-base-en-v1.5; cập nhật block cuối; proxy độ dài 128','Checkpoint và embeddings; BERTopic lịch sử'],
      ['BatteryBERT pretrained','batterybert-cased; chưa cập nhật theo corpus dự án','Embeddings chunked và catalog đối chứng'],
      ['BatteryBERT fine-tuned cũ','Title–Abstract contrastive; train tối đa 128 token','Checkpoint dùng cho optimization 23/10'],
      ['BatteryBERT full-document','Continuation block cuối; tất cả chunks Abstract','Checkpoint và 13.702 embeddings mới'],
    ],[1.4,2.35,1.5])
    sub('Kết quả proxy BGE lịch sử')
    table(['Phép thử','Pretrained','Fine-tuned'],[['Test Group MRR@10',f(bgepre['group_mrr_at_10'],6),f(bgeft['group_mrr_at_10'],6)],['Queries / candidates','1.368 / 1.368','1.368 / 1.368'],['Độ dài proxy','128 token','128 token']],[2,1,1])
    p('Đây là proxy Title → Abstract trong cùng split lịch sử. Không so trực tiếp điểm BGE 128 token với proxy full-document có Title context để kết luận encoder nào tốt hơn về Search.','small')
    source('outputs/reports/stage3/pretrained_test_metrics.json; outputs/reports/stage3/finetuned_test_metrics.json')

    section('7 BERTopic từ embedding đến topic và từ khóa')
    p('BERTopic kết hợp biểu diễn ngữ nghĩa và mô tả từ vựng. Pipeline sử dụng embeddings đã encode trước, UMAP giảm chiều, HDBSCAN tìm cụm theo mật độ, rồi c-TF-IDF để mô tả từ khóa [2, 5].')
    table(['Bước','Cấu hình thực tế','Ý nghĩa'],[
      ['Embedding','BGE hoặc BatteryBERT; chunked khi tương ứng','Vị trí tài liệu trong không gian vector'],
      ['UMAP','15 neighbors; 5 chiều; min_dist 0; cosine; seed 42','Tạo không gian để phân cụm'],
      ['HDBSCAN','min_cluster_size 30; min_samples 10; eom','Tìm cụm và outlier −1'],
      ['c-TF-IDF','Frozen vocabulary; reduce_frequent_words=True','Xếp hạng từ khóa trong cụm'],
      ['Gộp topic','Gộp về 10 nếu ban đầu có nhiều hơn','Giữ nguyên outlier để so sánh K10'],
    ],[1.1,2.3,1.7])
    eq('ctfidf',r'w_{c,t}=\sqrt{n_{c,t}/\sum_v n_{c,v}}\log(1+A/f_t)')
    p('n_ct là số lần từ t xuất hiện trong cụm c; A là trung bình tổng token của các tài liệu cấp cụm; f_t là tổng tần suất từ qua các cụm. Căn bậc hai của TF giảm sự áp đảo của từ quá thường xuyên. Đây là công thức tương ứng nhánh c-TF-IDF mặc định có reduce_frequent_words, không áp dụng thay cho công thức keywords của nhánh optimized.')
    p('Trong bản xuất so sánh, ma trận document–topic của BERTopic là cosine affinity giữa BoW và c-TF-IDF, chuẩn hóa L1. Nó không phải posterior xác suất như LDA. Assignments và outlier lấy từ clustering đã lưu, không suy lại bằng argmax affinity.')
    p('Hình UMAP hai chiều phục vụ hiển thị. Clustering được fit trong không gian UMAP năm chiều; khoảng cách và độ tách cụm trên hình hai chiều chỉ có tính mô tả.','small')
    source('src/model_comparison.py; src/stage3_bertopic_model.py; configs/model_comparison.yaml')

    section('8 Kết quả BERTopic đối chứng lịch sử')
    fig('05_historical_comparison','So sánh lịch sử LDA K10 và hai nhánh BERTopic. C_v và diversity cao không đủ để xác nhận chất lượng khoa học.','outputs/final_system/topic_comparison.csv',maxh=224)
    table(['Model','C_v','Diversity','Outlier'],[
      ['LDA K10',f(historical.iloc[0].C_v),f(historical.iloc[0].Topic_Diversity),'N/A'],
      ['BERTopic–BGE',f(historical.iloc[1].C_v),f(historical.iloc[1].Topic_Diversity),pc(historical.iloc[1].Outlier_Rate)],
      ['BERTopic–BatteryBERT pretrained',f(historical.iloc[2].C_v),f(historical.iloc[2].Topic_Diversity),pc(historical.iloc[2].Outlier_Rate)],
    ],[2.6,.8,.85,.85])
    p('BERTopic–BGE có các nhóm Prussian blue, điện ly, TiO2/titanate và MXene, nhưng 3.120 tài liệu là outlier. Tỷ lệ này mô tả phần corpus không được đưa vào topic, không tự chứng minh tất cả outlier là dữ liệu kém.')
    p('BERTopic–BatteryBERT pretrained có 109 outlier và C_v cao nhất trong ba nhánh lịch sử. Tuy nhiên, hai cụm lớn nhất gồm 7.261 và 5.449 tài liệu, tổng 92,76% corpus. Chúng được diễn giải là nhóm publication về hiệu năng và nhóm patent về phương pháp chế tạo. Đây là dấu hiệu các nhóm lớn có thể bị chi phối bởi nguồn hoặc cách viết.')
    p('Catalog cũng còn nhóm sodium cobaltate và siêu dẫn. Cần rà soát phạm vi corpus và tài liệu đại diện trước khi gọi đây là các hướng nghiên cứu pin sodium ion đã được xác nhận.')
    p('Nhánh BERTopic–BatteryBERT fine-tuned có trong code nhưng báo cáo model_comparison lịch sử ghi not_completed tại thời điểm xuất. Không dùng kết quả BatteryBERT KMeans để thay thế cho nhánh BERTopic chưa có kết quả hoàn tất.','small')
    source('docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; outputs/model_comparison_v2/reports/MODEL_COMPARISON_REPORT.md')

    section('9 Sáu catalog và cấu trúc phân bố topic')
    fig('04_six_distributions','Số tài liệu theo topic ở sáu catalog. Cùng thang tung giúp thấy sự tập trung ở BERTopic–BatteryBERT.','docs/TOP10_TOPICS_ALL_MODELS_20261024.csv',maxh=325)
    rows=[]
    outliers=[0,0,0,0,3120,109]
    short=['Full-document trained','Fine-tuned optimized','Pretrained KMeans','LDA K10','BERTopic–BGE','BERTopic–BatteryBERT']
    for name,label,out in zip(catalogs.model.unique(),short,outliers):
        part=catalogs[catalogs.model.eq(name)]
        rows.append([label,n(part.documents.max()),pc(part.documents.nlargest(2).sum()/13702),n(out)])
    table(['Catalog','Cụm lớn nhất','Hai cụm / corpus','Outlier'],rows,[2.3,1,1.2,.7])
    p('Topic ID chỉ có nghĩa trong từng catalog. Topic 0 của LDA không tương ứng topic 0 của BatteryBERT. Bảng 10 topic ở phụ lục liệt kê toàn bộ nhóm của mỗi catalog theo ID, không xếp hạng 10 chủ đề tốt nhất.')
    p('KMeans gán mọi tài liệu vào một trong K cụm, vì vậy outlier bằng 0 là đặc điểm thuật toán. Với LDA, tài liệu vẫn có posterior hỗn hợp dù bảng catalog chỉ đếm topic trội nhất.','small')

    section('10 Hiệu chỉnh hình học và BatteryBERT KMeans')
    p('Nhánh optimized kiểm tra giả thuyết embedding bị chi phối bởi nguồn. Phép source projection fit trên train loại thành phần nằm trên hướng chênh lệch trung bình hai nguồn rồi chuẩn hóa vector.')
    eq('geometry',r'u=(\mu_1-\mu_2)/\|\mu_1-\mu_2\|,\quad z=norm[(x-\mu)-((x-\mu)^Tu)u]')
    p('μ là trung bình toàn bộ train; μ1 và μ2 là trung bình theo nguồn. Phép biến đổi chỉ loại một hướng tuyến tính; hướng này có thể chứa cả phong cách viết lẫn nội dung có ích. Do đó cần đánh giá tác động thay vì mặc định đây là khử toàn bộ thiên lệch.')
    eq('kmeans',r'\min_{a,\nu}\sum_i\|z_i-\nu_{a_i}\|^2')
    p('KMeans tìm 10 tâm cụm sao cho tổng bình phương khoảng cách từ mỗi tài liệu đến tâm được gán nhỏ nhất. K = 10 là cấu hình tổ chức catalog, không phải kết luận lĩnh vực có đúng 10 chủ đề tự nhiên.')
    fig('12_validation_tradeoff','Các recipe center/source của nhánh full-document trên validation. Điểm NPMI cao vẫn có thể trượt gate lệch nguồn.',''+RUN+'/topics/validation_comparison.csv',maxh=230)
    p('Có 12 recipe: hai geometry × ba mức exclusivity × hai mức phrase boost. Cấu hình được chọn là source_ex0.25_pb1.0. Các cấu hình center có source Cramér V khoảng 0,988 trên validation; vì vậy không được chọn chỉ vì NPMI cao.','small')

    section('11 Huấn luyện BatteryBERT đọc đủ Abstract')
    p('Nhánh ngày 24/10 tiếp tục train từ checkpoint fine-tuned đã có. Mục tiêu là mọi chunk Abstract đóng góp vào biểu diễn tài liệu và gradient của block cuối, thay cho train chỉ trên phần đầu 128 token.')
    table(['Thành phần','Cấu hình'],[
      ['Cửa sổ / overlap','256 / 64 token; ngân sách Abstract phụ thuộc Title và special tokens'],
      ['Block cập nhật','Block cuối; 11 block đầu đóng băng, cache float32'],
      ['Batch loss / microbatch chunk','32 cặp / 2 chunks; gradient caching hai tầng'],
      ['Temperature / distillation','0,05 / 0,02'],
      ['Learning rate được chọn','5e-6; pilot thêm 2e-6'],
      ['Epoch / optimizer','Một lượt train; AdamW; warmup/decay; clipping norm 1'],
    ],[1.6,3.3])
    eq('pool',r'c_j=norm(\sum_t m_{jt}h_{jt}/\sum_t m_{jt}),\quad d=norm(\sum_j w_jc_j/\sum_jw_j)')
    p('m_jt chọn token Abstract và loại SEP; Title vẫn là context trong encoder nhưng không trực tiếp vào mean pooling của chunk Abstract. w_j là số token mới được bao phủ. Overlap vẫn ảnh hưởng hidden state, vì vậy weighting này không tương đương mỗi token chỉ ảnh hưởng đúng một lần.')
    eq('loss',r'S_{ij}=q_i^Td_j/\tau,\quad L_{q\to d}=-(1/B)\sum_i\log[\exp(S_{ii})/\sum_{j\in A_i}\exp(S_{ij})]')
    p('Loss lấy trung bình hai chiều Title → document và document → Title, có mask false negatives cùng group hoặc title rất giống; thêm khoảng cách bình phương tới embedding tham chiếu. Gradient caching tính loss chung của batch rồi replay từng nhóm chunk, giữ tập negatives lớn với RAM giới hạn [6]. Dropout tắt để replay nhất quán.','small')
    source('configs/batterybert_full_document.yaml; src/batterybert_full_document.py')

    section('12 Output huấn luyện và lựa chọn checkpoint')
    fig('06_learning_curve','Validation Group MRR@10 theo optimizer updates của hai learning rate. Đây là proxy Title → document có Title context.',''+RUN+'/all_validation_checkpoints.csv',maxh=245)
    table(['Output thực thi','Giá trị'],[
      ['Tài liệu train đã tham gia',n(train['train_documents'])],
      ['Chunk đã backpropagate',n(train['actual_chunks_backpropagated'])],
      ['Token Abstract duy nhất theo tài liệu',n(train['actual_unique_abstract_tokens'])],
      ['Số lần val và test vào optimizer',str(train['validation_and_test_optimizer_appearances'])],
      ['Checkpoint được chọn','Step '+str(sel['best_step'])+'; learning rate '+str(sel['learning_rate'])],
      ['Toàn corpus được encode lại','13.702 vector tài liệu'],
    ],[3,1.6])
    p('Checkpoint được chọn sau 342 optimizer updates và đã thấy toàn bộ 10.921 tài liệu của lượt continuation. Validation chọn learning rate và checkpoint; test không dùng để chọn. Test vẫn là split lịch sử đã được sử dụng trước đó, không phải external test mới.')
    p('Các output chứng minh độ bao phủ và thay đổi trọng số thật: training_coverage.json, training_document_coverage.csv, selection.json và embedding_success.json. Kiểm thử kỹ thuật kiểm tra gradient cache, ảnh hưởng phần cuối văn bản và parity giữa cache với full forward.')
    source(RUN+'/training_coverage.json; '+RUN+'/selection.json; '+RUN+'/embedding_success.json')

    section('13 Hiệu quả continuation trên proxy cố định')
    fig('07_training_effect','Baseline và checkpoint mới trên cùng full-document pooling; khoảng bootstrap dùng split_group_id.',''+RUN+'/paired_training_effect.json',maxh=238)
    table(['Split','Baseline','Sau train','Gain','Khoảng 95% của gain'],[
      [r['split'],f(r['baseline_mrr_at_10'],6),f(r['selected_mrr_at_10'],6),f(r['paired_gain'],6),f(r['cluster_bootstrap_percentile_95_interval'][0],6)+' đến '+f(r['cluster_bootstrap_percentile_95_interval'][1],6)] for r in effects
    ],[.85,1,1,1,1.8])
    p('So sánh ghép cặp giữ cùng dữ liệu và pooling, vì vậy gain phù hợp để mô tả tác động của continuation trong protocol này. Bootstrap 10.000 lần theo split_group_id phản ánh biến động lấy mẫu có điều kiện trên model và tập đã dùng.')
    p('Khoảng validation chịu ảnh hưởng lựa chọn checkpoint trên chính validation. Khoảng test không bao hàm toàn bộ uncertainty do annotation, lựa chọn thiết kế trước đó hoặc training seed.')
    sub('Giới hạn ý nghĩa')
    p('Document chunks chứa Title context trong các block đóng băng. Do đó proxy Title → document có thể hưởng lợi từ thông tin Title. Điểm test 0,924704 là Group MRR@10 của phép ghép cặp, không phải Search chính xác 92,47%.')
    p('Không quy mọi chênh lệch giữa proxy 128 token cũ và proxy full-document mới cho training, vì độ dài và pooling khác nhau. Kết quả trong bảng này sử dụng baseline cùng protocol để tránh nhầm lẫn đó.')

    section('14 Output catalog topic full document')
    fig('08_topic_sources','Phân bố nguồn trong 10 topic đang kích hoạt. Topic 6 chủ yếu gồm Patent.',''+RUN+'/topics/topic_source_distribution.csv',maxh=233)
    current=catalogs[catalogs.model.eq('BatteryBERT full-document · trained')].sort_values('topic_id')
    table(['ID','Diễn giải topic','Tài liệu'],[[str(int(r.topic_id)),r.topic_label,n(r.documents)] for r in current.itertuples()],[.45,4,.8],small=True)
    p('Cụm lớn nhất chiếm '+pc(topic['largest_topic_share'])+'; hai cụm lớn nhất chiếm '+pc(topic['top2_topic_share'])+'; source Cramér V toàn corpus = '+f(topic['source_cramers_v'])+'. Nhãn là diễn giải tạm thời từ keywords và tài liệu đại diện; agent review chưa phải expert ground truth.','small')

    section('15 Top 10 keywords được sinh và kiểm tra thế nào')
    eq('keywords',r'S(c,w)=TF(c,w)\log(1+N/df(w))[(df(c,w)+1)/(df(w)+10)]^{0.25}')
    p('Công thức trên là recipe cuối của nhánh full-document: TF tương đối trong cụm, IDF theo số tài liệu train, và exclusivity để ưu tiên từ đặc trưng cho cụm. Phrase boost của cấu hình được chọn bằng 1. Mỗi keyword có ít nhất 5 tài liệu train trong cụm hỗ trợ; tránh các từ/cụm từ bao hàm nhau.')
    fig('09_keywords','Top 10 keywords và trọng số ở bốn topic minh họa. Thang điểm mô tả từ khóa, không phải xác suất khoa học.',''+RUN+'/topics/topic_keywords.csv',maxh=355)
    p('Nhánh scientific keyword stream đọc raw fields và giữ ranh giới stopword, dấu câu và Title/Abstract trước khi tạo ngram. Điều này tránh tạo phrase giả do xóa từ ở giữa. Chuỗi token LDA lịch sử đã mất thông tin khoảng cách không thể dùng để xác nhận phrase liền kề trong văn bản gốc.','small')
    source('src/scientific_keyword_stream.py; scripts/evaluate_batterybert_full_document.py; docs/KEYWORD_ADJACENCY_CORRECTION_20261024.md')

    section('16 Đánh giá sau khi đã sinh Top 10 từ khóa')
    p('Điểm dùng để lấy Top 10 keywords không đồng nghĩa với chất lượng topic. Sau khi sinh từ khóa, đánh giá cần dùng thống kê corpus và kiểm tra tài liệu độc lập với thứ tự trọng số đã chọn.')
    eq('npmi',r'NPMI(w_i,w_j)=\log[P(w_i,w_j)/(P(w_i)P(w_j))]/[-\log P(w_i,w_j)]')
    table(['Chỉ số','Dữ liệu tham chiếu','Diễn giải và giới hạn'],[
      ['C_v','Top 10 keywords; cửa sổ 110 token','Coherence gián tiếp; cao không đảm bảo cụm có ý nghĩa khoa học'],
      ['U_Mass','BoW và đồng xuất hiện cấp tài liệu','Cao hơn thường tốt hơn trong cùng protocol'],
      ['Document NPMI','Có/không của keywords trong tài liệu','Dùng chọn recipe mới; khác sliding-window C_v'],
      ['Diversity','Từ khóa khác nhau / 100 vị trí','Ít lặp không tự chứng minh relevance'],
      ['ARI','Assignments qua các seed','Agreement của phân hoạch; không phải accuracy'],
      ['Source Cramér V','Bảng topic × source','Chẩn đoán lệch nguồn; cần đọc nội dung'],
    ],[1.1,1.6,2.6])
    p('Protocol coherence lịch sử dùng tokens Stage1. Bản common-reference dùng quy tắc phrase và dictionary tham chiếu đã sửa cho các danh sách từ khóa. C_v 0,537876 lịch sử của LDA và 0,529681 trong common-reference là hai phép tính khác protocol; không trộn thành một bảng xếp hạng duy nhất.')
    p('Đánh giá chuyên môn cần người đọc xem các tài liệu đại diện, độ đồng nhất nội dung, mức khớp keywords–label và các trường hợp nằm sát biên. Trong dự án hiện tại, nhãn khoa học chưa được chuyên gia xác nhận.')
    source('scripts/evaluate_topic_candidates.py; '+RUN+'/topics/scientific_review_protocol.json')

    section('17 So sánh topic mới với bản optimized trước')
    fig('10_topic_tradeoffs','Các chỉ số có đánh đổi giữa bản optimized 23/10 và full-document 24/10. Không có chiến thắng trên mọi thước đo.',''+RUN+'/topics/common_coherence.csv; '+OLD+'/final_test.json',maxh=225)
    table(['Thước đo','Optimized trước','Full-document'],[
      ['C_v toàn corpus cùng reference',f(common.iloc[0].c_v,6),f(common.iloc[1].c_v,6)],
      ['Keyword diversity','0,80','0,77'],
      ['Test stability ARI',f(oldtest['stability_ari'],6),f(topic['heldout']['stability_ari'],6)],
      ['Test document NPMI',f(oldtest['document_npmi'],6),f(topic['heldout']['document_npmi'],6)],
      ['Test source Cramér V',f(oldtest['source_cramers_v'],6),f(topic['heldout']['source_cramers_v'],6)],
      ['Test top 2 share',pc(oldtest['top2_topic_share']),pc(topic['heldout']['top2_topic_share'])],
    ],[2.4,1.3,1.3])
    p('Nhánh mới có phân bố cân hơn theo top2 share và source V trên test giảm nhẹ, trong khi ARI, NPMI, diversity và C_v giảm. Việc kích hoạt là quyết định theo mục tiêu đọc đầy đủ và cổng kỹ thuật đa tiêu chí, không phải chứng minh catalog mới tốt nhất về khoa học.')
    p('Các gate đặt giới hạn cho source, stability, coherence, độ lớn cụm, coverage và diversity. Promotion gate của output hiện tại đạt; vẫn cần đánh giá chuyên môn độc lập. Các catalog cũ được giữ để đối chứng.')

    section('18 Pipeline Search và tiêu chí xếp hạng Top 10')
    p('Query → lọc nguồn, năm, required terms và topic nếu có → encode query đầy đủ qua các cửa sổ → lấy tối đa 100 ứng viên dense từ Qdrant → lấy tối đa 100 ứng viên TF-IDF → hợp nhất bằng RRF → trả Top 10.')
    eq('rrf',r's(q,d)=q^Td/(\|q\|\|d\|),\quad RRF(d)=\sum_{m:d\in L_m}1/(60+rank_m(d))')
    table(['Nhánh','Bằng chứng dùng xếp hạng','Output điểm'],[
      ['TF-IDF','Từ ngữ trùng giữa query và Title–Abstract','Cosine lexical'],
      ['Dense','Embedding query và document đúng checkpoint','Cosine semantic'],
      ['Hybrid','Đồng thuận thứ hạng lexical và dense','RRF k = 60'],
    ],[1,2.8,1.2])
    p('Cosine của các vector L2-normalized bằng dot product. RRF cộng nghịch đảo thứ hạng, giúp kết hợp hai thang điểm khác nhau. Bài thiếu trong một danh sách không có đóng góp từ danh sách đó. Cả cosine lẫn RRF đều chưa được hiệu chuẩn thành xác suất relevance.')
    p('Ví dụ tính tay: A có lexical rank 1 và dense rank 10 thì RRF = 1/61 + 1/70 ≈ 0,03068. B có rank 3 ở cả hai nhánh thì RRF = 2/63 ≈ 0,03175. B đứng trước do đồng thuận hai nhánh cao hơn.')
    p('Required terms là bằng chứng từ vựng; có đủ token chưa xác nhận quan hệ hóa học, vai trò anode/cathode hay phủ định. Topic chỉ lọc phạm vi tài liệu khi người dùng chọn; không có topic bonus trong ranking.')
    source('src/full_document_retrieval.py; src/full_document_inference.py; src/final_search.py')

    section('19 Output thực tế của một truy vấn Search')
    query='Q01';method='BATTERYBERT_FULLDOC_HYBRID'
    selected=rankings[rankings.query_id.eq(query)&rankings.method.eq(method)].sort_values('rank')
    p('Query Q01: '+str(selected.iloc[0]['query']))
    p('Method: BatteryBERT full-document hybrid. Bảng dưới lấy trực tiếp từ rankings_top10.csv; giữ nguyên thứ hạng và điểm. Đây là kết quả hệ thống đã sinh, chưa phải danh sách đã được người chấm xác nhận relevance.')
    table(['Hạng','Tiêu đề bài','Nguồn năm','RRF'],[
      [str(int(r['rank'])),r['title'],str(r['source'])+' '+str(int(r['year'])) if pd.notna(r['year']) else str(r['source']),f(r['rrf_score'],5)] for _,r in selected.iterrows()
    ],[.45,3.9,1,.7],small=True)
    source(RUN+'/rankings_top10.csv; query_id Q01; method BATTERYBERT_FULLDOC_HYBRID')
    p('DOI/identifier, Abstract và đoạn trích evidence của từng kết quả nằm trong CSV. Đoạn evidence được chọn bằng lexical overlap nên không được gọi là lời giải thích đầy đủ cho quyết định của neural encoder.','small')

    section('20 Đánh giá relevance của Search Top 10')
    p('Sau khi hệ thống sinh Top 10, cần đánh giá từng cặp query–document theo intent người dùng. Protocol hiện tại dùng nhãn 0 không liên quan, 1 liên quan một phần, 2 liên quan trực tiếp; người chấm không thấy method, rank và score, đồng thời cần verifier độc lập.')
    eq('ndcg',r'P@10=\sum_{r=1}^{10}1[g_r=2]/10,\quad DCG@10=\sum_{r=1}^{10}(2^{g_r}-1)/\log_2(r+1),\quad nDCG=DCG/IDCG')
    p('MRR@10 lấy nghịch đảo hạng đầu tiên có grade 2, bằng 0 nếu không có trong Top 10, rồi trung bình theo query. nDCG dùng IDCG của pool chung giữa các method; đây là đánh giá theo pool đã thu thập, không phải qrels đầy đủ toàn corpus.')
    table(['Output đánh giá hiện tại','Giá trị'],[
      ['Số query benchmark',str(status['total_queries'])],
      ['Cặp query–document trong pool',str(status['pool_pairs'])],
      ['Nhãn người được chấp nhận',str(status['accepted_human_labels'])],
      ['Query có pool nhãn hoàn chỉnh',str(status['eligible_queries'])],
      ['Trạng thái',status['status']],
      ['P@10 / nDCG@10 / MRR@10 relevance','Chưa có kết quả hợp lệ để công bố'],
    ],[2.8,2])
    p('Nhãn trống không được xem là 0. Không dùng coherence, cosine, proxy Title–document hoặc AI tự chấm thay cho ground truth người dùng. Không báo Recall@10 vì chưa biết toàn bộ tài liệu relevant trong corpus.')
    p('Các query hiện do tác giả soạn và đã dùng cho khảo sát hệ thống; chưa phải một benchmark chuyên gia bên ngoài. Tập đánh giá đã khóa không được tái dùng làm dữ liệu chọn cấu hình sau khi có nhãn.')
    source(RUN+'/evaluation_status.json; '+RUN+'/annotation_protocol.json; src/research_completion_evaluation.py')

    section('21 Hiệu năng chạy và kiểm chứng kỹ thuật')
    fig('11_latency','Độ trễ từ requests benchmark đã lưu. Request đầu tách riêng để không lẫn với thống kê warm.',''+RUN+'/benchmark_runs.csv',maxh=230)
    table(['Method','Requests warm','Median giây','Mean giây','Max giây'],[
      ['Hybrid' if 'HYBRID' in r.method else 'Dense',str(int(r.count)),f(r.median,3),f(r.mean,3),f(r.max,3)] for r in latency.itertuples()
    ],[1.3,1,1,1,1])
    p('Thống kê warm bỏ request đầu mỗi method, còn 19 requests. Những số này mô tả một máy và workload đã chạy; chưa phải p95 production hay khả năng chịu nhiều người dùng đồng thời. Thứ tự chạy và cache cũng có thể ảnh hưởng chênh lệch giữa dense và hybrid.')
    table(['Kiểm chứng đã lưu','Kết quả','Phạm vi'],[
      ['Backend và web tests',str(check['passed'])+' / '+str(check['distinct_tests'])+' PASS','Kiểm tra các contract đã mã hóa trong tests'],
      ['Model reload parity',str(tests.get('status','PASS')),'Cache prefix + last block so với full forward'],
      ['Coverage train',train['status'],'Mọi train ID và chunk của protocol được xử lý'],
      ['Deployment Search / Topics',str(deployment['activate_search'])+' / '+str(deployment['activate_topics']),'Trạng thái artifacts hiện có'],
    ],[1.7,1.2,2.3])
    p('112 tests PASS là kết quả lần chạy được lưu trong verification_summary.json, không phải tuyên bố vừa chạy lại trong lúc biên soạn báo cáo. Các tests xác minh kỹ thuật, không chứng minh relevance hoặc độ đúng khoa học.','small')

    section('22 Thảo luận và hướng phát triển')
    p('Dự án đã xây được chuỗi xử lý có dữ liệu rõ nguồn, nhiều baseline topic, huấn luyện neural có kiểm tra độ bao phủ, catalog chủ đề và Search hybrid. Bằng chứng mạnh nhất hiện tại là khả năng tái lập output, continuation thật và cải thiện proxy trên cùng giao thức.')
    p('Hạn chế lớn nhất là thiếu đánh giá relevance và nhãn topic khoa học độc lập. Corpus chỉ có Title–Abstract; một số tài liệu có thể nằm ngoài mục tiêu pin sodium ion. Mean pooling qua nhiều chunk có thể làm mất chi tiết quan hệ, và các chunk không có attention toàn cục.')
    sub('Các thí nghiệm tiếp theo có thể kiểm chứng')
    table(['Ưu tiên','Thí nghiệm','Cách xác nhận'],[
      ['1','Hoàn thiện nhãn relevance 0–2','Hai người chấm và verifier; P@10, nDCG@10, MRR@10 trên pool khóa'],
      ['1','Đánh giá topic bởi người có chuyên môn','Đọc tài liệu đại diện và biên; ghi độ đồng nhất, nhãn sai và ngoài phạm vi'],
      ['2','Tách tác động encoder và clustering','Giữ embedding để so KMeans/HDBSCAN; giữ clustering để so BGE/BatteryBERT'],
      ['2','Ablation pooling và source projection','Cùng split, checkpoint và ngân sách; công bố cả kết quả giảm'],
      ['3','Truy hồi nhiều vector theo chunk','Đo chi tiết bị bỏ bởi pooling và đánh đổi latency/index'],
      ['3','Mở rộng toàn văn khi có dữ liệu','Version corpus mới; đánh giá riêng Methods, Results và quan hệ dài'],
    ],[.5,2.2,2.7])
    p('Báo cáo không chọn một model thắng mọi mục tiêu. LDA cung cấp baseline xác suất; BERTopic cung cấp cụm ngữ nghĩa có outlier; các nhánh KMeans tổ chức catalog cố định; Search cần được kết luận bằng relevance có nhãn. Những hướng trên là đề xuất từ chẩn đoán output, chưa phải kết quả đã thực nghiệm.')

    section('23 Đối chiếu output và tài liệu phương pháp')
    table(['Nội dung','Output chính trong workspace'],[
      ['Dữ liệu và kiểm tra','outputs/reports/stage1_summary.json'],
      ['LDA chọn K','outputs/tables/stage2/lda_base_metrics.csv; outputs/reports/stage2/selected_model.json'],
      ['LDA và BERTopic đối chứng','outputs/final_system/topic_comparison.csv; outputs/model_comparison_v2/'],
      ['Sáu catalog và 600 keywords','docs/TOP10_TOPICS_ALL_MODELS_20261024.csv'],
      ['Training và proxy mới',RUN+'/training_coverage.json; selection.json; paired_training_effect.json'],
      ['Topic mới',RUN+'/topics/final_metrics.json; topic_keywords.csv; topic_assignments.csv'],
      ['Search và nhãn',RUN+'/rankings_top10.csv; evaluation_status.json; annotation_pool.csv'],
      ['Hiệu năng và tests',RUN+'/benchmark_latency.csv; verification_summary.json'],
    ],[1.4,3.7],small=True)
    sub('Tài liệu tham khảo phương pháp')
    refs=[
      '[1] Blei, Ng và Jordan. Latent Dirichlet Allocation. JMLR 3, 2003. https://www.jmlr.org/papers/v3/blei03a.html',
      '[2] Grootendorst. BERTopic Neural topic modeling with a class based TF IDF procedure. 2022. https://arxiv.org/abs/2203.05794',
      '[3] BAAI. Model card bge base en v1.5. https://huggingface.co/BAAI/bge-base-en-v1.5',
      '[4] BatteryData. Checkpoint BatteryBERT cased. https://huggingface.co/batterydata/batterybert-cased',
      '[5] BERTopic documentation. Algorithm và c-TF-IDF implementation. https://maartengr.github.io/BERTopic/algorithm/algorithm.html ; https://maartengr.github.io/BERTopic/api/ctfidf.html',
      '[6] Gao và cộng sự. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup. 2021. https://arxiv.org/abs/2101.06983',
    ]
    for ref in refs:p(ref,'small')
    p('Nguồn phương pháp giải thích thuật toán. Các số đo thực nghiệm trong báo cáo lấy từ artifacts của dự án, không lấy từ các bài báo tham khảo. Manifest kèm theo ghi SHA-256 của những file đã dùng; hình mới được vẽ lại từ output đã lưu, không refit model.','small')

    appendix_names=list(catalogs.model.unique())
    for index,name in enumerate(appendix_names):
        section('Phụ lục '+chr(65+index)+' '+name.replace(' · ',' '))
        part=catalogs[catalogs.model.eq(name)].sort_values('topic_id')
        p('Output catalog gồm đầy đủ 10 topic và 10 keywords mỗi topic. Thứ tự dòng là ID topic; thứ tự từ khóa là rank trong chính model. Không ghép topic giữa các model bằng ID.','small')
        table(['ID','Diễn giải và số tài liệu','Top 10 keywords'],[
          [str(int(r.topic_id)),r.topic_label+'\n'+n(r.documents)+' tài liệu',str(r.top10_keywords).replace('; ', ', ')] for r in part.itertuples()
        ],[.35,2,3],small=True)
        out=[0,0,0,0,3120,109][index]
        p('Tài liệu ngoài 10 topic được gán outlier −1: '+n(out)+'. Nhãn là diễn giải từ khóa và tài liệu đại diện, chưa được chuyên gia xác nhận.','small')
        source('docs/TOP10_TOPICS_ALL_MODELS_20261024.csv; model = '+name)

    PDF.parent.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    def footer(c,doc):
        c.saveState();c.setStrokeColor(colors.HexColor('#d9d9d9'));c.line(left,37,W-right,37)
        c.setFont('Arial',8);c.setFillColor(colors.HexColor('#53616c'))
        c.drawString(left,25,'Sodium ion Research Explorer  |  Kết quả đã lưu đến 25/10/2026')
        c.drawRightString(W-right,25,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=left,rightMargin=right,topMargin=43,bottomMargin=49,
       title='Báo cáo pipeline khám phá chủ đề và tìm kiếm tài liệu pin sodium ion',author='Sodium ion Research Explorer',allowSplitting=True)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader, PdfWriter
    reader=PdfReader(PDF); writer=PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.add_outline_item('Bìa báo cáo',0)
    for index,title in enumerate(page_info):
        writer.add_outline_item(title,index+1)
    pending=PDF.with_suffix('.pending.pdf')
    with pending.open('wb') as handle:writer.write(handle)
    pending.replace(PDF)
    MD.write_text('\n'.join(md),encoding='utf-8')
    sources_extra=['src/pipeline.py','src/stage2.py','src/model_comparison.py','src/batterybert_full_document.py',
       'src/full_document_retrieval.py','src/scientific_keyword_stream.py','scripts/evaluate_batterybert_full_document.py',
       'configs/stage2.yaml','configs/stage3.yaml','configs/model_comparison.yaml','configs/batterybert_full_document.yaml',
       RUN+'/text_length_summary.csv',RUN+'/all_validation_checkpoints.csv',RUN+'/benchmark_runs.csv',
       RUN+'/topics/validation_comparison.csv',RUN+'/topics/topic_keywords.csv',
       OLD+'/common_coherence.csv','outputs/tables/stage2/lda_comparison.csv']
    SOURCES.update(ROOT/p for p in sources_extra)
    manifest=[]
    for path in sorted(SOURCES):
        if path.is_file():
            manifest.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    (WORK/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    meta={'pdf':str(PDF),'markdown':str(MD),'pages':len(PdfReader(PDF).pages),
          'figures':figure_count,'tables':table_count,'sections':page_info,'sources':len(manifest)}
    (WORK/'build_summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['figures','build']);args=parser.parse_args()
    if args.phase=='figures':figures()
    else:build()
