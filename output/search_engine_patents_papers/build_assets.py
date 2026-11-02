"""Read saved artifacts only; build the standalone LaTeX search section assets."""
from pathlib import Path
import hashlib,json,re,shutil,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FULL=ROOT/'outputs/batterybert_full_document_20261024_v1'
SILVER=ROOT/'outputs/ai_silver_labels_703_20261026_v1'
REV=ROOT/'outputs/research_revision_20261025_v1'
SOURCES={};FIGURES=[]
for f in ['figures','tables','build','qa']:(HERE/f).mkdir(exist_ok=True)


def record(p):
    p=Path(p);SOURCES[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return p


def readcsv(p):return pd.read_csv(record(p),encoding='utf-8-sig')
def readjson(p):return json.loads(record(p).read_text(encoding='utf-8-sig'))
def esc(x):
    x=str(x).replace('−','-').replace('–','--').replace('—','---')
    m={'\\':r'\textbackslash{}','_':r'\_','&':r'\&','%':r'\%','#':r'\#','$':r'\$','{':r'\{','}':r'\}'}
    return ''.join(m.get(c,c) for c in x)
def num(x,n=4):return f'{x:.{n}f}'.replace('.',',')


def table(name,caption,headers,rows,spec,source,long=False):
    lines=[r'\begingroup\normalsize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}']
    if long:
        lines += [r'\begin{longtable}{'+spec+'}',r'\caption{'+esc(caption)+r'}\\',r'\toprule '+' & '.join(map(esc,headers))+r'\\\midrule\endfirsthead',r'\toprule '+' & '.join(map(esc,headers))+r'\\\midrule\endhead',r'\bottomrule\endfoot']
    else:
        lines += [r'\begin{table}[htbp]\centering\caption{'+esc(caption)+'}',r'\begin{tabularx}{\textwidth}{'+spec+'}',r'\toprule '+' & '.join(map(esc,headers))+r'\\\midrule']
    lines += [' & '.join(map(esc,r))+r'\\' for r in rows]
    lines += [r'\end{longtable}' if long else r'\bottomrule\end{tabularx}\end{table}',r'\sesource{'+source+'}',r'\endgroup']
    (HERE/'tables'/name).write_text('\n'.join(lines)+'\n',encoding='utf8')


def copyfig(source,name):
    shutil.copy2(record(source),HERE/'figures'/name)
    FIGURES.append(dict(file=name,source=source.relative_to(ROOT).as_posix(),generated=False))


def result_figure(ranks,query,name,topn):
    q=ranks.query('query_id == @query and method == "BATTERYBERT_FULLDOC_HYBRID"').sort_values('rank').head(topn)
    assert len(q)==topn
    plt.rcParams['font.family']='DejaVu Sans'
    if topn==10:figsize=(8.8,10.2);wrap=61;height=.078
    else:figsize=(9.2,6.8);wrap=66;height=.22
    f,ax=plt.subplots(figsize=figsize);ax.axis('off')
    rows=[[str(r.rank),'\n'.join(textwrap.wrap(r.title,wrap)),f'{r.score:.5f}',str(int(r.ai_label))] for r in q.itertuples()]
    t=ax.table(cellText=rows,colLabels=['Rank','Title','RRF','Silver'],colWidths=[.06,.76,.1,.08],loc='center',cellLoc='left')
    t.auto_set_font_size(False);t.set_fontsize(11 if topn==10 else 12)
    for (i,j),c in t.get_celld().items():
        c.set_height(height if i else .055 if topn==10 else .09)
        c.set_edgecolor('#d5dde5');c.set_facecolor('#183b56' if i==0 else '#f0f5f8' if i%2 else 'white')
        if i==0:c.get_text().set_color('white')
    ax.set_title(f'{query} · Full-document hybrid · frozen Top {topn}\nReconstructed from saved output · AI Silver labels',fontsize=13,pad=10)
    f.tight_layout();f.savefig(HERE/'figures'/name,bbox_inches='tight',dpi=180);plt.close(f)
    FIGURES.append(dict(file=name,source='outputs/ai_silver_labels_703_20261026_v1/evaluation/rankings_with_silver_labels.csv',generated=True))


def main():
    corpus=pd.read_parquet(record(ROOT/'data/processed/main_corpus.parquet'))
    assert len(corpus)==13702 and corpus.doc_id.is_unique
    rows=[]
    for source,display in [('Patent','Patents'),('Publication','Papers')]:
        g=corpus.query('source == @source');a=g.abstract.notna()&g.abstract.astype(str).str.strip().ne('')
        rows.append((display,f'{len(g):,}'.replace(',','.'),f'{int(a.sum()):,}'.replace(',','.'),int((~a).sum())))
    table('corpus.tex','Dữ liệu Title--Abstract dùng cho search.', ['Nguồn','Tài liệu','Có Abstract','Thiếu Abstract'],rows,'Xrrr','data/processed/main_corpus.parquet')
    table('directions.tex','Bốn họ phương pháp và trạng thái trong dự án.', ['Hướng','Phương pháp và trạng thái'],[
        ('H1. Lexical','TF-IDF cosine là baseline hiện hành; BM25 thuộc Search v2 lịch sử.'),
        ('H2. Semantic','Dense BatteryBERT pretrained, BGE và BatteryBERT full-document; đối chứng biểu diễn.'),
        ('H3. Hybrid','TF-IDF + dense, hợp RRF60. Full-document hybrid đang là cấu hình mặc định.'),
        ('H4. Hybrid + rerank','MiniLM xếp lại Top30/Top50. Top30 không qua gate và không được bật.')],
        'p{.24\\textwidth}X','src/final_search.py; src/full_document_retrieval.py; outputs/research_revision_20261025_v1/deployment.json')
    effect=readjson(FULL/'paired_training_effect.json')
    table('encoder_results.tex','Group MRR@10 proxy trên cùng full-document pooling.', ['Split','Baseline','Selected','Gain'],
          [(r['split'].capitalize(),num(r['baseline_mrr_at_10'],6),num(r['selected_mrr_at_10'],6),num(r['paired_gain'],6)) for r in effect],
          'Xrrr','outputs/batterybert_full_document_20261024_v1/paired_training_effect.json')
    methods={'TFIDF_COSINE':'TF-IDF','BATTERYBERT_SEMANTIC':'BatteryBERT semantic','BATTERYBERT_HYBRID':'BatteryBERT hybrid','BGE_SEMANTIC':'BGE semantic','BATTERYBERT_FULLDOC_SEMANTIC':'Full-document semantic','BATTERYBERT_FULLDOC_HYBRID':'Full-document hybrid'}
    metrics=readcsv(SILVER/'evaluation/metrics_summary.csv');assert len(metrics)==6 and metrics.queries.eq(20).all()
    table('search_metrics.tex','Kết quả trên703 nhãn AI Silver, 20 queries; chưa xác minh bởi chuyên gia.', ['Phương pháp','P@10','nDCG@10','MRR@10'],
          [(methods[r.method],num(r.Precision_at_10),num(r.nDCG_at_10),num(r.MRR_at_10)) for r in metrics.itertuples()],
          'Xrrr','outputs/ai_silver_labels_703_20261026_v1/evaluation/metrics_summary.csv')
    gate=readcsv(REV/'comparison/search_proxy_comparison.csv')
    rows=[]
    for s in ['dev','gate']:
        q=gate.query('split == @s').set_index('method');rows.append((s.capitalize()+'16',num(q.loc['baseline','group_mrr_at_10'],6),num(q.loc['abstract_rerank30','group_mrr_at_10'],6)))
    table('reranker.tex','Group MRR@10 proxy: baseline và rerank Top30.', ['Tập','Hybrid','Rerank30'],rows,'Xrr','outputs/research_revision_20261025_v1/comparison/search_proxy_comparison.csv')
    latency=pd.concat([readcsv(ROOT/'outputs/research_completion_v1/latency.csv'),readcsv(FULL/'benchmark_latency.csv')])
    table('latency.tex','Độ trễ warm requests đã lưu, đơn vị giây; khác phiên chạy.', ['Phương pháp','n','Median','Max'],
          [(methods[r.method],int(r.count),num(r.median,3),num(r.max,3)) for r in latency.itertuples()],
          'Xrrr','outputs/research_completion_v1/latency.csv; outputs/batterybert_full_document_20261024_v1/benchmark_latency.csv')
    ranks=readcsv(SILVER/'evaluation/rankings_with_silver_labels.csv')
    for qid in ['Q01','Q16']:
        q=ranks.query('query_id == @qid and method == "BATTERYBERT_FULLDOC_HYBRID"').sort_values('rank').head(3)
        table(qid.lower()+'.tex','Ba kết quả đầu '+qid+' của full-document hybrid.', ['Rank','Tiêu đề','Grade'],
              [(r.rank,r.title,int(r.ai_label)) for r in q.itertuples()],
              'p{.08\\textwidth}p{.74\\textwidth}p{.07\\textwidth}',
              'outputs/ai_silver_labels_703_20261026_v1/evaluation/rankings_with_silver_labels.csv',long=True)
        result_figure(ranks,qid,qid.lower()+'_results.png',10 if qid=='Q01' else 3)
    table('work_plan.tex','Các việc tiếp theo và đầu ra cần có.', ['Hạng mục','Công việc và tiêu chí hoàn thành'],[
        ('Nhãn và query mới','Review nhãn bất định; khóa query-family phát triển và giữ riêng; chấm độc lập.'),
        ('Patents / papers','Chạy filter trước retrieval; tạo pool và IDCG riêng cho từng chế độ nguồn.'),
        ('Ablation','Giữ một yếu tố thay đổi: checkpoint/pooling, fusion, candidate budget hoặc reranker.'),
        ('Intent và điều kiện','Kiểm vai trò, công đoạn, cấu hình; tách lỗi ứng viên, xếp hạng và metadata.'),
        ('Vận hành','Đo latency, bộ nhớ, concurrency; kiểm phiên bản, filter và rollback.'),
        ('Bàn giao','PDF và nguồn LaTeX đủ hình, bảng, công thức và sơ đồ; số liệu có nguồn.')],
        'p{.22\\textwidth}p{.70\\textwidth}',
        'docs/KE_HOACH_HOAN_THIEN_SEARCH_ENGINE_PATENTS_PAPERS.md',long=True)
    for p,name in [(FULL/'learning_curve.png','learning_curve.png'),
                   (REV/'visualizations/search_gate_rejection.png','search_gate_rejection.png'),
                   (SILVER/'evaluation/figures_v2/method_comparison.png','method_comparison.png'),
                   (SILVER/'evaluation/figures_v2/query_heatmaps.png','query_heatmaps.png'),
                   (SILVER/'evaluation/figures_v2/threshold_sensitivity.png','threshold_sensitivity.png')]:copyfig(p,name)
    src=[]
    for f in FIGURES:
        src.append(r'\expandafter\def\csname seorigin'+f['file']+r'\endcsname{\sesource{'+f['source']+'}}')
        f['sha256']=hashlib.sha256((HERE/'figures'/f['file']).read_bytes()).hexdigest()
    (HERE/'figure_sources.tex').write_text('\n'.join(src)+'\n',encoding='utf8')
    shutil.copy2(record(ROOT/'course_report_latex/references.bib'),HERE/'references.bib')
    for p in ['docs/KE_HOACH_HOAN_THIEN_SEARCH_ENGINE_PATENTS_PAPERS.md','src/full_document_retrieval.py','src/final_search.py','src/full_document_inference.py','configs/batterybert_full_document.yaml','outputs/research_revision_20261025_v1/deployment.json','outputs/research_revision_20261025_v1/search/src/full_document_rerank.py','outputs/ai_silver_labels_703_20261026_v1/label_summary.json','outputs/ai_silver_labels_703_20261026_v1/evaluation/confidence_intervals.csv','outputs/ai_silver_labels_703_20261026_v1/evaluation/paired_deltas.csv','outputs/ai_silver_labels_703_20261026_v1/evaluation/sensitivity_summary.csv','outputs/batterybert_full_document_20261024_v1/training_coverage.json']:
        record(ROOT/p)
    # Improve prose spacing only, protecting paths, IDs and all math.
    words=['trên','Trên','cùng','có','Có','đủ','đạt','là','điểm','Điểm','grade','batch','microbatch','step','temperature','gain','khoảng','query','queries','nhãn','từ','gain','hạng','rank','baseline','Bootstrap','seed','nDCG','P@10','MRR','trong','thành','pháp','phần','này','các','Các','thời','full-document','Ma trận','chấm','chunk','overlap','Top','RRF','khóa','còn']
    protected=r'\\(?:seartifact|sesource|input|label|cite)\{[^}]*\}|\$[^$]*\$|\\begin\{(?:equation|align)\}[\s\S]*?\\end\{(?:equation|align)\}'
    pattern=re.compile('(?P<p>'+protected+')|(?P<w>\\b(?:'+'|'.join(map(re.escape,words))+'))(?P<n>[0-9])')
    for p in [HERE/'search_engine_section.tex',*sorted((HERE/'tables').glob('*.tex'))]:
        t=p.read_text(encoding='utf8');t=pattern.sub(lambda m:m.group(0) if m.group('p') else m.group('w')+' '+m.group('n'),t);p.write_text(t,encoding='utf8')
    manifest=dict(snapshot='2026-10-26 Asia/Bangkok',body_font='Times New Roman',body_size_pt=13,raster_figures=FIGURES,tikz_diagrams=3,sources=SOURCES,new_training=False,new_inference=False,new_evaluation=False)
    (HERE/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='BUILT',raster_figures=len(FIGURES),tikz_diagrams=3,tables=len(list((HERE/'tables').glob('*.tex'))),sources=len(SOURCES))))


if __name__=='__main__':main()
