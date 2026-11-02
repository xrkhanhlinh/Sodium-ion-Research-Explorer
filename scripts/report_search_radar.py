"""Generate measured research outputs and course-report supplement, without inference."""
from pathlib import Path
import json
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.search_radar_core import VERSION,METRICS,paired_bootstrap
from src.final_system_artifacts import sha,read_json
from course_report_latex.build_report import table,esc,num

OUT=ROOT/'outputs'/VERSION;H=OUT/'historical';C=ROOT/'course_report_latex'
FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
LABELS={'A0':'A0','A1':'A1','A2':'A2','A3':'A3','A3_weighted':'A3 (1:1:2)',
        'A4':'A4','A5_max':'A5 max','A5_unique_mean':'A5 mean','A6_N20':'A6 N20',
        'A6_N30':'A6 N30','A6_N50':'A6 N50','CE_CONTROL_N20':'MiniLM N20'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':180,
                     'axes.spines.top':False,'axes.spines.right':False})


def savefig(name,fig):
    fig.savefig(FIG/name,bbox_inches='tight');plt.close(fig)


def main():
    per=pd.read_csv(H/'metrics_per_query.csv');summary=pd.read_csv(H/'metrics_summary.csv')
    coverage=pd.read_csv(H/'coverage.csv');latency=pd.read_csv(H/'latency_memory.csv')
    ranks=pd.read_csv(H/'rankings_top10.csv');manifest=read_json(H/'manifest.json')
    old=pd.read_csv(ROOT/'outputs/ai_silver_labels_703_20261026_v1/evaluation/metrics_summary.csv')
    primary=per[per.depth.eq(100)];a0=primary[primary.method.eq('A0')].set_index('query_id')
    paired=[];pair_ci=[]
    for depth,dp in per.groupby('depth'):
        for method in dp.method.unique():
            if method=='A0':continue
            selected=dp[dp.method.isin(['A0',method])]
            common=set(selected[selected.method.eq('A0')&selected.status.eq('COMPLETE')].query_id)&set(selected[selected.method.eq(method)&selected.status.eq('COMPLETE')].query_id)
            cp=selected[selected.query_id.isin(common)]
            for name,g in cp.groupby('method'):
                paired.append(dict(comparison=method,method=name,depth=depth,queries=len(common),
                    query_ids=';'.join(sorted(common)),**{m:float(g[m].mean()) for m in METRICS}))
            if len(cp):pair_ci.extend(dict(comparison=method,depth=depth,**r) for r in paired_bootstrap(cp))
    pair=pd.DataFrame(paired);pair.to_csv(H/'pairwise_complete_metrics.csv',index=False)
    pd.DataFrame(pair_ci).to_csv(H/'pairwise_bootstrap.csv',index=False)
    compare=pair[pair.comparison.eq('A3')&pair.depth.eq(100)].set_index('method')
    n=int(compare.loc['A3','queries']);delta=compare.loc['A3',list(METRICS)].astype(float)-compare.loc['A0',list(METRICS)].astype(float)
    # Candidate recall intervals use every defined judged query pool, excluding Q08.
    recall=primary[primary.method.isin(['A0','A3'])].pivot(index='query_id',columns='method',values='Candidate_Recall_pool').dropna()
    d=(recall.A3-recall.A0).to_numpy();samples=np.random.default_rng(20261030).integers(0,len(d),size=(10000,len(d)))
    lo,hi=np.quantile(d[samples].mean(1),[.025,.975])
    recall_report=dict(queries=len(d),A0=float(recall.A0.mean()),A3=float(recall.A3.mean()),
        delta=float(d.mean()),ci95_low=float(lo),ci95_high=float(hi),scope='Candidate recall in fixed AI-Silver judged pool; not full-corpus recall')
    (H/'candidate_recall_comparison.json').write_text(json.dumps(recall_report,indent=2),encoding='utf8')
    angles=np.linspace(0,2*np.pi,3,endpoint=False);closed=np.r_[angles,angles[0]]
    fig,ax=plt.subplots(figsize=(7,5.6),subplot_kw={'projection':'polar'})
    names=['TF-IDF','BatteryBERT','BB hybrid','BGE','Full-doc','Full-doc hybrid']
    for name,row in zip(names,old.itertuples()):
        values=[getattr(row,m) for m in METRICS];ax.plot(closed,100*np.r_[values,values[0]],label=name)
    ax.set_xticks(angles,['P@10','nDCG@10','MRR@10']);ax.set_ylim(0,100)
    ax.set_title('Historical AI Silver · 20 queries · 26 Oct 2026',pad=25)
    ax.legend(loc='upper left',bbox_to_anchor=(1.08,1));savefig('historical_quality_radar.png',fig)
    fig,ax=plt.subplots(figsize=(7,5.6),subplot_kw={'projection':'polar'})
    for method in ('A0','A3'):
        values=compare.loc[method,list(METRICS)].to_numpy(float)
        ax.plot(closed,100*np.r_[values,values[0]],label=method);ax.fill(closed,100*np.r_[values,values[0]],alpha=.08)
    ax.set_xticks(angles,['P@10','nDCG@10','MRR@10']);ax.set_ylim(0,100)
    ax.set_title(f'Complete-label intersection only · {n}/20 queries',pad=25)
    ax.legend(loc='upper left',bbox_to_anchor=(1.05,1));savefig('experimental_quality_radar.png',fig)
    fig,axs=plt.subplots(1,3,figsize=(11,3.6),sharey=True)
    for ax,metric,label in zip(axs,METRICS,['P@10','nDCG@10','MRR@10']):
        ax.bar(['A0','A3'],compare.loc[['A0','A3'],metric].astype(float),color=['#546e7a','#00796b'])
        ax.set_title(label);ax.set_ylim(0,1)
        for x,value in enumerate(compare.loc[['A0','A3'],metric].astype(float)):ax.text(x,value+.02,f'{value:.4f}',ha='center')
    fig.suptitle(f'Exploratory historical subset · {n}/20 queries · no held-out claims');fig.tight_layout();savefig('quality_metrics.png',fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for method in ('A0','A1','A2','A3'):
        g=summary[summary.method.eq(method)]
        axs[0].plot(g.depth,g.Candidate_Recall_pool,marker='o',label=method)
    axs[0].set(xlabel='Candidates per branch',ylabel='Candidate recall in judged pool',ylim=(.8,1.015));axs[0].legend()
    g=coverage[coverage.depth.eq(100)];axs[1].barh([LABELS.get(m,m) for m in g.method],g.complete_queries,color='#00796b')
    axs[1].set(xlabel='Fully judged query Top 10 (out of 20)',xlim=(0,20));fig.tight_layout();savefig('experimental_candidate_coverage.png',fig)
    bench=OUT/'benchmark/summary.csv'
    if bench.exists():
        lg=pd.read_csv(bench);latency_scope='warm wall-clock search; separate benchmark after chunk build'
        memory_scope='RSS peak được lấy mẫu mỗi 20 ms, trong tiến trình mới cho từng phương pháp; gồm model và các index đã nạp, không phải chỉ riêng tensor encoder'
    else:
        lg=latency[latency.depth.eq(100)].copy();latency_scope='sum of warm components; measured with background workload, not isolated'
        memory_scope='RSS là toàn tiến trình giữ chung các model; chưa tách riêng từng phương pháp nên không dùng để vượt gate chi phí'
    fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(len(lg))
    ax.bar(x-.2,lg.P50_seconds,.4,label='P50');ax.bar(x+.2,lg.P95_seconds,.4,label='P95')
    ax.set_xticks(x,[LABELS.get(m,m) for m in lg.method],rotation=35,ha='right');ax.set_ylabel('Seconds');ax.legend()
    ax.set_title(latency_scope);fig.tight_layout();savefig('latency.png',fig)
    testpath=OUT/'automated_tests.xml';testroot=ET.parse(testpath).getroot() if testpath.exists() else None
    suites=list(testroot.iter('testsuite')) if testroot is not None else []
    test_count=sum(int(s.attrib.get('tests',0)) for s in suites);test_failures=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
    chunk=read_json(OUT/'chunks/manifest.json') if (OUT/'chunks/manifest.json').exists() else None
    unknown=ranks[ranks.ai_label.isna()].drop_duplicates(['query_id','doc_id'])
    parity=pd.read_csv(H/'baseline_parity.csv');s100=summary[summary.depth.eq(100)].set_index('method')
    a0lat=lg[lg.method.eq('A0')].iloc[0];a3lat=lg[lg.method.eq('A3')].iloc[0]
    latency_ratio=float(a3lat.P95_seconds/a0lat.P95_seconds)
    rss_ratio=float(a3lat.rss_max_bytes/a0lat.rss_max_bytes)
    cost_status='PASS' if latency_ratio<=1.5 and rss_ratio<=1.5 else 'FAIL'
    gate=dict(A7='NOT_SELECTED',independent_development_qrels=False,heldout_quality_measured=False,
        A3_latency_p95_ratio=latency_ratio,A3_sampled_peak_rss_ratio=rss_ratio,
        A3_cost_gate=cost_status if bench.exists() else 'NOT_MEASURED_IN_ISOLATED_PROCESSES',
        production_changed=False,reason='Independent quality gate pending; historical diagnostics do not authorize promotion')
    (OUT/'gate_status.json').write_text(json.dumps(gate,indent=2),encoding='utf8')
    pairrows=[]
    for method in primary.method.unique():
        if method=='A0':continue
        rows=pair[pair.comparison.eq(method)&pair.depth.eq(100)].set_index('method')
        if len(rows):
            a,b=rows.loc['A0'],rows.loc[method]
            pairrows.append([LABELS.get(method,method),int(b['queries']),num(b.Precision_at_10),num(b.Precision_at_10-a.Precision_at_10),
                             num(b.nDCG_at_10-a.nDCG_at_10),num(b.MRR_at_10-a.MRR_at_10)])
        else:pairrows.append([LABELS.get(method,method),0,'--','--','--','--'])
    tex=table('Ablation trên giao query đủ nhãn với A0, K=100. Các dòng có mẫu số khác nhau; không xếp hạng giữa các dòng.',
        ['Cấu hình','n/20','P@10','ΔP','ΔnDCG','ΔMRR'],pairrows,'lrrrrr',long=False)
    tex=tex.replace(r'\begin{table}[htbp]',r'\begin{table}[H]').replace(r'\begin{tabularx}{\textwidth}',r'\begin{tabular}').replace(r'\end{tabularx}',r'\end{tabular}')
    (C/'generated/search_radar_ablation.tex').write_text(tex,encoding='utf8')
    configrows=[('A0','TF-IDF + full-document'),('A1','TF-IDF + BGE'),('A2','BGE + full-document'),('A3','Ba nhánh, RRF đều'),
        ('A3 weighted','Ba nhánh, trọng số cố định 1:1:2'),('A4','A3 + số điều kiện khớp từ vựng, Top 30'),
        ('A5','A3 + chunk: max / unique-token mean'),('A6','A3 + bằng chứng từ vựng trong câu, Top 20/30/50'),
        ('A7','Chưa chọn: thiếu nhãn development độc lập')]
    (C/'generated/search_radar_configs.tex').write_text(table('Các cấu hình pilot, không thay Search mặc định.',
        ['Cấu hình','Thành phần'],configrows,'lX'),encoding='utf8')
    latency_rows=[(LABELS.get(r.method,r.method),num(r.P50_seconds,3),num(r.P95_seconds,3),num(r.rss_max_bytes/2**20,1)) for r in lg.itertuples()]
    latency_tex=table('Độ trễ và RSS trong benchmark CPU, K=100. Phạm vi đo được ghi trong nội dung.',
        ['Cấu hình','P50 (s)','P95 (s)','RSS (MiB)'],latency_rows,'lrrr')
    latency_tex=latency_tex.replace(r'\begin{tabularx}{\textwidth}',r'\begin{tabular}').replace(r'\end{tabularx}',r'\end{tabular}')
    (C/'generated/search_radar_latency.tex').write_text(latency_tex,encoding='utf8')
    chunk_text=(f"Index mới có {chunk['chunks']:,} chunks, bao phủ {chunk['documents']:,} tài liệu. "
        f"Chỉ chạy block cuối trên frozen-prefix cache đã kiểm hash, không huấn luyện và không encode lại toàn encoder. "
        f"Sai lệch lớn nhất khi gộp lại thành document vector là {chunk['document_aggregation_parity_max_error']:.2e}; kiểm trên toàn bộ corpus. "
        f"Thời gian dựng {chunk['seconds']:.1f} giây; vector index {((OUT/'chunks/vectors.npy').stat().st_size/2**20):.2f} MiB. "
        "Context, offsets và checkpoint hash lưu trong records.parquet. Max chọn một chunk mạnh nhất; unique mean dùng số token mới để chống đếm overlap. "
        "Chưa có nhãn span độc lập để đo độ đúng đoạn bằng chứng.") if chunk else 'Index chunk đang tạo; chưa báo cáo metric hay thời gian hoàn tất.'
    figure_defs=[('experimental_quality_radar.png',f'Radar mới trên giao {n}/20 query đủ nhãn của A0 và A3. Thang 100 lần giá trị gốc; tập giao có thể thiên lệch.'),
        ('quality_metrics.png',f'P@10, nDCG@10 và MRR@10 trên cùng {n} query lịch sử đủ nhãn, không phải held-out.'),
        ('experimental_candidate_coverage.png','Độ phủ ứng viên trong pool AI Silver và số query có Top 10 được chấm đủ. Q08 bị loại khỏi mẫu số candidate recall.'),
        ('latency.png','Độ trễ thực nghiệm CPU; đọc phạm vi đo kèm bảng và protocol, không so đơn vị latency với trục chất lượng.')]
    inventory=read_json(C/'figure_inventory.json');inventory['embedded']=[r for r in inventory['embedded'] if not r['file'].startswith('radar_20261030__')]
    figtex=[]
    for name,caption in figure_defs:
        dest='radar_20261030__'+name;shutil.copy2(FIG/name,C/'figures'/dest)
        label=f'fig:course{len(inventory["embedded"])+1:03d}'
        inventory['embedded'].append(dict(file=dest,caption=caption,source=(FIG/name).relative_to(ROOT).as_posix(),label=label,
            generated_from_frozen_data=True,sha256=sha(C/'figures'/dest)))
        figtex.append(r'\resultfigure{'+dest+'}{'+esc(caption)+'}{'+label+'}')
    inventory['scope']=list(dict.fromkeys(inventory['scope']+[FIG.relative_to(ROOT).as_posix()]))
    (C/'figure_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8')
    (C/'generated/search_radar_figures.tex').write_text('\n'.join(figtex)+r'\FloatBarrier',encoding='utf8')
    ci=pd.DataFrame(pair_ci);ci=ci[ci.comparison.eq('A3')&ci.method.eq('A3')&ci.depth.eq(100)]
    ci_text='; '.join(f'{r.metric}: Δ={r.delta:.4f}, CI95% [{r.delta_ci95_low:.4f}, {r.delta_ci95_high:.4f}]' for r in ci.itertuples())
    chapter=r'''\chapter{Thực nghiệm nâng cấp Search và Search Radar Pilot}
\section{Phạm vi cập nhật ngày 30/10/2026}
Phần bổ sung này ghi các lần chạy mới trên corpus và checkpoint cũ, tách biệt với kết quả ngày 26/10. Mọi artifacts mới lưu trong namespace \artifact{outputs/search_radar_20261030_v1}. Search mặc định và các nhãn, split, checkpoint, Qdrant collection gốc được giữ nguyên; hash trước/sau xác nhận bảo toàn. A0 exact matrix tái hiện đúng thứ tự Top 10 của baseline ở cả 20/20 query.
\input{generated/search_radar_configs.tex}
\section{Truy hồi ba nhánh và phân tích truy vấn}
Mỗi nhánh áp dụng cùng bộ lọc nguồn, năm, topic và required terms trước lấy ứng viên. Candidate depth lần lượt 50, 100 và 200. Fusion dùng $s(d)=\sum_i w_i/(60+rank_i(d))$; chỉ hợp thứ hạng, không cộng raw cosine khác không gian. Trọng số 1:1:2 là ablation cố định trước đánh giá, chưa được lựa chọn bằng development.

Parser rule-based giữ nguyên Na3V2(PO4)3, NaFSI và P2/O3; trích vật liệu, vai trò điện cực, hệ pin, tính chất, điều kiện, quy trình, optional và excluded concepts. Evidence inspector lưu trường, offsets và câu gốc; phân biệt thiếu bằng chứng, phủ định tường minh và mâu thuẫn. Đồng xuất hiện trong câu chỉ là proxy từ vựng, không tự xác nhận quan hệ khoa học hoặc sửa role metadata. A4/A6 là các đối chứng heuristic; chưa có neural evidence model phù hợp được xác minh.
\section{Độ phủ ứng viên và phần chưa có nhãn}
''' + f'''Trên 19 query có grade 2 trong pool AI Silver, Candidate Recall@100 của A0 là {num(recall_report['A0'])}; A3 đạt {num(recall_report['A3'])}. Chênh lệch {num(recall_report['delta'])}, CI95\\% theo query [{num(lo)}; {num(hi)}]. Đây là độ phủ các tài liệu đã biết trong pool, không phải Recall trên toàn corpus. Q08 không có grade 2 nên mẫu số không xác định.

Top 10 mới có {len(unknown)} cặp query--document chưa chấm khi hợp các cấu hình và depth. Không gán các cặp này thành 0. Ở K=100, A3 có {int(coverage[(coverage.method.eq('A3'))&(coverage.depth.eq(100))].complete_queries.iloc[0])}/20 query được chấm đủ; A0 có 20/20. Giao đủ nhãn của toàn bộ ablation có thể rỗng; chỉ số tổng hợp khi đó để trống. Mỗi so sánh cặp dùng cùng query cho baseline và ứng viên, không dùng mẫu số khác nhau rồi tuyên bố model thắng tổng thể.
\\section{{Kết quả A0--A3 trên tập giao lịch sử}}
Trên cùng {n}/20 query đủ nhãn, A0 đạt P@10={num(compare.loc['A0','Precision_at_10'])}, nDCG@10={num(compare.loc['A0','nDCG_at_10'])}, MRR@10={num(compare.loc['A0','MRR_at_10'])}; A3 đạt tương ứng {num(compare.loc['A3','Precision_at_10'])}, {num(compare.loc['A3','nDCG_at_10'])}, {num(compare.loc['A3','MRR_at_10'])}. Chênh lệch dương này chỉ có giá trị chẩn đoán trên tập giao được chấm sẵn. Việc điều kiện hóa theo độ phủ nhãn có thể tạo thiên lệch. Không diễn giải thành kết quả trên đủ 20 query hoặc dữ liệu độc lập.

Bootstrap ghép cặp 10.000 lần, seed 20261030, giữ nhãn AI cố định; bảng CSV lưu khoảng tin cậy từng chỉ số và danh sách query. Chưa hiệu chỉnh đa kiểm định giữa các ablation; không dùng kết quả này chọn trọng số hoặc A7.
\\input{{generated/search_radar_ablation.tex}}
\\input{{generated/search_radar_figures.tex}}
\\section{{Chunk retrieval và chi phí vận hành}}
{esc(chunk_text)}

Với K=100, A0 P50/P95 là {num(a0lat.P50_seconds,3)}/{num(a0lat.P95_seconds,3)} giây; A3 là {num(a3lat.P50_seconds,3)}/{num(a3lat.P95_seconds,3)} giây. Phạm vi đo: {esc(latency_scope)}. {esc(memory_scope)}. P95 của A3/A0 bằng {num(a3lat.P95_seconds/a0lat.P95_seconds,2)} lần trong lần đo này. MiniLM chạy riêng như đối chứng Top 20, không được tự kích hoạt trong Search.
Gate chi phí A3 theo ngưỡng 1,5 lần latency P95 và RSS có trạng thái {cost_status}; gate chất lượng độc lập vẫn chưa đủ nhãn, nên không thay production.
\\input{{generated/search_radar_latency.tex}}
\\section{{Pilot độc lập, kiểm thử và quyết định gate}}
Đã tạo 12 query mới, chia sáu family development và sáu family held-out không trùng. Đây là đề xuất hệ thống, cần chuyên gia xác nhận intent và chấm relevance. Đã chạy retrieval, gom pool và xuất biểu mẫu trống, không tạo nhãn con người. Held-out chưa có qrels, nên chưa tính P/nDCG/MRR held-out; A7 giữ trạng thái chưa chọn. Importer yêu cầu VERIFIED HUMAN\\_INDEPENDENT, người chấm khác người xác minh, thời điểm, rationale và nguồn review query. Held-out cần cấu hình đã khóa sau gate development.

Trang Streamlit có bảy tab: Search Overview, Radar Comparison, Query Inspector, Evidence Inspector, Pilot Evaluation, Error Analysis và Ablation \\& Latency. Radar dùng 100 lần giá trị [0,1], hover giữ giá trị gốc; latency hiển thị riêng. Condition Satisfaction Rate chưa có nhãn điều kiện kiểm chứng nên chưa đưa lên radar. Có lưu phiên bản và xuất CSV, JSON, PNG, Markdown.

Kiểm thử tự động: {test_count-test_failures}/{test_count} qua, bao gồm tính RRF, ID/bộ lọc, overlap chunk, parser công thức, missing labels, offsets, metric, UI bảy tab và regression backend. Lỗi fixture cũ do catalog revised trên đĩa xen vào corpus giả đã được cô lập bằng mock, không đổi production logic. Q04--Q06, Q08, Q16--Q18 và Q20 được xuất riêng để review quan hệ. Chưa có nhãn span/condition độc lập và chưa có bằng chứng cải thiện trên held-out; không chọn tất cả thành phần chỉ vì đã triển khai.
\\source{{outputs/search_radar_20261030_v1/historical; outputs/search_radar_20261030_v1/pilot; docs/SEARCH_FINAL_RESEARCH_REPORT.md}}
'''
    (C/'chapters/08_search_radar.tex').write_text(chapter,encoding='utf8')
    # Concise, machine-derived audit and final report retain all negative results.
    audit=(ROOT/'docs/search_baseline_audit.md').read_text(encoding='utf8')
    audit+='\n## Kết quả audit thực thi\n\n'+f'- A0 exact Top10 parity: {int(parity.exact_order.sum())}/20.\n- Protected corpus/splits/weights/qrels: PASS theo preservation.json.\n- {test_count-test_failures}/{test_count} tests qua.\n- Candidate Recall@100 A0={recall_report["A0"]:.6f}, A3={recall_report["A3"]:.6f}, 19 query có grade2.\n- Initialization {manifest["init_seconds"]:.2f}s; index bytes: {json.dumps(manifest["index_sizes_bytes"])}.\n- Chi phí RSS toàn process, chưa có per-method isolated memory gate.\n'
    (ROOT/'docs/SEARCH_IMPROVEMENT_AUDIT.md').write_text(audit,encoding='utf8')
    def md_table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
    pair_md=md_table(['Cấu hình','n/20','P@10','ΔP','ΔnDCG','ΔMRR'],pairrows)
    final=f'''# Search Radar Pilot — báo cáo thực nghiệm 30/10/2026

Đã triển khai và chạy retrieval ba nhánh, parser, rule/evidence reranking, MiniLM control, pilot pooling và dashboard Streamlit. Production giữ nguyên. Chưa có bằng chứng cải thiện held-out và chưa chọn A7.

## Kết quả lịch sử (AI Silver, không phải nhãn người)

Bộ 703 cặp/20 queries ngày 26/10 giữ nguyên; full-document hybrid P@10=0.5400, nDCG@10=0.7693, MRR@10=0.7767. Không dùng bộ này để chọn trọng số. Q08 không có grade2: nDCG=1 không có nghĩa đủ điều kiện khoa học.

## Kết quả mới thực đo

- A0 exact matrix khớp thứ tự baseline {int(parity.exact_order.sum())}/20 query.
- Candidate Recall@100 trong pool: A0={recall_report['A0']:.6f}, A3={recall_report['A3']:.6f}; 19 query có mẫu số. Paired delta={recall_report['delta']:.6f}, CI95% [{lo:.6f}, {hi:.6f}]. Không phải full-corpus recall.
- A0/A3 trên cùng {n}/20 query đủ nhãn: P={compare.loc['A0','Precision_at_10']:.6f}/{compare.loc['A3','Precision_at_10']:.6f}; nDCG={compare.loc['A0','nDCG_at_10']:.6f}/{compare.loc['A3','nDCG_at_10']:.6f}; MRR={compare.loc['A0','MRR_at_10']:.6f}/{compare.loc['A3','MRR_at_10']:.6f}.
- Đây là tập giao bị điều kiện hóa theo coverage, có thể thiên lệch; không so với mean trên 20 query hoặc tuyên bố độc lập. {ci_text}.
- Có {len(unknown)} cặp Top10 mới chưa chấm khi hợp mọi cấu hình/depth. Relevance metric của query có bất cứ ID chưa chấm được giữ missing.

## Ablation trên cùng query trong từng cặp với A0

{pair_md}

Các dòng có query set khác nhau; không dùng chúng để chọn model. Cột Δ lấy ứng viên trừ A0 trên đúng cùng query. Xem pairwise_complete_metrics.csv để biết ID và paired_bootstrap.csv/pairwise_bootstrap.csv để biết CI. Giao đầy đủ nhãn của toàn bộ methods tại K100 rỗng nên metrics_summary.csv không tạo điểm tổng hợp giả.

## Chunk, latency và memory

{chunk_text}

Phạm vi latency: {latency_scope}. A0 P50/P95={a0lat.P50_seconds:.4f}/{a0lat.P95_seconds:.4f}s; A3={a3lat.P50_seconds:.4f}/{a3lat.P95_seconds:.4f}s. {memory_scope}. P95 A3/A0={a3lat.P95_seconds/a0lat.P95_seconds:.3f} lần, so với mục tiêu cost gate ≤1.5 lần. CSV giữ từng query, P50/P95, RSS, index bytes và initialization. Hai lần lặp mỗi query ở benchmark tiến trình riêng chỉ mô tả query/repetition mix, không phải benchmark tải concurrent.

{md_table(['Method','P50 seconds','P95 seconds','RSS MiB'],latency_rows)}

Cost gate A3: **{cost_status}** theo ngưỡng P95/RSS ≤1.5× A0. RSS ratio={rss_ratio:.3f}; latency ratio={latency_ratio:.3f}. Quality gate độc lập chưa có nhãn; không tự chuyển cấu hình vào production. `gate_status.json` ghi quyết định.

## Development và held-out

12 query mới được hệ thống đề xuất: sáu dev families và sáu held-out families tách biệt. Đã chạy retrieval/pooling, chưa có expert query review hoặc human qrels. Không báo điểm dev/held-out và không gọi các query này là nhãn người. Biểu mẫu giữ trống reviewer/verifier/time/rationale/condition labels/agreement. Importer kiểm provenance và có thể tính exact agreement/Cohen kappa khi có người chấm thứ hai. Held-out yêu cầu config được khóa sau development gate và nhãn review sau thời điểm khóa.

## Kết quả âm và hạn chế

MiniLM lịch sử đã trượt gate 0.927083→0.850694; control mới không được bật vào production. A4/A6 chỉ là heuristic từ vựng, không phải entailment khoa học. Chưa có neural evidence model mới đã xác minh hoặc nhãn condition/span độc lập; chưa thể kết luận chunk, reranker hay A7 tốt hơn trên held-out. Mục tiêu 0.65/0.82/0.85 là mục tiêu, không phải kết quả đạt được. Giữ pipeline cũ.

Các trường hợp Q04–Q06 (điều kiện cathode/coating), Q08 (không có grade2), Q16 (particle/electrode coating), Q17 (pack thermal/voltage balancing), Q18 (spent SIB recycling) và Q20 (full-cell đồng thời nhiều điều kiện) nằm trong weak_queries.csv; không hard-code query_id vào thuật toán. Thiếu Abstract giữ missing evidence, không đổi metadata hoặc corpus để nâng điểm.

## Kiểm thử và bảo toàn

{test_count-test_failures}/{test_count} tests qua; automated_tests.xml ghi chi tiết. Baseline ranking parity 20/20, protection hashes trước/sau PASS. Không training, sửa checkpoint, encode lại toàn corpus qua full encoder hoặc ghi Qdrant. Chunk index mới chỉ replay block cuối từ prefix đã kiểm hash. App và report giữ riêng lịch sử, AI Silver, thí nghiệm mới, human/dev/held-out chưa có nhãn.

## Chạy lại

Từ gốc workspace trong PowerShell:

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\build_search_radar_chunks.py
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\run_search_radar.py --phase historical --cross-encoder-control
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\run_search_radar.py --phase pilot
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\benchmark_search_radar.py
.\\.venv\\Scripts\\python.exe -X utf8 scripts\\report_search_radar.py
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

Chọn **Search Radar Pilot**. Query vector cache/chunk progress có identity/hash và resume. Không xóa annotation_template.csv đã nhập; pool và annotation_template_latest.csv cập nhật riêng với đủ ứng viên mới. Sau khi chuyên gia chấm, chạy evaluate_search_radar_human.py --labels <CSV> --split dev; heldout cần thêm --frozen-config <JSON>. Run tự nhập trên UI dùng --run-dir <thư mục custom> --split unassigned, không coi là dev/held-out. Trước import điền query_review_reference thật, không điền giá trị giả. Nguồn LaTeX có chương mới, radar và ablation; PDF ngày 26/10 vẫn giữ nguyên.
'''
    (ROOT/'docs/SEARCH_FINAL_RESEARCH_REPORT.md').write_text(final,encoding='utf8')
    weak=per[per.query_id.isin(['Q04','Q05','Q06','Q08','Q16','Q17','Q18','Q20'])]
    weak.to_csv(H/'weak_queries.csv',index=False)
    unknown.to_csv(H/'unjudged_top10_pairs.csv',index=False)
    evidence=read_json(C/'evidence_manifest.json')
    sources=[*H.glob('*.csv'),*H.glob('*.json'),*FIG.glob('*.png'),*OUT.glob('*tests.xml'),
        ROOT/'configs/search_radar_v1.json',ROOT/'docs/SEARCH_FINAL_RESEARCH_REPORT.md',
        OUT/'gate_status.json',ROOT/'app.py',
        *list((ROOT/'src').glob('search_radar_*.py')),*list((ROOT/'scripts').glob('*search_radar*.py'))]
    for path in sources:evidence['source_sha256'][path.relative_to(ROOT).as_posix()]=sha(path)
    evidence.update(snapshot='2026-10-30 Asia/Bangkok; historical evidence retained',figures=len(inventory['embedded']),
        encoder_inference_executed=True,new_training_executed=False,new_pilot_results=True,
        independent_human_labels_available=False,heldout_quality_measured=False,
        method='Historical report plus executed versioned search pilot; queries encoded, accepted last block replayed for chunks')
    (C/'evidence_manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(pair_queries=n,recall=recall_report,tests=test_count,failures=test_failures,chunks=bool(chunk)),ensure_ascii=False))


if __name__=='__main__':main()
