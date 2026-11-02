"""Search Radar Pilot page; historical diagnostics and new unlabeled runs stay separate."""
import json
import threading
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from .search_radar_core import METRICS,parse_query,condition_evidence
from .search_radar_retrieval import OUT,ROOT


@st.cache_resource
def pilot_engine():
    from .search_radar_retrieval import RadarRetriever
    return RadarRetriever(),threading.Lock()


def frame(path):
    return pd.read_csv(path,keep_default_na=True) if path.exists() else pd.DataFrame()


def radar_chart(data,axes):
    fig=go.Figure()
    for row in data.to_dict('records'):
        values=[float(row[a]) for a in axes]
        fig.add_trace(go.Scatterpolar(r=[100*v for v in values+[values[0]]],theta=axes+[axes[0]],
            customdata=values+[values[0]],name=row['method'],fill='toself',
            hovertemplate='%{theta}: %{customdata:.4f}<extra>%{fullData.name}</extra>'))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0,100])),height=520)
    return fig


def render():
    st.header('Search Radar Pilot')
    st.caption('Thử nghiệm có phiên bản · Search mặc định giữ nguyên · AI Silver khác nhãn chuyên gia')
    tabs=st.tabs(['Search Overview','Radar Comparison','Query Inspector','Evidence Inspector',
                  'Pilot Evaluation','Error Analysis','Ablation & Latency'])
    folder=OUT/'historical';summary=frame(folder/'metrics_summary.csv');ranks=frame(folder/'rankings_top10.csv')
    per=frame(folder/'metrics_per_query.csv');queries=frame(folder/'queries.csv')
    with tabs[0]:
        st.write('TF-IDF + BGE + BatteryBERT full-document → Weighted RRF → phân tích điều kiện / chunk / evidence.')
        st.info('A7 chưa được chọn: cần development và held-out có nhãn độc lập. Điều kiện khớp từ vựng chưa xác nhận quan hệ khoa học.')
        if len(summary):st.dataframe(summary,hide_index=True,width='stretch')
        else:st.info('Chạy scripts/run_search_radar.py để tạo artifacts thực nghiệm.')
        audit=ROOT/'docs/SEARCH_IMPROVEMENT_AUDIT.md'
        if audit.exists():st.download_button('Audit (.md)',audit.read_bytes(),audit.name)
    with tabs[1]:
        scope=st.radio('Bộ đánh giá',['Lịch sử · 703 AI Silver','Thực nghiệm · cùng query đầy đủ nhãn'],key='radar_scope')
        if scope.startswith('Lịch sử'):
            data=frame(ROOT/'outputs/ai_silver_labels_703_20261026_v1/evaluation/metrics_summary.csv')
            axes=list(METRICS)
            st.caption('20 queries, sáu phương pháp, nhãn AI; đây là kết quả lịch sử ngày 26/10.')
        else:
            depth=st.selectbox('Candidate depth',[50,100,200],index=1,key='radar_depth')
            data=summary[summary.depth.eq(depth)].copy() if len(summary) else pd.DataFrame()
            axes=[]
            st.caption('Chỉ tính relevance trên giao query có đủ nhãn của mọi phương pháp. Candidate recall chỉ trong pool có nhãn; Q08 không có mẫu số grade 2.')
        if len(data):
            methods=st.multiselect('Phương pháp',data.method.tolist(),default=data.method.tolist()[:4],key='radar_methods_'+scope)
            data=data[data.method.isin(methods)]
            if not scope.startswith('Lịch sử') and len(data):
                selected_per=per[per.depth.eq(depth)&per.method.isin(methods)]
                common=set(queries.query_id)
                for method in methods:
                    common &= set(selected_per[selected_per.method.eq(method)&selected_per.status.eq('COMPLETE')].query_id)
                for metric in METRICS:
                    data[metric]=data.method.map(selected_per[selected_per.query_id.isin(common)].groupby('method')[metric].mean())
                data['query_coverage']=len(common)/len(queries)
                data['common_evaluated_queries']=len(common)
                axes=[m for m in (*METRICS,'Candidate_Recall_pool','query_coverage') if data[m].notna().all()]
                st.caption(f'Giao query đủ nhãn của các phương pháp đã chọn: {len(common)}/{len(queries)}. Tập này có thể bị thiên lệch do độ phủ nhãn; không đại diện toàn bộ query.')
            if len(data) and len(axes)>=3:st.plotly_chart(radar_chart(data,axes),width='stretch')
            else:st.info('Chưa đủ ba trục hợp lệ để vẽ radar; các giá trị đã đo được giữ trong bảng.')
            st.dataframe(data,hide_index=True,width='stretch')
        st.caption('Mỗi trục = 100 × giá trị gốc [0,1]; hover giữ giá trị gốc. Không tính điểm tổng hợp. Condition Satisfaction Rate chưa có nhãn kiểm chứng nên chưa hiển thị.')
        for name in ('historical_quality_radar.png','experimental_candidate_coverage.png','quality_metrics.png'):
            path=OUT/'figures'/name
            if path.exists():st.download_button('Tải '+name,path.read_bytes(),name,'image/png')
    with tabs[2]:
        if len(queries) and len(ranks):
            qid=st.selectbox('Query',queries.query_id.tolist(),key='radar_query')
            q=queries.set_index('query_id').loc[qid,'query'];st.write(q);st.json(parse_query(q),expanded=False)
            methods=st.multiselect('So sánh Top 10',sorted(ranks.method.unique()),default=['A0','A3'],key='inspector_methods')
            selected=ranks[ranks.query_id.eq(qid)&ranks.depth.eq(100)&ranks.method.isin(methods)]
            st.dataframe(selected[['method','rank','doc_id','title','ai_label','original_rank','rank_change','missing_conditions','provenance']],hide_index=True,width='stretch')
            st.dataframe(per[per.query_id.eq(qid)&per.depth.eq(100)&per.method.isin(methods)],hide_index=True,width='stretch')
        else:st.info('Chưa có bảng xếp hạng.')
    with tabs[3]:
        if len(ranks):
            qid=st.selectbox('Evidence query',queries.query_id.tolist(),key='evidence_query')
            subset=ranks[ranks.query_id.eq(qid)&ranks.depth.eq(100)].drop_duplicates('doc_id')
            doc=st.selectbox('Tài liệu',subset.doc_id.tolist(),key='evidence_doc')
            row=subset.set_index('doc_id').loc[doc];st.write(row.title)
            st.json(condition_evidence(parse_query(row.query),row.title,row.abstract),expanded=True)
            st.caption('Offsets theo trường Title/Abstract gốc. Missing/ambiguous khác contradicted. Quan hệ cần người đọc xác minh.')
    with tabs[4]:
        pilot=frame(OUT/'pilot/queries.csv')
        if len(pilot):st.dataframe(pilot,hide_index=True,width='stretch')
        st.caption('Các query mới là đề xuất chưa được chuyên gia xác nhận. Dev và held-out có family tách biệt; không có nhãn người được tự tạo.')
        with st.form('new_pilot_search'):
            query=st.text_input('Truy vấn tiếng Anh',value='NaFSI ether electrolytes for low-temperature sodium-ion batteries')
            family=st.text_input('Query family',value='custom_review_pending')
            category=st.selectbox('Loại query',['materials','conditions','processes','equipment','multi-constraint'])
            source=st.selectbox('Nguồn',['All','Patent','Publication'],key='pilot_source')
            use_year=st.checkbox('Lọc năm',key='pilot_use_year')
            year_range=st.slider('Khoảng năm',1900,2026,(2020,2026),key='pilot_years')
            topic=st.selectbox('Topic',['All',*range(10)],key='pilot_topic')
            required=st.text_input('Từ bắt buộc, cách nhau bằng dấu chấm phẩy')
            depth=st.selectbox('Top K ứng viên',[50,100,200],index=1,key='pilot_depth')
            methods=st.multiselect('Cấu hình',['A0','A1','A2','A3','A3_weighted','A4','A5','A6'],default=['A0','A3'],key='pilot_methods')
            submitted=st.form_submit_button('Chạy và lưu pilot')
        if submitted:
            try:
                if not methods or not family.strip():raise ValueError('Cần family và ít nhất một cấu hình')
                engine,lock=pilot_engine()
                with st.spinner('Đang so sánh các cấu hình...'),lock:
                    results=[engine.search(query,m,depth,source=source,years=year_range if use_year else None,
                        required_terms=[t.strip() for t in required.split(';') if t.strip()],topic_id=None if topic=='All' else topic) for m in methods]
                stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
                saved=OUT/'custom'/stamp;saved.mkdir(parents=True,exist_ok=False)
                payload=dict(version=stamp,query_id=stamp,family=family,category=category,created_at=datetime.now(timezone.utc).isoformat(),results=results,
                    label_status='PENDING_REVIEW',split='unassigned')
                (saved/'run.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
                flat=[];candidate_rows=[]
                for result in results:
                    flat.extend(dict(query_id=stamp,query=query,category=category,split='unassigned',method=result['method'],depth=depth,**row) for row in result['results'])
                    for branch,hits in result['candidates'].items():
                        candidate_rows.extend(dict(query_id=stamp,depth=depth,branch=branch,rank=i,doc_id=d,score=s) for i,(d,s) in enumerate(hits,1))
                pd.DataFrame(flat).to_csv(saved/'rankings_top10.csv',index=False,encoding='utf-8-sig')
                pd.DataFrame(candidate_rows).to_csv(saved/'candidates.csv',index=False,encoding='utf-8-sig')
                pd.DataFrame([dict(query_id=stamp,query=query,family=family,category=category,split='unassigned')]).to_csv(saved/'queries.csv',index=False)
                if flat:
                    pool=pd.DataFrame(flat).drop_duplicates(['query_id','doc_id'])[['query_id','query','category','split','doc_id','title','abstract','source','year']].copy()
                    for name in ('relevance_grade','annotator_id','verifier_id','verification_reference','label_method','label_status','reviewed_at','rationale','query_review_reference','condition_labels','agreement'):pool[name]=''
                    pool.to_csv(saved/'candidate_annotation_pool.csv',index=False,encoding='utf-8-sig')
                    pool.to_csv(saved/'annotation_template.csv',index=False,encoding='utf-8-sig')
                st.session_state['radar_live']=payload
            except (ValueError,FileNotFoundError) as error:st.error(str(error))
        if 'radar_live' in st.session_state:
            payload=st.session_state['radar_live'];flat=[]
            for result in payload['results']:
                st.subheader(result['method']);st.write(f"{result['seconds']:.3f} s · {result['eligible_documents']} tài liệu thỏa bộ lọc")
                st.dataframe(pd.DataFrame(result['results']).drop(columns=['evidence','provenance'],errors='ignore'),hide_index=True,width='stretch')
                flat.extend(dict(query_id=payload['query_id'],query=result['query'],method=result['method'],**row) for row in result['results'])
            st.download_button('Kết quả JSON',json.dumps(payload,ensure_ascii=False,indent=2),payload['version']+'.json')
            if flat:
                export=pd.DataFrame(flat);st.download_button('Kết quả CSV',export.to_csv(index=False).encode('utf-8-sig'),payload['version']+'.csv')
                template=export.drop_duplicates('doc_id')[['query_id','query','doc_id','title','abstract','source','year']].copy()
                for name in ('relevance_grade','annotator_id','verifier_id','verification_reference','label_method','label_status','reviewed_at','rationale','query_review_reference','condition_labels','agreement'):template[name]=''
                st.download_button('Biểu mẫu chấm relevance',template.to_csv(index=False).encode('utf-8-sig'),payload['version']+'_annotation.csv')
        for phase in ('historical','pilot'):
            for name in ('annotation_template_latest.csv','annotation_template.csv','candidate_annotation_pool.csv','protocol.json'):
                path=OUT/phase/name
                if path.exists():st.download_button(phase+' · '+name,path.read_bytes(),phase+'_'+name)
        st.caption('Reviewer, verifier, thời điểm, rationale, condition labels và agreement được nhập trong biểu mẫu; importer kiểm provenance trước đánh giá. Nhãn điều kiện trống không được suy ra từ parser.')
    with tabs[5]:
        if len(per):
            weak=['Q04','Q05','Q06','Q08','Q16','Q17','Q18','Q20']
            st.dataframe(per[per.depth.eq(100)&per.query_id.isin(weak)],hide_index=True,width='stretch')
            st.warning('Q08 không có grade 2 trong pool: nDCG=1 không có nghĩa đã thỏa các điều kiện truy vấn.')
        st.write('Particle/electrode coating, electrolyte/sacrificial cathode additive, spent SIB recycling, pack thermal management và full/half cell cần đọc quan hệ trong Abstract.')
    with tabs[6]:
        latency=frame(folder/'latency_memory.csv')
        if len(latency):
            selected=latency[latency.depth.eq(100)]
            st.plotly_chart(px.bar(selected,x='method',y=['P50_seconds','P95_seconds'],barmode='group'),width='stretch')
            st.dataframe(latency,hide_index=True,width='stretch')
        st.caption('Độ trễ là tổng thành phần warm đo tuần tự trên CPU; không gồm queue/UI. RSS là toàn tiến trình đang giữ chung models, không phải RAM riêng từng phương pháp. Không dùng RSS này để vượt gate tự động.')
        report=ROOT/'docs/SEARCH_FINAL_RESEARCH_REPORT.md'
        if report.exists():st.download_button('Báo cáo thực nghiệm (.md)',report.read_bytes(),report.name)
