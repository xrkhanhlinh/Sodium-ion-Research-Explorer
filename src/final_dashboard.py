"""Final-system views mounted in the existing Streamlit app shell."""
import json
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .final_system_artifacts import ROOT, OUT, TOPICS, topic_data, read_json
from .final_search import METHODS,RESEARCH_METHODS,RESEARCH_OUT
from .research_completion import TOPIC_NAME
from . import optimized_topics
from . import full_document_topics
from . import revised_topics


@st.cache_resource(show_spinner=False)
def final_service():
    from .final_search import FinalSearchService
    return FinalSearchService()


@st.cache_data(show_spinner=False)
def topics(name):
    if name==revised_topics.NAME:return revised_topics.topic_data()
    if name==full_document_topics.NAME:return full_document_topics.topic_data()
    if name==optimized_topics.NAME:
        return optimized_topics.topic_data()
    if name==TOPIC_NAME:
        from .research_completion import topic_data as research_topics
        return research_topics()
    return topic_data(name)


def csv_frame(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def search_page(filters, document):
    st.header('Semantic Search · Top 10')
    st.write('Tìm tài liệu Patent và Publication bằng truy vấn tiếng Anh.')
    source,years=filters('search')
    available=list(RESEARCH_METHODS)
    from . import full_document_inference
    if full_document_inference.ready():available+=list(full_document_inference.METHODS)
    if st.checkbox('Hiện phương pháp lịch sử V2/V3',key='search_legacy'):
        available+=['SEARCH_V2','SEARCH_V3']
    research_ready=(RESEARCH_OUT/'topics/topic_labels.csv').exists()
    labels=pd.read_csv(RESEARCH_OUT/'topics/topic_labels.csv').set_index('topic_id').topic_label.to_dict() if research_ready else {}
    if optimized_topics.ready():
        labels=optimized_topics.assignments().drop_duplicates('topic_id').set_index('topic_id').topic_label.to_dict()
    if full_document_topics.ready():
        labels=full_document_topics.assignments().drop_duplicates('topic_id').set_index('topic_id').topic_label.to_dict()
    if revised_topics.ready():
        labels=revised_topics.assignments().drop_duplicates('topic_id').set_index('topic_id').topic_label.to_dict()
    with st.form('search_form'):
        query=st.text_input('English query',key='search_query',placeholder='Hard carbon anodes with high initial Coulombic efficiency')
        required=st.text_input('Từ/công thức bắt buộc, cách nhau bằng dấu ;',key='search_required_terms')
        default=available.index('BATTERYBERT_FULLDOC_HYBRID') if full_document_inference.ready() else 2
        method=st.selectbox('Phương pháp',available,index=default,key='search_method')
        selected_topic=st.selectbox('Lọc theo topic',[None]+list(range(10)),
            format_func=lambda t:'Tất cả topic' if t is None else f'{t} · {labels.get(t,"Topic")}',key='search_topic')
        submit=st.form_submit_button('Tìm kiếm',type='primary')
    descriptions={
        'BATTERYBERT_FULLDOC_HYBRID':'Hybrid FULLDOC: TF-IDF + BatteryBERT full-document/Qdrant → RRF k=60.',
        'BATTERYBERT_FULLDOC_SEMANTIC':'Semantic FULLDOC: BatteryBERT full-document/Qdrant → cosine.',
        'BATTERYBERT_HYBRID':'Hybrid: TF-IDF + BatteryBERT pretrained/Qdrant → RRF k=60.',
        'BATTERYBERT_SEMANTIC':'Semantic: BatteryBERT pretrained/Qdrant → cosine.',
        'BGE_SEMANTIC':'Semantic: BGE pretrained → cosine.',
        'TFIDF_COSINE':'Lexical: TF-IDF → cosine.',
        'SEARCH_V2':'V2 lịch sử: BM25 + BGE → RRF → MiniLM.',
        'SEARCH_V3':'V3 lịch sử: BM25 + BatteryBERT fine-tuned → RRF → MiniLM.'}
    st.caption(descriptions[method]+' Topic chỉ dùng để duyệt và lọc tài liệu; điểm ranking không phải xác suất relevance.')
    if full_document_inference.ready():
        st.caption('Nhánh FULLDOC dùng checkpoint huấn luyện trên toàn bộ Abstract sẵn có và đọc toàn bộ truy vấn bằng nhiều đoạn. Dữ liệu hiện chưa có toàn văn bài báo; không có đánh giá relevance độc lập cho nhánh mới.')
    if revised_topics.ready():
        st.caption('Topic chính dùng KMeans10 với 50 khởi tạo, đã kiểm tra seed giữ riêng; công thức từ khóa được bảo toàn bằng parser mới. Tên topic là diễn giải AI, cần chuyên gia rà soát.')
    elif full_document_topics.ready():
        st.caption('Topic chính: BatteryBERT được train với toàn bộ các đoạn Abstract, dùng catalog mới đã kiểm tra. Nhãn chủ đề vẫn cần chuyên gia thẩm định.')
    elif optimized_topics.ready():
        st.caption('Topic chính: BatteryBERT fine-tuned + phép chiếu giảm ảnh hưởng nguồn + KMeans10; từ khóa đã tối ưu. Encoder topic và encoder Search được ghi riêng.')
    if submit:
        try:
            with st.spinner('Đang tìm kiếm trên corpus thật…'):
                st.session_state['search_response']=final_service().search(query,method,source,years,
                    required_terms=[x.strip() for x in required.split(';') if x.strip()],topic_id=selected_topic)
        except (ValueError,OSError,RuntimeError) as exc:
            st.session_state.pop('search_response',None);st.error(str(exc))
            if method.startswith('BATTERYBERT'):
                st.caption('Nếu collection Qdrant đang được process khác sử dụng, đóng process đó hoặc dùng TF-IDF trong lúc kiểm tra index.')
    response=st.session_state.get('search_response')
    if not response:return
    st.subheader(f"{len(response['results'])} tài liệu · {response['method']}")
    st.write('Query:',response['query'])
    st.caption(f"{response['eligible_documents']:,} tài liệu thỏa bộ lọc · {response.get('system_seconds',response['seconds']):.3f} s toàn request (gồm tải model ở lượt đầu)")
    st.caption(response['score_method']+' · Score không phải xác suất tài liệu phù hợp. Xem nhãn đã xác minh và độ phủ benchmark tại Evaluation.')
    if response.get('query_coverage'):
        cover=response['query_coverage']
        st.caption(f"Đã đọc đủ {cover['tokens']:,} tokens truy vấn qua {cover['chunks']} đoạn; không cắt phần cuối.")
    if response.get('topic_filter') is not None:st.caption(f"Bộ lọc đã áp dụng: Topic {response['topic_filter']}")
    if not response['results']:
        st.info('Không có tài liệu thỏa bộ lọc hoặc không có từ vựng khớp truy vấn. Hãy điều chỉnh bộ lọc/từ khóa.')
        return
    for result in response['results']:
        document(result)
        st.caption(f"Topic {result.get('topic_id','')}: "+result['topic_label']+' · '+result['topic_encoder_variant'])
    frame=pd.DataFrame([{k:v for k,v in r.items() if k!='topics'} for r in response['results']])
    frame['query']=response['query']
    frame['method']=response['method']
    st.download_button('Export Top 10 CSV',frame.to_csv(index=False).encode('utf-8-sig'),'search_top10.csv','text/csv')
    with st.expander('Phân bố trong kết quả'):
        left,right=st.columns(2)
        left.bar_chart(frame.groupby('source').size().rename('Documents'))
        years_frame=frame.dropna(subset=['year'])
        if len(years_frame):right.bar_chart(years_frame.groupby('year').size().rename('Documents'))
        st.bar_chart(frame.groupby('topic_label').size().rename('Documents'))
        st.plotly_chart(px.line(frame,x='rank',y='score',markers=True,labels={'score':response['score_method']}),width='stretch')
        st.caption('Điểm chỉ có ý nghĩa trong phương pháp đang hiển thị; không so sánh giá trị tuyệt đối giữa các thang điểm khác nhau.')


def topic_page(document,catalog):
    st.header('Topic Analysis')
    choices=([revised_topics.NAME] if revised_topics.ready() else [])+([full_document_topics.NAME] if full_document_topics.ready() else [])+([optimized_topics.NAME] if optimized_topics.ready() else [])
    if full_document_topics.available() and not full_document_topics.ready():choices.append(full_document_topics.NAME)
    choices+=([TOPIC_NAME] if (RESEARCH_OUT/'topics/topic_keywords.csv').exists() else [])+list(TOPICS)
    name=st.radio('Mô hình topic',choices,horizontal=True,key='topic_model')
    data=topics(name)
    summary=data['summary'].copy()
    summary['topic_label']=summary.topic_id.map(data['labels'])
    normal=summary[summary.topic_id.ne(-1)]
    metrics=data['metrics']
    cols=st.columns(4)
    for col,label,value in zip(cols,['Topics','C_v','Topic Diversity','Outlier rate'],
        [str(metrics['topics']),f"{metrics['c_v']:.4f}",f"{metrics['topic_diversity']:.4f}",
         'N/A (LDA)' if metrics['outlier_ratio'] is None else f"{metrics['outlier_ratio']:.2%}"]):col.metric(label,value)
    st.caption('Tái sử dụng assignments và metrics đã lưu; không fit lại topic model. C_v không phải chất lượng Search.')
    st.caption('Nhãn diễn giải từ keywords và representative documents, cần chuyên gia rà soát. Topic ID chỉ có ý nghĩa trong từng mô hình.')
    if 'BatteryBERT' in name:
        if name==revised_topics.NAME:
            metric=data['full_metrics']
            st.info('Cùng encoder full-document đã lưu; phân cụm được chạy lại với 50 khởi tạo. Từ khóa giữ công thức thống kê đã có, sửa lỗi ký hiệu hóa học; không áp dụng các cách lọc từ đã trượt kiểm tra.')
            st.caption(f"Gate seed giữ riêng: ARI={metric['heldout']['stability_ari']:.3f}. C_v={metric['c_v']:.3f} dùng reference canonical mới; không so trực tiếp với C_v lịch sử khác protocol. Không phải accuracy khoa học.")
            review_path=revised_topics.CATALOG/'assignment_diagnostics.csv'
            st.download_button('Tải chẩn đoán gán topic',review_path.read_bytes(),'revised_topic_diagnostics.csv','text/csv')
            st.caption('Topic 7 đã đổi nội dung sau phân cụm; ID chỉ là mã. Nhóm phần cứng và các hệ pin natri liên quan vẫn được nêu rõ, không tự xóa để làm đẹp kết quả.')
        elif name==full_document_topics.NAME:
            full=read_json(full_document_topics.OUT/'topics/final_metrics.json')
            st.info('Encoder được tiếp tục train bằng gradient trên toàn bộ các đoạn Abstract sẵn có; không phải toàn văn bài báo. KMeans và từ khóa được fit trên train; nhãn còn tạm thời.')
            if not full_document_topics.ready():st.warning('Catalog ứng viên để đối chiếu; chưa được kích hoạt làm topic chính. Xem cổng đánh giá tại Comparison.')
            st.caption(f"Test: source V={full['heldout']['source_cramers_v']:.3f}; ARI seed mới={full['heldout']['stability_ari']:.3f}. C_v so sánh cùng reference; 0% outlier là thuộc tính KMeans.")
            review_path=full_document_topics.OUT/'topics/review_priority.csv'
            if review_path.exists():
                st.download_button('Tải tài liệu cần rà soát',review_path.read_bytes(),'topic_review_priority.csv','text/csv')
                st.caption('Ưu tiên đọc lại các tài liệu ở xa trọng tâm hoặc gần hai topic; ngưỡng lấy từ train. Đây là gợi ý rà soát, không phải xác suất đúng/sai.')
        elif name==optimized_topics.NAME:
            result=read_json(optimized_topics.OUT/'final_test.json')
            st.info('Pipeline BatteryBERT fine-tuned ngày 23/10. Phép chiếu và KMeans fit trên train; validation chọn cấu hình, test chỉ kiểm tra cuối. 0% outlier là đặc tính KMeans.')
            st.caption(f"Test: source Cramér's V={result['source_cramers_v']:.3f}; ARI seed mới={result['stability_ari']:.3f}; hai cụm lớn nhất={result['top2_topic_share']:.2%}. Nhãn còn tạm thời; có tài liệu lệch nhóm và một nhóm phần cứng pin.")
        elif name==TOPIC_NAME:
            st.info('Giữ nguyên 10 cụm KMeans; chỉ cập nhật từ khóa. Nhãn còn tạm thời; 0% outlier là đặc tính gán cụm của KMeans.')
            st.caption('Ảnh hưởng Patent/Publication và một số topic chồng lấn vẫn còn. Coherence không xác nhận chất lượng khoa học hay relevance Search.')
        else:
            st.info('Kết quả HDBSCAN lịch sử trên embeddings pretrained; không phải 10 cụm KMeans mới.')
            st.caption('Hai cụm lớn nhất chiếm phần lớn corpus và phản ánh mạnh thể loại Patent/Publication.')
    st.plotly_chart(px.bar(normal,x='topic_id',y='documents',hover_data=['topic_label']),width='stretch')
    if summary.topic_id.eq(-1).any():st.caption(f"Outlier -1: {int(summary.loc[summary.topic_id.eq(-1),'documents'].sum()):,} tài liệu, không tính là topic nghiên cứu.")
    selected=st.selectbox('Chọn topic',normal.topic_id.tolist(),format_func=lambda t:f'{t} · {data["labels"][int(t)]}',key='topic_id_'+name)
    keyword=data['keywords'][data['keywords'].topic_id.eq(selected)].sort_values('rank')
    if not st.checkbox('Hiện đủ 15 từ khóa nếu có',key='show15_'+name):keyword=keyword.head(10)
    weight='LDA topic-word probability' if name=='LDA' else 'Topic keyword weight (frequency × IDF × exclusivity)' if name in (optimized_topics.NAME,full_document_topics.NAME,revised_topics.NAME) else 'c-TF-IDF weight'
    st.subheader(f'Top {len(keyword)} keywords và weights')
    st.plotly_chart(px.bar(keyword,x='weight',y='token',orientation='h',labels={'weight':weight}),width='stretch')
    st.dataframe(keyword,hide_index=True,width='stretch')
    if name==revised_topics.NAME:
        with st.expander('Từ khóa cần đọc trong ngữ cảnh'):
            audit=data['keyword_audit'].query('topic_id == @selected')
            st.dataframe(audit,hide_index=True,width='stretch')
            st.caption('Cờ đánh dấu từ rộng, thao tác, cấu trúc hoặc ký hiệu chưa hỗ trợ; không tự kết luận chúng là nhiễu. Một số từ như carbon hoặc electrode vẫn mô tả đúng cụm. Danh sách thống kê không bị giấu hoặc thay đổi thứ tự ở đây.')
    st.subheader('Xu hướng theo năm')
    years=data['years'][data['years'].topic_id.ne(-1)].copy()
    years['topic']=years.topic_id.map(data['labels'])
    st.plotly_chart(px.line(years,x='year',y='documents',color='topic',markers=True),width='stretch')
    st.caption('Số tài liệu theo năm được ghi trong metadata; năm thiếu không được suy diễn hay điền bổ sung.')
    sources=data['sources'][data['sources'].topic_id.eq(selected)]
    st.bar_chart(sources.set_index('source').documents)
    # All radar axes are explicitly defined ratios in [0,1], no invented
    # semantic confidence or incomparable normalized ranking scores.
    axes=['Corpus share','Publication share','Source balance','Recent share (≥2020)']
    radar=go.Figure()
    for t in normal.topic_id:
        size=int(normal.loc[normal.topic_id.eq(t),'documents'].iloc[0])
        s=data['sources'][data['sources'].topic_id.eq(t)].set_index('source').documents.to_dict()
        publication=s.get('Publication',0)/size
        dated=data['years'][data['years'].topic_id.eq(t)]
        dated_total=dated.documents.sum()
        recent=dated.loc[dated.year.ge(2020),'documents'].sum()/dated_total if dated_total else 0.
        values=[size/13702,publication,1-abs(2*publication-1),float(recent)]
        radar.add_trace(go.Scatterpolar(r=values+[values[0]],theta=axes+[axes[0]],name=f'Topic {t}',
                                       visible=True if t==selected else 'legendonly'))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])))
    st.plotly_chart(radar,width='stretch')
    st.caption('Radar: Corpus share = topic/13.702; Publication share = Publication/topic; Source balance = 1−|2×Publication share−1|; Recent share = tài liệu ≥2020/tài liệu có năm. Không phải relevance hay độ tin cậy khoa học.')
    st.subheader('Representative documents')
    for result in data['representatives'][data['representatives'].topic_id.eq(selected)].to_dict('records'):
        meta=catalog().loc[result['doc_id']]
        result['identifier']=meta.identifier
        document(result,score=False)
    st.dataframe(summary,hide_index=True,width='stretch')


def comparison_page(filters,document):
    st.header('Model Comparison')
    if revised_topics.available():
        st.subheader('Topic revision · kiểm tra seed và công thức')
        st.dataframe(csv_frame(revised_topics.CATALOG/'keyword_comparison.csv'),hide_index=True,width='stretch')
        st.caption('Hai danh sách từ khóa được đo trên cùng canonical reference. Tăng coherence không đồng nghĩa expert accuracy; diversity có đánh đổi. Reranker thử nghiệm không được kích hoạt vì trượt gate giữ riêng.')
    if (full_document_topics.OUT/'selection.json').exists():
        st.subheader('BatteryBERT full-document training')
        selection=read_json(full_document_topics.OUT/'selection.json')
        st.write('Validation proxy: paired Title → full document có Title context; không phải relevance Search độc lập.')
        st.dataframe(csv_frame(full_document_topics.OUT/'training_comparison.csv'),hide_index=True,width='stretch')
        if (full_document_topics.OUT/'topics/common_coherence.csv').exists():
            st.dataframe(csv_frame(full_document_topics.OUT/'topics/common_coherence.csv'),hide_index=True,width='stretch')
        st.caption(f"Chấp nhận checkpoint train mới: {selection['accepted']}; phần còn lại cần nhãn khoa học độc lập.")
    if (full_document_topics.OUT/'comparison_rankings_top10.csv').exists():
        st.subheader('Full-document Search · benchmark mới')
        st.dataframe(csv_frame(full_document_topics.OUT/'benchmark_latency.csv'),hide_index=True,width='stretch')
        st.caption('Pool mở rộng gồm kết quả model mới; nhãn người còn thiếu nên chưa kết luận phương pháp thắng.')
    if optimized_topics.ready():
        st.subheader('BatteryBERT topic optimization · test riêng')
        st.dataframe(csv_frame(optimized_topics.OUT/'inductive_test_comparison.csv'),hide_index=True,width='stretch')
        st.caption('Cùng train/validation/test theo nhóm tài liệu. NPMI, cân bằng và lệch nguồn là chẩn đoán topic; không phải relevance Search. Cấu hình mới đã qua tiêu chí kỹ thuật, còn cần thẩm định chuyên gia.')
        with st.expander('Coherence chung và lịch sử thử nghiệm'):
            st.dataframe(csv_frame(optimized_topics.OUT/'common_coherence.csv'),hide_index=True,width='stretch')
            st.dataframe(csv_frame(optimized_topics.OUT/'validation_comparison.csv'),hide_index=True,width='stretch')
    if (RESEARCH_OUT/'retrieval_evaluation.csv').exists():
        st.subheader('Topic representation · giữ nguyên 10 cụm KMeans')
        st.dataframe(csv_frame(RESEARCH_OUT/'topics/topic_evaluation.csv'),hide_index=True,width='stretch')
        st.caption('Ba biến thể trong bảng dùng cùng giao thức coherence; từ khóa tốt hơn không đồng nghĩa phân cụm tốt hơn.')
        st.subheader('Top-10 retrieval · cùng 20 truy vấn')
        state=read_json(RESEARCH_OUT/'evaluation_status.json')
        if state['status']!='COMPLETE':st.warning(state['status']+' · Chưa kết luận phương pháp tìm kiếm tốt hơn.')
        st.dataframe(csv_frame(RESEARCH_OUT/'retrieval_evaluation.csv'),hide_index=True,width='stretch')
        latency=csv_frame(RESEARCH_OUT/'latency.csv')
        if len(latency):st.dataframe(latency,hide_index=True,width='stretch')
        full_benchmark=full_document_topics.OUT/'comparison_rankings_top10.csv'
        saved=csv_frame(full_benchmark if full_benchmark.exists() else RESEARCH_OUT/'rankings_top10.csv')
        choices=saved[['query_id','query']].drop_duplicates().set_index('query_id')['query'].to_dict()
        q=st.selectbox('Truy vấn benchmark',list(choices),format_func=lambda k:f'{k} · {choices[k]}',key='comparison_query_id')
        rows=saved[saved.query_id.eq(q)]
        matrix=rows.pivot(index=['doc_id','title','source'],columns='method',values='rank').reset_index()
        st.dataframe(matrix,hide_index=True,width='stretch')
        st.download_button('Export bảng thứ hạng',matrix.to_csv(index=False).encode('utf-8-sig'),'comparison_top10.csv','text/csv')
        for method in saved.method.drop_duplicates():
            with st.expander(method):
                for row in rows[rows.method.eq(method)].sort_values('rank').to_dict('records'):document(row)
        return
    st.subheader('Topic Modeling · artifacts đã xác minh')
    st.dataframe(csv_frame(OUT/'topic_comparison.csv'),hide_index=True,width='stretch')
    state=read_json(OUT/'evaluation_status.json') if (OUT/'evaluation_status.json').exists() else {'status':'PENDING_INDEPENDENT_LABELS'}
    if state['status']!='COMPLETE':st.warning(state['status']+' · Không thay relevance labels bằng similarity, MiniLM hoặc topic coherence.')
    st.dataframe(csv_frame(OUT/'relevance_summary.csv'),hide_index=True,width='stretch')
    latency=csv_frame(OUT/'latency_warm.csv')
    if len(latency):
        st.subheader('Latency thực nghiệm · CPU')
        st.dataframe(latency,hide_index=True,width='stretch')
        st.caption('Đo request thật; bảng warm bỏ lượt đầu mỗi phương pháp, không cache query embeddings. Không phải benchmark đa người dùng.')
    saved=csv_frame(OUT/'rankings_top10.csv')
    if len(saved):
        st.subheader('Top 10 đã chạy trên cùng benchmark')
        available=saved[['query_id','query']].drop_duplicates()
        choices=available.set_index('query_id')['query'].to_dict()
        q=st.selectbox('Truy vấn benchmark',list(choices),format_func=lambda k:f'{k} · {choices[k]}',key='comparison_query_id')
        rows=saved[saved.query_id.eq(q)]
        matrix=rows.pivot(index=['doc_id','title','source'],columns='method',values='rank').reset_index()
        st.dataframe(matrix,hide_index=True,width='stretch')
        st.download_button('Export bảng thứ hạng',matrix.to_csv(index=False).encode('utf-8-sig'),'comparison_ranks.csv','text/csv')
        for method in METHODS:
            with st.expander(method):
                for row in rows[rows.method.eq(method)].sort_values('rank').to_dict('records'):document(row)
    with st.expander('Chạy so sánh trực tiếp trên truy vấn mới'):
        source,years=filters('comparison')
        with st.form('comparison_form'):
            query=st.text_input('English query',key='comparison_query')
            submit=st.form_submit_button('So sánh 3 phương pháp')
        if submit:
            try:
                with st.spinner('Đang chạy 3 phương pháp…'):st.session_state['comparison_response']=final_service().compare(query,source,years)
            except (ValueError,FileNotFoundError,RuntimeError) as exc:
                st.session_state.pop('comparison_response',None)
                st.error(str(exc))
        if st.session_state.get('comparison_response'):
            from .search_v2_service import rank_matrix
            st.caption('Query: '+st.session_state['comparison_response'][0]['query'])
            st.dataframe(rank_matrix(st.session_state['comparison_response']),hide_index=True,width='stretch')


def evaluation_page():
    st.header('Relevance Evaluation · @10')
    pilot=ROOT/'outputs/ai_silver_labels_20261026_v1'
    if (pilot/'batch_0001_summary.json').exists():
        with st.expander('AI Silver Labels · batch thử 20 cặp'):
            summary=read_json(pilot/'batch_0001_summary.json')
            st.write({'đã xử lý':summary['processed'],'nhãn':summary['label_counts'],'cần kiểm tra':summary['needs_review'],'nhãn trống':summary['blank_labels']})
            st.caption('Đây là nhãn AI, chưa xác minh bởi con người. Cả 20 cặp đầu đều thuộc Q01, không đại diện toàn benchmark và không được dùng như human qrels hoặc để chọn model.')
            st.download_button('Tải batch nhãn AI',(pilot/'batch_0001_labels_20.csv').read_bytes(),'ai_silver_batch_0001.csv','text/csv')
    expanded=full_document_topics.OUT
    if (expanded/'annotation_protocol.json').exists():
        st.subheader('Pool mở rộng · có BatteryBERT full-document')
        status=read_json(expanded/'evaluation_status.json')
        (st.success if status['status']=='COMPLETE' else st.warning)(status['status'])
        st.caption('Cùng 20 truy vấn và pool chung cho 6 phương pháp. Grade 0–2; P@10/MRR@10 dùng grade≥2, nDCG dùng gain 2^grade−1. Chỉ tính khi toàn bộ pool của query có nhãn người đã xác minh; nhãn trống không phải 0.')
        st.dataframe(csv_frame(expanded/'retrieval_evaluation.csv'),hide_index=True,width='stretch')
        st.dataframe(csv_frame(expanded/'annotation_coverage.csv'),hide_index=True,width='stretch')
        st.download_button('Tải phiếu pool mới 0–2',(expanded/'annotation_template_0_2.csv').read_bytes(),'fulldoc_annotation_0_2.csv','text/csv')
        uploaded=st.file_uploader('Import nhãn pool mới (CSV)',type=['csv'],key='full_document_labels')
        st.caption('Cần label_method=HUMAN_INDEPENDENT, label_status=VERIFIED, annotator_id, verifier_id khác người chấm và verification_reference. Nhãn chỉ dùng đánh giá cuối.')
        if st.button('Kiểm tra và nhập nhãn pool mới',disabled=uploaded is None):
            try:
                from .full_document_evaluation import import_labels
                import_labels(uploaded.getvalue());st.success('Đã nhập nhãn hợp lệ.');st.rerun()
            except (ValueError,KeyError) as exc:st.error(str(exc))
        st.divider();st.subheader('Pool lịch sử · 4 phương pháp')
    if (RESEARCH_OUT/'annotation_protocol.json').exists():
        status=read_json(RESEARCH_OUT/'evaluation_status.json')
        (st.success if status['status']=='COMPLETE' else st.warning)(status['status'])
        st.caption('Nhãn 0: không liên quan; 1: liên quan một phần; 2: liên quan trực tiếp. P@10/MRR@10 dùng grade≥2; nDCG dùng gain 2^grade−1 và pool chung đầy đủ. Nhãn trống không phải 0; chưa tính Recall toàn corpus.')
        st.dataframe(csv_frame(RESEARCH_OUT/'retrieval_evaluation.csv'),hide_index=True,width='stretch')
        st.dataframe(csv_frame(RESEARCH_OUT/'annotation_coverage.csv'),hide_index=True,width='stretch')
        st.download_button('Tải phiếu gán nhãn 0–2',(RESEARCH_OUT/'annotation_template_0_2.csv').read_bytes(),'annotation_template_0_2.csv','text/csv')
        st.caption('Phiếu ẩn thứ hạng/điểm/phương pháp. Nhãn chỉ dùng đánh giá cuối, không dùng tuning. Cần HUMAN_INDEPENDENT, VERIFIED, annotator_id, verifier_id khác người chấm và verification_reference.')
        uploaded=st.file_uploader('Import nhãn relevance 0–2 (CSV)',type=['csv'],key='research_labels')
        if st.button('Kiểm tra và nhập nhãn',disabled=uploaded is None):
            try:
                from .research_completion_evaluation import import_labels
                import_labels(uploaded.getvalue())
                st.success('Đã nhập nhãn; metrics chỉ tính trên query có pool được chấm đầy đủ.');st.rerun()
            except (ValueError,KeyError) as exc:st.error(str(exc))
        if status['eligible_queries'] and (RESEARCH_OUT/'retrieval_evaluation.png').exists():
            st.image(str(RESEARCH_OUT/'retrieval_evaluation.png'),width='stretch')
        return
    status=read_json(OUT/'evaluation_status.json') if (OUT/'evaluation_status.json').exists() else {'status':'PENDING_INDEPENDENT_LABELS'}
    (st.success if status['status']=='COMPLETE' else st.warning)(status['status'])
    st.caption('nDCG: gain 2^grade−1, IDCG từ pool chung; P@10 và MRR@10: grade≥2. Tất cả cặp trong pool của query phải được chấm độc lập; nhãn trống không tính là 0.')
    st.dataframe(csv_frame(OUT/'relevance_summary.csv'),hide_index=True,width='stretch')
    st.dataframe(csv_frame(OUT/'label_coverage.csv'),hide_index=True,width='stretch')
    template=OUT/'annotation_template.csv'
    if template.exists():st.download_button('Tải phiếu gán nhãn 0–3',template.read_bytes(),'annotation_template.csv','text/csv')
    uploaded=st.file_uploader('Import relevance labels đã xác minh (CSV)',type=['csv'],key='independent_labels')
    st.caption('Các cột bắt buộc: relevance_grade (0–3), annotator_id, verifier_id khác annotator, verification_reference, label_method=HUMAN_INDEPENDENT, label_status=VERIFIED. Có thể nhập từng phần; query thiếu nhãn vẫn PENDING.')
    if st.button('Kiểm tra và nhập nhãn',disabled=uploaded is None):
        try:
            from .final_system_evaluation import validate_labels,evaluate
            import io
            raw=uploaded.getvalue()
            frame=pd.read_csv(io.BytesIO(raw),keep_default_na=False)
            validate_labels(frame,pd.read_csv(OUT/'annotation_pool.csv',keep_default_na=False))
            path=OUT/('labels_import_'+hashlib.sha256(raw).hexdigest()[:16]+'.csv')
            path.write_bytes(raw)
            evaluate(path)
            from .final_system_reports import report
            report()
            st.success('Đã nhập nhãn hợp lệ và cập nhật kết quả theo độ phủ thực tế.');st.rerun()
        except (ValueError,KeyError) as exc:st.error(str(exc))
    intervals=csv_frame(OUT/'paired_bootstrap.csv')
    if len(intervals):st.dataframe(intervals,hide_index=True,width='stretch')
    audit=csv_frame(OUT/'label_source_audit.csv')
    if len(audit):
        with st.expander('Trạng thái các nguồn nhãn cũ'):st.dataframe(audit,hide_index=True,width='stretch')
