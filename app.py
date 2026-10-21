"""Sodium-ion Research Explorer — current retrieval and research results."""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
from src.search_v2_service import METHODS
st.set_page_config(page_title='Sodium-ion Research Explorer',page_icon='🔋',layout='wide')

@st.cache_data(show_spinner=False)
def table(path):return pd.read_csv(ROOT/path,keep_default_na=False)
@st.cache_data(show_spinner=False)
def catalog():return pd.read_parquet(ROOT/'data/processed/main_corpus.parquet',columns=['doc_id','source','year','identifier']).set_index('doc_id')
@st.cache_resource(show_spinner=False)
def service():
    from src.search_v2_service import SearchServiceV2
    return SearchServiceV2()
def image(path):
    if (ROOT/path).exists():st.image(str(ROOT/path),width='stretch')
def clear_response(prefix):
    st.session_state.pop(prefix+'_response',None)

def filters(prefix):
    left,right=st.columns([1,2])
    source=left.selectbox('Nguồn tài liệu',['All','Patent','Publication'],key=prefix+'_source',on_change=clear_response,args=(prefix,))
    use_year=right.checkbox('Lọc theo năm',key=prefix+'_use_year',on_change=clear_response,args=(prefix,))
    years=None
    if use_year:
        y=catalog().year.dropna();years=st.slider('Khoảng năm',int(y.min()),int(y.max()),(int(y.min()),int(y.max())),key=prefix+'_years',on_change=clear_response,args=(prefix,))
        st.caption('Lọc trước khi xếp hạng; tài liệu thiếu năm sẽ không nằm trong khoảng lọc.')
    return source,years
def document(result,score=True):
    with st.container(border=True):
        st.markdown(f"**{result.get('rank','')} · {result['title']}**")
        year=result.get('year');year='không rõ' if year is None or pd.isna(year) or year=='' else str(int(year))
        st.caption(f"{result['source']} · {year} · {result['doc_id']}")
        st.write('Identifier:',str(result.get('identifier','')) or 'Không có trong dữ liệu gốc')
        if 'lda_topic' in result:st.write(f"LDA topic: {result['lda_topic']} | BERTopic: {result['bertopic_topic']}"+(' (outlier)' if result['bertopic_topic']==-1 else ''))
        if score:st.write(f"Điểm xếp hạng: {result['score']:.6f}")
        with st.expander('Abstract',expanded=result.get('rank')==1):st.write(result.get('abstract') or 'Không có Abstract trong dữ liệu gốc.')
        if result.get('evidence_passage'):
            with st.expander('Đoạn được dùng để xếp hạng'):
                st.write(result['evidence_passage'])
                st.caption('Đoạn mô hình đọc để xếp hạng; chưa phải bằng chứng relevance đã được người chấm xác minh.')
def response_details(response):
    st.caption(f"Nguồn: {response['source_filter']} · Năm: {response['year_filter'] or 'Tất cả'} · {response['eligible_documents']:,} tài liệu thỏa bộ lọc")
    if response.get('required_terms'):
        st.caption('Từ/công thức bắt buộc: '+ '; '.join(response['required_terms']))
    st.write(f"Thời gian xử lý query: **{response['seconds']*1000:.1f} ms**")
    st.caption('Không gồm khởi tạo engine hoặc thời gian chờ worker; lượt đầu có thể gồm tải model. Reranker đọc tối đa hai đoạn mỗi tài liệu, có thể mất vài giây trên CPU. Thời gian demo không thay thế benchmark warm.')
    if not response['results']:
        if response['eligible_documents']==0:
            st.info('Không có tài liệu thỏa bộ lọc nguồn, năm hoặc từ/công thức bắt buộc. Hãy nới bộ lọc và tìm lại.')
        else:
            st.info('Không có kết quả có từ vựng khớp truy vấn. Hãy thử từ khóa khác; kết quả rỗng chưa chứng minh corpus không có tài liệu phù hợp.')
def search_page():
    st.header('Search Engine')
    st.write('Tìm tài liệu Patent và Publication bằng truy vấn tiếng Anh.')
    source,years=filters('search')
    with st.form('search_form'):
        query=st.text_input('English query',placeholder='Hard carbon anodes with high initial Coulombic efficiency',key='search_query')
        required=st.text_input('Từ/công thức bắt buộc (tùy chọn)',placeholder='NaFSI; ether',key='search_required_terms')
        method=st.selectbox('Phương pháp',METHODS,index=METHODS.index('RERANK_PRETRAINED'),key='search_method')
        submit=st.form_submit_button('Tìm kiếm',type='primary')
    st.caption('Mặc định dùng pretrained + hybrid + reranker, không cần fine-tune của dự án. Score không phải xác suất liên quan. Các nhánh fine-tuned/chunked phục vụ đối chứng.')
    st.caption('Với phương pháp V2, dùng dấu ; để tách các điều kiện bắt buộc. Bộ lọc kiểm tra từ trong Title/Abstract, có thể bỏ sót tài liệu chỉ nêu trong full text; chưa kiểm chứng quan hệ hoặc phủ định.')
    if submit:
        st.session_state.pop('search_response',None)
        if not query.strip():st.error('Hãy nhập truy vấn tiếng Anh trước khi tìm kiếm.')
        else:
            try:
                with st.spinner('Đang truy hồi và xếp hạng tài liệu…'):st.session_state.search_response=service().search(query,method,source,years,required_terms=tuple(t.strip() for t in required.split(';') if t.strip()))
            except Exception as exc:st.error(f'Không thể tìm kiếm: {exc}')
    if 'search_response' in st.session_state:
        response=st.session_state.search_response
        st.subheader(f"Kết quả · {len(response['results'])} tài liệu · {response['method']}");st.write('Truy vấn:',response['query']);response_details(response)
        for result in response['results']:document(result)
        if response['results']:
            st.download_button('Tải kết quả CSV',pd.DataFrame(response['results']).to_csv(index=False).encode('utf-8-sig'),'search_results.csv','text/csv')
def comparison_page():
    st.header('Search Comparison')
    source,years=filters('comparison')
    with st.form('comparison_form'):
        query=st.text_input('English query',placeholder='Prussian blue cathodes with low vacancy content',key='comparison_query')
        required=st.text_input('Từ/công thức bắt buộc cho V2 (tùy chọn)',key='comparison_required_terms')
        selected=st.multiselect('Các phương pháp',METHODS,default=['BM25_CHEMICAL','HYBRID_PRETRAINED','RERANK_PRETRAINED'],key='comparison_methods')
        submit=st.form_submit_button('So sánh',type='primary')
    st.info('So sánh doc_id, thứ hạng và thời gian phản hồi. Không so trực tiếp raw scores khác thang đo; thời gian lượt đầu có thể gồm tải checkpoint.')
    st.caption('Các phương pháp trong cùng lượt so sánh dùng chung query embedding và điểm truy hồi khi tương thích. Thời gian từng nhánh có thể dùng lại phần tính trước đó; không phải benchmark độc lập. Cache được giải phóng khi kết thúc lượt so sánh.')
    if submit:
        st.session_state.pop('comparison_response',None)
        if not query.strip() or not selected:st.error('Nhập truy vấn và chọn ít nhất một phương pháp.')
        else:
            try:
                with st.spinner('Đang chạy các phương pháp tuần tự để tiết kiệm RAM…'):st.session_state.comparison_response=service().compare(query,selected,source,years,required_terms=tuple(t.strip() for t in required.split(';') if t.strip()))
            except Exception as exc:st.error(f'Không thể so sánh: {exc}')
    if 'comparison_response' in st.session_state:
        from src.search_v2_service import rank_matrix
        responses=st.session_state.comparison_response;st.write('Truy vấn:',responses[0]['query'])
        timing=pd.DataFrame([dict(method=r['method'],results=len(r['results']),response_ms=round(r['seconds']*1000,2)) for r in responses])
        st.dataframe(timing,hide_index=True,width='stretch')
        matrix=rank_matrix(responses)
        if len(matrix):
            st.subheader('Tài liệu trùng nhau và khác biệt thứ hạng')
            st.caption('Số phương pháp là số lần tài liệu xuất hiện trong Top-5; dấu — nghĩa là không có trong Top-5 phương pháp đó. Xem Title/Abstract ở kết quả từng phương pháp bên dưới hoặc trong CSV.')
            methods=[r['method'] for r in responses]
            shown=matrix.reindex(columns=['title','source','doc_id','appearances',*methods]).copy()
            for name in methods:shown[name]=shown[name].map(lambda value:'—' if pd.isna(value) else str(int(value)))
            columns={'title':st.column_config.TextColumn('Tiêu đề',width='large'),'source':st.column_config.TextColumn('Nguồn',width='small'),'doc_id':st.column_config.TextColumn('doc_id',width='medium'),'appearances':st.column_config.NumberColumn('Số phương pháp',width='small')}
            columns.update({name:st.column_config.TextColumn(name,width='small') for name in methods})
            st.dataframe(shown,column_config=columns,hide_index=True,width='stretch')
            st.download_button('Tải bảng thứ hạng',matrix.to_csv(index=False).encode('utf-8-sig'),'comparison_ranks.csv','text/csv')
        for response in responses:
            with st.expander(f"{response['method']} · {len(response['results'])} kết quả",expanded=False):
                response_details(response)
                for result in response['results']:document(result,score=False)
def topic_page():
    st.header('Topic Modeling Dashboard')
    model=st.radio('Mô hình topic',['LDA','BERTopic full-document','BERTopic baseline'],horizontal=True,key='topic_model')
    if model=='LDA':
        summary=table('outputs/tables/stage2/topic_summary.csv');reps=table('outputs/tables/stage2/topic_representative_documents.csv')
        st.info('LDA: 20 topics. Số tài liệu là dominant-topic argmax; xác suất topic không phải search relevance.')
        image('outputs/figures/stage2/lda_02_topic_distribution.png')
    elif model=='BERTopic full-document':
        summary=table('outputs/bertopic_chunked/topic_summary.csv');reps=table('outputs/bertopic_chunked/representative_documents.csv')
        st.info('BERTopic full-document: 84 topics, 3.120 outliers (22,77%). Coherence cao hơn baseline, nhưng chưa chứng minh tốt hơn tổng thể.')
        image('outputs/bertopic_chunked/figures/topic_distribution.png')
    else:
        summary=table('outputs/tables/stage3/bertopic_topic_summary.csv');reps=table('outputs/tables/stage3/bertopic_representative_documents.csv')
        st.info('BERTopic: 92 topics và 2.751 outliers (20,08%). Topic -1 không phải một chủ đề khoa học có nhãn. Membership strength không phải relevance probability.')
        image('outputs/figures/stage3/bertopic_01_distribution.png')
    st.dataframe(summary,hide_index=True,width='stretch')
    topic=st.selectbox('Chọn topic',summary.topic_id.tolist(),index=1 if model.startswith('BERTopic') else 0,key='topic_id_'+model)
    row=summary[summary.topic_id.eq(topic)].iloc[0];st.write('Top-10 keywords:',row.top10_keywords)
    selected=reps[reps.topic_id.eq(topic)]
    st.subheader('Tài liệu đại diện')
    if topic==-1:st.caption('Không chọn tài liệu đại diện cho nhóm outlier.')
    for result in selected.to_dict('records'):
        c=catalog().loc[result['doc_id']];result.update(year=c.year,identifier=c.identifier)
        document(result,score=False)
    if model=='BERTopic full-document':
        with st.expander('UMAP full-document'):
            image('outputs/bertopic_chunked/figures/umap_2d.png')
    elif model=='BERTopic baseline':
        with st.expander('UMAP và topic similarity'):
            image('outputs/figures/stage3/bertopic_04_umap_2d.png');image('outputs/figures/stage3/bertopic_03_similarity_heatmap.png')
    else:
        with st.expander('LDA theo năm và so sánh K'):
            image('outputs/figures/stage2/lda_04_topic_year_heatmap.png');image('outputs/figures/stage2/lda_01_model_comparison.png')
def evaluation_page():
    st.header('Evaluation Dashboard')
    st.warning('PENDING_LABELS — chưa có relevance labels độc lập được kiểm chứng. nDCG@5, Precision@5 và MRR@5 chưa được tính chính thức.')
    upgrade=ROOT/'outputs/search_upgrade_20261018'
    if (upgrade/'run_status.json').exists():
        from src.stage5_common import current_status
        state=current_status()
        st.subheader('Search v2')
        st.write('Trạng thái:',state['status'])
        metrics=upgrade/'evaluation_14_methods/summary.csv'
        if metrics.exists():
            st.dataframe(pd.read_csv(metrics),hide_index=True,width='stretch')
            st.caption('14 phương pháp gồm 10 cấu hình mới và 4 baseline cũ từ kết quả đã lưu. Baseline lịch sử không còn là backend của app.')
        latency=upgrade/'live_latency_uncached.csv'
        if latency.exists():
            st.subheader('Thời gian tìm kiếm hiện tại')
            st.dataframe(pd.read_csv(latency),hide_index=True,width='stretch')
            st.caption('6 requests warm trên 3 query, encode query và rerank thật, không cache query/pair. Không gồm model loading; chưa đo throughput nhiều người dùng.')
        saved_path='outputs/search_upgrade_20261018/top5_comparison.csv'
        if (ROOT/saved_path).exists():
            with st.expander('Tra cứu Top-5 benchmark đã lưu'):
                saved=table(saved_path)
                queries=saved[['query_id','query']].drop_duplicates().set_index('query_id')['query'].to_dict()
                query_id=st.selectbox('Truy vấn benchmark',list(queries),format_func=lambda key:f'{key} · {queries[key]}',key='evaluation_query')
                method=st.selectbox('Phương pháp benchmark',METHODS,index=METHODS.index('RERANK_PRETRAINED'),key='evaluation_method')
                selected=saved[saved.query_id.eq(query_id)&saved.method.eq(method)].sort_values('rank')
                st.caption(f'{len(selected)} tài liệu trong kết quả đã lưu; chưa có đánh giá relevance độc lập. Truy vấn và bộ lọc Search hiện tại không áp dụng vào bảng này.')
                if selected.empty:st.info('Phương pháp này không trả về kết quả trong lượt benchmark đã lưu.')
                for result in selected.to_dict('records'):document(result)
                if len(selected):st.download_button('Tải Top-5 benchmark',selected.to_csv(index=False).encode('utf-8-sig'),'benchmark_top5.csv','text/csv')
        labels=upgrade/'annotator_A_NEW.csv'
        if labels.exists():
            st.download_button('Tải phiếu chấm độc lập — pool 942 cặp',labels.read_bytes(),'annotator_A_NEW.csv','text/csv')

st.title('🔋 Sodium-ion Research Explorer')
st.caption('Nghiên cứu Patent & Publication · Models và dữ liệu thực nghiệm đã lưu')
cols=st.columns(4)
for col,label,value in zip(cols,['Tài liệu','LDA topics','BERTopic full-document','Outliers full-document'],['13.702','20','84','22,77%']):col.metric(label,value)
st.sidebar.title('Nghiên cứu Sodium-ion')
page=st.sidebar.radio('Điều hướng',['Search','Comparison','Topics','Evaluation'],key='navigation')
st.sidebar.caption('13.702 tài liệu đã làm sạch. Tìm kiếm: lexical + dense + reranker. Topic modeling là thực nghiệm riêng, không phải điểm relevance.')
st.sidebar.caption('Local demo · một worker · chưa có relevance labels độc lập')
if (ROOT/'docs/FINAL_RESEARCH_REPORT.md').exists():
    st.sidebar.download_button('Báo cáo kỹ thuật (.md)',(ROOT/'docs/FINAL_RESEARCH_REPORT.md').read_bytes(),'FINAL_RESEARCH_REPORT.md','text/markdown')
{'Search':search_page,'Comparison':comparison_page,'Topics':topic_page,'Evaluation':evaluation_page}[page]()
