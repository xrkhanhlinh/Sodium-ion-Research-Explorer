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
        if score:st.write(f"{result.get('score_method','Điểm xếp hạng')}: {result['score']:.6f}")
        if score and result.get('semantic_cosine') is not None and not pd.isna(result.get('semantic_cosine')):
            st.write(f"Dense cosine similarity: {result['semantic_cosine']:.6f}")
        if score and result.get('rrf_score') is not None and not pd.isna(result.get('rrf_score')):
            st.caption('RRF kết hợp thứ hạng TF-IDF và BatteryBERT; không cộng trực tiếp hai cosine score.')
        abstract=result.get('abstract')
        with st.expander('Abstract',expanded=result.get('rank')==1):
            st.write(abstract if isinstance(abstract,str) and abstract.strip() else 'Không có Abstract trong dữ liệu gốc.')
        if result.get('evidence_passage'):
            lexical_excerpt=str(result.get('passage_method','')).startswith('Lexical')
            with st.expander('Trích đoạn khớp từ khóa' if lexical_excerpt else 'Đoạn được dùng để xếp hạng'):
                st.write(result['evidence_passage'])
                st.caption('Trích đoạn gợi ý theo từ khóa, không phải giải thích điểm semantic.' if lexical_excerpt else 'Đoạn mô hình đọc để xếp hạng; chưa phải relevance đã được người chấm xác minh.')
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
    from src.final_dashboard import search_page as render
    render(filters, document)

def comparison_page():
    from src.final_dashboard import comparison_page as render
    render(filters, document)

def topic_page():
    from src.final_dashboard import topic_page as render
    render(document, catalog)

def evaluation_page():
    from src.final_dashboard import evaluation_page as render
    render()

st.title('🔋 Sodium-ion Research Explorer')
st.caption('Nghiên cứu Patent & Publication · Models và dữ liệu thực nghiệm đã lưu')
cols=st.columns(4)
for col,label,value in zip(cols,['Documents','LDA topics','BERTopic-BGE topics','BatteryBERT topics (pretrained)'],['13,702','10','10','10']):col.metric(label,value)
st.sidebar.title('Nghiên cứu Sodium-ion')
page=st.sidebar.radio('Điều hướng',['Search','Comparison','Topics','Evaluation','Search Radar Pilot'],key='navigation')
st.sidebar.caption('13.702 tài liệu. Hybrid TF-IDF + BatteryBERT + Qdrant + RRF. Topic chỉ hỗ trợ duyệt/lọc, không làm điểm relevance.')
st.sidebar.caption('Qdrant Local persistent · một worker · đọc nguồn nhãn và độ phủ khi so sánh metrics')
if (ROOT/'outputs/research_completion_v1/COMPLETION_REPORT.md').exists():
    st.sidebar.download_button('Báo cáo hệ thống (.md)',(ROOT/'outputs/research_completion_v1/COMPLETION_REPORT.md').read_bytes(),'COMPLETION_REPORT.md','text/markdown')
elif (ROOT/'outputs/final_system/FINAL_SYSTEM_REPORT.md').exists():
    st.sidebar.download_button('Báo cáo hệ thống (.md)',(ROOT/'outputs/final_system/FINAL_SYSTEM_REPORT.md').read_bytes(),'FINAL_SYSTEM_REPORT.md','text/markdown')
def radar_page():
    from src.search_radar_dashboard import render
    render()

{'Search':search_page,'Comparison':comparison_page,'Topics':topic_page,'Evaluation':evaluation_page,'Search Radar Pilot':radar_page}[page]()
