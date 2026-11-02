"""Test final UI mounted in the existing app; V2 core has separate regression checks."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]


def test_research_method_ui_filters_topics_and_pending_evaluation():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=180).run()
    try:
        assert not app.exception
        from src import full_document_inference as full
        expected=['TFIDF_COSINE','BATTERYBERT_SEMANTIC','BATTERYBERT_HYBRID','BGE_SEMANTIC']
        if full.ready():expected+=list(full.METHODS)
        assert app.selectbox(key='search_method').value==('BATTERYBERT_FULLDOC_HYBRID' if full.ready() else 'BATTERYBERT_HYBRID')
        assert app.selectbox(key='search_method').options==expected
        app.selectbox(key='search_source').set_value('Publication')
        app.checkbox(key='search_use_year').check().run()
        app.slider(key='search_years').set_range(2020,2023)
        app.selectbox(key='search_method').set_value('TFIDF_COSINE')
        app.text_input(key='search_query').input('Na2FeP2O7 cathode cycling stability')
        app.button[0].click().run()
        assert not app.exception
        response=app.session_state['search_response']
        assert response['method']=='TFIDF_COSINE' and len(response['results'])==10
        assert all(r['source']=='Publication' and 2020<=r['year']<=2023 for r in response['results'])
        assert any('10 tài liệu' in item.value for item in app.subheader)
        app.slider(key='search_years').set_range(2021,2023).run()
        assert 'search_response' not in app.session_state
        app.text_input(key='search_query').input('zzzxqvzzzz987654')
        app.button[0].click().run()
        assert not app.exception and app.session_state['search_response']['results']==[]
        app.sidebar.radio[0].set_value('Topics').run()
        assert not app.exception
        for name in ['BatteryBERT KMeans (pretrained)','LDA','BERTopic–BGE','BERTopic–BatteryBERT (pretrained)']:
            app.radio(key='topic_model').set_value(name).run()
            assert not app.exception
            assert any(m.label=='Topics' and m.value=='10' for m in app.metric)
        app.sidebar.radio[0].set_value('Evaluation').run()
        assert not app.exception
        assert any('PENDING_HUMAN_ANNOTATION' in w.value for w in app.warning)
        app.sidebar.radio[0].set_value('Comparison').run()
        assert not app.exception
        assert len(app.dataframe)>=2
        import pandas as pd
        saved=pd.read_csv(ROOT/'outputs/research_completion_v1/rankings_top10.csv')
        missing_query=saved.loc[saved.abstract.isna(),'query_id'].iloc[0]
        app.selectbox(key='comparison_query_id').set_value(missing_query).run()
        assert not app.exception
        assert any(item.value=='Không có Abstract trong dữ liệu gốc.' for item in app.markdown)
    finally:
        from src.final_dashboard import final_service
        final_service().close()
        import streamlit as st
        st.cache_resource.clear()


@pytest.mark.skipif(not (ROOT/'outputs/final_system/batterybert_finetuned/embedding_success.json').exists(),
                    reason='Real V3 corpus inference must finish first')
def test_real_v3_top10_in_streamlit():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=240).run()
    try:
        app.checkbox(key='search_legacy').check().run()
        app.selectbox(key='search_method').set_value('SEARCH_V3')
        app.text_input(key='search_query').input('Hard carbon anodes derived from biomass for sodium-ion batteries')
        app.button[0].click().run()
        assert not app.exception
        response=app.session_state['search_response']
        assert response['method']=='SEARCH_V3' and len(response['results'])==10
        assert response['checkpoint']=='batterybert_finetuned'
        assert len({r['doc_id'] for r in response['results']})==10
        assert all(r['score_method']=='MiniLM cross-encoder score (V3)' for r in response['results'])
    finally:
        from src.final_dashboard import final_service
        final_service().close()
        import streamlit as st
        st.cache_resource.clear()


def test_real_pretrained_hybrid_topic_filter_in_streamlit():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=240).run()
    try:
        app.selectbox(key='search_method').set_value('BATTERYBERT_HYBRID')
        app.selectbox(key='search_topic').set_value(3)
        app.text_input(key='search_query').input('MoS2 carbon anode sodium storage')
        app.button[0].click().run()
        assert not app.exception
        response=app.session_state['search_response']
        assert response['method']=='BATTERYBERT_HYBRID'
        assert response['checkpoint']=='batterybert_pretrained'
        assert response['topic_filter']==3 and len(response['results'])==10
        assert all(r['topic_id']==3 and r['semantic_cosine'] is not None for r in response['results'])
        assert all(r['rrf_score']==r['score'] and 'RRF' in r['score_method'] for r in response['results'])
    finally:
        from src.final_dashboard import final_service
        final_service().close()
        import streamlit as st
        st.cache_resource.clear()


@pytest.mark.skipif(not (ROOT/'outputs/batterybert_full_document_20261024_v1/deployment.json').exists(),
    reason='Full-document model/index deployment must finish')
def test_real_full_document_hybrid_long_query_and_combined_filters_in_streamlit():
    import pandas as pd
    from src import full_document_inference as full,full_document_topics,optimized_topics
    if not full.ready():pytest.skip('Continuation was not accepted; retain the existing serving model')
    assigned=(full_document_topics if full_document_topics.ready() else optimized_topics).assignments()
    subset=assigned[assigned.source.eq('Publication')&assigned.year.between(2020,2023)]
    selected=int(subset.groupby('topic_id').size().idxmax())
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=240).run()
    try:
        assert not app.exception
        assert app.selectbox(key='search_method').value=='BATTERYBERT_FULLDOC_HYBRID'
        app.selectbox(key='search_source').set_value('Publication')
        app.checkbox(key='search_use_year').check().run()
        app.slider(key='search_years').set_range(2020,2023)
        app.selectbox(key='search_topic').set_value(selected)
        app.text_input(key='search_required_terms').input('sodium')
        app.text_input(key='search_query').input('Sodium ion battery electrodes and energy storage. '*40+'Hard carbon anodes from biomass. '*20)
        app.button[0].click().run()
        assert not app.exception
        response=app.session_state['search_response']
        assert response['method']=='BATTERYBERT_FULLDOC_HYBRID' and response['checkpoint']=='batterybert_fulldoc'
        assert response['query_coverage']['tokens']>128
        assert response['query_coverage']['chunks']>1 and not response['query_coverage']['truncated']
        assert len(response['results'])==10 and len({r['doc_id'] for r in response['results']})==10
        assert all(r['source']=='Publication' and 2020<=r['year']<=2023 and r['topic_id']==selected for r in response['results'])
        assert all('sodium' in (r['title']+' '+str(r['abstract'])).casefold() for r in response['results'])
        assert all(r['rrf_score']==r['score'] and r['semantic_cosine'] is not None for r in response['results'])
    finally:
        from src.final_dashboard import final_service
        final_service().close()
        import streamlit as st
        st.cache_resource.clear()
