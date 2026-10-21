"""Test actual v2 UI routing; a lexical query does not open historical Qdrant."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_v2_default_and_filtered_lexical_query():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
    try:
        assert not app.exception
        assert app.selectbox(key='search_method').value == 'RERANK_PRETRAINED'
        assert 'SE3' not in app.selectbox(key='search_method').options
        app.selectbox(key='search_source').set_value('Publication')
        app.checkbox(key='search_use_year').check().run()
        app.selectbox(key='search_method').set_value('BM25_CHEMICAL')
        app.slider(key='search_years').set_range(2020, 2023)
        app.text_input(key='search_query').input('Na2FeP2O7 cathode cycling stability')
        app.button[0].click().run()
        assert not app.exception
        response = app.session_state['search_response']
        assert response['version'] == 'search_v2'
        assert response['quality_status'] == 'PENDING_INDEPENDENT_LABELS'
        assert len(response['results']) == 5
        assert all(r['source'] == 'Publication' and 2020 <= r['year'] <= 2023
                   for r in response['results'])
        assert any('5 tài liệu' in item.value for item in app.subheader)
        # Changing filters must not leave a ranking from the previous scope.
        app.slider(key='search_years').set_range(2021, 2023).run()
        assert 'search_response' not in app.session_state
        app.button[0].click().run()
        assert all(2021 <= r['year'] <= 2023
                   for r in app.session_state['search_response']['results'])
        app.checkbox(key='search_use_year').uncheck().run()
        assert 'search_response' not in app.session_state
        app.button[0].click().run()
        app.selectbox(key='search_source').set_value('All').run()
        assert 'search_response' not in app.session_state
        app.text_input(key='search_query').input('zzzxqvzzzz987654')
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state['search_response']['results'] == []
        assert any('0 tài liệu' in item.value for item in app.subheader)
        assert any('từ vựng khớp' in item.value for item in app.info)
        app.text_input(key='search_required_terms').input('zzzxqvzzzz987654')
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state['search_response']['eligible_documents'] == 0
        assert any('Không có tài liệu thỏa bộ lọc' in item.value for item in app.info)
        app.sidebar.radio[0].set_value('Evaluation').run()
        assert not app.exception
        assert any('PENDING_LABELS' in w.value for w in app.warning)
        assert not any('Stage 4:' in item.value for item in app.success)
        app.selectbox(key='evaluation_query').set_value('Q19').run()
        assert not app.exception
        assert any('Fluorine-Free Water-in-Salt' in item.value for item in app.markdown)
        app.selectbox(key='evaluation_query').set_value('Q24').run()
        assert not app.exception
        assert any('không trả về kết quả' in item.value for item in app.info)
        app.sidebar.radio[0].set_value('Topics').run()
        app.radio(key='topic_model').set_value('BERTopic full-document').run()
        assert not app.exception
        assert any('84 topics' in item.value for item in app.info)
        app.sidebar.radio[0].set_value('Comparison').run()
        app.multiselect(key='comparison_methods').set_value(['BM25_CHEMICAL'])
        app.text_input(key='comparison_query').input('Na2FeP2O7 cathode cycling stability')
        app.button[0].click().run()
        assert not app.exception
        assert {'title', 'source', 'doc_id'} <= set(app.dataframe[1].value.columns)
        app.selectbox(key='comparison_source').set_value('Publication').run()
        assert 'comparison_response' not in app.session_state
        app.text_input(key='comparison_query').input('zzzxqvzzzz987654')
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state['comparison_response'][0]['results'] == []
    finally:
        # No resource ownership leak across other UI/integration tests.
        import streamlit as st
        st.cache_resource.clear()
