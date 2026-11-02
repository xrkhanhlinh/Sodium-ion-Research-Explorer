from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_radar_page_seven_tabs_and_historical_metrics():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=120).run()
    try:
        app.sidebar.radio[0].set_value('Search Radar Pilot').run()
        assert not app.exception
        assert [tab.label for tab in app.tabs]==['Search Overview','Radar Comparison','Query Inspector',
            'Evidence Inspector','Pilot Evaluation','Error Analysis','Ablation & Latency']
        assert any('A7' in item.value for item in app.info)
        assert len(app.dataframe)>0
        app.radio(key='radar_scope').set_value('Thực nghiệm · cùng query đầy đủ nhãn').run()
        assert not app.exception
    finally:
        from src.final_dashboard import final_service
        final_service().close()
