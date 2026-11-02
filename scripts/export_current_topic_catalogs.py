"""Export the six current dashboard topic catalogs without refitting anything."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src import full_document_topics, optimized_topics, research_completion
from src.final_system_artifacts import TOPICS, topic_data


def main():
    catalogs = [(full_document_topics.NAME, full_document_topics.topic_data()),
                (optimized_topics.NAME, optimized_topics.topic_data()),
                (research_completion.TOPIC_NAME, research_completion.topic_data())]
    catalogs += [(name, topic_data(name)) for name in TOPICS]
    rows = []
    text = ['# Mười topic của từng pipeline — 24/10/2026', '',
            'Xuất trực tiếp từ sáu catalog đang có trong ứng dụng; không train/refit.',
            'ID topic chỉ có ý nghĩa bên trong từng pipeline, không phải phép ghép cặp giữa model.',
            'Nhãn là diễn giải từ keywords và tài liệu đại diện, chưa có chuyên gia xác nhận.',
            'BatteryBERT full-document · trained là topic chính hiện tại. Dữ liệu huấn luyện có Title–Abstract đầy đủ, chưa có toàn văn bài báo.',
            'LDA dưới đây là bản đối chứng K10; bản chọn K của Stage2 lịch sử là một thí nghiệm riêng.', '']
    for name, data in catalogs:
        text += ['## ' + name, '', '| ID | Diễn giải trong catalog | Tài liệu | Top10 keywords |',
                 '|---|---|---:|---|']
        for topic_id in range(10):
            label = data['labels'][topic_id]
            keywords = data['keywords'].query('topic_id == @topic_id').sort_values('rank').head(10).token.astype(str).tolist()
            count = int((data['assignments'].topic_id == topic_id).sum())
            assert len(keywords) == 10
            row = dict(model=name, topic_id=topic_id, topic_label=label,
                       documents=count, top10_keywords='; '.join(keywords))
            rows.append(row)
            text.append(f"| {topic_id} | {label} | {count:,} | {', '.join(keywords)} |")
        outliers = int((data['assignments'].topic_id == -1).sum())
        text += ['', f'Ngoài 10 topic: {outliers:,} tài liệu outlier (ID −1).', '']
    pd.DataFrame(rows).to_csv(ROOT/'docs/TOP10_TOPICS_ALL_MODELS_20261024.csv', index=False, encoding='utf-8-sig')
    (ROOT/'docs/TOP10_TOPICS_ALL_MODELS_20261024.md').write_text('\n'.join(text), encoding='utf8')
    print('EXPORTED: 6 catalogs, 60 topics, 600 keyword slots')


if __name__ == '__main__':
    main()
