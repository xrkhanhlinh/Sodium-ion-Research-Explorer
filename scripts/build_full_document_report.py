"""Current whole-pipeline print report, built only from completed run evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import build_full_pipeline_report as b
import json
import hashlib
import re
import pandas as pd
from html import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4

RUN='outputs/batterybert_full_document_20261024_v1'
OLD='outputs/batterybert_optimization_20261023_v1'
NAME='BAO_CAO_PIPELINE_BATTERYBERT_FULLDOC_20261024'


def j(name):return b.read_json(RUN+'/'+name)
def c(name):return b.read_csv(RUN+'/'+name)


def build():
    b.OUT.mkdir(parents=True,exist_ok=True)
    selection=j('selection.json');coverage=j('coverage_audit.json');train=j('training_coverage.json')
    deploy=j('deployment.json');proof=j('embedding_success.json');parity=j('model_reload_parity.json')
    baseline=j('baseline/validation/metrics.json');baseline_test=j('baseline/test/metrics.json');test=j('final_test/metrics.json')
    effects=j('paired_training_effect.json');checks=j('verification_summary.json')
    topic=j('topics/final_metrics.json');topic_choice=j('topics/selection.json')
    cv=c('topics/common_coherence.csv');val=c('topics/validation_comparison.csv')
    status=j('evaluation_status.json');latency=c('benchmark_latency.csv')
    lengths=c('text_length_summary.csv');scope=j('text_length_audit.json')
    adjacency=j('keyword_adjacency_audit.json');preservation=j('preservation_check.json')
    old_full=b.read_json(OLD+'/full_corpus_diagnostics.json')
    old_test=b.read_json(OLD+'/final_test.json');old_cv=b.read_csv(OLD+'/common_coherence.csv')
    bge_pre=b.read_json('outputs/reports/stage3/pretrained_test_metrics.json')
    bge_fine=b.read_json('outputs/reports/stage3/finetuned_test_metrics.json')
    stage1=b.read_json('outputs/reports/stage1_summary.json')
    lda_selection=b.read_json('outputs/reports/stage2/selected_model.json')
    active=RUN+'/topics' if deploy['activate_topics'] else OLD
    assignments=b.read_csv(active+'/topic_assignments.csv')
    keywords=b.read_csv(active+'/topic_keywords.csv');labels=b.read_csv(active+'/topic_labels.csv').set_index('topic_id')
    sizes=assignments.groupby('topic_id').size()
    active_metrics=topic if deploy['activate_topics'] else old_full
    active_name='BatteryBERT full-document trained' if deploy['activate_topics'] else 'BatteryBERT fine-tuned/projection bản 23/10'
    p,h,table,section,note=b.p,b.h,b.table,b.section,b.note
    b.markdown.append('# Báo cáo toàn bộ pipeline và huấn luyện BatteryBERT full-document - 24/10/2026\n')
    b.story.append(Spacer(1,18))
    b.story.append(Paragraph('BÁO CÁO TOÀN BỘ PIPELINE<br/>BATTERYBERT FULL-DOCUMENT',ParagraphStyle(
        'fullcover',fontName='Arial-Bold',fontSize=25,leading=33,textColor=b.NAVY,spaceAfter=20)))
    p('Sodium-ion Research Explorer','h1')
    p('Kết quả thực thi, đánh giá, sửa lỗi và giới hạn - 24/10/2026','h2')
    p('Bản cập nhật sau huấn luyện thật, thay cho việc chỉ đề xuất cách train dài hơn. Các số đo được lấy từ artifacts của lượt chạy mới; model và dữ liệu lịch sử được giữ để đối chứng.')
    table(['Corpus','Lượt train đầy đủ','Văn bản train được bao phủ'],[[
        '13.702 Title-Abstract',b.count(train['train_documents'])+' tài liệu',
        b.count(train['actual_unique_abstract_tokens'])+' token Abstract']], [1,1,1.5])
    h('Kết quả chính')
    p(f"Validation proxy cùng giao thức: {b.fmt(baseline['group_mrr_at_10'],6)} → {b.fmt(selection['best_score'],6)}; test checkpoint được chọn: {b.fmt(test['group_mrr_at_10'],6)}. Checkpoint continuation được chấp nhận: {selection['accepted']}.")
    p(f"Search full-document được bật: {deploy['activate_search']}. Topic chính đang dùng: {active_name}. Model mới chỉ được kích hoạt theo cổng đã kiểm tra; không gọi một encoder là tốt nhất ở mọi chỉ số.")
    note('Dữ liệu hiện có Title và Abstract, chưa có toàn văn bài báo. “Đọc toàn bộ” trong thí nghiệm nghĩa là đọc hết văn bản sẵn có qua nhiều cửa sổ, không phải đã thu thập và học phần Methods/Results của mọi bài báo.')
    p('Nhãn chuyên gia và relevance người dùng độc lập còn thiếu. MRR proxy, coherence, seed agreement và tests kỹ thuật không được gọi là accuracy khoa học.','small')

    section(1,'Bài toán và pipeline toàn chương trình')
    p('Hai đầu ra chính: (1) 10 cụm topic, mỗi cụm có top10 keywords và đại diện; (2) Top10 tài liệu cho truy vấn. Hai đầu ra có protocol đánh giá riêng.')
    b.story.append(b.pipeline_drawing());b.story.append(Spacer(1,10))
    p('Excel → sửa cấu trúc bảng và metadata → sàng lọc/ngôn ngữ/trùng lặp/thu hồi → corpus. Nhánh lexical tạo LDA, TF-IDF và keywords; nhánh neural tạo BGE/BatteryBERT embeddings. Topic được fit trên train; Search lọc tập doc_id trước retrieval và hợp nhất thứ hạng.')
    p('Các bước train, encode corpus và xây index chạy offline. Web đọc artifacts; mỗi truy vấn chỉ encode query, lọc, tìm kiếm và hiển thị. Topic hỗ trợ duyệt/lọc, không cộng điểm vào ranking.')
    table(['Khối','Implementation chính'],[
        ['Dữ liệu','src/pipeline.py; text_processing.py; quality.py'],
        ['LDA/BGE/BatteryBERT lịch sử','stage2.py; stage3_neural.py; model_comparison.py'],
        ['Training đầy đủ mới','batterybert_full_document.py; train_batterybert_full_document.py'],
        ['Keywords/topic mới','scientific_keyword_stream.py; evaluate_batterybert_full_document.py'],
        ['Serving','final_search.py; full_document_inference.py; full_document_retrieval.py'],
        ['Giao diện/đánh giá','final_dashboard.py; full_document_evaluation.py'],
    ],[1,2.7])

    section(2,'Dữ liệu, chất lượng và split')
    table(['Nguồn/phạm vi','Bản ghi','Nguồn/phạm vi','Bản ghi'],[
        ['Patent Excel','8.657','Publication Excel','11.919'],
        ['Clean master bảo toàn nguồn','20.576','Corpus chính','13.702'],
        ['Cặp Title-Abstract','13.650','Title-only inference','52']],[1.5,.7,1.5,.7])
    p('Corpus chính gồm 5.933 Patent và 7.769 Publication.','small')
    p('Làm sạch giữ doc_id, Title/Abstract, source/year/identifier, provenance và split_group_id. Sàng lọc CORE/RELATED là heuristic, chưa phải nhãn chuyên gia. Không tạo Abstract còn thiếu, không thêm full article giả và không thay corpus để làm đẹp điểm.')
    dispositions={}
    for row in stage1['dispositions']:dispositions[row['disposition']]=dispositions.get(row['disposition'],0)+row['count']
    items=list(dispositions.items())+[('TỔNG',sum(dispositions.values()))]
    table(['Disposition','Bản ghi','Disposition','Bản ghi'],[
        [items[i][0],b.count(items[i][1]),items[i+1][0],b.count(items[i+1][1])]
        for i in range(0,len(items),2)],[1.6,.6,1.6,.6])
    p('Kiểm tra cấu trúc Excel, ngôn ngữ, thu hồi và trùng lặp theo thứ tự ưu tiên. Tổng disposition bằng20.576; số nhãn heuristic và disposition có thể khác do quy tắc ưu tiên.','small')
    table(['Split','Tài liệu','Token Abstract','Đoạn'],[[r.split,b.count(r.documents),b.count(r.abstract_content_tokens),b.count(r.chunks)] for r in lengths.itertuples()],[1.3,.9,1.3,.9])
    p('Train cập nhật optimizer và fit geometry/KMeans/vocabulary; validation chọn learning rate/checkpoint/topic representation; test chỉ đo sau khi chốt. 52 title-only không tham gia optimizer. Các split là split lịch sử, không phải external test mới.')
    p('Grouped split hạn chế trùng nhóm giữa tập nhưng không chứng minh đã nhận diện mọi patent family hay semantic duplicate. Cần review ngoài phạm vi, khác biệt vai trò anode/cathode và trường hợp nhiều chủ đề trong cùng tài liệu.')
    p('Catalog đối chứng còn có nhóm sodium cobaltate/siêu dẫn. Đây là dấu hiệu cần rà soát phạm vi khoa học của corpus; quy tắc liên quan ở Stage1 không chứng minh mọi tài liệu đều trực tiếp nghiên cứu pin sodium-ion. Không xóa các nhóm này sau khi xem test để làm đẹp kết quả.','small')

    section(3,'BatteryBERT cũ chưa tốt ở đâu và đã sửa gì')
    table(['Vấn đề','Thay đổi đã chạy'],[
        ['Train cắt 128 token','Train dùng toàn bộ các đoạn 256/64; mỗi đoạn tham gia pooling và backward'],
        ['Batch2, accumulation không tăng negatives trong loss','Gradient caching với batch thực 32 cặp, tối đa 31 negatives/cặp trước masking'],
        ['Train/inference lệch cách đọc','Query đầy đủ; document pooling nhất quán lúc train, export và serving'],
        ['Nguồn Patent/Publication chi phối topic','Tiếp tục kiểm tra train-fit source projection; không coi full-text loss tự xóa lệch nguồn'],
        ['Ngram trên tokens đã xóa stopword','Đọc lại raw fields; giữ ranh giới từ bị loại, dấu câu và Title/Abstract'],
        ['Báo cáo/model/encoder dễ bị trộn','Hash, ordered IDs, frozen protocol, replay parity và version riêng'],
    ],[1.2,2.8])
    p('Giữ checkpoint fine-tuned lịch sử làm điểm khởi đầu; khóa 11 block đầu và train block cuối. Cache float32 của 11 block được tái sử dụng đúng hash, đọc read-only; chỉ tính bổ sung các prefix còn thiếu. Đây là cập nhật neural thật, không thay pretrained vector rồi gọi là fine-tuned.')
    p('Không xóa hàng loạt các từ như solution/dry/connect để tạo danh sách đẹp. Chúng có thể là nội dung quy trình hoặc phần cứng thật; lỗi gán cụm cần được đánh giá qua tài liệu, không chỉ tên topic.')

    section(4,'Cơ chế đọc dài và huấn luyện')
    protocol=j('training_protocol.json')
    p(f"Lượt chạy dùng CPU, {protocol['config']['threads']} threads, torch {protocol['versions']['torch']}, transformers {protocol['versions']['transformers']}. Chỉ cache 11 block cố định; block cuối được replay với graph thật để giảm RAM mà giữ gradient. Không dùng GPU trong lượt này.")
    note(f"Độ dài thực: {b.count(scope['abstract_content_tokens'])} token Abstract trên corpus. Tối đa 1.673 token Abstract; số đoạn tối đa của một tài liệu là14 (không phải cùng tài liệu, vì độ dài Title thay đổi payload). Không có Title trong corpus bị cắt context. Native query prefixes có {b.count(scope['query_native_tokens_including_cls_sep'])} token kể cả CLS/SEP; raw Title content có {b.count(scope['raw_title_content_tokens'])} token.")
    p('Mỗi document chunk dùng template native BERT với Title làm context. Pool chỉ token Abstract, loại SEP; title-only có fallback pool nội dung Title. Chuẩn hóa vector từng chunk, gộp với trọng số số token mới, rồi chuẩn hóa tài liệu.')
    note('chunk = L2[ masked mean(hidden states của nội dung) ]\ndocument = L2[ Σj wj × chunkj / Σj wj ]\nwj = số token nội dung mới được bao phủ ở đoạn j')
    p('Trọng số không cộng lặp độ dài overlap; token overlap vẫn ảnh hưởng ngữ cảnh và chunk mean. Toàn bộ văn bản được bao phủ nhưng không có attention toàn cục giữa mọi đoạn.')
    h('Gradient caching hai tầng')
    p('Forward không lưu graph cho toàn batch → tính contrastive loss đối xứng với cả batch → lấy gradient query/document embeddings → truyền gradient qua phép pooling → replay từng nhóm query/chunk để cập nhật BERT. Dropout tắt để replay khớp; tests so sánh với full-batch gradient, gồm BERT attention thật.')
    table(['Thông số','Giá trị'],[['Optimizer','AdamW; weight decay 0,01; clip norm 1'],['Learning rates pilot','2e-6 và 5e-6, mỗi cấu hình 24 updates'],
        ['Learning rate chọn',str(selection['learning_rate'])],['Epoch / số update lượt thắng','1 / 342'],
        ['Batch / microbatch chunk','32 cặp / 2 đoạn; batch cuối có 9 cặp'],['Temperature / distillation','0,05 / 0,02'],
        ['Lịch LR','Warmup rồi decay; giữ best validation kể cả baseline']],[1.3,2.4])
    p('Batch phân tầng nguồn, mỗi ID train xuất hiện đúng một lần trong lượt hoàn chỉnh. Mask same-group và title gần trùng; chưa có nhãn khoa học để khẳng định đã loại mọi false negative.')

    section(5,'Kết quả encoder và đánh giá thống kê')
    table(['Phép đo cùng full-document protocol','MRR@10'],[
        ['Validation baseline',b.fmt(baseline['group_mrr_at_10'],6)],['Validation checkpoint được chọn',b.fmt(selection['best_score'],6)],
        ['Test baseline, cùng pooling',b.fmt(baseline_test['group_mrr_at_10'],6)],
        ['Test checkpoint được chọn',b.fmt(test['group_mrr_at_10'],6)]],[2.7,1])
    p(f"Checkpoint được chọn tại step {selection['best_step']}, đã thấy {b.count(selection['selected_checkpoint_rows_seen'])} tài liệu trong continuation. Toàn lượt train đã backward qua {b.count(train['actual_chunks_backpropagated'])} đoạn của {b.count(train['train_documents'])} tài liệu. Hai phạm vi này được ghi riêng.")
    p(f"L2 thay đổi block cuối được export: {b.fmt(proof['selected_last_block_delta_l2'],6)}. Validation/test xuất hiện trong optimizer: {train['validation_and_test_optimizer_appearances']}. Model reload parity: {parity['status']}.")
    for effect in effects:
        low,high=effect['cluster_bootstrap_percentile_95_interval']
        p(f"{effect['split']}: gain paired={b.fmt(effect['paired_gain'],6)}, khoảng bootstrap percentile 95% [{b.fmt(low,6)}; {b.fmt(high,6)}], resample theo {effect['groups']} split_group_id, 10.000 lượt.")
    image=ROOT/RUN/'learning_curve.png'
    b.SOURCES.append(image);b.story.append(b.Image(str(image),width=b.WIDTH,height=b.WIDTH*3.6/11))
    b.markdown.append('![Learning curve](../'+RUN+'/learning_curve.png)\n')
    note('Proxy là Title → document có Title context. Nó có thể được lợi từ Title; không phải relevance query độc lập. Khoảng validation còn chịu selection bias. Bootstrap trên test đánh giá proxy cho checkpoint cố định, không xác nhận accuracy khoa học hoặc thắng BGE trên Search.')
    p('Các số MRR lịch sử 128-token/BGE dùng protocol khác được giữ làm bối cảnh; không ghép validation của model này với test của model khác để kết luận encoder thắng tuyệt đối.')

    section(6,'Tối ưu 10 topic và từ khóa')
    p('Fit hai geometry center/source, KMeans10 seed42 và seed43 trên train. Với mỗi geometry, thử exclusivity 0/0,25/0,5 và phrase boost 1/1,5: tổng 12 cấu hình representation. Vocabulary chỉ fit train, min_df5, max_df0,6, tối đa40.000 features; mỗi keyword có ≥5 tài liệu train hỗ trợ.')
    table(['Ứng viên validation','V nguồn','ARI','NPMI','Div.','Đạt'],[[r.candidate,b.fmt(r.source_cramers_v,3),b.fmt(r.stability_ari,3),b.fmt(r.document_npmi,3),b.fmt(r.keyword_diversity,2),str(r.passes)] for r in val.itertuples()],[2.4,.65,.65,.7,.6,.55])
    p(f"Ứng viên được chọn: {topic_choice['candidate']}. Cổng đặt trước: lớn nhất≤30%, top2≤45%, nhỏ nhất≥0,5%, V≤0,5, ARI≥0,6, diversity≥0,75, support coverage≥95%, NPMI≥0. Seed137 kiểm tra test sau chọn; không refit toàn corpus.")
    p(f"Cổng topic test: {all(topic['heldout_gates'].values())}; gate triển khai tổng: {topic['promotion_gate']}. Topic mới chỉ thay catalog chính khi qua gate và có review agent theo đủ 10 nhóm. Gate kỹ thuật không phải chuẩn accuracy khoa học.")
    p(f"Có {b.count(topic['review_priority_documents'])} tài liệu được đánh dấu ưu tiên rà soát vì xa centroid hoặc gần hai topic; ngưỡng lấy từ train. Danh sách không thay assignments và không được hiểu là xác suất model sai.")

    section(7,'So sánh topic và chọn model chính')
    table(['Cùng reference mới','C_v','Diversity','Keyword coverage'],[[r.model,b.fmt(r.c_v),b.fmt(r.topic_diversity,2),b.pct(r.supported_keyword_fraction)] for r in cv.itertuples()],[1.8,.7,.8,1.1])
    p('Reference lần này dùng tokens nguồn có ranh giới, phrase thật và cùng window110/top10. Chỉ so hai dòng trong bảng trực tiếp với nhau; C_v cũ tính trên reference khác được ghi dưới dạng lịch sử.')
    p('So sánh topic là so pipeline: encoder, document pooling, geometry và trọng số keywords có thể cùng khác. Không quy toàn bộ chênh lệch topic cho neural fine-tuning. Phép so encoder baseline/continuation ở mục5 mới giữ cùng full-document pooling.')
    p('ARI ở đây đo agreement giữa các lần fit KMeans với seed khác; không phải tỷ lệ phân loại đúng. Neural training mới chỉ dùng một seed, nên chưa đo biến thiên qua nhiều lần train BERT độc lập.','small')
    table(['Pipeline lịch sử','C_v lịch sử','Giới hạn'],[
        ['BatteryBERT pretrained/HDBSCAN','0,7436','Hai cụm 92,76%; source V≈0,9986'],
        ['BatteryBERT pretrained/KMeans','0,6941','Kích thước tốt hơn, vẫn lệch nguồn rất mạnh'],
        ['BatteryBERT fine-tuned/projection 23/10','0,5601','Test ARI 0,9245; còn tài liệu lệch nhóm'],
        ['LDA K10','0,5297','Đối chứng lexical, không có encoder'],
        ['BERTopic BGE K10','0,6455','Outlier 22,77% trong nhánh lịch sử']],[1.6,.8,1.6])
    p(f"Catalog chính hiện hành: {active_name}. Trên toàn corpus: source V={b.fmt(active_metrics['source_cramers_v'])}, top2={b.pct(active_metrics['top2_topic_share'])}. KMeans luôn gán cụm; 0% outlier không chứng minh mọi tài liệu phù hợp.")
    p('Model tốt nhất phụ thuộc đầu ra: baseline C_v cao có thể collapse theo nguồn; encoder thắng proxy chưa chắc thắng relevance. Chọn BatteryBERT làm model chính theo yêu cầu đề tài, đồng thời giữ BGE/LDA và checkpoint cũ làm đối chứng trung thực.')
    p(f"Stage2 chọn LDA K={lda_selection['selected_k']} bằng protocol riêng. LDA K10 trong bảng trên là đối chứng ép cùng10 topic; không thay kết quả Stage2 bằng K10. Các pilot AI-judge và synthetic source-ID recovery lịch sử không được gộp thành relevance có nhãn người.", 'small')
    table(['Proxy test lịch sử, protocol riêng','Group MRR@10'],[
        ['BGE pretrained',b.fmt(bge_pre['group_mrr_at_10'])],['BGE fine-tuned',b.fmt(bge_fine['group_mrr_at_10'])]],[2.7,1])
    p('BGE fine-tuned dẫn trong phép so BGE lịch sử này. Các số trên không phải benchmark Search có nhãn người và không so trực tiếp với FULLDOC có Title context để kết luận encoder nào tốt nhất tuyệt đối.','small')

    # Preserve each catalog's own IDs; a shared row number is not an alignment.
    catalogs=b.read_csv('docs/TOP10_TOPICS_ALL_MODELS_20261024.csv')
    for model,group in catalogs.groupby('model',sort=False):
        if model=='BatteryBERT full-document · trained':continue
        b.story.append(b.PageBreak())
        h('10 topic đối chứng: '+model)
        p('Tên diễn giải lấy từ catalog hiện hữu. ID không tương ứng giữa các model; chưa có chuyên gia xác nhận. Bảng đầy đủ top10 keywords được xuất riêng tại docs/TOP10_TOPICS_ALL_MODELS_20261024.md.','small')
        table(['ID','Nhãn trong catalog','Tài liệu','Ba keywords đầu'],[
            [str(r.topic_id),r.topic_label,b.count(r.documents),'; '.join(r.top10_keywords.split('; ')[:3])]
            for r in group.sort_values('topic_id').itertuples()],[.3,2.2,.6,1.4])

    for start in [0,5]:
        section(8 if start==0 else 9,f'10 topic hiện hành: nhóm {start}-{start+4}')
        p('ID chỉ có ý nghĩa trong catalog đang hiển thị. Nhãn là diễn giải tạm thời, chưa phải nhãn chuyên gia. Số lượng lấy từ toàn corpus.','small')
        for t in range(start,start+5):
            h(f"Topic {t}: {labels.loc[t,'topic_label']}")
            p(f"{b.count(sizes.loc[t])} tài liệu ({b.pct(sizes.loc[t]/len(assignments))}).")
            p('Top10: '+'; '.join(keywords.loc[keywords.topic_id.eq(t)].sort_values('rank').token.astype(str))+'.')

    b.story.append(b.PageBreak())
    h('Phân bố nguồn trong 10 topic hiện hành')
    p('Các nhóm không bị ép có kích thước bằng nhau. Phân bố theo nguồn được kiểm tra riêng vì topic có thể học văn phong Patent/Publication.')
    source_counts=assignments.groupby(['topic_id','source']).size().unstack(fill_value=0).reindex(range(10),fill_value=0)
    b.story.append(b.source_chart(source_counts))
    table(['ID','Patent','Publication','Tổng'],[[str(t),b.count(source_counts.loc[t,'Patent']),
        b.count(source_counts.loc[t,'Publication']),b.count(sizes.loc[t])] for t in range(10)],[.5,1,1,1])

    section(10,'Search Top10, bộ lọc và relevance')
    table(['Nhánh','Cơ chế'],[['FULLDOC semantic','Query nhiều cửa sổ → checkpoint mới → Qdrant cosine'],
        ['FULLDOC hybrid','TF-IDF Top100 + FULLDOC dense Top100 → RRF k=60 → Top10'],
        ['4 phương pháp nghiên cứu cũ','TF-IDF, BatteryBERT pretrained semantic/hybrid, BGE semantic'],
        ['V2/V3 lịch sử','BM25 + BGE/BatteryBERT fine-tuned + MiniLM']],[1.2,2.8])
    note('RRF(d) = Σnhánh 1 / [60 + ranknhánh(d)]. Cosine, TF-IDF, RRF và MiniLM khác thang đo; chúng không phải xác suất relevance.')
    p('Nguồn/năm/từ bắt buộc/topic tạo tập doc_id hợp lệ trước khi rank ở cả lexical/dense. Gắn nhãn topic hiện hành sau retrieval bằng doc_id. Encoder query phải đúng checkpoint/pooling với corpus vectors; không thay pretrained cùng768 chiều rồi coi là tương đương.')
    p(f"Benchmark chung: 20 query, 6 phương pháp, 1.200 hàng Top10; pool có {status['pool_pairs']} cặp. Nhãn người được chấp nhận: {status['accepted_human_labels']}; query đủ pool: {status['eligible_queries']}/20. Trạng thái: {status['status']}.")
    table(['FULLDOC warm CPU','Mean giây','Median','Requests'],[[r.method,b.fmt(r.mean,3),b.fmt(r.median,3),str(r.count)] for r in latency.itertuples()],[2.3,.8,.8,.7])
    p('Warm loại request đầu mỗi phương pháp. Đo tuần tự, chưa đo concurrency. Phiếu mới ẩn method/rank/score; grade0–2, P@10/MRR@10 dùng grade≥2, nDCG gain2^grade−1. Chỉ tính trên query có toàn pool được người chấm và người xác minh khác nhau chấm đủ. Nhãn trống không phải0; chưa báo Recall toàn corpus.')
    search_review=j('search_agent_review.json')
    b.story.append(b.PageBreak())
    h('Tự đọc kết quả Search và kiểm tra hiệu năng')
    p(f"Agent đã xem {search_review['title_rows_read']} title Top10 của 20 truy vấn FULLDOC hybrid, đọc đủ Title–Abstract của {search_review['complete_abstract_rows_read']} kết quả Top1 được kiểm tra sâu. Không sinh grade relevance và không dùng để đổi model sau benchmark.")
    for finding in search_review['findings']:p(finding,'small')
    filtering=j('filter_optimization.json')
    p(f"Tối ưu phục vụ sau benchmark: đổi tập doc_id hợp lệ thành point IDs đã xác minh, vẫn lọc trước ranking. Ba lượt trên collection thật ({filtering['eligible_documents']} tài liệu hợp lệ) giữ nguyên Top100 và scores; median riêng bước retrieval {b.fmt(filtering['median_old_seconds'],3)} → {b.fmt(filtering['median_new_seconds'],3)} giây, nhanh {b.fmt(filtering['speedup'],2)} lần. Đây không phải gain relevance và không thay 1.200 thứ hạng benchmark đã lưu.",'small')
    p('Lượt web đầu có một timeout 240 giây ở truy vấn dài/lọc kết hợp; XML thất bại được giữ tại web_initial_tests.xml. Chạy riêng đã qua; profiler xác định cold import/model loading và bộ lọc Qdrant Local có chi phí đáng kể. Kết quả kiểm thử cuối được ghi riêng sau tối ưu. Chưa quy toàn bộ timeout đầu cho một nguyên nhân duy nhất.','small')

    section(11,'Tự kiểm tra, lỗi còn lại và nhận xét')
    review=j('topics/agent_review.json')
    p(f"Đã review {len(set(review['reviewed_document_ids']))} tài liệu đại diện ở 10 topic mới: 3 gần centroid và 3 random có seed mỗi topic. Đây là nhận xét agent, không phải expert ground truth hay ước lượng precision. Phạm vi đọc và các ca lỗi được lưu cùng review.")
    for issue in review.get('main_findings',[]):p(str(issue))
    p(f"Audit keywords cũ: {adjacency['old_keywords']} từ khác nhau đều có support nguồn; không xác nhận danh sách top10 cũ có phrase giả. Sửa mới bảo toàn ranh giới để ngram tương lai không nối qua stopword/field. Số file dữ liệu/checkpoint được xác minh giữ nguyên: {preservation['files']}.")
    table(['Kiểm chứng kỹ thuật mới','Kết quả'],[['Tests riêng biệt',f"{checks['distinct_tests']} qua / {checks['failures']} lỗi / {checks['errors']} errors"],
        ['Numerical parity','Prefix/last-block, full BERT export, query và document API'],
        ['Full-text contracts','Coverage, gradient tail, same-first128/different-tail query'],
        ['Data/topic/search','Hash/ID train-only, keyword support, predict reload, filters'],
        ['Web/HTTP','AppTest và server smoke thực tế']],[1.4,2.7])
    note('Điều kiện dừng là hoàn thành protocol đã định trước và kiểm tra các gate; không tuyên bố đã đạt tối ưu tuyệt đối. Chưa có toàn văn bài báo, nhãn topic/relevance độc lập, kiểm chứng ngoài corpus hoặc đo production đa người dùng.')

    section(12,'Tái lập và hướng nghiên cứu tiếp')
    p('.venv/Scripts/python.exe run_demo.py\n\nChạy train trong thư mục MỚI:\n.venv/Scripts/python.exe scripts/train_batterybert_full_document.py all\n  --output outputs/batterybert_full_document_REPLAY\n.venv/Scripts/python.exe scripts/evaluate_batterybert_full_document.py\n  --output outputs/batterybert_full_document_REPLAY','small')
    p('Giữ đủ frozen corpus/split, checkpoint khởi đầu, cache 11 block và versions. Resume có fingerprint, không sửa source/config rồi tiếp tục dùng tên run cũ. Artifact index/catalog đang phục vụ nằm trong run v1, không tự tải hoặc train khi mở web.')
    table(['Artifacts','Nội dung'],[['training_protocol / code_snapshot','Config, version, SHA nguồn/input và protocol đã đóng băng'],
        ['training_document_coverage / training_coverage','ID optimizer, số đoạn/token thực sự backward'],
        ['selection / paired_training_effect','Lựa chọn validation và so sánh paired sau chọn'],
        ['model / document_embeddings / ordered_doc_ids','Checkpoint export thật và vectors đúng thứ tự'],
        ['topics / annotation_pool / rankings','Topic candidates, đại diện, pool 6 phương pháp và kết quả thật'],
        ['verification_summary / preservation_check','Kiểm thử riêng biệt và bảo toàn dữ liệu']],[1.4,2.7])
    h('Các bước tiếp theo cần thêm bằng chứng')
    p('Ưu tiên tập scientific-topic độc lập có nhãn chuyên gia và các ca giao thoa/ngoài phạm vi. Tạo positives cùng chủ đề giữa Patent/Publication và hard negatives tránh false negatives; chọn bằng validation khoa học. Với full article, cần ingest/version thân bài và đánh giá mới, không tái dùng cache Abstract cho văn bản khác.')
    p('Có thể so sánh hierarchical pooling/adapters/nhiều block và batch lớn hơn khi có tài nguyên; chỉ gọi là cải thiện sau phép so cùng protocol. Giữ baseline và lỗi quan sát được, không sửa test để bắt BatteryBERT thắng.')
    h('Phương pháp và nguồn')
    p('Gao et al. (2021), Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://arxiv.org/abs/2101.06983. Model card: https://huggingface.co/batterydata/batterybert-cased. BERTopic clustering: https://maartengr.github.io/BERTopic/getting_started/clustering/clustering.html','small')
    p('Các số đo nội bộ được đọc từ artifacts; evidence manifest đi kèm ghi SHA-256. Review agent và kế hoạch nghiên cứu được tách khỏi kết quả có nhãn. Báo cáo cũ ngày23/10 được giữ như snapshot lịch sử, có đính chính về tính liền kề tokens.','small')
    target=b.OUT/(NAME+'.pdf')
    SimpleDocTemplate(str(target),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=54,bottomMargin=51,
        title='Báo cáo toàn bộ pipeline và BatteryBERT full-document',author='Sodium-ion Research Explorer').build(b.story,canvasmaker=b.NumberedCanvas)
    import fitz
    pdf=fitz.open(target);outline=[[1,'Tóm tắt điều hành',1]]
    for i,page in enumerate(pdf):
        first=page.get_text().splitlines()[0]
        if re.match(r'^\d{2} / ',first):outline.append([1,first,i+1])
    pdf.set_toc(outline);temporary=target.with_suffix('.outline.pdf');pdf.save(temporary,garbage=4,deflate=True);pdf.close();temporary.replace(target)
    (ROOT/'docs'/(NAME+'.md')).write_text('\n'.join(b.markdown),encoding='utf8')
    sources={p.relative_to(ROOT).as_posix():dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(set(b.SOURCES))}
    (b.OUT/(NAME+'_evidence.json')).write_text(json.dumps(dict(pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        sources=sources,scope='Actual full-document training run and identified historical comparators'),ensure_ascii=False,indent=2),encoding='utf8')
    print(str(target),flush=True)


if __name__=='__main__':build()
