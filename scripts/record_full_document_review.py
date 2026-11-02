"""Record completed agent reading notes; never manufactures expert labels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.model_comparison_io import write_json

OUT=ROOT/'outputs/batterybert_full_document_20261024_v1/topics'
LABELS={0:'Oxide Ti/Sn, titanate và composite điện cực',1:'Prussian blue và các hợp chất điện cực chứa sắt',
2:'Oxide lớp P2/O3 và biến đổi cấu trúc điện cực',3:'Điện ly, polymer, muối và phụ gia giao diện',
4:'Phosphate vanadium, NVP/NASICON và composite carbon',5:'Anode hợp kim Sn/Sb, graphene và MXene',
6:'Phần cứng cell và thiết bị chế tạo, kiểm tra pin',7:'Cấu trúc pha, trật tự vacancy và dẫn ion',
8:'Hard carbon, carbon xốp và tiền chất sinh khối',9:'Sulfide/selenide, MoS2 và composite carbon'}
NOTES={
'patent:201566599A':('OVERLAP','Patent liệt kê nhiều oxide/phosphate/carbon/sulfide; không phải một tài liệu TiO2 chuyên biệt.'),
'publication:WOS:000489039800001':('OVERLAP','Vanadium oxide/rGO; cùng nền oxide-composite nhưng ngoài trọng tâm Ti/Sn.'),
'publication:WOS:000453488500037':('OVERLAP','Fe2(MoO4)3 cathode; cùng Fe/precipitation nhưng không phải Prussian blue.'),
'patent:202193782F':('MISPLACED','Fe7S8/carbon, khác framework PB; raw patent gọi cathode, không tự sửa cực tính.'),
'patent:202220171B':('MISPLACED','Iron-sodium phosphate, không phải PB; còn mâu thuẫn tên lithium/sodium và anode trong metadata.'),
'patent:202387232U':('UNCERTAIN','O3-layered oxide phù hợp family; raw patent gọi anode, cần chuyên gia xác minh vai trò.'),
'publication:WOS:000484981000037':('MISPLACED','Conjugated coordination polymer redox-active electrode, không chứng minh là electrolyte.'),
'publication:WOS:000974150900001':('OVERLAP','Polymer binder/SEI trên hard carbon; giao thoa vật liệu điện cực và điện ly-giao diện.'),
'patent:201650828K':('MISPLACED','Metal-oxide/polypyrrole composite, không có NVP/phosphate được nêu trong abstract.'),
'publication:WOS:000434625200002':('OVERLAP','Review vanadium-based cathodes gồm cả oxide, phosphate và fluorophosphate.'),
'patent:202162319E':('MISPLACED','Formula P2-Na0.7MnO2 là trọng tâm oxide lớp; không suy ra phosphate từ tên dịch có phosphorus.'),
'patent:202161705B':('MISPLACED','Sodium nickel-metal oxide/carbon, khác trọng tâm hợp kim Sn/Sb.'),
'patent:202374381T':('OVERLAP','Thiết bị sấy chế tạo điện cực, thuộc hardware/manufacturing rộng hơn cấu trúc cell.'),
'publication:WOS:000874406600005':('OVERLAP','Vanadate glass/glass-ceramics, nghiên cứu cấu trúc-conductivity; khác NASICON đơn nhất.'),
'publication:WOS:000929116700021':('OVERLAP','Zn borate@carbon anode; carbon coating không đồng nghĩa vật liệu chính là hard carbon.'),
'patent:201815458Q':('UNCERTAIN','Biomass porous carbon phù hợp chất liệu, nhưng patent gọi cathode; không sửa nguồn bằng suy đoán.'),
'patent:202215856L':('UNCERTAIN','Selenide phù hợp family; patent gọi cathode trong khi trọng tâm nhóm nhiều publication anode.')}


def main():
    reps=pd.read_csv(OUT/'representative_documents.csv',keep_default_na=False)
    if len(reps)!=60 or not reps.groupby('topic_id').size().eq(6).all():raise ValueError('Review sample differs from the completed reading')
    reps['reviewer_type']='AGENT_NOT_EXPERT';reps['reading_scope']='Complete available Title and Abstract; no full article'
    reps['agent_status']=[NOTES.get(d,('ALIGNED','Title/Abstract khớp trọng tâm provisional topic; không phải xác nhận chuyên gia.'))[0] for d in reps.doc_id]
    reps['agent_note']=[NOTES.get(d,('ALIGNED','Title/Abstract khớp trọng tâm provisional topic; không phải xác nhận chuyên gia.'))[1] for d in reps.doc_id]
    reps.to_csv(OUT/'scientific_review.csv',index=False)
    pd.DataFrame([dict(topic_id=t,topic_label=label,label_status='AGENT_INTERPRETATION_NOT_EXPERT') for t,label in LABELS.items()]).to_csv(OUT/'topic_labels.csv',index=False)
    findings=[
        '30 mẫu central cho thấy trọng tâm nội dung của10 nhóm; mẫu random còn giao thoa và lệch nhóm. Đây là nhận xét agent, không phải ước lượng precision.',
        'Nhóm7 có đại diện NASICON, trật tự pha/vacancy và dẫn ion; đây là chủ đề cơ chế/cấu trúc, không phải một họ vật liệu duy nhất. Từ khóa na/compound/active vẫn chưa đặc trưng.',
        'Nhóm6 thực sự có cấu trúc cell, thiết bị sấy và kiểm tra. connect/plate/end là ngôn ngữ phần cứng; blacklist không sửa được thành viên bị gán lệch.',
        'Các ca lệch đã thấy: Fe-phosphate trong nhóm trọng tâm PB, P2 oxide trong nhóm NVP, sodium-nickel oxide trong nhóm Sn/Sb, polymer điện cực hoạt tính trong nhóm điện ly.',
        'Một số patent gọi anode/cathode chưa nhất quán với mô tả vật liệu; không tự sửa cực tính của nguồn bằng suy đoán.',
        'Top2 toàn corpus giảm27,49%→24,59%; nhưng ARI test giảm0,9245→0,6888, diversity0,80→0,77 và C_v chung reference0,560261→0,553022. Không tuyên bố topic mới thắng mọi chỉ số.',
        'Tích hợp tạm thời vì encoder đã học các đoạn đầy đủ và qua cổng kỹ thuật/review định trước. Giữ topic cũ làm đối chứng ổn định seed tốt hơn; còn chờ nhãn chuyên gia độc lập.'
    ]
    write_json(OUT/'agent_review.json',dict(reviewer_type='AGENT_NOT_EXPERT',independent_ground_truth=False,
        reviewed_document_ids=reps.doc_id.drop_duplicates().tolist(),reviewed_topic_ids=list(range(10)),
        reading_scope='Complete available Title and Abstract, all 60 sample rows; no article full text',
        sample_rows=len(reps),distinct_documents=reps.doc_id.nunique(),status_counts=reps.agent_status.value_counts().to_dict(),
        recommendation_checks=dict(two_central_per_label=True,content_not_just_genre=True,hardware_named_honestly=True,overlap_documented=True),
        recommend_as_main_topic_pipeline=True,main_findings=findings,
        labels_used_for_training_or_selection=False,review_is_accuracy_estimate=False))
    print(reps.agent_status.value_counts().to_string(),flush=True)


if __name__=='__main__':main()
