"""Record the agent's completed qualitative reading; never generate relevance qrels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from src.final_system_artifacts import sha
from src.model_comparison_io import write_json

OUT=ROOT/'outputs/batterybert_full_document_20261024_v1'
NOTES={
    'Q03':'Top1 abstract xác nhận anionic redox trong P2 oxide dù title không nêu redox; ví dụ title-only có thể thiếu bằng chứng.',
    'Q05':'Top1 là review NASICON NVP có carbon coating trong abstract; không phải nghiên cứu thực nghiệm coating duy nhất.',
    'Q08':'Top1 là NaFSI/PEO polymer electrolyte, abstract nêu phép đo ở nhiệt độ cao; chưa thỏa rõ yêu cầu ether liquid và low-temperature. Truy vấn nhiều ràng buộc còn yếu.',
    'Q16':'Top1 là thấm electrolyte bằng hot/cold rolling của cell; chưa phải chính xác thiết bị phủ/ép điện cực. Có lexical overlap nhưng sai công đoạn.',
    'Q17':'Top1 là giảm voltage drop của pack; abstract không chứng minh thermal management. Tài liệu thermal cụ thể ở hạng thấp hơn.',
    'Q18':'Top1 mô tả tái chế vật liệu cathode; nhưng các title Top10 còn cathode review và tái chế khẩu trang, cần review relevance đầy đủ.'}


def main():
    path=OUT/'rankings_top10.csv';rankings=pd.read_csv(path)
    hybrid=rankings.query("method == 'BATTERYBERT_FULLDOC_HYBRID'").sort_values(['query_id','rank'])
    assert len(hybrid)==200 and hybrid.groupby('query_id').size().eq(10).all()
    skim=hybrid[['query_id','query','rank','doc_id','title']].copy()
    skim['reviewer_type']='AGENT_NOT_EXPERT'
    skim['reading_scope']='Top10 title skim, not a relevance grade'
    skim.to_csv(OUT/'search_title_review.csv',index=False)
    sample=hybrid[hybrid['rank'].eq(1)&hybrid.query_id.isin(NOTES)].copy()
    sample['agent_note']=sample.query_id.map(NOTES)
    sample['reading_scope']='Complete available Title and Abstract'
    sample['reviewer_type']='AGENT_NOT_EXPERT'
    sample.to_csv(OUT/'search_abstract_review.csv',index=False)
    write_json(OUT/'search_agent_review.json',dict(
        reviewer_type='AGENT_NOT_EXPERT',independent_ground_truth=False,
        rankings_sha256=sha(path),title_rows_read=len(skim),queries=20,
        complete_abstract_rows_read=len(sample),abstract_review_query_ids=list(NOTES),
        reviewed_abstract_document_ids=sample.doc_id.tolist(),
        labels_used_for_training_or_selection=False,relevance_qrels_generated=False,
        review_is_accuracy_estimate=False,
        findings=list(NOTES.values())+[
            'Các title về hard carbon, MoS2, Sn4P3 và NASICON nhìn chung cùng miền truy vấn; chưa kiểm định mọi ràng buộc trên toàn bộ abstract.',
            'Q20 có hai kết quả cùng title Guidelines for Air-Stable Lithium/Sodium Layered Oxide Cathodes; khác doc_id cần kiểm tra DOI/family trước gộp.',
            'Không dùng các truy vấn benchmark này để chỉnh model sau khi đọc kết quả; giữ nguyên lỗi và thứ hạng làm bằng chứng.']))
    print('RECORDED: 200 title rows and 6 complete abstract rows; no relevance grades')


if __name__=='__main__':main()
