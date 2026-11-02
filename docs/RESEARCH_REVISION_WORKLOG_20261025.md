# Rà soát và cải thiện nghiên cứu ngày 25 tháng 10 năm 2026

## Phạm vi được thực hiện

Người dùng yêu cầu rà soát toàn bộ bài, sửa các điểm chưa logic, kiểm tra chất lượng topic và từ nhiễu, kiểm tra Top 10 tìm kiếm và ý nghĩa các chỉ số, chạy lại hoặc tinh chỉnh mô hình khi có căn cứ. Việc cải thiện được thực hiện trên phiên bản mới; không sửa corpus, trọng số, rankings, qrels và số đo đã đóng băng để làm đẹp kết quả.

## Những vấn đề đã xác nhận trước thí nghiệm

1. Bản LaTeX `report_latex/main.tex` và các chương chốt ngày 19/10, mô tả Search v2 Top 5 và BERTopic 84/92 topic. Pipeline ngày 24/10 đã chuyển sang Search Top 10 và có các catalog K10; cần tách snapshot lịch sử khỏi bài hiện hành.
2. LDA Stage2 chọn K20, trong khi bảng catalog đối chứng dùng K10. Đây là hai thí nghiệm khác nhau, không được mô tả như một kết quả duy nhất.
3. BERTopic–BatteryBERT pretrained có coherence cao nhưng hai nhóm chứa khoảng 92,76% corpus. Coherence không đủ làm tiêu chí lựa chọn model duy nhất.
4. Nhánh full-document đọc hết Abstract nhưng một số chỉ số topic kém bản optimized 23/10. Việc đồng nhất encoder Search với encoder topic không có căn cứ bắt buộc.
5. Bộ 100 keywords hiện hành có từ chung và thao tác như `solution`, `dry`, `type`, các từ miền rất rộng như `sodium`, `carbon`, và một số cặp alias chiếm nhiều vị trí. Đây là vấn đề đại diện topic; việc chỉ giấu từ không cải thiện thành viên cụm.
6. Keyword `Na3V2(PO4)` cần kiểm tra vì raw fields có cách viết `Na3V2(PO4)(3)`. Phải bảo toàn công thức và lưu provenance khi chuẩn hóa các cách viết tương đương.
7. Search có lỗi truy vấn nhiều điều kiện trong review cũ. Điểm cosine/RRF/MiniLM là điểm xếp hạng, chưa phải relevance ground truth.
8. Pool Search hiện tại có 703 cặp, 20 query và 0 nhãn người được chấp nhận. Không có điểm relevance chính thức thấp để tùy ý tối ưu; các trường trống không phải 0.

## Ba giả thuyết và thí nghiệm tách biệt

- Keywords: giữ assignments và encoder, cải thiện mô tả bằng cụm từ có bằng chứng train và độ phân biệt topic. Đo proxy dự đoán cluster từ Top 10 keywords trên dev; giữ một tập nhóm riêng cho gate của coordinator.
- Geometry: giữ embeddings và công thức keywords, chạy lại KMeans với nhiều khởi tạo và source whitening 32/64 chiều. So stability nhiều seed cùng các ràng buộc coherence, source và kích thước cụm.
- Search: giữ encoder, corpus và benchmark Q01–Q20, thử MiniLM reranking theo passage trên một proxy mới Title→Abstract. Title bị loại khỏi đoạn đưa vào reranker; vẫn phải công bố bias vì vector tài liệu ban đầu chứa Title.

Mỗi hướng có protocol đóng băng trước khi tìm ứng viên. Các executor chạy trong Git worktree riêng; coordinator ghi Arbor và thực hiện gate. Các split/corpus đã được sử dụng trong lịch sử; kết quả mới là chẩn đoán phát triển có gate giữ riêng trong lần sửa này, không phải đánh giá ngoài độc lập.

## Nguồn và trạng thái

- Snapshot mã trước sửa: `outputs/research_revision_20261025_v1/material`, commit `cf0b428`.
- Worktrees: `keywords`, `geometry`, `search` dưới cùng thư mục.
- Kết quả trước sửa và báo cáo 31 trang được giữ để đối chiếu.
- Bản báo cáo cuối sẽ phân biệt thay đổi được chấp nhận, ứng viên bị loại, giới hạn chưa giải quyết và công việc cần nhãn chuyên gia.

## Hoàn tất đợt tiếp tục ngày 26/10/2026

Đã đối chiếu trạng thái thực tế và nhận lại các output hoàn tất trước gián đoạn. Không train lại encoder, không tính lại embeddings, không thay corpus hoặc bảng xếp hạng benchmark đã đóng băng. Các thử nghiệm và output mới được tách phiên bản; 16 artifact gốc có hash giữ nguyên.

### Quyết định theo bằng chứng

- **Chấp nhận geometry source_n50:** giữ K=10 và source projection fit trên train, tăng số khởi tạo KMeans lên 50. Gate seed giữ riêng có ARI 0,686587 → 0,991808; document NPMI theo protocol geometry 0,079191 → 0,110046. Source association và top2 share tăng nhẹ, vẫn trong ràng buộc; không tuyên bố thắng mọi mặt.
- **Chấp nhận sửa parser công thức:** chuẩn hóa các biến thể được hỗ trợ như Na3V2(PO4)(3) thành Na3V2(PO4)3, giữ nguyên span chưa hỗ trợ và raw fields. Hoàn tất 54 shards cho 13.702 tài liệu; provenance đầy đủ. Catalog có 10 topic, 100 keywords, 60 representatives và audit từ chung/nhiễu. Tên topic được diễn giải bằng AI từ keywords và Title–Abstract, không phải nhãn chuyên gia.
- **Loại cả 5 recipe keyword thay thế:** giảm từ chung không đủ để chấp nhận nếu coherence hoặc coverage giảm. Giữ weighting thống kê gốc và công khai audit; không giấu từ để làm đẹp kết quả.
- **Loại reranker:** dev MRR@10 tăng nhưng gate giảm 0,927083 → 0,850694. Search hybrid mặc định được giữ. Không mở thêm vòng tuning trên gate.
- **Đối chứng:** báo cáo phân biệt LDA K20 được chọn và catalog K10, BERTopic pretrained có hai cụm rất lớn, các mô hình lịch sử và giới hạn của proxy Title→document. NPMI của geometry và NPMI trên canonical reference là hai phép đo khác nhau.

### Silver Labels theo yêu cầu pilot

Không có file mới đính kèm trong lượt yêu cầu, nên dùng `outputs/batterybert_full_document_20261024_v1/annotation_pool.csv` hiện có: 703 cặp duy nhất, không thiếu query/title, 28 abstract trống toàn pool. Đọc và đánh giá độc lập 20 cặp đầu theo thứ tự; cả 20 thuộc Q01. Không dùng score, thứ hạng hoặc tên Search model để quyết định nhãn.

Kết quả: 11 nhãn 2, 4 nhãn 1, 4 nhãn 0, 1 nhãn trống; 8 trường hợp review_required=TRUE. Một bản ghi có bằng chứng vai trò cathode/negative electrode mâu thuẫn được để trống. CSV đúng 7 cột yêu cầu, giữ nguyên 20 query_id–doc_id; kiểm tra mọi evidence là trích dẫn trong Title hoặc Abstract. Checkpoint mỗi 5 nhãn. 683 cặp còn lại pending ngoài phạm vi pilot; không nhập Silver Labels vào nhãn người hoặc dùng chọn mô hình.

### Kiểm chứng, resume và bàn giao

- 102 tests của các phần bị ảnh hưởng PASS, 0 failures/errors/skips; routing smoke có 10 kết quả đúng topic mới, không gọi encoder.
- Runner lưu completed/running/failed/pending, receipt/hash, PID lock, thời gian và logs; nhận lại output hợp lệ hoặc tiếp tục các shard thiếu.
- Báo cáo v2 có 22 trang, 8 hình kết quả và 22 bảng; đã xem toàn bộ trang render, sửa số trang và vị trí nhãn biểu đồ trong phiên bản mới, giữ bản cũ.
- Lần QA đầu báo mismatch duy nhất ở `execution/pipeline_state.json` vì log đang tiến triển sau khi build báo cáo. Giữ nguyên log/validation thất bại; validator cuối tách live state khỏi nguồn khoa học bất biến và kiểm tra mọi upstream stage completed. Không thay kết quả nghiên cứu hoặc PDF để vượt kiểm tra.
- Báo cáo hiện hành: `output/pdf/BAO_CAO_NGHIEN_CUU_CAI_THIEN_SODIUM_ION_20261026_v2.pdf`; nguồn Markdown cùng tên dưới `docs/`. Trạng thái, receipts và logs tại `outputs/research_revision_20261025_v1/execution/`.

Relevance người dùng và độ đúng của tên topic vẫn cần nhãn chuyên gia/người dùng độc lập. Coherence, ARI, Silver Labels và tests kỹ thuật không được gọi là accuracy khoa học hoặc relevance ground truth.
