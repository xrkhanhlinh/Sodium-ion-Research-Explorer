"""Finalize Stage 1 only after a fresh, successful real-data test suite."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .data_io import sha256
from .pipeline import progress, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "outputs/reports"
    summary_path = report_dir / "stage1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test_file = report_dir / "tests.xml"
    if not test_file.exists() or test_file.stat().st_mtime < summary_path.stat().st_mtime:
        raise RuntimeError("Run the complete test suite after the latest pipeline execution")
    tree = ET.parse(test_file)
    cases = tree.findall(".//testcase")
    failures = tree.findall(".//failure") + tree.findall(".//error")
    skipped = tree.findall(".//skipped")
    if not cases or failures or skipped:
        progress(root, "test suite requires resolution", "PARTIAL", f"{len(cases)} tests, {len(failures)} failures/errors, {len(skipped)} skipped")
        raise RuntimeError("All unit and integration tests must pass without skips")
    validation = json.loads((report_dir / "data_validation_report.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS":
        raise RuntimeError("Runtime data validation has not passed")
    for source, digest in summary["input_hashes"].items():
        if sha256(root / f"{source}.xlsx") != digest:
            raise RuntimeError("An input workbook changed after validation")
    summary["status"] = "COMPLETED"
    summary["tests"] = dict(passed=len(cases), failed=0, skipped=0)
    write_json(report_dir / "stage1_summary.json", summary)
    lines = ["# STAGE 1 COMPLETION REPORT", "", "Trạng thái: **COMPLETED** — chỉ Stage 1; Báo cáo phạm vi xử lý dữ liệu; xem docs/FINAL_RESEARCH_REPORT.md cho hệ thống hiện tại.", "", "## Số lượng bản ghi", "", "| Nguồn | Đầu vào | Clean master giữ toàn bộ | Corpus chính |", "|---|---:|---:|---:|"]
    for source in ["Patent", "Publication"]:
        lines.append(f"| {source} | {summary['input_counts'].get(source,0):,} | {summary['clean_master_counts'].get(source,0):,} | {summary['main_corpus_counts'].get(source,0):,} |")
    lines += ["", f"Corpus LDA và BERT có cùng **{summary['main_corpus_total']:,} doc_id**. Có **{summary['finetuning_eligible']:,}** bản ghi đủ điều kiện tạo cặp Title–Abstract ở giai đoạn sau; chưa chia tập hoặc huấn luyện.", "", "## Xử lý và bản ghi cần xem xét", "", f"- Phục hồi cấu trúc có kiểm chứng: {summary['repaired']} bản ghi; lỗi cấu trúc chưa giải quyết: {summary['unresolved_structure']}.", f"- Loại khỏi corpus chính do trùng bản ghi được xác nhận: {summary['confirmed_duplicate_exclusions']}; tất cả vẫn còn trong clean master.", f"- Gắn cờ rút lại theo nguồn: {summary['retracted']} bản ghi; không đưa vào corpus chính.", f"- Có {summary['needs_review']:,} bản ghi thuộc các nhóm chờ xem xét, lưu trong `outputs/reports/needs_review.csv`.", "- Giữ nguyên dữ liệu gốc, giữ literal NA và công thức hóa học; không tạo Abstract giả hoặc loại bản ghi do thiếu Keywords.", "- Patent khác native ID không bị gộp chỉ vì cùng Title/Abstract. Nhóm exact/near-duplicate và liên kết sáng chế được lưu để hỗ trợ chia tập sau này.", "", "### Phân nhóm xử lý không cộng trùng", "", "| Nguồn | Nhóm | Số bản ghi |", "|---|---|---:|"]
    for row in summary["dispositions"]:
        lines.append(f"| {row['source']} | {row['disposition']} | {row['count']:,} |")
    lines += ["", "### Relevance heuristic, chưa được chuyên gia xác nhận", "", "| Nguồn | Nhãn | Số bản ghi |", "|---|---|---:|"]
    for row in summary["relevance"]:
        lines.append(f"| {row['source']} | {row['relevance']} | {row['count']:,} |")
    lines += ["", "## Features và chính sách corpus", "", "Giữ 14 feature: doc_id, source, title, abstract, text_for_model, year, authors, organization, keywords, classification, identifier, document_type, language, source_name. Thêm provenance, chất lượng, eligibility, original_doc_id và split_group_id. Metadata mở rộng giữ tất cả trường gốc cùng ngữ nghĩa nguồn.", "", "Clean master giữ toàn bộ bản ghi. Corpus chính chọn CORE/RELATED, văn bản được nhận diện English đủ ngưỡng, không có lỗi chưa giải quyết, không rút lại và không phải bản ghi trùng đã xác nhận. Fine-tuning còn yêu cầu Title và Abstract thực tế đều có nội dung. Thiếu Keywords không phải tiêu chí loại.", "", "## Đầu ra", "", "- Raw: `data/interim/patent_raw.parquet`, `publication_raw.parquet`, `combined_raw.parquet`.", "- Clean masters: `data/processed/patent_clean.parquet`, `publication_clean.parquet`, `combined_clean.parquet`.", "- Corpus: `data/processed/main_corpus.parquet`, `lda_tokens.parquet`, `bert_texts.parquet`.", "- Metadata: `data/processed/metadata_extended.parquet`.", "- Reports: `outputs/reports/data_quality_report.csv`, `duplicate_report.csv`, `relevance_audit.csv`, `cleaning_log.csv`, `cleaning_summary.csv`, `needs_review.csv`, `structural_issues.csv`, `retracted_records.csv`, `language_audit.csv`, `text_quality_audit.csv`, `data_validation_report.json/csv`, `input_audit.json`, `environment_report.json`, `tests.xml`.", "- Configuration: `configs/config.yaml`; dependency pins: `requirements.txt`; specification: `docs/PROJECT_SPEC.md`.", "", "## Sáu biểu đồ và bảng nguồn", ""]
    manifest = json.loads((report_dir / "figure_manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        lines.append(f"- `{item['figure']}` — {item['dpi']} DPI; bảng: `{item['table']}`.")
    lines += ["", "## Kiểm chứng", "", f"- **PASS: {len(cases)} unit/integration tests**, 0 FAIL, 0 SKIP.", f"- **PASS: {len(validation['checks'])} kiểm tra dữ liệu và file đầu ra**.", "- Corpus LDA/BERT cùng thứ tự IDs; tất cả công thức được bộ nhận diện phát hiện trong corpus đều được kiểm tra bảo toàn.", "- SHA-256 của cả hai workbook gốc không thay đổi.", "", "## Hạn chế còn tồn tại", "", "- Relevance và nhận diện ngôn ngữ là heuristic, không phải ground truth; cần chuyên gia rà soát CORE/RELATED cũng như hàng đợi UNCERTAIN.", "- Near-duplicate dùng chặn ứng viên theo Title; chưa bao phủ mọi bản sao ngữ nghĩa có Title khác biệt lớn.", "- Liên kết priority/PN là ứng viên họ sáng chế, chưa phải family ID được xác minh.", "- Retraction chỉ phản ánh nhãn trong tệp; không kiểm tra trạng thái nhà xuất bản theo thời gian thực.", "- Năm Patent lấy ngày công bố hợp lệ sớm nhất trong PD cho mỗi bản ghi; không đại diện mọi thành viên họ sáng chế.", "- Công thức hóa học được bảo vệ bằng quy tắc từ vựng và bộ kiểm thử, chưa phải parser hóa học đầy đủ; chưa đo độ dài bằng tokenizer BERT.", "- GPU chưa xác minh vì quyền CIM bị hạn chế và không tìm thấy nvidia-smi; Stage 1 đã chạy CPU.", "", "Báo cáo này chỉ mô tả bước chuẩn bị dữ liệu, không thay thế báo cáo tổng hợp hiện tại."]
    (root / "outputs/reports/stage1").mkdir(parents=True, exist_ok=True)
    (root / "outputs/reports/stage1/COMPLETION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress(root, "Stage 1 completed and verified", "COMPLETED", f"{len(cases)} tests PASS; {summary['main_corpus_total']:,} main-corpus records; all {sum(summary['input_counts'].values()):,} source records preserved. See outputs/reports/stage1/COMPLETION_REPORT.md. Stage-local report exported.")
    output_manifest = []
    for folder in ["data", "outputs", "configs", "docs", "src", "tests"]:
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.name in {"output_manifest.json", "stage1.log"}:
                continue
            output_manifest.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha256(path)))
    for name in ["requirements.txt", "run_stage1.py", "pytest.ini", ".gitignore"]:
        path = root / name
        output_manifest.append(dict(path=name, bytes=path.stat().st_size, sha256=sha256(path)))
    write_json(report_dir / "output_manifest.json", dict(stage=1, status="COMPLETED", artifacts=output_manifest))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
