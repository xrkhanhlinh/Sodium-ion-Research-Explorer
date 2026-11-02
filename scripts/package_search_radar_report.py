"""Package the visually reviewed course report and standalone LaTeX sources."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1];C=ROOT/'course_report_latex'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--visual-reviewed',action='store_true');args=parser.parse_args()
    if not args.visual_reviewed:raise ValueError('Render and visually review final pages before packaging')
    validation=json.loads((C/'validation.json').read_text(encoding='utf8'))
    if validation.get('missing_glyphs') or validation.get('overfull_boxes') or not validation.get('source_hashes_unchanged'):
        raise ValueError('Automated PDF checks must pass before packaging')
    pdf=ROOT/'output/pdf/BAO_CAO_KET_THUC_HOC_PHAN_SODIUM_ION_20261030_TNR13.pdf'
    if sha(pdf)!=validation['pdf_sha256']:raise ValueError('PDF changed since validation')
    validation.update(status='PASS_AUTOMATED_AND_VISUAL_REVIEW',visual_review_new_search_chapter=True)
    (C/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf8')
    target=ROOT/'output/latex/BAO_CAO_KET_THUC_HOC_PHAN_SODIUM_ION_20261030_TNR13_LATEX.zip'
    files=[C/n for n in ('main.tex','typography_13pt.tex','references.bib','README.md','build_report.py','validate_report.py',
                         'evidence_manifest.json','figure_inventory.json','validation.json','topic_catalog_70.csv')]
    files += [p for folder in ('chapters','generated','figures') for p in (C/folder).rglob('*') if p.is_file()]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(files):archive.write(path,path.relative_to(C).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC check failed')
        if 'chapters/08_search_radar.tex' not in archive.namelist():raise ValueError('Missing new chapter')
    result=dict(status='PASS',pdf=str(pdf),source_zip=str(target),pdf_pages=validation['pages'],
        figures=validation['figures'],tables=validation['tables'],chapters=9,topics=70,
        body_font='Times New Roman',body_size_tex_pt=13,cover_preserved=validation['cover_preserved_exactly'],
        source_zip_files=len(files),pdf_sha256=sha(pdf),source_zip_sha256=sha(target),
        new_search_pilot_results=True,human_heldout_quality_claimed=False)
    (ROOT/'output/latex/course_report_delivery_20261030.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
