"""Refresh PDF compatibility copy, offline HTML viewer and source ZIP.

Run only after successful compilation and validate_report.py.
"""
from pathlib import Path
import base64
import hashlib
import json
import zipfile
import fitz

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'Sodium_ion_Research_Report.pdf'
    doc = fitz.open(source)
    compatible = ROOT / 'Sodium_ion_Report_compatible.pdf'
    doc.save(compatible, garbage=4, clean=True, deflate=True, use_objstms=0)
    # PDF cleaning can change the in-memory resource cache. Compare two freshly
    # opened on-disk files rather than that modified source document object.
    doc.close()
    doc = fitz.open(source)
    check = fitz.open(compatible)
    assert len(check) == len(doc) and not check.is_encrypted and not check.is_repaired
    for i, page in enumerate(check):
        assert page.get_text() == doc[i].get_text()
        page.get_pixmap(matrix=fitz.Matrix(.2, .2))
    title = f'Báo cáo Sodium-ion · {len(doc)} trang'
    parts = ['<!doctype html><html lang="vi"><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>' + title + '</title><style>'
             'body{margin:0;background:#e5e7eb;font-family:Arial,sans-serif}'
             'header{position:sticky;top:0;background:#183b56;color:white;padding:12px 20px;z-index:1}'
             'header a{color:white}main{max-width:1000px;margin:auto}figure{margin:18px 8px;scroll-margin-top:65px}'
             'img{display:block;width:100%;height:auto;background:white;box-shadow:0 2px 9px #888}'
             'figcaption{text-align:center;padding:7px;font-size:14px}input{width:60px}button{padding:5px 10px}'
             '@media print{header,figcaption{display:none}figure{margin:0;break-after:page}img{box-shadow:none}}'
             '</style><header><strong>' + title + '</strong> &nbsp; '
             '<a href="Sodium_ion_Report_compatible.pdf">Mở PDF</a> &nbsp; Trang '
             f'<input id="pg" type="number" min="1" max="{len(doc)}" value="1">'
             '<button onclick="document.getElementById(\'p\'+document.getElementById(\'pg\').value)?.scrollIntoView({behavior:\'smooth\',block:\'start\'})">Đến</button></header><main>']
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        data = base64.b64encode(pix.tobytes('png')).decode('ascii')
        parts.append(f'<figure id="p{i+1}"><img loading="lazy" width="{pix.width}" height="{pix.height}" '
                     f'alt="Trang {i+1}" src="data:image/png;base64,{data}">'
                     f'<figcaption>Trang {i+1} / {len(doc)}</figcaption></figure>')
    parts.append('</main></html>')
    viewer = ROOT / 'Sodium_ion_Report_view.html'
    viewer.write_text('\n'.join(parts), encoding='utf-8')
    result = {'original_pdf_valid': True, 'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'compatibility_mode': 'rewritten, no object streams', 'compatible_pages': len(check),
              'all_pages_rendered': True, 'text_identical': True, 'html_embedded_pages': len(doc),
              'html_requires_network': False,
              'files': {p.name: p.stat().st_size for p in [source, compatible, viewer]}}
    (ROOT / 'viewer_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    files = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in {'.tex', '.bib', '.md', '.py', '.json'}]
    for d in ['chapters', 'tables', 'figures']:
        files.extend(p for p in (ROOT / d).rglob('*') if p.is_file())
    archive = ROOT / 'report_latex_source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, 'report_latex/' + p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({'pages': len(doc), 'html_pages': len(doc), 'source_zip_files': len(files),
                      'compatibility_and_zip_checks': 'PASS'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
