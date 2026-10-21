"""Validate this report and render final PDF locally with PyMuPDF/Pillow.

No model, network, API or database operations. This does not replace human
scientific review or visual inspection of the emitted page images.
"""
from pathlib import Path
import collections
import hashlib
import json
import re
import shutil

import fitz
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    manifest = json.loads((OUT / 'evidence_manifest.json').read_text(encoding='utf-8'))
    changed = [p for p, r in manifest['sources'].items() if digest(ROOT / p) != r['sha256']]
    assert not changed, changed
    paths = [OUT / 'main.tex', *sorted((OUT / 'chapters').glob('*.tex')), *sorted((OUT / 'tables').glob('*.tex'))]
    text = '\n'.join(p.read_text(encoding='utf-8') for p in paths)
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text)
    dup = [x for x, n in collections.Counter(labels).items() if n > 1]
    assert not dup, dup
    assert not set(refs) - set(labels), set(refs) - set(labels)
    for p in re.findall(r'\\input\{([^}]+)\}', text):
        assert (OUT / p).is_file(), p
    catalog = json.loads((OUT / 'figure_catalog.json').read_text(encoding='utf-8'))
    for figure in catalog['items']:
        assert digest(OUT / 'figures' / figure['dest']) == digest(ROOT / figure['source'])
    bib = (OUT / 'references.bib').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    cited = {x.strip() for s in re.findall(r'\\cite\{([^}]+)\}', text) for x in s.split(',')}
    assert cited <= keys, cited - keys
    for p in (OUT / 'chapters').glob('*.tex'):
        s = p.read_text(encoding='utf-8')
        for env in ['equation', 'align', 'figure']:
            assert s.count('\\begin{' + env + '}') == s.count('\\end{' + env + '}'), (p, env)
    # Never read api.env or any credential; scan common key literal shapes only.
    for p in paths + [OUT / 'references.bib']:
        s = p.read_text(encoding='utf-8')
        assert not re.search(r'AIza[0-9A-Za-z_-]{30,}', s), f'Possible credential: {p.name}'
    log = (OUT / 'build/main.log').read_text(encoding='utf-8', errors='replace')
    blg = (OUT / 'build/main.blg').read_text(encoding='utf-8', errors='replace')
    forbidden = ['Undefined control sequence', 'LaTeX Error:', 'Missing character:',
                 'There were undefined references', 'There were undefined citations', 'Overfull \\hbox', 'Overfull \\vbox']
    assert not [x for x in forbidden if x in log], [x for x in forbidden if x in log]
    assert 'Warning--' not in blg and 'error message' not in blg, blg
    pdfpath = OUT / 'build/main.pdf'
    pdf = fitz.open(pdfpath)
    assert len(pdf) > 20
    qa = OUT / 'qa'
    qa.mkdir(exist_ok=True)
    for old in qa.glob('*.png'):
        if re.fullmatch(r'(page|contact|detail)_\d+\.png', old.name):
            old.unlink()  # Only this script's generated previews in report_latex/qa.
    page_checks = []
    thumbnails = []
    for i, page in enumerate(pdf):
        textpage = page.get_text()
        assert '\ufffd' not in textpage and '\x00' not in textpage, i + 1
        assert not re.search(r'\?\?', textpage), f'Unresolved marker on page {i+1}'
        assert len(textpage.strip()) > 20, f'Blank page {i+1}'
        outside = []
        for b in page.get_text('dict')['blocks']:
            for line in b.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if x0 < 10 or y0 < 10 or x1 > page.rect.width - 10 or y1 > page.rect.height - 10:
                        outside.append(span['text'])
        assert not outside, {'page': i + 1, 'outside': outside}
        pix = page.get_pixmap(matrix=fitz.Matrix(0.65, 0.65), alpha=False)
        pix.save(qa / f'page_{i+1:02d}.png')
        thumb = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        thumb.thumbnail((285, 405))
        thumbnails.append(thumb)
        page_checks.append({'page': i + 1, 'word_count': len(textpage.split()), 'text_inside_page': True})
    contacts = []
    for start in range(0, len(thumbnails), 12):
        sheet = Image.new('RGB', (1200, 1305), '#e5e7eb')
        draw = ImageDraw.Draw(sheet)
        for j, thumb in enumerate(thumbnails[start:start+12]):
            x, y = (j % 4) * 300 + 7, (j // 4) * 435 + 20
            sheet.paste(thumb, (x, y))
            draw.text((x, y - 16), f'PDF page {start+j+1}', fill='black')
        dest = qa / f'contact_{start//12+1:02d}.png'
        sheet.save(dest)
        contacts.append(dest.name)
    # High-resolution selected pages for equations, results, tables, references.
    selected = {0}
    needles = ['Top5 Q19 của', 'Relevance AI một phần:', 'LDA grid',
               'Suy luận biến phân', 'Khảo sát LDA ở', 'TÀI LIỆU THAM KHẢO',
               'Exact source-ID recovery', 'UMAP 2D minh họa']
    for i, page in enumerate(pdf):
        if any(n.casefold() in page.get_text().casefold() for n in needles):
            selected.add(i)
    for i in selected:
        pdf[i].get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(qa / f'detail_{i+1:02d}.png')
    shutil.copy2(pdfpath, OUT / 'Sodium_ion_Research_Report.pdf')
    aux = (OUT / 'build/main.aux').read_text(encoding='utf-8')
    actual_figures = len(re.findall(r'\\newlabel\{fig:', aux))
    assert actual_figures == len(catalog['items']) + 1, 'PDF/AUX stale or figures missing; finish compilation first'
    result = {'status': 'PASS_AUTOMATED_REQUIRES_VISUAL_REVIEW', 'pdf_pages': len(pdf),
              'numbered_figures': actual_figures,
              'numbered_tables': len(re.findall(r'\\newlabel\{tab:', aux)),
              'output_figures_embedded': len(catalog['items']),
              'source_files': len(paths), 'citations_resolved': len(cited), 'labels_unique': len(labels),
              'all_source_hashes_unchanged': True, 'source_hash_count': len(manifest['sources']),
              'compile': 'Tectonic 0.17.0, successful PDF and BibTeX, cached packages',
              'undefined_references': 0, 'missing_glyph_warnings': 0, 'overfull_boxes': 0,
              'underfull_hbox_warnings': log.count('Underfull \\hbox'),
              'portability_note': 'Microsoft fonts are system-local; fallback fonts documented in README',
              'secrets_check': 'No API credential read; common literal key pattern absent from report source',
              'new_experiments_executed': False, 'gemini_called': False, 'models_modified': False,
              'rendered_all_pages': True, 'contact_sheets': contacts,
              'detail_pages': [i + 1 for i in sorted(selected)], 'page_checks': page_checks,
              'pdf_sha256': digest(OUT / 'Sodium_ion_Research_Report.pdf')}
    (OUT / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'page_checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
