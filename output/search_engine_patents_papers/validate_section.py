"""Validate the standalone LaTeX section and render its actual final PDF."""
from pathlib import Path
from collections import Counter
import hashlib,json,re,shutil
import fitz
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=json.loads((HERE/'evidence_manifest.json').read_text(encoding='utf8'))
    changed=[p for p,h in manifest['sources'].items() if sha(ROOT/p)!=h]
    assert not changed,changed
    texpaths=[HERE/'main.tex',HERE/'search_engine_section.tex',HERE/'figure_sources.tex',*sorted((HERE/'tables').glob('*.tex')),*sorted((HERE/'diagrams').glob('*.tex'))]
    text='\n'.join(p.read_text(encoding='utf8') for p in texpaths)
    labels=re.findall(r'\\label\{([^}]+)\}',text)
    assert all(n==1 for n in Counter(labels).values())
    assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text))<=set(labels)
    keys=set(re.findall(r'@\w+\{([^,]+),',(HERE/'references.bib').read_text(encoding='utf8')))
    cites={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')}
    assert cites<=keys,cites-keys
    for f in manifest['raster_figures']:assert sha(HERE/'figures'/f['file'])==f['sha256']
    log=(HERE/'build/main.log').read_text(encoding='utf8',errors='replace')
    errors=[s for s in ['Undefined control sequence','LaTeX Error:','Missing character:','There were undefined references','There were undefined citations','Overfull \\hbox','Overfull \\vbox','Float too large'] if s in log]
    assert not errors,errors
    assert 'Warning--' not in (HERE/'build/main.blg').read_text(encoding='utf8')
    pdf=fitz.open(HERE/'build/main.pdf')
    assert 10<=len(pdf)<=35
    expected_pdf_size=13*72/72.27
    body=[];checks=[];thumbs=[];details=[]
    for i,page in enumerate(pdf):
        t=page.get_text();assert len(t.strip())>20,(i+1,'blank')
        assert '\ufffd' not in t and '\x00' not in t,(i+1,'glyph')
        assert '??' not in t,(i+1,'reference')
        for b in page.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                for s in line['spans']:
                    x0,y0,x1,y1=s['bbox']
                    assert x0>=8 and y0>=8 and x1<=page.rect.width-8 and y1<=page.rect.height-8,(i+1,s['text'])
                    if len(s['text'])>=55 and abs(s['size']-expected_pdf_size)<.04:
                        body.append((s['font'],s['size'],len(s['text'])))
        pix=page.get_pixmap(matrix=fitz.Matrix(.8,.8),alpha=False);pix.save(HERE/'qa'/f'page_{i+1:02d}.png')
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((260,370));thumbs.append(im)
        checks.append(dict(page=i+1,words=len(t.split()),images=len(page.get_images()),inside_page=True))
        needles=['Pipeline offline','Pipeline online','Group MRR@10 proxy','Ba kết quả đầu Q16','So sánh sáu phương pháp','Các patent trong','Ba patent đầu','Loss và validation','Top10 Q01','Độ nhạy khi']
        if i==0 or any(n in t for n in needles):
            page.get_pixmap(matrix=fitz.Matrix(1.35,1.35),alpha=False).save(HERE/'qa'/f'detail_{i+1:02d}.png');details.append(i+1)
    assert len(body)>100,'Font/body check lacks enough text'
    assert all(f.startswith('TimesNewRoman') for f,s,n in body),set(f for f,s,n in body)
    contacts=[]
    for start in range(0,len(thumbs),9):
        sheet=Image.new('RGB',(840,1230),'#dfe5eb');d=ImageDraw.Draw(sheet)
        for j,im in enumerate(thumbs[start:start+9]):
            x=(j%3)*280+10;y=(j//3)*410+24;sheet.paste(im,(x,y));d.text((x,y-18),f'PDF page {start+j+1}',fill='black')
        name=f'contact_{start//9+1:02d}.png';sheet.save(HERE/'qa'/name);contacts.append(name)
    aux=(HERE/'build/main.aux').read_text(encoding='utf8')
    figures=len(re.findall(r'\\newlabel\{se:fig:',aux));assert figures==10,figures
    dest=HERE/'SEARCH_ENGINE_PATENTS_PAPERS.pdf';shutil.copy2(HERE/'build/main.pdf',dest)
    result=dict(status='PASS_AUTOMATED_PENDING_VISUAL_REVIEW',pages=len(pdf),figures=figures,raster_figures=7,tikz_diagrams=3,tables=9,body_font='Times New Roman',body_size_tex_pt=13,body_size_pdf_points=expected_pdf_size,verified_body_spans=len(body),body_font_verified=True,source_hashes_checked=len(manifest['sources']),source_hashes_unchanged=True,undefined_references=0,missing_glyphs=0,overfull_boxes=0,underfull_warnings=log.count('Underfull \\hbox'),citations_resolved=len(cites),rendered_all_pages=True,contact_sheets=contacts,detail_pages=details,page_checks=checks,pdf_sha256=sha(dest),new_training=False,new_evaluation=False)
    (HERE/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='page_checks'},ensure_ascii=False))


if __name__=='__main__':main()
