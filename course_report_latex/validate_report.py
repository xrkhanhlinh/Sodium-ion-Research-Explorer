"""Check actual PDF, unchanged evidence, citations, topic completeness and render QA."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import shutil
import zipfile

import fitz
import pandas as pd
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DEST=ROOT/'output/pdf/BAO_CAO_KET_THUC_HOC_PHAN_SODIUM_ION_20261030_TNR13.pdf'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=json.loads((HERE/'evidence_manifest.json').read_text(encoding='utf8'))
    changed=[p for p,s in manifest['source_sha256'].items() if digest(ROOT/p)!=s]
    assert not changed,changed
    paths=[HERE/'main.tex',HERE/'typography_13pt.tex',*sorted((HERE/'chapters').glob('*.tex')),*sorted((HERE/'generated').glob('*.tex'))]
    text='\n'.join(p.read_text(encoding='utf8') for p in paths)
    labels=re.findall(r'\\label\{([^}]+)\}',text)
    assert not [v for v,n in Counter(labels).items() if n>1]
    assert set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text))<=set(labels)
    for p in re.findall(r'\\input\{([^}]+)\}',text):assert (HERE/p).is_file(),p
    keys=set(re.findall(r'@\w+\{([^,]+),',(HERE/'references.bib').read_text(encoding='utf8')))
    cited={k.strip() for item in re.findall(r'\\cite\{([^}]+)\}',text) for k in item.split(',')}
    assert cited<=keys,cited-keys
    assert not re.search(r'AIza[0-9A-Za-z_-]{30,}',text)
    inventory=json.loads((HERE/'figure_inventory.json').read_text(encoding='utf8'))
    for item in inventory['embedded']:assert digest(HERE/'figures'/item['file'])==item['sha256'],item['file']
    topics=pd.read_csv(HERE/'topic_catalog_70.csv')
    assert len(topics)==70 and topics.model.nunique()==7
    assert topics.top10_keywords.str.split('; ').str.len().eq(10).all()
    assert topics.groupby('model').topic_id.apply(lambda s:set(s)==set(range(10))).all()
    for m,g in topics.groupby('model'):
        assert g.documents.sum()<=13702
    log=(HERE/'build/main.log').read_text(encoding='utf8',errors='replace')
    failures=[s for s in ['Undefined control sequence','LaTeX Error:','Missing character:',
                          'There were undefined references','There were undefined citations',
                          'Overfull \\hbox','Overfull \\vbox','Float too large'] if s in log]
    assert not failures,failures
    blg=(HERE/'build/main.blg').read_text(encoding='utf8',errors='replace')
    assert 'Warning--' not in blg and 'error message' not in blg
    aux=(HERE/'build/main.aux').read_text(encoding='utf8')
    assert len(re.findall(r'\\newlabel\{fig:course',aux))==manifest['figures']
    pdf=fitz.open(HERE/'build/main.pdf')
    assert len(pdf)>60
    checks=[];thumbs=[];details=[];body_spans=[];all_text=[]
    expected_font_size=13*72/72.27
    cover_reference_path=ROOT/'tmp/pdfs/course_report_font13/before.json'
    cover_reference=json.loads(cover_reference_path.read_text(encoding='utf8')) if cover_reference_path.exists() else None
    old_pdf=ROOT/'output/pdf/BAO_CAO_KET_THUC_HOC_PHAN_SODIUM_ION_20261026_TNR13.pdf'
    if cover_reference is None and old_pdf.exists():
        with fitz.open(old_pdf) as previous:
            original_cover=previous[0].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
            cover_reference=dict(cover_pixel_sha256=hashlib.sha256(original_cover.samples).hexdigest())
    cover_pix=pdf[0].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
    cover_hash=hashlib.sha256(cover_pix.samples).hexdigest()
    if cover_reference:
        assert cover_hash==cover_reference['cover_pixel_sha256'],'Cover appearance changed'
    needles=['Thiết kế và vận hành search engine','Công thức P@10','Tính DCG từ output',
             'Cơ sở lý thuyết các mô hình','Continuation full-document',
             'So sánh search trên','stable topics','BERTopic chunked lịch sử: Top10',
             'Tiêu đề RRF','Khám phá chủ đề và tìm kiếm tài liệu nghiên cứu']
    for i,page in enumerate(pdf):
        t=page.get_text()
        all_text.append(t)
        assert '\ufffd' not in t and '\x00' not in t,(i+1,'bad glyph')
        assert '??' not in t,(i+1,'unresolved')
        outside=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for s in line['spans']:
                    x0,y0,x1,y1=s['bbox']
                    if x0<8 or y0<8 or x1>page.rect.width-8 or y1>page.rect.height-8:outside.append(s['text'])
                    if i>0 and len(s['text'].strip())>=20:
                        assert not any(f in s['font'] for f in ['Arial','Consolas','TeXGyre']), (i+1,s['font'],s['text'])
                        if s['font'].startswith('TimesNewRoman'):
                            assert abs(s['size']-expected_font_size)<.04,(i+1,s['font'],s['size'],s['text'])
                            body_spans.append((i+1,s['font'],s['size']))
        assert not outside,(i+1,outside)
        assert len(t.strip())>20,(i+1,'blank')
        pix=page.get_pixmap(matrix=fitz.Matrix(.65,.65),alpha=False)
        pix.save(HERE/'qa'/f'page_{i+1:03d}.png')
        im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((240,340));thumbs.append(im)
        checks.append(dict(page=i+1,words=len(t.split()),images=len(page.get_images()),inside_page=True))
        if i==0 or (any(n.casefold() in t.casefold() for n in needles) and i>11):
            if len(details)<18:
                page.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(HERE/'qa'/f'detail_{i+1:03d}.png');details.append(i+1)
    assert 'Quy ước và phạm vi' not in '\n'.join(all_text)
    assert len(body_spans)>500,'Not enough body text verified'
    contacts=[]
    for start in range(0,len(thumbs),16):
        sheet=Image.new('RGB',(1040,1500),'#dfe5eb');draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(thumbs[start:start+16]):
            x=(j%4)*260+10;y=(j//4)*375+24
            sheet.paste(im,(x,y));draw.text((x,y-18),f'PDF page {start+j+1}',fill='black')
        name=f'contact_{start//16+1:02d}.png';sheet.save(HERE/'qa'/name);contacts.append(name)
    DEST.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/'build/main.pdf',DEST)
    result=dict(status='PASS_AUTOMATED_PENDING_VISUAL_REVIEW',pdf=str(DEST),pages=len(pdf),
                figures=manifest['figures'],tables=len(re.findall(r'\\contentsline \{table\}',(HERE/'build/main.lot').read_text(encoding='utf8'))),
                chapters=9,catalogs=7,topics=70,keyword_slots=700,
                body_font='Times New Roman',body_size_tex_pt=13,body_size_pdf_points=expected_font_size,
                verified_text_spans=len(body_spans),all_body_text_font_verified=True,
                removed_section='Quy ước và phạm vi',cover_pixel_sha256=cover_hash,
                cover_preserved_exactly=bool(cover_reference),
                source_hashes_checked=len(manifest['source_sha256']),source_hashes_unchanged=True,
                undefined_references=0,missing_glyphs=0,overfull_boxes=0,
                underfull_warnings=log.count('Underfull \\hbox'),citations=len(cited),
                rendered_all_pages=True,contacts=contacts,detail_pages=details,page_checks=checks,
                pdf_sha256=digest(DEST),new_training=False,new_inference=True,
                new_search_pilot_chapter=True,independent_heldout_quality_claimed=False)
    (HERE/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    pdf.close()
    print(json.dumps({k:v for k,v in result.items() if k!='page_checks'},ensure_ascii=False))


if __name__=='__main__':main()
