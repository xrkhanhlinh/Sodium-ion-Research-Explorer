"""Render and inspect document geometry and preserved source identities."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

import fitz
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT/'evidence_manifest.json').read_text(encoding='utf-8'))
root = Path(manifest['source_root'])
changed = [p for p,h in manifest['sources'].items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
assert not changed, changed
log=(OUT/'main.log').read_text(encoding='utf-8',errors='replace')
bad=[x for x in ['Overfull \\hbox','Overfull \\vbox','Missing character:',
                 'Undefined control sequence','LaTeX Error:', 'undefined references'] if x in log]
assert not bad, bad
pdf=fitz.open(OUT/'main.pdf')
assert len(pdf)==manifest['main_slides']+manifest['appendix_slides']
qa=OUT/'qa';qa.mkdir(exist_ok=True)
outside=[];texts=[]
for i,p in enumerate(pdf):
    assert abs(p.rect.width/p.rect.height-16/9)<0.01
    text=p.get_text();texts.append(text)
    assert len(text.strip())>60, i+1
    for b in p.get_text('dict')['blocks']:
        for line in b.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if min(x0,y0)<-0.1 or x1>p.rect.width+0.5 or y1>p.rect.height+0.5:
                    outside.append((i+1,span['text'],span['bbox']))
    p.get_pixmap(matrix=fitz.Matrix(2.7,2.7),alpha=False).save(qa/f'slide_{i+1:03d}.png')
assert not outside,outside
combined='\n'.join(texts)
for literal in ['109.482.240','7.087.872','33.433','MAIN_CORPUS','6.596','RERANK_PRETRAINED']:
    assert literal in combined, ('Missing rendered content', literal)
inventory=json.loads((OUT/'slide_inventory.json').read_text(encoding='utf-8'))
for slide in inventory:
    if slide['title'].startswith(('LDA K20: topic','BERTopic baseline: topic','BERTopic chunked: topic')):
        expected=re.findall(r' & ([0-9.]+) & ',slide['body'])
        rendered=texts[slide['number']-1]
        assert all(n in rendered for n in expected), ('Missing table values',slide['number'],expected)
(qa/'extracted_text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
for group in range(math.ceil(len(pdf)/9)):
    sheet=Image.new('RGB',(1920,1185),'#dde4e9')
    draw=ImageDraw.Draw(sheet)
    for slot in range(9):
        idx=group*9+slot
        if idx>=len(pdf): break
        with Image.open(qa/f'slide_{idx+1:03d}.png') as im:
            im.thumbnail((628,354))
            x=(slot%3)*640+6;y=(slot//3)*395+25
            sheet.paste(im,(x,y))
            draw.text((x,y-18),str(idx+1),fill='#12304a')
    sheet.save(qa/f'contact_{group+1:02d}.png')
with (OUT/'topic_catalog.csv').open(encoding='utf-8-sig',newline='') as f:
    catalog=list(csv.DictReader(f))
counts={name:sum(r['model']==name and r['is_outlier']=='False' for r in catalog)
        for name in ['LDA K20','BERTopic baseline','BERTopic chunked']}
assert list(counts.values())==[20,92,84],counts
assert all(len(r['top10_keywords'].split('; '))==10 for r in catalog)
result=dict(pages=len(pdf),main_slides=manifest['main_slides'],appendix_slides=manifest['appendix_slides'],
            figures=manifest['figures'],source_hashes_unchanged=True,tex_errors_or_overfull=bad,
            out_of_page_spans=outside,topic_counts=counts,topic_catalog_rows=len(catalog),
            all_pages_rendered=True,pdf_sha256=hashlib.sha256((OUT/'main.pdf').read_bytes()).hexdigest(),
            scope='Document QA only. Historical model training, search benchmarks and project tests were not rerun.')
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
