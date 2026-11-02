"""Extract evidence checks and rasterize every page for visual PDF review."""
from pathlib import Path
import json
import hashlib
import re
import fitz
import pandas as pd
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
NAME='BAO_CAO_PIPELINE_BATTERYBERT_FULLDOC_20261024'
PDF=ROOT/'output/pdf'/(NAME+'.pdf')
RUN=ROOT/'outputs/batterybert_full_document_20261024_v1'
QA=ROOT/'tmp/pdfs/batterybert_full_document_20261024'


def run():
    QA.mkdir(parents=True,exist_ok=True);pdf=fitz.open(PDF)
    text='\n'.join(page.get_text() for page in pdf)
    deploy=json.loads((RUN/'deployment.json').read_text())
    active=RUN/'topics' if deploy['activate_topics'] else ROOT/'outputs/batterybert_optimization_20261023_v1'
    keys=pd.read_csv(active/'topic_keywords.csv')
    selection=json.loads((RUN/'selection.json').read_text())
    expected=['13.702','10.921','1.361','1.368','3.420.215','26.147',
        'PENDING_HUMAN_ANNOTATION',f"{selection['best_score']:.6f}".replace('.',','),'1.673']
    missing=[value for value in expected if value not in text]
    missing_keys=[word for word in keys.token if word not in text]
    bounds=[];headings=[];inventory=[]
    for i,page in enumerate(pdf):
        first=page.get_text().splitlines()[0]
        if re.match(r'^\d{2} / ',first):headings.append(first)
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0<35 or x1>page.rect.width-35 or y0<12 or y1>page.rect.height-12:
                        bounds.append(dict(page=i+1,text=span['text'],bbox=span['bbox']))
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(QA/f'page_{i+1:02d}.png')
        inventory.append(dict(page=i+1,first=first,characters=len(page.get_text())))
    for start in range(0,len(pdf),4):
        sheet=Image.new('RGB',(1120,1640),'#dbe3e8');draw=ImageDraw.Draw(sheet)
        for i in range(start,min(start+4,len(pdf))):
            im=Image.open(QA/f'page_{i+1:02d}.png');im.thumbnail((540,775))
            x=(i-start)%2*560+10;y=(i-start)//2*820+28
            sheet.paste(im,(x,y));draw.text((x,y-20),f'PAGE {i+1}',fill='black')
        sheet.save(QA/f'contact_{start//4+1}.png')
    manifest=json.loads((PDF.parent/(NAME+'_evidence.json')).read_text(encoding='utf8'))
    changed=[name for name,value in manifest['sources'].items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value['sha256']]
    result=dict(status='PASS',pages=len(pdf),numbered_sections=len(headings),outline_entries=len(pdf.get_toc()),
        expected_metrics_missing=missing,keywords_missing=missing_keys,bounds_violations=bounds,
        replacement_glyphs='\ufffd' in text,source_files_changed=changed,source_files=len(manifest['sources']),
        pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),inventory=inventory,
        visual_review='Rendered every page; contact sheets await final human-agent visual inspection')
    assert len(headings)==12 and len(pdf.get_toc())==13
    assert not missing and not missing_keys and not bounds and not changed and '\ufffd' not in text
    (PDF.parent/(NAME+'_validation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'}),flush=True)


if __name__=='__main__':run()
