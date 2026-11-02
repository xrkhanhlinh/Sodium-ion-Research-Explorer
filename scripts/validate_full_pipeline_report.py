"""Text/evidence checks and raster QA pages for the generated print report."""
from pathlib import Path
import json
import re
import hashlib
import fitz
import pandas as pd
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT/'output/pdf/BAO_CAO_TOAN_BO_PIPELINE_20261024.pdf'
QA=ROOT/'tmp/pdfs/full_pipeline_20261024'


def run():
    QA.mkdir(parents=True,exist_ok=True)
    pdf=fitz.open(PDF)
    text='\n'.join(page.get_text() for page in pdf)
    expected=['13.702','10.921','1.361','1.368','0,5601','0,7436','0,9245','0,3375',
        '0,2975','27,49%','92,76%','PENDING_HUMAN_ANNOTATION','GEMINI_LLM_JUDGE']
    missing=[t for t in expected if t not in text]
    keys=pd.read_csv(ROOT/'outputs/batterybert_optimization_20261023_v1/topic_keywords.csv')
    missing_keys=[str(w) for w in keys.token if str(w) not in text]
    headings=[];bounds=[];invent=[]
    for i,page in enumerate(pdf):
        content=page.get_text(); first=content.splitlines()[0]
        if re.match(r'^\d{2} / ',first):headings.append(first)
        # Text spans must stay on the physical A4 page and above its footer.
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0<35 or x1>page.rect.width-35 or y0<12 or y1>page.rect.height-12:
                        bounds.append(dict(page=i+1,text=span['text'],bbox=span['bbox']))
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(QA/f'page_{i+1:02d}.png')
        invent.append(dict(page=i+1,chars=len(content),first=first))
    for start in range(0,len(pdf),4):
        sheet=Image.new('RGB',(1120,1640),'#dbe3e8');draw=ImageDraw.Draw(sheet)
        for j in range(start,min(start+4,len(pdf))):
            im=Image.open(QA/f'page_{j+1:02d}.png');im.thumbnail((540,775))
            x=(j-start)%2*560+10;y=(j-start)//2*820+28
            sheet.paste(im,(x,y));draw.text((x,y-20),f'PAGE {j+1}',fill='black')
        sheet.save(QA/f'contact_{start//4+1}.png')
    manifest=json.loads((PDF.parent/'evidence_manifest_20261024.json').read_text(encoding='utf8'))
    changed=[p for p,v in manifest['sources'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=v['sha256']]
    result=dict(status='PASS',pages=len(pdf),numbered_sections=len(headings),outline_entries=len(pdf.get_toc()),
        expected_metrics_missing=missing,keyword_slots=100,keywords_missing=missing_keys,
        replacement_glyphs='\ufffd' in text,bounds_violations=bounds,source_files=len(manifest['sources']),
        source_files_changed=changed,pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
        model_training_performed=False,model_tests_rerun_during_report_authoring=False,
        visual_review='Render produced for every page; final visual inspection recorded separately',inventory=invent)
    assert len(pdf)==18 and len(headings)==17 and len(pdf.get_toc())==18
    assert not missing and not missing_keys and not bounds and not changed and '\ufffd' not in text
    (PDF.parent/'validation_20261024.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'},ensure_ascii=True))


if __name__=='__main__':run()
