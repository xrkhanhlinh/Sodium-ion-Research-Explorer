"""Validate final PDF, reviewed page images, Silver pilot and source preservation."""
from pathlib import Path
import hashlib
import json
import re
import sys
import subprocess
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/'outputs/research_revision_20261025_v1'
OUT=REV/'report_v2'
RUNTIME=Path('C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')


def main():
    # PDF libraries belong to the bundled artifact runtime.
    if Path(sys.executable).resolve()!=RUNTIME.resolve():
        raise SystemExit(subprocess.call([str(RUNTIME),str(Path(__file__).resolve())],cwd=ROOT))
    import pdfplumber
    from pypdf import PdfReader
    meta=json.loads((OUT/'build_summary.json').read_text(encoding='utf8'))
    pdf=Path(meta['pdf']);markdown=Path(meta['markdown'])
    reader=PdfReader(pdf);checks={}
    checks['page_count_22']=len(reader.pages)==22
    review_path=OUT/'visual_review.json'
    if not review_path.exists():raise ValueError('PNG inspection record missing; inspect every rendered page before accepting report')
    visual=json.loads(review_path.read_text(encoding='utf8'))
    checks['visual_review_covers_every_page']=visual['reviewed_pages']==list(range(1,len(reader.pages)+1)) and visual['status']=='PASS'
    checks['visual_review_matches_pdf']=visual['pdf_sha256']==hashlib.sha256(pdf.read_bytes()).hexdigest()
    texts=[page.extract_text() or '' for page in reader.pages]
    text='\n'.join(texts)
    checks['main_outputs_and_failure_results_present']=all(value in text for value in ['0,991808','0,850694','Na3V2(PO4)3','Silver','102 PASS'])
    checks['no_empty_pages']=all(len(t)>300 for t in texts)
    checks['no_replacement_glyphs']='\ufffd' not in text and '\u25a0' not in text and '\x00' not in text
    checks['all_sections_present']=all(title in re.sub(r'\s+',' ',text) for title in meta['sections'])
    bounds=[];footers=[]
    with pdfplumber.open(pdf) as doc:
        for i,page in enumerate(doc.pages,1):
            footer=(page.crop((page.width-70,page.height-40,page.width,page.height-10)).extract_text() or '').strip()
            footers.append({'page':i,'printed_number':footer})
            for char in page.chars:
                if char['x0']<40 or char['x1']>page.width-40 or char['top']<34 or char['bottom']>page.height-13:
                    bounds.append({'page':i,'text':char['text'],'x0':char['x0'],'x1':char['x1'],'top':char['top']})
    checks['printed_page_numbers_match_physical_pages']=all(item['printed_number']==str(item['page']) for item in footers)
    checks['text_inside_margins']=not bounds
    source=json.loads((OUT/'source_manifest.json').read_text(encoding='utf8'))
    # The report records the pipeline state at build time. That live state must
    # change while subsequent QA stages run; scientific inputs remain immutable.
    mutable_state='outputs/research_revision_20261025_v1/execution/pipeline_state.json'
    scientific_sources=[item for item in source if item['path']!=mutable_state]
    checks['scientific_source_manifest_unchanged']=all(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'] for item in scientific_sources)
    live_state=json.loads((ROOT/mutable_state).read_text(encoding='utf8'))
    checks['upstream_pipeline_stages_completed']=all(stage['status']=='completed' for name,stage in live_state['stages'].items() if name!='report_visual_qa')
    images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',markdown.read_text(encoding='utf8'))
    checks['all_markdown_images_exist']=all((markdown.parent/path).resolve().is_file() for path in images)
    pilot=pd.read_csv(ROOT/'outputs/ai_silver_labels_20261026_v1/batch_0001_labels_20.csv',keep_default_na=False,dtype=str)
    pool=pd.read_csv(ROOT/'outputs/batterybert_full_document_20261024_v1/annotation_pool.csv',keep_default_na=False,dtype=str).head(20)
    checks['silver_pilot_exact_twenty_ordered_pairs']=len(pilot)==20 and list(zip(pilot.query_id,pilot.doc_id))==list(zip(pool.query_id,pool.doc_id)) and not pilot.duplicated(['query_id','doc_id']).any()
    checks['silver_pilot_every_quote_is_source_evidence']=all(all(piece in (raw.title+'\n'+raw.abstract) for piece in judged.evidence.split(' | ')) for raw,judged in zip(pool.itertuples(),pilot.itertuples()))
    protected=json.loads((REV/'protected_artifacts_before.json').read_text(encoding='utf8'))
    checks['original_artifacts_unchanged']=all(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'] for item in protected)
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'visual_review':visual,
        'page_count':len(reader.pages),'footer_numbers':footers,'boundary_findings':bounds,
        'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pdf_bytes':pdf.stat().st_size,
        'mutable_source_exemption':{'path':mutable_state,'reason':'Live execution log advances after report creation; all scientific source hashes must still match.'}}
    # Preserve the first failed QA attempt; the final receipt is separate.
    destination=OUT/'validation_final.json'
    if destination.exists() and json.loads(destination.read_text(encoding='utf8'))!=result:raise ValueError('Existing validation differs; preserve prior record')
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':result['status'],'checks':checks,'pages':len(reader.pages),'boundary_count':len(bounds)},ensure_ascii=True,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)


if __name__=='__main__':main()
