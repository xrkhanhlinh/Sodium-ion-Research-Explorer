from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/ai_silver_labels_703_20261026_v1';E=OUT/'evaluation'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import pdfplumber
    from pypdf import PdfReader
    pdf=ROOT/'output/pdf/DANH_GIA_SEARCH_SILVER_703_20261026_v2.pdf'
    visual=json.loads((E/'pdf_qa/visual_review_v2.json').read_text(encoding='utf8'))
    reader=PdfReader(pdf);text='\n'.join(p.extract_text() for p in reader.pages);bounds=[];footers=[]
    with pdfplumber.open(pdf) as doc:
        for i,p in enumerate(doc.pages,1):
            footers.append((p.crop((p.width-65,p.height-35,p.width,p.height-10)).extract_text() or '').strip()==str(i))
            bounds.extend({'page':i,'text':c['text']} for c in p.chars if c['x0']<40 or c['x1']>p.width-40 or c['top']<35 or c['bottom']>p.height-12)
    manifest=json.loads((E/'report_manifest_v2.json').read_text(encoding='utf8'))
    checks={'six_pages':len(reader.pages)==6,'visual_review_all_pages':visual['reviewed_pages']==[1,2,3,4,5,6] and visual['status']=='PASS',
       'visual_pdf_hash_matches':visual['pdf_sha256']==h(pdf),'physical_footers':all(footers),'inside_margins':not bounds,
       'no_missing_glyphs':not any(s in text for s in ['\ufffd','\u25a0','\x00']),
       'actual_results_present':all(s in text for s in ['703/703','143','302','258','0,5400','0,7693','0,7767','Q08','Q20','70']),
       'source_hashes_unchanged':all(h(ROOT/x['path'])==x['sha256'] for x in manifest['artifacts'])}
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'page_count':6,'pdf_sha256':h(pdf),'bounds':bounds,
       'draft_v1':'Kept unchanged; colorbar overlap found on page4 and corrected in separate v2 figure/PDF.'}
    path=E/'pdf_validation_v2.json';payload=json.dumps(result,ensure_ascii=False,indent=2).encode('utf8')
    assert not path.exists() or path.read_bytes()==payload;path.write_bytes(payload)
    print(json.dumps(result,ensure_ascii=True));raise SystemExit(0 if result['status']=='PASS' else 1)
if __name__=='__main__':main()
