"""Validate the generated research artifact against its saved source outputs."""
from pathlib import Path
import hashlib
import json
import re
import sys
import pandas as pd
import pdfplumber
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'outputs/report_pipeline_20261025'
PDF=ROOT/'output/pdf/BAO_CAO_PIPELINE_SODIUM_ION_20261025.pdf'
MD=ROOT/'docs/BAO_CAO_PIPELINE_SODIUM_ION_20261025.md'
RUN=ROOT/'outputs/batterybert_full_document_20261024_v1'
checks={}
reader=PdfReader(PDF)
checks['page_count_31']=len(reader.pages)==31
checks['outline_count_31']=len(reader.outline)==31
text='\n'.join(page.extract_text() or '' for page in reader.pages)
checks['all_main_sections_present']=all(name in text for name in [
    'LDA','BERTopic','BatteryBERT','KMeans','Top 10','703','0,924704','0,887839','PENDING_HUMAN_ANNOTATION'])
checks['all_appendices_present']=all('Phụ lục '+c in text for c in 'ABCDEF')
checks['no_replacement_characters']='\ufffd' not in text and '\u25a0' not in text
checks['no_near_empty_pages']=all(len(p.extract_text() or '')>300 for p in reader.pages)
headings=json.loads((WORK/'build_summary.json').read_text(encoding='utf8'))['sections']
checks['section_page_mapping']=all(heading in (reader.pages[i+1].extract_text() or '') for i,heading in enumerate(headings))
boundary=[]
with pdfplumber.open(PDF) as document:
    for i,page in enumerate(document.pages,1):
        for char in page.chars:
            if char['x0']<40 or char['x1']>page.width-40 or char['top']<35 or char['bottom']>page.height-15:
                boundary.append({'page':i,'character':char['text'],'x0':char['x0'],'x1':char['x1'],'top':char['top']})
checks['text_inside_page_margins']=not boundary
markdown=MD.read_text(encoding='utf8')
images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',markdown)
checks['twelve_result_figures']=len(images)==12
checks['all_markdown_images_exist']=all((MD.parent/path).resolve().is_file() for path in images)
catalog=pd.read_csv(ROOT/'docs/TOP10_TOPICS_ALL_MODELS_20261024.csv')
checks['six_catalogs_sixty_topics']=catalog.model.nunique()==6 and len(catalog)==60 and catalog.groupby('model').size().eq(10).all()
checks['six_hundred_keyword_slots']=catalog.top10_keywords.str.split(';').map(len).eq(10).all()
outliers=[0,0,0,0,3120,109]
checks['catalog_document_totals']=all(int(group.documents.sum())+outlier==13702 for (_,group),outlier in zip(catalog.groupby('model',sort=False),outliers))
ranks=pd.read_csv(RUN/'rankings_top10.csv')
q=ranks[ranks.query_id.eq('Q01')&ranks.method.eq('BATTERYBERT_FULLDOC_HYBRID')].sort_values('rank')
checks['query_example_ten_ranked_results']=q['rank'].tolist()==list(range(1,11)) and q.rrf_score.is_monotonic_decreasing
checks['query_example_titles_present']=all(title in text.replace('\n',' ') for title in q.title.str.replace(r'\s+',' ',regex=True))
status=json.loads((RUN/'evaluation_status.json').read_text(encoding='utf8'))
checks['pending_labels_described_correctly']=status['accepted_human_labels']==0 and status['eligible_queries']==0 and status['pool_pairs']==703
manifest=json.loads((WORK/'source_manifest.json').read_text(encoding='utf8'))
checks['all_source_hashes_match']=all(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'] for item in manifest)
result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':{k:bool(v) for k,v in checks.items()},
        'boundary_findings':boundary,'pages':len(reader.pages),'pdf_bytes':PDF.stat().st_size,
        'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
        'visual_review':'All original 30 pages reviewed as contact sheets; detailed pages inspected. Final added contents and corrected learning-rate glyphs visually inspected.'}
(WORK/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=True,indent=2))
sys.exit(0 if result['status']=='PASS' else 1)
