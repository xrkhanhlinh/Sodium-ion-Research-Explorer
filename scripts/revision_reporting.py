"""Evidence comparison, semantic annotations and delivery for the revision."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.resumable_pipeline import atomic_json,sha256
REV=ROOT/'outputs/research_revision_20261025_v1'
BASE=ROOT/'outputs/batterybert_full_document_20261024_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Do not overwrite completed report input: '+str(path))
    else:atomic_json(path,value)


def immutable_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=pd.read_csv(path,keep_default_na=False)
        # A committed artifact is validated by its manifest. No replacement.
        if old.columns.tolist()!=frame.columns.tolist() or len(old)!=len(frame):raise ValueError('Existing table schema changed')
        return
    pending=path.with_suffix('.pending.csv');frame.to_csv(pending,index=False,encoding='utf-8-sig');pending.replace(path)


def commit_manifest(folder,paths,extra=None):
    folder=Path(folder)
    value={'artifacts':[{'path':str(Path(path).relative_to(folder)).replace('\\','/'),'sha256':sha256(path),'bytes':Path(path).stat().st_size} for path in paths],
       'scope':'Saved original artifacts and frozen revision experiments; no fabricated human labels',**(extra or {})}
    immutable_json(folder/'artifact_manifest.json',value)


def semantic_review():
    folder=REV/'review';folder.mkdir(parents=True,exist_ok=True)
    representatives=pd.read_csv(REV/'final_topics/representative_documents.csv')
    keys=pd.read_csv(REV/'final_topics/topic_keywords.csv')
    names=[
      'Anode oxide titan và titanate',
      'Prussian blue và vật liệu điện cực chứa sắt',
      'Cathode oxide lớp và biến đổi cấu trúc pha',
      'Điện ly polymer và giao diện trong các hệ pin natri',
      'Cathode phosphate vanadium và NASICON',
      'Anode chứa Sn và Sb cùng composite chuyển hóa',
      'Cấu trúc cell bộ góp dòng và thiết bị pin',
      'Anode MoS2 và vật liệu hai chiều graphene MXene',
      'Hard carbon và tiền chất sinh khối',
      'Sulfide selenide kim loại chuyển tiếp và composite carbon']
    evidence=[
      'anatase/bronze TiO2 nanocrystal wall',
      'Fe-based Prussian blue material (FePB)',
      'O3-type layered sodium transition metal oxides',
      'sodium-metal battery enabled by a nonaqueous polyelectrolyte solution',
      'Off-stoichiometric Na3V2-x(PO4)3/C',
      'Sb2S3/SnO(2)bundle-like nanocomposite',
      'current collecting element includes a housing',
      'nitrogen-doped MXene substrates (MXene-MoS2)',
      'biomass-derived hard carbon made from corn straw piths',
      'Ni3Se4@CoSe2@C/CNTs']
    notes=[
      'Có từ chỉ quy trình như solution và từ rộng anode; không suy mọi thành viên đều là TiO2.',
      'Có thể chồng lấn các vật liệu chứa sắt khác; từ iron không tự xác nhận Prussian blue.',
      'Tên nhóm phản ánh oxide lớp, không coi mọi phase hoặc metal là từ nhiễu.',
      'Đại diện có sodium-metal, khác sodium-ion thuần; tên nhóm phải phản ánh phạm vi natri rộng và cần rà soát nguồn.',
      'Công thức NVP được bảo toàn; không tự diễn giải mọi phosphorus hoặc phosphate là NVP.',
      'Đại diện có sulfide/oxide Sn/Sb, nên tên nhóm rộng hơn chỉ hợp kim.',
      'Đây là nội dung phần cứng thật trong corpus và thường đa hệ pin; không gọi là chủ đề vật liệu điện cực thuần.',
      'ID7 gần như đổi nội dung so với catalog cũ; không giữ nhãn pha/vacancy cũ chỉ vì cùng ID.',
      'Biomass và hard carbon có căn cứ; carbonize/carbonization có thể là alias, chưa tự động loại khi thí nghiệm không đạt.',
      'Nhóm chứa nhiều chalcogenide và composite carbon, không đặt tên như một hợp chất duy nhất.']
    rows=[]
    for topic,(name,quote,note) in enumerate(zip(names,evidence,notes)):
        central=representatives[representatives.topic_id.eq(topic)&representatives.sample_role.eq('central')].iloc[0]
        if quote not in str(central.title)+'\n'+str(central.abstract):raise ValueError('Semantic annotation quote mismatch topic '+str(topic))
        rows.append({'topic_id':topic,'topic_label':name,'label_status':'AI_INTERPRETATION_NOT_EXPERT_VERIFIED',
           'evidence_doc_id':central.doc_id,'evidence':quote,'reading_scope':'Full Title and Abstract of one central representative plus all10 topic keywords',
           'reviewer_type':'AI_ASSISTANT_NOT_EXPERT_GROUND_TRUTH','note':note})
    frame=pd.DataFrame(rows);immutable_csv(folder/'topic_labels.csv',frame)
    immutable_json(folder/'summary.json',{'status':'completed','topics_reviewed':10,'keyword_slots_read':len(keys),
      'complete_central_abstracts_read':10,'all_representatives_expert_verified':False,'human_ground_truth':False,
      'not_a_topic_accuracy_estimate':True,'membership_not_modified':True,
      'source_keywords_sha256':sha256(REV/'final_topics/topic_keywords.csv'),
      'source_representatives_sha256':sha256(REV/'final_topics/representative_documents.csv')})


def comparison():
    folder=REV/'comparison';folder.mkdir(parents=True,exist_ok=True)
    geo=REV/'geometry/evidence/geometry';rows=[]
    for name in ['baseline_source_n10','source_n50','source_whiten32_n50','source_whiten64_n50']:
        metric=read(geo/name/'metrics.json')
        rows.append({'model':name,'split':'validation','stability_ari':metric['stability_ari'],'document_npmi':metric['document_npmi'],
         'source_cramers_v':metric['source_cramers_v'],'diversity':metric['keyword_diversity'],
         'largest_share':metric['largest_topic_share'],'top2_share':metric['top2_topic_share'],
         'cosine_silhouette':metric['cosine_silhouette'],'passed_development':metric['passes'],
         'protocol':'Original gap-preserving stream; seeds42/43/47; identical train-fit keyword recipe'})
    old=read(REV/'geometry/evidence/old_comparator/metrics.json')
    rows.append({'model':'old_optimized_23oct','split':'validation','stability_ari':old['stability_ari'],'document_npmi':old['document_npmi'],
         'source_cramers_v':old['source_cramers_v'],'diversity':old['keyword_diversity'],
         'largest_share':old['largest_topic_share'],'top2_share':old['top2_topic_share'],
         'cosine_silhouette':old['cosine_silhouette'],'passed_development':'historical descriptive comparator',
         'protocol':'Same seeds and keyword recipe, historical encoder'})
    for name in ['baseline_source_n10','source_n50']:
        metric=read(geo/name/'gate_metrics.json')
        rows.append({'model':name,'split':'test_gate','stability_ari':metric['stability_ari'],'document_npmi':metric['document_npmi'],
         'source_cramers_v':metric['source_cramers_v'],'diversity':metric['keyword_diversity'],
         'largest_share':metric['largest_topic_share'],'top2_share':metric['top2_topic_share'],
         'cosine_silhouette':metric['cosine_silhouette'],'passed_development':True,
         'protocol':'Original stream; coordinator independent seed20261029; reused historical test'})
    immutable_csv(folder/'geometry_comparison.csv',pd.DataFrame(rows))
    keyroot=REV/'keywords/experiment_outputs';keyrows=[]
    for name in ['baseline_dev','contrastive_df','title_evidence','contrastive_phrases']:
        value=read(keyroot/name/'dev_metrics.json')
        keyrows.append({'candidate':name,**{k:value[k] for k in ['macro_f1','document_npmi','macro_own_topic_coverage','diversity','procedural_unigram_slots','alias_redundant_slots']},'decision':'baseline retained' if name=='baseline_dev' else 'rejected'})
    for value in read(keyroot/'cycle2/comparison.json'):
        keyrows.append({k:v for k,v in value.items() if k in keyrows[0] or k=='candidate'}|{'decision':'rejected'})
    immutable_csv(folder/'keyword_experiments.csv',pd.DataFrame(keyrows))
    search=[]
    for split,names in [('dev',['baseline','abstract_rerank30','abstract_rerank50']),('gate',['baseline','abstract_rerank30'])]:
        for name in names:
            value=read(REV/'search/experiment_outputs'/split/(name+'_summary.json'))
            search.append({k:value[k] for k in ['method','queries','group_mrr_at_10','success_at_1','success_at_5','success_at_10','median_rank_seconds']}|{'split':split,'timing_scope':'Post-retrieval only; baseline cached time0 is NOT end-to-end speed'})
    immutable_csv(folder/'search_proxy_comparison.csv',pd.DataFrame(search))
    immutable_csv(folder/'historical_topic_comparison.csv',pd.read_csv(ROOT/'outputs/final_system/topic_comparison.csv'))
    immutable_csv(folder/'canonical_keyword_comparison.csv',pd.read_csv(REV/'final_topics/keyword_comparison.csv'))
    audit=pd.read_csv(REV/'final_topics/keyword_audit.csv')
    immutable_csv(folder/'keyword_flags_per_topic.csv',audit.groupby('topic_id')[[c for c in audit if c in ['weak_procedural_or_structural_unigram','broad_domain_unigram','alias_redundancy','unsupported_formula_preserved']]].sum().reset_index())
    immutable_json(folder/'conclusions.json',{'geometry_accepted':read(REV/'decisions/geometry.json')['accepted'],
       'reranker_accepted':read(REV/'decisions/search_gate.json')['accepted'],'keyword_suppression_candidates_accepted':0,
       'same_reference_keyword_npmi':read(REV/'final_topics/final_metrics.json')['heldout']['document_npmi'],
       'original_human_relevance':read(BASE/'evaluation_status.json'),
       'new_ai_silver_pilot':read(ROOT/'outputs/ai_silver_labels_20261026_v1/batch_0001_summary.json'),
       'metric_protocols_are_not_interchangeable':True})
    commit_manifest(folder,sorted(path for path in folder.iterdir() if path.is_file() and path.name!='artifact_manifest.json'))


def visualizations():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=REV/'visualizations';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    navy,teal,orange='#2c5069','#198477','#bd7b39'
    def save(fig,name):fig.savefig(folder/(name+'.png'),dpi=200,bbox_inches='tight',facecolor='white');plt.close(fig)
    geom=pd.read_csv(REV/'comparison/geometry_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(11.2,3.9))
    for ax,metric,title in zip(axes,['stability_ari','document_npmi'],['Độ ổn định ARI','NPMI từ khóa cùng protocol gốc']):
        for pos,(split,label) in enumerate([('validation','Dev'),('test_gate','Gate')]):
            for offset,(model,color) in enumerate([('baseline_source_n10',navy),('source_n50',teal)]):
                value=float(geom[(geom.model.eq(model))&geom.split.eq(split)][metric].iloc[0])
                ax.bar(pos+offset*.3,value,.28,color=color,label=model if pos==0 else None)
                ax.text(pos+offset*.3,value+.012,f'{value:.3f}',ha='center',fontsize=9)
        ax.set(title=title,xticks=[.15,1.15],xticklabels=['Dev','Gate'],ylim=(0,1.08 if metric=='stability_ari' else .14));ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();save(fig,'geometry_before_after')
    key=pd.read_csv(REV/'comparison/keyword_experiments.csv')
    fig,ax=plt.subplots(figsize=(10.5,4.4))
    points=ax.scatter(key.macro_own_topic_coverage,key.macro_f1,c=key.document_npmi,cmap='viridis',s=95)
    for _,row in key.iterrows():ax.annotate(row.candidate,(row.macro_own_topic_coverage,row.macro_f1),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Tỷ lệ tài liệu có keyword của chính topic',ylabel='Macro F1 dự đoán cluster từ keywords',title='Danh sách ít từ chung chưa chắc đại diện topic tốt hơn',xlim=(.4,1.04),ylim=(.49,.61));fig.colorbar(points,ax=ax,label='Document NPMI')
    fig.tight_layout();save(fig,'keyword_failed_hypotheses')
    dist=pd.read_csv(REV/'final_topics/topic_source_distribution.csv')
    if 'documents' in dist:
        dist=dist.pivot(index='topic_id',columns='source',values='documents').fillna(0).reset_index()
    fig,ax=plt.subplots(figsize=(10.5,4))
    ax.bar(dist.topic_id,dist.Patent,color=orange,label='Patent');ax.bar(dist.topic_id,dist.Publication,bottom=dist.Patent,color=teal,label='Publication')
    ax.set(xticks=range(10),xlabel='ID topic đã căn chỉnh trên train',ylabel='Tài liệu',title='Catalog mới sau KMeans50 và bảo toàn công thức');ax.legend(frameon=False,ncol=2)
    fig.tight_layout();save(fig,'revised_topic_sources')
    keys=pd.read_csv(REV/'final_topics/topic_keywords.csv')
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.0))
    for ax,topic,title in zip(axes.flat,[4,5,7,8],['Phosphate và NASICON','Vật liệu chứa Sn và Sb','MoS2 và vật liệu hai chiều','Hard carbon và biomass']):
        part=keys[keys.topic_id.eq(topic)].sort_values('rank').iloc[::-1]
        ax.barh(part.token,part.weight,color=teal);ax.set(title=title,xlabel='Trọng số TF × IDF × exclusivity');ax.tick_params(axis='y',labelsize=9)
    fig.tight_layout(w_pad=2.6,h_pad=1.7);save(fig,'revised_top10_keywords')
    canonical=pd.read_csv(REV/'comparison/canonical_keyword_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.7,3.7))
    for i,split in enumerate(['validation','test']):
        part=canonical[canonical.split.eq(split)]
        for j,(name,color) in enumerate([('raw_statistical',navy),('canonical_v2',teal)]):
            value=float(part[part.keywords.eq(name)].document_npmi.iloc[0]);axes[0].bar(i+j*.3,value,.28,color=color,label=name if i==0 else None)
    axes[0].set(title='Cùng canonical reference',xticks=[.15,1.15],xticklabels=['Validation','Test'],ylabel='Document NPMI');axes[0].legend(frameon=False,fontsize=8)
    coherence=pd.read_csv(REV/'final_topics/common_coherence.csv');axes[1].bar(['Raw keywords','Canonical v2'],coherence.c_v,color=[navy,teal]);axes[1].set(title='C_v trên cùng corpus tham chiếu',ylim=(0,.7))
    for i,v in enumerate(coherence.c_v):axes[1].text(i,v+.01,f'{v:.4f}',ha='center')
    fig.tight_layout();save(fig,'canonical_correctness_effect')
    search=pd.read_csv(REV/'comparison/search_proxy_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    for i,split in enumerate(['dev','gate']):
        part=search[search.split.eq(split)&search.method.isin(['baseline','abstract_rerank30'])]
        for j,(name,color) in enumerate([('baseline',navy),('abstract_rerank30',teal)]):
            value=float(part[part.method.eq(name)].group_mrr_at_10.iloc[0]);axes[0].bar(i+j*.3,value,.28,color=color,label=name if i==0 else None);axes[0].text(i+j*.3,value+.015,f'{value:.3f}',ha='center',fontsize=9)
    axes[0].set(title='Reranker tăng dev nhưng giảm gate',xticks=[.15,1.15],xticklabels=['Dev16 query','Gate16 query'],ylim=(0,1.08),ylabel='Proxy Group MRR@10');axes[0].legend(frameon=False,fontsize=8)
    extra=search[search.split.eq('dev')&search.method.ne('baseline')]
    axes[1].bar(['Top30','Top50'],extra.median_rank_seconds,color=[teal,orange]);axes[1].set(title='Chi phí rerank bổ sung trên dev',ylabel='Median giây; không gồm encode query')
    fig.tight_layout();save(fig,'search_gate_rejection')
    pilot=read(ROOT/'outputs/ai_silver_labels_20261026_v1/batch_0001_summary.json')
    fig,ax=plt.subplots(figsize=(8.5,3.5));labels=['Label0','Label1','Label2','Trống'];counts=[pilot['label_counts'][str(i)] for i in [0,1,2]]+[pilot['blank_labels']]
    ax.bar(labels,counts,color=[orange,navy,teal,'#9aa7b0']);ax.set(title='Silver pilot20 cặp Q01 theo thứ tự gốc',ylabel='Số cặp',ylim=(0,13))
    for i,v in enumerate(counts):ax.text(i,v+.2,str(v),ha='center')
    fig.tight_layout();save(fig,'silver_pilot20')
    commit_manifest(folder,sorted(folder.glob('*.png')),{'plots':7,'new_model_fits':0})


def verification():
    folder=REV/'verification';folder.mkdir(parents=True,exist_ok=True)
    xml=folder/'affected_tests.xml';inputs=['src/resumable_pipeline.py','src/scientific_keyword_stream_v2.py','src/revised_topics.py','src/final_search.py','src/final_dashboard.py',
       'tests/test_resumable_pipeline.py','tests/test_scientific_keyword_stream_v2.py','tests/test_revised_topics.py']
    identity={name:sha256(ROOT/name) for name in inputs};marker=folder/'tests_receipt.json'
    if marker.exists():
        previous=read(marker)
        if previous['inputs']!=identity or previous['xml_sha256']!=sha256(xml):raise ValueError('Verified code/tests changed; use a new verification stage')
    else:
        command=[str(ROOT/'.venv/Scripts/python.exe'),'-m','pytest','tests/test_resumable_pipeline.py','tests/test_scientific_keyword_stream_v2.py','tests/test_revised_topics.py',
             'tests/test_scientific_keyword_stream.py','tests/test_full_document_retrieval.py','tests/test_full_document_evaluation.py','tests/test_research_evaluation.py','-q','--junitxml',str(xml)]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
        (folder/'affected_tests.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
        if result.returncode:raise RuntimeError('Affected tests failed; inspect '+str(folder/'affected_tests.log'))
        immutable_json(marker,{'inputs':identity,'xml_sha256':sha256(xml)})
    suites=ET.parse(xml).getroot().findall('.//testsuite')
    counts={field:sum(int(suite.get(field,0)) for suite in suites) for field in ['tests','failures','errors','skipped']}
    protected=read(REV/'protected_artifacts_before.json')
    changed=[item['path'] for item in protected if sha256(ROOT/item['path'])!=item['sha256']]
    if changed:raise ValueError('Historical artifact changed: '+str(changed))
    pilot=read(ROOT/'outputs/ai_silver_labels_20261026_v1/batch_0001_summary.json')
    value={'status':'PASS','tests':counts,'preservation':{'checked':len(protected),'changed':changed},
       'human_labels_accepted':read(BASE/'evaluation_status.json')['accepted_human_labels'],
       'ai_silver_labels':pilot['processed'],'ai_labels_not_imported_as_human':True,
       'encoder_forward_repeated':0,'existing_embeddings_recomputed':0,'scientific_accuracy_proven':False}
    if counts['failures'] or counts['errors'] or counts['skipped']:raise ValueError('Required verification not complete')
    immutable_json(folder/'summary.json',value)


def deployment():
    if read(REV/'verification/summary.json')['status']!='PASS':raise ValueError('Code verification required')
    geom=read(REV/'decisions/geometry.json');search=read(REV/'decisions/search_gate.json')
    value={'activate_revised_topics':geom['accepted'],'topic_geometry_gate_accepted':geom['accepted'],
       'activate_search_rerank':False,'search_rerank_gate_accepted':search['accepted'],'search_rerank_depth':30,
       'search_default':'BATTERYBERT_FULLDOC_HYBRID','search_reranker_reason':'Rejected on coordinator diagnostic gate; remains isolated experiment',
       'topic_catalog_manifest_sha256':sha256(REV/'final_topics/artifact_manifest.json'),
       'topic_label_review_sha256':sha256(REV/'review/topic_labels.csv'),
       'reviewer_type':'AI_NOT_EXPERT_GROUND_TRUTH','independent_human_relevance_validated':False,
       'original_model_weights_and_vectors_unchanged':True,'original_qdrant_index_rebuilt':False,
       'topic_filter_policy':'Updated assignments converted into allowed doc_ids before search; no stale topic-payload filter',
       'scientific_keyword_suppression_promoted':False}
    immutable_json(REV/'deployment.json',value)


def routing_smoke():
    from src import revised_topics
    from src.final_search import FinalSearchEngine
    if not revised_topics.ready():raise ValueError('Accepted revised catalog not ready')
    engine=FinalSearchEngine()
    try:
        response=engine.search('MoS2 MXene sodium','TFIDF_COSINE',topic_id=7)
        if not response['results'] or any(result['topic_id']!=7 for result in response['results']):raise ValueError('New topic filter was not applied')
        labels=revised_topics.assignments(engine.ids).set_index('doc_id').topic_id
        if any(int(labels.loc[result['doc_id']])!=7 for result in response['results']):raise ValueError('Stale topic payload used for filtering')
        value={'status':'PASS','method':'TFIDF_COSINE','topic_filter':7,'results':len(response['results']),
          'catalog':revised_topics.NAME,'encoder_forward_calls':0,'topic_label':response['results'][0]['topic_label'],
          'all_returned_doc_ids_match_revised_catalog':True,'result_doc_ids':[r['doc_id'] for r in response['results']]}
    finally:engine.close()
    immutable_json(REV/'verification/routing_smoke.json',value)


def report():
    runtime=Path('C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
    result=subprocess.run([str(runtime),str(ROOT/'scripts/revision_report_document.py')],cwd=ROOT)
    if result.returncode:raise RuntimeError('Report builder failed')


def report_v2():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=REV/'visualizations_v2';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    key=pd.read_csv(REV/'comparison/keyword_experiments.csv')
    destination=folder/'keyword_failed_hypotheses.png'
    if not destination.exists():
        fig,ax=plt.subplots(figsize=(10.5,4.4));points=ax.scatter(key.macro_own_topic_coverage,key.macro_f1,c=key.document_npmi,cmap='viridis',s=95)
        offsets={'baseline_dev':(-125,-23),'conservative_tf1':(-125,-5),'contrastive_df':(12,-25),'title_evidence':(15,16),'contrastive_phrases':(12,-19),'conservative_sqrt_tf':(-105,15)}
        for _,row in key.iterrows():
            ax.annotate(row.candidate,(row.macro_own_topic_coverage,row.macro_f1),xytext=offsets[row.candidate],textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':'#89979e','lw':.7})
        ax.set(xlabel='Tỷ lệ tài liệu có keyword của chính topic',ylabel='Macro F1 dự đoán cluster từ keywords',title='Ít từ chung chưa chắc đại diện topic tốt hơn',xlim=(.4,1.04),ylim=(.49,.61));fig.colorbar(points,ax=ax,label='Document NPMI');fig.tight_layout();fig.savefig(destination,dpi=200,bbox_inches='tight');plt.close(fig)
    destination=folder/'revised_topic_sources.png'
    if not destination.exists():
        data=pd.read_csv(REV/'final_topics/topic_source_distribution.csv')
        if 'documents' in data:data=data.pivot(index='topic_id',columns='source',values='documents').fillna(0).reset_index()
        fig,ax=plt.subplots(figsize=(10.5,4));ax.bar(data.topic_id,data.Patent,color='#bd7b39',label='Patent');ax.bar(data.topic_id,data.Publication,bottom=data.Patent,color='#198477',label='Publication')
        ax.set(xticks=range(10),xlabel='ID topic đã căn chỉnh trên train',ylabel='Tài liệu',title='Catalog KMeans K=10 với n_init=50 và bảo toàn công thức');ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(destination,dpi=200,bbox_inches='tight');plt.close(fig)
    commit_manifest(folder,sorted(folder.glob('*.png')),{'reason':'Separate figure revision fixes label collisions and clarifies K10 versus50 initializations; original plots unchanged'})
    runtime=Path('C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
    result=subprocess.run([str(runtime),str(ROOT/'scripts/revision_report_document.py'),'--revision','v2'],cwd=ROOT)
    if result.returncode:raise RuntimeError('Report layout revision failed')


def run_job(name):globals()[name]()
