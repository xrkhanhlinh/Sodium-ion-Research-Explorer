"""Reproducible, non-promoting pilot experiments over a frozen corpus.

Run --phase historical first; --phase pilot creates a separate unlabelled pool.
No API calls, fabricated grades, training, parameter selection or Qdrant writes.
"""
from pathlib import Path
import argparse
import itertools
import json
import platform
import sys
import time
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import psutil
from src.final_system_artifacts import sha,read_json
from src.full_document_inference import OUT as FULL
from src.search_radar_core import VERSION,METRICS,parse_query,condition_evidence,weighted_rrf,evaluate_ranking,paired_bootstrap
from src.search_radar_retrieval import RadarRetriever,OUT,BRANCHES
from src.model_comparison_io import write_json

SILVER=ROOT/'outputs/ai_silver_labels_703_20261026_v1'
PILOT=[
 ('D01','dev','manganese_dissolution','materials','Manganese dissolution suppression in sodium manganese oxide cathodes'),
 ('D02','dev','fast_charge_plating','conditions','Sodium plating detection during fast charging of sodium-ion full cells'),
 ('D03','dev','dry_processing','processes','Solvent-free dry processing of sodium-ion electrodes with PTFE binders'),
 ('D04','dev','gas_evolution','equipment','Operando gas analysis of sodium-ion pouch cells'),
 ('D05','dev','self_discharge','multi-constraint','Self-discharge mechanisms of aqueous sodium-ion full cells at room temperature'),
 ('D06','dev','first_cycle_sodium','multi-constraint','Presodiation methods compensating first-cycle sodium loss in hard carbon full cells'),
 ('H01','heldout','transition_metal_migration','materials','Transition metal migration in oxygen-redox sodium layered oxide cathodes'),
 ('H02','heldout','humidity_storage','conditions','Humidity stability of sodium-ion cathode powders during air storage'),
 ('H03','heldout','laser_patterning','processes','Laser patterning of thick sodium-ion battery electrodes'),
 ('H04','heldout','acoustic_monitoring','equipment','Acoustic monitoring of swelling in sodium-ion pouch cells'),
 ('H05','heldout','separator_shrinkage','multi-constraint','Thermal shrinkage of ceramic-coated separators in sodium-ion batteries'),
 ('H06','heldout','calendar_aging','multi-constraint','Calendar aging of sodium-ion full cells at high state of charge and elevated temperature'),
]


def csv(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,encoding='utf-8-sig')


def protected():
    paths=[ROOT/'data/processed/main_corpus.parquet',ROOT/'data/stage3/split_assignments.parquet',
        SILVER/'silver_labels_703.csv',FULL/'document_embeddings.npy',FULL/'ordered_doc_ids.json',
        FULL/'model/model.safetensors',FULL/'accepted_human_labels.csv',FULL/'annotation_pool.csv',
        FULL/'comparison_rankings_top10.csv',FULL/'deployment.json',ROOT/'data/stage4/pretrained_embeddings.npy',
        ROOT/'outputs/final_system/tfidf/matrix.npz',ROOT/'models/bge_pretrained/model.safetensors']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def pilot_queries():
    result=pd.DataFrame(PILOT,columns=['query_id','split','family','category','query'])
    result['origin']='system-proposed; expert review pending';result['expert_verified']=False
    return result


def vector_cache(retriever,queries,folder):
    folder.mkdir(parents=True,exist_ok=True)
    identity=dict(queries=queries[['query_id','query']].to_dict('records'),
        bge_weights=sha(ROOT/'models/bge_pretrained/model.safetensors'),
        bge_tokenizer=sha(ROOT/'models/bge_pretrained/tokenizer.json'),
        full_weights=sha(FULL/'model/model.safetensors'),full_tokenizer=read_json(FULL/'embedding_success.json')['tokenizer_files'],
        inference_contract=sha(ROOT/'outputs/reports/stage3/inference_contract.json'),
        full_inference_source=sha(ROOT/'src/full_document_inference.py'),version=VERSION)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    vectors={};timings=[]
    for q in queries.itertuples():
        target=folder/(q.query_id+'.npz');marker=folder/(q.query_id+'.json')
        if target.exists() and marker.exists():
            info=read_json(marker)
            if info['fingerprint']!=fingerprint or info['sha256']!=sha(target):raise ValueError('Query vector cache mismatch')
            with np.load(target) as data:vectors[q.query_id]={k:np.array(data[k]) for k in ('bge','fulldoc')}
            timings.extend(info['timings']);continue
        # First pass may include model load; second is warmed query encoding.
        _,cold=retriever.encode(q.query);values,warm=retriever.encode(q.query)
        vectors[q.query_id]=values;np.savez(target,**values)
        rows=[dict(query_id=q.query_id,branch=name,cold_seconds=cold[name],warm_seconds=warm[name]) for name in ('bge','fulldoc')]
        timings.extend(rows);write_json(marker,dict(fingerprint=fingerprint,identity=identity,sha256=sha(target),timings=rows))
        print('ENCODE',q.query_id,flush=True)
    return vectors,pd.DataFrame(timings)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['historical','pilot'],default='historical')
    parser.add_argument('--cross-encoder-control',action='store_true');args=parser.parse_args()
    target=OUT/args.phase;target.mkdir(parents=True,exist_ok=True)
    before=protected();write_json(target/'protected_before.json',before)
    queries=pd.read_csv(FULL/'queries.csv',keep_default_na=False) if args.phase=='historical' else pilot_queries()
    if args.phase=='historical':queries['split']='historical_diagnostic';queries['family']=queries.category
    csv(target/'queries.csv',queries)
    protocol=dict(version=VERSION,phase=args.phase,parameter_selection=False,production_enabled=False,
        labels='AI_SILVER_DIAGNOSTIC' if args.phase=='historical' else 'PENDING_INDEPENDENT_REVIEW',
        candidate_depths=[50,100,200],rerank_depths=[20,30,50],rrf_k=60,
        fixed_weight_ablation=[1,1,2],weights_selected_from_labels=False,
        unjudged_policy='Withhold relevance metrics for any top10 with unjudged IDs; never impute zero',
        aggregation='Same complete query intersection for all methods at each candidate depth',
        candidate_recall='Known grade2 retrieved / all grade2 in frozen judged query pool; undefined if denominator=0',
        bootstrap='10000 paired query resamples; seed20261030; fixed AI labels; not annotation uncertainty',
        A4='Lexical condition-count sorting of A3 Top30',A6='Conservative sentence-local lexical evidence, not scientific entailment',
        A7='NOT_SELECTED_PENDING_INDEPENDENT_DEVELOPMENT_LABELS',
        chunks='Separate index; accepted last block + hashed frozen prefix; no encoder training',
        heldout_policy='Pilot heldout queries may be pooled now, but quality evaluation remains unopened until dev config is frozen; no heldout labels present')
    write_json(target/'protocol.json',protocol)
    retriever=RadarRetriever();vectors,encode_times=vector_cache(retriever,queries,target/'query_vectors')
    csv(target/'encoding_latency.csv',encode_times)
    labels=pd.read_csv(SILVER/'silver_labels_703.csv') if args.phase=='historical' else pd.DataFrame(columns=['query_id','doc_id','ai_label'])
    qrels={qid:dict(zip(g.doc_id,g.ai_label.astype(int))) for qid,g in labels.groupby('query_id')}
    historical=pd.read_csv(FULL/'comparison_rankings_top10.csv')
    candidates=[];ranks=[];per=[];runs=[];overlaps=[];parity=[];evidence_rows=[]
    chunk_ready=(OUT/'chunks/manifest.json').exists()
    if chunk_ready:retriever.load_chunks()
    for q in queries.itertuples():
        for depth in (50,100,200):
            start=time.perf_counter()
            branches,bt,mask,parsed=retriever.branches(q.query,depth,vectors=vectors[q.query_id])
            for name,branch in branches.items():
                for rank,(doc,score) in enumerate(branch,1):candidates.append(dict(query_id=q.query_id,depth=depth,branch=name,rank=rank,doc_id=doc,score=score))
            for a,b in itertools.combinations(branches,2):
                sa={d for d,s in branches[a]};sb={d for d,s in branches[b]}
                overlaps.append(dict(query_id=q.query_id,depth=depth,left=a,right=b,intersection=len(sa&sb),union=len(sa|sb),jaccard=len(sa&sb)/len(sa|sb) if sa|sb else None))
            methods={};candidate_ids={};method_times={}
            for method,names in BRANCHES.items():
                tick=time.perf_counter();selected={n:branches[n] for n in names}
                weights=dict(tfidf=1.,bge=1.,fulldoc=2.) if method=='A3_weighted' else None
                methods[method]=weighted_rrf(selected,weights);method_times[method]=time.perf_counter()-tick
                candidate_ids[method]=set().union(*[{d for d,s in selected[n]} for n in selected])
            tick=time.perf_counter();methods['A4']=retriever.rerank(q.query,methods['A3'],30,'conditions')
            method_times['A4']=method_times['A3']+time.perf_counter()-tick;candidate_ids['A4']=candidate_ids['A3']
            for n in (20,30,50):
                method='A6_N'+str(n);tick=time.perf_counter();methods[method]=retriever.rerank(q.query,methods['A3'],n,'evidence')
                method_times[method]=method_times['A3']+time.perf_counter()-tick;candidate_ids[method]=candidate_ids['A3']
            if chunk_ready:
                for mode in ('max','unique_mean'):
                    tick=time.perf_counter();chunk=retriever.chunk_branch(vectors[q.query_id]['fulldoc'],mask,depth,mode)
                    method='A5_'+mode;methods[method]=weighted_rrf({**branches,'chunk':[(d,s) for d,s,_ in chunk]})
                    method_times[method]=time.perf_counter()-tick
                    candidate_ids[method]=candidate_ids['A3']|{d for d,s,r in chunk}
                    for rank,(doc,score,record) in enumerate(chunk,1):
                        candidates.append(dict(query_id=q.query_id,depth=depth,branch='chunk_'+mode,rank=rank,doc_id=doc,score=score,
                            chunk_id=record['global_chunk_id'],offset_field=record['offset_field'],char_start=record['char_start'],char_end=record['char_end']))
            if args.cross_encoder_control and depth==100:
                tick=time.perf_counter();control=retriever.engine.v2.rerank(q.query,[(d,s) for d,s,p in methods['A3'][:20]])
                # Distinct score scale retained; only ranks are compared.
                methods['CE_CONTROL_N20']=[(d,s,dict(control='MiniLM cross-encoder; never promoted')) for d,s in control]+methods['A3'][20:]
                method_times['CE_CONTROL_N20']=time.perf_counter()-tick;candidate_ids['CE_CONTROL_N20']=candidate_ids['A3']
            for method,fused in methods.items():
                ids=[d for d,s,p in fused[:10]]
                metrics=evaluate_ranking(ids,qrels.get(q.query_id,{}),candidate_ids[method])
                row=dict(query_id=q.query_id,category=q.category,split=q.split,method=method,depth=depth,**metrics)
                row['unjudged_ids']=';'.join(row['unjudged_ids']);row['missing_relevant_candidates']=';'.join(row['missing_relevant_candidates']);per.append(row)
                names=BRANCHES.get(method,BRANCHES['A3'])
                enc=float(encode_times[(encode_times.query_id.eq(q.query_id)) & encode_times.branch.isin([n for n in names if n!='tfidf'])].warm_seconds.sum())
                seconds=enc+sum(bt[n+'_retrieval'] for n in names)+method_times[method]
                runs.append(dict(query_id=q.query_id,method=method,depth=depth,warm_component_seconds=seconds,
                    ranking_seconds=method_times[method],rss_bytes=psutil.Process().memory_info().rss,
                    latency_scope='sum of measured warm encoding, branch retrieval and fusion/rerank; sequential CPU; no UI/queue'))
                for rank,(doc,score,provenance) in enumerate(fused[:10],1):
                    meta=retriever.meta.loc[doc];ev=condition_evidence(parsed,meta.title,meta.abstract)
                    ranks.append(dict(query_id=q.query_id,query=q.query,category=q.category,split=q.split,method=method,depth=depth,rank=rank,doc_id=doc,
                        title=meta.title,abstract=meta.abstract,source=meta.source,year=meta.year,score=score,
                        ai_label=qrels.get(q.query_id,{}).get(doc),provenance=json.dumps(provenance,ensure_ascii=False),
                        original_rank=provenance.get('original_rank'),rank_change=provenance.get('rank_change'),
                        missing_conditions=';'.join(ev['missing_conditions'])))
                    if depth==100:evidence_rows.append(dict(query_id=q.query_id,method=method,doc_id=doc,evidence=ev))
            if depth==100 and args.phase=='historical':
                old=historical[historical.query_id.eq(q.query_id)&historical.method.eq('BATTERYBERT_FULLDOC_HYBRID')].sort_values('rank').doc_id.tolist()
                new=[d for d,s,p in methods['A0'][:10]]
                parity.append(dict(query_id=q.query_id,exact_order=old==new,intersection=len(set(old)&set(new)),old=';'.join(old),new=';'.join(new)))
            print('RANK',q.query_id,depth,f'{time.perf_counter()-start:.2f}s',flush=True)
        # Checkpoint per query; resume query inference from verified cache.
        csv(target/'rankings_top10.csv',pd.DataFrame(ranks));csv(target/'metrics_per_query.csv',pd.DataFrame(per))
    frames=dict(candidates=pd.DataFrame(candidates),rankings_top10=pd.DataFrame(ranks),metrics_per_query=pd.DataFrame(per),
        runs=pd.DataFrame(runs),candidate_overlap=pd.DataFrame(overlaps),baseline_parity=pd.DataFrame(parity))
    for name,frame in frames.items():csv(target/(name+'.csv'),frame)
    write_json(target/'evidence.json',evidence_rows)
    summary=[];cis=[];coverage=[];grouped=[]
    for depth,dp in frames['metrics_per_query'].groupby('depth'):
        methods=dp.method.unique();common=set(queries.query_id)
        for method in methods:common &= set(dp[dp.method.eq(method)&dp.status.eq('COMPLETE')].query_id)
        complete=dp[dp.query_id.isin(common)]
        for method,g in dp.groupby('method'):
            cg=g[g.query_id.isin(common)]
            summary.append(dict(method=method,depth=depth,queries_total=len(queries),common_evaluated_queries=len(common),
                query_coverage=len(common)/len(queries),Candidate_Recall_pool=g.Candidate_Recall_pool.mean() if g.Candidate_Recall_pool.notna().any() else None,
                candidate_recall_defined_queries=int(g.Candidate_Recall_pool.notna().sum()),
                judged_fraction=g.judged_fraction.mean(),**{m:float(cg[m].mean()) if len(cg) else None for m in (*METRICS,'Success_at_1','Success_at_10')}))
            coverage.append(dict(method=method,depth=depth,complete_queries=int(g.status.eq('COMPLETE').sum()),
                common_complete_queries=len(common),query_ids_common=';'.join(sorted(common)),queries_total=len(queries)))
            for category,gg in g.groupby('category'):
                gc=gg[gg.query_id.isin(common)]
                grouped.append(dict(method=method,depth=depth,category=category,queries=len(gg),common_evaluated_queries=len(gc),
                    **{m:float(gc[m].mean()) if len(gc) else None for m in METRICS}))
        if len(complete):cis.extend(dict(depth=depth,**row) for row in paired_bootstrap(complete))
    csv(target/'metrics_summary.csv',pd.DataFrame(summary));csv(target/'coverage.csv',pd.DataFrame(coverage));csv(target/'metrics_by_category.csv',pd.DataFrame(grouped))
    csv(target/'paired_bootstrap.csv',pd.DataFrame(cis,columns=['depth','method','metric','mean','ci95_low','ci95_high','delta','delta_ci95_low','delta_ci95_high','queries']))
    latency=frames['runs'].groupby(['method','depth']).agg(P50_seconds=('warm_component_seconds','median'),
        P95_seconds=('warm_component_seconds',lambda x:x.quantile(.95)),rss_max_bytes=('rss_bytes','max'),queries=('query_id','count')).reset_index()
    csv(target/'latency_memory.csv',latency)
    pool=frames['rankings_top10'].drop_duplicates(['query_id','doc_id'])[['query_id','query','category','split','doc_id','title','abstract','source','year']].copy()
    for field in ('relevance_grade','annotator_id','verifier_id','verification_reference','label_method','label_status','reviewed_at','rationale','query_review_reference','condition_labels','agreement'):
        pool[field]=''
    template=target/'annotation_template.csv'
    if not template.exists():csv(template,pool)
    csv(target/'annotation_template_latest.csv',pool)
    # Preserve user annotations: refreshed candidate pool is separate from template.
    csv(target/'candidate_annotation_pool.csv',pool)
    after=protected()
    if before!=after:raise ValueError('Protected artifacts changed')
    write_json(target/'preservation.json',dict(status='PASS',hashes=after))
    write_json(target/'manifest.json',dict(version=VERSION,phase=args.phase,queries=len(queries),
        configuration_sha256=sha(ROOT/'configs/search_radar_v1.json'),
        script_sha256=sha(Path(__file__)),source_sha256={p.name:sha(p) for p in (ROOT/'src').glob('search_radar_*.py')},
        chunks_enabled=chunk_ready,cross_encoder_control=args.cross_encoder_control,init_seconds=retriever.init_seconds,
        python=platform.python_version(),platform=platform.platform(),index_sizes_bytes=dict(tfidf=(ROOT/'outputs/final_system/tfidf/matrix.npz').stat().st_size,
        bge=(ROOT/'data/stage4/pretrained_embeddings.npy').stat().st_size,fulldoc=(FULL/'document_embeddings.npy').stat().st_size,
        chunks=(OUT/'chunks/vectors.npy').stat().st_size if chunk_ready else None),
        files={name:sha(target/name) for name in ('queries.csv','encoding_latency.csv','candidates.csv',
            'rankings_top10.csv','metrics_per_query.csv','runs.csv','candidate_overlap.csv','baseline_parity.csv',
            'metrics_summary.csv','coverage.csv','metrics_by_category.csv','paired_bootstrap.csv',
            'latency_memory.csv','candidate_annotation_pool.csv','annotation_template_latest.csv')},
        new_training=False,production_changed=False,A7='NOT_SELECTED_PENDING_LABELS'))
    retriever.close();print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__=='__main__':main()
