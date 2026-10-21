"""Reproducible exploratory LDA experiments on the frozen Stage 1 corpus."""
import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from .data_io import sha256

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


def csv(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def checkpoint(message,status='RUNNING'):
    timestamp=datetime.now(timezone(timedelta(hours=7))).isoformat()
    write_json(ROOT/'outputs/reports/stage2/progress.json', dict(stage=2,status=status,updated=timestamp,message=message))
    with (ROOT/'docs/EXPERIMENT_LOG.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n- {timestamp} - Stage 2 {status}: {message}\n')


def topic_sets(model,topn=10):
    return [set(w for w,p in model.show_topic(i,topn=topn)) for i in range(model.num_topics)]


def match_stability(left,right):
    if len(left)!=len(right):
        raise ValueError('Stability requires equal topic counts')
    scores=np.array([[len(a&b)/max(1,len(a|b)) for b in right] for a in left])
    rows,cols=linear_sum_assignment(-scores)
    return float(scores[rows,cols].mean()),[(int(a),int(b),float(scores[a,b])) for a,b in zip(rows,cols)]


def _fit_lda(bow,dictionary,k,seed,settings):
    return LdaModel(corpus=bow,id2word=dictionary,num_topics=k,random_state=seed,
        passes=settings['passes'],iterations=settings['iterations'],chunksize=settings['chunksize'],
        update_every=settings['update_every'],alpha=settings['alpha'],eta=settings['eta'],
        eval_every=settings['eval_every'],gamma_threshold=settings['gamma_threshold'],minimum_probability=0.0)


def run_model(k,seed,bow,texts,dictionary,config,input_fp,logger):
    folder=ROOT/f'models/lda/k{k:02d}_seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    settings={x:y for x,y in config['lda'].items() if x not in ['topics','base_seed','additional_seeds']}
    signature=fingerprint(dict(k=k,seed=seed,settings=settings,input=input_fp,gensim=importlib.metadata.version('gensim'),numpy=np.__version__,scipy=importlib.metadata.version('scipy'),trainer=inspect.getsource(_fit_lda)))
    success=folder/'training_success.json'
    model_path=folder/'model.gensim'
    if success.exists():
        manifest=json.loads(success.read_text(encoding='utf-8'))
        if manifest['fingerprint']!=signature:
            raise RuntimeError(f'Existing successful run differs from requested configuration: {folder}. Preserve it; choose a new experiment directory before changing training parameters.')
        if any(not (folder/item['file']).exists() or sha256(folder/item['file'])!=item['sha256'] for item in manifest['model_files']):
            raise RuntimeError(f'Saved model integrity failure: {folder}; do not silently retrain')
        model=LdaModel.load(str(model_path))
        logger.info('REUSE successful model K=%s seed=%s',k,seed)
    else:
        checkpoint(f'Training Gensim LDA K={k}, seed={seed}; completed runs preserved.')
        logger.info('TRAIN START K=%s seed=%s passes=%s iterations=%s',k,seed,settings['passes'],settings['iterations'])
        start=time.perf_counter()
        model=_fit_lda(bow,dictionary,k,seed,settings)
        model.save(str(model_path))
        reread=LdaModel.load(str(model_path))
        assert reread.num_topics==k and np.isfinite(reread.get_topics()).all()
        files=[dict(file=p.name,sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(folder.glob('model.gensim*'))]
        manifest=dict(fingerprint=signature,k=k,seed=seed,settings=settings,training_seconds=time.perf_counter()-start,model_files=files,documents=len(bow),dictionary_size=len(dictionary),gensim=importlib.metadata.version('gensim'))
        write_json(success,manifest)
        logger.info('TRAIN COMPLETE K=%s seed=%s seconds=%.1f',k,seed,manifest['training_seconds'])
    evaluation_signature=fingerprint(dict(training=signature,evaluation=config['evaluation'],texts=input_fp))
    metric_path=folder/'metrics.json'
    if metric_path.exists() and json.loads(metric_path.read_text(encoding='utf-8')).get('evaluation_fingerprint')==evaluation_signature:
        metrics=json.loads(metric_path.read_text(encoding='utf-8'))
        logger.info('REUSE metrics K=%s seed=%s',k,seed)
    else:
        logger.info('EVALUATE K=%s seed=%s full corpus, coherence topn=%s',k,seed,config['evaluation']['topn'])
        start=time.perf_counter()
        topn=config['evaluation']['topn']
        cv=CoherenceModel(model=model,texts=texts,dictionary=dictionary,coherence='c_v',topn=topn,window_size=config['evaluation']['cv_window_size'],processes=1)
        cv_topics=cv.get_coherence_per_topic()
        um=CoherenceModel(model=model,corpus=bow,dictionary=dictionary,coherence='u_mass',topn=topn,processes=1)
        um_topics=um.get_coherence_per_topic()
        words=topic_sets(model,topn)
        metrics=dict(k=k,seed=seed,c_v=float(np.mean(cv_topics)),u_mass=float(np.mean(um_topics)),topic_diversity=len(set.union(*words))/(k*topn),c_v_per_topic=list(map(float,cv_topics)),u_mass_per_topic=list(map(float,um_topics)),evaluation_seconds=time.perf_counter()-start,training_seconds=manifest['training_seconds'],evaluation_fingerprint=evaluation_signature)
        if not all(np.isfinite(metrics[key]) for key in ['c_v','u_mass','topic_diversity']):
            raise RuntimeError(f'Nonfinite metrics K={k}, seed={seed}')
        write_json(metric_path,metrics)
        logger.info('METRICS K=%s seed=%s C_v=%.5f U_Mass=%.5f diversity=%.5f',k,seed,metrics['c_v'],metrics['u_mass'],metrics['topic_diversity'])
    csv(folder/'top10_keywords.csv',pd.DataFrame([dict(topic_id=i,rank=r+1,token=w,weight=float(p)) for i in range(k) for r,(w,p) in enumerate(model.show_topic(i,topn=config['evaluation']['topn']))]))
    return model,metrics


def eda(main,tokens,table_dir):
    stats=[]
    terms=[]
    ngrams=[]
    keywords=[]
    classifications=[]
    year_terms=[]
    targets=['hard_carbon','cathode','anode','electrolyte','sodiation']
    for source,group in main.groupby('source'):
        indices=group.index.tolist()
        docs=[tokens[i] for i in indices]
        lengths=[len(x) for x in docs]
        stats.append(dict(source=source,documents=len(group),year_min=int(group.year.min()),year_max=int(group.year.max()),missing_year=int(group.year.isna().sum()),missing_abstract=int(group.abstract.eq('').sum()),missing_keywords=int(group.keywords.eq('').sum()),total_tokens=sum(lengths),unique_tokens=len(set(itertools.chain.from_iterable(docs))),mean_tokens=float(np.mean(lengths)),median_tokens=float(np.median(lengths)),p95_tokens=float(np.percentile(lengths,95)),mean_title_chars=float(group.title.str.len().mean()),mean_abstract_chars=float(group.abstract.str.len().mean())))
        tf=Counter(itertools.chain.from_iterable(docs)); df=Counter(w for doc in docs for w in set(doc))
        terms.extend(dict(source=source,term=w,term_frequency=n,document_frequency=df[w],tf_per_10000=10000*n/sum(lengths),document_share=df[w]/len(group)) for w,n in tf.most_common())
        for n in [2,3]:
            tf_n=Counter(); df_n=Counter()
            for doc in docs:
                grams=list(zip(*(doc[i:] for i in range(n))))
                tf_n.update(grams);df_n.update(set(grams))
            ngrams.extend(dict(source=source,n=n,ngram=' '.join(g),count=count,document_frequency=df_n[g],document_share=df_n[g]/len(group)) for g,count in tf_n.most_common(100))
        keycounts=Counter(k for text in group.keywords for k in set(k.strip().casefold() for k in text.split(';') if k.strip()))
        keywords.extend(dict(source=source,keyword=k,documents=n,document_share=n/len(group)) for k,n in keycounts.most_common())
        cc=Counter(c.strip() for text in group.classification for c in set(text.split(';')) if c.strip())
        classifications.extend(dict(source=source,classification=c,documents=n,classification_system='IPC' if source=='Patent' else 'WoS Categories') for c,n in cc.most_common())
    for (source,year),group in main.dropna(subset=['year']).groupby(['source','year']):
        for term in targets:
            count=sum(term in set(tokens[i]) for i in group.index)
            year_terms.append(dict(source=source,year=int(year),term=term,documents=count,total_documents=len(group),document_share=count/len(group)))
    csv(table_dir/'eda_source_summary.csv',pd.DataFrame(stats))
    csv(table_dir/'eda_term_frequencies.csv',pd.DataFrame(terms))
    csv(table_dir/'eda_ngrams.csv',pd.DataFrame(ngrams))
    csv(table_dir/'eda_author_keywords.csv',pd.DataFrame(keywords,columns=['source','keyword','documents','document_share']))
    csv(table_dir/'eda_classification.csv',pd.DataFrame(classifications))
    csv(table_dir/'eda_year_term_trends.csv',pd.DataFrame(year_terms))
    csv(table_dir/'eda_year_source_counts.csv',main.groupby(['source','year']).size().reset_index(name='documents'))
    csv(table_dir/'eda_document_types.csv',main.groupby(['source','document_type']).size().reset_index(name='documents'))
    csv(table_dir/'eda_document_lengths.csv',pd.DataFrame(dict(doc_id=main.doc_id,source=main.source,token_count=list(map(len,tokens)),title_chars=main.title.str.len(),abstract_chars=main.abstract.str.len())) )
    top_org=[]
    for source,group in main.groupby('source'):
        counts=Counter(org.strip() for s in group.organization for org in set(s.split(';')) if org.strip())
        top_org.extend(dict(source=source,organization=org,documents=n,role='assignee' if source=='Patent' else 'author_affiliation') for org,n in counts.most_common(50))
    csv(table_dir/'eda_top_organizations.csv',pd.DataFrame(top_org))


def export_topics(model,main,bow,table_dir,data_dir):
    k=model.num_topics
    vectors=np.zeros((len(main),k),dtype=np.float32)
    for i,doc in enumerate(bow):
        for topic,p in model.get_document_topics(doc,minimum_probability=0.0):
            vectors[i,topic]=p
    np.save(data_dir/'lda_topic_vectors.npy',vectors)
    vector_df=pd.DataFrame(vectors,columns=[f'topic_{i:02d}' for i in range(k)])
    vector_df.insert(0,'doc_id',main.doc_id.values)
    vector_df.insert(1,'source',main.source.values)
    vector_df.to_parquet(data_dir/'lda_topic_vectors.parquet',index=False)
    csv(table_dir/'document_topic_distributions.csv',vector_df)
    dominant=vectors.argmax(axis=1)
    confidence=vectors.max(axis=1)
    assignments=main[['doc_id','source','year','title','source_file','source_sheet','source_row','split_group_id']].copy()
    assignments['dominant_topic']=dominant
    assignments['dominant_probability']=confidence
    assignments['topic_entropy']=-np.sum(vectors*np.log(np.maximum(vectors,1e-30)),axis=1)
    csv(table_dir/'document_topic_assignments.csv',assignments)
    assignments.to_parquet(data_dir/'document_topic_assignments.parquet',index=False)
    csv(table_dir/'topic_keywords.csv',pd.DataFrame([dict(topic_id=t,rank=r+1,token=w,weight=float(p)) for t in range(k) for r,(w,p) in enumerate(model.show_topic(t,topn=10))]))
    summaries=[]; by_source=[]; reps=[]; trends=[]
    for t in range(k):
        summaries.append(dict(topic_id=t,top10_keywords='; '.join(w for w,p in model.show_topic(t,topn=10)),dominant_documents=int((dominant==t).sum()),soft_document_mass=float(vectors[:,t].sum())))
        for source in ['Patent','Publication']:
            mask=main.source.eq(source).values
            by_source.append(dict(topic_id=t,source=source,dominant_documents=int(((dominant==t)&mask).sum()),soft_document_mass=float(vectors[mask,t].sum()),share_of_source_documents=float(vectors[mask,t].mean())))
        eligible=np.flatnonzero(dominant==t)
        fallback=len(eligible)==0
        if fallback: eligible=np.arange(len(main))
        ranked=eligible[np.argsort(-vectors[eligible,t],kind='stable')[:3]]
        for rank,i in enumerate(ranked):
            row=main.iloc[i]
            reps.append(dict(topic_id=t,rank=rank+1,doc_id=row.doc_id,source=row.source,year=row.year,title=row.title,abstract=row.abstract,topic_probability=float(vectors[i,t]),dominant_topic=int(dominant[i]),fallback_for_empty_topic=fallback))
    for (source,year),group in main.dropna(subset=['year']).groupby(['source','year']):
        indices=group.index.to_numpy()
        for t in range(k):
            trends.append(dict(source=source,year=int(year),topic_id=t,documents=len(group),dominant_documents=int((dominant[indices]==t).sum()),soft_document_mass=float(vectors[indices,t].sum()),topic_share=float(vectors[indices,t].mean())))
    csv(table_dir/'topic_summary.csv',pd.DataFrame(summaries))
    csv(table_dir/'topic_source_distribution.csv',pd.DataFrame(by_source))
    csv(table_dir/'topic_representative_documents.csv',pd.DataFrame(reps))
    csv(table_dir/'topic_year_trends.csv',pd.DataFrame(trends))
    return vectors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/stage2.yaml')
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'))
    if config['stage']!=2: raise ValueError('Stage 2 only')
    for folder in ['outputs/reports/stage2','outputs/tables/stage2','outputs/figures/stage2','outputs/logs','models/lda','data/stage2']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s %(message)s',handlers=[logging.FileHandler(ROOT/'outputs/logs/stage2.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)],force=True)
    # Keep inference/training diagnostics in the persistent log without flooding console.
    for name in ['gensim.models.ldamodel','gensim.models.coherencemodel','gensim.topic_coherence','gensim.corpora.dictionary','gensim.utils']:
        logging.getLogger(name).setLevel(logging.WARNING)
    training_logger=logging.getLogger('gensim.models.ldamodel')
    training_logger.setLevel(logging.INFO)
    training_logger.propagate=False
    training_logger.addHandler(logging.getLogger().handlers[0])
    logger=logging.getLogger('stage2')
    protected_paths=sorted(list((ROOT/'data/interim').glob('*.parquet'))+list((ROOT/'data/processed').glob('*.parquet'))+[ROOT/'patent.xlsx',ROOT/'publication.xlsx',ROOT/'configs/config.yaml'])
    hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in protected_paths}
    checkpoint('User provisionally approved the current Stage 1 corpus. Starting EDA and five requested LDA configurations; acceptance proposals not applied.')
    try:
        with threadpool_limits(limits=1):
            main_data=pd.read_parquet(ROOT/config['input_main']).reset_index(drop=True)
            token_data=pd.read_parquet(ROOT/config['input_tokens']).reset_index(drop=True)
            assert len(main_data)==config['expected_main_records'] and main_data.doc_id.tolist()==token_data.doc_id.tolist()
            texts=[list(x) for x in token_data.tokens]
            dictionary=corpora.Dictionary(texts)
            bow=[dictionary.doc2bow(doc) for doc in texts]
            assert all(bow) and sum(sum(n for w,n in doc) for doc in bow)==sum(map(len,texts))
            data_dir=ROOT/'data/stage2'; tables=ROOT/'outputs/tables/stage2'; reports=ROOT/'outputs/reports/stage2'
            dictionary.save(str(ROOT/'models/lda/dictionary.gensim'))
            corpora.MmCorpus.serialize(str(data_dir/'corpus_bow.mm'),bow)
            csv(tables/'dictionary_tokens.csv',pd.DataFrame([dict(token_id=i,token=dictionary[i],document_frequency=dictionary.dfs[i],term_frequency=dictionary.cfs[i]) for i in range(len(dictionary))]))
            master=pd.read_parquet(ROOT/config['input_clean_master'])
            uncertain=master[master.relevance.eq('UNCERTAIN')].copy()
            uncertain.to_parquet(data_dir/'uncertain_sensitivity_pool.parquet',index=False)
            csv(tables/'uncertain_sensitivity_index.csv',uncertain[['doc_id','source','relevance','relevance_reason','disposition','split_group_id']])
            del master,uncertain
            write_json(reports/'input_manifest.json',dict(protected_hashes=hashes,config=config,documents=len(main_data),dictionary_size=len(dictionary),tokens=sum(map(len,texts)),input_fingerprint=fingerprint(hashes),dictionary_policy='All Stage 1 tokens; no dictionary filtering, cleaning, phrase learning or document removal',versions={p:importlib.metadata.version(p) for p in ['gensim','numpy','scipy','pandas','matplotlib']}))
            eda(main_data,texts,tables)
            logger.info('EDA complete; documents=%s vocabulary=%s tokens=%s',len(main_data),len(dictionary),sum(map(len,texts)))
            input_fp=fingerprint(dict(main=hashes[config['input_main']],tokens=hashes[config['input_tokens']],dictionary=dictionary.token2id))
            baseline=[]
            for k in config['lda']['topics']:
                model,metrics=run_model(k,config['lda']['base_seed'],bow,texts,dictionary,config,input_fp,logger)
                baseline.append(metrics)
                csv(tables/'lda_base_metrics.csv',pd.DataFrame([{key:m[key] for key in ['k','seed','c_v','u_mass','topic_diversity','training_seconds','evaluation_seconds']} for m in baseline]))
            ranked=sorted(baseline,key=lambda m:(-m['c_v'],-m['u_mass'],m['k']))
            promising=[m['k'] for m in ranked[:config['evaluation']['promising_count']]]
            write_json(reports/'promising_configurations.json',dict(selection='Top two base-seed C_v; ties U_Mass then smaller K',topics=promising))
            all_metrics=list(baseline); stability_rows=[]; matches=[]
            for k in promising:
                models={}
                for seed in [config['lda']['base_seed']]+config['lda']['additional_seeds']:
                    model,metric=run_model(k,seed,bow,texts,dictionary,config,input_fp,logger)
                    models[seed]=topic_sets(model,config['evaluation']['topn'])
                    if seed!=config['lda']['base_seed']: all_metrics.append(metric)
                for left,right in itertools.combinations(models,2):
                    value,pairs=match_stability(models[left],models[right])
                    stability_rows.append(dict(k=k,seed_a=left,seed_b=right,mean_matched_jaccard=value))
                    matches.extend(dict(k=k,seed_a=left,seed_b=right,topic_a=a,topic_b=b,jaccard=p) for a,b,p in pairs)
            csv(tables/'lda_all_run_metrics.csv',pd.DataFrame([{key:m[key] for key in ['k','seed','c_v','u_mass','topic_diversity','training_seconds','evaluation_seconds']} for m in all_metrics]))
            stab=pd.DataFrame(stability_rows)
            csv(tables/'lda_stability_pairs.csv',stab)
            csv(tables/'lda_stability_topic_matches.csv',pd.DataFrame(matches))
            comparison=pd.DataFrame([{key:m[key] for key in ['k','seed','c_v','u_mass','topic_diversity','training_seconds']} for m in baseline])
            comparison['stability']=comparison.k.map(stab.groupby('k').mean_matched_jaccard.mean())
            comparison['stability_std']=comparison.k.map(stab.groupby('k').mean_matched_jaccard.std(ddof=0))
            for metric in ['c_v','u_mass']:
                low,high=comparison[metric].min(),comparison[metric].max()
                comparison[metric+'_normalized']=1.0 if high==low else (comparison[metric]-low)/(high-low)
            s=config['selection']
            comparison['selection_score']=s['cv_minmax_weight']*comparison.c_v_normalized+s['umass_minmax_weight']*comparison.u_mass_normalized+s['diversity_weight']*comparison.topic_diversity+s['stability_weight']*comparison.stability
            selected=comparison.dropna(subset=['selection_score']).sort_values(['selection_score','k'],ascending=[False,True]).iloc[0]
            k=int(selected.k)
            csv(tables/'lda_comparison.csv',comparison)
            selection=dict(selected_k=k,seed=config['lda']['base_seed'],promising=promising,score=float(selected.selection_score),policy=config['selection'],stability_seeds=[config['lda']['base_seed']]+config['lda']['additional_seeds'],selected_model=f'models/lda/k{k:02d}_seed{config["lda"]["base_seed"]}/model.gensim',dictionary='models/lda/dictionary.gensim',probability_topic_order=list(range(k)),exploratory_full_corpus_metrics=True)
            write_json(reports/'selected_model.json',selection)
            logger.info('SELECTED K=%s selection_score=%.6f',k,selected.selection_score)
            model=LdaModel.load(str(ROOT/selection['selected_model']))
            vectors=export_topics(model,main_data,bow,tables,data_dir)
            from .stage2_figures import build_figures
            manifest=build_figures(tables,ROOT/'outputs/figures/stage2',config['visualization']['dpi'])
            write_json(reports/'figure_manifest.json',manifest)
            assert vectors.shape==(len(main_data),k) and np.isfinite(vectors).all() and np.allclose(vectors.sum(axis=1),1,atol=2e-5)
            assert all(sha256(ROOT/p)==digest for p,digest in hashes.items())
            write_json(reports/'runtime_validation.json',dict(status='PASS',protected_files_unchanged=True,documents=len(main_data),vector_shape=list(vectors.shape),vectors_nonnegative=bool((vectors>=0).all()),max_probability_sum_error=float(np.abs(vectors.sum(axis=1)-1).max()),all_five_K_evaluated=len(baseline)==5,successful_training_runs=len(all_metrics),stability_pairs=len(stability_rows),figures=len(manifest),source_counts=main_data.source.value_counts().to_dict()))
            checkpoint(f'Stage 2 experiments and exports complete: selected K={k}; {len(all_metrics)} successful runs. Runtime checks PASS; final tests and report pending.',status='VALIDATED_PENDING_TESTS')
    except Exception as exc:
        logger.exception('Stage 2 interrupted; successful training runs remain reusable')
        checkpoint(f'Interrupted: {type(exc).__name__}: {exc}. Rerun run_stage2.py; completed configurations are reused.',status='PARTIAL')
        raise


if __name__=='__main__': main()
