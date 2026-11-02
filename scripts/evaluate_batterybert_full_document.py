"""Evaluate saved full-document vectors on topic diagnostics, without Torch.

Validation selects geometry/word weighting; test and common-reference C_v are
computed only after that choice. No expert labels are manufactured.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='4'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json
from itertools import product
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from src.batterybert_optimization import Geometry,TopicPipeline,ContiguousScientificVectorizer,document_npmi
from src.batterybert_topic_tools import distribution,phrase_reference,GENERIC,PROTECTED,BOUNDARY
from src.scientific_keyword_stream import prepare_corpus_stream
from src.model_comparison_io import write_json

OUT=ROOT/'outputs/batterybert_full_document_20261024_v1'
OLD=ROOT/'outputs/batterybert_optimization_20261023_v1'


def keywords_from_counts(counts,binary,vocabulary,labels,exponent,phrase_boost=1.):
    member=sparse.csr_matrix((np.ones(len(labels)),(labels,np.arange(len(labels)))),shape=(10,len(labels)))
    tf=(member@counts).toarray();tf/=tf.sum(1,keepdims=True)
    df=(member@binary).toarray();total=np.asarray(binary.sum(0)).ravel()
    score=tf*np.log1p(len(labels)/np.maximum(total,1))[None,:]*((df+1)/(total[None,:]+10))**exponent
    score*=np.asarray([phrase_boost if ' ' in word else 1. for word in vocabulary])[None,:]
    result=[]
    for topic in range(10):
        chosen=[]
        for at in np.argsort(-score[topic],kind='stable'):
            term=vocabulary[at];parts=set(term.split())
            if df[topic,at]<5 or any(parts<=old or old<=parts for old in chosen):continue
            chosen.append(parts);result.append(dict(topic_id=topic,rank=len(chosen),token=term,weight=float(score[topic,at]),train_document_support=int(df[topic,at])))
            if len(chosen)==10:break
        if len(chosen)!=10:raise ValueError('Insufficient supported scientific keywords')
    return pd.DataFrame(result)


def topic_phase(out=OUT):
    OUT=Path(out)
    dest=OUT/'topics';dest.mkdir(exist_ok=True)
    if (dest/'final_metrics.json').exists():print('Topic final evaluation already frozen');return
    plan=dict(geometry=['center','source'],exclusivity=[0.,.25,.5],phrase_boost=[1.,1.5],seed=42,stability_seed=43,final_seed=137,
        n_init=10,fit='train',select='validation',test_not_used_for_selection=True,
        gates=dict(source_v_max=.5,ari_min=.6,diversity_min=.75,npmi_min=0.,largest_max=.30,top2_max=.45,smallest_min=.005,coverage_min=.95),
        representation='Immutable raw Title and Abstract; lemmatization with stopword/punctuation/field boundaries',
        stopping='No final test when no validation candidate passes; freeze selection before final test',
        promotion=dict(common_reference_cv_noninferiority_margin=.02,expert_validation=False))
    if (dest/'plan.json').exists() and json.loads((dest/'plan.json').read_text())!=plan:raise ValueError('Frozen topic protocol changed')
    write_json(dest/'plan.json',plan)
    from src.final_system_artifacts import sha
    provenance=dict(evaluator_sha256=sha(Path(__file__)),vectors_sha256=sha(OUT/'document_embeddings.npy'),
        geometry_implementation_sha256=sha(ROOT/'src/batterybert_optimization.py'),
        topic_tools_sha256=sha(ROOT/'src/batterybert_topic_tools.py'),
        lexical_stream_sha256=sha(ROOT/'src/scientific_keyword_stream.py'),
        text_processing_sha256=sha(ROOT/'src/text_processing.py'),
        ordered_ids_sha256=sha(OUT/'ordered_doc_ids.json'),selection_sha256=sha(OUT/'selection.json'),
        corpus_sha256=sha(ROOT/'data/processed/main_corpus.parquet'),
        splits_sha256=sha(ROOT/'data/stage3/split_assignments.parquet'))
    provenance_path=dest/'input_provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text())!=provenance:
        raise ValueError('Topic evaluator/input changed: preserve this version and use a new output directory')
    write_json(provenance_path,provenance)
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    splits=pd.read_parquet(ROOT/'data/stage3/split_assignments.parquet').set_index('doc_id').loc[main.doc_id].split.to_numpy()
    train,val,test=[splits==name for name in ('train','validation','test')]
    vectors=np.load(OUT/'document_embeddings.npy',mmap_mode='r')
    if json.loads((OUT/'ordered_doc_ids.json').read_text())!=main.doc_id.tolist():raise ValueError('Embedding IDs mismatch')
    phrases=yaml.safe_load((ROOT/'configs/config.yaml').read_text(encoding='utf8'))['preprocessing']['domain_phrases']
    texts=prepare_corpus_stream(OUT,main,phrases)
    docs=np.asarray([' '.join(t) for t in texts],dtype=object)
    vectorizer=ContiguousScientificVectorizer(lowercase=False,ngram_range=(1,2),min_df=5,max_df=.6,max_features=40000,
        stop_words=sorted((set(ENGLISH_STOP_WORDS)|set(GENERIC))-PROTECTED))
    counts=vectorizer.fit_transform(docs[train]);binary=counts.copy();binary.data[:]=1
    candidates=[]
    for mode in plan['geometry']:
        geometry=Geometry(mode).fit(vectors[train],main.source.to_numpy()[train])
        xtrain=geometry.transform(vectors[train]);xval=geometry.transform(vectors[val])
        cluster=KMeans(n_clusters=10,n_init=10,random_state=42).fit(xtrain)
        alternate=KMeans(n_clusters=10,n_init=10,random_state=43).fit(xtrain)
        labels=cluster.predict(xval);ari=float(adjusted_rand_score(labels,alternate.predict(xval)))
        val_distribution,_,base=distribution(labels,main.source.to_numpy()[val])
        base['smallest_topic_share']=float(val_distribution.corpus_share.min())
        for exponent,phrase_boost in product(plan['exclusivity'],plan['phrase_boost']):
            name=f'{mode}_ex{exponent}_pb{phrase_boost}'
            folder=dest/name;folder.mkdir(exist_ok=True)
            keys=keywords_from_counts(counts,binary,vectorizer.get_feature_names_out(),cluster.labels_,exponent,phrase_boost)
            metric=dict(**base,**document_npmi(docs[val].tolist(),keys),candidate=name,geometry=mode,
                exclusivity=exponent,phrase_boost=phrase_boost,stability_ari=ari,keyword_diversity=keys.token.nunique()/100)
            metric['passes']=(metric['source_cramers_v']<=.5 and ari>=.6 and metric['keyword_diversity']>=.75
                and metric['document_npmi']>=0 and metric['largest_topic_share']<=.3 and metric['top2_topic_share']<=.45
                and metric['smallest_topic_share']>=.005 and metric['supported_keyword_fraction']>=.95)
            keys.to_csv(folder/'keywords.csv',index=False)
            joblib.dump(TopicPipeline(geometry,cluster,'batterybert_full_document',main.doc_id[train].tolist()),folder/'pipeline.joblib')
            write_json(folder/'metrics.json',metric);candidates.append(metric)
            print(name,'NPMI',metric['document_npmi'],'V',metric['source_cramers_v'],'ARI',ari,'passes',metric['passes'],flush=True)
    pd.DataFrame([{k:v for k,v in m.items() if not isinstance(v,list)} for m in candidates]).to_csv(dest/'validation_comparison.csv',index=False)
    selected=max(candidates,key=lambda m:(m['passes'],m['document_npmi']))
    write_json(dest/'selection.json',selected)
    if not selected['passes']:
        raise RuntimeError('No topic candidate passed validation; final test remains unopened. Preserve this development run before revising the protocol.')
    folder=dest/selected['candidate'];pipeline=joblib.load(folder/'pipeline.joblib');keys=pd.read_csv(folder/'keywords.csv')
    labels=pipeline.predict(vectors);summary,sources,metrics=distribution(labels,main.source.tolist())
    final=KMeans(n_clusters=10,n_init=10,random_state=137).fit(pipeline.geometry.transform(vectors[train]))
    test_distribution,_,heldout=distribution(labels[test],main.source.to_numpy()[test])
    heldout['smallest_topic_share']=float(test_distribution.corpus_share.min())
    heldout.update(document_npmi(docs[test].tolist(),keys),stability_ari=float(adjusted_rand_score(labels[test],final.predict(pipeline.geometry.transform(vectors[test])))))
    heldout['keyword_diversity']=keys.token.nunique()/100
    heldout_gates=dict(source=heldout['source_cramers_v']<=.5,
        stability=heldout['stability_ari']>=.6,npmi=heldout['document_npmi']>=0,
        largest=heldout['largest_topic_share']<=.30,top2=heldout['top2_topic_share']<=.45,
        smallest=heldout['smallest_topic_share']>=.005,coverage=heldout['supported_keyword_fraction']>=.95,
        diversity=heldout['keyword_diversity']>=.75)
    old_keys=pd.read_csv(OLD/'topic_keywords.csv');features=set(keys.token)|set(old_keys.token)
    reference=[[word for word in doc if word!=BOUNDARY] for doc in phrase_reference(texts,features)]
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    dictionary=Dictionary(reference)
    coherence=[]
    for name,frame in [('previous',old_keys),('full_document',keys)]:
        topics=[[w.replace(' ','\u2581') for w in group.sort_values('rank').token] for t,group in frame.groupby('topic_id',sort=True)]
        values=CoherenceModel(topics=topics,texts=reference,dictionary=dictionary,coherence='c_v',window_size=110,topn=10,processes=1).get_coherence_per_topic()
        coherence.append(dict(model=name,c_v=float(np.mean(values)),topic_diversity=frame.token.nunique()/100,
            supported_keyword_fraction=sum(w.replace(' ','\u2581') in dictionary.token2id for w in frame.token)/len(frame)))
    pd.DataFrame(coherence).to_csv(dest/'common_coherence.csv',index=False)
    training=json.loads((OUT/'selection.json').read_text())
    metrics.update(c_v=coherence[1]['c_v'],topic_diversity=keys.token.nunique()/100,
        heldout=heldout,heldout_gates=heldout_gates,validation_pass=selected['passes'],encoder_training_accepted=training['accepted'],
        promotion_gate=(selected['passes'] and training['accepted'] and all(heldout_gates.values())
            and all(row['supported_keyword_fraction']==1. for row in coherence)
            and coherence[1]['c_v']>=coherence[0]['c_v']-.02),
        scientific_validation='Pending independent expert labels; common-reference C_v is descriptive')
    assigned=main[['doc_id','source','year']].copy();assigned['topic_id']=labels
    assigned.to_csv(dest/'topic_assignments.csv',index=False);keys.to_csv(dest/'topic_keywords.csv',index=False)
    summary.to_csv(dest/'topic_distribution.csv',index=False);sources.to_csv(dest/'topic_source_distribution.csv')
    joblib.dump(pipeline,dest/'topic_pipeline.joblib')
    distances=pipeline.cluster.transform(pipeline.geometry.transform(vectors));reps=[]
    nearest=np.argsort(distances,axis=1,kind='stable')[:,:2]
    neighbor=np.where(nearest[:,0]!=labels,nearest[:,0],nearest[:,1])
    own=distances[np.arange(len(labels)),labels];other=distances[np.arange(len(labels)),neighbor]
    gap=np.maximum(0.,other-own)/np.maximum(other,1e-12)
    thresholds=dict(train_distance_p95=float(np.quantile(own[train],.95)),train_relative_gap_p05=float(np.quantile(gap[train],.05)),
        interpretation='Uncalibrated review priority, not probability/confidence; no assignments are changed')
    diagnostic=main[['doc_id','source','year','title','abstract']].copy()
    diagnostic['topic_id']=labels;diagnostic['next_topic_id']=neighbor
    diagnostic['centroid_distance']=own;diagnostic['relative_gap_to_next_topic']=gap
    diagnostic['review_priority']=(own>thresholds['train_distance_p95'])|(gap<thresholds['train_relative_gap_p05'])
    diagnostic.to_csv(dest/'assignment_diagnostics.csv',index=False)
    diagnostic[diagnostic.review_priority].sort_values('relative_gap_to_next_topic').to_csv(dest/'review_priority.csv',index=False)
    write_json(dest/'review_priority_thresholds.json',thresholds)
    metrics['review_priority_documents']=int(diagnostic.review_priority.sum())
    for topic in range(10):
        ids=np.flatnonzero(labels==topic);near=ids[np.argsort(distances[ids,topic])[:3]]
        random=np.random.default_rng(20261024+topic).choice(ids,min(3,len(ids)),replace=False)
        for role,chosen in [('central',near),('random',random)]:
            for i in chosen:
                row=main.iloc[i];reps.append(dict(topic_id=topic,sample_role=role,doc_id=row.doc_id,title=row.title,abstract=row.abstract,source=row.source,year=row.year))
    pd.DataFrame(reps).to_csv(dest/'representative_documents.csv',index=False)
    pd.DataFrame([dict(topic_id=t,topic_label=' / '.join(g.sort_values('rank').token.head(3)),label_status='AUTOMATIC_KEYWORD_SUMMARY_NOT_EXPERT_LABEL')
        for t,g in keys.groupby('topic_id')]).to_csv(dest/'topic_labels.csv',index=False)
    write_json(dest/'final_metrics.json',metrics)
    print('TOPIC FINAL',json.dumps(metrics),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    topic_phase(parser.parse_args().output)
