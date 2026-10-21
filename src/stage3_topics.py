"""BERTopic on saved fine-tuned BGE embeddings; no transformer training here."""
import os
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).resolve().parents[1]/'.cache/numba'))
os.environ.setdefault('NUMBA_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import time
import inspect
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from joblib import parallel_backend
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from umap import UMAP
from gensim import corpora
from gensim.models import CoherenceModel
from .stage3_common import ROOT,REPORT,TABLE,DATA,config,read_json,write_json,csv,digest,progress,signature,verify_frozen,require_cache_fingerprint
from .stage3_bertopic_model import FrozenTokensBERTopic

CFG=config();MODELS=ROOT/'models/bertopic'

def fit(seed,docs,embeddings,dictionary):
    folder=MODELS/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'training_success.json';model_path=folder/'model.pickle'
    identity=signature(dict(settings=CFG['bertopic'],seed=seed,representation_documents_sha256=signature(docs),embedding_manifest=read_json(REPORT/'embedding_manifest.json'),dictionary_sha256=digest(ROOT/'models/lda/dictionary.gensim'),trainer=inspect.getsource(fit),text_preprocessing=inspect.getsource(FrozenTokensBERTopic._preprocess_text),versions={n:importlib.metadata.version(n) for n in ['bertopic','umap-learn','hdbscan','scikit-learn']}))
    if marker.exists():
        manifest=read_json(marker);assert manifest['fingerprint']==identity
        assert digest(model_path)==manifest['model_sha256']
        progress(f'Reusing successful BERTopic seed={seed}; no refitting.')
        return BERTopic.load(str(model_path)),manifest
    settings=CFG['bertopic']
    umap=UMAP(n_neighbors=settings['umap_neighbors'],n_components=settings['umap_components'],min_dist=settings['umap_min_dist'],metric='cosine',random_state=seed,n_jobs=1,low_memory=True)
    cluster=HDBSCAN(min_cluster_size=settings['min_cluster_size'],min_samples=settings['min_samples'],metric='euclidean',cluster_selection_method=settings['cluster_selection_method'],prediction_data=True,core_dist_n_jobs=1)
    vectorizer=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,vocabulary=dictionary.token2id)
    model=FrozenTokensBERTopic(embedding_model=None,umap_model=umap,hdbscan_model=cluster,vectorizer_model=vectorizer,ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),top_n_words=10,nr_topics=None,calculate_probabilities=False,verbose=True)
    progress(f'Fitting real BERTopic seed={seed}: fine-tuned BGE embeddings + UMAP + HDBSCAN + c-TF-IDF, nr_topics=None.')
    start=time.perf_counter()
    with parallel_backend('sequential',n_jobs=1):
        assignments,probabilities=model.fit_transform(docs,embeddings=embeddings)
    assert np.isfinite(model.ctfidf_model._idf_diag.data).all() and np.isfinite(model.c_tf_idf_.data).all(),'c-TF-IDF must be finite without changing Stage 1 tokens'
    model.save(str(model_path),serialization='pickle')
    # Full pickle preserves UMAP/HDBSCAN for inference with explicit embeddings.
    # Compact representation is additional, not a substitute for the fitted pipeline.
    model.save(str(folder/'compact'),serialization='safetensors',save_ctfidf=True)
    np.save(folder/'umap_5d.npy',model.umap_model.embedding_)
    labels=np.asarray(assignments,dtype=np.int32);np.save(folder/'assignments.npy',labels)
    if probabilities is not None: np.save(folder/'membership_strength.npy',np.asarray(probabilities,dtype=np.float32))
    manifest=dict(fingerprint=identity,seed=seed,seconds=time.perf_counter()-start,topics=len(set(labels)-{-1}),outliers=int(np.sum(labels==-1)),outlier_ratio=float(np.mean(labels==-1)),model_sha256=digest(model_path),settings=settings,embedding_model='Externally supplied fine-tuned BGE; use explicit embeddings for transform')
    write_json(marker,manifest)
    progress(f'BERTopic seed={seed} saved: {manifest["topics"]} topics, outlier ratio={manifest["outlier_ratio"]:.4f}.')
    return model,manifest

def topwords(model):
    return {int(topic):[w for w,p in model.get_topic(topic)[:10] if w and p>0] for topic in sorted(model.get_topics()) if topic!=-1}

def stability(left,right,labels1,labels2):
    a=list(left);b=list(right)
    matrix=np.array([[len(set(left[i])&set(right[j]))/max(1,len(set(left[i])|set(right[j]))) for j in b] for i in a])
    if matrix.size:
        rows,cols=linear_sum_assignment(-matrix)
        matches=[dict(base_topic=a[i],other_topic=b[j],jaccard=float(matrix[i,j])) for i,j in zip(rows,cols)]
        score=float(matrix[rows,cols].sum()/max(len(a),len(b)))
        matched_mean=float(matrix[rows,cols].mean())
    else:matches=[];score=0.;matched_mean=0.
    shared=(labels1!=-1)&(labels2!=-1)
    return dict(hungarian_top10_jaccard_unmatched_zero=score,matched_topics_jaccard_mean=matched_mean,base_topics=len(a),other_topics=len(b),matched_topics=len(matches),ari_all_labels_including_outlier=float(adjusted_rand_score(labels1,labels2)),common_inlier_documents=int(shared.sum()),ari_common_inliers=float(adjusted_rand_score(labels1[shared],labels2[shared])) if shared.sum()>1 else None,method='Two UMAP seeds, fixed fine-tuned embeddings/corpus; rectangular Hungarian top10 Jaccard divided by max topic count. Unmatched topics score 0. Outlier excluded from word matching.'),matches

def metrics_cache_identity(model_sha256, representation_sha256):
    return signature(dict(model_sha256=model_sha256,representation_sha256=representation_sha256,
        dictionary_sha256=digest(ROOT/'models/lda/dictionary.gensim'),
        bow_sha256=digest(ROOT/'data/stage2/corpus_bow.mm'),
        evaluation=dict(topn=10,c_v_window=110,processes=1,outliers='excluded'),
        evaluator=inspect.getsource(run),topwords=inspect.getsource(topwords),
        versions={n:importlib.metadata.version(n) for n in ['gensim','numpy']}))

def run():
    verify_frozen();assert read_json(ROOT/'models/bert_finetuned/training_success.json')['real_gradient_updates']
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    tokens=pd.read_parquet(ROOT/'data/processed/lda_tokens.parquet')
    assert tokens.doc_id.tolist()==main.doc_id.tolist()
    texts=[list(t) for t in tokens['tokens']]
    assert all(len(t)==int(n) for t,n in zip(texts,tokens.token_count)),'Stage 1 token counts must match the representation field'
    docs=[' '.join(t) for t in texts]
    embeddings=np.load(DATA/'embeddings.npy',mmap_mode='r')
    assert embeddings.shape==(len(main),768)
    dictionary=corpora.Dictionary.load(str(ROOT/'models/lda/dictionary.gensim'))
    metrics_path=REPORT/'bertopic_metrics.json'
    representation_sha256=signature(docs)
    # Check existing caches before fitting or exporting anything. A missing model
    # marker cannot establish the identity of an existing evaluation cache.
    if metrics_path.exists():
        model_marker=MODELS/f"seed{CFG['bertopic']['seed']}"/'training_success.json'
        model_sha256=read_json(model_marker)['model_sha256'] if model_marker.exists() else None
        require_cache_fingerprint(metrics_path,metrics_cache_identity(model_sha256,representation_sha256))
    projection_path=DATA/'umap_2d.npy'
    projection_marker=projection_path.with_suffix('.cache.json')
    projection_settings=dict(n_neighbors=15,n_components=2,min_dist=.1,metric='cosine',random_state=42,n_jobs=1,low_memory=True)
    projection_identity=signature(dict(ordered_doc_ids=main.doc_id.astype(str).tolist(),
        embeddings_sha256=digest(DATA/'embeddings.npy'),settings=projection_settings,
        umap_version=importlib.metadata.version('umap-learn')))
    projection_metadata=require_cache_fingerprint(projection_path,projection_identity,projection_marker)
    if projection_metadata is not None and projection_metadata.get('projection_sha256')!=digest(projection_path):
        raise ValueError('Projection cache content mismatch; preserve historical artifacts and recompute in an isolated output directory.')
    model,manifest=fit(CFG['bertopic']['seed'],docs,embeddings,dictionary)
    words=topwords(model)
    labels=np.asarray(model.topics_)
    strengths=np.asarray(model.probabilities_) if model.probabilities_ is not None else np.full(len(main),np.nan)
    assignments=main[['doc_id','source','year','split_group_id']].copy()
    assignments['topic_id']=labels;assignments['membership_strength']=strengths
    assignments.to_parquet(DATA/'bertopic_assignments.parquet',index=False);csv('bertopic_assignments.csv',assignments)
    summary=[];keywords=[];representatives=[]
    for topic in sorted(set(labels)):
        indices=np.flatnonzero(labels==topic)
        tw=[w for w,p in model.get_topic(int(topic))[:10] if w and p>0]
        summary.append(dict(topic_id=int(topic),documents=len(indices),document_share=len(indices)/len(main),top10_keywords='; '.join(tw),is_outlier=topic==-1))
        for rank,(word,weight) in enumerate(model.get_topic(int(topic))[:10],1):
            keywords.append(dict(topic_id=int(topic),rank=rank,token=word,ctfidf_weight=float(weight)))
        if topic==-1:continue
        order=indices[np.argsort(-strengths[indices],kind='stable')][:3]
        for rank,index in enumerate(order,1):
            row=main.iloc[index]
            representatives.append(dict(topic_id=int(topic),rank=rank,doc_id=row.doc_id,source=row.source,title=row.title,abstract=row.abstract,membership_strength=float(strengths[index]),selection_method='Highest HDBSCAN membership strength among assigned documents; not BERTopic internal c-TF-IDF representatives'))
    csv('bertopic_topic_summary.csv',pd.DataFrame(summary));csv('bertopic_keywords.csv',pd.DataFrame(keywords));csv('bertopic_representative_documents.csv',pd.DataFrame(representatives))
    csv('bertopic_topic_sources.csv',assignments.groupby(['topic_id','source']).size().rename('documents').reset_index())
    csv('bertopic_topic_years.csv',assignments.groupby(['year','topic_id','source']).size().rename('documents').reset_index())
    metrics_identity=metrics_cache_identity(manifest['model_sha256'],representation_sha256)
    if metrics_path.exists():metrics=require_cache_fingerprint(metrics_path,metrics_identity)
    else:
        start=time.perf_counter();topic_lists=list(words.values())
        assert topic_lists and all(len(w)==10 for w in topic_lists),'Need ten valid words per non-outlier topic'
        cv=CoherenceModel(topics=topic_lists,texts=texts,dictionary=dictionary,coherence='c_v',window_size=110,topn=10,processes=1).get_coherence_per_topic()
        bow=corpora.MmCorpus(str(ROOT/'data/stage2/corpus_bow.mm'))
        um=CoherenceModel(topics=topic_lists,corpus=bow,dictionary=dictionary,coherence='u_mass',topn=10,processes=1).get_coherence_per_topic()
        metrics=dict(topics=len(words),c_v=float(np.mean(cv)),u_mass=float(np.mean(um)),topic_diversity=len(set(w for topic in topic_lists for w in topic))/(10*len(words)),outliers=manifest['outliers'],outlier_ratio=manifest['outlier_ratio'],coherence_seconds=time.perf_counter()-start,coherence_corpus='Identical Stage 1 LDA tokens and full corpus used for Stage 2; outlier topic excluded',ctfidf_reduce_frequent_words=True)
        csv('bertopic_coherence_per_topic.csv',pd.DataFrame([dict(topic_id=t,c_v=float(c),u_mass=float(u)) for t,c,u in zip(words,cv,um)]))
        metrics['cache_fingerprint']=metrics_identity
        write_json(metrics_path,metrics)
    all_stability=[]
    for seed in CFG['bertopic']['stability_seeds']:
        other,other_manifest=fit(seed,docs,embeddings,dictionary)
        outcome,matches=stability(words,topwords(other),labels,np.asarray(other.topics_))
        all_stability.append(dict(seed=seed,**outcome))
        csv(f'bertopic_stability_matches_seed{seed}.csv',pd.DataFrame(matches))
        del other
    write_json(REPORT/'bertopic_stability.json',all_stability)
    lda=read_json(ROOT/'outputs/reports/stage2/stage2_summary.json')
    comparison=pd.DataFrame([dict(model='LDA K=20 seed42',topics=20,**{k:lda['selected_metrics'][k] for k in ['c_v','u_mass','topic_diversity']},outlier_ratio=0.,stability=lda['selected_metrics']['stability'],stability_runs=3),dict(model='BERTopic fine-tuned BGE seed42',**{k:metrics[k] for k in ['topics','c_v','u_mass','topic_diversity','outlier_ratio']},stability=all_stability[0]['hungarian_top10_jaccard_unmatched_zero'] if all_stability else None,stability_runs=1+len(all_stability))])
    comparison['stability_method']=['Fixed K, three LDA seeds/three pairs, Hungarian top10 Jaccard mean','Two UMAP seeds, fixed fine-tuned embeddings; rectangular Hungarian top10 Jaccard, unmatched=0']
    comparison['coherence_reference']='Full frozen Stage 1 LDA tokens; top10, C_v window110; BERTopic outlier topic excluded'
    csv('lda_vs_bertopic.csv',comparison)
    topic_ids=list(words)
    # Similarity in the semantic embedding space, excluding outliers.
    centroids=np.stack([np.mean(embeddings[labels==t],axis=0) for t in topic_ids])
    similarity=cosine_similarity(centroids)
    sim_table=pd.DataFrame(similarity,columns=[f'topic_{t}' for t in topic_ids]);sim_table.insert(0,'topic_id',topic_ids)
    csv('bertopic_similarity.csv',sim_table)
    if projection_path.exists():projection=np.load(projection_path)
    else:
        with parallel_backend('sequential',n_jobs=1):
            projection=UMAP(**projection_settings).fit_transform(embeddings)
        np.save(projection_path,projection)
        write_json(projection_marker,dict(cache_fingerprint=projection_identity,projection_sha256=digest(projection_path),settings=projection_settings))
    scatter=main[['doc_id','source']].copy();scatter['topic_id']=labels;scatter['x']=projection[:,0];scatter['y']=projection[:,1]
    csv('bertopic_umap_2d.csv',scatter)
    verify_frozen();write_json(REPORT/'topic_success.json',dict(status='TRAINED_EXPORTED',topics=metrics['topics'],outlier_ratio=metrics['outlier_ratio'],seed=42,embedding_model_sha256=read_json(REPORT/'embedding_manifest.json')['model_sha256']))
    progress(f'BERTopic trained and exported: {metrics["topics"]} actual topics; C_v={metrics["c_v"]:.6f}, outliers={metrics["outlier_ratio"]:.4f}. LDA unchanged. Figures/tests/report pending.')

if __name__=='__main__':run()
