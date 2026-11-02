"""Bounded completion of existing topic/search artifacts; no encoder training."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ.setdefault(_key,'4')
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from .final_system_artifacts import ROOT, corpus, sha, read_json, vector_spec
from .model_comparison_io import write_json

OUT = ROOT/'outputs/research_completion_v1'
AUDIT = ROOT/'outputs/batterybert_topic_audit_20261022_v1'
TOPIC_NAME = 'BatteryBERT KMeans (pretrained)'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    main = corpus()
    inputs = [ROOT/'data/processed/main_corpus.parquet',ROOT/'data/processed/lda_tokens.parquet',
        AUDIT/'bertopic_final.pickle',AUDIT/'topic_assignments.csv',
        ROOT/'outputs/model_comparison_v2/batterybert_pretrained/document_embeddings.npy',
        ROOT/'outputs/final_system/tfidf/matrix.npz',ROOT/'outputs/final_system/tfidf/vectorizer.pkl',
        ROOT/'src/search_v2.py',ROOT/'configs/search_v2.json']
    identity = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    marker = OUT/'inputs.json'
    if marker.exists() and read_json(marker)!=identity:
        raise ValueError('Frozen completion inputs changed; create a new version')
    write_json(marker,identity)
    dest = OUT/'topics';dest.mkdir(exist_ok=True)
    # Initial preparation uses the existing reviewed labels; representation-only
    # optimization preserves every assignment and these evidence-based labels.
    assignments = pd.read_csv(AUDIT/'topic_assignments.csv')
    if assignments.doc_id.tolist()!=main.doc_id.tolist() or set(assignments.topic_id)!=set(range(10)):
        raise ValueError('KMeans topic IDs/count/order mismatch')
    if not (dest/'topic_assignments.csv').exists():assignments.to_csv(dest/'topic_assignments.csv',index=False)
    from .final_search import build_tfidf
    build_tfidf()  # Validated reuse; refuses a different pre-existing artifact.
    bge=vector_spec('bge_pretrained')
    write_json(OUT/'bge_validation.json',dict(status='PASS',documents=len(bge['ids']),dimension=bge['dimension'],
        vectors_sha256=bge['embeddings_sha256'],weights_sha256=bge['weights_sha256'],ids_aligned=True))
    queries = pd.read_parquet(ROOT/'data/stage4/queries.parquet')
    # Exactly the first20 existing non-edge queries; IDs/text are unchanged.
    selected = queries[~queries.is_edge_case].head(20).copy()
    if len(selected)!=20 or not selected.query_id.is_unique:raise ValueError('Expected20 frozen queries')
    if (OUT/'queries.csv').exists():
        old=pd.read_csv(OUT/'queries.csv')
        if old[['query_id','query']].to_dict('records')!=selected[['query_id','query']].to_dict('records'):
            raise ValueError('Evaluation query set changed')
    else:selected.to_csv(OUT/'queries.csv',index=False)
    write_json(OUT/'audit.json',dict(documents=len(main),unique_ids=main.doc_id.is_unique,
        active_topic_model='Existing KMeans10 on pretrained BatteryBERT, no PCA/UMAP',
        encoder=read_json(ROOT/'outputs/model_comparison_v2/batterybert_pretrained/embedding_manifest.json'),
        search_before='TF-IDF, V2 BM25+BGE+RRF+MiniLM, V3 BM25+fine-tuned BatteryBERT+RRF+MiniLM',
        missing_before=['Pretrained BatteryBERT standalone retrieval','TF-IDF + pretrained BatteryBERT RRF','0..2 annotation protocol','KMeans topics mounted in web'],
        corpus_changed=False,training_invoked=False,embeddings_regenerated=False))
    print('PREPARE PASS:20 unchanged queries,13702 aligned IDs, frozen TF-IDF reused',flush=True)


def index():
    from .final_system_vectors import VectorDatabase
    topics=pd.read_csv(OUT/'topics/topic_assignments.csv')
    db=VectorDatabase(OUT/'qdrant',artifact_dir=OUT,topics=topics)
    try:
        db.build('batterybert_pretrained')
        db.verify()
    finally:db.close()


def optimize_topics():
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from .model_comparison import corpus as load_tokens
    from .batterybert_topic_tools import (BOUNDARY, ScientificCountVectorizer, repair_phrase_placeholders,
        phrase_reference, centroid_scores, GENERIC)
    import yaml
    main,texts=load_tokens()
    phrases=yaml.safe_load((ROOT/'configs/config.yaml').read_text(encoding='utf-8'))['preprocessing']['domain_phrases']
    texts,repairs=repair_phrase_placeholders(texts,phrases)
    docs=[' '.join(t)+' '+BOUNDARY for t in texts]
    spec=vector_spec('batterybert_pretrained');vectors=np.load(spec['matrix'],mmap_mode='r')
    labels=pd.read_csv(AUDIT/'topic_assignments.csv').topic_id.to_numpy()
    stop=read_json(AUDIT/'stopwords.json')
    vocabulary=read_json(AUDIT/'vocabulary_repaired_2.json')
    dest=OUT/'topics';dest.mkdir(exist_ok=True)
    all_keywords={}
    all_keywords['before']=pd.read_csv(AUDIT/'experiments/kmeans10_full_repaired/keywords.csv')
    for bm25 in (False,True):
        name='ctfidf_bm25_'+str(int(bm25))
        folder=dest/name;folder.mkdir(exist_ok=True)
        if (folder/'keywords.csv').exists():
            all_keywords[name]=pd.read_csv(folder/'keywords.csv');continue
        model=BERTopic.load(str(AUDIT/'bertopic_final.pickle'))
        before=np.asarray(model.topics_).copy()
        model.update_topics(docs,top_n_words=15,
            vectorizer_model=ScientificCountVectorizer(ngram_range=(1,2),lowercase=False,stop_words=stop,
                vocabulary=vocabulary,min_df=1),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True,bm25_weighting=bm25))
        if not np.array_equal(before,labels) or not np.array_equal(before,model.topics_):
            raise ValueError('Representation update changed a KMeans membership')
        rows=[dict(topic_id=int(t),rank=i,token=w,weight=float(s)) for t in range(10)
            for i,(w,s) in enumerate(model.get_topic(t)[:15],1) if w and s>0]
        frame=pd.DataFrame(rows)
        if not frame.groupby('topic_id').size().ge(10).all():raise ValueError('Fewer than10 valid keywords')
        frame.to_csv(folder/'keywords.csv',index=False)
        model.save(str(folder/'model.pickle'),serialization='pickle')
        write_json(folder/'config.json',dict(ngram_range=[1,2],top_n_words=15,reduce_frequent_words=True,
            bm25_weighting=bm25,assignments_unchanged=True,vocabulary_source=str((AUDIT/'vocabulary_repaired_2.json').relative_to(ROOT)),
            stopwords_source=str((AUDIT/'stopwords.json').relative_to(ROOT)),clustering_fit=False))
        all_keywords[name]=frame
        del model
    features={w for frame in all_keywords.values() for w in frame.loc[frame['rank']<=10,'token']}
    reference=list(phrase_reference(texts,features));dictionary=Dictionary(reference)
    bow=[dictionary.doc2bow(t) for t in reference]
    rows=[]
    for name,keys in all_keywords.items():
        top=keys[keys['rank']<=10]
        words=[[w.replace(' ','\u2581') for w in g.sort_values('rank').token] for _,g in top.groupby('topic_id')]
        if len(words)!=10 or any(len(w)!=10 or any(x not in dictionary.token2id for x in w) for w in words):
            raise ValueError('Coherence reference lacks complete top10 keyword coverage')
        cv=CoherenceModel(topics=words,texts=reference,dictionary=dictionary,coherence='c_v',window_size=110,topn=10,processes=1).get_coherence()
        um=CoherenceModel(topics=words,corpus=bow,dictionary=dictionary,coherence='u_mass',topn=10,processes=1).get_coherence()
        rows.append(dict(version=name,c_v=float(cv),u_mass=float(um),keyword_diversity_at_10=top.token.nunique()/100,
            generic_keyword_slots=int(top.token.isin(GENERIC).sum()),multiword_keywords=int(top.token.str.contains(' ').sum()),
            topics=10,outliers=0,largest_topic_share=float(pd.Series(labels).value_counts().max()/len(labels)),
            assignments_unchanged=True,coherence_topics=10))
        print(name,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(dest/'topic_evaluation.csv',index=False)
    # BM25 is a controlled representation candidate, not automatic metric selection.
    # Selection follows inspection of actual keywords and unchanged representatives.
    write_json(dest/'representation_status.json',dict(status='CANDIDATES_EVALUATED',selection_pending=True))
    scores,redundancy=centroid_scores(vectors,labels)
    centers=np.asarray([vectors[labels==t].mean(axis=0) for t in range(10)])
    centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    similarity=centers@centers.T
    pd.DataFrame(similarity,index=range(10),columns=range(10)).rename_axis('topic_id').to_csv(dest/'topic_similarity.csv')
    pd.DataFrame([dict(topic_a=a,topic_b=b,centroid_cosine=float(similarity[a,b]),
        conclusion='Centroid diagnostic only; inspect representative content before merging') for a,b in ((1,5),(2,3))]).to_csv(dest/'overlap_diagnostics.csv',index=False)
    reps=[]
    for topic in range(10):
        ids=np.flatnonzero(labels==topic);ordered=ids[np.argsort(-scores[ids],kind='stable')][:5]
        for rank,i in enumerate(ordered,1):
            row=main.iloc[i]
            reps.append(dict(topic_id=topic,rank=rank,doc_id=row.doc_id,title=row.title,abstract=row.abstract,
                source=row.source,year=row.year,centroid_cosine=float(scores[i])))
    pd.DataFrame(reps).to_csv(dest/'topic_representative_documents.csv',index=False)
    write_json(dest/'coherence_protocol.json',dict(reference='Repaired frozen Stage1 tokens + exact observed top10 phrases for all three variants',
        phrase_encoding='U+2581',c_v_window=110,topn=10,processes=1,full_topic_coverage=True,
        old_scores_not_mixed=True,independent_scientific_validation=False))


def select_topics(name):
    import shutil
    dest=OUT/'topics';folder=dest/name
    if name not in ('ctfidf_bm25_0','ctfidf_bm25_1'):raise ValueError('Unknown representation')
    keys=pd.read_csv(folder/'keywords.csv');keys.to_csv(dest/'topic_keywords.csv',index=False)
    labels=pd.read_csv(AUDIT/'topic_labels.csv')
    # Keep previously reviewed names unless new representative evidence warrants change.
    labels.to_csv(dest/'topic_labels.csv',index=False)
    summary=pd.read_csv(AUDIT/'topic_distribution.csv')
    summary=summary.drop(columns=['topic_label'],errors='ignore').merge(labels[['topic_id','topic_label']],on='topic_id',validate='one_to_one')
    summary.to_csv(dest/'topic_distribution.csv',index=False)
    assignments=pd.read_csv(AUDIT/'topic_assignments.csv').drop(columns=['topic_label'],errors='ignore')
    assignments['topic_label']=assignments.topic_id.map(labels.set_index('topic_id').topic_label)
    assignments.to_csv(dest/'topic_assignments.csv',index=False)
    assignments.groupby(['topic_id','source']).size().rename('documents').reset_index().to_csv(dest/'topic_sources.csv',index=False)
    assignments.groupby(['topic_id','year'],dropna=False).size().rename('documents').reset_index().to_csv(dest/'topic_years.csv',index=False)
    shutil.copy2(folder/'model.pickle',dest/'model.pickle')
    write_json(dest/'representation_status.json',dict(status='SELECTED_PROVISIONAL',selected=name,
        assignments_unchanged=True,clustering_rerun=False,labels='Tentative; prior document reviews and new representative exports retained'))
    from scripts.audit_batterybert_topics import render_topics
    plotfolder=dest/'plot_input';plotfolder.mkdir(exist_ok=True)
    keys[keys['rank']<=10].to_csv(plotfolder/'keywords.csv',index=False)
    summary.to_csv(plotfolder/'topic_distribution.csv',index=False)
    render_topics(plotfolder,dest/'topic_keywords_10_topics.png',labels.set_index('topic_id').topic_label.to_dict(),
        subtitle='Fixed existing memberships; representation-only update; topic names remain tentative')


def topic_data():
    dest=OUT/'topics'
    result={key:pd.read_csv(dest/file) for key,file in dict(summary='topic_distribution.csv',keywords='topic_keywords.csv',
        representatives='topic_representative_documents.csv',assignments='topic_assignments.csv',
        sources='topic_sources.csv',years='topic_years.csv').items()}
    result['summary']=result['summary'].rename(columns={'corpus_share':'document_share'})
    result['labels']=pd.read_csv(dest/'topic_labels.csv').set_index('topic_id').topic_label.to_dict()
    status=read_json(dest/'representation_status.json')
    row=pd.read_csv(dest/'topic_evaluation.csv').set_index('version').loc[status['selected']]
    result['metrics']=dict(topics=10,c_v=float(row.c_v),topic_diversity=float(row.keyword_diversity_at_10),outlier_ratio=0.)
    return result
