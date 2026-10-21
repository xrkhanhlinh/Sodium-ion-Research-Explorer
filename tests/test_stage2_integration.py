import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from gensim import corpora
from gensim.models import LdaModel
from PIL import Image
from src.data_io import sha256
from src.stage2 import match_stability,topic_sets

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'outputs/tables/stage2'
REPORTS=ROOT/'outputs/reports/stage2'


@pytest.fixture(scope='module')
def selection():
    return json.loads((REPORTS/'selected_model.json').read_text(encoding='utf-8'))


def test_frozen_inputs_and_separate_uncertain_pool():
    manifest=json.loads((REPORTS/'input_manifest.json').read_text(encoding='utf-8'))
    assert all(sha256(ROOT/p)==digest for p,digest in manifest['protected_hashes'].items())
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    tokens=pd.read_parquet(ROOT/'data/processed/lda_tokens.parquet')
    assert len(main)==13702 and main.doc_id.tolist()==tokens.doc_id.tolist()
    uncertain=pd.read_parquet(ROOT/'data/stage2/uncertain_sensitivity_pool.parquet')
    clean=pd.read_parquet(ROOT/'data/processed/combined_clean.parquet')
    assert len(uncertain)==6598 and uncertain.relevance.eq('UNCERTAIN').all()
    assert uncertain.doc_id.tolist()==clean.loc[clean.relevance.eq('UNCERTAIN'),'doc_id'].tolist()
    assert set(uncertain.doc_id).isdisjoint(main.doc_id)


def test_dictionary_and_bow_preserve_all_stage1_tokens():
    tokens=pd.read_parquet(ROOT/'data/processed/lda_tokens.parquet')
    dictionary=corpora.Dictionary.load(str(ROOT/'models/lda/dictionary.gensim'))
    assert set(dictionary.token2id)==set(w for doc in tokens.tokens for w in doc)
    bow=corpora.MmCorpus(str(ROOT/'data/stage2/corpus_bow.mm'))
    assert bow.num_docs==len(tokens) and bow.num_terms==len(dictionary)
    assert sum(sum(count for _,count in doc) for doc in bow)==tokens.token_count.sum()


def test_actual_runs_and_saved_models(selection):
    baseline=pd.read_csv(TABLES/'lda_base_metrics.csv')
    assert set(baseline.k)=={10,15,20,25,30} and len(baseline)==5
    runs=pd.read_csv(TABLES/'lda_all_run_metrics.csv')
    assert len(runs)==9 and not runs.duplicated(['k','seed']).any()
    assert np.isfinite(runs[['c_v','u_mass','topic_diversity','training_seconds']]).all().all()
    assert runs.topic_diversity.between(0,1).all() and runs.training_seconds.gt(0).all()
    for row in runs.itertuples():
        folder=ROOT/f'models/lda/k{row.k:02d}_seed{row.seed}'
        state=json.loads((folder/'training_success.json').read_text(encoding='utf-8'))
        assert state['documents']==13702
        assert all(sha256(folder/a['file'])==a['sha256'] for a in state['model_files'])
        model=LdaModel.load(str(folder/'model.gensim'))
        assert model.num_topics==row.k
        assert np.isfinite(model.get_topics()).all() and np.allclose(model.get_topics().sum(axis=1),1,atol=1e-5)


def test_stability_and_selection_recomputed(selection):
    comparison=pd.read_csv(TABLES/'lda_comparison.csv')
    pairs=pd.read_csv(TABLES/'lda_stability_pairs.csv')
    assert len(pairs)==6 and len(comparison.dropna(subset=['stability']))==2
    assert pairs.mean_matched_jaccard.between(0,1).all()
    for k in selection['promising']:
        sets={seed:topic_sets(LdaModel.load(str(ROOT/f'models/lda/k{k:02d}_seed{seed}/model.gensim'))) for seed in selection['stability_seeds']}
        for a,b in itertools.combinations(sets,2):
            actual,_=match_stability(sets[a],sets[b])
            saved=pairs[(pairs.k==k)&(pairs.seed_a==a)&(pairs.seed_b==b)].iloc[0]
            assert abs(actual-saved.mean_matched_jaccard)<1e-12
    s=selection['policy']
    for metric in ['c_v','u_mass']:
        low,high=comparison[metric].min(),comparison[metric].max()
        values=np.ones(len(comparison)) if high==low else (comparison[metric]-low)/(high-low)
        assert np.allclose(values,comparison[metric+'_normalized'])
    scores=s['cv_minmax_weight']*comparison.c_v_normalized+s['umass_minmax_weight']*comparison.u_mass_normalized+s['diversity_weight']*comparison.topic_diversity+s['stability_weight']*comparison.stability
    assert np.allclose(scores.dropna(),comparison.selection_score.dropna())
    best=comparison.dropna(subset=['selection_score']).sort_values(['selection_score','k'],ascending=[False,True]).iloc[0]
    assert selection['selected_k']==int(best.k) and selection['seed']==42


def test_document_vectors_counts_and_inference(selection):
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet')
    vectors=pd.read_parquet(ROOT/'data/stage2/lda_topic_vectors.parquet')
    matrix=np.load(ROOT/'data/stage2/lda_topic_vectors.npy')
    assert vectors.doc_id.tolist()==main.doc_id.tolist()
    assert matrix.shape==(13702,selection['selected_k'])
    assert np.array_equal(matrix,vectors.filter(regex=r'^topic_').to_numpy())
    assert np.isfinite(matrix).all() and (matrix>=0).all()
    assert np.allclose(matrix.sum(axis=1),1,atol=2e-5)
    summary=pd.read_csv(TABLES/'topic_summary.csv')
    assert len(summary)==selection['selected_k'] and summary.dominant_documents.sum()==13702
    assert np.isclose(summary.soft_document_mass.sum(),13702,atol=.02)
    assignments=pd.read_csv(TABLES/'document_topic_assignments.csv')
    assert np.array_equal(assignments.dominant_topic,matrix.argmax(axis=1))
    model=LdaModel.load(str(ROOT/selection['selected_model']))
    dictionary=corpora.Dictionary.load(str(ROOT/selection['dictionary']))
    tokens=pd.read_parquet(ROOT/'data/processed/lda_tokens.parquet').tokens.iloc[0]
    prediction=np.array([p for t,p in model.get_document_topics(dictionary.doc2bow(tokens),minimum_probability=0)])
    assert np.allclose(prediction,matrix[0],atol=1e-5)


def test_topic_keywords_representatives_trends(selection):
    k=selection['selected_k']
    keywords=pd.read_csv(TABLES/'topic_keywords.csv',keep_default_na=False)
    assert len(keywords)==10*k and keywords.groupby('topic_id').size().eq(10).all()
    assert keywords.weight.gt(0).all()
    model=LdaModel.load(str(ROOT/selection['selected_model']))
    for topic,group in keywords.groupby('topic_id'):
        expected=model.show_topic(topic,topn=10)
        assert group.sort_values('rank').token.tolist()==[word for word,weight in expected]
        assert np.allclose(group.sort_values('rank').weight,[weight for word,weight in expected],atol=1e-9)
    reps=pd.read_csv(TABLES/'topic_representative_documents.csv',keep_default_na=False)
    main=pd.read_parquet(ROOT/'data/processed/main_corpus.parquet').set_index('doc_id')
    assert reps.groupby('topic_id').size().between(1,3).all()
    assert set(reps.doc_id).issubset(main.index)
    assert all(main.at[r.doc_id,'title']==r.title for r in reps.itertuples())
    trends=pd.read_csv(TABLES/'topic_year_trends.csv')
    assert np.allclose(trends.groupby(['source','year']).topic_share.sum(),1,atol=2e-5)
    assert trends.dominant_documents.sum()==13702
    distribution=pd.read_csv(TABLES/'topic_source_distribution.csv')
    assert np.allclose(distribution.groupby('source').share_of_source_documents.sum(),1,atol=2e-5)


def test_eda_and_figures():
    summary=pd.read_csv(TABLES/'eda_source_summary.csv')
    assert summary.documents.sum()==13702 and summary.total_tokens.sum()==1725417
    ngrams=pd.read_csv(TABLES/'eda_ngrams.csv',keep_default_na=False)
    assert set(ngrams.n)=={2,3} and ngrams.document_share.between(0,1).all()
    terms=pd.read_csv(TABLES/'eda_term_frequencies.csv',keep_default_na=False)
    assert terms.document_share.between(0,1).all()
    manifest=json.loads((REPORTS/'figure_manifest.json').read_text(encoding='utf-8'))
    assert len(manifest)>=10
    for item in manifest:
        with Image.open(ROOT/item['figure']) as image:
            assert min(image.info['dpi'])>=299
            image.verify()
        for table in item['tables']:
            assert len(pd.read_csv(ROOT/table,keep_default_na=False))>0
    runtime=json.loads((REPORTS/'runtime_validation.json').read_text(encoding='utf-8'))
    assert runtime['status']=='PASS' and runtime['protected_files_unchanged']
