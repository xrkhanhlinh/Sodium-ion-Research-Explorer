"""Frozen source-document recovery benchmark, distinct from relevance judgments.

Legacy formulas are replayed read-only with exact NumPy cosine; the adapter must
match historical rankings before it can label a synthetic run as comparable.
"""
import json
import pickle
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .ai_judge import ROOT, SYNTHETIC_SCHEMA, SYNTHETIC_SYSTEM, fingerprint, response_json, write_json, read_json
from .ai_retrieval_evaluation import METHODS


def select_documents(corpus, seed=42, each=100):
    eligible = corpus[corpus.abstract.fillna('').str.strip().ne('')].sort_values('doc_id')
    # Document sampling, not family sampling; the report states that dependency.
    chosen = []
    for source in ('Patent', 'Publication'):
        part = eligible[eligible.source.eq(source)]
        if len(part) < each:
            raise ValueError('Not enough eligible abstracts')
        chosen.append(part.sample(each, random_state=seed))
    selected = pd.concat(chosen, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    selected.insert(0, 'synthetic_id', ['S%03d' % (i+1) for i in range(len(selected))])
    return selected[['synthetic_id', 'doc_id', 'source', 'title', 'abstract', 'split_group_id']]


def validate_synthetic(value, row):
    if not isinstance(value, dict) or set(value) != {'status','query','reason'}:
        raise ValueError('Invalid synthetic JSON')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('Missing synthetic generation reason')
    if value['status'] == 'UNRESOLVED' and value['query'] == '':
        return value
    if value['status'] != 'OK' or not isinstance(value['query'], str):
        raise ValueError('Invalid generation status/query')
    query = value['query'].strip()
    if not 10 <= len(query.split()) <= 35 or len(query) > 400:
        raise ValueError('Synthetic query must be 10..35 words and <=400 chars')
    normalize = lambda s: re.sub(r'[^a-z0-9]+',' ',s.casefold()).strip()
    q, title = normalize(query), normalize(row['title'])
    if str(row['doc_id']).casefold() in query.casefold() or (title and (title in q or q in title)):
        raise ValueError('Query leaks ID or copies the title')
    # Reject long copied title phrases; chemical names alone can be unavoidable.
    words = title.split()
    if any(' '.join(words[i:i+8]) in q for i in range(max(0,len(words)-7))):
        raise ValueError('Query copies an eight-word title phrase')
    tokenizer = _tokenizer()
    from .search_v2 import normalize
    from .text_processing import normalize_text
    prefix = read_json(ROOT/'outputs/reports/stage3/inference_contract.json')['query_prefix']
    if max(len(tokenizer(prefix+normalize(query), truncation=False)['input_ids']),
           len(tokenizer(prefix+normalize_text(query), truncation=False)['input_ids'])) > 128:
        raise ValueError('Synthetic query exceeds frozen neural context')
    return dict(value, query=query)


def _tokenizer():
    # A module singleton avoids loading the tokenizer for each generated query.
    if not hasattr(_tokenizer, 'value'):
        from transformers import AutoTokenizer
        _tokenizer.value = AutoTokenizer.from_pretrained(str(ROOT/'models/bge_pretrained'),local_files_only=True)
    return _tokenizer.value


def generate_query(client, row, config):
    # NO title, ID, rank, method or retrieval result sent to the generator.
    payload = {'abstract': row['abstract']}
    saved = None
    for attempt in range(6):
        system = SYNTHETIC_SYSTEM
        if attempt:
            system += ('\nFormatting/overlap repair attempt '+str(attempt)+
                       ': produce a fresh paraphrase. Do not copy eight consecutive words from the Abstract. '
                       'Keep 10..35 words. No search has been run; do not optimize for retrieval scores.')
        if attempt>=3:
            system += (' Reorganize the wording around another research aspect explicitly present in the Abstract. '
                       'Break up long material descriptions instead of reproducing their noun phrases; '
                       'reorder ingredient lists and use different grammatical structure. Do not invent findings or chemistry.')
        saved = client.generate('SYNTHETIC_GENERATION', config['generator'], system, payload, SYNTHETIC_SCHEMA)
        try:
            value = validate_synthetic(response_json(saved), row)
            break
        except (ValueError,KeyError,TypeError) as exc:
            write_json(client.output/'synthetic/rejections'/(row['synthetic_id']+'_'+str(attempt)+'.json'),
                       {'request_sha256':saved['request_sha256'],'reason':str(exc),'attempt':attempt,
                        'status':'REJECTED_BEFORE_RETRIEVAL'})
    else:
        return {'status':'INVALID_RESPONSE','query':'','reason':'Six validation attempts failed before any retrieval',
                'synthetic_id':row['synthetic_id'],'request_sha256':saved['request_sha256'],
                'model_id':saved['requested_model'],'model_version':saved['server_model_version'],
                'abstract_sha256':fingerprint(row['abstract']),'query_sha256':fingerprint('')}
    return dict(value, synthetic_id=row['synthetic_id'], request_sha256=saved['request_sha256'],
                model_id=saved['requested_model'], model_version=saved['server_model_version'],
                abstract_sha256=fingerprint(row['abstract']), query_sha256=fingerprint(value['query']))


def recovery_metrics(ids, expected):
    if len(ids) > 10 or len(ids) != len(set(ids)):
        raise ValueError('Need unique Top10 IDs')
    rank = ids.index(expected)+1 if expected in ids else None
    return {'Hit_at_1': int(rank == 1), 'Hit_at_5': int(rank is not None and rank <= 5),
            'Hit_at_10': int(rank is not None), 'MRR_at_10': 1/rank if rank else 0.}


def recovery_bootstrap(per_query, selection, repetitions=2000, seed=42):
    metrics=['Hit_at_1','Hit_at_5','Hit_at_10','MRR_at_10']
    methods=[m for m in METHODS if m in set(per_query.method)]
    ids=sorted(set(per_query.synthetic_id))
    arrays={}
    for method in methods:
        frame=per_query[per_query.method.eq(method)].set_index('synthetic_id')
        if frame.index.duplicated().any() or set(frame.index)!=set(ids):
            raise ValueError('Synthetic bootstrap requires common complete query/method coverage')
        arrays[method]=frame.loc[ids,metrics].to_numpy(dtype=float)
        if not np.isfinite(arrays[method]).all():raise ValueError('Missing synthetic metric')
    groups=selection.set_index('synthetic_id').loc[ids,'split_group_id'].astype(str).to_numpy()
    unique,inverse=np.unique(groups,return_inverse=True)
    if len(unique)<2:return pd.DataFrame(),pd.DataFrame()
    rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(unique),size=(repetitions,len(unique)))
    sizes=np.bincount(inverse).astype(float)
    boots={}
    summary=[];pairs=[]
    for method,array in arrays.items():
        sums=np.zeros((len(unique),len(metrics)))
        np.add.at(sums,inverse,array)
        boot=sums[samples].sum(axis=1)/sizes[samples].sum(axis=1)[:,None]
        boots[method]=boot
        for index,metric in enumerate(metrics):
            low,high=np.quantile(boot[:,index],[.025,.975])
            summary.append(dict(method=method,metric=metric,mean=float(array[:,index].mean()),
                                ci95_low=float(low),ci95_high=float(high),queries=len(ids),clusters=len(unique)))
    for left,right in combinations(methods,2):
        boot=boots[left]-boots[right]
        for index,metric in enumerate(metrics):
            low,high=np.quantile(boot[:,index],[.025,.975])
            pairs.append(dict(left=left,right=right,metric=metric,
                mean_delta=float((arrays[left]-arrays[right])[:,index].mean()),
                ci95_low=float(low),ci95_high=float(high),queries=len(ids),clusters=len(unique)))
    return pd.DataFrame(summary),pd.DataFrame(pairs)


class LegacyReplay:
    def __init__(self, engine, dense_encoder=None):
        import nltk, yaml
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel
        nltk.data.path.insert(0, str(ROOT/'.cache/nltk_data'))
        self.engine = engine
        self.dense_encoder=dense_encoder or (lambda query:engine.encode(query,'finetuned'))
        self.config = read_json(ROOT/'configs/stage4.json')
        self.preprocessing = yaml.safe_load((ROOT/'configs/config.yaml').read_text(encoding='utf-8'))['preprocessing']
        self.dictionary = Dictionary.load(str(ROOT/'models/lda/dictionary.gensim'))
        self.lda = LdaModel.load(str(ROOT/'models/lda/k20_seed42/model.gensim'))
        with (ROOT/'data/stage4/bm25.pkl').open('rb') as stream:
            self.bm25 = pickle.load(stream)  # trusted, frozen local artifact
        self.ids = pd.read_parquet(ROOT/'data/processed/lda_tokens.parquet').doc_id.tolist()
        order = pd.read_parquet(ROOT/'data/stage2/lda_topic_vectors.parquet').doc_id.tolist()
        if self.ids != engine.ids or order != self.ids:
            raise ValueError('Legacy vector/token order mismatch')
        array = np.load(ROOT/'data/stage2/lda_topic_vectors.npy')
        self.lda_vectors = array/np.linalg.norm(array,axis=1,keepdims=True)
        self.centroids = np.load(ROOT/'data/stage4/topic_centroids.npy')
        self.dense_norms=None
        self.topics = pd.read_parquet(ROOT/'data/stage3/bertopic_assignments.parquet').set_index('doc_id').loc[self.ids,'topic_id'].to_dict()

    def tokens(self, query):
        from .text_processing import lda_tokens, normalize_text
        return lda_tokens(normalize_text(query), self.preprocessing['domain_phrases'], self.preprocessing['technical_stopwords'])

    def sorted_scores(self, values):
        return sorted(zip(self.ids, map(float,values)),key=lambda pair:(-pair[1],pair[0]))[:self.config['candidate_depth']]

    def rank(self, query, method):
        from .search_v2 import reciprocal_rank_fusion
        if method == 'BM25':
            tokens = self.tokens(query)
            return self.sorted_scores(self.bm25.get_scores(tokens)) if any(t in self.bm25.idf for t in tokens) else []
        first, second = [], []
        if method in ('SE1', 'SE3'):
            bow = self.dictionary.doc2bow(self.tokens(query))
            if bow:
                self.lda.random_state = np.random.RandomState(self.config['query_seed'])
                with threadpool_limits(limits=1):
                    vec = np.array([p for _,p in self.lda.get_document_topics(bow,minimum_probability=0)],dtype='float32')
                first = self.sorted_scores(self.lda_vectors @ (vec/np.linalg.norm(vec)))
        if method in ('SE2', 'SE3'):
            vector = self.dense_encoder(query)
            with threadpool_limits(limits=4):
                matrix=self.engine._matrix('finetuned')
                if self.dense_norms is None:self.dense_norms=np.linalg.norm(matrix,axis=1)
                second = self.sorted_scores((matrix @ vector)/(self.dense_norms*np.linalg.norm(vector)))
            affinity = self.centroids @ vector
            second = [(doc,score+(self.config['topic_weight']*float(affinity[self.topics[doc]]) if self.topics[doc]!=-1 else 0.)) for doc,score in second]
            second.sort(key=lambda pair:(-pair[1],pair[0]))
        return first if method=='SE1' else second if method=='SE2' else reciprocal_rank_fusion([first,second], self.config['rrf_k'])


def readonly_engine():
    from .search_v2 import SearchEngineV2, CACHE, digest
    class ReadOnlyEngine(SearchEngineV2):
        def _load_lexical(self):
            marker, path = CACHE/'lexical_manifest.json', CACHE/'lexical.pkl'
            identity = dict(corpus_sha256=digest(ROOT/self.config['corpus']),
                            tokenizer_source_sha256=digest(ROOT/'src/search_v2.py'),
                            k1=self.config['bm25_k1'],b=self.config['bm25_b'])
            if not marker.exists() or read_json(marker)['identity']!=identity or digest(path)!=read_json(marker)['sha256']:
                raise ValueError('Frozen lexical cache absent/mismatched; evaluation will not rebuild it')
            with path.open('rb') as stream:
                self.title_index,self.abstract_index=pickle.load(stream)
    return ReadOnlyEngine(use_query_cache=True)


def verify_legacy_parity(output):
    engine = readonly_engine()
    legacy = LegacyReplay(engine)
    rankings = pd.read_csv(ROOT/'outputs/search_upgrade_20261018/evaluation_14_methods/top5_fourteen_methods.csv',keep_default_na=False)
    queries = pd.read_csv(ROOT/'outputs/search_upgrade_20261018/queries.csv',keep_default_na=False)
    rows = []
    for row in queries.itertuples():
        for method in ('SE1','SE2','SE3','BM25'):
            expected=rankings[rankings.query_id.eq(row.query_id)&rankings.method.eq(method)].sort_values('rank').doc_id.tolist()
            actual=[doc for doc,_ in legacy.rank(row.query,method)[:5]]
            rows.append({'query_id':row.query_id,'method':method,'matches':actual==expected,
                         'expected':expected,'actual':actual})
    write_json(Path(output)/'legacy_parity.json', {'status':'PASS' if all(r['matches'] for r in rows) else 'FAIL',
                'backend':'Read-only NumPy exact cosine, historical formulas; no Qdrant opened', 'checks':rows})
    return all(row['matches'] for row in rows)


def retrieve_synthetic(output, selected, generations):
    output=Path(output)
    parity=read_json(output/'legacy_parity.json') if (output/'legacy_parity.json').exists() else {}
    if parity.get('status')!='PASS':
        raise ValueError('PENDING_LEGACY_PARITY: run --phase parity before synthetic retrieval')
    accepted=[r for r in generations if r['status']=='OK']
    if not accepted:
        return
    engine=readonly_engine()
    from .search_v2 import normalize
    from .text_processing import normalize_text
    original_encode=engine.encode
    query_vectors={}
    vecdir=output/'synthetic/query_vectors';vecdir.mkdir(parents=True,exist_ok=True)
    # Encode all queries per checkpoint once; keep only one BGE checkpoint in RAM.
    for kind in ('pretrained','finetuned'):
        for row in accepted:
            path=vecdir/(fingerprint({'kind':kind,'query':row['query']})+'.npy')
            if path.exists():
                vector=np.load(path)
            else:
                vector=original_encode(row['query'],kind)
                with path.with_suffix('.tmp').open('wb') as stream:np.save(stream,vector)
                path.with_suffix('.tmp').replace(path)
            if vector.shape!=(768,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1,atol=2e-5):
                raise ValueError('Invalid synthetic query vector')
            query_vectors[(kind,normalize(row['query']))]=vector
    legacy_vectors={}
    for row in accepted:
        text=normalize_text(row['query'])
        if text==normalize(row['query']):
            legacy_vectors[row['query']]=query_vectors[('finetuned',normalize(row['query']))]
            continue
        path=vecdir/(fingerprint({'kind':'legacy_finetuned_NFC','query':text})+'.npy')
        if path.exists():vector=np.load(path)
        else:
            # Old Stage4 uses NFC+subscript repair; preserve its treatment of
            # compatibility characters rather than silently using v2's NFKC.
            original_encode(row['query'],'finetuned')
            vector=engine.encoder.encode([engine.prefix+text],normalize_embeddings=True,show_progress_bar=False)[0]
            with path.with_suffix('.tmp').open('wb') as stream:np.save(stream,vector)
            path.with_suffix('.tmp').replace(path)
        if vector.shape!=(768,) or not np.isfinite(vector).all():raise ValueError('Invalid legacy query vector')
        legacy_vectors[row['query']]=vector
    engine.encode=lambda query,kind: query_vectors[('finetuned' if kind=='chunked' else kind,normalize(query))]
    legacy=LegacyReplay(engine,dense_encoder=lambda query:legacy_vectors[query])
    sources=selected.set_index('synthetic_id')
    runs=output/'synthetic/runs';runs.mkdir(parents=True,exist_ok=True)
    rows=[]
    for record in accepted:
        sid,query=record['synthetic_id'],record['query']
        for method in METHODS:
            path=runs/(sid+'_'+method+'.json')
            identity={'synthetic_id':sid,'query_sha256':fingerprint(query),'method':method}
            if path.exists():
                result=read_json(path)
                if result['identity']!=identity:raise ValueError('Synthetic query drift after retrieval')
                ids=result['ids']
            else:
                if method in ('SE1','SE2','SE3','BM25'):
                    ranking=legacy.rank(query,method)
                    ids=[doc for doc,_ in ranking[:10]]
                else:
                    ids=[r['doc_id'] for r in engine.search(query,method=method,top_k=10)['results']]
                write_json(path,{'identity':identity,'ids':ids,'metrics':recovery_metrics(ids,sources.loc[sid,'doc_id'])})
            rows.append(dict(synthetic_id=sid,method=method,source=sources.loc[sid,'source'],
                             **recovery_metrics(ids,sources.loc[sid,'doc_id'])))
        print('Synthetic retrieval checkpoint '+sid,flush=True)
    per=pd.DataFrame(rows)
    splits=pd.read_parquet(ROOT/'data/stage3/split_assignments.parquet',columns=['doc_id','split'])
    split_lookup=selected[['synthetic_id','doc_id']].merge(splits,on='doc_id',validate='one_to_one')
    per=per.merge(split_lookup[['synthetic_id','split']],on='synthetic_id',validate='many_to_one')
    per.to_csv(output/'synthetic/per_query.csv',index=False,encoding='utf-8-sig')
    metrics=['Hit_at_1','Hit_at_5','Hit_at_10','MRR_at_10']
    summary=per.groupby('method')[metrics].mean().reindex(METHODS).reset_index()
    summary['queries_evaluated']=len(accepted);summary['queries_planned']=len(selected)
    summary['status']='SYNTHETIC_SOURCE_RECOVERY_ONLY' if len(accepted)==len(selected) else 'PARTIAL_SYNTHETIC_SELECTED_QUERIES'
    summary.to_csv(output/'synthetic/summary.csv',index=False,encoding='utf-8-sig')
    by_source=per.groupby(['method','source'])[metrics].mean()
    by_source['queries']=per.groupby(['method','source']).size()
    by_source.to_csv(output/'synthetic/by_source.csv',encoding='utf-8-sig')
    by_split=per.groupby(['method','split'])[metrics].mean()
    by_split['queries']=per.groupby(['method','split']).size()
    by_split.to_csv(output/'synthetic/by_training_split.csv',encoding='utf-8-sig')
    ci,pairs=recovery_bootstrap(per,selected)
    ci.to_csv(output/'synthetic/confidence_intervals.csv',index=False,encoding='utf-8-sig')
    pairs.to_csv(output/'synthetic/paired_deltas.csv',index=False,encoding='utf-8-sig')
    write_json(output/'synthetic/metric_scope.json',{'metrics':'Exact source doc_id recovery, not relevance or corpus recall',
        'confidence_intervals':'Paired cluster bootstrap by split_group_id, 2000 repetitions, seed42; document-weighted means',
        'limitations':'Conditional on fixed generated queries/checkpoints; no generation-seed uncertainty. Document sampling may include families. Source proportions may vary under cluster resampling. Many comparisons exploratory, no multiplicity correction.'})
