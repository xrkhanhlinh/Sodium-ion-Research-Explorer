"""Pooled @10 relevance evaluation with strict independent-label provenance."""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from .final_system_artifacts import ROOT, OUT, read_json, sha, corpus
from .model_comparison_io import write_json

METHODS = ('TFIDF_COSINE','SEARCH_V2','SEARCH_V3')
METRICS = ('nDCG_at_10','Precision_at_10','MRR_at_10')
LABELS = ('relevance_grade','annotator_id','verifier_id','verification_reference','label_method','label_status')
PLACEHOLDERS = {'','nan','none','null','unknown','tbd','n/a','na'}


def keys(frame):
    frame = frame.copy()
    for field in ('query_id','doc_id'):
        if field not in frame or frame[field].isna().any(): raise ValueError('Missing ' + field)
        frame[field] = frame[field].map(str).str.strip()
        if frame[field].str.casefold().isin(PLACEHOLDERS).any(): raise ValueError('Placeholder ' + field)
    if frame.duplicated(['query_id','doc_id']).any(): raise ValueError('Duplicate query-document pairs')
    return frame


def validate_labels(frame, pool, max_grade=3):
    frame = keys(frame)
    if any(field not in frame for field in LABELS): raise ValueError('Missing relevance/provenance columns')
    valid_pairs = set(zip(pool.query_id,pool.doc_id))
    if set(zip(frame.query_id,frame.doc_id)) - valid_pairs: raise ValueError('Labels contain out-of-pool pairs')
    # Blank grades remain absent; an annotation template may be imported partly.
    present = frame.relevance_grade.notna() & frame.relevance_grade.map(str).str.strip().ne('')
    frame = frame[present].copy()
    if frame.relevance_grade.map(lambda x:isinstance(x,(bool,np.bool_))).any(): raise ValueError('Boolean grade')
    grades = pd.to_numeric(frame.relevance_grade,errors='raise')
    if max_grade not in (2,3): raise ValueError('Unsupported grade scale')
    if not np.isfinite(grades).all() or not grades.isin(list(range(max_grade+1))).all(): raise ValueError(f'Grades must be integers 0..{max_grade}')
    frame['relevance_grade'] = grades.astype(int)
    for field in LABELS[1:]:
        frame[field] = frame[field].fillna('').astype(str).str.strip()
        if frame[field].str.casefold().isin(PLACEHOLDERS).any(): raise ValueError('Missing ' + field)
    if not frame.label_status.eq('VERIFIED').all() or not frame.label_method.eq('HUMAN_INDEPENDENT').all():
        raise ValueError('Evaluated labels must be VERIFIED HUMAN_INDEPENDENT')
    if frame.annotator_id.eq(frame.verifier_id).any(): raise ValueError('Independent verifier must differ from annotator')
    return frame


def metrics_at_10(grades, pool_grades):
    for values in (grades,pool_grades):
        raw=np.asarray(values,dtype=object)
        if raw.ndim!=1 or any(isinstance(x,(bool,np.bool_)) for x in raw):
            raise ValueError('Grades must be one-dimensional integers, not booleans')
    g, pool = np.asarray(grades,dtype=float), np.asarray(pool_grades,dtype=float)
    if g.ndim != 1 or pool.ndim != 1 or len(g)>10: raise ValueError('Invalid @10 grade shape')
    if not np.isfinite(g).all() or not np.isfinite(pool).all() or not np.isin(g,[0,1,2,3]).all() or not np.isin(pool,[0,1,2,3]).all():
        raise ValueError('Invalid grade domain')
    if Counter(g)-Counter(pool): raise ValueError('Ranked grades must belong to judged pool')
    ideal = np.sort(pool)[::-1][:10]
    dcg = np.sum((2.**g-1)/np.log2(np.arange(len(g))+2))
    idcg = np.sum((2.**ideal-1)/np.log2(np.arange(len(ideal))+2))
    relevant = np.flatnonzero(g>=2)
    return dict(nDCG_at_10=float(dcg/idcg) if idcg else 0., Precision_at_10=float(np.sum(g>=2)/10),
                MRR_at_10=float(1/(relevant[0]+1)) if len(relevant) else 0.)


def audit_label_sources():
    candidates = sorted({p for root in (ROOT/'outputs',ROOT/'data') for pattern in ('*qrels*.csv','*annotator*.csv','*labels*.csv')
                         for p in root.rglob(pattern) if not p.is_relative_to(OUT)})
    rows = []
    for path in candidates:
        frame = pd.read_csv(path,keep_default_na=False)
        if 'relevance_grade' not in frame: continue
        grades = pd.to_numeric(frame.relevance_grade,errors='coerce')
        verified = frame.get('label_status',pd.Series('',index=frame.index)).eq('VERIFIED')
        independent = frame.get('label_method',pd.Series('',index=frame.index)).eq('HUMAN_INDEPENDENT')
        rows.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),pairs=len(frame),
            numeric_grades=int(grades.notna().sum()),verified_independent_candidates=int((verified&independent&grades.notna()).sum()),
            label_methods='; '.join(sorted(set(frame.get('label_method',pd.Series(dtype=str)).astype(str))))))
    pd.DataFrame(rows,columns=['path','sha256','pairs','numeric_grades','verified_independent_candidates','label_methods']).to_csv(OUT/'label_source_audit.csv',index=False)
    return rows


def create_pool():
    ranks = pd.read_csv(OUT/'rankings_top10.csv',keep_default_na=False)
    pool = ranks.drop_duplicates(['query_id','doc_id'])[['query_id','query','doc_id','title','abstract','source','year']].copy()
    # Retain the earlier common pool to reuse genuine old independent judgments
    # when available and avoid changing IDCG merely because a method was removed.
    old = pd.read_csv(ROOT/'outputs/search_upgrade_20261018/shared_pool_UNVERIFIED.csv',keep_default_na=False)
    available = [c for c in pool.columns if c in old]
    pool = pd.concat([pool,old[available]],ignore_index=True).drop_duplicates(['query_id','doc_id'])
    pool = keys(pool)
    queries = pd.read_parquet(ROOT/'data/stage4/queries.parquet')
    pool = pool[pool.query_id.isin(queries.query_id)].sort_values(['query_id','doc_id']).reset_index(drop=True)
    # Canonical query text always comes from the frozen benchmark.
    pool['query'] = pool.query_id.map(queries.set_index('query_id')['query'])
    meta=corpus().set_index('doc_id')
    if not pool.doc_id.isin(meta.index).all(): raise ValueError('Annotation pool contains IDs outside the frozen corpus')
    for column in ('title','abstract','source','year'):
        pool[column]=pool.doc_id.map(meta[column])
    pool.to_csv(OUT/'annotation_pool.csv',index=False)
    write_json(OUT/'annotation_pool_manifest.json',dict(pairs=len(pool),deduplicated=True,
        rankings_sha256=sha(OUT/'rankings_top10.csv'),
        previous_pool_sha256=sha(ROOT/'outputs/search_upgrade_20261018/shared_pool_UNVERIFIED.csv'),
        query_protocol_sha256=sha(ROOT/'data/stage4/queries.parquet'),
        corpus_sha256=sha(ROOT/'data/processed/main_corpus.parquet'),
        pool_sha256=sha(OUT/'annotation_pool.csv'),grade_generation='None; independent annotation required'))
    template = pool.copy()
    for field in (*LABELS,'reviewer_notes'): template[field] = ''
    template['label_status'] = 'UNVERIFIED'
    # Do not overwrite a user's imported or partially annotated template.
    if not (OUT/'annotation_template.csv').exists(): template.to_csv(OUT/'annotation_template.csv',index=False)
    audit_label_sources()
    return pool


def evaluate(labels_path=None, repetitions=2000):
    committed=OUT/'benchmark_success.json'
    if committed.exists():
        record=read_json(committed)
        if (sha(OUT/'rankings_top10.csv')!=record['rankings_sha256']
                or sha(OUT/'run_manifest.csv')!=record['runs_sha256']):
            raise ValueError('Committed retrieval outputs changed')
    pool_marker=OUT/'annotation_pool_manifest.json'
    if pool_marker.exists() and sha(OUT/'annotation_pool.csv')!=read_json(pool_marker)['pool_sha256']:
        raise ValueError('Committed annotation pool changed; import grades separately')
    pool = keys(pd.read_csv(OUT/'annotation_pool.csv',keep_default_na=False))
    ranks = pd.read_csv(OUT/'rankings_top10.csv',keep_default_na=False)
    queries = pd.read_parquet(ROOT/'data/stage4/queries.parquet')
    runs = pd.read_csv(OUT/'run_manifest.csv',keep_default_na=False)
    if runs.duplicated(['query_id','method']).any(): raise ValueError('Duplicate retrieval run')
    expected = {(q,m) for q in queries.query_id for m in METHODS}
    if set(zip(runs.query_id,runs.method)) != expected or not runs.status.eq('OK').all():
        raise ValueError('Incomplete query/method retrieval matrix; empty results need a successful run record')
    if set(ranks.method)-set(METHODS) or ranks.duplicated(['query_id','method','doc_id']).any():
        raise ValueError('Unknown method or duplicate ranked document')
    pool_pairs = set(zip(pool.query_id,pool.doc_id))
    if set(zip(ranks.query_id,ranks.doc_id))-pool_pairs: raise ValueError('Ranking outside pool')
    for (q,m), group in ranks.groupby(['query_id','method']):
        if sorted(group['rank']) != list(range(1,len(group)+1)) or len(group)>10: raise ValueError('Noncontiguous or invalid ranks')
    for run in runs.itertuples():
        if len(ranks[ranks.query_id.eq(run.query_id)&ranks.method.eq(run.method)]) != run.result_count:
            raise ValueError('Run count disagrees with ranking')
    if labels_path:
        labels = validate_labels(pd.read_csv(labels_path,keep_default_na=False),pool)
        label_source = str(Path(labels_path).resolve())
    else:
        labels = pd.DataFrame(columns=['query_id','doc_id',*LABELS])
        # Reuse only genuinely independent and VERIFIED old labels. AI and
        # synthetic labels remain an audited separate research artifact.
        for candidate in audit_label_sources():
            if candidate['verified_independent_candidates']:
                frame = pd.read_csv(ROOT/candidate['path'],keep_default_na=False)
                frame = frame[frame.label_status.eq('VERIFIED')&frame.label_method.eq('HUMAN_INDEPENDENT')]
                frame = frame[[pair in pool_pairs for pair in zip(frame.query_id,frame.doc_id)]]
                checked = validate_labels(frame,pool)
                labels = pd.concat([labels,checked],ignore_index=True)
        if labels.duplicated(['query_id','doc_id']).any():
            raise ValueError('Multiple old verified label versions; import a resolved version explicitly')
        label_source = 'Audited existing independent labels only'
    labels.to_csv(OUT/'accepted_independent_labels.csv',index=False)
    lookup = {(r.query_id,r.doc_id):int(r.relevance_grade) for r in labels.itertuples()}
    per_query, coverage = [], []
    for q in queries.itertuples():
        documents = pool.loc[pool.query_id.eq(q.query_id),'doc_id'].tolist()
        judged = sum((q.query_id,doc) in lookup for doc in documents)
        complete = bool(documents) and judged == len(documents) and not q.is_edge_case
        status = 'EXCLUDED_EDGE_CONTROL' if q.is_edge_case else 'COMPLETE_INDEPENDENT_POOL' if complete else 'PENDING_INDEPENDENT_LABELS'
        coverage.append(dict(query_id=q.query_id,pool_pairs=len(documents),independent_judged=judged,status=status))
        for method in METHODS:
            row = dict(query_id=q.query_id,method=method,status=status,**{m:None for m in METRICS})
            if complete:
                selected = ranks[ranks.query_id.eq(q.query_id)&ranks.method.eq(method)].sort_values('rank')
                row.update(metrics_at_10([lookup[(q.query_id,doc)] for doc in selected.doc_id],
                                        [lookup[(q.query_id,doc)] for doc in documents]))
            per_query.append(row)
    frame = pd.DataFrame(per_query)
    frame.to_csv(OUT/'relevance_per_query.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'label_coverage.csv',index=False)
    complete = frame[frame.status.eq('COMPLETE_INDEPENDENT_POOL')]
    n = complete.query_id.nunique()
    total = int((~queries.is_edge_case).sum())
    summary = [dict(method=method,status='PENDING_INDEPENDENT_LABELS' if n==0 else 'COMPLETE' if n==total else 'PARTIAL_INDEPENDENT_QUERY_COVERAGE',
        eligible_queries=n,total_queries=total,**{metric:float(complete.loc[complete.method.eq(method),metric].mean()) if n else None for metric in METRICS}) for method in METHODS]
    pd.DataFrame(summary).to_csv(OUT/'relevance_summary.csv',index=False)
    intervals = []
    if n>=2:
        rng = np.random.default_rng(42)
        for metric in METRICS:
            paired = complete.pivot(index='query_id',columns='method',values=metric)
            delta = (paired.SEARCH_V3-paired.SEARCH_V2).to_numpy(dtype=float)
            boot = np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(repetitions)])
            intervals.append(dict(metric=metric,contrast='V3 minus V2',queries=n,mean_delta=float(delta.mean()),
                lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),repetitions=repetitions))
    pd.DataFrame(intervals,columns=['metric','contrast','queries','mean_delta','lower95','upper95','repetitions']).to_csv(OUT/'paired_bootstrap.csv',index=False)
    write_json(OUT/'evaluation_status.json',dict(status=summary[0]['status'],eligible_queries=n,total_queries=total,
        accepted_independent_labels=len(labels),pool_pairs=len(pool),label_source=label_source,
        missing_labels_are_zero=False,relevance_threshold=2,ndcg_gain='2^grade - 1',
        idcg_scope='Shared annotation pool; not full-corpus exhaustive judgments',
        bootstrap='Paired query bootstrap when at least 2 queries have complete independent labels'))
