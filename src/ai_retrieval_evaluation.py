"""AI-only adapter around the unchanged, strict research metric primitives."""
import json
import numpy as np
import pandas as pd

from .ai_judge import ERROR_TYPES, validate_judgment
from .research_evaluation import (METRICS, clean_keys, metrics_at_5,
                                  paired_bootstrap, validate_rankings)

METHODS = ('BM25_CHEMICAL', 'DENSE_PRETRAINED', 'DENSE_FINETUNED', 'DENSE_CHUNKED',
           'HYBRID_PRETRAINED', 'HYBRID_FINETUNED', 'HYBRID_CHUNKED',
           'RERANK_PRETRAINED', 'RERANK_FINETUNED', 'RERANK_CHUNKED', 'SE1', 'SE2', 'SE3', 'BM25')
DEVELOPMENT = {'Q01', 'Q08', 'Q16', 'Q19', 'Q23'}
AI_METHOD = 'GEMINI_LLM_JUDGE'
AI_STATUS = 'AI_CONSENSUS_UNVERIFIED'


def numeric(record):
    return record is not None and type(record['judgment']['grade']) is int


def consensus_record(document, a=None, b=None, c=None):
    for record in (a, b, c):
        if record is not None:
            validate_judgment(record['judgment'], document, record['requested_model'])
            if record.get('evidence_valid') is not True or not record.get('request_sha256'):
                raise ValueError('Missing actual call/evidence provenance')
    result = {'query_id': document['query_id'], 'doc_id': document['doc_id'],
              'relevance_grade': None, 'label_method': AI_METHOD, 'label_status': 'PENDING_JUDGES',
              'expert_verified': False, 'resolution': 'PENDING', 'evidence': [], 'reason': '',
              'confidence': None, 'axes': {},
              **{f'judge_{role}_request': r['request_sha256'] if r else ''
                 for role, r in [('a', a), ('b', b), ('c', c)]}}
    selected = None
    if a is not None and b is not None:
        if a['request_sha256'] == b['request_sha256']:
            raise ValueError('A and B must be separate calls')
        if numeric(a) and numeric(b) and a['judgment']['grade'] == b['judgment']['grade']:
            selected, result['resolution'] = a, 'AB_AGREEMENT'
        elif a['judgment']['grade'] == b['judgment']['grade'] == 'UNRESOLVED':
            result.update(label_status='UNRESOLVED', resolution='AB_INSUFFICIENT',
                          reason='A: '+a['judgment']['reason']+'; B: '+b['judgment']['reason'])
        elif c is not None:
            if c['request_sha256'] in (a['request_sha256'], b['request_sha256']):
                raise ValueError('Adjudication must be a separate call')
            selected, result['resolution'] = c, 'C_ADJUDICATION'
        else:
            result['label_status'] = 'PENDING_ADJUDICATION'
    if selected is not None:
        j = selected['judgment']
        result.update(relevance_grade=j['grade'] if numeric(selected) else None,
                      label_status=AI_STATUS if numeric(selected) else 'UNRESOLVED',
                      evidence=j['evidence'], reason=j['reason'], confidence=j['confidence'], axes=j['axes'],
                      model_id=selected['requested_model'], model_version=selected['model_version'])
    # On agreement A and B may disagree on error axes. Mark that explicitly rather
    # than attributing A's diagnosis to both raters.
    if result['resolution'] == 'AB_AGREEMENT':
        result['axes'] = {key: a['judgment']['axes'][key] if a['judgment']['axes'][key] == b['judgment']['axes'][key]
                          else 'JUDGES_DISAGREE' for key in ERROR_TYPES}
    result['axes'] = json.dumps(result['axes'], ensure_ascii=False)
    result['evidence'] = json.dumps(result['evidence'], ensure_ascii=False)
    return result


def agreement(documents, votes):
    a, b = [], []
    completed = unresolved = 0
    for doc in documents:
        pair = (doc['query_id'], doc['doc_id'])
        left, right = votes.get(('A', *pair)), votes.get(('B', *pair))
        if left is not None and right is not None:
            completed += 1
            if numeric(left) and numeric(right):
                a.append(left['judgment']['grade']); b.append(right['judgment']['grade'])
            else:
                unresolved += 1
    result = {'status': 'COMPLETE' if completed == len(documents) else 'PENDING_OR_PARTIAL',
              'pool_pairs': len(documents), 'completed_both_calls': completed,
              'paired_numeric_ratings': len(a), 'pairs_with_unresolved_vote': unresolved,
              'exact_agreement': None, 'quadratic_weighted_kappa': None,
              'scope': 'Raw A/B before adjudication; Gemini family, not expert or independent-family agreement',
              'kappa_note': 'PENDING_NO_PAIRED_NUMERIC_RATINGS'}
    if not a:
        return result
    matrix = np.zeros((4, 4), dtype=float)
    np.add.at(matrix, (a, b), 1)
    weights = (np.arange(4)[:, None] - np.arange(4)[None, :]) ** 2 / 9.
    observed = matrix / len(a)
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / len(a) ** 2
    denominator = float((weights * expected).sum())
    result.update(exact_agreement=float(np.mean(np.array(a) == np.array(b))),
                  quadratic_weighted_kappa=1 - float((weights * observed).sum()) / denominator if denominator else None,
                  kappa_note='Numeric pairs only; partial coverage may be selected' if denominator else 'Undefined: constant rating marginals')
    return result


def evaluate_ai(rankings, queries, pool, labels, run_manifest, methods=METHODS, repetitions=2000):
    ranks, domain, pool = validate_rankings(rankings, queries, pool, methods, run_manifest)
    labels = clean_keys(labels)
    if set(map(tuple, labels[['query_id', 'doc_id']].values)) != set(map(tuple, pool[['query_id', 'doc_id']].values)):
        raise ValueError('AI qrels must contain exactly the frozen pool')
    if not labels.label_method.eq(AI_METHOD).all():
        raise ValueError('Only explicitly AI-labeled provenance accepted')
    if not labels.label_status.isin([AI_STATUS,'UNRESOLVED','PENDING_JUDGES','PENDING_ADJUDICATION']).all():
        raise ValueError('Unknown AI label status')
    resolved = labels.label_status.eq(AI_STATUS)
    raw = labels.relevance_grade
    if raw[~resolved].map(lambda x: not (x is None or pd.isna(x) or str(x).strip() == '')).any():
        raise ValueError('Unresolved/pending labels must have empty grades')
    from .research_evaluation import grades_array
    if resolved.any():
        grades_array(raw[resolved])
        if labels.loc[resolved, 'expert_verified'].map(lambda v: str(v).lower() != 'false').any():
            raise ValueError('AI labels cannot claim expert verification')
        for col in ('evidence', 'reason', 'judge_a_request', 'judge_b_request', 'resolution'):
            if labels.loc[resolved, col].fillna('').astype(str).str.strip().isin(['', '[]']).any():
                raise ValueError('Missing AI consensus provenance')
    labels = labels.copy()
    labels['resolved'] = resolved
    complete = [q for q in domain if labels.loc[labels.query_id.eq(q), 'resolved'].all()]
    qrels = labels[resolved].set_index(['query_id', 'doc_id']).relevance_grade
    rows, coverage = [], []
    for qid in domain:
        group = labels[labels.query_id.eq(qid)]
        coverage.append({'query_id': qid, 'pool_pairs': len(group), 'resolved': int(group.resolved.sum()),
                         'included_all_methods': qid in complete,
                         'split': 'development' if qid in DEVELOPMENT else 'remaining'})
        for method in methods:
            values = {key: None for key in METRICS}
            if qid in complete:
                ordered = ranks[ranks.query_id.eq(qid) & ranks.method.eq(method)].sort_values('rank')
                values = metrics_at_5([qrels.loc[(qid, d)] for d in ordered.doc_id], group.relevance_grade)
            rows.append({'query_id': qid, 'method': method, **values,
                         'status': AI_STATUS if qid in complete else 'PENDING_COMPLETE_QUERY_POOL'})
    per = pd.DataFrame(rows)
    summary, intervals, deltas = [], [], []
    for split, ids in [('all', domain), ('development', [q for q in domain if q in DEVELOPMENT]),
                       ('remaining', [q for q in domain if q not in DEVELOPMENT])]:
        included = [q for q in ids if q in complete]
        for method in methods:
            vals = per[per.method.eq(method) & per.query_id.isin(included)]
            summary.append({'split': split, 'method': method, 'queries_planned': len(ids),
                            'queries_evaluated': len(included), 'status': 'PENDING' if not included else
                            (AI_STATUS if len(included) == len(ids) else 'PARTIAL_AI_SELECTED_QUERIES'),
                            **{k: float(vals[k].mean()) if included else None for k in METRICS}})
        if len(included) >= 2:
            ci, pairs = paired_bootstrap(per[per.query_id.isin(included)], included, methods, repetitions)
            ci['split'] = split; pairs['split'] = split
            intervals.append(ci); deltas.append(pairs)
    return {'summary': pd.DataFrame(summary), 'per_query': per, 'query_coverage': pd.DataFrame(coverage),
            'confidence_intervals': pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame(columns=['split','method','metric','mean','ci95_low','ci95_high','queries']),
            'paired_deltas': pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame(columns=['split','left','right','metric','mean_delta','ci95_low','ci95_high','queries']),
            'scope': {'label_method': AI_METHOD, 'expert_verified': False, 'domain_queries': len(domain),
                      'complete_query_pools': complete, 'unresolved_not_zero': True,
                      'missing_policy': 'Exclude whole incomplete query pool for ALL methods; report selection/coverage bias',
                      'ndcg': '2^grade-1, ideal from full frozen shared pool; zero IDCG kept with nDCG=0',
                      'ci': 'Paired query bootstrap, fixed AI labels/checkpoints; 91 comparisons exploratory, no multiplicity adjustment',
                      'generalization': 'Remaining 18 are authored queries, not independent external test; synthetic results separate'}}


def error_analysis(rankings, labels):
    joined = rankings.merge(labels, on=['query_id', 'doc_id'], how='left', validate='many_to_one')
    rows = []
    for row in joined.to_dict('records'):
        if row.get('label_status') != AI_STATUS or float(row['relevance_grade']) >= 2:
            continue
        axes = json.loads(row['axes'])
        rows.append({key: row[key] for key in ('query_id','method','rank','doc_id','title','relevance_grade','reason')} |
                    {key: axes.get(key, 'INSUFFICIENT') for key in ERROR_TYPES} |
                    {'scope': 'AI_DIAGNOSIS_NOT_EXPERT_CAUSAL_ERROR'})
    return pd.DataFrame(rows, columns=['query_id','method','rank','doc_id','title','relevance_grade','reason',*ERROR_TYPES,'scope'])
