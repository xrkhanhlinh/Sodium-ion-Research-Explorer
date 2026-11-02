"""Auditable pilot primitives; no benchmark IDs, learned weights or implicit labels."""
import math
import re
from collections import defaultdict

import numpy as np

VERSION = 'search_radar_20261030_v1'
METRICS = ('Precision_at_10', 'nDCG_at_10', 'MRR_at_10')
# Patterns describe observable concepts, not scientific entailment.
CONCEPTS = {
    'materials': {
        'hard carbon': r'hard[ -]carbon', 'ether': r'ether|glyme',
        'layered oxide': r'layered[ -]oxides?', 'Prussian blue': r'Prussian[ -]blue',
        'iron phosphate': r'iron[ -]phosphate', 'pyrophosphate': r'pyrophosphates?',
        'MXene': r'MXenes?', 'quinone': r'quinones?', 'NASICON': r'NASICON',
    },
    'electrode_role': {'anode': r'anodes?|negative electrodes?', 'cathode': r'cathodes?|positive electrodes?'},
    'battery_system': {'sodium-ion': r'sodium[ -]ion|Na[ -]ion|\bSIBs?\b',
                       'lithium-ion': r'lithium[ -]ion|Li[ -]ion|\bLIBs?\b',
                       'full cell': r'full[ -]cells?', 'half cell': r'half[ -]cells?'},
    'properties': {'ionic conductivity': r'ionic conductivity', 'initial Coulombic efficiency': r'initial coulombic efficiency',
                   'low vacancies': r'low vacancies|low vacancy|vacancy[ -]free',
                   'low water': r'low water|water[ -]free', 'volume expansion': r'volume expansion',
                   'anionic redox': r'anionic redox', 'interlayer spacing': r'interlayer spacing'},
    'experimental_conditions': {'low temperature': r'low[ -]temperature|sub[ -]zero',
                                'aqueous': r'aqueous', 'water-in-salt': r'water[ -]in[ -]salt',
                                'low cost': r'low[ -]cost'},
    'processes': {'particle coating': r'particle coating|carbon coating|coated particles?',
                  'electrode coating': r'electrode coating|slurry coating', 'roll pressing': r'roll pressing|calendering',
                  'electrolyte additive': r'electrolyte additives?|additives? (?:in|for) (?:the )?electrolyte',
                  'sacrificial cathode additive': r'sacrificial (?:cathode )?additives?|cathode sacrificial',
                  'spent battery recycling': r'recycl\w* spent|spent .{0,35}recycl',
                  'recycling': r'recycl\w*', 'thermal management': r'thermal management',
                  'voltage balancing': r'voltage balancing|cell balancing'},
    'device': {'battery pack': r'battery packs?|packs?'}
}


def bounded(pattern):
    return r'(?<!\w)(?:' + pattern + r')(?!\w)'


def parse_query(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 10000:
        raise ValueError('Query must contain 1..10000 characters')
    result = dict(query=query, parser_version=VERSION, optional_constraints=[], negated_concepts=[],
                  required_relationships=[], conditions=[])
    # Preserve original strings and case, including formula parentheses and P2/O3.
    formula = r'(?<!\w)(?:[A-Z][a-z]?\d*)+(?:\((?:[A-Z][a-z]?\d*)+\)\d*)*(?!\w)'
    result['chemical_formulas'] = list(dict.fromkeys(m.group() for m in re.finditer(formula, query)
        if (any(c.isdigit() for c in m.group()) or m.group() in ('NaFSI','NaTFSI','FEC'))))
    for category, concepts in CONCEPTS.items():
        result[category] = []
        for name, pattern in concepts.items():
            match = re.search(bounded(pattern), query, re.I)
            if not match:
                continue
            prefix = query[max(0, match.start()-30):match.start()]
            excluded = bool(re.search(r'(?:without|excluding|exclude|not|no)\s+(?:any\s+)?$', prefix, re.I))
            optional = bool(re.search(r'(?:optionally|optional|preferably)\s*$', prefix, re.I))
            item = dict(name=name, category=category, pattern=bounded(pattern), excluded=excluded,
                        optional=optional, query_span=[match.start(), match.end()])
            result['conditions'].append(item)
            result[category].append(name)
            if excluded: result['negated_concepts'].append(name)
            if optional: result['optional_constraints'].append(name)
    for value in result['chemical_formulas']:
        match=re.search(bounded(re.escape(value)),query)
        prefix=query[max(0,match.start()-30):match.start()]
        excluded=bool(re.search(r'(?:without|excluding|exclude|not|no)\s+(?:any\s+)?$',prefix,re.I))
        optional=bool(re.search(r'(?:optionally|optional|preferably)\s*$',prefix,re.I))
        result['conditions'].append(dict(name=value, category='chemical_formulas',
            pattern=bounded(re.escape(value)), excluded=excluded, optional=optional))
        if excluded:result['negated_concepts'].append(value)
        if optional:result['optional_constraints'].append(value)
    # Explicit relation templates are query-derived; sentence co-occurrence is
    # reported as lexical evidence only, never as verified entailment.
    active = [c['name'] for c in result['conditions'] if not c['excluded'] and not c['optional']]
    if len(active) > 1:
        result['required_relationships'] = [dict(concepts=active, scope='same_document',
            verification='requires_review; same-sentence co-occurrence is only a lexical proxy')]
    result['salt'] = [f for f in result['chemical_formulas'] if f in ('NaFSI','NaTFSI')]
    return result


def condition_evidence(parsed, title, abstract):
    title = title if isinstance(title, str) else ''
    abstract = abstract if isinstance(abstract, str) else ''
    rows = []
    supported_sentences = defaultdict(set)
    required = [c for c in parsed['conditions'] if not c['optional']]
    for condition in parsed['conditions']:
        hits = []
        for field, text in (('title', title), ('abstract', abstract)):
            for sentence in re.finditer(r'[^.!?]+(?:[.!?]|$)', text):
                for match in re.finditer(condition['pattern'], sentence.group(), re.I):
                    start = sentence.start() + match.start(); end = sentence.start() + match.end()
                    prefix = sentence.group()[max(0,match.start()-35):match.start()]
                    negated = bool(re.search(r'(?:no|not|without|excluding|absence of)\s+(?:any\s+)?$', prefix, re.I))
                    hits.append(dict(field=field, start=start, end=end, text=text[start:end],
                                     sentence=text[sentence.start():sentence.end()],
                                     sentence_start=sentence.start(), sentence_end=sentence.end(), negated=negated))
        positive = [h for h in hits if not h['negated']]
        negative = [h for h in hits if h['negated']]
        if condition['excluded']:
            status = 'contradicted' if positive else 'lexically_supported' if negative else 'insufficient_evidence'
        else:
            status = 'ambiguous' if positive and negative else 'lexically_supported' if positive else 'contradicted' if negative else 'insufficient_evidence'
        if status == 'lexically_supported' and not condition['optional']:
            for h in (negative if condition['excluded'] else positive):
                supported_sentences[(h['field'],h['sentence_start'],h['sentence_end'])].add(condition['name'])
        rows.append(dict(condition=condition['name'], status=status, optional=condition['optional'], spans=hits))
    required_names = {c['name'] for c in required}
    same_sentence = any(required_names <= names for names in supported_sentences.values()) if required_names else False
    return dict(conditions=rows, required_conditions=len(required),
        supported=sum(r['status']=='lexically_supported' and not r['optional'] for r in rows),
        contradicted=sum(r['status']=='contradicted' and not r['optional'] for r in rows),
        missing_conditions=[r['condition'] for r in rows if not r['optional'] and r['status'] in ('insufficient_evidence','ambiguous')],
        same_sentence_lexical_support=same_sentence, scientific_entailment='NOT_VERIFIED',
        abstract_available=bool(abstract.strip()))


def weighted_rrf(branches, weights=None, k=60):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('RRF k must be a positive integer')
    weights = {name:1. for name in branches} if weights is None else weights
    if set(weights) != set(branches) or any(isinstance(w,bool) or not math.isfinite(w) or w < 0 for w in weights.values()) or not any(weights.values()):
        raise ValueError('One finite nonnegative weight per branch; at least one positive')
    scores = defaultdict(float); provenance = defaultdict(dict)
    for name, ranking in branches.items():
        seen = set()
        for rank, (doc, raw) in enumerate(ranking, 1):
            if doc in seen: raise ValueError('Duplicate ID within retrieval branch')
            if not math.isfinite(raw): raise ValueError('Nonfinite branch score')
            seen.add(doc)
            scores[doc] += weights[name]/(k+rank)
            provenance[doc][name] = dict(rank=rank, score=float(raw), weight=weights[name])
    return [(doc,score,provenance[doc]) for doc,score in sorted(scores.items(), key=lambda item:(-item[1],item[0]))]


def aggregate_chunks(scores, records, mode='max'):
    if mode not in ('max','unique_mean'): raise ValueError('Unknown chunk aggregation')
    scores = np.asarray(scores)
    if scores.shape != (len(records),) or not np.isfinite(scores).all(): raise ValueError('Invalid chunk scores')
    groups = defaultdict(list)
    for score, row in zip(scores, records): groups[row['doc_id']].append((float(score),row))
    out=[]
    for doc, group in groups.items():
        ordered=sorted(group,key=lambda v:v[1]['token_start']); previous=0; weights=[]
        for _, row in ordered:
            start, end=int(row['token_start']),int(row['token_end'])
            if start>previous or end<=previous: raise ValueError('Chunk overlap coverage invalid')
            weights.append(end-max(previous,start));previous=end
        score=max(s for s,_ in ordered) if mode=='max' else float(np.average([s for s,_ in ordered],weights=weights))
        best=max(group,key=lambda v:v[0])[1]
        out.append((doc,score,best))
    return sorted(out,key=lambda v:(-v[1],v[0]))


def evaluate_ranking(ranking, qrels, candidates=()):
    """Absent labels are absent; candidate recall is defined only in judged pool."""
    from .final_system_evaluation import metrics_at_10
    ids = list(ranking)[:10]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate ranked IDs')
    if any(isinstance(g,(bool,np.bool_)) or g not in (0,1,2) for g in qrels.values()): raise ValueError('Grades must be 0..2')
    judged=sum(doc in qrels for doc in ids)
    result=dict(returned=len(ids),judged_top10=judged,judged_fraction=judged/len(ids) if ids else None,
                unjudged_ids=[doc for doc in ids if doc not in qrels], status='COMPLETE' if qrels and judged==len(ids) else 'PENDING_LABELS')
    result.update({m:None for m in METRICS})
    result.update(Success_at_1=None,Success_at_10=None)
    if result['status']=='COMPLETE':
        grades=[qrels[doc] for doc in ids]
        result.update(metrics_at_10(grades,list(qrels.values())))
        result.update(Success_at_1=float(bool(grades) and grades[0]==2),Success_at_10=float(2 in grades))
    relevant={doc for doc,g in qrels.items() if g==2}
    result['Candidate_Recall_pool']=len(set(candidates)&relevant)/len(relevant) if relevant else None
    result['relevant_in_pool']=len(relevant)
    result['missing_relevant_candidates']=sorted(relevant-set(candidates))
    return result


def paired_bootstrap(per_query, baseline='A0', seed=20261030, repetitions=10000):
    """Caller supplies the common complete query set for every method."""
    methods=sorted(per_query.method.unique()); qids=sorted(per_query.query_id.unique())
    if not qids: return []
    if per_query.duplicated(['method','query_id']).any():raise ValueError('Duplicate query/method metrics')
    if any(set(per_query[per_query.method.eq(m)].query_id)!=set(qids) for m in methods):raise ValueError('Use common queries')
    if per_query[list(METRICS)].isna().any().any():raise ValueError('Bootstrap requires complete judgments')
    samples=np.random.default_rng(seed).integers(0,len(qids),size=(repetitions,len(qids)))
    base=per_query[per_query.method.eq(baseline)].set_index('query_id').loc[qids,list(METRICS)].to_numpy(float)
    rows=[]
    for method in methods:
        values=per_query[per_query.method.eq(method)].set_index('query_id').loc[qids,list(METRICS)].to_numpy(float)
        for j,metric in enumerate(METRICS):
            means=values[samples,j].mean(axis=1);delta=(values-base)[samples,j].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975]);dl,dh=np.quantile(delta,[.025,.975])
            rows.append(dict(method=method,metric=metric,mean=float(values[:,j].mean()),ci95_low=lo,ci95_high=hi,
                delta=float((values-base)[:,j].mean()),delta_ci95_low=dl,delta_ci95_high=dh,queries=len(qids)))
    return rows
