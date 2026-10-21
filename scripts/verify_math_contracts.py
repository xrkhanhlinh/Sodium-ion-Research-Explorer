"""Independent exact references for current code; no models or relevance labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import pandas as pd
    import sympy as sp
    from rank_bm25 import BM25Okapi
    from src.search_v2 import reciprocal_rank_fusion, select_passages
    from src.research_evaluation import (
        METRICS, metrics_at_5, paired_bootstrap, paired_cluster_bootstrap, annotation_agreement,
    )
    from src.ai_retrieval_evaluation import agreement

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new evidence file; preserve previous runs')
    checks = []

    def record(name, evidence):
        checks.append(dict(name=name, status='PASS', evidence=evidence))

    k = sp.Symbol('k', nonnegative=True)
    r = sp.Symbol('r', integer=True, positive=True)
    decay = sp.factor(1/(k+r)-1/(k+r+1))
    assert decay == 1/((k+r)*(k+r+1)) and decay.is_positive is True
    expected = {'a': sp.Rational(1,61), 'b': sp.Rational(1,61)+sp.Rational(1,62), 'c': sp.Rational(1,62)}
    actual = dict(reciprocal_rank_fusion([[('a',99),('b',0)],[('b',-3),('c',8)]]))
    assert all(abs(actual[name]-float(value)) < 1e-14 for name,value in expected.items())
    record('RRF exact scores and monotone rank contribution', {name:str(value) for name,value in expected.items()})

    f,k1,b,length,average,idf = sp.symbols('f k1 b L A I', positive=True)
    bm25 = idf*f*(k1+1)/(f+k1*(1-b+b*length/average))
    corpus = [['a','a','b'],['b','c'],['d'],['e'],['f']]
    index = BM25Okapi(corpus,k1=1.5,b=.75)
    reference = [float(bm25.subs({f:doc.count('a'),k1:sp.Rational(3,2),b:sp.Rational(3,4),
        length:len(doc),average:sp.Rational(8,5),idf:sp.log(3)})) for doc in corpus]
    assert np.allclose(index.get_scores(['a']),reference,atol=1e-14,rtol=0)
    clipped_example = BM25Okapi([['sodium']]*3).get_scores(['sodium'])
    assert not np.maximum(clipped_example,0).any()
    record('Installed BM25 algebra; clipping counterexample retained',
           dict(formula=str(bm25),raw_common_term_scores=clipped_example.tolist(),
                limitation='Toy corpus, not evidence of failure on the actual corpus; weighted separate fields are not full BM25F'))

    x,y = sp.Matrix([3,4]), sp.Matrix([4,-3])
    assert (x/x.norm()).dot(y/y.norm()) == x.dot(y)/(x.norm()*y.norm()) == 0
    norm_evidence = {}
    for kind in ('pretrained','finetuned'):
        vectors = np.load(ROOT/f'data/stage4/{kind}_query_embeddings.npy',mmap_mode='r')
        norms = np.linalg.norm(vectors,axis=1)
        assert vectors.shape == (24,768) and np.isfinite(vectors).all()
        assert np.allclose(norms,1,atol=2e-5,rtol=0)
        norm_evidence[kind] = dict(shape=list(vectors.shape),min_norm=float(norms.min()),max_norm=float(norms.max()))
    record('Cosine identity for nonzero vectors and actual cached query norms',norm_evidence)

    rng = np.random.default_rng(1729)
    for _ in range(150):
        pool = rng.integers(0,4,size=int(rng.integers(0,9))).tolist()
        chosen = rng.choice(len(pool),size=min(len(pool),int(rng.integers(0,6))),replace=False)
        grades = [pool[i] for i in chosen]
        dcg = sum((sp.Integer(2)**v-1)/sp.log(i+2,2) for i,v in enumerate(grades))
        idcg = sum((sp.Integer(2)**v-1)/sp.log(i+2,2) for i,v in enumerate(sorted(pool,reverse=True)[:5]))
        relevant = [i for i,v in enumerate(grades,1) if v >= 2]
        reference = dict(Precision_at_5=sp.Rational(len(relevant),5),
            MRR_at_5=sp.Rational(1,relevant[0]) if relevant else 0,nDCG_at_5=dcg/idcg if idcg else 0)
        actual = metrics_at_5(grades,pool)
        assert all(abs(actual[name]-float(value)) < 1e-13 for name,value in reference.items())
    for bad in (None,np.nan,True,2.5,-1,4):
        try:
            metrics_at_5([bad],[0,1,2,3])
        except ValueError:
            continue
        raise AssertionError(f'Accepted invalid grade: {bad}')
    record('Precision/MRR/nDCG vs exact SymPy definitions',dict(cases=150,seed=1729,missing='rejected',zero_idcg='0 by declared convention'))

    per = pd.DataFrame([dict(query_id=q,method=m,**{name:i/4 for name in METRICS})
        for i,q in enumerate(['q1','q2','q3']) for m in ('A','B')])
    _,delta = paired_bootstrap(per,['q1','q2','q3'],['A','B'],100,42)
    assert (delta[['mean_delta','ci95_low','ci95_high']].to_numpy() == 0).all()
    record('Paired query bootstrap cancellation',dict(identical_methods_delta=0,repetitions=100))
    cluster = paired_cluster_bootstrap(np.zeros((3,1)),np.array([[0.],[1.],[1.]]),['a','b','b'],['gain'],100,42)
    assert abs(cluster.iloc[0].mean_delta-float(sp.Rational(2,3))) < 1e-15
    assert cluster.iloc[0].ci95_low == 0 and cluster.iloc[0].ci95_high == 1
    record('Unequal-cluster ratio estimator',dict(expected_mean_delta='2/3',possible_bootstrap_values=['0','2/3','1']))

    a,b = [0,1,2,3],[0,2,1,3]
    observed = sum(sp.Rational((u-v)**2,9) for u,v in zip(a,b))/4
    expected = sum(sp.Rational((u-v)**2,9) for u in a for v in b)/16
    kappa = 1-observed/expected
    left = pd.DataFrame([dict(query_id='q',doc_id=str(i),relevance_grade=v,annotator_id='a') for i,v in enumerate(a)])
    right = left.copy();right['relevance_grade']=b;right['annotator_id']='b'
    human = annotation_agreement(left,right)
    docs = [dict(query_id='q',doc_id=str(i)) for i in range(4)]
    votes = {(role,'q',str(i)):{'judgment':{'grade':v}} for role,values in [('A',a),('B',b)] for i,v in enumerate(values)}
    ai = agreement(docs,votes)
    assert abs(human['quadratic_weighted_kappa']-float(kappa)) < 1e-15
    assert abs(ai['quadratic_weighted_kappa']-float(kappa)) < 1e-15
    record('Quadratic weighted kappa in both evaluators',dict(expected=str(kappa),synthetic_only=True))

    class Tokenizer:
        def __call__(self,text,**kwargs):
            spans = [m.span() for m in re.finditer(r'\S+',text)]
            return dict(input_ids=list(range(len(spans))),offset_mapping=spans)
    try:
        select_passages(' '.join(['title']*80),' '.join(['abstract']*600),
                        ' '.join(['query']*128),Tokenizer(),width=400,overlap=301)
    except ValueError:
        pass
    else:
        raise AssertionError('Stalled-window counterexample was not rejected')
    record('Original nontermination counterexample now rejects before iteration',dict(effective_width=301,overlap=301))

    u,v,c = sp.symbols('u v c',real=True)
    probability = sp.exp(u)/(sp.exp(u)+sp.exp(v))
    shifted = sp.exp(u+c)/(sp.exp(u+c)+sp.exp(v+c))
    assert sp.simplify(probability-shifted) == 0
    loss = -u+sp.log(sp.exp(u)+sp.exp(v))
    assert sp.factor(sp.diff(loss,u)).is_negative is True
    assert sp.factor(sp.diff(loss,v)).is_positive is True
    record('Two-logit cross-entropy algebra',dict(scope='Symbolic softmax shift invariance and derivative signs; not a training convergence/gradient implementation proof'))

    # All C_v values lie in [0,1]; C is dominated by A.
    old_a,old_b = sp.Rational(1,2),sp.Rational(15,100)
    new_a = sp.Rational(1,2)
    new_b = sp.Rational(1,2)*sp.Rational(49,50)+sp.Rational(15,100)
    assert old_a > old_b and new_a < new_b
    record('Candidate-set sensitivity of minmax composite selection',
           dict(A=dict(c_v='.50',u_mass='-10'),B=dict(c_v='.49',u_mass='-9'),
                dominated_C=dict(c_v='0',u_mass='-10'),old_scores=[str(old_a),str(old_b)],
                new_scores=[str(new_a),str(new_b)],scope='Diversity/stability held equal; structural sensitivity, not arithmetic error'))

    source_names = ['src/search_v2.py','src/research_evaluation.py','src/ai_retrieval_evaluation.py','src/stage3_common.py']
    result = dict(scope='Bounded symbolic and numerical references, never relevance accuracy',sympy_version=sp.__version__,
        checks=checks,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(checks_passed=len(checks),output=str(args.output))))


if __name__ == '__main__':
    main()
