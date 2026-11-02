"""Complete existing research artifacts using explicit, resumable phases."""
import argparse
import json
import time
from pathlib import Path


def benchmark():
    import pandas as pd
    import psutil
    from src.final_search import FinalSearchEngine,RESEARCH_METHODS
    from src.research_completion import OUT
    from src.final_system_artifacts import sha,read_json
    from src.model_comparison_io import write_json
    if (OUT/'benchmark_success.json').exists():
        marker=read_json(OUT/'benchmark_success.json')
        if sha(OUT/'rankings_top10.csv')!=marker['rankings_sha256'] or sha(OUT/'run_manifest.csv')!=marker['runs_sha256']:
            raise ValueError('Completed benchmark changed; do not overwrite')
        print('Reusing completed frozen benchmark',flush=True);return
    queries=pd.read_csv(OUT/'queries.csv')
    engine=FinalSearchEngine();rows=[];runs=[]
    try:
        # Group by method to keep only one neural encoder resident in RAM.
        for method in RESEARCH_METHODS:
            for query in queries.itertuples():
                started=time.perf_counter()
                response=engine.search(query.query,method)
                results=response['results']
                if len(results)!=10 or len({r['doc_id'] for r in results})!=10:raise ValueError('Expected10 distinct actual results')
                for result in results:
                    rows.append(dict(query_id=query.query_id,query=query.query,method=method,
                                     **{k:v for k,v in result.items() if k!='topics'}))
                runs.append(dict(query_id=query.query_id,method=method,status='OK',result_count=len(results),
                    seconds=time.perf_counter()-started,process_rss_mib=psutil.Process().memory_info().rss/1024**2,
                    first_for_method=bool(query.Index==0)))
                print(f'{method} {query.query_id}:10 documents {runs[-1]["seconds"]:.3f}s',flush=True)
    finally:engine.close()
    pd.DataFrame(rows).to_csv(OUT/'rankings_top10.csv',index=False)
    pd.DataFrame(runs).to_csv(OUT/'run_manifest.csv',index=False)
    write_json(OUT/'benchmark_success.json',dict(queries=len(queries),methods=list(RESEARCH_METHODS),requests=len(runs),ranked_rows=len(rows),
        rankings_sha256=sha(OUT/'rankings_top10.csv'),runs_sha256=sha(OUT/'run_manifest.csv'),
        queries_sha256=sha(OUT/'queries.csv'),query_cache=False,method_order=list(RESEARCH_METHODS),
        labels_used_for_ranking=False,neural_training=False))
    from src.research_completion_evaluation import create_pool,evaluate
    create_pool();evaluate()


def report():
    from src.research_completion import OUT,AUDIT
    from src.final_system_artifacts import ROOT,read_json,sha
    from src.model_comparison_io import write_json
    import pandas as pd
    import xml.etree.ElementTree as ET
    status=read_json(OUT/'evaluation_status.json')
    qs=read_json(OUT/'qdrant_validation.json');bench=read_json(OUT/'benchmark_success.json')
    topics=pd.read_csv(OUT/'topics/topic_evaluation.csv')
    selection=read_json(OUT/'topics/representation_status.json')
    before=topics.set_index('version').loc['before'];after=topics.set_index('version').loc[selection['selected']]
    runs=pd.read_csv(OUT/'run_manifest.csv')
    latency=runs[~runs.first_for_method].groupby('method').seconds.agg(['count','mean','median','max']).reset_index()
    latency.to_csv(OUT/'latency.csv',index=False)
    tests=[];unique_testcases=set()
    for path in sorted(OUT.glob('*tests.xml')):
        if path.name=='core_tests.xml':continue  # Superseded by the combined backend suite.
        suites=ET.parse(path).getroot().findall('testsuite')
        for s in suites:tests.append(dict(file=path.name,tests=int(s.get('tests')),failures=int(s.get('failures')),errors=int(s.get('errors')),skipped=int(s.get('skipped'))))
        for suite in suites:
            unique_testcases.update((case.get('classname'),case.get('name')) for case in suite.findall('testcase'))
    inputs=read_json(OUT/'inputs.json')
    changed=[p for p,h in inputs.items() if sha(ROOT/p)!=h]
    write_json(OUT/'preservation_check.json',dict(status='PASS' if not changed else 'FAIL',changed=changed,files=len(inputs)))
    if changed:raise ValueError('Frozen inputs changed')
    testtext='; '.join(f"{t['file']}: {t['tests']} tests, {t['failures']} failures, {t['errors']} errors, {t['skipped']} skipped" for t in tests)
    overlap=pd.read_csv(OUT/'topics/overlap_diagnostics.csv')
    text=f'''# Research completion report

The existing engine and Streamlit shell were extended; no encoder or clustering model was retrained. The13,702-document corpus, original TF-IDF matrix, BGE vectors and existing BatteryBERT vectors are reused. Historical V2/V3 remain available for comparisons.

## Topic representation

The existing KMeans10 memberships are frozen. Two targeted representation variants use ngram_range=(1,2), top_n_words=15, reduce_frequent_words=True, and BM25 c-TF-IDF weighting False/True. The branch vocabulary and chemical-aware frozen tokens (with verified placeholder recovery) are reused. Stopwords apply only to representation. Scientific/context-sensitive terms and units are retained; recipe words such as water, solution and stir are not blindly deleted because they can describe synthesis or electrolytes.

Selected representation: {selection['selected']}. Same-reference C_v: {before.c_v:.6f} -> {after.c_v:.6f}; keyword diversity@10: {before.keyword_diversity_at_10:.2f} -> {after.keyword_diversity_at_10:.2f}; generic inspected keyword slots: {int(before.generic_keyword_slots)} -> {int(after.generic_keyword_slots)}. All memberships and topic sizes are unchanged; outlier fraction is0 because KMeans assigns every document, not because relevance is perfect. Top15 keywords are saved, and Top10 are used for evaluation. Scores were recomputed on one shared repaired token/phrase reference; old incompatible coherence scores were not mixed.

Centroid overlap1/5={overlap.loc[overlap.topic_a.eq(1),'centroid_cosine'].iloc[0]:.6f}; overlap2/3={overlap.loc[overlap.topic_a.eq(2),'centroid_cosine'].iloc[0]:.6f}. High pooled-BERT cosine is not sufficient evidence to merge topics. Five actual nearest-centroid Title-Abstract documents per topic are exported. All50 central examples have agent inspection notes in topics/representative_review.csv, and topics/overlap_review.md explains pair-specific evidence. Scope/misplacement limitations from the prior22-record inspection remain explicit. Topics2/3 and3/4 are broad/overlapping; Patent/Publication style effects remain unresolved. Labels are tentative agent interpretations, not independent scientific ground truth. The web displays this uncertainty. No forced reclustering or invented technical keyword was used.

## Search and Qdrant

- TFIDF_COSINE: original L2-normalized full Title-Abstract TF-IDF cosine.
- BATTERYBERT_SEMANTIC: pretrained batterydata/batterybert-cased, pinned revision0106ed8cca65ec6302ada7123048be9c37e31a7d; native tokenizer; attention-mask mean + L2, matching document pooling. Query maximum256 tokens with explicit rejection rather than silent truncation; documents retain their existing256-token overlapping chunk coverage.
- BATTERYBERT_HYBRID: independent Top100 TF-IDF and pretrained BatteryBERT rankings, deduplicated reciprocal-rank fusion k=60, finalTop10. No MiniLM reranking, raw-score addition or topic relevance bonus.
- BGE_SEMANTIC: existing pretrained BGE query protocol and existing normalized document vectors reused as the comparison baseline.

The new pretrained BatteryBERT collection uses persisted Qdrant Local cosine distance. It is built once from saved vectors and reused for queries; no per-query upsert/build is performed. Payloads include doc_id, title, full available abstract, source, year, identifier, topic_id and topic_label. All13,702 points/payloads/vectors were checked; {qs['cases']} exact-retrieval/filter probes passed with score/tie tolerance2e-6. Original Qdrant storage was left on its prior path; the additional pretrained checkpoint requires its own compatible collection under this versioned output. BGE's existing memmap retrieval is reused rather than duplicating its index.

Source/year/required-term/topic filters apply before ranking in both channels. Empty lexical support does not gate semantic retrieval. RRF uses rank positions only, and missing semantic Top100 membership is distinguished from an actual cosine score. Query encoders are switched serially to bound RAM. One worker owns each local database; stop the serving app before running another process against the same new Qdrant directory.

## Top10 evaluation

Reused the first20 existing non-edge query IDs/texts without rewriting them. Executed{bench['requests']} real requests across4 methods and exported{bench['ranked_rows']} ranked rows. A shared deduplicated Top10 pool contains{status['pool_pairs']} query-document pairs. The annotation file hides method, rank, score and topic labels to reduce anchoring.

Human grade scale:0 not relevant,1 partial,2 directly relevant. Precision@10 and MRR@10 use grade>=2; nDCG uses gain2^grade-1 and IDCG from the complete shared pool. A query is eligible only when its entire pool is independently judged and verified. Blank labels remain absent. Recall@10 is not reported because full-corpus relevant sets are unknown. The protocol is locked for final evaluation, with no human-label tuning in this completion run.

Current status: **{status['status']}**; accepted human labels={status['accepted_human_labels']}; complete query pools={status['eligible_queries']}/{status['total_queries']}. Official P@10/nDCG@10/MRR@10 remain blank until labels arrive. No fake relevance labels, accuracy bars or claims that one retrieval method is superior were generated. The metric implementation is exercised with explicitly synthetic unit fixtures stored only in temporary test directories.

`annotation_template_0_2.csv` can be completed and imported in Evaluation or passed to the CLI. Required provenance fields follow the existing independent-human protocol: annotator_id, distinct verifier_id, verification_reference, label_method=HUMAN_INDEPENDENT and label_status=VERIFIED.

## Web and validation

The existing web shell now exposes TF-IDF, pretrained BatteryBERT and Hybrid, with BGE in the same selector and historical methods optionally available. Results show full title/abstract, identifiers, source/year, fixed-KMeans topic ID/label, actual dense cosine and explicitly named RRF scores where applicable. Topic filtering occurs before retrieval and the Topics view includes all10 saved clusters. Evaluation downloads/imports the0..2 pool and updates metrics only with valid human labels.

Confirmed presentation defects were corrected: lexical excerpts were previously described as passages used for ranking even for full-document/vector search; they now have an explicit excerpt label. Empty abstracts read from saved CSV become NaN (26 rows in this benchmark), which previously rendered a literal nan; the existing card now displays the missing-abstract message. No abstract text was invented. Previously working V2/V3 are preserved; their fine-tuned/BM25/reranker design was a difference from the new requested protocol, not evidence that those algorithms were broken.

Changed files: src/final_search.py (additive research modes, pretrained query encoder, pre-ranking topic filter, separate cosine/RRF fields); src/final_system_vectors.py (isolated registry/output directory and complete topic/abstract metadata); src/final_system_evaluation.py (explicit0..2 validation option, legacy default unchanged); src/final_dashboard.py and app.py (reuse the web shell, new selector/topics/evaluation and corrected score/excerpt/abstract presentation). Added src/research_completion.py, src/research_completion_evaluation.py and run_research_completion.py for bounded preparation, representation, indexing, benchmark and evaluation. Tests and run documentation were updated. No large parameter search, LDA/BGE retraining or corpus edit was executed.

Executed tests: {testtext}. There are {len(unique_testcases)} distinct test cases; focused reruns are listed separately rather than added as new tests. HTTP/app smoke results are saved separately. Technical tests verify contracts and rankings; they do not establish relevance quality.

Measured warm request latency (first request per method excluded, grouped serial method order, not a randomized performance comparison):

{latency.to_string(index=False)}

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe run_research_completion.py prepare
.\\.venv\\Scripts\\python.exe run_research_completion.py topics
.\\.venv\\Scripts\\python.exe run_research_completion.py select --representation {selection['selected']}
.\\.venv\\Scripts\\python.exe run_research_completion.py index
.\\.venv\\Scripts\\python.exe run_research_completion.py benchmark
.\\.venv\\Scripts\\python.exe run_research_completion.py evaluate --labels PATH_TO_ACTUAL_HUMAN_LABELS.csv
.\\.venv\\Scripts\\python.exe run_demo.py
```

The frozen completed benchmark is reused when checksums match. Stop the web app before index/benchmark. Without --labels, evaluation retains the last accepted immutable human-label source if one exists; otherwise it remains PENDING_HUMAN_ANNOTATION. Existing old annotation scales/results remain separate. Required outputs are under outputs/research_completion_v1/topics/ and this directory; source/configuration hashes and an artifact manifest are retained.
'''
    (OUT/'COMPLETION_REPORT.md').write_text(text,encoding='utf-8')
    write_json(OUT/'final_status.json',dict(technical_tests=tests,unique_testcases=len(unique_testcases),retrieval=status,topic_representation=selection,
        index_validation=qs['status'],training=False,clustering_refit=False,scientific_labels='Tentative',
        independent_relevance_quality='Pending' if not status['eligible_queries'] else 'See coverage'))
    sources=['src/final_search.py','src/final_system_vectors.py','src/final_dashboard.py','src/final_system_evaluation.py',
        'src/research_completion.py','src/research_completion_evaluation.py','run_research_completion.py','app.py']
    write_json(OUT/'artifact_manifest.json',dict(
        files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(OUT/'qdrant') and not p.is_relative_to(OUT/'logs') and p.name!='artifact_manifest.json'},
        source_sha256={p:sha(ROOT/p) for p in sources},qdrant_live_files_excluded=True))
    print('Completion report written',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','topics','select','index','benchmark','evaluate','report'])
    parser.add_argument('--representation',default='ctfidf_bm25_1')
    parser.add_argument('--labels')
    args=parser.parse_args()
    from src import research_completion as completion
    completion.OUT.mkdir(parents=True,exist_ok=True)
    if args.phase in ('prepare','index'):getattr(completion,args.phase)()
    elif args.phase=='topics':completion.optimize_topics()
    elif args.phase=='select':completion.select_topics(args.representation)
    elif args.phase=='benchmark':benchmark()
    elif args.phase=='evaluate':
        from src.research_completion_evaluation import evaluate
        print(evaluate(args.labels).to_string(index=False))
    elif args.phase=='report':report()


if __name__=='__main__':main()
