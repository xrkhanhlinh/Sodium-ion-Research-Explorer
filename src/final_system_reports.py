"""Generate all final reports exclusively from observed run artifacts."""
from pathlib import Path
import platform
import xml.etree.ElementTree as ET
from importlib.metadata import version
import pandas as pd
from .final_system_artifacts import ROOT,OUT,OLD,TOPICS,read_json,sha
from .model_comparison_io import write_json


def table(path):
    if not path.exists():return '*Not executed yet.*'
    frame=pd.read_csv(path).fillna('—')
    if frame.empty:return '*No eligible observations.*'
    def cell(value):
        if isinstance(value,float):return f'{value:.6g}'
        return str(value).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join('---' for _ in frame.columns)+' |\n'+'\n'.join(
        '| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))


def info(path):
    return read_json(path) if path.exists() else {'status':'NOT_EXECUTED'}


def report():
    OUT.mkdir(exist_ok=True,parents=True)
    evaluation=info(OUT/'evaluation_status.json')
    qdrant=info(OUT/'qdrant_validation.json')
    embedding=info(OUT/'batterybert_finetuned/embedding_success.json')
    preservation=info(OUT/'preservation_check.json')
    optimization=info(OUT/'optimization_parity.json')
    loading=read_json(OUT/'model_loading.json') if (OUT/'model_loading.json').exists() else []
    tests=[]
    for filename in ('tests.xml','dashboard_tests.xml'):
        if (OUT/filename).exists():
            tree=ET.parse(OUT/filename)
            suites=list(tree.getroot().iter('testsuite'))
            tests.append(dict(file=filename,tests=sum(int(s.get('tests',0)) for s in suites),
                failures=sum(int(s.get('failures',0)) for s in suites),errors=sum(int(s.get('errors',0)) for s in suites),
                skipped=sum(int(s.get('skipped',0)) for s in suites)))
    training=info(OLD/'batterybert_finetuned/training_success.json')
    commands='''```powershell
.venv/Scripts/python.exe run_final_system.py prepare
# Infer ONLY missing BatteryBERT vectors from the existing checkpoint; no training:
.venv/Scripts/python.exe run_final_system.py embeddings
.venv/Scripts/python.exe run_final_system.py topics
.venv/Scripts/python.exe run_final_system.py qdrant
.venv/Scripts/python.exe run_final_system.py benchmark
.venv/Scripts/python.exe run_final_system.py evaluate
# Optional: import partially/fully independently verified grades (0–3):
.venv/Scripts/python.exe run_final_system.py evaluate --labels PATH_TO_LABELS.csv
.venv/Scripts/python.exe run_final_system.py optimize
.venv/Scripts/python.exe run_final_system.py smoke
.venv/Scripts/python.exe run_final_system.py models
.venv/Scripts/python.exe run_final_system.py validate
.venv/Scripts/python.exe -m pytest tests/test_search_v2_app.py -q --junitxml=outputs/final_system/dashboard_tests.xml
.venv/Scripts/python.exe scripts/smoke_final_dashboard_server.py
.venv/Scripts/python.exe run_final_system.py preserve
.venv/Scripts/python.exe run_final_system.py report
.venv/Scripts/python.exe -m streamlit run app.py --server.headless true
```
Stop the dashboard before CLI Qdrant build/validation: Local persistent storage has one process owner. The app serializes requests in one worker. If Local storage is already locked, the error is surfaced; no encoder/database fallback occurs.
'''
    architecture=f'''# System architecture

The frozen corpus has 13,702 Patent/Publication documents. Source data, old models, checkpoints and old outputs are read-only in this integration. All new derived files live under `outputs/final_system/`.

```mermaid
flowchart LR
  C[Frozen Title–Abstract corpus] --> T[Existing LDA / BERTopic artifacts]
  T --> D[Topic Analysis dashboard]
  C --> E[Existing checkpoint / saved embeddings]
  E --> Q[Versioned persistent Qdrant / exact cosine]
  U[Query + source/year/required terms] --> L[BM25 Top 100]
  U --> B[Fine-tuned BatteryBERT query encoder]
  B --> Q
  L --> R[RRF k=60 / Top 50]
  Q --> R
  R --> M[MiniLM reranking / Top 10]
  M --> S[Search dashboard + CSV]
  M --> A[Shared annotation pool]
  A --> V[Independent verified grades / @10 evaluation]
  V --> D2[Model Comparison dashboard]
```

Three configurations share the same corpus, query protocol and filters: TF-IDF/Cosine, original Search V2 (BM25+BGE pretrained→RRF→MiniLM), and Search V3 (BM25+BatteryBERT fine-tuned/Qdrant→RRF→MiniLM). Topic assignments/keywords/labels annotate results after ranking and never add ranking bonuses.

`src/final_system_artifacts.py` validates order/checksums and exports existing topics. `src/final_system_vectors.py` owns Qdrant. `src/final_search.py` delegates original V2 without algorithm changes and implements TF-IDF/V3. `src/final_system_evaluation.py` validates independent labels. Existing `app.py` retains the shell, filters and document renderer; `src/final_dashboard.py` supplies updated views.

BatteryBERT query/document representations use checkpoint `outputs/model_comparison_v2/batterybert_finetuned/model`, native cased tokenizer, attention-mask mean including native special tokens, then L2. Full document vectors average L2 chunk vectors and normalize again (256 tokens, 64 overlap). Queries use no BGE instruction and reject >128 tokens rather than silently truncate. V2 retains its original BGE CLS/instruction/128-token inference contract. This end-to-end comparison therefore includes pooling/chunking differences; it is not a pure encoder-only ablation.

No training, pretraining or BERTopic refit is invoked by `run_final_system.py`. The missing fine-tuned corpus embeddings are inferred once with resume/checksum support. Completed embeddings are reused. Exact float32 frozen first-11-block cache rows are read-only, per-row SHA checked, matched to frozen weight hashes; the updated last block is recomputed and checked against full forward. Missing cache rows use full forward and do not mutate the earlier experiment.

## Reproduction

{commands}
'''
    qr=f'''# Qdrant report

Status of currently verified collections: **{qdrant['status']}**, coverage **{len(qdrant.get('collections',[]))}/2 required collections**. Qdrant client {version('qdrant-client')}; Local persistent directory `outputs/final_system/qdrant/`. This is a real persisted Qdrant database, with SQLite-backed Local storage and exact search. It is not a mocked service or an HNSW benchmark. Local mode keeps vectors in memory; only two 13,702×768 collections are needed. Abstract text remains in the corpus table rather than being duplicated in every payload.

Collections are named from encoder, version, model SHA and vector SHA. Cosine distance, float32, 768 dimensions, 64-point upsert batches, stable integer point ID=corpus row. Upserts replace the same IDs, so resume does not duplicate points. Payload: `doc_id`, corpus row, source, numeric/nullable year, title, identifier and encoder. The collection manifest binds corpus/model/vector hashes. New versions do not overwrite historical Qdrant storage under `data/stage4/qdrant`.

Source/year filters are executed inside Qdrant before Top-K. Missing years are excluded from year intervals. Explicit required-term constraints reuse V2's chemical lexical masks and are passed as allowed doc IDs. All 13,702 payload/vector pairs in every collection are checked against the corpus and source embeddings. Probe cases compare Top100 against NumPy exact cosine, including source/year filters. Raw ID overlap is reported; score-level float32 cutoff ties within 2e-6 are checked separately, requiring every strictly better point.

Observed validation:

```json
{__import__('json').dumps(qdrant,ensure_ascii=False,indent=2)}
```

Detailed cases: `qdrant_exact_parity.csv`; collection definitions and provenance: `qdrant_collections.json`. HNSW was not tried on this 8 GB CPU environment.

References: [official persistent Local client](https://github.com/qdrant/qdrant-client), [official exact search API](https://qdrant.tech/documentation/search/).
'''
    topic=f'''# Topic model comparison

{table(OUT/'topic_comparison.csv')}

All assignments and metrics above are reused from the listed model directories; no LDA/UMAP/HDBSCAN/BERTopic fit or topic reduction was executed for this integration. Ten research topics per branch exclude outlier -1. LDA K10 is a reusable checkpoint; it does not overwrite the old selected K20.

BERTopic–BatteryBERT is explicitly **pretrained**. Its fine-tuned topic-model branch has no completed saved artifact, despite the initial task assumption that all models had finished. It is not inferred from other assignments and is not silently relabeled as fine-tuned. The main Search encoder is independently the existing fine-tuned BatteryBERT checkpoint. Dashboard labels are conservative descriptions grounded in saved keywords/representatives; they are not expert scientific annotations.

C_v uses the frozen Stage1 tokens, top10 keywords, window110 and the recorded original coherence protocol. Topic Diversity = distinct top10 tokens/(10×research topics). Outlier Rate = count(topic=-1)/13,702 for BERTopic; not applicable to LDA. Top10 BERTopic keyword weights are actual saved c-TF-IDF; LDA weights are topic-word probabilities. These weights are not TF-IDF search scores or relevance probabilities.

The pretrained BatteryBERT model has high C_v but two very large clusters strongly separate Publication performance text from Patent preparation/claim text. Inspect size/source balance and representatives before concluding semantic superiority. Sodium-cobaltate superconductivity and lithium-cell hardware appear in the frozen corpus; no documents were removed to improve metrics.

Exports: `topics/lda`, `topics/bge`, `topics/batterybert`: summary, keyword weights, representatives, per-year/source distributions and assignments. Radar axes are explicit [0,1] ratios: corpus share, Publication share, source balance=1−|2p−1|, and ≥2020 share among documents with known year. Missing years are not imputed.
'''
    search=f'''# Search comparison

| Configuration | Lexical / dense retrieval | Fusion / reranking | Final score |
| --- | --- | --- | --- |
| TFIDF_COSINE | Full cleaned Title–Abstract, chemical-aware tokens, smoothed IDF, L2 | Sparse cosine Top10 | TF-IDF cosine |
| SEARCH_V2 | Original BM25 + original BGE pretrained, Top100 each | Original RRF k=60 → Top50 → MiniLM | MiniLM cross-encoder score |
| SEARCH_V3 | Same BM25 + BatteryBERT fine-tuned/Qdrant, Top100 each | RRF k=60 → Top50 → same MiniLM | MiniLM cross-encoder score |

No raw BM25/dense/MiniLM score addition. RRF(d)=Σ 1/(60+rank_i(d)). Each result is unique. Original normalization, case-sensitive chemical lexical channels, source/year and explicit required-term constraints are reused. V2's conservative lexical-support gate is retained for V3; valid semantic paraphrases with no lexical overlap may abstain. MiniLM sees at most two query-aware abstract passages per candidate; it does not read every full text. TF-IDF excerpts are lexical-overlap passages, not neural explanations.

Outputs contain rank, title, abstract, source, year, doc_id, identifier, passage, topic label/variant, score and score method. Similarity and model scores are not calibrated relevance probabilities. `rankings_top10.csv`, `query_runs/*.json`, `run_manifest.csv` preserve actual outputs, failures are not filled with fake records. Query protocol is the frozen 24-query set (23 domain queries + one OOV edge control). All 3 methods use the same protocol. The earlier Top5 is verified against original saved rankings in `v2_top5_regression.csv`; core V2 files/config are unchanged.

## Relevance

{table(OUT/'relevance_summary.csv')}

## Actual warm query latency

{table(OUT/'latency_warm.csv')}

These serial CPU requests include actual query encoding/reranking; query vectors and neural scores are not cached between queries. The first request per method, including lazy loading, is separately retained in `latency_all_requests.csv` and the run manifest. Engine initialization is separately profiled. Process peak RSS is sampled every 50 ms. Concurrent desktop load/page-cache state can affect latency; this is not a multiuser load test. `v2_v3_rank_changes.csv` describes overlap/rank changes without claiming relevance improvement.
'''
    ev=f'''# Retrieval evaluation

Status: **{evaluation['status']}**. Accepted independent labels: {evaluation.get('accepted_independent_labels','not evaluated')}; eligible domain queries: {evaluation.get('eligible_queries',0)}/{evaluation.get('total_queries',23)}.

{table(OUT/'relevance_summary.csv')}

The newest available label sources were inspected and checksummed in `label_source_audit.csv`. Existing AI judgments, synthetic source-document recovery and title–abstract proxy MRR are not independent human relevance. They are not used to populate official @10 scores. Unlabeled pairs remain absent; unlabeled ≠ irrelevant. Pending cells are blank, never synthetic zeros.

The common pool unions all 3 methods' Top10 and the earlier shared annotation pool, deduplicated by query/doc ID. `annotation_template.csv` has blank grade/provenance fields. Import via dashboard or `run_final_system.py evaluate --labels file.csv`. Grades must be integer 0–3; evaluated rows require HUMAN_INDEPENDENT, VERIFIED, annotator, a distinct verifier and verification reference. Duplicates, unknown pairs, invalid/boolean grades and missing provenance are rejected. Partial imports are supported; a query enters metrics only when its entire shared pool has verified independent labels for all methods.

For k=10: DCG=Σ(2^grade_r−1)/log2(r+1); IDCG sorts the common judged pool and takes the best10; nDCG=DCG/IDCG (0 when IDCG=0). P@10=count(grade≥2)/10, even if fewer results. MRR@10 is 1/rank of first grade≥2, or 0. Empty successful retrievals are represented in the run manifest. The OOV edge control is excluded from domain quality means. All methods average the same eligible query set. Pool-based IDCG does not establish exhaustive corpus Recall and can have pooling/coverage bias.

Per-query values/status: `relevance_per_query.csv`; coverage: `label_coverage.csv`. Paired bootstrap resamples complete queries (seed42, 2000 replicates, V3−V2), only when ≥2 queries are eligible. No confidence interval is manufactured while labels are missing. The query set is researcher-authored and not an external benchmark.

{table(OUT/'paired_bootstrap.csv')}
'''
    opt=f'''# Optimization report

Observed parity status: **{optimization['status']}**.

{table(OUT/'optimization_profiles.csv')}

The system reuses valid corpus embeddings, existing BM25 indexes and topic exports. TF-IDF uses a normalized sparse CSR matrix; query scoring never creates a corpus×vocabulary dense matrix. Qdrant upserts in bounded batches and queries Top100. Existing partition-based Top-K preserves the full-sort score/doc-ID tie convention. RRF operates on bounded ranked lists. MiniLM batches pairs (original batch4) and reranks only the Top50 with at most two passages each. CPU threads=4. Streamlit caches topic data and a single search service, not unbounded query results. BGE and BatteryBERT encoders are released when switching to the other encoder; only one large query encoder is active. There is no per-query corpus encoding.

Before/after microbenchmarks above compare full sort vs partition with identical scores and MiniLM batch1 vs batch4 on the same real query-passages. They record wall latency and sampled process-wide peak RSS; they are not relevance experiments and do not establish independent memory savings. Pair scores/order and TopK IDs must match at recorded tolerances. No quantization, HNSW approximation or accuracy-reducing pruning was enabled. The full prefix inference cache is exact float32 and read-only, with direct-forward parity evidence in `batterybert_finetuned/prefix_parity.json`.

Live system latency/RAM:

{table(OUT/'latency_all_requests.csv')}

Model loading and initialization are separate; serial warm requests are in `latency_warm.csv`. RAM figures cover the Python process, not the entire OS. This host has 8 GB RAM and no CUDA execution; other applications affect available memory.
'''
    components=dict(topic_exports='PASS' if (OUT/'topic_comparison.csv').exists() else 'NOT_EXECUTED',
        fine_tuned_document_embeddings=embedding['status'],qdrant=qdrant['status'] if len(qdrant.get('collections',[]))==2 else 'NOT_COMPLETE',
        live_three_method_benchmark=info(OUT/'benchmark_success.json')['status'],
        filtered_three_method_smoke=info(OUT/'filtered_search_smoke.json')['status'],
        dashboard_http_startup=info(OUT/'dashboard_server_smoke.json')['status'],
        independent_relevance=evaluation['status'],optimization=optimization['status'],preservation=preservation['status'],
        model_loading='PASS' if len(loading)==5 and all(r['status']=='PASS' for r in loading) else 'NOT_COMPLETE',
        tests='PASS' if len(tests)==2 and all(not r['failures'] and not r['errors'] and not r['skipped'] for r in tests) else 'NOT_COMPLETE')
    final=f'''# Final system report

## Verified component status

{chr(10).join('- '+k+': **'+v+'**' for k,v in components.items())}

The integration uses the real frozen 13,702-document corpus and existing checkpoints. No training or topic fitting was executed. The original model-comparison run had saved the trained BatteryBERT weights but stopped at tokenizer serialization checking (only runtime padding/truncation differed). The new integration verifies native tokenizer semantics and frozen weights without modifying any old tokenizer/model/cache. Missing fine-tuned corpus vectors are inferred to a new directory, never substituted with pretrained vectors. Their successful validation is required for Search V3.

Existing BatteryBERT checkpoint evidence: validation title–abstract group MRR@10={training.get('best_score')}, step={training.get('best_step')}, weight SHA={training.get('weights',{}).get('model.safetensors')}. This selects among checkpoints for that capped proxy; it is not verified Search relevance and does not establish a best retrieval encoder. The prior one-epoch training happened before this integration; this task performs zero gradient updates.

Fine-tuned embeddings status={embedding['status']}, shape={embedding.get('documents','pending')}×{embedding.get('dimension','pending')}, full abstract coverage={embedding.get('full_abstract_coverage','pending')}. Existing pretrained/full-doc BGE vectors remain distinct in `artifact_inventory.json`. Topic exports use LDA K10, BERTopic–BGE K10 and BERTopic–BatteryBERT **pretrained** K10. Fine-tuned BERTopic artifacts are missing and were not fitted. Search V2 remains the original pretrained-BGE ranking and supports both Top5 and Top10; Search V3 uses the fine-tuned BatteryBERT dense channel.

## Validation evidence

{chr(10).join('- '+str(row) for row in tests) or '- Final tests not executed yet.'}

`model_loading.json` records isolated real reloads of LDA, both BERTopic models, BGE and BatteryBERT. `qdrant_validation.json` checks every stored payload/vector and exact retrieval probes. `v2_top5_regression.csv` compares actual outputs to original saved rankings. `preservation_check.json` rehashes {preservation.get('protected_files','pending')} protected old files; changed={preservation.get('changed','not checked')}. Historical initial test failures, including an early index-build race, are retained separately; acceptance uses the final XML results above.

## Remaining scientific limits

Official nDCG@10/P@10/MRR@10 and confidence intervals require enough independent verified judgments: current status **{evaluation['status']}**. No empty score cells were filled with AI, similarity or coherence. BERTopic fine-tuned output is unavailable and the existing pretrained topic model is explicitly identified. No new topic fitting or HNSW trial was authorized. No winner is claimed. V2/V3 comparison includes encoder tokenization/pooling/document-context differences. Topic labels need expert review and some frozen-corpus topics are outside the core sodium-ion scope. MiniLM passage selection and the lexical-support gate can miss relevant paraphrases/evidence. CPU latency is serial on one desktop, not production throughput.

## Artifacts and running

New outputs: `outputs/final_system/`; reports: SYSTEM_ARCHITECTURE.md, QDRANT_REPORT.md, TOPIC_MODEL_COMPARISON.md, SEARCH_COMPARISON.md, RETRIEVAL_EVALUATION.md, OPTIMIZATION_REPORT.md, FINAL_SYSTEM_REPORT.md. Existing research artifacts remain at their original paths, recorded in the inventory and protected SHA manifest.

{commands}
'''
    for name,content in dict(SYSTEM_ARCHITECTURE=architecture,QDRANT_REPORT=qr,TOPIC_MODEL_COMPARISON=topic,
        SEARCH_COMPARISON=search,RETRIEVAL_EVALUATION=ev,OPTIMIZATION_REPORT=opt,FINAL_SYSTEM_REPORT=final).items():
        (OUT/(name+'.md')).write_text(content,encoding='utf-8')
    source_paths=[Path(p) for p in ('app.py','run_final_system.py','scripts/prepare_final_battery_vectors.py',
        'scripts/smoke_final_dashboard_server.py',
        'src/batterybert.py','src/bertopic_chunked.py','src/model_comparison_io.py','src/stage3_common.py',
        'src/search_v2.py','src/search_v2_constraints.py','configs/model_comparison.yaml','configs/search_v2.json')]
    source_paths += [
                  *[p.relative_to(ROOT) for p in (ROOT/'src').glob('final_*.py')]]
    write_json(OUT/'implementation_manifest.json',dict(python=platform.python_version(),
        dependencies={name:version(name) for name in ('numpy','pandas','torch','transformers','streamlit','qdrant-client','scikit-learn')},
        source_sha256={p.as_posix():sha(ROOT/p) for p in source_paths},reports_sha256={p.name:sha(p) for p in OUT.glob('*.md')}))
    write_json(OUT/'final_status.json',dict(components=components,tests=tests,training_invoked=False,topic_fit_invoked=False,
        quality_claim='No independent relevance superiority claim; pending labels stay blank'))
    write_json(OUT/'artifact_manifest.json',dict(
        files_sha256={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*')
            if p.is_file() and not p.is_relative_to(OUT/'qdrant') and p.name!='artifact_manifest.json'},
        qdrant_storage='Validated by full payload/vector comparison; live SQLite/lock bytes excluded'))
