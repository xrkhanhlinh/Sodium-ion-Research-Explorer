# Isolated BatteryBERT topic audit

This is the historical 22 October audit. The main topic catalog was subsequently updated by the [23 October BatteryBERT optimization](BATTERYBERT_OPTIMIZATION_20261023.md); this audit's data/model exports remain unchanged for comparison.

Outputs: [`outputs/batterybert_topic_audit_20261022_v1`](../outputs/batterybert_topic_audit_20261022_v1/).

Read [root-cause audit](../outputs/batterybert_topic_audit_20261022_v1/root_cause_audit.md) and [optimization report](../outputs/batterybert_topic_audit_20261022_v1/optimization_report.md). The report distinguishes actual code defects from representation and corpus limitations.

The exported `bertopic_final.pickle` is a **provisional fixed-10 comparison**, not a scientifically validated replacement for the web model. No acceptable set of ten independently validated research themes has been established. Production LDA, BGE, BatteryBERT, Qdrant and Search artifacts remain on their existing paths. No encoder training or full embedding inference occurs in this audit.

The baseline and 14 new variants use the same 13,702 documents and pretrained BatteryBERT vectors. Experiments include controlled representation changes, HDBSCAN EOM/leaf, two UMAP configurations, natural and reduced topic counts, full-vector/UMAP KMeans, placeholder recovery and seed42/43 robustness checks. Parameter files and complete row assignments are saved in each `experiments/<name>/` directory. Topic IDs are local to each candidate.

```powershell
.\.venv\Scripts\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_NEW --phase all
.\.venv\Scripts\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_NEW --phase stability --candidate kmeans10_full
.\.venv\Scripts\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_NEW --phase stability --candidate kmeans10_umap
```

`--phase all` completes audit, bounded duplicate/chemical diagnostics, representation experiments, clustering experiments, shortlist representation repair and same-reference coherence evaluation. It deliberately stops before scientific labels: inspect the real representative documents and write `topic_labels.csv` plus `scientific_review.csv`. Labels must cover every selected topic and reviewed document IDs/titles must match the frozen corpus. Agent inspection is explicitly distinguished from independent expert validation. Do not copy topic labels between different assignments.

For the frozen run with recorded review:

```powershell
.\.venv\Scripts\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_20261022_v1 --phase finalize --candidate kmeans10_full_repaired
.\.venv\Scripts\python.exe -m pytest tests/test_batterybert_topic_audit.py tests/test_model_comparison_v2.py -q --junitxml=outputs/batterybert_topic_audit_20261022_v1/tests.xml
.\.venv\Scripts\python.exe -m pytest tests/test_batterybert_topic_audit_outputs.py -q --junitxml=outputs/batterybert_topic_audit_20261022_v1/integration_tests.xml
.\.venv\Scripts\python.exe scripts/audit_batterybert_topics.py --output outputs/batterybert_topic_audit_20261022_v1 --phase report
```

A new run refuses an unrelated populated output directory. Completed experiments are retained, and input fingerprint mismatches require a new version. Inputs and vectors are read-only; model files are saved only under the chosen experiment directory. Do not put the output directory inside an existing baseline directory.

Keyword extraction uses its own vocabulary fitted on the **original documents**, with min_df=3 and max_features=60,000. That fixed vocabulary is then used for aggregated topic documents with min_df=1. Technical words are protected; common words such as material, source, method and performance are retained when context-sensitive. Cross-document n-grams are blocked. The legacy hyphenated phrase-placeholder leak is repaired only in the isolated representation text; frozen LDA tokens are preserved.

Coherence uses a single reference with exact observed keyword phrases encoded atomically; missing support and partial coverage are explicit. Old baseline coherence from another reference is stored separately. Seed checks are same-corpus robustness diagnostics, not held-out accuracy. Detailed review examples and scope flags are preserved without deleting documents.

The final model was loaded and queried with explicit existing vectors; all12 probes matched saved assignments. When using `BERTopic.load`, retain the recorded local package versions and project import path. Always provide embeddings from the pinned BatteryBERT checkpoint and the recorded chunked masked-mean/L2 pooling protocol; there is no automatic fallback encoder.
