# Reproducing the LDA / BGE / BatteryBERT topic comparison

Run from `D:/daln` with the existing Python 3.10 environment:

```powershell
.venv/Scripts/python.exe run_model_comparison.py
```

Outputs are isolated under `outputs/model_comparison_v2/`. The runner hashes
the old data, models, outputs and original configuration files. It never calls
the Stage1 processing pipeline, the Stage2 selection writer or BGE fine-tuning.
Completed phases are reused only when their code/config/dependency fingerprints
and artifact SHA-256 hashes match. A stale process lock is detected by PID plus
process creation time; a second active runner is refused.

Each expensive phase runs in a fresh subprocess. This releases the transformer
before BERTopic imports UMAP/HDBSCAN, and releases clustering memory before the
next encoder loads. No transformer runs in parallel with another transformer.

To run a selected sequence, keep its dependency order:

```powershell
.venv/Scripts/python.exe run_model_comparison.py --phases preflight lda bge
.venv/Scripts/python.exe run_model_comparison.py --phases download battery_preflight
.venv/Scripts/python.exe run_model_comparison.py --phases proxy_pretrained embed_pretrained topics_pretrained train proxy_finetuned embed_finetuned topics_finetuned verify report
```

The first download requires access to Hugging Face. The revision is pinned in
`batterybert_pretrained/model/download_provenance.json`; later loads are local.
Only that public model is downloaded. Project documents are not uploaded.

To inspect progress without loading a model:

```powershell
Get-Content outputs/model_comparison_v2/run_status.json
Get-Content outputs/model_comparison_v2/logs/proxy_pretrained.log -Tail 10
Get-Content outputs/model_comparison_v2/logs/embed_pretrained.log -Tail 10
Get-Content outputs/model_comparison_v2/batterybert_finetuned/training_progress.json
```

`--phases report` can refresh the partial comparison while a worker is running.
A report lists absent experiments as `not_completed`, never as zero-valued
metrics. `verification.json` distinguishes verification of existing artifacts
from completion of all four topic branches (LDA, BGE and both BatteryBERT variants).

## Resume boundaries

- Document encoding commits every 32 whole documents. A partial last interval
  is safely recomputed; completed document/chunk vectors use memory maps.
- Atomic JSON/CSV and training-checkpoint replacement retries transient
  Windows sharing violations, including a reader holding a progress file open.
  Permanent denial raises and preserves the previous committed file and the
  pending temporary file; it is never treated as successful progress.
- Proxy encoding and validation prefix caches commit every 64 sequences.
- Training commits weights, AdamW state, RNG, histories and the next batch at
  optimizer boundaries every 100 microbatches, and after validation.
- Shuffling is deterministic. Every training pair is seen exactly once,
  including the odd final row, merged into a three-pair last microbatch.
- Validation caches contain only frozen-prefix hidden states in float32. Every
  validation recomputes the current last block and checks it against a full
  forward pass. No test information selects the model.
- Full-document inference also shares float32 states from the first eleven
  blocks between pretrained and fine-tuned branches. Its disk-backed cache is
  about 24 GiB for this corpus, with a SHA-256 for every padded chunk state.
  Changed frozen weights, tokenizer or chunks invalidate it; the last block
  is always recomputed. Validity bits commit after data and checksums flush.
  This cache is temporary execution support, not a new encoder configuration.
- A saved tokenizer may have different JSON formatting after Transformers
  serialization. The runner checks backend equality, then preserves the original
  tokenizer bytes so shared chunk-cache provenance remains identical.

Use a different `--output outputs/<new-name>` for a different configuration.
Do not delete markers or edit completed artifacts to force reuse.

## Scientific interpretation

The topic comparison always uses the full frozen corpus and dictionary. C_v
uses the original Stage1 token lists (window110/top10); U_Mass uses the original
Stage2 BoW; diversity uses ten keywords per research topic. Topic `-1` is excluded
from topic counts and coherence, while its document assignments are retained.
The installed BERTopic implementation includes its outlier slot in `nr_topics`,
so a ten-topic reduction requests eleven slots when outliers exist.

BatteryBERT token pooling is masked mean plus L2 in both training and inference.
The CPU training/proxy budget uses single sequences capped at 128 tokens;
full-document topic embeddings cover the full abstract with 256-token windows,
overlap64, and mean/L2 aggregation. Proxy results measure Title-to-Abstract
group matching inside a held-out split. They do not measure deployed Search@10.

LDA's document-topic matrix is a variational posterior. BERTopic's exported
document-topic matrix is an L1-normalized cosine affinity between frozen BoW
and topic c-TF-IDF, with an explicit list of topic columns. It is not a
calibrated posterior. Saved assignments, rather than argmax of this affinity
matrix, identify the density-cluster membership and outliers.

The five representative documents are selected from assigned documents.
For LDA they have the highest dominant posterior; for BERTopic they have the
highest cosine similarity to their topic's document-embedding centroid.
Interpretive labels retain the keyword/title evidence and require expert review.

Topic counts, keyword diversity and coherence can trade off. No single metric
selects a universal best model. Extra topic-stability seeds are not run in this
CPU-budgeted task, and old stability scores are not attributed to reduced models.

Technical contract tests:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_model_comparison_v2.py -q
```
