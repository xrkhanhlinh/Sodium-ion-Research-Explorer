"""Infer only MISSING fine-tuned corpus vectors; never train or modify old artifacts.

Read-only, SHA-checked frozen-prefix reuse is exact float32. Missing prefix rows
use a full forward pass and are NOT written to the earlier experiment cache.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['USE_TF'] = os.environ['USE_FLAX'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from src.batterybert import BatteryEncoder, tensor_hash, runtime
from src.stage3_common import digest, signature, read_json
from src.model_comparison_io import write_json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/model_comparison_v2'
OUT = ROOT / 'outputs/final_system/batterybert_finetuned'


def tokenizer_identity(tokenizer):
    state = json.loads(tokenizer.backend_tokenizer.to_str())
    return {k: v for k, v in state.items() if k not in ('padding', 'truncation')}


def run():
    runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    trained = OLD / 'batterybert_finetuned/model'
    pretrained = OLD / 'batterybert_pretrained'
    success = read_json(OLD / 'batterybert_finetuned/training_success.json')
    assert success['status'] == 'PASS' and success['real_gradient_updates']
    assert digest(trained / 'model.safetensors') == success['weights']['model.safetensors']
    identity = dict(weights=success['weights'], chunks=digest(pretrained / 'chunks.parquet'),
                    corpus=digest(ROOT / 'data/processed/main_corpus.parquet'),
                    source=digest(Path(__file__)), pooling='masked mean + L2; mean chunks + L2')
    done = OUT / 'embedding_success.json'
    if done.exists():
        info = read_json(done)
        assert info['identity'] == identity
        assert digest(OUT / 'document_embeddings.npy') == info['sha256']
        print('REUSE VERIFIED fine-tuned embeddings', flush=True)
        return
    adapter = BatteryEncoder(trained)
    from transformers import AutoTokenizer
    native = AutoTokenizer.from_pretrained(pretrained / 'model', local_files_only=True)
    assert tokenizer_identity(adapter.tokenizer) == tokenizer_identity(native), 'Tokenizer semantics changed'
    # Transformers persists the most recent padding/truncation call on save.
    # These runtime defaults do not affect explicit saved native token ID windows.
    adapter.tokenizer.backend_tokenizer.no_padding()
    adapter.tokenizer.backend_tokenizer.no_truncation()
    samples = ['Na3V2(PO4)3 cathodes and NaFSI ether electrolytes', 'Hard carbon ' * 200]
    for text in samples:
        assert adapter.tokenizer(text, truncation=False)['input_ids'] == native(text, truncation=False)['input_ids']
    chunks = pd.read_parquet(pretrained / 'chunks.parquet')
    docs = pd.read_parquet(pretrained / 'document_chunk_counts.parquet')
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet', columns=['doc_id'])
    assert docs.doc_id.tolist() == main.doc_id.tolist() and len(main) == 13702
    cm = read_json(pretrained / 'chunking_success.json')
    assert cm['chunks_sha256'] == identity['chunks'] and cm['coverage'] == 1.
    assert digest(pretrained / 'document_chunk_counts.parquet') == cm['documents_sha256']
    prefix = pretrained / 'frozen_document_prefix'
    pm = read_json(prefix / 'manifest.json')
    frozen = tensor_hash([(n, p) for n, p in adapter.model.named_parameters()
                          if not n.startswith('encoder.layer.11.')])
    assert frozen == pm['frozen_hash'] == success['frozen_hash_after']
    assert pm['chunks_sha256'] == identity['chunks']
    hidden = np.load(prefix / 'hidden.npy', mmap_mode='r')
    valid = np.load(prefix / 'valid.npy', mmap_mode='r')
    hashes = np.load(prefix / 'row_sha256.npy', mmap_mode='r')
    marker = OUT / 'embedding_progress.json'
    state = read_json(marker) if marker.exists() else dict(identity=identity, completed_documents=0, seconds=0.,
        reused_prefix_chunks=0, full_forward_chunks=0)
    assert state['identity'] == identity, 'Resume identity mismatch'
    vectors = np.lib.format.open_memmap(OUT / 'document_embeddings.npy', mode='r+' if marker.exists() else 'w+',
                                       dtype='float32', shape=(len(main), 768))
    parity = False
    tick = time.perf_counter()
    with torch.inference_mode():
        for index in range(state['completed_documents'], len(main)):
            doc = docs.iloc[index]
            start, end = int(doc.chunk_start), int(doc.chunk_start + doc.chunk_count)
            rows = chunks.iloc[start:end].to_dict('records')
            parts = []
            for offset in range(0, len(rows), 2):
                batch = rows[offset:offset + 2]
                features = adapter.record_features(batch)
                indices = list(range(start + offset, start + offset + len(batch)))
                if all(valid[i] for i in indices):
                    states = []
                    width = features['input_ids'].shape[1]
                    for i in indices:
                        row = hidden[i]
                        assert hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest().encode() == hashes[i]
                        assert np.isfinite(row).all()
                        states.append(np.array(row[:width], copy=True))
                    result = adapter.from_prefix(torch.from_numpy(np.stack(states)), features['attention_mask'])
                    state['reused_prefix_chunks'] += len(indices)
                    if not parity:
                        full = adapter.forward(features)
                        error = float(torch.max(torch.abs(full - result)))
                        assert torch.allclose(full, result, atol=2e-5)
                        write_json(OUT / 'prefix_parity.json', dict(status='PASS', max_abs_error=error,
                            native_tokenizer_semantics_identical=True, old_cache_read_only=True))
                        parity = True
                else:
                    result = adapter.forward(features)
                    state['full_forward_chunks'] += len(indices)
                parts.append(result.numpy())
            vector = np.concatenate(parts).mean(axis=0)
            vector /= np.linalg.norm(vector)
            vectors[index] = vector
            if (index + 1) % 32 == 0 or index + 1 == len(main):
                vectors.flush()
                state.update(completed_documents=index + 1, seconds=state['seconds'] + time.perf_counter() - tick)
                write_json(marker, state)
                tick = time.perf_counter()
                print(f"EMBED {index + 1}/13702 seconds={state['seconds']:.1f}", flush=True)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)
    write_json(OUT / 'ordered_doc_ids.json', main.doc_id.tolist())
    base = np.load(pretrained / 'document_embeddings.npy', mmap_mode='r')
    cosine = np.sum(vectors * base, axis=1)
    write_json(done, dict(status='PASS', identity=identity, sha256=digest(OUT / 'document_embeddings.npy'),
        ordered_doc_ids_sha256=digest(OUT / 'ordered_doc_ids.json'), documents=len(main), dimension=768,
        full_abstract_coverage=True, chunks=len(chunks), pooling=identity['pooling'],
        query_instruction=False, model_path=str(trained.relative_to(ROOT)),
        pretrained_finetuned_cosine_mean=float(cosine.mean()),
        pretrained_finetuned_cosine_min=float(cosine.min()), **{k:v for k,v in state.items() if k != 'identity'}))
    print('COMPLETE: fine-tuned embeddings; no training or topic fitting', flush=True)


if __name__ == '__main__':
    run()
