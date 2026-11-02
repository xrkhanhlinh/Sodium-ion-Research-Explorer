"""CPU BatteryBERT mean-pooling adapter, resumable embeddings and contrastive fit.

No SentenceTransformer or BGE instructions are used. Only the last encoder
block is trained. Validation's frozen prefix is cached as float32 memmaps;
the updated last block is always evaluated afresh and checked against forward.
"""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import random
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '4'

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from .stage3_common import digest, signature, read_json, group_retrieval_metrics
from .model_comparison import cfg, corpus, validate_vectors, log
from .model_comparison_io import write_json, write_csv as table, replace_with_retry


def runtime():
    torch.set_num_threads(cfg()['threads'])
    torch.set_num_interop_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def masked_mean(hidden, attention_mask):
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1), p=2, dim=1)


class BatteryEncoder:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(str(folder), local_files_only=True,
                                               add_pooling_layer=False, trust_remote_code=False)
        self.model.eval()
        assert self.model.config.model_type == 'bert'
        self.dimension = self.model.config.hidden_size
        self.maximum = self.model.config.max_position_embeddings

    def features(self, texts, max_length=128):
        return self.tokenizer(list(texts), padding=True, truncation=True,
                              max_length=max_length, return_tensors='pt')

    def forward(self, features):
        output = self.model(**features).last_hidden_state
        assert output.shape[-1] == self.dimension
        return masked_mean(output, features['attention_mask'])

    def record_features(self, records):
        clean = [{k: [int(value) for value in row[k]] for k in ('input_ids', 'token_type_ids')} for row in records]
        return self.tokenizer.pad(clean, padding=True, return_tensors='pt')

    def prefix(self, features):
        h = self.model.embeddings(input_ids=features['input_ids'], token_type_ids=features.get('token_type_ids'))
        att, _ = self.model._create_attention_masks(features['attention_mask'], None, h, None, None)
        for layer in self.model.encoder.layer[:-1]:
            h = layer(h, attention_mask=att)
            if isinstance(h, tuple):
                h = h[0]
        return h

    def from_prefix(self, hidden, attention):
        att, _ = self.model._create_attention_masks(attention, None, hidden, None, None)
        output = self.model.encoder.layer[-1](hidden, attention_mask=att)
        if isinstance(output, tuple):
            output = output[0]
        return masked_mean(output, attention)


def weight_identity(folder):
    paths = [p for pattern in ('*.safetensors', 'pytorch_model*.bin') for p in Path(folder).glob(pattern)]
    assert paths, 'No checkpoint weights'
    return {p.name: digest(p) for p in paths}


def tensor_hash(named):
    h = hashlib.sha256()
    for name, value in named:
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class DocumentPrefixCache:
    """Exact float32 disk cache of the frozen first N-1 BERT blocks.

    Validity bits are committed only after hidden states and their per-row
    SHA-256 hashes are flushed. Missing rows can be filled during either
    pretrained or fine-tuned inference; updated last-block states are never
    cached here. This trades disk space for CPU time without quantization.
    """
    def __init__(self, folder, adapter, chunk_sha256, count):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        last = 'encoder.layer.' + str(len(adapter.model.encoder.layer) - 1) + '.'
        frozen_hash = tensor_hash([(n, p) for n, p in adapter.model.named_parameters()
                                  if not n.startswith(last)])
        self.shape = (count, 256, adapter.dimension)
        identity = signature(dict(chunks=chunk_sha256, frozen_hash=frozen_hash,
            shape=self.shape, dtype='float32', tokenizer=digest(adapter.folder / 'tokenizer.json')))
        marker = folder / 'manifest.json'
        existing = marker.exists()
        if existing:
            info = read_json(marker)
            if info['fingerprint'] != identity:
                raise ValueError('Full-document prefix cache differs from frozen model/tokenizer/chunks')
        else:
            required = int(np.prod(self.shape)) * 4
            if shutil.disk_usage(folder).free < required + 2 * 1024 ** 3:
                raise RuntimeError(f'Frozen-prefix cache needs {required / 1024**3:.2f} GiB plus 2 GiB free headroom')
        mode = 'r+' if existing else 'w+'
        self.hidden = np.lib.format.open_memmap(folder / 'hidden.npy', mode=mode,
            dtype='float32', shape=self.shape)
        self.hashes = np.lib.format.open_memmap(folder / 'row_sha256.npy', mode=mode,
            dtype='S64', shape=(count,))
        self.valid = np.lib.format.open_memmap(folder / 'valid.npy', mode=mode,
            dtype='bool', shape=(count,))
        self.pending = set()
        if not existing:
            self.valid[:] = False
            self.valid.flush()
            write_json(marker, dict(fingerprint=identity, frozen_hash=frozen_hash,
                shape=list(self.shape), dtype='float32', chunks_sha256=chunk_sha256,
                integrity='SHA-256 per full padded hidden-state row; validity bits flushed after data/hashes',
                final_encoder_block_cached=False))
        self.identity = identity

    @staticmethod
    def row_hash(row):
        return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest().encode('ascii')

    def get(self, adapter, features, indices):
        width = features['input_ids'].shape[1]
        if all(self.valid[i] for i in indices):
            values = []
            for i in indices:
                row = self.hidden[i]
                if self.row_hash(row) != self.hashes[i]:
                    raise ValueError(f'Frozen-prefix cache SHA-256 mismatch at chunk {i}')
                if not np.isfinite(row).all():
                    raise ValueError('Nonfinite frozen-prefix states')
                values.append(np.array(row[:width], copy=True))
            return torch.from_numpy(np.stack(values))
        hidden = adapter.prefix(features)
        for j, i in enumerate(indices):
            if not self.valid[i]:
                self.hidden[i] = 0
                self.hidden[i, :width] = hidden[j].detach().cpu().numpy()
                self.hashes[i] = self.row_hash(self.hidden[i])
                self.pending.add(i)
        return hidden

    def flush(self):
        if self.pending:
            self.hidden.flush()
            self.hashes.flush()
            self.valid[list(self.pending)] = True
            self.valid.flush()
            self.pending.clear()


def encode_texts(adapter, texts, path):
    """Proxy single-sequence embeddings, with atomic progress and exact cache ID."""
    maximum = cfg()['batterybert']['training_max_tokens']
    identity = signature(dict(weights=weight_identity(adapter.folder), texts=list(texts),
        maximum=maximum, pooling=cfg()['batterybert']['pooling'], tokenizer=digest(adapter.folder / 'tokenizer.json')))
    marker = path.with_suffix('.progress.json')
    state = read_json(marker) if marker.exists() else dict(fingerprint=identity, completed=0)
    if state['fingerprint'] != identity:
        raise ValueError('Proxy embedding cache differs from model/text/pooling configuration')
    arr = np.lib.format.open_memmap(path, mode='r+' if marker.exists() else 'w+',
                                   dtype='float32', shape=(len(texts), adapter.dimension))
    batch_size = cfg()['batterybert']['batch_size']
    adapter.model.eval()
    with torch.inference_mode():
        for start in range(state['completed'], len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            arr[start:end] = adapter.forward(adapter.features(texts[start:end], maximum)).numpy()
            if end % 64 == 0 or end == len(texts):
                arr.flush()
                state['completed'] = end
                write_json(marker, state)
                log(f'PROXY {path.name}: {end}/{len(texts)}')
    validate_vectors(path, list(range(len(texts))), list(range(len(texts))), adapter.dimension)
    return arr


def frozen_validation(adapter, texts, folder, field):
    """Memory-bounded exact prefix cache, shared between before/after weights."""
    maximum = cfg()['batterybert']['training_max_tokens']
    named = [(n, p) for n, p in adapter.model.named_parameters() if not n.startswith('encoder.layer.' + str(len(adapter.model.encoder.layer) - 1) + '.')]
    identity = signature(dict(texts=list(texts), frozen_hash=tensor_hash(named), maximum=maximum,
                             dimension=adapter.dimension, pooling=cfg()['batterybert']['pooling']))
    folder.mkdir(parents=True, exist_ok=True)
    path, maskpath = folder / (field + '.npy'), folder / (field + '_mask.npy')
    marker = folder / (field + '.progress.json')
    state = read_json(marker) if marker.exists() else dict(fingerprint=identity, completed=0)
    if state['fingerprint'] != identity:
        raise ValueError('Validation frozen-prefix cache does not match the current frozen weights/texts')
    mode = 'r+' if marker.exists() else 'w+'
    hidden = np.lib.format.open_memmap(path, mode=mode, dtype='float32', shape=(len(texts), maximum, adapter.dimension))
    masks = np.lib.format.open_memmap(maskpath, mode=mode, dtype='int64', shape=(len(texts), maximum))
    bs = cfg()['batterybert']['batch_size']
    adapter.model.eval()
    with torch.inference_mode():
        for start in range(state['completed'], len(texts), bs):
            end = min(start + bs, len(texts))
            features = adapter.features(texts[start:end], maximum)
            h = adapter.prefix(features)
            width = h.shape[1]
            hidden[start:end] = 0
            masks[start:end] = 0
            hidden[start:end, :width] = h.numpy()
            masks[start:end, :width] = features['attention_mask'].numpy()
            if end % 64 == 0 or end == len(texts):
                hidden.flush()
                masks.flush()
                state['completed'] = end
                write_json(marker, state)
                log(f'VALIDATION frozen {field}: {end}/{len(texts)}')
        result = np.empty((len(texts), adapter.dimension), dtype=np.float32)
        for start in range(0, len(texts), 4):
            end = min(start + 4, len(texts))
            width = int(masks[start:end].sum(axis=1).max())
            h = torch.from_numpy(np.array(hidden[start:end, :width], copy=True))
            attention = torch.from_numpy(np.array(masks[start:end, :width], copy=True))
            result[start:end] = adapter.from_prefix(h, attention).numpy()
        direct = adapter.forward(adapter.features(texts[:2], maximum)).numpy()
        error = float(np.max(np.abs(result[:2] - direct)))
        assert np.allclose(result[:2], direct, atol=2e-5), 'Frozen prefix does not reproduce the full encoder'
        write_json(folder / (field + '_parity.json'), dict(status='PASS', max_abs_error=error,
            full_precision_cache=True, last_block_recomputed=True))
    return result


def pairs_for(split):
    frame = pd.read_parquet(ROOT / 'data/stage3/contrastive_pairs.parquet')
    assert frame.groupby('split_group_id')['split'].nunique().max() == 1
    return frame[frame.split.eq(split)].reset_index(drop=True)


def evaluate_proxy(adapter, out, variant):
    folder = out / ('batterybert_' + variant)
    target = folder / 'proxy'
    target.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in ('validation', 'test'):
        frame = pairs_for(split)
        if split == 'validation':
            cache = out / 'batterybert_pretrained/validation_frozen'
            q = frozen_validation(adapter, frame.title.tolist(), cache, 'titles')
            d = frozen_validation(adapter, frame.abstract.tolist(), cache, 'abstracts')
            np.save(target / (split + '_titles.npy'), q)
            np.save(target / (split + '_abstracts.npy'), d)
        else:
            q = encode_texts(adapter, frame.title.tolist(), target / (split + '_titles.npy'))
            d = encode_texts(adapter, frame.abstract.tolist(), target / (split + '_abstracts.npy'))
        metrics, ranks = group_retrieval_metrics(q, d, frame.split_group_id.to_numpy(), frame.doc_id.to_numpy())
        metrics.update(split=split, proxy_only=True,
            candidate_scope='All abstracts within the same held-out split; all same-group matches relevant',
            max_tokens=cfg()['batterybert']['training_max_tokens'], query_instruction=False)
        results[split] = metrics
        table(target / (split + '_ranks.csv'), ranks)
        table(target / (split + '_ordered_doc_ids.csv'), frame[['doc_id', 'split_group_id']])
        log(f'BatteryBERT {variant} {split} Group MRR@10 = {metrics["group_mrr_at_10"]:.6f}')
    write_json(folder / 'proxy_metrics.json', results)


def preflight(out):
    folder = out / 'batterybert_pretrained'
    adapter = BatteryEncoder(folder / 'model')
    provenance = read_json(folder / 'model/download_provenance.json')
    for name, expected in provenance['hashes'].items():
        assert digest(folder / 'model' / name) == expected
    tokenizer = adapter.tokenizer
    from .bertopic_chunked import chunk_document, bert_features
    assert tokenizer.is_fast and adapter.maximum >= 256
    for title, abstract in [('Sodium-ion cells', 'Hard carbon anode.'), ('Na+ transport', None)]:
        first = tokenizer(title, add_special_tokens=False)['input_ids']
        second = tokenizer(abstract, add_special_tokens=False)['input_ids'] if abstract else None
        expected = tokenizer(title, abstract) if abstract else tokenizer(title)
        built = bert_features(tokenizer, first, second)
        assert built['input_ids'] == expected['input_ids']
        assert built['token_type_ids'] == expected['token_type_ids']
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet', columns=['title', 'abstract'])
    texts = main.title.iloc[:2].tolist()
    with torch.inference_mode():
        features = adapter.features(texts)
        actual = adapter.forward(features)
        h = adapter.model(**features).last_hidden_state
        manual = F.normalize(torch.stack([x[m.bool()].mean(dim=0) for x, m in zip(h, features['attention_mask'])]), dim=1)
        assert torch.allclose(actual, manual, atol=1e-6)
        single = adapter.forward(adapter.features(texts[:1]))
        assert torch.allclose(actual[:1], single, atol=2e-5)
        cached = adapter.from_prefix(adapter.prefix(features), features['attention_mask'])
        assert torch.allclose(actual, cached, atol=2e-5)
    # Test actual long corpus input, including final abstract tokens.
    index = main.abstract.str.len().idxmax()
    row = main.loc[index]
    chunks, stats = chunk_document(tokenizer, row.title, row.abstract)
    assert chunks[-1]['token_end'] == stats['abstract_tokens']
    assert stats['covered_abstract_tokens'] == stats['abstract_tokens']
    write_json(folder / 'preflight.json', dict(status='PASS', model_id=cfg()['batterybert']['model_id'],
        revision=provenance['revision'], actual_hidden_dimension=int(h.shape[-1]),
        hidden_dimension_config=adapter.dimension, maximum_positions=adapter.maximum,
        tokenizer_class=type(tokenizer).__name__, do_lower_case=getattr(tokenizer, 'do_lower_case', None),
        checks=['Real AutoModel forward', 'Manual masked-mean parity', 'Padding invariance',
                'Frozen prefix + final block parity', 'Tokenizer template parity', 'Full long abstract coverage'],
        long_abstract_tokens=stats['abstract_tokens'], long_document_chunks=len(chunks),
        initialization='Public pretrained weights; no pretraining from scratch'))
    log(f'BatteryBERT real preflight PASS; actual hidden dimension={h.shape[-1]}')


def embed(out, variant):
    folder = out / ('batterybert_' + variant)
    adapter = BatteryEncoder(folder / 'model')
    main = pd.read_parquet(ROOT / 'data/processed/main_corpus.parquet')
    assert len(main) == 13702 and main.doc_id.is_unique
    from .bertopic_chunked import chunk_document
    # Chunking is encoder-weight independent, so the fine-tuned branch reuses it.
    chunk_folder = out / 'batterybert_pretrained'
    chunk_path = chunk_folder / 'chunks.parquet'
    doc_path = chunk_folder / 'document_chunk_counts.parquet'
    chunk_marker = chunk_folder / 'chunking_success.json'
    chunk_identity = signature(dict(ids=main.doc_id.tolist(),
        corpus=digest(ROOT / 'data/processed/main_corpus.parquet'),
        tokenizer=digest(adapter.folder / 'tokenizer.json'), tokens=256, overlap=64))
    if chunk_marker.exists():
        info = read_json(chunk_marker)
        assert info['fingerprint'] == chunk_identity
        assert digest(chunk_path) == info['chunks_sha256'] and digest(doc_path) == info['documents_sha256']
    else:
        rows, docs = [], []
        for index, row in enumerate(main.itertuples()):
            chunks, stats = chunk_document(adapter.tokenizer, row.title, row.abstract)
            docs.append(dict(doc_index=index, doc_id=row.doc_id, chunk_start=len(rows), **stats))
            for record in chunks:
                rows.append(dict(doc_index=index, doc_id=row.doc_id, **record))
            if (index + 1) % 2000 == 0:
                log(f'BatteryBERT CHUNK {index + 1}/13702')
        pd.DataFrame(rows).to_parquet(chunk_path, index=False)
        frame = pd.DataFrame(docs)
        frame.to_parquet(doc_path, index=False)
        assert frame.abstract_tokens.sum() == frame.covered_abstract_tokens.sum()
        write_json(chunk_marker, dict(fingerprint=chunk_identity, chunks_sha256=digest(chunk_path),
            documents_sha256=digest(doc_path), chunks=len(rows), documents=len(frame),
            abstract_tokens=int(frame.abstract_tokens.sum()), coverage=1.,
            title_context_truncated_documents=int(frame.title_truncated.sum()),
            title_only_documents=int(frame.title_only.sum())))
        del rows, docs, frame
    chunks = pd.read_parquet(chunk_path)
    docs = pd.read_parquet(doc_path)
    assert docs.doc_id.tolist() == main.doc_id.tolist()
    identity = signature(dict(chunks=digest(chunk_path), docs=digest(doc_path), weights=weight_identity(adapter.folder),
                              settings=cfg()['batterybert'], dimension=adapter.dimension))
    marker = folder / 'embedding_progress.json'
    state = read_json(marker) if marker.exists() else dict(fingerprint=identity, completed_documents=0, seconds=0.)
    if state['fingerprint'] != identity:
        raise ValueError('Document embeddings cannot resume with different model/chunks/config')
    embeddings = np.lib.format.open_memmap(folder / 'document_embeddings.npy',
        mode='r+' if marker.exists() else 'w+', dtype='float32', shape=(len(main), adapter.dimension))
    chunk_embeddings = np.lib.format.open_memmap(folder / 'chunk_embeddings.npy',
        mode='r+' if marker.exists() else 'w+', dtype='float32', shape=(len(chunks), adapter.dimension))
    bs = cfg()['batterybert']['batch_size']
    prefix = DocumentPrefixCache(chunk_folder / 'frozen_document_prefix', adapter, digest(chunk_path), len(chunks))
    parity_checked = False
    adapter.model.eval()
    tick = time.perf_counter()
    with torch.inference_mode():
        for index in range(state['completed_documents'], len(main)):
            doc = docs.iloc[index]
            start, end = int(doc.chunk_start), int(doc.chunk_start + doc.chunk_count)
            records = chunks.iloc[start:end].to_dict('records')
            for at in range(0, len(records), bs):
                batch = records[at:at + bs]
                features = adapter.record_features(batch)
                indices = list(range(start + at, start + at + len(batch)))
                hidden = prefix.get(adapter, features, indices)
                vectors = adapter.from_prefix(hidden, features['attention_mask'])
                if not parity_checked:
                    direct = adapter.forward(features)
                    error = float(torch.max(torch.abs(vectors - direct)))
                    assert torch.allclose(vectors, direct, atol=2e-5), 'Document prefix-cache forward parity failed'
                    write_json(folder / 'document_prefix_parity.json', dict(status='PASS', max_abs_error=error,
                        checked_doc_id=doc.doc_id, chunk_count=len(batch), float32=True))
                    parity_checked = True
                chunk_embeddings[start + at:start + at + len(batch)] = vectors.numpy()
            vector = np.asarray(chunk_embeddings[start:end]).mean(axis=0)
            vector /= np.linalg.norm(vector)
            embeddings[index] = vector
            if (index + 1) % 32 == 0 or index + 1 == len(main):
                prefix.flush()
                embeddings.flush()
                chunk_embeddings.flush()
                state.update(completed_documents=index + 1, seconds=state['seconds'] + time.perf_counter() - tick,
                             total_documents=len(main))
                write_json(marker, state)
                tick = time.perf_counter()
                log(f'BatteryBERT {variant} EMBED {index + 1}/13702; committed seconds={state["seconds"]:.1f}')
    ids = main.doc_id.tolist()
    write_json(folder / 'ordered_doc_ids.json', ids)
    validation = validate_vectors(folder / 'document_embeddings.npy', ids, read_json(out / 'ordered_doc_ids.json'), adapter.dimension)
    for start in range(0, len(chunks), 256):
        part = chunk_embeddings[start:start + 256]
        assert np.isfinite(part).all() and np.allclose(np.linalg.norm(part, axis=1), 1, atol=2e-5)
    for doc in docs.itertuples():
        vector = np.asarray(chunk_embeddings[doc.chunk_start:doc.chunk_start + doc.chunk_count]).mean(axis=0)
        vector /= np.linalg.norm(vector)
        assert np.allclose(embeddings[doc.doc_index], vector, atol=1e-6)
    provenance = read_json(out / 'batterybert_pretrained/model/download_provenance.json')
    write_json(folder / 'embedding_manifest.json', dict(model_id=cfg()['batterybert']['model_id'],
        revision=provenance['revision'], variant=variant, weights=weight_identity(adapter.folder),
        dimension=adapter.dimension, config=cfg()['batterybert'], ordered_doc_ids_sha256=signature(ids),
        chunks_sha256=digest(chunk_path), chunk_metadata_path=str(chunk_path),
        pooling='Attention-mask mean + L2 per chunk; mean normalized chunks + L2 per document',
        full_abstract_coverage=True, query_instruction=False,
        frozen_document_prefix_cache=dict(path=str(prefix.folder), fingerprint=prefix.identity,
            cached_chunks_at_branch_completion=int(prefix.valid.sum()), total_chunks=len(chunks),
            precision='float32; per-chunk SHA-256; last block always recomputed'),
        chunk_embeddings_sha256=digest(folder / 'chunk_embeddings.npy'), seconds=state['seconds']))
    write_json(folder / 'embedding_success.json', dict(status='PASS', **validation))
    log(f'BatteryBERT {variant} embeddings complete: {validation}')


def false_negative_mask(frame, threshold=.8):
    words = [set(re.findall(r'\w+', title.casefold())) for title in frame.title]
    mask = np.zeros((len(frame), len(frame)), dtype=bool)
    for i in range(len(frame)):
        for j in range(i):
            same = frame.iloc[i].split_group_id == frame.iloc[j].split_group_id
            lexical = len(words[i] & words[j]) / max(1, len(words[i] | words[j])) >= threshold
            mask[i, j] = mask[j, i] = same or lexical
    return mask


def contrastive_loss(queries, documents, mask, temperature):
    scores = queries @ documents.T / temperature
    scores = scores.masked_fill(torch.as_tensor(mask, device=scores.device), -1e4)
    targets = torch.arange(len(scores), device=scores.device)
    return (F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets)) / 2


def batches_for(size, batch_size, seed=42):
    if size < 2 or batch_size < 2:
        raise ValueError('In-batch negatives require at least two pairs')
    order = np.random.default_rng(seed).permutation(size)
    batches = [order[i:i + batch_size] for i in range(0, size, batch_size)]
    if len(batches[-1]) == 1:
        singleton = batches.pop()
        batches[-1] = np.concatenate([batches[-1], singleton])
    return batches


def train(out):
    settings = cfg()['batterybert']
    assert settings['epochs'] == 1 and settings['trainable_encoder_blocks'] == 1
    folder = out / 'batterybert_finetuned'
    folder.mkdir(exist_ok=True)
    adapter = BatteryEncoder(out / 'batterybert_pretrained/model')
    model = adapter.model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.layer[-1].parameters():
        param.requires_grad = True
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    frozen_hash = tensor_hash(frozen)
    before = {n: p.detach().clone() for n, p in named}
    frame = pairs_for('train')
    assert len(frame) == settings['expected_splits']['train']
    batches = batches_for(len(frame), settings['batch_size'])
    accum = settings['gradient_accumulation']
    optimizer = torch.optim.AdamW([p for _, p in named], lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    identity = signature(dict(config=cfg(), pretrained=weight_identity(adapter.folder), frozen_hash=frozen_hash,
        split=digest(ROOT / 'data/stage3/contrastive_pairs.parquet'), code=digest(Path(__file__))))
    resume_path = folder / 'resume.pt'
    state = dict(next_batch=0, optimizer_updates=0, rows_seen=0, best_score=-1., best_step=None,
                 seconds=0., masked_pairs=0)
    history, validations = [], []
    if resume_path.exists():
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        assert saved['fingerprint'] == identity, 'Training resume identity mismatch'
        for n, p in named:
            p.data.copy_(saved['weights'][n])
        optimizer.load_state_dict(saved['optimizer'])
        state, history, validations = saved['state'], saved['history'], saved['validations']
        torch.set_rng_state(saved['torch_rng'])
        log(f'RESUME BatteryBERT batch {state["next_batch"]}/{len(batches)}')
    write_json(folder / 'training_plan.json', dict(config=settings, fingerprint=identity,
        training_pairs=len(frame), batches=len(batches), frozen_hash=frozen_hash,
        trainable_parameters=sum(p.numel() for _, p in named),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_names=[n for n, _ in named], selection_uses_test=False,
        max_tokens=settings['training_max_tokens'],
        pooling='Attention-mask mean + L2, same token pooling as document inference',
        negatives='Within each microbatch only; gradient accumulation does not enlarge the negative pool'))
    def save_resume():
        temp = resume_path.with_suffix('.tmp')
        torch.save(dict(fingerprint=identity, weights={n: p.detach().cpu() for n, p in named},
            optimizer=optimizer.state_dict(), state=state, torch_rng=torch.get_rng_state(),
            history=history, validations=validations), temp)
        replace_with_retry(temp, resume_path)
        table(folder / 'training_history.csv', pd.DataFrame(history))
        write_json(folder / 'validation_history.json', validations)
        write_json(folder / 'training_progress.json', dict(**state, total_batches=len(batches)))
    optimizer.zero_grad(set_to_none=True)
    for index in range(state['next_batch'], len(batches)):
        started = time.perf_counter()
        model.eval()
        model.encoder.layer[-1].train()
        batch = frame.iloc[batches[index]]
        q = adapter.forward(adapter.features(batch.title.tolist(), settings['training_max_tokens']))
        d = adapter.forward(adapter.features(batch.abstract.tolist(), settings['training_max_tokens']))
        mask = false_negative_mask(batch, settings['title_jaccard_false_negative_threshold'])
        loss = contrastive_loss(q, d, mask, settings['temperature'])
        assert torch.isfinite(loss), 'Nonfinite training loss'
        group_size = min(accum, len(batches) - (index // accum) * accum)
        (loss / group_size).backward()
        boundary = (index + 1) % accum == 0 or index + 1 == len(batches)
        if boundary:
            torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state['optimizer_updates'] += 1
        seconds = time.perf_counter() - started
        state.update(next_batch=index + 1, rows_seen=state['rows_seen'] + len(batch),
                     seconds=state['seconds'] + seconds, masked_pairs=state['masked_pairs'] + int(mask.sum()))
        history.append(dict(step=index + 1, loss=float(loss.detach()), batch_pairs=len(batch),
            masked_off_diagonal=int(mask.sum()), learning_rate=settings['learning_rate'], seconds=seconds))
        if (index + 1) % 25 == 0 or index == 0:
            log(f'BatteryBERT TRAIN {index + 1}/{len(batches)} loss={float(loss.detach()):.6f}; seconds={state["seconds"]:.1f}')
        if boundary and ((index + 1) % settings['validation_every_steps'] == 0 or index + 1 == len(batches)):
            model.eval()
            validation = pairs_for('validation')
            cache = out / 'batterybert_pretrained/validation_frozen'
            qval = frozen_validation(adapter, validation.title.tolist(), cache, 'titles')
            dval = frozen_validation(adapter, validation.abstract.tolist(), cache, 'abstracts')
            metrics, _ = group_retrieval_metrics(qval, dval, validation.split_group_id.to_numpy())
            validations.append(dict(step=index + 1, **metrics))
            if metrics['group_mrr_at_10'] > state['best_score']:
                state.update(best_score=metrics['group_mrr_at_10'], best_step=index + 1)
                temp = folder / 'best_trainable.tmp'
                torch.save({n: p.detach().cpu() for n, p in named}, temp)
                replace_with_retry(temp, folder / 'best_trainable.pt')
            save_resume()
            log(f'BatteryBERT VALIDATION step={index + 1} Group MRR@10={metrics["group_mrr_at_10"]:.6f}; best={state["best_step"]}')
        elif boundary and (index + 1) % settings['checkpoint_every_steps'] == 0:
            save_resume()
    assert state['rows_seen'] == len(frame) and state['best_step'] is not None
    selected = torch.load(folder / 'best_trainable.pt', map_location='cpu', weights_only=True)
    for n, p in named:
        p.data.copy_(selected[n])
    assert tensor_hash(frozen) == frozen_hash, 'Frozen parameters changed'
    delta = sum(float(((p.detach() - before[n]) ** 2).sum()) for n, p in named) ** .5
    assert delta > 0 and np.isfinite(delta), 'No verified real fine-tuning update'
    model.eval()
    model.save_pretrained(folder / 'model', safe_serialization=True)
    adapter.tokenizer.save_pretrained(folder / 'model')
    # The saved model reloads and produces the same actual embedding.
    with torch.inference_mode():
        reference = adapter.forward(adapter.features(frame.title.iloc[:2].tolist())).numpy()
    del adapter, model, optimizer, named, frozen, before
    gc.collect()
    reloaded = BatteryEncoder(folder / 'model')
    with torch.inference_mode():
        actual = reloaded.forward(reloaded.features(frame.title.iloc[:2].tolist())).numpy()
    assert np.allclose(reference, actual, atol=1e-6)
    write_json(folder / 'training_success.json', dict(status='PASS', **state, fingerprint=identity,
        selected_weight_delta_l2=delta, frozen_hash_before=frozen_hash, frozen_hash_after=frozen_hash,
        real_gradient_updates=True, selection_uses_test=False, weights=weight_identity(folder / 'model'),
        reload_parity_max_error=float(np.max(np.abs(reference - actual))), config=settings))
    log('BatteryBERT fine-tuning COMPLETE and model reload verified')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['battery_preflight', 'proxy_pretrained', 'embed_pretrained',
                                        'train', 'proxy_finetuned', 'embed_finetuned'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must stay within outputs/')
    runtime()
    if args.phase == 'battery_preflight':
        preflight(out)
    elif args.phase.startswith('proxy_'):
        variant = args.phase.removeprefix('proxy_')
        if variant == 'finetuned':
            assert read_json(out / 'batterybert_finetuned/training_success.json')['real_gradient_updates']
        evaluate_proxy(BatteryEncoder(out / ('batterybert_' + variant) / 'model'), out, variant)
    elif args.phase.startswith('embed_'):
        embed(out, args.phase.removeprefix('embed_'))
    else:
        train(out)


if __name__ == '__main__':
    main()
