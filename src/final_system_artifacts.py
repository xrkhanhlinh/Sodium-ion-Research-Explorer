"""Read-only access to frozen research artifacts and versioned final outputs."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from .model_comparison_io import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/final_system'
OLD = ROOT / 'outputs/model_comparison_v2'
CORPUS = ROOT / 'data/processed/main_corpus.parquet'
TOPICS = {'LDA': OLD / 'lda_k10', 'BERTopic–BGE': OLD / 'bertopic_bge_k10',
          'BERTopic–BatteryBERT (pretrained)': OLD / 'bertopic_batterybert_k10/pretrained'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def corpus():
    frame = pd.read_parquet(CORPUS)
    if len(frame) != 13702 or not frame.doc_id.is_unique:
        raise ValueError('Frozen corpus must contain 13,702 unique document IDs')
    return frame


def vector_spec(kind):
    specs = {
        'batterybert_fulldoc': (ROOT / 'outputs/batterybert_full_document_20261024_v1/document_embeddings.npy',
            ROOT / 'outputs/batterybert_full_document_20261024_v1/ordered_doc_ids.json',
            ROOT / 'outputs/batterybert_full_document_20261024_v1/model',
            ROOT / 'outputs/batterybert_full_document_20261024_v1/embedding_success.json'),
        'batterybert_finetuned': (OUT / 'batterybert_finetuned/document_embeddings.npy',
            OUT / 'batterybert_finetuned/ordered_doc_ids.json', OLD / 'batterybert_finetuned/model',
            OUT / 'batterybert_finetuned/embedding_success.json'),
        'batterybert_pretrained': (OLD / 'batterybert_pretrained/document_embeddings.npy',
            OLD / 'batterybert_pretrained/ordered_doc_ids.json', OLD / 'batterybert_pretrained/model',
            OLD / 'batterybert_pretrained/embedding_success.json'),
        'bge_pretrained': (ROOT / 'data/stage4/pretrained_embeddings.npy',
            ROOT / 'data/stage4/pretrained_embedding_ids.parquet', ROOT / 'models/bge_pretrained',
            ROOT / 'outputs/reports/stage4/pretrained_embedding_manifest.json'),
        'bge_finetuned': (ROOT / 'data/stage3/embeddings.npy',
            ROOT / 'data/stage3/embeddings.parquet', ROOT / 'models/bert_finetuned',
            ROOT / 'outputs/reports/stage3/embedding_manifest.json'),
        'bge_finetuned_chunked': (ROOT / 'outputs/bertopic_chunked/document_embeddings.npy',
            ROOT / 'outputs/bertopic_chunked/ordered_doc_ids.json', ROOT / 'models/bert_finetuned',
            ROOT / 'outputs/bertopic_chunked/embedding_success.json'),
    }
    matrix, order, model, marker = specs[kind]
    ids = read_json(order) if order.suffix == '.json' else pd.read_parquet(order, columns=['doc_id']).doc_id.tolist()
    info = read_json(marker)
    expected = info.get('sha256', info.get('numpy_sha256', info.get('document_embeddings_sha256')))
    if not expected or sha(matrix) != expected:
        raise ValueError(f'{kind}: missing or mismatched embedding checksum')
    if ids != corpus().doc_id.tolist():
        raise ValueError(f'{kind}: ordered document IDs mismatch')
    vectors = np.load(matrix, mmap_mode='r')
    if vectors.shape != (13702, 768) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError(f'{kind}: invalid float32 vectors')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5):
        raise ValueError(f'{kind}: vectors are not normalized')
    weights = next(iter(model.glob('*.safetensors')), model / 'pytorch_model.bin')
    actual_weight_sha = sha(weights)
    model_expected = info.get('model_sha256', info.get('identity',{}).get('model_sha256'))
    if kind == 'batterybert_fulldoc':
        model_expected = info['weights_sha256']
    elif kind == 'batterybert_finetuned':
        model_expected = info['identity']['weights'][weights.name]
    elif kind == 'batterybert_pretrained':
        model_expected = read_json(model.parent/'embedding_manifest.json')['weights'][weights.name]
    if not model_expected or model_expected != actual_weight_sha:
        raise ValueError(f'{kind}: checkpoint differs from embedding provenance')
    return dict(kind=kind, matrix=matrix, order=order, model=model, ids=ids,
                embeddings_sha256=expected, weights_sha256=actual_weight_sha, dimension=768)


def snapshot():
    """Freeze all pre-existing data/model/output bytes, excluding the new branch."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'protected_artifacts.json'
    if target.exists():
        return read_json(target)
    paths = [p for root in ('data', 'models', 'outputs') for p in (ROOT / root).rglob('*')
             if p.is_file() and not p.is_relative_to(OUT)]
    protected = {}
    for i, p in enumerate(paths):
        protected[p.relative_to(ROOT).as_posix()] = dict(sha256=sha(p), bytes=p.stat().st_size)
        if i % 500 == 0:
            print(f'PROTECT {i}/{len(paths)}', flush=True)
    write_json(target, protected)
    return protected


def verify_protected():
    old = read_json(OUT / 'protected_artifacts.json')
    changed = [name for name, info in old.items() if not (ROOT / name).exists() or sha(ROOT / name) != info['sha256']]
    write_json(OUT / 'preservation_check.json', dict(status='PASS' if not changed else 'FAIL',
                                                   protected_files=len(old), changed=changed))
    if changed:
        raise ValueError('Protected artifact changes: ' + repr(changed[:10]))


def topic_data(name):
    folder = TOPICS[name]
    result = {key: pd.read_csv(folder / file) for key, file in dict(summary='topic_summary.csv',
        keywords='topic_keywords.csv', representatives='representative_documents.csv',
        years='topic_years.csv', sources='topic_sources.csv').items()}
    assignment = pd.read_parquet(folder / 'topic_assignments.parquet')
    if assignment.doc_id.tolist() != corpus().doc_id.tolist():
        raise ValueError('Topic document order mismatch')
    result['assignments'] = assignment
    result['metrics'] = read_json(folder / 'metrics.json')
    # Tentative descriptions grounded in saved keywords and representatives.
    labels = {int(row.topic_id): ' / '.join(str(row.top10_keywords).split('; ')[:3])
              for row in result['summary'].itertuples() if row.topic_id != -1}
    # Reuse the evidence-based interpretations already recorded with these
    # exact assignments in reports/INTERPRETATION_NOTES.md.
    if name=='LDA':
        labels.update(dict(enumerate([
            'Sodium-storage anodes and carbon composites',
            'Non-aqueous electrolytes and ionic transport',
            'Cathode electrochemistry and vanadium phosphates',
            'Electrode-material inventions (mixed materials)',
            'Transition-metal and Prussian-white formulations',
            'Layered oxide cathodes and P2/O3 structures',
            'Active electrode materials and characterization',
            'Material synthesis and preparation processes',
            'Battery devices, housings and assembly',
            'Hard-carbon preparation and related materials'])))
    elif name=='BERTopic–BGE':
        labels.update(dict(enumerate([
            'Electrode-material preparation patents',
            'Layered-oxide and polyanion cathodes',
            'Alloy/conversion and carbon-based anodes',
            'Sulfide/selenide and carbon-composite anodes',
            'Prussian-blue analogue cathodes',
            'Titanate and TiO2 anodes',
            'Electrolyte salts, solvents and additives',
            'Battery hardware and manufacturing devices',
            'Sodium cobalt oxides and superconductivity',
            'MXene-based sodium-storage electrodes'])))
    if 'BatteryBERT' in name:
        labels.update({0:'Electrode performance (broad Publication cluster)',
            1:'Material preparation methods (broad Patent cluster)', 2:'Reviews and research outlook',
            3:'2D anodes and first-principles calculations', 4:'Fixtures, protection and inspection equipment',
            5:'Assembly, heating and mechanical equipment', 6:'Battery packs, management and control',
            7:'Sodium cobaltate superconductivity', 8:'Cell housing and terminals',
            9:'Mixed high-performance electrode materials'})
    labels[-1] = 'Outlier'
    result['labels'] = labels
    return result


def export_topics():
    rows = []
    for name, folder in TOPICS.items():
        data = topic_data(name)
        info = data['metrics']
        manifest = read_json(folder / 'manifest.json')
        for path, expected in manifest['sha256'].items():
            if sha(folder / path) != expected:
                raise ValueError('Topic artifact checksum mismatch: ' + path)
        dest = OUT / 'topics' / ('lda' if name == 'LDA' else 'batterybert' if 'BatteryBERT' in name else 'bge')
        dest.mkdir(parents=True, exist_ok=True)
        for key in ('summary','keywords','representatives','years','sources'):
            frame = data[key].copy()
            frame['topic_label'] = frame.topic_id.map(data['labels'])
            frame.to_csv(dest / (key + '.csv'), index=False)
        data['assignments'].to_parquet(dest / 'assignments.parquet', index=False)
        write_json(dest / 'provenance.json', dict(source=str(folder.relative_to(ROOT)), fit_rerun=False,
            label_status='Descriptive interpretation; not expert-validated',
            keyword_weight='LDA topic-word probability' if name == 'LDA' else 'c-TF-IDF',
            encoder_variant='pretrained' if 'BatteryBERT' in name else 'fine-tuned' if 'BGE' in name else None))
        rows.append(dict(model=name, topics=info['topics'], C_v=info['c_v'],
            Topic_Diversity=info['topic_diversity'], Outlier_Rate=info['outlier_ratio'],
            largest_topic_share=float(data['summary'].query('topic_id != -1').document_share.max()),
            source=str(folder.relative_to(ROOT))))
    pd.DataFrame(rows).to_csv(OUT / 'topic_comparison.csv', index=False)
    inventory = []
    for kind in ('bge_pretrained','bge_finetuned','bge_finetuned_chunked','batterybert_pretrained','batterybert_finetuned'):
        try:
            spec = vector_spec(kind)
            inventory.append({k:str(v.relative_to(ROOT)) if isinstance(v, Path) else v
                              for k,v in spec.items() if k != 'ids'} | dict(status='VALID'))
        except FileNotFoundError as exc:
            inventory.append(dict(kind=kind, status='MISSING', reason=str(exc)))
    write_json(OUT / 'artifact_inventory.json', dict(vectors=inventory,
        batterybert_finetuned_topics='MISSING; no BERTopic refit authorized; existing pretrained topics reused',
        checkpoint_selection='BatteryBERT selected on held-out title–abstract proxy MRR@10; not Search relevance',
        original_search_v2_default='RERANK_PRETRAINED; unchanged'))
