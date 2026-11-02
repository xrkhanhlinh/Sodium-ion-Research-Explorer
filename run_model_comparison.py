"""Run independent, memory-bounded stages; completed stages resume by SHA-256.

Usage: .venv/Scripts/python.exe run_model_comparison.py
Use --phases preflight lda bge for the non-neural branches, or --phases report
to refresh the honest partial-results report while a long CPU run is active.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parent
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '4'

from src.stage3_common import digest, signature, read_json
from src.model_comparison_io import write_json

PHASES = ['preflight', 'lda', 'bge', 'download', 'battery_preflight',
          'proxy_pretrained', 'embed_pretrained', 'topics_pretrained', 'train',
          'proxy_finetuned', 'embed_finetuned', 'topics_finetuned', 'verify', 'report']
NEURAL = {'battery_preflight', 'proxy_pretrained', 'embed_pretrained', 'train',
          'proxy_finetuned', 'embed_finetuned'}
DIRECTORIES = dict(lda='lda_k10', bge='bertopic_bge_k10',
    download='batterybert_pretrained/model', battery_preflight='batterybert_pretrained/preflight.json',
    proxy_pretrained='batterybert_pretrained/proxy_metrics.json',
    embed_pretrained='batterybert_pretrained/embedding_success.json',
    topics_pretrained='bertopic_batterybert_k10/pretrained',
    train='batterybert_finetuned/training_success.json',
    proxy_finetuned='batterybert_finetuned/proxy_metrics.json',
    embed_finetuned='batterybert_finetuned/embedding_success.json',
    topics_finetuned='bertopic_batterybert_k10/finetuned', preflight='preflight.json')
DEPENDENCIES = dict(lda=['preflight'], bge=['preflight'], battery_preflight=['download', 'preflight'],
    proxy_pretrained=['battery_preflight'], embed_pretrained=['battery_preflight'],
    topics_pretrained=['embed_pretrained'], train=['proxy_pretrained'],
    proxy_finetuned=['train'], embed_finetuned=['train'], topics_finetuned=['embed_finetuned'])


def phase_identity(phase, out):
    source = 'src/batterybert.py' if phase in NEURAL else 'scripts/download_batterybert.py' if phase == 'download' else 'src/model_comparison.py'
    return signature(dict(phase=phase, config=digest(ROOT / 'configs/model_comparison.yaml'),
        source=digest(ROOT / source), shared=digest(ROOT / 'src/stage3_common.py'),
        inputs=digest(out / 'input_manifest.json') if (out / 'input_manifest.json').exists() else None,
        dependencies={p: digest(out / '.phases' / (p + '.json')) for p in DEPENDENCIES.get(phase, [])
                      if (out / '.phases' / (p + '.json')).exists()}))


def verify_marker(marker, identity, out):
    info = read_json(marker)
    if info['fingerprint'] != identity:
        raise ValueError(f'Completed phase differs from code/config/inputs: {marker}; preserve it and use a new experiment directory')
    for name, expected in info['hashes'].items():
        if digest(out / name) != expected:
            raise ValueError('Completed artifact integrity mismatch: ' + name)


def run_command(phase, out, logs):
    if phase == 'download':
        command = [sys.executable, 'scripts/download_batterybert.py', '--output', str(out)]
    else:
        module = 'src.batterybert' if phase in NEURAL else 'src.model_comparison'
        command = [sys.executable, '-m', module, phase, '--output', str(out)]
    with (logs / (phase + '.log')).open('a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND ' + json.dumps(command) + '\n')
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        if phase == 'report' and result == 0:
            result = subprocess.run([sys.executable, 'scripts/finalize_model_comparison_report.py',
                '--output', str(out)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        return result


def canonicalize_tokenizer(out):
    """Keep byte-identical tokenization provenance for shared chunk caches.

    Transformers reserializes tokenizer.json on save even when its backend is
    identical. Verify semantic identity, then preserve the original bytes.
    """
    from transformers import AutoTokenizer
    original = out / 'batterybert_pretrained/model'
    trained = out / 'batterybert_finetuned/model'
    before = AutoTokenizer.from_pretrained(original, local_files_only=True)
    after = AutoTokenizer.from_pretrained(trained, local_files_only=True)
    if before.backend_tokenizer.to_str() != after.backend_tokenizer.to_str():
        raise ValueError('Fine-tuned tokenizer changed; cannot share frozen chunk metadata')
    old_sha = digest(trained / 'tokenizer.json')
    shutil.copy2(original / 'tokenizer.json', trained / 'tokenizer.json')
    write_json(trained / 'tokenizer_serialization.json', dict(semantic_backend_identical=True,
        saved_serialization_sha256=old_sha, canonical_sha256=digest(trained / 'tokenizer.json'),
        reason='Preserve byte-identical pretrained tokenizer provenance for shared token windows'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/model_comparison_v2')
    parser.add_argument('--phases', nargs='+', choices=PHASES, default=PHASES)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must stay inside outputs/')
    out.mkdir(parents=True, exist_ok=True)
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    # Pure reporting may safely run beside an active execution.
    if args.phases == ['report']:
        return run_command('report', out, logs)
    lock = out / '.run.lock'
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError:
        import psutil
        owner = read_json(lock)
        if psutil.pid_exists(owner['pid']) and abs(psutil.Process(owner['pid']).create_time() - owner['created']) < 1:
            raise RuntimeError('An experiment runner is already active')
        lock.unlink()
        handle = lock.open('x', encoding='utf-8')
    import psutil
    with handle:
        json.dump(dict(pid=os.getpid(), created=psutil.Process().create_time()), handle)
    state_path = out / 'run_status.json'
    state = read_json(state_path) if state_path.exists() else dict(phases={})
    state.pop('failed_phase', None)
    state.pop('error', None)
    state.update(status='RUNNING', pid=os.getpid(), requested_phases=args.phases)
    write_json(state_path, state)
    try:
        for phase in args.phases:
            marker = out / '.phases' / (phase + '.json')
            identity = phase_identity(phase, out)
            if phase not in ('verify', 'report') and marker.exists():
                verify_marker(marker, identity, out)
                print('REUSE VERIFIED', phase, flush=True)
                continue
            for dep in DEPENDENCIES.get(phase, []):
                dep_marker = out / '.phases' / (dep + '.json')
                if not dep_marker.exists():
                    raise RuntimeError(f'{phase} requires completed {dep}; run that phase first')
                verify_marker(dep_marker, phase_identity(dep, out), out)
            state.update(current_phase=phase, status='RUNNING')
            state['phases'][phase] = dict(status='RUNNING', started=time.time())
            write_json(state_path, state)
            print('START', phase, 'log:', logs / (phase + '.log'), flush=True)
            result = run_command(phase, out, logs)
            if result:
                state['phases'][phase].update(status='FAILED', exit_code=result, log=str(logs / (phase + '.log')))
                state.update(status='INCOMPLETE', failed_phase=phase)
                write_json(state_path, state)
                run_command('report', out, logs)
                return result
            if phase == 'train':
                canonicalize_tokenizer(out)
            target = out / DIRECTORIES.get(phase, 'verification.json')
            if phase not in ('verify', 'report'):
                paths = [target] if target.is_file() else [p for p in target.rglob('*') if p.is_file() and '.cache' not in p.parts]
                if not paths:
                    raise RuntimeError('No artifacts produced by ' + phase)
                # Include data/model files behind embedding/training summary markers.
                extras = []
                if phase.startswith('embed_'):
                    extras = [target.parent / n for n in ('document_embeddings.npy', 'ordered_doc_ids.json', 'embedding_manifest.json')]
                if phase == 'train':
                    extras = [p for p in (target.parent / 'model').rglob('*') if p.is_file()]
                    extras += [target.parent / 'training_history.csv', target.parent / 'validation_history.json']
                if phase.startswith('proxy_'):
                    extras = [p for p in (target.parent / 'proxy').glob('*') if p.is_file()]
                # Preflight identity includes its just-created input manifest.
                write_json(marker, dict(fingerprint=phase_identity(phase, out),
                    hashes={p.relative_to(out).as_posix(): digest(p) for p in paths + extras}))
            state['phases'][phase].update(status='COMPLETED', finished=time.time())
            write_json(state_path, state)
            print('DONE', phase, flush=True)
            if phase != 'report':
                run_command('report', out, logs)
        all_done = all((out / '.phases' / (p + '.json')).exists() for p in PHASES if p not in ('verify', 'report'))
        state.pop('failed_phase', None)
        state.pop('error', None)
        state.update(status='COMPLETED' if all_done and (out / 'verification.json').exists() else 'PARTIAL', current_phase=None)
        write_json(state_path, state)
        run_command('report', out, logs)
        return 0
    except BaseException as exc:
        state.update(status='INCOMPLETE', error=repr(exc))
        write_json(state_path, state)
        run_command('report', out, logs)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.exit(main())
