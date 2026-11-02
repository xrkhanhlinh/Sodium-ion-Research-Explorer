"""Read-only progress monitor; does not import Torch or load model artifacts."""
import argparse
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def snapshot(out):
    state = read(out / 'run_status.json')
    phase = state.get('current_phase')
    progress = {}
    if phase and phase.startswith('embed_'):
        info = read(out / ('batterybert_' + phase.removeprefix('embed_')) / 'embedding_progress.json')
        progress = {k: info[k] for k in ('completed_documents', 'total_documents', 'seconds') if k in info}
    elif phase == 'train':
        info = read(out / 'batterybert_finetuned/training_progress.json')
        progress = {k: info[k] for k in ('next_batch', 'total_batches', 'rows_seen', 'best_score') if k in info}
    result = dict(status=state.get('status'), phase=phase, **progress)
    if state.get('failed_phase') and state.get('status') == 'INCOMPLETE':
        result['failed_phase'] = state['failed_phase']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/model_comparison_v2'))
    parser.add_argument('--interval', type=float, default=50.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        info = snapshot(args.output)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        if args.once or info['status'] != 'RUNNING':
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
