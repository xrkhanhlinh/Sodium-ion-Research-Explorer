"""Download a pinned public BatteryBERT checkpoint into the new experiment only."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/model_comparison_v2'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root / 'outputs'):
        raise ValueError('Experiment must be inside outputs/')
    folder = output / 'batterybert_pretrained' / 'model'
    marker = folder / 'download_provenance.json'
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                h.update(block)
        return h.hexdigest()
    if marker.exists():
        info = json.loads(marker.read_text(encoding='utf-8'))
        if not all(digest(folder / k) == v for k, v in info['hashes'].items()):
            raise ValueError('Downloaded checkpoint failed integrity check')
        print('REUSE pinned BatteryBERT', info['revision'], flush=True)
        return
    repo = 'batterydata/batterybert-cased'
    info = HfApi().model_info(repo)
    files = {s.rfilename for s in info.siblings}
    weight = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
    patterns = [weight, 'config.json', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', 'vocab.txt', 'README.md']
    print('Downloading public checkpoint', repo, info.sha, weight, flush=True)
    snapshot_download(repo, revision=info.sha, local_dir=folder,
                      allow_patterns=patterns, max_workers=1)
    hashes = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
    marker.write_text(json.dumps(dict(repo_id=repo, revision=info.sha, hashes=hashes,
                      source='https://huggingface.co/' + repo,
                      trust_remote_code=False), indent=2), encoding='utf-8')
    print('DOWNLOAD VERIFIED', info.sha, flush=True)


if __name__ == '__main__':
    main()
