"""Download one pinned public reranker; never upload project data."""
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import HfApi, snapshot_download

if __name__ == '__main__':
    repo = 'cross-encoder/ms-marco-MiniLM-L6-v2'
    revision = HfApi().model_info(repo).sha
    print('Public reranker revision:', revision, flush=True)
    root = Path(__file__).resolve().parents[1]
    folder = snapshot_download(
        repo, revision=revision, local_dir=root / 'models/reranker_minilm_l6',
        allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json',
                        'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'README.md'],
        max_workers=2,
    )
    Path(folder, 'download_provenance.json').write_text(json.dumps({
        'repo_id': repo, 'revision': revision,
        'source': 'https://huggingface.co/' + repo,
        'use': 'Inference only; no project data uploaded; trust_remote_code=False',
    }, indent=2), encoding='utf-8')
    print('Downloaded', folder, flush=True)
