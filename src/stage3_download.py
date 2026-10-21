"""Download the pinned pretrained snapshot without importing neural runtimes."""
import os
from .stage3_common import ROOT,REPORT,config,read_json,write_json,digest,progress
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING','1')

def main():
    from huggingface_hub import HfApi,snapshot_download
    cfg=config();marker=REPORT/'pretrained_model.json'
    if marker.exists():
        info=read_json(marker)
        assert all(digest(ROOT/p)==h for p,h in info['files'].items())
        print('REUSE pinned pretrained snapshot',info['revision'],flush=True);return
    revision=HfApi().model_info(cfg['model_id']).sha
    folder=ROOT/'models/bge_pretrained'
    snapshot_download(cfg['model_id'],revision=revision,local_dir=str(folder),allow_patterns=['*.json','*.txt','model.safetensors','1_Pooling/*'],max_workers=2)
    files={p.relative_to(ROOT).as_posix():digest(p) for p in folder.rglob('*') if p.is_file() and '.cache' not in p.parts}
    assert (folder/'model.safetensors').exists(),'Missing real pretrained weights'
    write_json(marker,dict(model=cfg['model_id'],revision=revision,local_path=folder.relative_to(ROOT).as_posix(),files=files))
    progress(f'Downloaded pinned real BGE pretrained weights: revision={revision}.')

if __name__=='__main__':main()
