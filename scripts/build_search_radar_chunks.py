"""Resume exact last-block inference from verified frozen prefixes, without training."""
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.batterybert_full_document import FeatureStore,LastBlock,normal_pool,sha,read,signature,LEGACY,OUT as FULL
from src.model_comparison_io import write_json
from src.search_radar_core import VERSION


def main():
    torch.set_num_threads(4)
    target=ROOT/'outputs'/VERSION/'chunks';target.mkdir(parents=True,exist_ok=True)
    source=FULL/'experiments'/Path(read(FULL/'selection.json')['selected_checkpoint'].replace('\\','/')).parent.name/'best_last_block.pt'
    store=FeatureStore();block=LastBlock(FULL/'tokenizer')
    block.load_state_dict(torch.load(source,map_location='cpu',weights_only=True));block.eval()
    # Match the accepted exported model exactly, rather than trusting path names.
    from safetensors import safe_open
    with safe_open(FULL/'model/model.safetensors',framework='pt',device='cpu') as exported:
        for name,value in block.layer.state_dict().items():
            if not torch.equal(value,exported.get_tensor('encoder.layer.11.'+name)):
                raise ValueError('Last block differs from accepted exported encoder')
    identity=dict(weights_sha256=sha(FULL/'model/model.safetensors'),last_block_sha256=sha(source),
        chunks_sha256=sha(LEGACY/'chunks.parquet'),cache_identity=read(FULL/'features/identity.json'),
        prefix_manifest=read(LEGACY/'frozen_document_prefix/manifest.json'),
        ordered_doc_ids_sha256=sha(FULL/'ordered_doc_ids.json'),pooling='abstract_only_l2_per_chunk',version=VERSION)
    if identity['chunks_sha256']!=identity['prefix_manifest']['chunks_sha256'] or identity['cache_identity']['frozen_hash']!=identity['prefix_manifest']['frozen_hash']:
        raise ValueError('Frozen prefix and chunk manifest provenance disagree')
    fingerprint=signature(identity);progress=target/'progress.json';count=len(store.chunks)
    state=read(progress) if progress.exists() else dict(fingerprint=fingerprint,completed=0,seconds=0.)
    if state['fingerprint']!=fingerprint:raise ValueError('Chunk resume identity mismatch')
    if (target/'manifest.json').exists():
        marker=read(target/'manifest.json')
        if marker['fingerprint']!=fingerprint:raise ValueError('Chunk manifest identity mismatch')
        for n,digest in marker['files'].items():
            if sha(target/n)!=digest:raise ValueError('Chunk output changed')
        print('Verified completed chunk index',flush=True);return
    matrix=np.lib.format.open_memmap(target/'vectors.npy',mode='r+' if progress.exists() else 'w+',dtype='float32',shape=(count,768))
    start=time.perf_counter();offset=state['completed'];prior=state['seconds']
    with torch.inference_mode():
        for at in range(offset,count,8):
            end=min(at+8,count);matrix[at:end]=block(*store.packed('chunk',range(at,end))).numpy()
            if end%128==0 or end==count:
                matrix.flush();state.update(completed=end,seconds=prior+time.perf_counter()-start);write_json(progress,state)
                if end%1024==0 or end==count:print(f'CHUNKS {end}/{count} {state["seconds"]:.1f}s',flush=True)
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix,axis=1),1,atol=2e-5):raise ValueError('Invalid chunk vectors')
    records=store.chunks.drop(columns=['input_ids','token_type_ids']).copy()
    records['global_chunk_id']=[f'{d}::chunk:{i}' for d,i in zip(records.doc_id,records.chunk_id)]
    records['checkpoint_sha256']=identity['weights_sha256']
    records['context']=records.doc_id.map(store.main.set_index('doc_id').title)
    records.to_parquet(target/'records.parquet',index=False)
    documents=np.load(FULL/'document_embeddings.npy',mmap_mode='r');errors=[]
    for i in range(len(store.main)):
        ids=store.chunk_ids(i);actual=normal_pool(torch.from_numpy(np.array(matrix[ids])),store.weights(i)).numpy()
        errors.append(float(np.max(np.abs(actual-documents[i]))))
    if max(errors)>2e-5:raise ValueError(f'Chunk/document aggregation parity failed: {max(errors)}')
    import psutil
    info=dict(**identity,fingerprint=fingerprint,chunks=count,documents=len(documents),dimension=768,
        seconds=state['seconds'],rss_bytes=psutil.Process().memory_info().rss,
        document_aggregation_parity_max_error=max(errors),verified_documents=len(errors),
        files={n:sha(target/n) for n in ('vectors.npy','records.parquet')},
        new_training=False,full_encoder_reencoding=False,prefix_hash_checked_per_row=True)
    write_json(target/'manifest.json',info)
    print(json.dumps(info,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
