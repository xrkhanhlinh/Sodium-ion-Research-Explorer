"""Real corpus identity/filter/chunk validation after the full index is committed."""
import json
import numpy as np
import pandas as pd
import pytest
from src.search_radar_retrieval import RadarRetriever,OUT


@pytest.fixture(scope='module')
def retriever():
    if not (OUT/'chunks/manifest.json').exists():pytest.skip('Chunk build must finish first')
    engine=RadarRetriever();engine.load_chunks()
    yield engine
    engine.close()


def test_real_chunk_manifest_and_document_identity(retriever):
    info=json.loads((OUT/'chunks/manifest.json').read_text(encoding='utf8'))
    assert info['document_aggregation_parity_max_error']<2e-5
    assert info['verified_documents']==info['documents']==13702
    assert info['chunks']==32669
    assert not info['new_training'] and not info['full_encoder_reencoding']
    assert info['chunks_sha256']==info['prefix_manifest']['chunks_sha256']
    assert info['cache_identity']['frozen_hash']==info['prefix_manifest']['frozen_hash']
    assert len({r['global_chunk_id'] for r in retriever.chunk_records})==32669


@pytest.mark.parametrize('depth',[50,100,200])
def test_real_all_channels_obey_same_source_year_required_topic_filter(retriever,depth):
    q='Hard carbon anodes derived from biomass for sodium-ion batteries'
    with np.load(OUT/'historical/query_vectors/Q01.npz') as data:vectors={k:np.array(data[k]) for k in ('bge','fulldoc')}
    # Use an existing topic represented by recent Publication hard-carbon docs.
    source_mask=retriever.mask('Publication',(2020,2026),['carbon'])
    doc=next(d for d,use in zip(retriever.ids,source_mask) if use)
    topic=retriever.engine._research_topics()[doc]['topic_id']
    branches,_,mask,_=retriever.branches(q,depth,'Publication',(2020,2026),['carbon'],topic,vectors)
    allowed={d for d,use in zip(retriever.ids,mask) if use}
    for branch in branches.values():
        assert len(branch)<=depth and {d for d,s in branch}<=allowed
    chunks=retriever.chunk_branch(vectors['fulldoc'],mask,depth)
    assert len(chunks)<=depth and {d for d,s,r in chunks}<=allowed
    assert len({d for d,s,r in chunks})==len(chunks)
    for d,s,record in chunks:
        row=retriever.meta.loc[d]
        if record['offset_field']=='abstract':
            start,end=int(record['char_start']),int(record['char_end'])
            assert 0<=start<end<=len(row.abstract)


def test_real_unjudged_pairs_remain_missing_and_a0_parity():
    per=pd.read_csv(OUT/'historical/metrics_per_query.csv')
    pending=per[per.status.eq('PENDING_LABELS')]
    assert len(pending)>0
    assert pending[['Precision_at_10','nDCG_at_10','MRR_at_10']].isna().all().all()
    assert pd.read_csv(OUT/'historical/baseline_parity.csv').exact_order.all()
