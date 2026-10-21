"""Technical correctness only; no configuration selection or baseline fitting."""
import numpy as np
import pytest
from src.bertopic_chunked import windows, coverage, chunk_document


@pytest.mark.parametrize('length,capacity',[(1,65),(64,65),(65,65),(66,65),(257,253),(1300,200),(20000,65)])
def test_windows_cover_tail_and_exact_overlap(length,capacity):
    spans = windows(length,capacity,[5,40,90,150,280,1000])
    assert coverage(length,spans)==length
    assert spans[-1][1]==length
    assert all(end-start<=capacity for start,end in spans)
    assert all(a-b==64 for (_,a),(b,_) in zip(spans,spans[1:]))


def test_sentence_preference_and_progress():
    assert windows(400,200,[120,190,250])[0]==(0,190)
    assert windows(400,200,[30])[0]==(0,200)
    with pytest.raises(ValueError): windows(300,64)
    with pytest.raises(AssertionError): coverage(10,[(0,3),(4,10)])


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    from src.bertopic_chunked import MODEL
    return AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True)


def test_original_wordpieces_preserved_with_title_and_tail(tokenizer):
    title='Na3V2(PO4)3 sodium-ion batteries'
    abstract='Hard carbon stores Na+ ions. '*200 + 'Unique final tail.'
    chunks,stats=chunk_document(tokenizer,title,abstract)
    ids=tokenizer(abstract,add_special_tokens=False)['input_ids']
    title_ids=tokenizer(title,add_special_tokens=False)['input_ids']
    assert len(chunks)>2 and stats['covered_abstract_tokens']==len(ids)
    for row in chunks:
        expected=[tokenizer.cls_token_id]+title_ids+[tokenizer.sep_token_id]+ids[row['token_start']:row['token_end']]+[tokenizer.sep_token_id]
        assert row['input_ids']==expected and len(expected)<=256
    assert chunks[-1]['token_end']==len(ids)


def test_extreme_title_reserves_payload_and_records_truncation(tokenizer):
    chunks,stats=chunk_document(tokenizer,'battery '*400,'sodium '*400)
    assert stats['title_truncated'] and stats['covered_abstract_tokens']==400
    assert all(len(r['input_ids'])<=256 for r in chunks)
    assert len(chunks)>1


@pytest.mark.parametrize('abstract',['',None,np.nan])
def test_title_only_has_real_input_and_correct_field(tokenizer,abstract):
    chunks,stats=chunk_document(tokenizer,'Hard carbon sodium battery',abstract)
    assert stats['title_only'] and stats['abstract_tokens']==0
    assert chunks[0]['offset_field']=='title'
    assert chunks[0]['input_ids']==tokenizer('Hard carbon sodium battery')['input_ids']


def test_empty_document_rejected(tokenizer):
    with pytest.raises(AssertionError): chunk_document(tokenizer,'','')
