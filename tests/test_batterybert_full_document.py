"""Numerical contracts for actual full-document gradient caching."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from src.batterybert_full_document import normal_pool,unique_token_weights,stratified_batches,embedding_gradients,pooling_gradients,symmetric_loss


def test_unique_weights_cover_the_complete_abstract_without_overlap_double_count():
    records=[dict(token_start=0,token_end=200),dict(token_start=136,token_end=336),dict(token_start=272,token_end=410)]
    weights=unique_token_weights(records)
    np.testing.assert_array_equal(weights,[200,136,74])
    assert weights.sum()==410
    with pytest.raises(ValueError):unique_token_weights([dict(token_start=2,token_end=20)])


def test_stratification_keeps_every_train_id_once_and_controls_batch_size():
    sources=np.array(['Patent']*43+['Publication']*58)
    batches=stratified_batches(np.arange(101),sources,16,42)
    assert sorted(np.concatenate(batches).tolist())==list(range(101))
    assert min(map(len,batches))>=2 and max(map(len,batches))<=16
    assert all(set(sources[b])=={'Patent','Publication'} for b in batches)
    again=stratified_batches(np.arange(101),sources,16,42)
    assert all(np.array_equal(a,b) for a,b in zip(batches,again))


@pytest.mark.parametrize('distill',[0.,.02])
def test_two_level_gradient_cache_matches_full_batch_gradient_and_updates_every_tail(distill):
    torch.manual_seed(14)
    layer=torch.nn.Linear(7,5,bias=False).double()
    original=layer.weight.detach().clone()
    queries=torch.randn(4,7,dtype=torch.float64)
    documents=[torch.randn(k,7,dtype=torch.float64) for k in [2,3,4,2]]
    weights=[torch.arange(1,len(x)+1,dtype=torch.float64) for x in documents]
    mask=np.zeros((4,4),dtype=bool)
    ref=(F.normalize(torch.randn(4,5,dtype=torch.float64),dim=1),F.normalize(torch.randn(4,5,dtype=torch.float64),dim=1))
    def vectors():
        q=F.normalize(layer(queries),dim=1)
        chunks=[F.normalize(layer(x),dim=1) for x in documents]
        d=torch.stack([normal_pool(c,w) for c,w in zip(chunks,weights)])
        return q,d,chunks
    q,d,chunks=vectors();loss=symmetric_loss(q,d,mask,.2)
    if distill:loss=loss+distill*((q-ref[0]).square().sum(1).mean()+(d-ref[1]).square().sum(1).mean())
    loss.backward();expected=layer.weight.grad.detach().clone();layer.zero_grad(set_to_none=True)
    with torch.no_grad():q,d,chunks=vectors()
    _,gq,gd=embedding_gradients(q,d,mask,.2,ref,distill)
    for i in range(4):F.normalize(layer(queries[i:i+1]),dim=1).backward(gq[i:i+1])
    tail_norms=[]
    for i in range(4):
        cached_gradient=pooling_gradients(chunks[i],weights[i],gd[i])
        tail_norms.append(float(cached_gradient[-1].norm()))
        for j in range(len(documents[i])):
            F.normalize(layer(documents[i][j:j+1]),dim=1).backward(cached_gradient[j:j+1])
    torch.testing.assert_close(layer.weight.grad,expected,atol=1e-10,rtol=1e-9)
    assert min(tail_norms)>0
    torch.optim.SGD(layer.parameters(),lr=.01).step()
    assert not torch.equal(layer.weight,original)


def test_tail_content_changes_document_embedding_and_cannot_be_truncated():
    first=F.normalize(torch.tensor([1.,.2,.3]),dim=0)
    tail=F.normalize(torch.tensor([.2,1.,.3]),dim=0)
    pooled=normal_pool(torch.stack([first,tail]),torch.tensor([128.,270.]))
    assert not torch.allclose(pooled,first)
    pooled2=normal_pool(torch.stack([first,-tail]),torch.tensor([128.,270.]))
    assert not torch.allclose(pooled,pooled2)


def test_pooling_replay_accepts_weights_cached_during_inference():
    with torch.inference_mode():weights=torch.tensor([128.,270.])
    chunks=F.normalize(torch.tensor([[1.,.2,.3],[.2,1.,.3]]),dim=1)
    gradients=pooling_gradients(chunks,weights,torch.tensor([1.,2.,3.]))
    assert torch.isfinite(gradients).all() and gradients[-1].norm()>0


def test_real_bert_attention_layer_cached_replay_matches_full_batch(tmp_path):
    """Exercise actual SDPA BERT, pooling masks and padding, not only a linear toy."""
    from transformers import BertConfig
    from src.batterybert_full_document import LastBlock
    torch.manual_seed(42)
    BertConfig(hidden_size=12,num_attention_heads=3,intermediate_size=24,num_hidden_layers=1,
        hidden_dropout_prob=.2,attention_probs_dropout_prob=.2).save_pretrained(tmp_path)
    block=LastBlock(tmp_path).double()
    query=torch.randn(3,5,12,dtype=torch.float64);body=torch.randn(7,6,12,dtype=torch.float64)
    qa=torch.ones(3,5,dtype=torch.int64);qa[2,-2:]=0
    da=torch.ones(7,6,dtype=torch.int64);da[::2,-1]=0
    pool=da.clone();pool[:,:2]=0
    spans=[(0,2),(2,5),(5,7)]
    weights=[torch.tensor(w,dtype=torch.float64) for w in [[190,90],[120,100,70],[180,100]]]
    mask=np.zeros((3,3),dtype=bool)
    def representations():
        q=block(query,qa,qa);chunks=block(body,da,pool)
        return q,torch.stack([normal_pool(chunks[a:b],w) for (a,b),w in zip(spans,weights)]),chunks
    q,d,_=representations();symmetric_loss(q,d,mask,.1).backward()
    expected={name:p.grad.clone() for name,p in block.named_parameters()}
    block.zero_grad(set_to_none=True)
    with torch.no_grad():q,d,chunks=representations()
    _,gq,gd=embedding_gradients(q,d,mask,.1)
    for i in range(3):block(query[i:i+1],qa[i:i+1],qa[i:i+1]).backward(gq[i:i+1])
    for i,((a,b),w) in enumerate(zip(spans,weights)):
        gradients=pooling_gradients(chunks[a:b],w,gd[i])
        assert gradients[-1].norm()>0
        for j in range(a,b):block(body[j:j+1],da[j:j+1],pool[j:j+1]).backward(gradients[j-a:j-a+1])
    for name,p in block.named_parameters():torch.testing.assert_close(p.grad,expected[name],atol=1e-10,rtol=1e-8)
