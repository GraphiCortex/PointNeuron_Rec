"""Fixed feature capture, loss semantics, synthetic fitting and strict reload."""
import copy
import pytest
import torch
from torch.nn import functional as F

from pointneuron.models.frozen_query_confidence import FrozenQueryConfidence, fixed_confidence_loss, confidence_checkpoint, load_confidence_checkpoint
from pointneuron.models.pointnext_query_cropnorm import PointNextCropNormQueryModel
from scripts.pointneuron2.query_confidence_experiment import cache_tensors, tensor_digests


def recipe():
    return {"positive_class_weight":8.,"confidence_weight":2.}


def test_loss_is_mean_of_weighted_crop_losses_and_head_can_fit_separable_fixed_labels():
    torch.set_num_threads(2)
    labels=torch.zeros(2,512,dtype=torch.long);labels[0,:6]=1;labels[1,:12]=1
    features=torch.zeros(2,512,128);features[:,:,0]=labels.float()*2-1
    head=FrozenQueryConfidence(torch.zeros(2,128),torch.zeros(2))
    logits=head(features)
    probe_logits=logits.clone();probe_logits[1,:,1]+=2.
    actual, per_crop=fixed_confidence_loss(probe_logits,labels,recipe())
    expected=torch.stack([F.cross_entropy(probe_logits[i],labels[i],weight=torch.tensor([1.,8.])) for i in range(2)])
    torch.testing.assert_close(per_crop,expected);torch.testing.assert_close(actual,2*expected.mean())
    pooled=F.cross_entropy(probe_logits.reshape(-1,2),labels.reshape(-1),weight=torch.tensor([1.,8.]))
    assert not torch.isclose(actual,2*pooled)
    cache={"features":features,"labels":labels,"geometry":torch.ones(2,512,3)}
    before=tensor_digests(cache)
    optimizer=torch.optim.SGD(head.parameters(),lr=.1)
    for _ in range(20):
        optimizer.zero_grad();loss,_=fixed_confidence_loss(head(features),labels,recipe());loss.backward();optimizer.step()
    assert torch.equal((head(features).softmax(-1)[...,1]>=.5).long(),labels)
    assert sum(p.numel() for p in head.parameters())==258
    assert tensor_digests(cache)==before
    cp=confidence_checkpoint(head,step=20,recipe=recipe(),provenance={"synthetic":True})
    loaded=load_confidence_checkpoint(cp)
    torch.testing.assert_close(head(features),loaded(features),rtol=0,atol=0)
    with pytest.raises(ValueError,match="diagnostic"):
        load_confidence_checkpoint({**cp,"schema":"pointneuron2-pointnext-query-cropnorm-v1"})
    with pytest.raises(ValueError,match="detached"):
        head(features.clone().requires_grad_(True))


def test_capture_replays_source_logits_and_classifier_updates_leave_model_and_geometry_fixed():
    torch.set_num_threads(2)
    model=PointNextCropNormQueryModel(0)
    records=[]
    for crop,count in enumerate((6,12)):
        i=torch.arange(256);xyz=torch.stack((i%16,i//16,torch.zeros_like(i)),-1).float()
        center=torch.tensor([[100.,200.,300.]])
        points=torch.cat((xyz+center,torch.full((256,1),255.)),-1)[None]
        nodes=torch.zeros(1,count,6);nodes[0,:,0]=torch.arange(1,count+1)
        nodes[0,:,1:4]=center+torch.arange(count)[:,None].float();nodes[:,:,4]=1;nodes[:,:,5]=-1
        records.append({"points":points,"centers":center,"scales":torch.tensor([255.]),"nodes":nodes,
            "mask":torch.ones(1,count,dtype=torch.bool),"metadata":{"patch_index":crop},
            "geometry":{"full_segments":torch.zeros(1,2,3).numpy(),"critical_points":torch.zeros(0,3).numpy(),"chains":[]}})
    state=copy.deepcopy(model.state_dict())
    cache,rows,delta=cache_tensors(model,records,{"query_loss":{}})
    assert delta<=1e-5 and cache["features"].shape==(2,512,128)
    assert cache["labels"].sum(1).tolist()==[6,12]
    assert all(len(r["assigned_queries"])==len(set(r["assigned_queries"])) for r in rows)
    before=tensor_digests(cache)
    head=FrozenQueryConfidence(cache["initial_weight"],cache["initial_bias"])
    torch.testing.assert_close(head(cache["features"]),cache["baseline_logits"],rtol=0,atol=1e-5)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0001)
    optimizer.zero_grad();loss,_=fixed_confidence_loss(head(cache["features"]),cache["labels"],recipe());loss.backward();optimizer.step()
    assert tensor_digests(cache)==before
    assert all(torch.equal(t,state[k]) for k,t in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
