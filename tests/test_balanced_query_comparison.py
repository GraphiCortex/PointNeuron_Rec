"""Loss-only gradient controls and placement-error candidate contract."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from pointneuron.models.balanced_query_loss import POLICY, balanced_query_loss, loss_from_assignments
from pointneuron.models.query_loss import query_set_loss
from pointneuron.models.pointnext_query_cropnorm import PointNextCropNormQueryModel
from pointneuron.models.query_fit_state import fit_checkpoint, restore_fit
from pointneuron.graph.proposal_graph import build_candidate_graph


@pytest.mark.parametrize("targets", [6, 418])
def test_only_confidence_gradients_change_and_assignment_geometry_are_identical(targets):
    torch.set_num_threads(2)
    g = torch.Generator().manual_seed(30)
    output = SimpleNamespace(center_proposals=torch.rand(1,512,3,generator=g,requires_grad=True),
        radius=(torch.rand(1,512,1,generator=g)+1).requires_grad_(),
        objectness_logits=torch.randn(1,512,2,generator=g,requires_grad=True))
    nodes=torch.zeros(1,targets,6); nodes[:,:,1:4]=torch.rand(1,targets,3,generator=g); nodes[:,:,4]=1
    mask=torch.ones(1,targets,dtype=torch.bool)
    original=query_set_loss(output,nodes,mask)
    balanced=balanced_query_loss(output,nodes,mask)
    assert original.assignments==balanced.assignments
    assert torch.equal(original.center,balanced.center) and torch.equal(original.radius,balanced.radius)
    variables=(output.center_proposals,output.radius,output.objectness_logits)
    a=torch.autograd.grad(original.total,variables,retain_graph=True)
    b=torch.autograd.grad(balanced.total,variables)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    assert not torch.equal(a[2],b[2]) and all(torch.isfinite(v).all() for v in b)


@pytest.mark.parametrize("targets", [6,418])
def test_equal_class_means_remove_density_prior_at_constant_half_probability(targets):
    bias=torch.tensor(0.,requires_grad=True)
    logits=torch.stack((torch.zeros(512),bias.expand(512)),dim=-1)[None]
    out=SimpleNamespace(center_proposals=torch.zeros(1,512,3),radius=torch.ones(1,512,1),objectness_logits=logits)
    nodes=torch.zeros(1,targets,6); nodes[:,:,4]=1
    assignments=((tuple(range(targets)),tuple(range(targets))),)
    loss=loss_from_assignments(out,nodes,torch.ones(1,targets,dtype=torch.bool),assignments)
    assert abs(float(loss.objectness)-np.log(2))<1e-6
    loss.total.backward()
    assert abs(float(bias.grad))<1e-6


def test_unknown_crops_are_skipped_and_single_class_remains_supervised():
    out=SimpleNamespace(center_proposals=torch.zeros(2,4,3,requires_grad=True),radius=torch.ones(2,4,1,requires_grad=True),
        objectness_logits=torch.zeros(2,4,2,requires_grad=True))
    nodes=torch.zeros(2,4,6); nodes[:,:,4]=1
    mask=torch.tensor([[False]*4,[True]*4])
    loss=balanced_query_loss(out,nodes,mask)
    assert loss.supervised_crops==1 and loss.matched_count==4
    assert abs(float(loss.objectness)-np.log(2))<1e-6
    loss.total.backward(); assert out.objectness_logits.grad[0].abs().sum()==0
    empty=balanced_query_loss(out,nodes,torch.zeros_like(mask))
    assert empty.supervised_crops==0 and float(empty.total.detach())==0


def test_radius_twelve_keeps_worst_case_trusted_adjacency_despite_neighbor_crowding():
    # GT endpoints 0/8; trusted predictions -2/10; each displaced by exactly 2.
    xyz=torch.zeros(42,3); xyz[0,0]=-2; xyz[1,0]=10
    xyz[2:22,0]=torch.linspace(-2.1,-1.9,20); xyz[22:,0]=torch.linspace(9.9,10.1,20)
    ids=torch.arange(42)
    small=build_candidate_graph(xyz,ids,neighbors=16,radius=8.)
    large=build_candidate_graph(xyz,ids,neighbors=16,radius=12.)
    a={tuple(e) for e in small.edges.tolist()}; b={tuple(e) for e in large.edges.tolist()}
    assert (0,1) not in a and (0,1) in b and a<=b


def test_checkpoint_policy_identity_rejects_baseline_resume(tmp_path):
    torch.set_num_threads(2)
    baseline=json.loads((ROOT/'configs/pointneuron2/branched_query_fit.json').read_text())
    recipe=dict(baseline,classification_policy=POLICY)
    model=PointNextCropNormQueryModel(0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    checkpoint=fit_checkpoint(model,optimizer,step=0,recipe=recipe,provenance={'test':'balanced'},backend={'device':'cpu'})
    path=tmp_path/'balanced.pt';torch.save(checkpoint,path)
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    restored,_,step=restore_fit(loaded,recipe=recipe,provenance={'test':'balanced'},backend={'device':'cpu'},device='cpu')
    assert step==0 and all(torch.equal(t,restored.state_dict()[k]) for k,t in model.state_dict().items())
    with pytest.raises(ValueError):
        restore_fit(loaded,recipe=baseline,provenance={'test':'balanced'},backend={'device':'cpu'},device='cpu')


def test_balanced_update_reaches_all_full_model_parameters_on_synthetic_cloud():
    from scripts.pointneuron2.balanced_query_experiment import update
    torch.set_num_threads(2)
    recipe=json.loads((ROOT/'configs/pointneuron2/branched_query_fit.json').read_text())
    g=torch.Generator().manual_seed(7)
    points=torch.rand(256,4,generator=g);points[:,:3]*=96;points[:,3]*=255
    nodes=np.array([[1,40,40,40,1,-1],[2,48,48,48,1,-1]],dtype=np.float32)
    record={'points':points.numpy(),'nodes':nodes,'meta':{'patch_center':[48.,48.,48.],'image_scale':255.}}
    model=PointNextCropNormQueryModel(0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    metrics=update(model,optimizer,record,recipe,'cpu')
    assert np.isfinite(metrics['total'])
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert len(optimizer.state)==len(list(model.parameters()))
