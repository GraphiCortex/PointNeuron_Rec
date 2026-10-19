"""Timing-only disk reload, mixed history, backend isolation and failure evidence."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_saved_adam_timing as timing
from pointneuron.models.query_fit_state import cpu_snapshot
from pointneuron.models.query_proposal import QueryProposalOutput
from pointneuron.models.joint_query_graph import JointOutput
from pointneuron.graph.proposal_graph import ProposalGraphInput,CandidateGraph


class SpecLinear(nn.Linear):
    def model_spec(self):return {'kind':'synthetic','in':self.in_features,'out':self.out_features}


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.proposal_model=SpecLinear(3,2);self.graph_model=SpecLinear(2,1)
    def forward(self,x):return self.graph_model(self.proposal_model(x))


RECIPE={'learning_rate':.0001,'weight_decay':.0001}


def fresh():
    model=Model();optimizer=torch.optim.AdamW(model.proposal_model.parameters(),lr=.0001,weight_decay=.0001)
    for p in model.proposal_model.parameters():p.grad=torch.ones_like(p)
    optimizer.step();model.zero_grad()
    source=cpu_snapshot(optimizer.state_dict())
    for state in source['state'].values():state['step'].fill_(4000)
    return model,timing.stability.mixed_optimizer(model,source,RECIPE)


def step(model,optimizer):
    optimizer.zero_grad();model(torch.ones(2,3)).square().sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()


@pytest.mark.parametrize('updates',[0,2])
def test_mixed_disk_reload_repeats_next_update_without_becoming_resumable(tmp_path,updates):
    model,optimizer=fresh()
    for _ in range(updates):step(model,optimizer)
    path=tmp_path/'timing.pt';cp=timing.timing_checkpoint(model,optimizer,updates,{}, {'backend':'cpu'})
    assert cp['resumable'] is False and cp['purpose']=='throwaway_timing_only'
    torch.save(cp,path);other=Model()
    restored=timing.restore_timing(torch.load(path,weights_only=True),other,RECIPE,{}, {'backend':'cpu'})
    assert timing.stability.cursors(other,restored)==timing.expected_cursors(updates)
    step(model,optimizer);step(other,restored)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),other.parameters()))
    assert timing.stability.control.optimizer_digests(optimizer)==timing.stability.control.optimizer_digests(restored)


def test_checkpoint_rejects_backend_fit_flag_and_wrong_counter():
    model,optimizer=fresh();cp=timing.timing_checkpoint(model,optimizer,0,{}, {'backend':'cpu'})
    with pytest.raises(ValueError,match='backend'):
        timing.restore_timing(cp,model,RECIPE,{}, {'backend':'cuda'})
    bad=copy.deepcopy(cp);bad['resumable']=True
    with pytest.raises(ValueError):timing.restore_timing(bad,model,RECIPE,{}, {'backend':'cpu'})
    bad=copy.deepcopy(cp);bad['optimizer']['state'][0]['step'].add_(1)
    with pytest.raises(ValueError,match='cursor'):timing.restore_timing(bad,model,RECIPE,{}, {'backend':'cpu'})
    with pytest.raises(ValueError):timing.timing_checkpoint(model,optimizer,9,{}, {})


def test_reload_wrapper_checks_predictions_and_removes_temporary_checkpoint(monkeypatch,tmp_path):
    model,optimizer=fresh()
    monkeypatch.setattr(timing.runner,'prediction_signature',lambda m,*args:[(m(torch.ones(2,3)).detach().clone(),)])
    result=timing.reload_check(model,optimizer,0,{},RECIPE,{},[],{},'cpu',tmp_path)
    assert result['prediction_max_delta']==0 and result['weights_and_optimizer_identical']
    assert not list(tmp_path.glob('*.pt*'))
    counter=[0]
    def different(m,*args):
        counter[0]+=1
        return [(m(torch.ones(2,3)).detach().clone()+counter[0],)]
    monkeypatch.setattr(timing.runner,'prediction_signature',different)
    with pytest.raises(ValueError,match='reload changed'):
        timing.reload_check(model,optimizer,0,{},RECIPE,{},[],{},'cpu',tmp_path)
    assert not list(tmp_path.glob('*.pt*'))


def test_original_supervision_stop_records_device_safe_incoming_predictions(monkeypatch,tmp_path):
    model,optimizer=fresh();before=timing.runner.state_hashes(model)
    xyz=torch.tensor([[0.,0,0],[3.,0,0],[10.,0,0]],requires_grad=True)
    radii=torch.ones(3,1,requires_grad=True);logits=torch.zeros(3,2,requires_grad=True);features=torch.zeros(3,128,requires_grad=True)
    proposals=QueryProposalOutput(xyz[None],logits[None],radii[None],xyz[None]/48.,features[None])
    packet=ProposalGraphInput(xyz,radii,logits,xyz/48.,features,torch.arange(3),{})
    output=JointOutput(proposals,packet,CandidateGraph(torch.tensor([[0,1],[0,2],[1,2]]),2,8.),torch.zeros(3,requires_grad=True))
    record={'nodes':np.array([[0,0,0,0,1,0],[1,3,0,0,1,0]],dtype=np.float32),'target_edges':np.array([[0,1]]),
        'target_components':np.array([0,0]),'identity':{'sample_index':0,'patch_index':0}}
    monkeypatch.setattr(timing.runner,'forward',lambda *args:output)
    attempt={'optimizer_step_returned':False};saved=[]
    with pytest.raises(ValueError,match='known positive and negative'):
        timing.instrumented_update(model,optimizer,record,{'query_loss':{}},{'edge_weight':1.,'trusted_match_tolerance':2.,'gradient_clip':10.},
            'cpu',attempt,lambda:saved.append(copy.deepcopy(attempt)),tmp_path)
    assert saved[-1]['incoming']['edge_supervision']['positive']==1 and saved[-1]['incoming']['edge_supervision']['negative']==0
    assert not attempt['optimizer_step_returned'] and timing.runner.state_hashes(model)==before
    assert timing.stability.cursors(model,optimizer)==timing.expected_cursors(0)
    with np.load(tmp_path/'failure_incoming_predictions.npz') as archive:
        assert np.array_equal(archive['center_proposals'],xyz.detach().numpy()[None])
