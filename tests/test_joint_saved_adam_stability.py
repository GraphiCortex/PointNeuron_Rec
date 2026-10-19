"""Mixed per-parameter optimizer history and unchanged joint update semantics."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import check_joint_saved_adam_stability as check
from pointneuron.models.query_fit_state import cpu_snapshot
from pointneuron.models.query_proposal import QueryProposalOutput
from pointneuron.models.joint_query_graph import JointOutput
from pointneuron.graph.proposal_graph import ProposalGraphInput,CandidateGraph
import numpy as np


class TwoParts(nn.Module):
    def __init__(self):
        super().__init__();self.proposal_model=nn.Linear(3,2);self.graph_model=nn.Linear(2,1)
    def forward(self,x):return self.graph_model(self.proposal_model(x))


def source(model):
    optimizer=torch.optim.AdamW(model.proposal_model.parameters(),lr=.0001,weight_decay=.0001)
    for p in model.proposal_model.parameters():p.grad=torch.ones_like(p)
    optimizer.step();model.zero_grad()
    return cpu_snapshot(optimizer.state_dict())


def test_mixed_state_matches_separate_adam_updates_with_one_global_clip():
    torch.manual_seed(5);model=TwoParts();saved=source(model);other=copy.deepcopy(model)
    recipe={'learning_rate':.0001,'weight_decay':.0001}
    mixed=check.mixed_optimizer(model,saved,recipe,expected_step=1)
    proposal=check.control.make_optimizer(other.proposal_model,saved,restored=True,expected_step=1)
    graph=torch.optim.AdamW(other.graph_model.parameters(),**{'lr':.0001,'weight_decay':.0001})
    assert check.cursors(model,mixed)=={'proposal':[1.],'gat':[0.]}
    assert len(mixed.param_groups)==1
    for _ in range(2):
        for network,optimizers in ((model,[mixed]),(other,[proposal,graph])):
            for optimizer in optimizers:optimizer.zero_grad()
            network(torch.ones(2,3)).square().sum().backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),.5)
            for optimizer in optimizers:optimizer.step()
    assert all(torch.equal(x,y) for x,y in zip(model.parameters(),other.parameters()))
    assert check.cursors(model,mixed)=={'proposal':[3.],'gat':[2.]}
    assert all(float(s['step'])==1 for s in saved['state'].values())


def test_rejects_changed_adam_hyperparameters():
    model=TwoParts();saved=source(model)
    with pytest.raises(ValueError,match='Different Adam options'):
        check.mixed_optimizer(model,saved,{'learning_rate':.01,'weight_decay':.0001},expected_step=1)


def test_stability_gate_does_not_confuse_completion_with_criteria():
    def probe(trusted=6,positive=5,negative=8):
        return {'edge_supervision':{'trusted_matches':trusted,'positive':positive,'negative':negative}}
    report={'completed_updates':8,'snapshots':[{'after_updates':i,'crops':[probe() for _ in range(4)]} for i in (4,8)]}
    assert check.stability_gate(report)['passed']
    report['snapshots'][0]['crops'][0]=probe(trusted=5)
    assert not check.stability_gate(report)['passed']
    report['snapshots'][0]['crops'][0]=probe();report['snapshots'][1]['crops'][3]=probe(negative=0)
    assert not check.stability_gate(report)['passed']
    report['completed_updates']=4
    assert not check.stability_gate(report)['passed']


def test_supervision_stop_persists_incoming_counts_and_predictions_before_any_step(monkeypatch,tmp_path):
    model=TwoParts();optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    before={k:v.clone() for k,v in model.state_dict().items()}
    xyz=torch.tensor([[0.,0,0],[3.,0,0],[10.,0,0]],requires_grad=True)
    radii=torch.ones(3,1,requires_grad=True);logits=torch.zeros(3,2,requires_grad=True)
    features=torch.zeros(3,128,requires_grad=True)
    proposals=QueryProposalOutput(xyz[None],logits[None],radii[None],xyz[None]/48.,features[None])
    packet=ProposalGraphInput(xyz,radii,logits,xyz/48.,features,torch.arange(3),{})
    candidates=CandidateGraph(torch.tensor([[0,1],[0,2],[1,2]]),2,8.)
    output=JointOutput(proposals,packet,candidates,torch.zeros(3,requires_grad=True))
    record={'nodes':np.array([[0,0,0,0,1,0],[1,3,0,0,1,0]],dtype=np.float32),
        'target_edges':np.array([[0,1]]),'target_components':np.array([0,0])}
    monkeypatch.setattr(check.runner,'forward',lambda *args:output)
    attempt={'optimizer_step_executed':False,'failure_output_directory':str(tmp_path)};saved=[]
    with pytest.raises(ValueError,match='known positive and negative'):
        check.instrumented_update(model,optimizer,record,{'query_loss':{}},
            {'edge_weight':1.,'trusted_match_tolerance':2.,'gradient_clip':10.},attempt,lambda:saved.append(copy.deepcopy(attempt)))
    assert saved[0]['before']['edge_supervision']['positive']==1
    assert saved[0]['before']['edge_supervision']['negative']==0
    assert not attempt['optimizer_step_executed'] and not optimizer.state
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    with np.load(tmp_path/'failure_incoming_predictions.npz') as archive:
        assert np.array_equal(archive['centers'],xyz.detach().numpy()[None])
