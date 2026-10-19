"""Capped mixed-state fitting: exact resume, prefix decisions and metric failure."""
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_saved_adam_fit as fit
from pointneuron.models.query_fit_state import cpu_snapshot


class SpecLinear(nn.Linear):
    def model_spec(self):return {'synthetic':True,'in':self.in_features,'out':self.out_features}


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.proposal_model=SpecLinear(3,2);self.graph_model=SpecLinear(2,1)
    def forward(self,x):return self.graph_model(self.proposal_model(x))


RECIPE={'learning_rate':.0001,'weight_decay':.0001}


def config():return fit.runner.base.read(fit.CONFIG)


def fresh():
    model=Model();proposal=torch.optim.AdamW(model.proposal_model.parameters(),lr=.0001,weight_decay=.0001)
    for p in model.proposal_model.parameters():p.grad=torch.ones_like(p)
    proposal.step();model.zero_grad();saved=cpu_snapshot(proposal.state_dict())
    for v in saved['state'].values():v['step'].fill_(4000)
    return model,fit.timing.stability.mixed_optimizer(model,saved,RECIPE)


def update(model,optimizer,crop):
    inputs=torch.randn(2,3)*(crop+1);optimizer.zero_grad()
    model(inputs).square().sum().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
    return inputs


@pytest.mark.parametrize('stage',[0,2])
def test_disk_resume_repeats_next_crop_rng_update_and_optimizer(tmp_path,stage):
    torch.manual_seed(3);np.random.seed(3);random.seed(3)
    model,optimizer=fresh()
    for i in range(stage):update(model,optimizer,i%4)
    cp=fit.checkpoint(model,optimizer,stage,config(),RECIPE,{'backend':'cpu'})
    path=tmp_path/'fit.pt';torch.save(cp,path)
    expected_inputs=update(model,optimizer,stage%4);random_values=(np.random.rand(),random.random())
    other=Model();restored,step,prefix=fit.restore(torch.load(path,weights_only=True),other,config(),RECIPE,{'backend':'cpu'},resume=True)
    assert step==stage and not prefix and cp['next_crop']==stage%4
    assert torch.equal(expected_inputs,update(other,restored,step%4))
    assert random_values==(np.random.rand(),random.random())
    assert fit.runner.state_hashes(model)==fit.runner.state_hashes(other)
    assert fit.timing.stability.control.optimizer_digests(optimizer)==fit.timing.stability.control.optimizer_digests(restored)
    assert fit.timing.stability.cursors(other,restored)==fit.timing.expected_cursors(stage+1)


def test_resume_rejects_changed_cap_backend_data_cursor_names_and_moments():
    model,optimizer=fresh();cp=fit.checkpoint(model,optimizer,0,config(),RECIPE,{'source':'same','backend':'cpu'})
    changed=config();changed['updates']=8000
    with pytest.raises(ValueError):fit.restore(cp,model,changed,RECIPE,{'source':'same','backend':'cpu'},resume=True)
    with pytest.raises(ValueError):fit.restore(cp,model,config(),RECIPE,{'source':'same','backend':'cuda'},resume=True)
    with pytest.raises(ValueError):fit.restore(cp,model,config(),RECIPE,{'source':'different','backend':'cpu'},resume=True)
    for mutate in (lambda c:c.update(next_crop=1),lambda c:c['parameter_names'].reverse(),
                   lambda c:c['optimizer']['state'].pop(0),lambda c:c['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))):
        bad=copy.deepcopy(cp);mutate(bad)
        with pytest.raises(ValueError):fit.restore(bad,Model(),config(),RECIPE,{'source':'same','backend':'cpu'},resume=True)


def test_terminal_and_check_only_are_not_resumable_and_prefix_proof_is_required():
    model,optimizer=fresh();checked=fit.checkpoint(model,optimizer,0,config(),RECIPE,{},purpose='check')
    with pytest.raises(ValueError,match='Terminal/check-only'):fit.restore(checked,model,config(),RECIPE,{},resume=True)
    with pytest.raises(ValueError,match='prefix'):fit.checkpoint(model,optimizer,0,config(),RECIPE,{},prefix_verified=True)
    update(model,optimizer,0)
    for p in model.proposal_model.parameters():optimizer.state[p]['step'].fill_(8000)
    for p in model.graph_model.parameters():optimizer.state[p]['step'].fill_(4000)
    with pytest.raises(ValueError,match='prefix'):fit.checkpoint(model,optimizer,4000,config(),RECIPE,{})
    terminal=fit.checkpoint(model,optimizer,4000,config(),RECIPE,{},prefix_verified=True)
    assert not terminal['resumable'] and terminal['stage_step']==4000
    with pytest.raises(ValueError,match='Terminal/check-only'):fit.restore(terminal,model,config(),RECIPE,{},resume=True)
    fit.restore(terminal,model,config(),RECIPE,{},resume=False)
    assert fit.timing.stability.cursors(model,optimizer)=={'proposal':[8000.],'gat':[4000.]}


def test_complete_checkpoint_survives_failed_evaluation(monkeypatch,tmp_path):
    model,optimizer=fresh();report={'prefix_verified':False,'evaluations':[]}
    def broken(*args,**kwargs):raise RuntimeError('deliberate metric failure')
    monkeypatch.setattr(fit.runner,'evaluate',broken)
    with pytest.raises(RuntimeError,match='deliberate'):
        fit.snapshot(model,optimizer,0,config(),RECIPE,{},report,tmp_path,[],[],{},'cpu')
    path=tmp_path/'step_0000.pt';assert path.exists() and not path.with_suffix('.pt.writing').exists()
    assert json.loads((tmp_path/'report.json').read_text())['last_checkpoint_step']==0
    cp=torch.load(path,weights_only=True);other=Model()
    restored,step,_=fit.restore(cp,other,config(),RECIPE,{},resume=True)
    assert step==0 and fit.timing.stability.cursors(other,restored)==fit.timing.expected_cursors(0)


def saved_prediction():
    edges=np.array([[0,1],[0,2],[1,2]],np.int64);scores=np.array([1.,-1e-7,1.],np.float32)
    return {'centers':np.array([[[0,0,0],[3,0,0],[6,0,0]]],np.float32),'radii':np.ones((1,3,1),np.float32),
        'logits':np.array([[[-1,1],[-1,1],[-1,1]]],np.float32),'decoder_features':np.zeros((1,3,128),np.float32),
        'edge_logits':scores,'retained_edge_logits':scores.copy(),'candidate_edges':edges,'retained_edges':edges.copy(),
        'edge_labels':np.array([1,0,1],np.float32),'evaluation_labels':np.array([1,0,1],np.float32),
        'pred_indices':np.arange(3),'target_indices':np.arange(3),'retained_node_ids':np.arange(3)}


def test_prefix_requires_same_discrete_decisions_even_inside_float_tolerance(tmp_path):
    actual=tmp_path/'actual';expected=tmp_path/'expected';actual.mkdir();expected.mkdir();saved=saved_prediction()
    np.savez(expected/'step_0008_crop_0.npz',**saved);np.savez(actual/'step_0008_crop_0.npz',**saved)
    records=[{'meta':{'patch_center':[0,0,0]}}];original={'patch_radius':48.}
    assert fit.compare_prefix(actual,expected,records,original)['passed']
    changed=copy.deepcopy(saved);changed['edge_logits'][1]=1e-7;changed['retained_edge_logits'][1]=1e-7
    np.savez(actual/'step_0008_crop_0.npz',**changed)
    with pytest.raises(ValueError,match='cutoff'):fit.compare_prefix(actual,expected,records,original)
    changed=copy.deepcopy(saved);changed['target_indices']=np.array([1,0,2]);np.savez(actual/'step_0008_crop_0.npz',**changed)
    with pytest.raises(ValueError,match='mapping'):fit.compare_prefix(actual,expected,records,original)
