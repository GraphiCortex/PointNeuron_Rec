"""Validate the sole intervention in the bounded optimizer-state control."""
import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2.compare_proposal_optimizer_state import make_optimizer,optimizer_digests
from pointneuron.models.query_fit_state import cpu_snapshot


def trained():
    model=nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    for p in model.parameters():p.grad=torch.ones_like(p)
    optimizer.step()
    return model,cpu_snapshot(optimizer.state_dict())


def test_only_moments_and_counter_differ_and_saved_state_is_not_mutated():
    model,saved=trained();before=copy.deepcopy(saved)
    fresh=make_optimizer(model,saved,restored=False,expected_step=1)
    restored=make_optimizer(model,saved,restored=True,expected_step=1)
    assert fresh.state_dict()['param_groups']==restored.state_dict()['param_groups']==saved['param_groups']
    assert optimizer_digests(fresh)=={} and len(optimizer_digests(restored))==2
    for state in restored.state.values():
        assert float(state['step'])==1
        state['exp_avg'].zero_()
    assert all(torch.equal(saved['state'][i]['exp_avg'],before['state'][i]['exp_avg']) for i in saved['state'])


def test_rejects_parameter_permutation_bad_moments_and_incomplete_state():
    model,saved=trained()
    bad=copy.deepcopy(saved);bad['param_groups'][0]['params'].reverse()
    with pytest.raises(ValueError,match='parameter order'):make_optimizer(model,bad,restored=True,expected_step=1)
    bad=copy.deepcopy(saved);bad['state'][0]['exp_avg']=torch.ones(99)
    with pytest.raises(ValueError,match='Misaligned'):make_optimizer(model,bad,restored=True,expected_step=1)
    bad=copy.deepcopy(saved);bad['state'].pop(1)
    with pytest.raises(ValueError,match='Incomplete'):make_optimizer(model,bad,restored=True,expected_step=1)
    with pytest.raises(ValueError,match='cursor'):make_optimizer(model,saved,restored=True,expected_step=4000)


def test_restored_state_matches_uninterrupted_next_adam_update():
    model,saved=trained();other=nn.Linear(3,2);other.load_state_dict(model.state_dict())
    a=make_optimizer(model,saved,restored=True,expected_step=1)
    b=torch.optim.AdamW(other.parameters());b.load_state_dict(cpu_snapshot(saved))
    for proposal,optimizer in ((model,a),(other,b)):
        optimizer.zero_grad()
        for p in proposal.parameters():p.grad=torch.full_like(p,.25)
        optimizer.step()
    assert all(torch.equal(x,y) for x,y in zip(model.parameters(),other.parameters()))
    assert optimizer_digests(a)==optimizer_digests(b)
