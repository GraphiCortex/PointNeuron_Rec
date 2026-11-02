"""A resumed stochastic Adam update must reproduce uninterrupted execution."""
import random

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from pn2.training import node_fit_state as state


def setup():
    model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
    return model,optimizer


def update(model,optimizer):
    x=torch.randn(5,3)+float(np.random.random())+random.random()
    optimizer.zero_grad();loss=model(x).square().mean();loss.backward();optimizer.step()
    return float(loss.detach())


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def test_saved_adam_rng_and_next_update_reproduce_uninterrupted_fit(tmp_path):
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    model,opt=setup();update(model,opt);update(model,opt)
    bindings={'maximum_updates':20,'budget':8192,'manifest_sha256':'dataset','schedule_sha256':'order'}
    path=tmp_path/'checkpoint.pt';state.save(path,model,opt,bindings,2)
    expected_loss=update(model,opt)
    expected_model={k:v.clone() for k,v in model.state_dict().items()};expected_adam=opt.state_dict()
    restored,restored_opt=setup()
    assert state.load(path,restored,restored_opt,bindings)==2
    assert update(restored,restored_opt)==expected_loss
    assert equal(restored.state_dict(),expected_model) and equal(restored_opt.state_dict(),expected_adam)


def test_wrong_budget_or_data_is_rejected_before_model_mutation(tmp_path):
    model,opt=setup();bindings={'maximum_updates':20,'budget':2048,'manifest_sha256':'a'}
    path=tmp_path/'checkpoint.pt';state.save(path,model,opt,bindings,0)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError,match='bindings'):
        state.load(path,model,opt,{**bindings,'budget':8192})
    with pytest.raises(ValueError,match='bindings'):
        state.load(path,model,opt,{**bindings,'manifest_sha256':'b'})
    assert equal(before,model.state_dict())


def test_partial_or_modified_checkpoints_are_not_accepted_and_files_not_overwritten(tmp_path):
    model,opt=setup();bindings={'maximum_updates':20}
    path=tmp_path/'checkpoint.pt';state.save(path,model,opt,bindings,0)
    with pytest.raises(FileExistsError):state.save(path,model,opt,bindings,0)
    path.write_bytes(b'interrupted')
    with pytest.raises(ValueError,match='incomplete or changed'):state.load(path,model,opt,bindings)


def test_update_cursor_is_bounded():
    model,opt=setup()
    for step in (-1,21,True):
        with pytest.raises(ValueError,match='cursor'):state.capture(model,opt,{'maximum_updates':20},step)
