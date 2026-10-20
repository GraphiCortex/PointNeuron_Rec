"""Full-state bounded stable-LR fit: replay arithmetic, resume and stopping decisions."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import stable_joint_learnability as fit
from pointneuron.models.partial_joint_loss import step_active_loss
from pointneuron.models.query_fit_state import cpu_snapshot,capture_rng
from test_joint_partial_stability import Model,initialized,RECIPE


def config():return fit.base.read(fit.CONFIG)


def start():
    model,optimizer=initialized()
    for group in optimizer.param_groups:group['lr']=1e-5
    return model,optimizer


def update(model,optimizer,active,x):
    optimizer.zero_grad(set_to_none=True);proposal,edge=model.losses(x)
    attempt={};fit.apply_update(model,optimizer,proposal+edge if active else proposal,active,attempt,lambda:None,dict(RECIPE,gradient_clip=10.))
    return attempt


def test_update_arithmetic_matches_completed_short_controller_exactly_including_skipped_gat():
    a,oa=start();b=Model();b.load_state_dict(a.state_dict());ob=fit.runner.optimizer_for(b,dict(RECIPE,learning_rate=1e-5))
    ob.load_state_dict(cpu_snapshot(oa.state_dict()))
    graph_steps=0
    for q in range(1,65):
        active=q not in (17,34);x=torch.randn(2,3)
        actual=update(a,oa,active,x)
        ob.zero_grad(set_to_none=True);proposal,edge=b.losses(x)
        expected=step_active_loss(b,ob,proposal+edge if active else proposal,edge_active=active)
        for key,value in expected.items():assert actual[key]==value
        graph_steps+=int(active);fit.validate_state(a,oa,q,graph_steps,config(),RECIPE)
        assert fit.control.equal(a.state_dict(),b.state_dict()) and fit.control.equal(oa.state_dict(),ob.state_dict())


@pytest.mark.parametrize('stage',[0,64,500])
def test_disk_resume_preserves_cursors_rng_and_next_update(tmp_path,stage):
    torch.manual_seed(5);np.random.seed(5);random.seed(5)
    model,optimizer=start()
    graph_stage=stage-1 if stage>64 else stage
    for p in model.proposal_model.parameters():optimizer.state[p]['step'].fill_(5000+stage)
    for p in model.graph_model.parameters():optimizer.state[p]['step'].fill_(1000+graph_stage)
    saved=fit.checkpoint(model,optimizer,stage,graph_stage,stage>64,config(),RECIPE,{'backend':'cpu'})
    file=tmp_path/'state.pt';torch.save(saved,file)
    inputs=torch.randn(2,3);draws=(np.random.rand(),random.random())
    update(model,optimizer,True,inputs)
    other=Model();loaded,q,g,prefix=fit.restore(torch.load(file,weights_only=True),other,config(),RECIPE,{'backend':'cpu'},resume=True)
    assert q==stage and g==graph_stage and saved['historical_stage']==1000+stage and saved['next_crop']==stage%4
    assert torch.equal(inputs,torch.randn(2,3)) and draws==(np.random.rand(),random.random())
    update(other,loaded,True,inputs)
    assert fit.control.equal(model.state_dict(),other.state_dict()) and fit.control.equal(optimizer.state_dict(),loaded.state_dict())
    fit.validate_state(other,loaded,stage+1,graph_stage+1,config(),RECIPE)


def test_cap_policy_provenance_parameter_names_and_moments_are_strict():
    model,optimizer=start();saved=fit.checkpoint(model,optimizer,0,0,False,config(),RECIPE,{'source':'fixed'})
    changed=config();changed['updates']=4000
    with pytest.raises(ValueError):fit.restore(saved,model,changed,RECIPE,{'source':'fixed'},resume=True)
    with pytest.raises(ValueError):fit.restore(saved,model,config(),RECIPE,{'source':'different'},resume=True)
    for mutate in (lambda s:s.update(historical_stage=1001),lambda s:s.update(next_crop=1),lambda s:s['parameter_names'].reverse(),
                   lambda s:s['optimizer']['state'][0]['exp_avg'].fill_(float('nan')),lambda s:s['optimizer']['param_groups'][0].update(lr=1e-4)):
        bad=copy.deepcopy(saved);mutate(bad)
        with pytest.raises(ValueError):fit.restore(bad,Model(),config(),RECIPE,{'source':'fixed'},resume=True)
    with pytest.raises(ValueError):fit.checkpoint(model,optimizer,0,0,True,config(),RECIPE,{})
    with pytest.raises(ValueError):fit.validate_state(model,optimizer,3001,3001,config(),RECIPE)


@pytest.mark.parametrize('purpose,stage',[('check',0),('geometry_stop',500),('fit',3000)])
def test_check_stop_and_terminal_checkpoints_cannot_resume(purpose,stage):
    model,optimizer=start()
    for p in model.proposal_model.parameters():optimizer.state[p]['step'].fill_(5000+stage)
    for p in model.graph_model.parameters():optimizer.state[p]['step'].fill_(1000+stage)
    saved=fit.checkpoint(model,optimizer,stage,stage,stage>=64,config(),RECIPE,{},purpose)
    assert not saved['resumable']
    with pytest.raises(ValueError,match='cannot resume'):fit.restore(saved,model,config(),RECIPE,{},resume=True)
    fit.restore(saved,model,config(),RECIPE,{},resume=False)


def test_checkpoint_survives_a_failed_metric_export(monkeypatch,tmp_path):
    model,optimizer=start();report={'proposal_updates':0,'graph_updates':0,'prefix_verified':False,'evaluations':[]}
    def fail(*args,**kwargs):raise RuntimeError('deliberate metric error')
    monkeypatch.setattr(fit.control,'evaluate',fail)
    with pytest.raises(RuntimeError,match='metric error'):fit.snapshot(model,optimizer,report,config(),RECIPE,{},tmp_path,[],[],{},'cpu')
    assert (tmp_path/'stage_1000.pt').exists()
    saved=torch.load(tmp_path/'stage_1000.pt',weights_only=True)
    _,q,g,_=fit.restore(saved,Model(),config(),RECIPE,{},resume=True)
    assert q==g==0


def test_optimizer_return_is_recorded_before_poststep_validation():
    model,optimizer=start();optimizer.zero_grad(set_to_none=True);proposal,edge=model.losses(torch.ones(2,3))
    phases=[];attempt={}
    fit.apply_update(model,optimizer,proposal+edge,True,attempt,lambda:phases.append((attempt.get('phase'),attempt.get('optimizer_step_returned'))),dict(RECIPE,gradient_clip=10.))
    assert phases[-1]==('optimizer_step_returned',True)


def test_probe_cadence_geometry_stop_and_tail_use_the_declared_windows():
    assert [q for q in range(1,101) if fit.is_probe_step(q)]==list(range(4,65,4))+[80,100]
    probes=[{'crops':[{'matched_distance_mean':4.1,'trusted_fraction':1.} for _ in range(4)]} for _ in range(3)]
    assert fit.geometry_stop(probes) and not fit.geometry_stop(probes[:2])
    report={'probes':[{'crops':[{'trusted_fraction':1.} for _ in range(4)]} for _ in range(8)],
        'updates':[{'crop':crop,'incoming':{'edge_supervision':{'trusted_matches':1,'targets':1}}} for _ in range(8) for crop in config()['training_crops']]}
    assert fit.stable_tail(report,config())
    report['updates'][-1]['incoming']['edge_supervision']['trusted_matches']=0
    assert not fit.stable_tail(report,config())


def test_exact_prefix_checks_model_adam_and_rng_not_only_prediction_metrics(monkeypatch,tmp_path):
    model,optimizer=start()
    for p in model.proposal_model.parameters():optimizer.state[p]['step'].fill_(5064)
    for p in model.graph_model.parameters():optimizer.state[p]['step'].fill_(1064)
    reference=tmp_path/'source';reference.mkdir()
    saved={'model':cpu_snapshot(model.state_dict()),'optimizer':cpu_snapshot(optimizer.state_dict()),'rng':capture_rng()}
    torch.save(saved,reference/'step_0064_terminal.pt')
    monkeypatch.setattr(fit.control,'read_arrays',lambda path:{'centers':np.zeros((1,1,3))})
    report={'prefix_verified':False}
    fit.verify_prefix(model,optimizer,report,tmp_path,reference,config())
    assert report['prefix_verified'] and report['prefix_model_optimizer_rng_exact']
    optimizer.state[next(model.parameters())]['exp_avg'].add_(1)
    with pytest.raises(ValueError,match='model/Adam/RNG'):fit.verify_prefix(model,optimizer,report,tmp_path,reference,config())
