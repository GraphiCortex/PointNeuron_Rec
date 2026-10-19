"""Partial labels, inactive Adam, independent fork state and noninterfering probes."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_partial_stability as control
from pointneuron.models.partial_joint_loss import partial_joint_loss,step_active_loss
from pointneuron.graph.proposal_graph import ProposalGraphInput,CandidateGraph
from pointneuron.models.query_fit_state import cpu_snapshot,capture_rng


RECIPE={'learning_rate':1e-4,'weight_decay':1e-4}


def config():return control.base.read(control.CONFIG)


class SpecLinear(nn.Linear):
    def model_spec(self):return {'synthetic':True,'in':self.in_features,'out':self.out_features}


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.proposal_model=SpecLinear(3,2);self.graph_model=SpecLinear(2,1)
    def losses(self,x):
        proposal=self.proposal_model(x)
        return proposal.square().sum(),self.graph_model(proposal).square().sum()


def initialized():
    model=Model();optimizer=control.runner.optimizer_for(model,RECIPE)
    optimizer.zero_grad(set_to_none=True);a,b=model.losses(torch.ones(2,3))
    step_active_loss(model,optimizer,a+b,edge_active=True)
    for p in model.proposal_model.parameters():optimizer.state[p]['step'].fill_(5000)
    for p in model.graph_model.parameters():optimizer.state[p]['step'].fill_(1000)
    return model,optimizer


def packet_case(edges,shift=0.):
    xyz=torch.tensor([[0.,0.,0.],[1.,0.,0.],[3.,0.,0.]],requires_grad=True)
    predicted=xyz+torch.tensor([shift,0.,0.]);radii=torch.ones(3,1,requires_grad=True)
    logits=torch.tensor([[-1.,1.],[-1.,1.],[-1.,1.]],requires_grad=True)
    features=torch.zeros(3,128,requires_grad=True)
    packet=ProposalGraphInput(predicted,radii,logits,predicted/48,features,torch.arange(3),{})
    scores=nn.Linear(128,1)(features).reshape(-1)[:len(edges)]
    output=SimpleNamespace(proposals=SimpleNamespace(center_proposals=predicted[None],radius=radii[None],objectness_logits=logits[None]),
        packet=packet,candidates=CandidateGraph(torch.tensor(edges,dtype=torch.long).reshape(-1,2),16,8.),edge_logits=scores)
    nodes=np.zeros((3,6),np.float32);nodes[:,1:4]=xyz.detach().numpy();nodes[:,4]=1
    record={'nodes':nodes,'target_edges':np.array([[0,1],[1,2]],np.int64),'target_components':np.zeros(3,np.int64)}
    return output,record,features


@pytest.mark.parametrize('edges,shift,positive,negative',[
    ([[0,1],[0,2],[1,2]],0.,2,1),([[0,1],[1,2]],0.,2,0),([[0,2]],0.,0,1),([[0,1],[0,2]],30.,0,0)])
def test_partial_loss_uses_only_available_labels_and_detaches_unknown_graph(edges,shift,positive,negative):
    output,record,features=packet_case(edges,shift)
    total,proposal,edge,labels=partial_joint_loss(output,record,{})
    assert (edge.positive_count,edge.negative_count)==(positive,negative)
    expected=[]
    if positive:expected.append(torch.nn.functional.softplus(-output.edge_logits[labels.labels==1]).mean())
    if negative:expected.append(torch.nn.functional.softplus(output.edge_logits[labels.labels==0]).mean())
    if expected:torch.testing.assert_close(total,proposal.total+torch.stack(expected).mean())
    else:
        assert total is proposal.total
        assert int((labels.labels==-1).sum())==len(edges)
    total.backward()
    assert (features.grad is None)==(not expected)


def test_missing_annotation_remains_unknown():
    output,record,_=packet_case([[0,1]])
    record['nodes']=np.empty((0,6),np.float32)
    with pytest.raises(ValueError,match='Missing annotation'):partial_joint_loss(output,record,{})


def test_proposal_only_step_preserves_nonzero_gat_adam_moments_weights_and_clock():
    model,optimizer=initialized();before=control.graph_state(model,optimizer)
    assert any(bool(s['exp_avg'].abs().sum()) for s in before[1].values())
    optimizer.zero_grad(set_to_none=True);proposal,_=model.losses(torch.randn(2,3))
    step_active_loss(model,optimizer,proposal,edge_active=False)
    assert control.equal(before,control.graph_state(model,optimizer))
    assert all(p.grad is None for p in model.graph_model.parameters())
    control.validate_state(model,optimizer,1,0,1e-4,RECIPE)
    optimizer.zero_grad(set_to_none=True);proposal,edge=model.losses(torch.randn(2,3))
    step_active_loss(model,optimizer,proposal+edge,edge_active=True)
    control.validate_state(model,optimizer,2,1,1e-4,RECIPE)


def test_inactive_gat_zero_grad_is_rejected_before_optimizer_step():
    model,optimizer=initialized();before=control.graph_state(model,optimizer)
    optimizer.zero_grad(set_to_none=True);proposal,edge=model.losses(torch.ones(2,3))
    with pytest.raises(ValueError,match='None gradients'):step_active_loss(model,optimizer,proposal+0*edge,edge_active=False)
    assert control.equal(before,control.graph_state(model,optimizer))


def test_active_missing_grad_and_nonfinite_loss_are_errors():
    model,optimizer=initialized();optimizer.zero_grad(set_to_none=True)
    proposal,_=model.losses(torch.ones(2,3))
    with pytest.raises(ValueError,match='active GAT'):step_active_loss(model,optimizer,proposal,edge_active=True)
    with pytest.raises(FloatingPointError):step_active_loss(model,optimizer,torch.tensor(float('nan')),edge_active=False)


def test_observe_preserves_modes_and_rng_and_rejects_rng_or_buffer_mutation():
    model=Model();model.train();model.graph_model.eval();rng=capture_rng()
    assert control.observe(model,lambda:42)==42
    assert model.training and not model.graph_model.training and control.equal(rng,capture_rng())
    with pytest.raises(ValueError,match='Observation changed'):control.observe(model,lambda:torch.rand(1))
    model.register_buffer('moving',torch.zeros(1))
    with pytest.raises(ValueError,match='Observation changed'):control.observe(model,lambda:model.moving.add_(1))


def test_diagnostic_disk_reload_preserves_next_rng_crop_update_and_independent_states(tmp_path):
    torch.manual_seed(4);np.random.seed(4);random.seed(4)
    model,optimizer=initialized();optimizer.zero_grad(set_to_none=True)
    proposal,_=model.losses(torch.randn(2,3));step_active_loss(model,optimizer,proposal,edge_active=False)
    saved=control.snapshot_state(model,optimizer,1,0,1e-4,config(),RECIPE,{'source':'fixed'})
    path=tmp_path/'diagnostic.pt';torch.save(saved,path)
    assert saved['next_crop']==1 and not saved['resumable']
    inputs=torch.randn(2,3);draws=(np.random.rand(),random.random())
    optimizer.zero_grad(set_to_none=True);a,b=model.losses(inputs);step_active_loss(model,optimizer,a+b,edge_active=True)
    other=Model();loaded=torch.load(path,weights_only=True)
    restored=control.restore_snapshot(loaded,other,config(),RECIPE,{'source':'fixed'})
    assert torch.equal(inputs,torch.randn(2,3)) and draws==(np.random.rand(),random.random())
    restored.zero_grad(set_to_none=True);a,b=other.losses(inputs);step_active_loss(other,restored,a+b,edge_active=True)
    assert control.equal(model.state_dict(),other.state_dict()) and control.equal(optimizer.state_dict(),restored.state_dict())
    assert control.equal(saved,loaded)  # Restoring/updating did not mutate the source checkpoint's tensors.
    control.validate_state(other,restored,2,1,1e-4,RECIPE)
    bad=copy.deepcopy(loaded);bad['config']['updates_per_arm']=128
    with pytest.raises(ValueError):control.restore_snapshot(bad,Model(),config(),RECIPE,{'source':'fixed'})
    with pytest.raises(ValueError):control.restore_snapshot(loaded,Model(),config(),RECIPE,{'source':'changed'})
    with pytest.raises(ValueError):control.restore_snapshot(loaded,Model(),config(),dict(RECIPE,weight_decay=.01),{'source':'fixed'})


def test_validate_state_rejects_false_clocks_and_cap_extensions():
    model,optimizer=initialized()
    with pytest.raises(ValueError):control.validate_state(model,optimizer,1,0,1e-4,RECIPE)
    with pytest.raises(ValueError):control.validate_state(model,optimizer,65,0,1e-4,RECIPE)
    with pytest.raises(ValueError):control.validate_state(model,optimizer,0,1,1e-4,RECIPE)


def baseline():
    rows=[]
    for targets,extras,correct,bad in ((6,1,5,1),(12,18,6,20),(16,5,13,7),(418,0,399,5)):
        rows.append({'assigned_nodes':targets,'unmatched_nodes':extras,'decoded_required_edges':correct,'decoded_nonreference_edges':bad,
            'retained_reference_junctions':0,'retained_junction_incidence_errors':0,'output_components_spanning_multiple_reference_crop_components':0,
            'reference_components':1,'decoding':{'components':1},'swc_roundtrip':{'passed':True},
            'geometry':{'tolerances':{'2.0':{'precision':.9,'recall':.9}}}})
    return {'crops':rows,'automatic_gate':False}


def passing_report():
    rows=[{'trusted_fraction':1.,'matched_distance_mean':.5} for _ in range(4)]
    updates=[{'crop':crop,'incoming':{'edge_supervision':{'trusted_matches':1,'targets':1}}}
        for _ in range(16) for crop in config()['training_crops']]
    return {'proposal_updates':64,'probes':[{'crops':copy.deepcopy(rows)} for _ in range(16)],'updates':updates,
        'evaluations':[baseline(),baseline()]}


def test_diagnostic_success_cannot_substitute_for_full_gate_or_hide_incoming_instability():
    report=passing_report();result=control.qualification(report,config())
    assert result['qualified'] and not result['full_automatic_gate']
    report['updates'][-1]['incoming']['edge_supervision']['trusted_matches']=0
    assert not control.qualification(report,config())['qualified']
    report=passing_report();report['proposal_updates']=60
    assert not control.qualification(report,config())['qualified']


def test_geometry_stop_requires_three_consecutive_cycle_violations():
    rows=[{'matched_distance_mean':4.1} for _ in range(4)]
    probes=[{'crops':copy.deepcopy(rows)} for _ in range(3)]
    assert not control.geometry_stop(probes[:2],config()) and control.geometry_stop(probes,config())
    probes[1]['crops']=[{'matched_distance_mean':1.} for _ in range(4)]
    assert not control.geometry_stop(probes,config())


@pytest.mark.parametrize('change',[
    lambda row:row.update(unmatched_nodes=row['unmatched_nodes']+1),
    lambda row:row.update(retained_junction_incidence_errors=1),
    lambda row:row['decoding'].update(components=2),
    lambda row:row['geometry']['tolerances']['2.0'].update(precision=.894),
    lambda row:row['geometry']['tolerances']['2.0'].update(recall=None)])
def test_output_preservation_detects_reconstruction_regressions(change):
    start=baseline();end=copy.deepcopy(start);change(end['crops'][0])
    assert not control.output_preservation(start,end,config())['passed']


def test_replay_rejects_missing_fields_discrete_changes_and_threshold_crossing():
    saved={'centers':np.ones((1,2,3),np.float32),'logits':np.array([[[-1.,1.],[-1.,1.]]],np.float32),
        'edge_logits':np.array([-1e-7],np.float32),'candidate_edges':np.array([[0,1]])}
    control.compare_arrays(saved,copy.deepcopy(saved))
    bad=copy.deepcopy(saved);bad['edge_logits'][0]=1e-7
    with pytest.raises(ValueError,match='cutoff'):control.compare_arrays(bad,saved)
    bad=copy.deepcopy(saved);bad.pop('candidate_edges')
    with pytest.raises(ValueError,match='fields'):control.compare_arrays(bad,saved)
    bad=copy.deepcopy(saved);bad['candidate_edges'][0]=[1,0]
    with pytest.raises(ValueError,match='discrete'):control.compare_arrays(bad,saved)


def test_decision_prefers_current_when_both_qualify_without_quality_advantage():
    a=passing_report();b=passing_report()
    for r in (a,b):r['qualification']=control.qualification(r,config())
    assert control.decision([a,b]).startswith('retain_current')
    b['evaluations'][-1]['crops'][0]['unmatched_nodes']=0
    assert control.decision([a,b]).startswith('reduced_lr')
    a['qualification']['qualified']=False;b['qualification']['qualified']=False
    assert control.decision([a,b]).startswith('neither')


def test_whole_control_replays_prefix_then_handles_single_class_and_inactive_adam(monkeypatch,tmp_path):
    """Exercise the bounded controller, using a synthetic learned model and real Adam."""
    model,optimizer=initialized()
    saved={'model':cpu_snapshot(model.state_dict()),'optimizer':cpu_snapshot(optimizer.state_dict()),
        'rng':capture_rng(),'parameter_names':[n for n,_ in model.named_parameters()],
        'proposal_spec':model.proposal_model.model_spec(),'graph_spec':model.graph_model.model_spec()}
    recipe=dict(RECIPE,gradient_clip=10.,edge_weight=1.)
    def forward(model,record,original,device):
        model.synthetic_step=getattr(model,'synthetic_step',0)+1
        value=model.proposal_model(torch.ones(3,3));edge=model.graph_model(value).reshape(-1)
        proposals=SimpleNamespace(center_proposals=value.mean().expand(1,3,3),radius=value.square().mean().add(1).expand(1,3,1),
            objectness_logits=value[None],decoder_features=value.mean().expand(1,3,128))
        return SimpleNamespace(proposals=proposals,candidates=CandidateGraph(torch.tensor([[0,1],[0,2],[1,2]]),16,8.),
            edge_logits=edge,proposal_loss=value.square().sum(),edge_loss=edge.square().sum(),step=model.synthetic_step)
    def describe(output,record,original,recipe):
        trusted=0 if output.step==20 else 1
        return {'crop':record,'loss':float(output.proposal_loss.detach()+(output.edge_loss.detach() if trusted else 0)),
            'edge_supervision':{'targets':1,'trusted_matches':trusted,'positive':trusted,'negative':int(trusted and output.step!=14)},
            'matched_distance_mean':.1,'matched_distance_max':.1,'matched_query_ids':[0],'matched_target_indices':[0]}
    def partial(output,record,recipe,**kwargs):
        active=output.step!=20
        return (output.proposal_loss+output.edge_loss if active else output.proposal_loss,None,SimpleNamespace(supervised_crops=int(active)),None)
    monkeypatch.setattr(control.runner,'forward',forward);monkeypatch.setattr(control.timing,'describe',describe)
    monkeypatch.setattr(control,'partial_joint_loss',partial);monkeypatch.setattr(control.runner,'warm_start',lambda *args:Model())
    monkeypatch.setattr(control,'probe',lambda *args:[{'trusted_fraction':1.,'matched_distance_mean':.1,
        'edge_supervision':{'trusted_matches':1,'targets':1}} for _ in range(4)])
    monkeypatch.setattr(control,'evaluate',lambda *args:baseline())
    synthetic_arrays={'centers':np.ones((1,3,3),np.float32)}
    monkeypatch.setattr(control,'read_arrays',lambda path:synthetic_arrays if path.name.startswith('step_') else original_failure)
    # Produce an independent reference trajectory and its incoming would-stop packet.
    history=[];source=Model();reference=control.imported_state(source,saved,recipe,1e-4,'cpu')
    crops=config()['training_crops']
    for step in range(1,14):
        reference.zero_grad(set_to_none=True);output=forward(source,crops[(step-1)%4],{},'cpu')
        incoming=describe(output,crops[(step-1)%4],{},recipe)
        norms=step_active_loss(source,reference,output.proposal_loss+output.edge_loss,edge_active=True)
        history.append({'stage_update':1000+step,'crop':crops[(step-1)%4],'incoming':incoming,
            **norms,'cursors_after':control.timing.stability.cursors(source,reference)})
    output=forward(source,crops[1],{},'cpu');incoming=describe(output,crops[1],{},recipe)
    original_failure={k:getattr(output.proposals,k).detach().numpy().copy() for k in ('center_proposals','radius','objectness_logits','decoder_features')}
    original_failure.update(candidate_edges=output.candidates.edges.numpy(),edge_logits=output.edge_logits.detach().numpy().copy())
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'updates.jsonl').write_text('\n'.join(json.dumps(r) for r in history),encoding='utf-8')
    (parent/'active_update.json').write_text(json.dumps({'incoming':incoming}),encoding='utf-8')
    report=control.run_arm(config()['arms'][0],config(),recipe,{'query_loss':{}},crops,[],saved,parent,{},tmp_path/'arm','cpu')
    assert report['status']=='completed' and report['proposal_updates']==64 and report['graph_updates']==63
    assert report['guard_transition_verified'] and report['replay_verified_steps']==list(range(1001,1014))
    assert report['qualification']['qualified'] and not report['qualification']['full_automatic_gate']
    assert control.equal(saved['optimizer'],optimizer.state_dict())
