"""Real gradient paths, late filtering, unknown labels and recoverable joint state."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import joint_query_graph_experiment as runner
from pointneuron.models.joint_query_graph import JointQueryGraph,retained_graph
from pointneuron.models.query_proposal import QueryProposalOutput
from pointneuron.models.gatv2_connectivity import GATv2Connectivity,ConnectivityConfig
from pointneuron.graph.proposal_graph import CandidateGraph


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class SmallProposal(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(4,128)
        self.proposal=nn.Linear(128,6)

    def model_spec(self):return {'synthetic':'eight-query-gradient-control'}

    def forward(self,points,centers,scales,radius):
        features=self.encoder(points/32.).tanh()
        outputs=self.proposal(features)*.05
        xyz=points[:,:,:3]+outputs[:,:,:3]
        return QueryProposalOutput(xyz,outputs[:,:,3:5],outputs[:,:,5:6].exp(),xyz/radius,features)


def fixture():
    torch.manual_seed(4)
    model=JointQueryGraph(SmallProposal(),GATv2Connectivity(ConnectivityConfig(hidden_channels=8,heads=2,layers=1)),neighbors=7)
    points=np.array([[0,0,0,1],[3,0,0,1],[6,0,0,1],[0,10,0,1],[3,10,0,1],
        [6,10,0,1],[9,10,0,1],[12,10,0,1]],dtype=np.float32)
    nodes=np.column_stack((np.arange(3),points[:3,:3],np.ones(3),np.zeros(3))).astype(np.float32)
    record={'points':points,'nodes':nodes,'target_edges':np.array([[0,1],[1,2]]),
        'target_components':np.zeros(3,dtype=np.int64),'meta':{'patch_center':[0,0,0],'image_scale':1.}}
    recipe=json.loads((ROOT/'configs/pointneuron2/joint_query_graph.json').read_text())
    original={'patch_radius':48.,'query_loss':{}}
    return model,record,original,recipe


def test_edge_only_gradients_reach_all_upstream_outputs_and_unmatched_context():
    model,record,original,recipe=fixture();output=runner.forward(model,record,original,'cpu')
    for t in (output.packet.positions,output.packet.radii,output.packet.objectness_logits,output.packet.decoder_features):t.retain_grad()
    _,_,edge,labels=runner.loss(output,record,original,recipe)
    assert (edge.positive_count,edge.negative_count,edge.unknown_count)==(2,1,25)
    assert labels.statistics['trusted_matches']==3 and len(output.packet.positions)==8
    edge.total.backward()
    for t in (output.packet.positions,output.packet.radii,output.packet.objectness_logits,output.packet.decoder_features):
        assert t.grad is not None and bool(torch.isfinite(t.grad).all()) and float(t.grad.abs().sum())>0
    # Unmatched proposals receive context gradients, while their incident edge labels stay unknown.
    assert float(output.packet.decoder_features.grad[3:].abs().sum())>0
    assert all(runner.grad_norm(m.parameters())>0 for m in (model.proposal_model.encoder,model.proposal_model.proposal,model.graph_model))


def test_final_filter_inherits_scores_and_edges_without_rebuilding_candidates():
    model,record,original,_=fixture();output=runner.forward(model,record,original,'cpu')
    logits=torch.tensor([[0.,1.],[1.,0.],[0.,1.]]+[[1.,0.]]*5)
    packet=replace(output.packet,objectness_logits=logits)
    graph=CandidateGraph(torch.tensor([[0,1],[1,2],[0,3]]),1,8.)
    selected,c,scores=retained_graph(replace(output,packet=packet,candidates=graph,edge_logits=torch.tensor([1.,2.,3.])))
    assert selected.node_ids.tolist()==[0,2] and c.edges.shape==(0,2) and scores.shape==(0,)
    graph=CandidateGraph(torch.tensor([[0,2],[1,2],[2,4]]),1,8.)
    selected,c,scores=retained_graph(replace(output,packet=packet,candidates=graph,edge_logits=torch.tensor([7.,8.,9.])))
    assert c.edges.tolist()==[[0,1]] and scores.tolist()==[7.]
    empty,empty_c,_=retained_graph(replace(output,packet=replace(packet,objectness_logits=torch.tensor([[1.,0.]]*8))))
    assert len(empty.positions)==0 and empty_c.edges.shape==(0,2)


def test_unknown_annotation_and_loss_class_collapse_stop_without_relabeling():
    model,record,original,recipe=fixture();output=runner.forward(model,record,original,'cpu')
    unknown=copy.deepcopy(record);unknown['nodes']=np.empty((0,6),np.float32)
    with pytest.raises(ValueError,match='Missing annotation is unknown'):runner.loss(output,unknown,original,recipe)
    no_edges=copy.deepcopy(record);no_edges['target_edges']=np.empty((0,2),dtype=np.int64)
    with pytest.raises(ValueError,match='known positive and negative'):runner.loss(output,no_edges,original,recipe)
    untrusted=copy.deepcopy(record);untrusted['nodes'][:,1:4]+=100
    with pytest.raises(ValueError,match='known positive and negative'):runner.loss(output,untrusted,original,recipe)


def test_both_models_update_and_exact_disk_resume_preserves_next_update(tmp_path):
    model,record,original,recipe=fixture();other,_,_,_=fixture()
    optimizer=runner.optimizer_for(model,recipe);before=runner.state_hashes(model);inputs=record['points'].copy()
    runner.update(model,optimizer,record,original,recipe,'cpu');after=runner.state_hashes(model)
    assert any(before[k]!=after[k] for k in before if k.startswith('proposal_model.'))
    assert any(before[k]!=after[k] for k in before if k.startswith('graph_model.'))
    path=tmp_path/'joint.pt';torch.save(runner.checkpoint(model,optimizer,1,recipe,{}),path)
    cp=torch.load(path,weights_only=True);restored,step=runner.restore(cp,other,recipe,{},resume=True)
    assert step==1
    a=runner.update(model,optimizer,record,original,recipe,'cpu');b=runner.update(other,restored,record,original,recipe,'cpu')
    assert a==b and runner.state_hashes(model)==runner.state_hashes(other)
    assert np.array_equal(record['points'],inputs)
    changed=copy.deepcopy(recipe);changed['updates']=8000
    with pytest.raises(ValueError):runner.restore(cp,other,changed,{},resume=True)
    cp['next_crop']=3
    with pytest.raises(ValueError):runner.restore(cp,other,recipe,{},resume=True)
    cp['next_crop']=1;cp['purpose']='timing';cp['resumable']=False
    with pytest.raises(ValueError,match='Terminal/check-only'):runner.restore(cp,other,recipe,{},resume=True)


def test_checkpoint_precedes_metric_failure(monkeypatch,tmp_path):
    model,record,original,recipe=fixture();optimizer=runner.optimizer_for(model,recipe);report={'evaluations':[]}
    def broken(*args,**kwargs):raise RuntimeError('deliberate evaluation failure')
    monkeypatch.setattr(runner,'evaluate',broken)
    with pytest.raises(RuntimeError,match='deliberate'):
        runner.snapshot(model,optimizer,0,recipe,{},report,tmp_path,[record],[],original,'cpu')
    assert (tmp_path/'step_0000.pt').exists()
    assert json.loads((tmp_path/'report.json').read_text())['last_checkpoint_step']==0


def test_automatic_gate_requires_geometry_nodes_graph_junctions_and_export():
    row={'geometry':{'tolerances':{'2.0':{'precision':.999,'recall':.999}}},'assigned_nodes':3,'unmatched_nodes':0,
        'decoded_missing_required_edges':0,'decoded_nonreference_edges':0,'decoding':{'components':1},'reference_components':1,
        'retained_trusted_junctions':1,'reference_junctions':1,'retained_junction_incidence_errors':0,'swc_roundtrip':{'passed':True}}
    assert runner.automatic_gate(row,3)
    for key,value in [('assigned_nodes',2),('unmatched_nodes',1),('decoded_missing_required_edges',1),
        ('decoded_nonreference_edges',1),('retained_trusted_junctions',0),('retained_junction_incidence_errors',1),
        ('decoding',{'components':2}),('swc_roundtrip',{'passed':False}),
        ('geometry',{'tolerances':{'2.0':{'precision':.99,'recall':.98}}})]:
        bad=copy.deepcopy(row);bad[key]=value;assert not runner.automatic_gate(bad,3)


def test_reload_check_compares_actual_predictions():
    model,record,original,_=fixture()
    before=runner.prediction_signature(model,[record],original,'cpu')
    assert runner.require_identical_predictions(before,runner.prediction_signature(model,[record],original,'cpu'))==0.
    with torch.no_grad():model.graph_model.edge_head[-1].bias.add_(1.)
    with pytest.raises(ValueError,match='reload changed'):runner.require_identical_predictions(before,runner.prediction_signature(model,[record],original,'cpu'))


def test_historical_replay_distinguishes_cross_backend_report_from_gpu_gate(tmp_path):
    old=tmp_path/'old/model';old.mkdir(parents=True);current=tmp_path/'new';current.mkdir()
    saved={'centers':np.ones((1,8,3),np.float32),'radii':np.ones((1,8,1),np.float32),
        'logits':np.zeros((1,8,2),np.float32),'decoder_features':np.ones((1,8,128),np.float32)}
    for i in range(4):
        np.savez(old/f'step_4000_crop_{i}.npz',**saved)
        different={k:v.copy() for k,v in saved.items()};different['centers']+=.077
        np.savez(current/f'step_0000_crop_{i}.npz',**different)
    recipe={'proposal_experiment':str(old.parent)}
    report=runner.historical_warm_replay(current,recipe,torch.device('cpu'))
    assert report[0]['max_center_coordinate_delta']>.07 and report[0]['comparison']=='cross_backend_report_only'
    with pytest.raises(AssertionError):runner.historical_warm_replay(current,recipe,torch.device('cuda'))
