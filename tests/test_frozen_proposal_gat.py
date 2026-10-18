"""Oracle-node isolation, incomplete references, topology and exact fit state."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import frozen_proposal_gat as runner
from pointneuron.graph.proposal_graph import build_candidate_graph


@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def recipe():
    return json.loads((ROOT/'configs/pointneuron2/frozen_proposal_gat.json').read_text())


def record():
    # Fork graph: 0--1--2 and 1--3. Saved confidence rejects every node.
    xyz=torch.tensor([[0.,0,0],[3.,0,0],[6.,0,0],[3.,3,0]])
    r={'positions':xyz,'radii':torch.ones(4,1),'objectness_logits':torch.tensor([[10.,-10.]]).repeat(4,1),
       'local_centers':xyz/48.,'decoder_features':torch.randn(4,128,generator=torch.Generator().manual_seed(3)),
       'node_ids':torch.tensor([2,9,17,408]),'sample_index':0,'patch_index':0,'partition':'train'}
    edges=build_candidate_graph(runner.packet(r).positions,r['node_ids']).edges
    required=np.array([[0,1],[1,2],[1,3]])
    r['edges']=edges
    r['labels'],r['required_edges']=runner.oracle_labels(edges,np.arange(4),required,np.zeros(4,int))
    r['reference_segments']=xyz[torch.from_numpy(required)].double()
    return r


def test_oracle_labels_keep_imperfect_positions_out_of_correspondence_and_unknowns_masked():
    edges=torch.tensor([[0,1],[0,2],[1,2]])
    # Target correspondence permuted; two different full-reference components.
    labels,required=runner.oracle_labels(edges,np.array([2,0,1]),np.array([[0,1]]),np.array([5,5,99]))
    assert labels.tolist()==[-1.,-1.,1.] and required.tolist()==[[1,2]]
    with pytest.raises(ValueError,match='one-to-one'):
        runner.oracle_labels(edges,np.array([0,0,2]),np.array([[0,1]]),np.zeros(3,int))
    with pytest.raises(ValueError,match='absent'):
        runner.oracle_labels(torch.tensor([[0,1]]),np.arange(3),np.array([[1,2]]),np.zeros(3,int))


def test_oracle_retention_is_decoder_only_and_graph_gate_detects_same_degree_rewiring(tmp_path):
    r=record(); before=runner.fingerprints([r]); config=recipe()
    logits=torch.where(r['labels']==1,10.,-10.)
    good=runner.graph_metrics(r,logits,config,tmp_path,'good')
    assert good['exact_reference_graph_gate'] and good['supervised_classification_gate']
    assert good['decoded_components']==1 and good['degree_errors']==0
    assert good['swc_roundtrip']['passed'] and runner.fingerprints([r])==before
    assert bool((runner.packet(r).objectness_logits[:,1]<0).all())
    # Same-component extra edge is rejected as an error even if Kruskal removes another.
    wrong=logits.clone(); wrong[0]=20.; wrong[-1]=20.
    assert not runner.graph_metrics(r,wrong,config)['exact_reference_graph_gate']


def test_masked_unknown_selection_is_reported_and_not_supervised_as_negative():
    r=record(); r['labels'][-1]=-1
    logits=torch.where(r['labels']==1,10.,-10.); logits[-1]=20.
    metrics=runner.graph_metrics(r,logits,recipe())
    assert metrics['unknown_edges_above_cutoff']==1
    assert metrics['known_negative_edges_accepted']==0
    assert metrics['supervised_classification_gate']
    assert not metrics['exact_reference_graph_gate']


def test_only_gat_parameters_change_and_disk_resume_repeats_next_update(tmp_path):
    r=record(); config=recipe(); before=runner.fingerprints([r]); provenance={'synthetic':True}
    model,opt=runner.fresh(config,'cpu')
    initial={k:v.clone() for k,v in model.state_dict().items()}
    runner.update(model,opt,r,config,'cpu')
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
    cp=runner.checkpoint(model,opt,1,config,provenance)
    path=tmp_path/'state.pt'; torch.save(cp,path)
    loaded,other,step=runner.restore(torch.load(path,weights_only=True),config,provenance,'cpu',resume=True)
    assert step==1
    a=runner.update(model,opt,r,config,'cpu'); b=runner.update(loaded,other,r,config,'cpu')
    assert a==b
    assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),loaded.state_dict().values()))
    assert runner.fingerprints([r])==before
    changed=copy.deepcopy(config); changed['updates']=8000
    with pytest.raises(ValueError): runner.restore(cp,changed,provenance,'cpu',resume=True)
    check=runner.checkpoint(model,opt,2,config,provenance,purpose='timing')
    with pytest.raises(ValueError,match='Terminal/check-only'): runner.restore(check,config,provenance,'cpu',resume=True)


def test_checkpoint_precedes_metric_failure(monkeypatch,tmp_path):
    r=record(); config=recipe(); model,opt=runner.fresh(config,'cpu'); report={'evaluations':[]}
    def broken(*args,**kwargs): raise RuntimeError('deliberate metric failure')
    monkeypatch.setattr(runner,'evaluate',broken)
    with pytest.raises(RuntimeError,match='deliberate'):
        runner.snapshot(model,opt,0,config,{},report,tmp_path,[r],'cpu')
    assert (tmp_path/'step_0000.pt').exists()
    assert json.loads((tmp_path/'report.json').read_text())['last_checkpoint_step']==0
