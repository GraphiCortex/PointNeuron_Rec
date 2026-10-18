"""Selection independence, causal edge accounting, unknown extras and isolates."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.pointneuron2 import evaluate_gat_automatic_selection as audit
from pointneuron.graph.proposal_graph import CandidateGraph
from pointneuron.graph.representation_oracle import ReferenceGraph


def example(ids):
    saved={'centers':np.array([[[0.,0,0],[3,0,0],[6,0,0],[3,3,0]]],np.float32),
           'radii':np.ones((1,4,1),np.float32),'logits':np.array([[[0,1],[0,-1],[0,1],[0,1]]],np.float32),
           'decoder_features':np.zeros((1,4,128),np.float32),'pred_indices':np.array([0,1,2]),'target_indices':np.arange(3)}
    nodes=np.zeros((3,6)); nodes[:,1:4]=saved['centers'][0,:3]
    original={'nodes':nodes,'target_edges':np.array([[0,1],[1,2]]),'target_components':np.zeros(3,int),
              'junction_mask':np.zeros(3,bool)}
    reference=ReferenceGraph([('target',i) for i in range(3)],nodes[:,1:4],np.ones(3),[(0,1),(1,2)])
    return saved,original,reference,audit.make_packet(saved,np.array(ids,dtype=np.int64),[0,0,0],48.)


def test_automatic_selection_ignores_assignment_and_keeps_inclusive_threshold():
    saved,_,_,_=example([0,2,3]); assert audit.automatic_ids(saved).tolist()==[0,2,3]
    saved['pred_indices']=np.array([3]);saved['target_indices']=np.array([99])
    assert audit.automatic_ids(saved).tolist()==[0,2,3]
    saved['logits'][0,1]=[0,0]
    assert audit.automatic_ids(saved).tolist()==[0,1,2,3]


def test_extra_bridge_is_unknown_not_negative_and_missing_nodes_are_separate():
    saved,orig,ref,p=example([0,2,3])
    c=CandidateGraph(torch.tensor([[0,2],[1,2]]),16,8.)
    row,decoded,labels=audit.assess(p,c,torch.tensor([10.,10.]),saved,orig,ref)
    assert row['missing_edge_decomposition']=={'node_filter':2,'candidate_graph':0,'edge_score':0,'forest_decode':0}
    assert row['decoded_nonreference_edges']==2 and labels.tolist()==[-1.,-1.]
    assert row['decoded_nonreference_edge_categories']['touches_unmatched_query']==2
    assert row['unmatched_nodes']==1 and row['isolated_unmatched_nodes']==0
    assert decoded.statistics['accepted_nodes']==3


def test_isolated_extra_survives_and_edge_loss_stages_add_up():
    saved,orig,ref,p=example([0,1,2,3])
    # Oracle retention bypasses only the otherwise rejected middle node.
    c=CandidateGraph(torch.tensor([[0,1],[0,2]]),16,8.)
    row,_,labels=audit.assess(p,c,torch.tensor([-10.,-10.]),saved,orig,ref,oracle_retention=True)
    assert row['missing_edge_decomposition']=={'node_filter':0,'candidate_graph':1,'edge_score':1,'forest_decode':0}
    assert row['isolated_unmatched_nodes']==1 and row['decoding']['components']==4
    assert labels.tolist()==[1.,0.]


def test_wrong_high_score_cycle_edge_can_displace_a_required_edge():
    saved,orig,ref,p=example([0,1,2])
    c=CandidateGraph(torch.tensor([[0,1],[1,2],[0,2]]),16,8.)
    before=copy.deepcopy(saved)
    row,_,_=audit.assess(p,c,torch.tensor([1.,2.,10.]),saved,orig,ref,oracle_retention=True)
    assert row['missing_edge_decomposition']=={'node_filter':0,'candidate_graph':0,'edge_score':0,'forest_decode':1}
    assert row['decoded_nonreference_edge_categories']['assigned_same_tree_nonadjacency']==1
    assert all(np.array_equal(before[k],saved[k]) for k in saved)
    json.dumps(row,allow_nan=False)


def test_empty_automatic_selection_is_counted_as_missing_not_excluded():
    saved,orig,ref,p=example([])
    c=CandidateGraph(torch.empty(0,2,dtype=torch.long),16,8.)
    row,decoded,labels=audit.assess(p,c,torch.empty(0),saved,orig,ref)
    assert decoded.status=='empty_reconstruction' and row['missing_assigned_targets']==3
    assert row['missing_edge_decomposition']['node_filter']==2
    assert row['geometry']['tolerances']['2.0']['precision'] is None
    assert row['geometry']['tolerances']['2.0']['recall']==0
    assert row['geometry']['reference_distance_unbounded_due_to_empty_prediction']
    json.dumps(row,allow_nan=False)
