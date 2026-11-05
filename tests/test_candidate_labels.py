import numpy as np
import pytest
from pn2.evaluation.swc import Morphology,empty_morphology
from pn2.supervision.candidate_labels import label_candidates,label_counts,EDGE_REASONS


def morphology(xyz,parents):
    p=np.asarray(xyz,float).reshape(-1,3);n=len(p)
    return Morphology(None,np.arange(1,n+1),np.zeros(n,int),p,np.ones(n),np.array(parents))


def labels(p,e,m):
    return label_candidates(p,e,m,partition='training')


def test_chain_consecutive_skips_coincident_and_bound():
    m=morphology([[0,0,0],[100,0,0]],[-1,0])
    a,_=labels([[0,0,0],[3,0,0],[6,0,0],[40,0,0],[3,.2,0]],
               [[0,1],[1,2],[0,2],[2,3],[1,4]],m)
    assert a['node_label'].tolist()==[1]*5
    assert a['edge_label'].tolist()==[1,1,0,0,-1]
    assert a['edge_reason'][3]==EDGE_REASONS['over_32']
    a,_=labels([[0,0,0],[32,0,0]],[[0,1]],m)
    assert a['edge_label'].tolist()==[1]


def test_node_tolerance_unknown_fragments_and_negative_precedence():
    m=morphology([[0,0,0],[10,0,0],[0,10,0],[10,10,0]],[-1,0,-1,2])
    a,_=labels([[2,2,0],[3,2.00001,0],[2,10,0]],[[0,2],[1,2]],m)
    assert a['node_label'].tolist()==[1,0,1]
    assert a['edge_label'].tolist()==[-1,0]
    assert a['edge_reason'][0]==EDGE_REASONS['fragment_relation_unknown']


def test_coincident_components_are_unknown_not_welded():
    m=morphology([[0,0,0],[10,0,0],[0,0,0],[0,10,0]],[-1,0,-1,2])
    a,_=labels([[0,0,0],[5,0,0],[0,5,0]],[[0,1],[1,2]],m)
    assert a['node_label'].tolist()==[-1,1,1]
    assert a['edge_label'].tolist()==[-1,-1]


def test_nearby_junction_anchor_labels_incidence_without_moving_candidates():
    m=morphology([[-5,0,0],[0,0,0],[5,0,0],[0,5,0]],[-1,0,1,1])
    p=np.array([[-4.,0,0],[-.7,0,0],[4,0,0],[0,4,0]])
    old=p.copy();a,s=labels(p,[[0,1],[1,2],[1,3],[0,2]],m)
    np.testing.assert_array_equal(p,old)
    assert a['edge_label'].tolist()==[1,1,1,-1]
    assert s['junctions_with_anchor']==1 and a['junction_anchor_vertex'][1]==1


def test_distinct_junction_anchors_never_reuse_a_candidate():
    m=morphology([[-5,0,0],[0,0,0],[1,0,0],[5,0,0],[0,5,0],[1,-5,0]],[-1,0,1,2,1,2])
    a,s=labels([[.5,0,0]],[],m)
    assert s['junctions']==2 and s['junctions_with_anchor']==1


def test_empty_isolated_and_partition_guard():
    a,_=labels([[0,0,0]],[],empty_morphology())
    assert label_counts(a)['node']['masked_fraction']==1
    m=morphology([[0,0,0],[10,0,0]],[-1,-1])
    a,_=labels([[0,0,0],[1,0,0],[10,0,0]],[[0,1],[1,2]],m)
    assert a['node_label'].tolist()==[1,1,1] and a['edge_label'].tolist()==[-1,-1]
    with pytest.raises(ValueError,match='training-only'):
        label_candidates([],[],m,partition='development')
