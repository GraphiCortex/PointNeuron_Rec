"""Whole-grid denominators, macro aggregation and sparse-target curve baseline."""
from types import SimpleNamespace

import numpy as np
import pytest

from pn2.evaluation.node_cohort import score_box,summarize


def ref(points,runs=()):
    return {'target_xyz':np.asarray(points,dtype=float).reshape(-1,3),
            'target_roles':[frozenset(('leaf',)) for _ in points],'reference_runs':list(runs)}


def score(cid,tile,points,reference):
    xyz=np.asarray(points,dtype=float).reshape(-1,3)
    return score_box(cid,tile,xyz,np.ones(len(xyz)),reference)


def dataset(cases):
    return SimpleNamespace(cases={cid:{'partition':'development','alias':i,'source_group':'fixture'}
        for i,(cid,counts) in enumerate(cases.items())},
        boxes={(cid,tile):{'target':{'count':n}} for cid,counts in cases.items() for tile,n in enumerate(counts)})


def test_unknown_boxes_remain_in_precision_denominator_and_are_reported_separately():
    data=dataset({'a':[1,0]})
    rows=[score('a',0,[[0,0,0]],ref([[0,0,0]])),score('a',1,[[30,0,0]],ref([]))]
    result=summarize(data,rows);n=result['neurons'][0]['tolerances']['2.0']
    assert n['full_grid']['precision']==.5 and n['reference_present_only']['precision']==1.
    assert result['partitions']['development']['unknown_box_accepted']==1


def test_macro_is_per_neuron_not_dominated_by_more_reference_nodes():
    points=[[i*10,0,0] for i in range(9)]
    data=dataset({'large':[9],'small':[1]})
    result=summarize(data,[score('large',0,points,ref(points)),score('small',0,[],ref([[0,0,0]]))])
    metric=result['partitions']['development']['tolerances']['2.0']
    assert metric['macro_recall']==.5 and metric['pooled_reference_agreement']['recall']==.9


def test_missing_unknown_box_is_not_accepted_as_a_complete_evaluation():
    data=dataset({'a':[1,0]})
    with pytest.raises(ValueError,match='every image box'):
        summarize(data,[score('a',0,[[0,0,0]],ref([[0,0,0]]))])


def test_duplicates_do_not_create_extra_recall_or_get_silently_removed():
    row=score('a',0,[[0,0,0],[0,0,0]],ref([[0,0,0]]))
    metric=row['tolerances']['2.0']['nodes']
    assert metric['matched']==1 and metric['precision']==.5 and metric['extras']==1
    assert row['duplicate_accepted_pairs_within_two_voxels']==1


def test_perfect_sparse_targets_expose_their_own_curve_support_ceiling():
    points=[[0,0,0],[8,0,0]];r=ref(points,[np.asarray(points,dtype=float)])
    row=score('a',0,points,r)['tolerances']['2.0']
    assert row['nodes']['f1']==1.
    assert row['curve_to_accepted_node_support']==.5
    assert row['curve_to_target_grid_support']==.5


def test_unknown_reference_does_not_fabricate_curve_or_node_recall():
    row=score('a',0,[[0,0,0]],ref([]))['tolerances']['2.0']
    assert row['nodes']['recall'] is None and row['curve_to_accepted_node_support'] is None
    assert row['accepted_near_reference_geometry'] is None


def test_point_on_continuous_curve_can_disagree_with_the_discrete_node_grid():
    points=[[0,0,0],[8,0,0]];r=ref(points,[np.asarray(points,dtype=float)])
    row=score('a',0,[[4,0,0]],r)['tolerances']['2.0']
    assert row['nodes']['matched']==0 and row['accepted_near_reference_geometry']==1
