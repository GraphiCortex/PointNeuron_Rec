"""Verify observation preserves selector output and restores original functions."""
import numpy as np
import pytest

from scripts.pointneuron2.record_graph_components import SelectionRecorder,initializer


def inputs():
    sizes=[300,250,75];n=sum(sizes)
    xyz=np.zeros((n,3),dtype=np.float32)
    geo=np.full((n,n),np.inf,dtype=np.float32)
    start=0
    for component,size in enumerate(sizes):
        idx=np.arange(start,start+size)
        xyz[idx,0]=component*1000+np.arange(size)
        geo[np.ix_(idx,idx)]=np.abs(np.arange(size)[:,None]-np.arange(size)[None,:])
        start+=size
    return dict(centers=xyz,scores=np.linspace(1,.5,n,dtype=np.float32),geodesic=geo,
        euclidean=np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=-1),max_nodes=128,max_geodesic_ratio=12.)


def test_observer_preserves_original_selection_and_records_zero_component():
    values=inputs()
    original=initializer.connected_coverage_selection
    expected=original(**values)
    recorder=SelectionRecorder({})
    with recorder.installed():
        actual=initializer.connected_coverage_selection(**values)
    np.testing.assert_array_equal(actual,expected)
    trace=recorder.component_trace
    assert [r['selected_count'] for r in trace['components']]==[61,67,0]
    assert [r['visited_in_allocation_loop'] for r in trace['components']]==[True,True,False]
    assert trace['eligible_zero_anchor_components']==[2]
    assert initializer.connected_coverage_selection is original


def test_observer_restores_functions_after_exception():
    original=(initializer.adaptive_connected_coverage_selection,initializer.connected_coverage_selection,initializer.coverage_truncate)
    recorder=SelectionRecorder({})
    with pytest.raises(RuntimeError):
        with recorder.installed():
            raise RuntimeError('Interrupted diagnostic')
    assert (initializer.adaptive_connected_coverage_selection,initializer.connected_coverage_selection,initializer.coverage_truncate)==original


def test_adaptive_wrapper_preserves_result_metadata_and_inputs(monkeypatch):
    values=inputs()
    values.update(radii=np.ones(625,dtype=np.float32),nms_distance=18.,nms_mode='distance',iou_threshold=.1,
        disconnected_multiplier=100.,candidate_k=6,max_euclidean_distance=0.,allow_unreachable_fallback=False,
        bridge_max_geodesic_ratio=0.,bridge_allow_unreachable_fallback=True)
    # Force the connected-coverage branch in this observer test only.
    monkeypatch.setattr(initializer,'selected_graph_pressure',lambda **kwargs:
        {'bridge_edges':8,'reachable_edge_fraction':.9,'components':1})
    original=initializer.adaptive_connected_coverage_selection
    expected,metadata=original(**values)
    pool={k:values[k].copy() for k in ['centers','radii','scores']}
    before={k:v.copy() for k,v in values.items() if isinstance(v,np.ndarray)}
    recorder=SelectionRecorder(pool)
    with recorder.installed():
        actual,actual_metadata=initializer.adaptive_connected_coverage_selection(**values)
    np.testing.assert_array_equal(actual,expected)
    assert actual_metadata==metadata==recorder.adaptive_metadata
    for key,array in before.items():
        np.testing.assert_array_equal(values[key],array)
        np.testing.assert_array_equal(recorder.pool[key],array)
    assert recorder.component_trace['eligible_zero_anchor_components']==[2]


def test_changed_input_pool_rejected_before_selection():
    values=inputs()
    pool={k:values[k].copy() for k in ['centers','scores']}
    pool['centers'][0,0]+=1
    recorder=SelectionRecorder(pool)
    with recorder.installed(),pytest.raises(ValueError,match='Actual input pool differs'):
        initializer.adaptive_connected_coverage_selection(**values,radii=np.ones(625),nms_distance=18.,
            nms_mode='distance',iou_threshold=.1,disconnected_multiplier=100.,candidate_k=6,
            max_euclidean_distance=0.,allow_unreachable_fallback=False,bridge_max_geodesic_ratio=0.,
            bridge_allow_unreachable_fallback=True)
