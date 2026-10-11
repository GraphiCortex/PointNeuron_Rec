"""Exercise repaired graph worker gates without loading images or tracing paths."""
from pathlib import Path
import json

import numpy as np
import pytest

from scripts.pointneuron2 import compare_component_allocation as comparison
from scripts.pointneuron2.sampling_ablation import write


def worker_fixture(tmp_path, monkeypatch, changed_matrix=False, changed_setting=False):
    n = 625
    centers = np.zeros((n,3), dtype=np.float32)
    geo = np.full((n,n), np.inf, dtype=np.float32)
    start = 0
    for number,size in enumerate([300,250,75]):
        idx = np.arange(start,start+size)
        centers[idx,0] = number*1000 + np.arange(size)
        geo[np.ix_(idx,idx)] = np.abs(np.arange(size)[:,None]-np.arange(size)[None,:])
        start += size
    pool = dict(centers=centers, radii=np.ones(n,dtype=np.float32), scores=np.linspace(1,.5,n,dtype=np.float32),
        geodesic=geo, euclidean=np.linalg.norm(centers[:,None,:]-centers[None,:,:],axis=-1),
        proposal_indices=np.arange(n,dtype=np.int64), selected_pool_indices=np.arange(128))
    meta = dict(nms_distance=18., nms_mode='distance', iou_threshold=.1, max_nodes=128,
        max_geodesic_ratio=12., disconnected_multiplier=100., candidate_k=6, max_euclidean_distance=0.,
        allow_unreachable_fallback=False, bridge_max_geodesic_ratio=0., bridge_allow_unreachable_fallback=True)
    original = {'graph_metadata':meta}
    matrix_path = tmp_path/'matrices.npz'
    np.savez_compressed(matrix_path,**pool)
    trace = {'matrix_artifact':{'path':str(matrix_path)}, 'component_trace':{'excluded_components':[]}}
    monkeypatch.setattr(comparison.initializer,'selected_graph_pressure',lambda **kwargs:
        {'bridge_edges':8,'reachable_edge_fraction':.9,'components':1})
    _, selected, adaptive, recorded = comparison.cached_repair(original,trace)
    selection_path = tmp_path/'selected.npz'
    np.savez_compressed(selection_path,repaired_indices=selected)
    diagnostic_path = tmp_path/'diagnostic.json'
    write(diagnostic_path,{'repaired_component_trace':recorded,'repaired_adaptive_metadata':adaptive})
    graph_path = tmp_path/'remaining_budget/sample_0079/sample_0079_geodesic_graph.npz'
    plan = {'status':'running','cached_selection':{'path':str(selection_path)},
        'cached_diagnostic':{'path':str(diagnostic_path)},'expected_repaired_graph':str(graph_path)}
    write(tmp_path/'comparison.json',plan)
    monkeypatch.setattr(comparison,'verify_inputs',lambda plan:(original,trace,{}))
    monkeypatch.setattr(comparison,'initializer_arguments',lambda *args:['--output',str(graph_path)])
    def fake_main():
        args = {k:pool[k].copy() for k in ['centers','radii','scores','geodesic','euclidean']}
        args.update(meta)
        if changed_matrix:
            args['euclidean'][0,1] += 1
        actual, metadata = comparison.initializer.adaptive_connected_coverage_selection(**args)
        out_meta = dict(meta,adaptive_selection=metadata)
        if changed_setting:
            out_meta['nms_distance'] = 19.
        np.savez_compressed(graph_path,centers=centers[actual],original_indices=pool['proposal_indices'][actual],
            metadata=np.array(json.dumps(out_meta)))
        return 0
    monkeypatch.setattr(comparison.initializer,'main',fake_main)
    return graph_path


def test_worker_requires_prepared_allocation_and_tags_version(tmp_path, monkeypatch):
    graph = worker_fixture(tmp_path,monkeypatch)
    original = comparison.initializer.connected_coverage_selection
    comparison.graph_worker(tmp_path)
    assert comparison.initializer.connected_coverage_selection is original
    with np.load(graph,allow_pickle=False) as z:
        assert json.loads(str(z['metadata']))['component_allocation_version'] == comparison.VERSION
    result = json.loads((tmp_path/'actual_repaired_component_trace.json').read_text())
    assert result['checks'] == 'passed'
    assert result['component_trace']['eligible_zero_anchor_components'] == []


@pytest.mark.parametrize('matrix,setting,match',[(True,False,'Recomputed euclidean differs'),(False,True,'Fixed graph metadata differs')])
def test_worker_rejects_changed_pool_or_settings_and_restores_selector(tmp_path,monkeypatch,matrix,setting,match):
    worker_fixture(tmp_path,monkeypatch,changed_matrix=matrix,changed_setting=setting)
    original = comparison.initializer.connected_coverage_selection
    with pytest.raises(ValueError,match=match):
        comparison.graph_worker(tmp_path)
    assert comparison.initializer.connected_coverage_selection is original
    assert not (tmp_path/'actual_repaired_component_trace.json').exists()
