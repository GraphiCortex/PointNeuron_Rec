"""Check budget starvation, feasibility, identity preservation and restoration."""
import numpy as np
import pytest

from scripts.pointneuron2.remaining_budget_selection import repaired_connected_coverage_selection, installed, initializer
from scripts.pointneuron2.record_graph_components import SelectionRecorder


def inputs(sizes, budget):
    n = sum(sizes)
    xyz = np.zeros((n, 3), dtype=np.float32)
    geo = np.full((n, n), np.inf, dtype=np.float32)
    start = 0
    for number, size in enumerate(sizes):
        idx = np.arange(start, start + size)
        xyz[idx, 0] = number * 1000 + np.arange(size)
        geo[np.ix_(idx, idx)] = np.abs(np.arange(size)[:, None] - np.arange(size)[None, :])
        start += size
    return dict(centers=xyz, scores=np.linspace(1, .5, n, dtype=np.float32), geodesic=geo,
        euclidean=np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1),
        max_nodes=budget, max_geodesic_ratio=12.)


@pytest.mark.parametrize('sizes,budget', [([300,250,75],128), ([250,250,125],128), ([429,44,31,9,6,4,4,3,3,3,3,3,3],128), ([90,3,3,3],4)])
def test_every_eligible_component_receives_an_anchor_when_feasible(sizes, budget):
    values = inputs(sizes, budget)
    before = {k:v.copy() for k,v in values.items() if isinstance(v, np.ndarray)}
    with installed():
        recorder = SelectionRecorder({})
        with recorder.installed():
            selected = initializer.connected_coverage_selection(**values)
    assert len(selected) == len(np.unique(selected)) == budget
    assert recorder.component_trace['eligible_zero_anchor_components'] == []
    for key, old in before.items():
        np.testing.assert_array_equal(values[key], old)
    np.testing.assert_array_equal(selected, repaired_connected_coverage_selection(**values))


def test_infeasible_budget_is_respected_and_size_rule_is_unchanged():
    values = inputs([6,5,4,2,1], 2)
    with installed():
        recorder = SelectionRecorder({})
        with recorder.installed():
            selected = initializer.connected_coverage_selection(**values)
    assert len(selected) == 2
    assert set(selected) <= set(range(15))
    assert [r['candidate_count'] for r in recorder.component_trace['components'] if not r['eligible']] == [2,1]


def test_single_component_matches_original_and_small_pool_keeps_every_row():
    values = inputs([30], 12)
    np.testing.assert_array_equal(repaired_connected_coverage_selection(**values), initializer.connected_coverage_selection(**values))
    values = inputs([2,1], 5)
    np.testing.assert_array_equal(repaired_connected_coverage_selection(**values), np.arange(3))


def test_scope_restores_original_after_exception():
    original = initializer.connected_coverage_selection
    with pytest.raises(RuntimeError), installed():
        assert initializer.connected_coverage_selection is repaired_connected_coverage_selection
        raise RuntimeError('interrupted')
    assert initializer.connected_coverage_selection is original
