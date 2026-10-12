from pathlib import Path

import numpy as np
import pytest

from pointneuron.data.field_guided_sampling import (
    ARMS, CandidatePools, FieldSamplingConfig, candidate_pools, coverage_indices, select_cloud,
)
from pointneuron.data.swc import SwcNode, SwcTree
from scripts.pointneuron2.field_sampling_metrics import (
    reference_geometry, reference_distances, measure_support, nearest_segments, unsupported_run,
)


def evidence():
    raw = np.zeros((9, 10, 25), dtype=np.uint16)
    valid = np.ones_like(raw, dtype=bool)
    raw[4, 5, :] = np.arange(25)*100
    raw[4, 6, :] = 4095
    valid[4, 5, 0] = False
    probability = np.zeros_like(raw, dtype=np.float32)
    probability[4, 5, :] = .8
    probability[0, 0, 0] = .99  # Valid predicted support can have zero raw intensity.
    probability[4, 5, 0] = .99  # Padding must still be excluded.
    return raw, valid, probability


def test_admission_uses_raw_threshold_and_prediction_only():
    raw, valid, p = evidence()
    pools = candidate_pools(raw, valid, p, 1000)
    assert np.ravel_multi_index((4, 5, 0), raw.shape) not in pools.expanded
    assert 0 in pools.guide and 0 not in pools.foreground
    assert np.ravel_multi_index((4, 5, 10), raw.shape) not in pools.foreground  # Strict > cutoff.
    assert np.ravel_multi_index((4, 5, 11), raw.shape) in pools.foreground
    np.testing.assert_array_equal(pools.expanded, np.union1d(pools.foreground, pools.guide))
    with pytest.raises(ValueError, match="full-precision"):
        candidate_pools(raw, valid, p.astype(np.float16), 1000)


@pytest.mark.parametrize("arm", ARMS)
def test_counts_coordinates_raw_intensity_reproducibility_and_no_mutation(arm):
    raw, valid, p = evidence()
    before = [x.copy() for x in (raw, valid, p)]
    origin = np.array([100, 200, 300])
    a = select_cloud(raw, valid, p, origin, 1000, arm, 20, 5)
    b = select_cloud(raw, valid, p, origin, 1000, arm, 20, 5)
    assert a.actual_count == 20 and len(np.unique(a.voxel_indices)) == 20
    np.testing.assert_array_equal(a.voxel_indices, b.voxel_indices)
    np.testing.assert_array_equal(a.selection_role, b.selection_role)
    z, y, x = np.unravel_index(a.voxel_indices, raw.shape)
    np.testing.assert_array_equal(a.points[:, :3], np.column_stack((x, y, z)) + origin)
    np.testing.assert_array_equal(a.points[:, 3], raw.ravel()[a.voxel_indices])
    np.testing.assert_array_equal(a.field_scores, p.ravel()[a.voxel_indices])
    for actual, previous in zip((raw, valid, p), before):
        np.testing.assert_array_equal(actual, previous)


def test_shortage_is_unique_and_guide_context_are_disjoint():
    raw, valid, p = evidence()
    pools = candidate_pools(raw, valid, p, 1000)
    cloud = select_cloud(raw, valid, p, [0, 0, 0], 1000, "guided_context", 10000, 0)
    assert cloud.actual_count == len(pools.expanded) < cloud.requested_count
    assert len(np.unique(cloud.voxel_indices)) == cloud.actual_count
    assert (cloud.selection_role == 1).sum() == min(cloud.actual_count//2, len(pools.guide))
    assert np.isin(cloud.voxel_indices[cloud.selection_role == 1], pools.guide).all()


def test_no_guide_fallback_is_identical_to_coverage_and_empty_is_explicit():
    raw, valid, p = evidence()
    p[:] = 0
    control = select_cloud(raw, valid, p, [0, 0, 0], 1000, "expanded_coverage", 20, 2)
    guided = select_cloud(raw, valid, p, [0, 0, 0], 1000, "guided_context", 20, 2)
    assert guided.fallback and not guided.selection_role.any()
    np.testing.assert_array_equal(control.voxel_indices, guided.voxel_indices)
    empty = select_cloud(raw*0, valid, p, [0, 0, 0], 1000, "guided_context", 512, 2)
    assert empty.actual_count == 0 and empty.points.shape == (0, 4)


def test_foreground_shortage_cannot_be_hidden_by_repeated_points():
    raw, valid, p = evidence()
    foreground = candidate_pools(raw, valid, p, 1000).foreground
    cloud = select_cloud(raw, valid, p, [0, 0, 0], 1000, "threshold_random", 512, 0)
    assert len(cloud.voxel_indices) == len(foreground)
    np.testing.assert_array_equal(cloud.voxel_indices, foreground)


def test_label_selected_pool_is_rejected_and_labels_are_not_an_argument():
    raw, valid, p = evidence()
    pools = candidate_pools(raw, valid, p, 1000)
    altered = CandidatePools(pools.foreground, pools.guide, pools.expanded[:2], pools.shape_zyx)
    with pytest.raises(ValueError, match="image-only"):
        select_cloud(raw, valid, p, [0, 0, 0], 1000, "expanded_coverage", 2, 0, pools=altered)
    with pytest.raises(TypeError):
        select_cloud(raw, valid, p, [0, 0, 0], 1000, "guided_context", 2, 0, reference_swc="forbidden")


def test_coverage_visits_cells_in_rounds_and_guide_prefers_scores_within_cells():
    shape = (1, 1, 24)
    pool = np.arange(24)
    selected = coverage_indices(pool, shape, 3, 0, 8)
    assert len(np.unique(selected//8)) == 3
    scores = np.arange(24, dtype=float)
    guided = coverage_indices(pool, shape, 3, 0, 8, scores=scores)
    assert set(guided) == {7, 15, 23}
    assert len(np.unique(coverage_indices(pool, shape, 24, 0))) == 24


def swc(rows):
    return SwcTree(Path("synthetic.swc"), tuple(SwcNode(i, 3, x, y, z, 1., parent) for i, x, y, z, parent in rows))


def test_branch_gaps_are_continuous_and_crop_pieces_do_not_bridge_outside_box():
    tree = swc([(1, -2, 1, 1, -1), (2, 4, 1, 1, 1), (3, 12, 1, 1, 2), (4, 4, 8, 1, 2)])
    geometry = reference_geometry(tree, [0, 0, 0], (3, 6, 10))
    assert len(geometry["chains"]) == 3
    assert all(c["boundary_clipped"] for c in geometry["chains"])
    empty = measure_support(np.empty((0, 3)), geometry)
    assert empty["reference_coverage_2"] == 0.
    assert empty["longest_unsupported_run_voxels"]["2"] == pytest.approx(5.)
    assert unsupported_run([False, False, True, False], [1., 2., 1., 4.]) == 4.


def test_reference_support_uses_nearest_points_not_fictional_connecting_edges():
    geometry = reference_geometry(swc([(1, 0, 0, 0, -1), (2, 20, 0, 0, 1)]), [0, 0, 0], (1, 1, 21))
    result = measure_support([[0, 0, 0], [20, 0, 0]], geometry)
    assert result["reference_coverage_2"] == pytest.approx(.2)
    assert result["longest_unsupported_run_voxels"]["2"] == pytest.approx(16.)


def test_exact_segment_distance_pruning_preserves_interior_and_far_geometry():
    rng = np.random.default_rng(0)
    segments = rng.uniform(-200, 200, (100, 2, 3))
    segments = np.concatenate((segments, [[[0., 0., 0.], [100., 0., 0.]], [[2., 2., 2.], [2., 2., 2.]]]))
    points = np.r_[rng.uniform(-20, 20, (200, 3)), [[50., .1, 0.], [2., 2., 2.]]]
    np.testing.assert_allclose(reference_distances(points, segments), nearest_segments(points, segments), atol=1e-10)


def test_no_reference_has_undefined_length_metrics_but_isolated_root_is_measured():
    geometry = reference_geometry(swc([(1, 1, 1, 1, -1)]), [0, 0, 0], (3, 3, 3))
    result = measure_support([[1, 1, 1]], geometry)
    assert result["reference_coverage_2"] is None
    assert result["critical_reference_coverage"]["1"] == 1.
    np.testing.assert_array_equal(reference_distances([[2, 1, 1]], geometry["full_segments"]), [1.])
