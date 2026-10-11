import numpy as np

from scripts.pointneuron2.audit_sampling import clipped_reference, reference_support


def test_edge_crossing_patch_is_retained_with_no_local_nodes():
    nodes = np.array([[1, -5, 2, 2, 1, -1], [2, 15, 2, 2, 1, 1]], dtype=float)
    curve, segments, crossings = clipped_reference(nodes, [0, 0, 0], [10, 10, 10])
    np.testing.assert_allclose(curve, np.column_stack((np.arange(11), np.full(11, 2), np.full(11, 2))))
    assert len(segments) == 1 and crossings == 1
    assert reference_support(curve, curve, np.ones(3))['within_1'] == 1


def test_parallel_outside_edge_has_no_false_reference():
    nodes = np.array([[1, -5, 12, 2, 1, -1], [2, 15, 12, 2, 1, 1]], dtype=float)
    curve, segments, crossings = clipped_reference(nodes, [0, 0, 0], [10, 10, 10])
    assert curve.shape == (0, 3) and segments == [] and crossings == 0
    assert reference_support(curve, np.ones((1, 3)), np.ones(3)) is None


def test_fractional_diagonal_clips_all_axes_and_bounds_resampling():
    nodes = np.array([[1, -2, -2, -2, 1, -1], [2, 12, 12, 12, 1, 1]], dtype=float)
    curve, segments, crossings = clipped_reference(nodes, [.5, 1.5, 2.5], [8.5, 9.5, 10.5])
    np.testing.assert_allclose(segments[0], [[2.5, 2.5, 2.5], [8.5, 8.5, 8.5]])
    assert crossings == 1 and np.max(np.linalg.norm(np.diff(curve, axis=0), axis=1)) <= 1
