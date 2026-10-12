import numpy as np
import pytest

from scripts.pointneuron2.audit_field_centerpaths import nearest_segments, distance_summary


def test_distance_uses_segment_interior_and_zero_length_geometry():
    points = np.array([[5., 2., 0.], [1., 0., 3.], [12., 0., 0.]])
    segments = np.array([[[1., 0., 0.], [9., 0., 0.]]])
    np.testing.assert_allclose(nearest_segments(points, segments), [2., 3., 3.])
    subdivisions = np.array([[[1., 0., 0.], [4., 0., 0.]], [[4., 0., 0.], [9., 0., 0.]]])
    np.testing.assert_allclose(nearest_segments(points, subdivisions), nearest_segments(points, segments))
    np.testing.assert_allclose(nearest_segments([[3., 4., 0.]], [[[0., 0., 0.], [0., 0., 0.]]]), [5.])


def test_summary_weights_by_length_not_number_of_vertices():
    result = distance_summary(np.array([0., 10.]), np.array([9., 1.]))
    assert result["length_weighted_mean_voxels"] == 1.
    assert result["length_weighted_median_voxels"] == 0.
    assert result["length_weighted_p95_voxels"] == 10.
    assert result["within_voxels"]["6"] == pytest.approx(.9)


def test_missing_paths_report_zero_support_and_null_distances():
    distances = nearest_segments([[1., 2., 3.]], np.empty((0, 2, 3)))
    assert np.isinf(distances).all()
    result = distance_summary(distances, np.ones(1))
    assert result["within_voxels"]["6"] == 0.
    assert result["length_weighted_mean_voxels"] is None
    assert result["length_weighted_p95_voxels"] is None
    assert distance_summary(np.empty(0), np.empty(0)) is None
