import numpy as np
import pytest

from scripts.pointneuron2.audit_field_centerpaths import nearest_segments, distance_summary, measure_paths
from scripts.pointneuron2.audit_centerpath_oracle import oracle_mask


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


def test_shared_path_scorer_measures_continuous_paths_in_both_directions():
    reference = np.array([[[0., 0., 0.], [10., 0., 0.]]])
    result = measure_paths(np.array([[0., 2., 0.], [10., 2., 0.]]), np.array([[0, 1]]), reference, reference)
    assert result["components"] == 1 and result["endpoints"] == 2
    assert result["branch_vertices"] == 0 and result["path_length_voxels"] == 10.
    assert result["reference_length_voxels"] == 10.
    for key in ("reference_to_paths", "paths_to_reference"):
        assert result[key]["length_weighted_mean_voxels"] == 2.
        assert result[key]["within_voxels"]["1"] == 0.
        assert result[key]["within_voxels"]["2"] == 1.


def test_oracle_is_exact_reference_tube_with_padding_excluded_and_no_input_mutation():
    distance = np.array([[[0., 1., 2., 3., np.inf]]])
    valid = np.array([[[False, True, True, True, True]]])
    target = (distance <= 2) & valid
    record = {"distance": distance, "valid": valid, "target": target}
    result = oracle_mask(record, 2)
    assert result.tolist() == [[[False, True, True, False, False]]]
    result[:] = False
    assert target.sum() == 2
    record["target"] = target.copy()
    record["target"][0, 0, 3] = True
    with pytest.raises(ValueError, match="disagree"):
        oracle_mask(record, 2)
