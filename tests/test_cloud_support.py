"""Independent analytic support/gap fixtures; no real references."""
import numpy as np
import pytest

from pn2.evaluation.cloud_support import CloudSupport, branch_paths
from pn2.evaluation.swc import Morphology


def morphology(xyz, parents):
    n = len(xyz)
    return Morphology(None, np.arange(n), np.ones(n, dtype=int), np.array(xyz, dtype=float),
                      np.ones(n), np.array(parents, dtype=np.int64))


def test_straight_curve_support_gap_and_subdivision_invariance():
    sparse = morphology([[0, 0, 0], [10, 0, 0]], [-1, 0])
    dense = morphology([[0, 0, 0], [1.3, 0, 0], [5.7, 0, 0], [10, 0, 0]], [-1, 0, 1, 2])
    cloud = [[0, 0, 0], [10, 0, 0]]
    a = CloudSupport(sparse, .5, (11, 2, 2)).measure(cloud, [2.])
    b = CloudSupport(dense, .5, (11, 2, 2)).measure(cloud, [2.])
    assert a["reference_length"] == 10
    assert a["tolerances"] == b["tolerances"]
    metrics = a["tolerances"]["2.0"]
    assert metrics["curve_support"] == .4
    assert metrics["longest_unsupported_run"] == 6.
    assert metrics["endpoint_support"] == 1 and metrics["junction_support"] is None
    assert metrics["selected_point_reference_agreement"] == 1


def test_branch_local_loss_cannot_hide_in_global_coverage():
    ref = morphology([[10, 10, 10], [0, 10, 10], [20, 10, 10], [10, 20, 10]], [-1, 0, 0, 0])
    assert len(branch_paths(ref)) == 3
    cloud = np.column_stack((np.arange(21), np.full(21, 10), np.full(21, 10)))
    result = CloudSupport(ref, .5, (21, 21, 21)).measure(cloud, [1.])["tolerances"]["1.0"]
    assert result["curve_support"] == .7
    assert result["worst_branch_support"] == .1
    assert result["longest_unsupported_run"] == 9.
    assert result["junction_support"] == 1


def test_outside_reference_is_reported_and_not_dropped():
    ref = morphology([[-5, 0, 0], [5, 0, 0]], [-1, 0])
    result = CloudSupport(ref, .5, (6, 2, 2)).measure([[0, 0, 0], [5, 0, 0]], [1.])
    assert result["outside_image_quadrature_samples"] == 9
    assert result["reference_length"] == 10.
    assert result["tolerances"]["1.0"]["curve_support"] == .3


def test_segmentless_reference_explicit_and_empty_cloud_rejected():
    ref = morphology([[0, 0, 0]], [-1])
    metric = CloudSupport(ref, .5, (2, 2, 2))
    assert metric.measure([[0, 0, 0]], [1.])["status"] == "unscorable_segmentless_reference"
    with pytest.raises(ValueError):
        metric.measure([], [1.])
