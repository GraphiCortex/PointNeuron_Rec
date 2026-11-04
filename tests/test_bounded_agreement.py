import numpy as np

from pn2.evaluation.formulation import segment_agreement
from pn2.evaluation.bounded_agreement import BoundedAgreement, bounded_distances
from pn2.evaluation.geometry import _CurveIndex


def test_random_decisions_and_segment_metrics_match_original():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(15, 2, 3))*9
    predicted = rng.normal(size=(19, 2, 3))*13
    index = _CurveIndex(reference, 4.)
    queries = rng.normal(size=(250, 3))*20
    old, new = index.distances(queries), bounded_distances(queries, index, chunk_size=7)
    for t in (2., 6.):
        np.testing.assert_array_equal(old <= t, new <= t)
    assert BoundedAgreement(reference).score(predicted) == segment_agreement(reference, predicted)


def test_exact_tolerance_boundaries_and_censoring():
    s = np.array([[[0., 0, 0], [20, 0, 0]]])
    p = np.array([[3, 2, 0], [3, 6, 0], [3, np.nextafter(6., np.inf), 0], [30, 0, 0]])
    d = bounded_distances(p, _CurveIndex(s, 4.))
    np.testing.assert_array_equal(d[:2], [2., 6.])
    assert np.isinf(d[2:]).all()
    for offset in (0., 2., 6., 6.001, 100.):
        t = s+np.array([0., offset, 0.])
        assert BoundedAgreement(s).score(t) == segment_agreement(s, t)


def test_empty_zero_length_subdivision_and_large_coordinates():
    empty = np.empty((0, 2, 3))
    whole = np.array([[[0., 0, 0], [40, 0, 0]]])+1e6
    split = np.array([[[0., 0, 0], [20, 0, 0]], [[20., 0, 0], [40, 0, 0]]])+1e6
    for a, b in [(empty, whole), (whole, empty), (empty, empty), (whole, split), (whole, np.ones((1, 2, 3)))]:
        assert BoundedAgreement(a).score(b) == segment_agreement(a, b)
