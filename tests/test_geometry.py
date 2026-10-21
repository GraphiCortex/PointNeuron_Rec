"""Analytic checks for evaluator meaning, numerical behavior and failure cases."""
from types import SimpleNamespace

import numpy as np
import pytest

from pn2.evaluation.geometry import _CurveIndex, geometry_metrics


def morphology(points=(), edges=()):
    return SimpleNamespace(xyz=np.asarray(points, dtype=float).reshape(-1, 3),
                           edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2))


def line(a=(0., 0., 0.), b=(10., 0., 0.)):
    return morphology([a, b], [(0, 1)])


def test_parallel_curves_have_analytic_distance_and_inclusive_tolerance():
    result = geometry_metrics(line(), line((0, 2, 0), (10, 2, 0)), tolerances=(1., 2.))
    assert result["reference_length"] == result["predicted_length"] == 10
    assert result["directed"]["reference_to_prediction"]["mean_distance"] == 2
    assert result["directed"]["prediction_to_reference"]["maximum_distance"] == 2
    assert result["tolerances"]["1"] == dict(precision=0., recall=0., f1=0., extra_length=10., missed_length=10.)
    assert result["tolerances"]["2"] == dict(precision=1., recall=1., f1=1., extra_length=0., missed_length=0.)


def test_distances_use_continuous_segments_not_target_vertices():
    result = geometry_metrics(line(), line((-100, 0, 0), (100, 0, 0)), tolerances=(0.,))
    row = result["tolerances"]["0"]
    assert row["recall"] == 1
    assert row["precision"] == .05
    assert row["extra_length"] == 190
    assert result["directed"]["prediction_to_reference"]["mean_distance"] == 45.25


def test_partial_curve_weights_length_not_node_count():
    predicted = morphology([(5, 0, 0), (5.1, 0, 0), (5.2, 0, 0), (10, 0, 0)], [(0, 1), (1, 2), (2, 3)])
    result = geometry_metrics(line(), predicted, tolerances=(0.,))
    assert result["tolerances"]["0"] == dict(precision=1., recall=.5, f1=2 / 3, extra_length=0., missed_length=5.)
    assert result["directed"]["reference_to_prediction"]["mean_distance"] == 1.25


def test_anisotropic_spacing_and_common_translation_preserve_geometry():
    reference, predicted = line(), line((0, 0, 1), (10, 0, 1))
    result = geometry_metrics(reference, predicted, spacing=(2, 1, 3), tolerances=(2., 3.))
    assert result["reference_length"] == result["predicted_length"] == 20
    assert result["directed"]["reference_to_prediction"]["mean_distance"] == 3
    assert result["tolerances"]["2"]["f1"] == 0
    assert result["tolerances"]["3"]["f1"] == 1
    shift = np.array([10000., -3000., 50.])
    shifted = geometry_metrics(morphology(reference.xyz + shift, reference.edges),
                               morphology(predicted.xyz + shift, predicted.edges),
                               spacing=(2, 1, 3), tolerances=(2., 3.))
    assert result == shifted


def test_arbitrary_collinear_subdivision_does_not_change_score():
    original = morphology([(0, 0, 0), (10, 0, 0), (10, 10, 0)], [(0, 1), (1, 2)])
    subdivided = morphology([(0, 0, 0), (.317, 0, 0), (4.142, 0, 0), (10, 0, 0),
                            (10, 2.131, 0), (10, 10, 0)], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    predicted = line((0, 3, 0), (11, 3, 0))
    first = geometry_metrics(original, predicted, tolerances=(1., 4.))
    second = geometry_metrics(subdivided, predicted, tolerances=(1., 4.))
    assert first["reference_length"] == second["reference_length"]
    assert first["tolerances"] == second["tolerances"]
    assert first["directed"] == second["directed"]
    assert first["structural_support"] == second["structural_support"]


def test_quadrature_converges_towards_analytic_curve_integrals():
    target = line((0, 1, 0), (0, 2, 0))
    boundary_x = 2.34567
    tolerance = np.sqrt(1 + boundary_x ** 2)
    coarse = geometry_metrics(line(), target, tolerances=(tolerance,), maximum_step=.5)
    fine = geometry_metrics(line(), target, tolerances=(tolerance,), maximum_step=.0625)
    analytic_mean = .5 * (10 * np.sqrt(101) + np.arcsinh(10)) / 10
    cd = coarse["directed"]["reference_to_prediction"]["mean_distance"]
    fd = fine["directed"]["reference_to_prediction"]["mean_distance"]
    assert abs(fd - analytic_mean) < abs(cd - analytic_mean)
    cr = next(iter(coarse["tolerances"].values()))["recall"]
    fr = next(iter(fine["tolerances"].values()))["recall"]
    assert abs(fr - boundary_x / 10) < abs(cr - boundary_x / 10)
    assert abs(fr - boundary_x / 10) <= .0625 / 10
    assert fine["numerical"]["maximum_distance_is_sampled"] is True


@pytest.mark.parametrize("reference", [morphology(), morphology([(0, 0, 0)]), line((0, 0, 0), (0, 0, 0))])
def test_reference_without_positive_curve_is_unscorable_not_background(reference):
    result = geometry_metrics(reference, line())
    assert result["status"] == "unscorable_reference"
    assert result["reference_length"] == 0
    assert all(value is None for row in result["tolerances"].values() for value in row.values())


@pytest.mark.parametrize("predicted", [morphology(), morphology([(0, 0, 0), (5, 0, 0), (10, 0, 0)]),
                                       line((0, 0, 0), (0, 0, 0))])
def test_empty_or_isolated_prediction_cannot_fake_supported_reference_curve(predicted):
    result = geometry_metrics(line(), predicted)
    assert result["status"] == "scorable"
    assert result["prediction_has_curve"] is False
    assert result["tolerances"]["2"] == dict(precision=0., recall=0., f1=0., extra_length=0., missed_length=10.)
    assert result["directed"]["reference_to_prediction"]["mean_distance"] is None
    assert result["directed"]["reference_to_prediction"]["distance_status"] == "no_target_curve"


def test_structural_support_is_one_to_one_and_isolated_counts_are_separate():
    predicted = morphology([(0, 0, 0), (.1, 0, 0), (20, 1, 0), (20, 2, 0), (100, 100, 100)], [(0, 2), (1, 3)])
    result = geometry_metrics(line(), predicted, tolerances=(.2,))
    endpoints = result["structural_support"]["endpoints"]["tolerances"]["0.2"]
    assert endpoints == dict(matched=1, precision=.25, recall=.5, f1=1 / 3)
    assert result["isolated"]["predicted"] == 1


def test_perfect_curve_agreement_does_not_establish_correct_connectivity():
    # Two crossing disconnected segments versus a connected four-way junction.
    reference = morphology([(-5, 0, 0), (5, 0, 0), (0, -5, 0), (0, 5, 0)], [(0, 1), (2, 3)])
    prediction = morphology([(-5, 0, 0), (5, 0, 0), (0, -5, 0), (0, 5, 0), (0, 0, 0)],
                            [(0, 4), (1, 4), (2, 4), (3, 4)])
    result = geometry_metrics(reference, prediction)
    assert result["tolerances"]["2"]["f1"] == 1
    assert result["structural_support"]["junctions"]["reference_count"] == 0
    assert result["structural_support"]["junctions"]["predicted_count"] == 1


def test_exact_spatial_index_matches_exhaustive_projection_for_long_and_short_segments():
    random = np.random.default_rng(490)
    segments = random.normal(size=(40, 2, 3)) * 20
    segments = np.concatenate((segments, [[[-1000, 0, 0], [1000, 0, 0]]]))
    points = np.concatenate((random.normal(size=(100, 3)) * 35, [[999.9, .1, 0], [2000, 2000, 2000]]))
    a, delta = segments[:, 0], segments[:, 1] - segments[:, 0]
    vectors = points[:, None, :] - a
    fraction = np.clip(np.einsum("qej,ej->qe", vectors, delta) / np.einsum("ej,ej->e", delta, delta), 0, 1)
    expected = np.linalg.norm(vectors - fraction[..., None] * delta, axis=2).min(axis=1)
    actual = _CurveIndex(segments, fragment_length=8).distances(points, chunk_size=17)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_projection_roundoff_does_not_erase_small_perpendicular_offsets():
    segments = np.array([[[0., 0., 0.], [1e9, 0., 0.]]])
    points = np.array([[5e8 + .1, 1e-8, 0.], [5e8 + .1, 0., 0.]])
    actual = _CurveIndex(segments, fragment_length=1e8).distances(points)
    np.testing.assert_allclose(actual, [1e-8, 0.], rtol=1e-12, atol=1e-20)


@pytest.mark.parametrize("settings", [dict(spacing=(1, 0, 1)), dict(spacing=(1, 1)),
                                       dict(tolerances=()), dict(tolerances=(-1,)),
                                       dict(tolerances=(2, 2)), dict(maximum_step=0),
                                       dict(chunk_size=0), dict(chunk_size=1.5)])
def test_invalid_settings_fail_explicitly(settings):
    with pytest.raises(ValueError):
        geometry_metrics(line(), line(), **settings)
