import numpy as np
import pytest
import importlib.util
from pathlib import Path

from pn2.evaluation.branch_evidence import arc_windows, measure, profile_window, source_priority_order


def tube(sigma, amplitude):
    z, y, x = np.indices((40, 40, 40))
    return 100.+amplitude*np.exp(-((y-20.)**2+(z-20.)**2)/(2*sigma**2))


@pytest.mark.parametrize("sigma", [.5, 2.])
@pytest.mark.parametrize("amplitude", [5., 50.])
def test_known_width_contrast_and_full_or_empty_retention(sigma, amplitude):
    image = tube(sigma, amplitude)
    window = profile_window(image, [20, 20, 20], [1, 0, 0], [0, 0, 0], [40, 40, 40])
    row = window["metadata"]
    assert row["profile_status"] == "resolved_apparent_profile"
    assert abs(row["apparent_width_voxel"]-2*np.sqrt(2*np.log(2))*sigma) < 1.
    assert row["inner_peak_contrast"] == pytest.approx(amplitude, rel=.001)
    assert all(r["retained_mass_fraction"] == 1 for r in measure(window, np.ones_like(image, bool)))
    assert all(r["retained_mass_fraction"] == 0 for r in measure(window, np.zeros_like(image, bool)))


def test_bright_evidence_outside_two_voxels_is_counted_at_four():
    image = tube(2., 50.)
    window = profile_window(image, [20, 20, 20], [1, 0, 0], [0, 0, 0], [40, 40, 40])
    selected = np.zeros_like(image, bool); selected[20, 23, 20] = True
    result = {r["radius"]: r for r in measure(window, selected)}
    assert result[2.]["retained_mass_fraction"] == 0
    assert result[4.]["retained_mass_fraction"] > 0


def test_constant_image_is_undefined_not_background_supervision():
    image = np.full((40, 40, 40), 17.)
    window = profile_window(image, [20, 20, 20], [1, 0, 0], [0, 0, 0], [40, 40, 40])
    assert window["metadata"]["apparent_width_voxel"] is None
    assert all(r["retained_mass_fraction"] is None for r in measure(window, np.ones_like(image, bool)))


def test_box_clipping_changes_denominator_without_padding_or_double_counting():
    image = tube(2., 50.)
    window = profile_window(image, [20, 20, 20], [1, 0, 0], [20, 0, 0], [40, 40, 40])
    for domain in window["domains"]:
        assert len(domain["ids"]) == len(np.unique(domain["ids"]))
    result = measure(window, np.ones((40, 40, 20), bool))
    assert all(r["input_box_fraction_of_source_window"] < 1 for r in result)
    assert all(r["retained_mass_fraction"] == 1 for r in result)


def test_hash_order_is_independent_of_selection_rank_and_subset():
    ids = np.array([200, 3, 7, 40, 8, 91])
    ordered = ids[source_priority_order(ids)]
    assert np.array_equal(ordered, ids[::-1][source_priority_order(ids[::-1])])
    subset = ids[[0, 2, 4]]
    assert np.array_equal(subset[source_priority_order(subset)], ordered[np.isin(ordered, subset)])


def test_arc_sampling_preserves_length_and_handles_zero_edges():
    samples = arc_windows([[0, 0, 0], [0, 0, 0], [10, 0, 0]], step=4)
    assert len(samples) == 3
    assert sum(s["weight"] for s in samples) == pytest.approx(10)
    assert all(np.array_equal(s["tangent"], [1, 0, 0]) for s in samples)


def test_independent_point_attribute_counts_match_voxel_window_measurement():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_real_branch_counts.py"
    spec = importlib.util.spec_from_file_location("point_count_validator", path)
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    image = tube(2., 50.)
    w = profile_window(image, [20, 20, 20], [1, 0, 0], [0, 0, 0], [40, 40, 40])
    record = dict(w["metadata"], domains=[{"available_positive_contrast_mass": float(d["mass"].sum())} for d in w["domains"]])
    points = np.array([[20, 20, 20], [20, 23, 20], [24, 20, 20], [22, 20, 21]])
    mask = np.zeros_like(image, bool); mask[tuple(points[:, ::-1].T)] = True
    intensity = image[tuple(points[:, ::-1].T)]
    expected = measure(w, mask)
    actual = validator.point_stats(points, intensity, [record], [1., 2., 4., 8.], 2.)[0]
    for a, b in zip(actual, expected):
        assert a["selected_points"] == b["selected_voxel_count"]
        assert a["retained_positive_contrast_mass"] == pytest.approx(b["retained_positive_contrast_mass"])
        assert a["retained_mass_fraction"] == pytest.approx(b["retained_mass_fraction"])


def test_independent_count_validator_accepts_empty_access_path():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_real_branch_counts.py"
    spec = importlib.util.spec_from_file_location("empty_count_validator", path)
    validator = importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    w = {"xyz": [0., 0., 0.], "tangent": [1., 0., 0.], "background": 0.,
         "domains": [{"available_positive_contrast_mass": 10.}]}
    row = validator.point_stats(np.empty((0, 3)), np.empty(0), [w], [4.], 2.)[0][0]
    assert row["selected_points"] == row["points_above_background"] == 0
    assert row["retained_mass_fraction"] == 0


def test_noise_only_image_has_positive_mass_without_foreground_truth():
    image = 100.+np.random.default_rng(0).normal(size=(40, 40, 40))
    w = profile_window(image, [20, 20, 20], [1, 0, 0], [0, 0, 0], [40, 40, 40])
    assert w["metadata"]["background_MAD"] > 0
    # Deliberate limitation: positive contrast mass does not certify a neurite.
    assert w["domains"][-1]["mass"].sum() > 0


def test_stronger_neighboring_tube_marks_width_unresolved():
    z, y, x = np.indices((40, 40, 40))
    image = tube(.5, 10.)+500*np.exp(-((y-25.)**2+(z-20.)**2)/.5)
    w = profile_window(image, [20, 20, 20], [1, 0, 0], [0, 0, 0], [40, 40, 40])
    assert w["metadata"]["profile_status"] == "outer_profile_exceeds_inner_peak"
    assert w["metadata"]["apparent_width_voxel"] is None
