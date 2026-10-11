import numpy as np
import pytest

from scripts.pointneuron2.compare_foreground import density_controls


def test_density_control_exposes_coverage_from_filling_the_whole_box():
    xyz = np.indices((8, 8, 8)).reshape(3, -1).T
    curve = np.column_stack((np.arange(8), np.full(8, 4), np.full(8, 4)))
    dense = density_controls(curve, xyz, xyz, [0, 0, 0], [7, 7, 7], 0, budget=8)
    full = dense['sets']['full_guide']
    assert full['reference_support']['within_1'] == 1
    assert full['null_mean_support']['within_1'] == 1
    assert full['support_above_null']['within_1'] == 0
    assert full['point_association']['within_1'] < .2
    thin = density_controls(curve, curve, curve, [0, 0, 0], [7, 7, 7], 0, budget=8)
    assert thin['sets']['full_guide']['support_above_null']['within_1'] > .5


def test_missing_annotation_and_empty_guides_remain_distinct():
    empty = np.empty((0, 3))
    no_reference = density_controls(empty, np.array([[0, 0, 0]]), empty, [0, 0, 0], [1, 1, 1], 2)
    assert no_reference['sets']['full_guide']['reference_support'] is None
    assert no_reference['sets']['full_guide']['null_mean_support'] is None
    no_prediction = density_controls(np.array([[0, 0, 0]]), empty, empty, [0, 0, 0], [1, 1, 1], 2)
    assert no_prediction['sets']['full_guide']['reference_support']['within_6'] == 0
    assert no_prediction['sets']['full_guide']['support_above_null']['within_6'] == 0


def test_null_counts_use_unique_points_not_padding_and_are_reproducible():
    curve = np.array([[2, 2, 2], [3, 3, 3]])
    points = np.repeat(curve, 4, axis=0)
    first = density_controls(curve, points, points, [0, 0, 0], [8, 8, 8], 9, budget=1)
    second = density_controls(curve, points, points, [0, 0, 0], [8, 8, 8], 9, budget=1)
    assert first == second
    assert first['sets']['full_guide']['unique_points'] == 2
    assert first['sets']['candidate_pool']['unique_points'] == 2
    assert first['sets']['budget_guide']['unique_points'] == 1
    with pytest.raises(ValueError):
        density_controls(curve, points, points, [0, 0, 0], [8, 8, 8], 9, repeats=1)
