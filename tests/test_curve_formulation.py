"""Curve/point distinction, geometric controls and image-only baseline behavior."""
from types import SimpleNamespace

import numpy as np
import pytest

from pn2.evaluation.formulation import (curve_samples, runs_graph, graph_segments, support,
    point_diagnostics, segment_agreement, resample_runs, clip_segments)
from pn2.evaluation.classical_skeleton import guide_mask, voxel_graph, prune_terminal_spurs, extract


def line(length=32.):return [np.array([[0., 0, 0], [length, 0, 0]])]


def test_correct_curve_with_sparse_nodes_does_not_get_perfect_node_support():
    runs = line();samples, weights = curve_samples(runs)
    nodes = resample_runs(runs, spacing=8.)[0]
    value, _, _ = support(nodes, samples, weights)
    assert value['fraction']['2.0'] == .5
    result = segment_agreement(graph_segments(runs_graph(runs)), graph_segments(runs_graph([nodes])))
    assert result['tolerances']['2.0']['f1'] == 1.


def test_bends_and_phase_subdivisions_preserve_segment_metrics():
    runs = [np.array([[0., 0, 0], [9, 0, 0], [9, 13, 0]])]
    changed = resample_runs(runs, spacing=8., phase=4., preserve_vertices=True)
    result = segment_agreement(graph_segments(runs_graph(runs)), graph_segments(runs_graph(changed)), tolerances=(.001,))
    assert result['tolerances']['0.001']['f1'] == 1.
    assert not np.array_equal(changed[0], runs[0])


def test_exact_segment_distance_not_nearest_resampled_node_distance():
    samples, weights = curve_samples(line(16.));segments = graph_segments(runs_graph(line(16.)))
    points = np.array([[4., 0, 0], [4., 3, 0], [8., 0, 0]])
    targets = np.array([[0., 0, 0], [8., 0, 0], [16., 0, 0]])
    result = point_diagnostics(points, targets, segments, samples, weights)
    assert result['point_near_curve']['2.0'] == 2
    assert result['point_near_grid_2'] == 1
    assert result['on_curve_beyond_grid_2'] == 1


def test_two_voxel_input_support_is_not_an_offset_prediction_ceiling():
    samples, weights = curve_samples(line(8.))
    seeds = np.column_stack((np.arange(9), np.full(9, 3.), np.zeros(9)))
    value, _, _ = support(seeds, samples, weights)
    assert value['fraction']['2.0'] == 0 and value['fraction']['6.0'] == 1
    shifted = seeds-np.array([0., 3., 0.])
    assert support(shifted, samples, weights)[0]['fraction']['2.0'] == 1


def test_geometry_cannot_distinguish_crossing_from_connected_junction():
    reference = SimpleNamespace(xyz=np.array([[-4., 0, 0], [4, 0, 0], [0, -4, 0], [0, 4, 0]]),
        edges=np.array([[0, 1], [2, 3]]))
    connected = SimpleNamespace(xyz=np.vstack((reference.xyz, [[0., 0, 0]])),
        edges=np.array([[0, 4], [1, 4], [2, 4], [3, 4]]))
    assert segment_agreement(graph_segments(reference), graph_segments(connected))['tolerances']['2.0']['f1'] == 1.


def test_unannotated_geometry_is_not_certified_background():
    result = point_diagnostics(np.ones((3, 3)), np.empty((0, 3)), np.empty((0, 2, 3)), np.empty((0, 3)), np.empty(0))
    assert not result['reference_geometry_present'] and result['point_near_curve'] is None
    assert result['fraction']['2.0'] is None
    metric = segment_agreement(np.empty((0, 2, 3)), np.array([[[0., 0, 0], [1, 0, 0]]]))
    assert metric['tolerances']['2.0']['precision'] is None


def test_empty_prediction_scores_zero_against_a_curve_and_isolated_reference_is_supported():
    metric = segment_agreement(graph_segments(runs_graph(line())), np.empty((0, 2, 3)))
    assert metric['tolerances']['2.0']['f1'] == 0.
    result = point_diagnostics(np.zeros((1, 3)), np.zeros((1, 3)), np.empty((0, 2, 3)), np.empty((0, 3)), np.empty(0))
    assert result['point_near_curve']['2.0'] == 1 and result['curve_length'] == 0


def test_clipping_preserves_cross_boundary_edge_without_snapping():
    segments = np.array([[[-2., 0, 0], [3., 0, 0]], [[0, 3, 0], [1, 3, 0]]])
    result = clip_segments(segments, [-.5, -.5, -.5], [1.5, 1.5, 1.5])
    assert result.shape == (1, 2, 3)
    np.testing.assert_allclose(result[0], [[-.5, 0, 0], [1.5, 0, 0]])


def test_voxel_graph_xyz_and_diagonal_adjacency():
    mask = np.zeros((3, 4, 5), bool);mask[0, 1, 2] = True;mask[1, 2, 3] = True
    graph = voxel_graph(mask)
    np.testing.assert_array_equal(graph.xyz, [[2, 1, 0], [3, 2, 1]])
    np.testing.assert_array_equal(graph.edges, [[0, 1]])
    assert voxel_graph(np.zeros((2, 2, 2), bool)).edges.shape == (0, 2)


def test_pruning_is_one_pass_and_never_deletes_a_standalone_component():
    graph = SimpleNamespace(xyz=np.array([[0., 0, 0], [3, 0, 0], [-3, 0, 0], [0, .5, 0], [10, 0, 0], [10.5, 0, 0]]),
        edges=np.array([[0, 1], [0, 2], [0, 3], [4, 5]]))
    original = graph.xyz.copy();pruned, count = prune_terminal_spurs(graph)
    assert count == 1 and len(pruned.xyz) == 5 and len(pruned.edges) == 3
    assert any(np.array_equal(x, [10.5, 0, 0]) for x in pruned.xyz)
    np.testing.assert_array_equal(graph.xyz, original)


def test_image_only_constant_guide_and_otsu_recipe(tmp_path):
    path = tmp_path/'guide.npy';np.save(path, np.zeros((5, 5, 5), dtype=np.float32))
    raw, pruned, metadata = extract(path)
    assert not len(raw.xyz) and not len(pruned.xyz) and metadata['foreground_voxels'] == 0
    assert metadata['references_opened_for_extraction'] is False
    image = np.zeros((5, 5, 5), dtype=np.float32);image[1:4, 2, 2] = 1.;np.save(path, image)
    mask, metadata = guide_mask(path)
    assert mask.sum() == 3 and metadata['threshold_method'] == 'whole_guide_otsu_including_zeros'


def test_invalid_reference_control_does_not_silently_rephase_targets():
    with pytest.raises(ValueError):resample_runs(line(), phase=9.)
    with pytest.raises(ValueError):resample_runs(line(), spacing=0.)
