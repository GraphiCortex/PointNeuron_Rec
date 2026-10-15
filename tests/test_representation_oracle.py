"""Falsifiable geometry/topology checks for the reference-only diagnostic."""
from pathlib import Path

import numpy as np
import pytest

from pointneuron.data.swc import SwcNode, SwcTree
from pointneuron.graph.representation_oracle import (
    ReferenceGraph, candidate_report, clipped_reference_graph, export_and_verify,
    distance_to_graph, geometry_report, target_graph, topology, topology_comparison,
)


def tree(rows):
    return SwcTree(Path("synthetic.swc"), tuple(SwcNode(i, 3, x, y, z, 1., p)
                   for i, x, y, z, p in rows))


def test_crossing_segment_without_local_swc_vertices_roundtrips(tmp_path):
    source = tree([(1, -10, 0, 0, -1), (2, 10, 0, 0, 1)])
    boxes = [(np.zeros(3), 2.)]
    reference, target = clipped_reference_graph(source, boxes), target_graph(source, boxes)
    assert len(target.xyz) == 2 and len(target.edges) == 1
    assert topology_comparison(reference, target)["passed"]
    assert geometry_report(reference, target)["tolerances"]["2.0"] == {"recall": 1., "precision": 1.}
    assert export_and_verify(target, tmp_path/"crossing.swc")["passed"]


def test_junction_and_distinct_coincident_forests_survive():
    source = tree([(1, 0, 0, 0, -1), (2, 4, 0, 0, 1),
                   (3, 4, 4, 0, 2), (4, 4, -4, 0, 2), (5, 4, 0, 0, -1)])
    reference, target = clipped_reference_graph(source), target_graph(source)
    assert topology_comparison(reference, target)["passed"]
    assert topology(target)["components"] == 2
    assert topology(target)["junctions"] == 1
    assert len(np.unique(target.xyz, axis=0)) < len(target.xyz)


def test_chain_exits_and_reenters_crop_without_false_shortcut():
    source = tree([(1, -1, 0, 0, -1), (2, 5, 0, 0, 1),
                   (3, 5, 1, 0, 2), (4, -1, 1, 0, 3)])
    boxes = [(np.zeros(3), 2.)]
    reference, target = clipped_reference_graph(source, boxes), target_graph(source, boxes)
    assert topology(reference)["components"] == topology(target)["components"] == 2
    assert topology_comparison(reference, target)["passed"]
    assert geometry_report(reference, target)["tolerances"]["2.0"]["recall"] == 1.


def test_touching_and_overlapping_crops_use_oracle_correspondence_once():
    source = tree([(1, -10, 0, 0, -1), (2, 10, 0, 0, 1)])
    for boxes in [[(np.array([-2., 0, 0]), 2.), (np.array([2., 0, 0]), 2.)],
                  [(np.array([-1., 0, 0]), 3.), (np.array([1., 0, 0]), 3.)]]:
        reference, target = clipped_reference_graph(source, boxes), target_graph(source, boxes)
        assert len(set(target.identities)) == len(target.identities)
        assert topology_comparison(reference, target)["passed"]
        assert topology(target)["components"] == 1


def test_candidate_knn_can_miss_known_edge_but_radius_includes_it():
    xyz = np.array([[0., 0, 0], [8., 0, 0], [0., .1, 0], [8., .1, 0]])
    graph = ReferenceGraph([("node", i) for i in range(4)], xyz, np.ones(4), [(0, 1), (2, 3)])
    result = candidate_report(graph, neighbors=1, radius=8.)
    assert result["knn"]["missing_required_edges"] == 2
    assert result["knn_union_radius"]["missing_required_edges"] == 0
    assert result["knn_union_radius"]["nonrequired_candidates"] > 0


def test_arc_length_metrics_expose_sparsification_and_are_density_independent():
    source = tree([(1, 0, 0, 0, -1), (2, 0, 3, 0, 1), (3, 3, 3, 0, 2)])
    reference, target = clipped_reference_graph(source), target_graph(source, step=8.)
    metric = geometry_report(reference, target, tolerances=(.5, 2.))
    assert metric["tolerances"]["0.5"]["recall"] < .9
    assert metric["tolerances"]["0.5"]["precision"] < .9
    assert metric["reference_length"] == pytest.approx(6.)
    subdivided = tree([(1, 0, 0, 0, -1), (4, 0, 1, 0, 1),
                        (2, 0, 3, 0, 4), (3, 3, 3, 0, 2)])
    dense_metric = geometry_report(clipped_reference_graph(subdivided), target, tolerances=(.5, 2.))
    assert dense_metric["tolerances"] == metric["tolerances"]


def test_rewiring_is_detected_even_when_degrees_and_component_counts_match():
    source = tree([(1, 0, 0, 0, -1), (2, 2, 0, 0, 1), (3, 2, 2, 0, 2),
                   (4, 2, -2, 0, 2), (5, 10, 0, 0, -1), (6, 12, 0, 0, 5)])
    reference, target = clipped_reference_graph(source), target_graph(source, step=20.)
    lookup = {key: i for i, key in enumerate(target.identities)}
    a, b, c, d = [lookup[("node", i)] for i in (2, 3, 5, 6)]
    target.edges = [edge for edge in target.edges if edge not in [(a, b), (c, d)]] + [(a, d), (c, b)]
    assert not topology_comparison(reference, target)["passed"]


def test_empty_output_is_missing_geometry_and_cycle_is_rejected():
    reference = clipped_reference_graph(tree([(1, 0, 0, 0, -1), (2, 8, 0, 0, 1)]))
    empty = ReferenceGraph([], np.empty((0, 3)), np.empty(0), [])
    report = geometry_report(reference, empty)
    assert report["tolerances"]["2.0"]["recall"] == 0.
    assert report["tolerances"]["2.0"]["precision"] is None
    cyclic = ReferenceGraph(reference.identities, reference.xyz, reference.radii, [(0, 1), (1, 0)])
    with pytest.raises(ValueError, match="cycle"):
        topology(cyclic)


def test_metric_uses_segment_interiors_not_nearest_vertices():
    graph = clipped_reference_graph(tree([(1, 0, 0, 0, -1), (2, 5, 0, 0, 1)]))
    points = np.array([[2., 0, 0], [3., 4., 0], [8., 0, 0], [-1., 0, 0]])
    assert distance_to_graph(points, graph) == pytest.approx([0., 4., 3., 1.])


def test_same_component_leaf_swap_fails_incidence_despite_equal_degrees():
    source = tree([(1, 0, 0, 0, -1), (2, 2, 0, 0, 1), (3, 2, 2, 0, 2),
                   (4, 4, 0, 0, 2), (5, 4, 2, 0, 4), (6, 4, -2, 0, 4)])
    reference, target = clipped_reference_graph(source), target_graph(source, step=20.)
    lookup = {key: i for i, key in enumerate(target.identities)}
    a, b, c, d = [lookup[("node", i)] for i in (2, 3, 4, 5)]
    target.edges = [edge for edge in target.edges if edge not in [(a, b), (c, d)]] + [(a, d), (c, b)]
    result = topology_comparison(reference, target)
    assert result["structural_degree_errors"] == result["detected_cross_component_edges"] == 0
    assert result["branch_incidence_errors"] > 0 and not result["passed"]
