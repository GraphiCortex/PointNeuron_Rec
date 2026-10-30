"""Target geometry invariants; no model or real dataset required."""
import numpy as np
import pytest

from pn2.evaluation.swc import Morphology
from pn2.supervision.curve_nodes import rectangular_curve_nodes


def tree(xyz, parents, ids=None):
    n = len(xyz)
    return Morphology(None, np.arange(1, n+1) if ids is None else np.asarray(ids),
                      np.ones(n, dtype=np.int64), np.asarray(xyz, dtype=float),
                      np.ones(n), np.asarray(parents, dtype=np.int64))


def test_rectangular_cell_bounds_and_true_intersections():
    result = rectangular_curve_nodes(tree([(-10, 2, 1), (30, 2, 1)], [-1, 0]),
                                     (0, 0, 0), (16, 4, 3))
    assert np.allclose(result.xyz[:, 1:], [2, 1])
    assert sorted(result.xyz[:, 0]) == [-.5, 6., 14., 15.5]
    assert sum("crop_boundary" in roles for roles in result.roles) == 2
    assert result.maximum_reference_to_chord_bound == 0
    assert result.maximum_parametric_deviation_bound == 0


def test_phase_and_geometry_survive_collinear_vertex_subdivision():
    first = tree([(0, 0, 0), (20, 0, 0)], [-1, 0], ids=[1, 2])
    second = tree([(0, 0, 0), (3.13, 0, 0), (20, 0, 0)], [-1, 0, 1], ids=[1, 9, 2])
    a = rectangular_curve_nodes(first, (-1, -1, -1), (30, 2, 2))
    b = rectangular_curve_nodes(second, (-1, -1, -1), (30, 2, 2))
    assert a.identities == b.identities
    assert np.array_equal(a.xyz, b.xyz)
    assert np.array_equal(a.edges, b.edges)
    assert a.roles == b.roles


def test_grid_is_anchored_before_cropping():
    reference = tree([(0, 0, 0), (30, 0, 0)], [-1, 0])
    a = rectangular_curve_nodes(reference, (-1, -1, -1), (20, 2, 2))
    b = rectangular_curve_nodes(reference, (5, -1, -1), (25, 2, 2))
    for result in (a, b):
        grid = result.xyz[["arc_sample" in r for r in result.roles], 0]
        assert 8 in grid and 16 in grid


def test_reference_leaving_and_reentering_has_no_outside_shortcut():
    reference = tree([(0, 1, 0), (10, 1, 0), (10, 10, 0), (0, 10, 0), (0, 1, 0)], [-1, 0, 1, 2, 3])
    result = rectangular_curve_nodes(reference, (-1, -1, -1), (6, 6, 2))
    assert len(result.reference_runs) == 2
    for u, v in result.edges:
        a, b = result.xyz[u], result.xyz[v]
        assert (a[1] == b[1] == 1) or (a[0] == b[0] == 0)


def test_coincident_components_and_zero_length_incidence_are_not_welded():
    reference = tree([(0, 0, 0), (0, 0, 0), (0, 0, 0), (10, 0, 0)], [-1, 0, -1, 2])
    result = rectangular_curve_nodes(reference, (-1, -1, -1), (15, 2, 2))
    for node_id in (1, 2, 3, 4):
        assert ("node", node_id) in result.identities
    index = {key: i for i, key in enumerate(result.identities)}
    assert (index[("node", 1)], index[("node", 2)]) in map(tuple, result.edges)
    assert len({tuple(result.xyz[index[("node", i)]]) for i in (1, 2, 3)}) == 1


def test_junction_and_isolated_node_roles_survive():
    reference = tree([(0, 0, 0), (4, 0, 0), (8, 4, 0), (8, -4, 0), (0, 0, 1)], [-1, 0, 1, 1, -1])
    result = rectangular_curve_nodes(reference, (-1, -5, -1), (12, 6, 3))
    assert "junction" in result.roles[result.identities.index(("node", 2))]
    assert result.roles[result.identities.index(("node", 5))] == {"root", "leaf"}


def test_bend_error_is_measured_instead_of_hidden_by_target_pruning():
    result = rectangular_curve_nodes(tree([(0, 0, 0), (4, 0, 0), (4, 8, 0)], [-1, 0, 1]),
                                     (-1, -1, -1), (10, 10, 2))
    assert result.maximum_reference_to_chord_bound == pytest.approx(np.sqrt(8))
    assert result.maximum_parametric_deviation_bound == pytest.approx(np.sqrt(8))


def test_short_endpoint_gap_is_retained_and_overflow_is_explicit():
    reference = tree([(0, 0, 0), (8.01, 0, 0)], [-1, 0])
    result = rectangular_curve_nodes(reference, (-1, -1, -1), (12, 2, 2))
    assert len(result.xyz) == 3  # No unreviewed canonical endpoint deletion.
    with pytest.raises(ValueError, match="overflow"):
        rectangular_curve_nodes(reference, (-1, -1, -1), (12, 2, 2), capacity=2)


def test_tiny_distinct_structural_span_keeps_both_identities_and_edge():
    reference = tree([(0, 0, 0), (1e-10, 0, 0)], [-1, 0])
    result = rectangular_curve_nodes(reference, (-1, -1, -1), (2, 2, 2))
    assert result.identities == [("node", 1), ("node", 2)]
    assert len(result.edges) == 1
    assert not np.array_equal(result.xyz[0], result.xyz[1])


@pytest.mark.parametrize("start,stop", [((0, 0, 0), (1, 0, 1)), ((0, 0), (1, 2)), ((0, 0, 0), (1, 2, np.nan))])
def test_invalid_domain_is_rejected(start, stop):
    with pytest.raises(ValueError):
        rectangular_curve_nodes(tree([(0, 0, 0)], [-1]), start, stop)
