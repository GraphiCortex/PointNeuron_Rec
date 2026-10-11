"""Geometry checks for the proposed target-count diagnostic, not model scores."""
from pathlib import Path

import numpy as np
import pytest

from scripts.pointneuron2.audit_query_targets import curve_geometry, patch_targets
from pointneuron.data.swc import SwcNode, SwcTree


def tree(rows):
    return SwcTree(Path("synthetic.swc"), tuple(SwcNode(i, 3, x, y, z, r, p)
                   for i, x, y, z, r, p in rows))


def positions(targets):
    return sorted(tuple(np.round(t["xyz"], 7)) for t in targets)


def test_crossing_edge_with_no_local_vertices():
    g = curve_geometry(tree([(1, -10, 0, 0, 1, -1), (2, 10, 0, 0, 3, 1)]))
    targets = patch_targets(g, [0, 0, 0], 2, 6)
    assert positions(targets) == [(-2., 0., 0.), (2., 0., 0.)]
    assert sorted(t["radius"] for t in targets) == pytest.approx([1.8, 2.2])
    assert all("crop_boundary" in t["roles"] for t in targets)


def test_subdivision_does_not_change_density_or_radius():
    a = tree([(1, 0, 0, 0, 1, -1), (2, 12, 0, 0, 3, 1)])
    b = tree([(1, 0, 0, 0, 1, -1), (3, 3, 0, 0, 1.5, 1),
              (4, 7, 0, 0, 13/6, 3), (2, 12, 0, 0, 3, 4)])
    ta, tb = (patch_targets(curve_geometry(t), [6, 0, 0], 20, 4) for t in (a, b))
    assert positions(ta) == positions(tb) == [(0., 0., 0.), (4., 0., 0.), (8., 0., 0.), (12., 0., 0.)]
    assert sorted(t["radius"] for t in ta) == pytest.approx(sorted(t["radius"] for t in tb))


def test_junction_is_preserved_once_with_multiple_chains():
    t = tree([(1, 0, 0, 0, 1, -1), (2, 4, 0, 0, 1, 1),
              (3, 4, 4, 0, 1, 2), (4, 4, -4, 0, 1, 2)])
    targets = patch_targets(curve_geometry(t), [0, 0, 0], 20, 4)
    assert len(targets) == 4
    assert sum("junction" in t["roles"] for t in targets) == 1
    assert sum("leaf" in t["roles"] for t in targets) == 2


def test_phase_is_anchored_before_crop():
    g = curve_geometry(tree([(1, 0, 0, 0, 1, -1), (2, 20, 0, 0, 1, 1)]))
    assert positions(patch_targets(g, [7, 0, 0], 2, 4)) == [(5., 0., 0.), (8., 0., 0.), (9., 0., 0.)]


def test_curve_on_box_face_does_not_force_every_vertex():
    t = tree([(1, -2, 2, 0, 1, -1), (2, -1, 2, 0, 1, 1),
              (3, 0, 2, 0, 1, 2), (4, 2, 2, 0, 1, 3)])
    assert positions(patch_targets(curve_geometry(t), [0, 0, 0], 2, 6)) == [(-2., 2., 0.), (2., 2., 0.)]


def test_isolated_coincident_reference_nodes_remain_distinct():
    t = tree([(1, 0, 0, 0, 1, -1), (2, 0, 0, 0, 2, -1)])
    targets = patch_targets(curve_geometry(t), [0, 0, 0], 2, 4)
    assert len(targets) == 2
    assert all(t["roles"] == {"root", "leaf"} for t in targets)


def test_zero_length_edge_is_supported():
    t = tree([(1, 0, 0, 0, 1, -1), (2, 0, 0, 0, 1, 1), (3, 8, 0, 0, 1, 2)])
    assert positions(patch_targets(curve_geometry(t), [0, 0, 0], 20, 4)) == [(0., 0., 0.), (4., 0., 0.), (8., 0., 0.)]


def test_unrooted_cycle_with_junction_is_rejected():
    t = tree([(1, 0, 0, 0, 1, -1), (2, 1, 0, 0, 1, 3),
              (3, 2, 0, 0, 1, 2), (4, 3, 0, 0, 1, 3)])
    with pytest.raises(ValueError, match="cycle"):
        curve_geometry(t)


def test_empty_crop_and_invalid_step():
    g = curve_geometry(tree([(1, 0, 0, 0, 1, -1)]))
    assert patch_targets(g, [50, 0, 0], 2, 4) == []
    with pytest.raises(ValueError, match="positive"):
        patch_targets(g, [0, 0, 0], 2, 0)
