"""Structural safety, geometry guards and crop consistency of optional removal."""
from pathlib import Path

import numpy as np
import pytest

from pointneuron.data.canonical_curve_targets import CanonicalTargetBuilder
from pointneuron.data.curve_targets import target_arrays
from pointneuron.data.swc import SwcNode, SwcTree
from pointneuron.graph.representation_oracle import candidate_report, geometry_report, clipped_reference_graph


def tree(rows):
    return SwcTree(Path('synthetic.swc'), tuple(SwcNode(i, 3, x, y, z, r, p) for i, x, y, z, r, p in rows))


def straight(length=16.1):
    return tree([(1, 0, 0, 0, 1, -1), (2, length, 0, 0, 1, 1)])


def keys(patch):
    return {tuple(t['reference_identity']) for t in patch.targets}


def test_endpoint_is_preserved_and_optional_sliver_removed():
    source = straight(); builder = CanonicalTargetBuilder(source)
    result = builder.patch([8, 0, 0], 20)
    assert len(builder.removed) == len(result.removed) == 1
    assert sorted(t['xyz'][0] for t in result.targets) == [0, 8, 16.1]
    assert ('node', 2) in keys(result) and result.topology['passed']
    assert candidate_report(result.graph)['knn_union_radius']['required_edge_recall'] == 1
    assert geometry_report(clipped_reference_graph(source), result.graph)['tolerances']['2.0'] == {'precision': 1., 'recall': 1.}


def test_nearby_distinct_branches_and_coincident_structural_nodes_are_preserved():
    source = tree([(1, 0, 0, 0, 1, -1), (2, 16.1, 0, 0, 1, 1),
                   (3, 0, .01, 0, 1, -1), (4, 16.1, .01, 0, 1, 3), (5, 16.1, 0, 0, 1, -1)])
    result = CanonicalTargetBuilder(source).patch([8, 0, 0], 20)
    assert len(result.removed) == 2 and result.topology['target']['components'] == 3
    assert {('node', i) for i in range(1, 6)} <= keys(result)


def test_junction_and_distinct_short_structural_branch_survive():
    source = tree([(1, 0, 0, 0, 1, -1), (2, 16.1, 0, 0, 1, 1),
                   (3, 16.1, .01, 0, 1, 2), (4, 16.1, 16.1, 0, 1, 2)])
    result = CanonicalTargetBuilder(source).patch([8, 8, 0], 30)
    assert {('node', i) for i in range(1, 5)} <= keys(result)
    assert result.topology['target']['junctions'] == 1 and result.topology['branch_incidence_errors'] == 0


def test_euclidean_nearness_without_arclength_nearness_does_not_remove():
    source = tree([(1, 0, 0, 0, 1, -1), (2, 8, 0, 0, 1, 1),
                   (3, 8, 4, 0, 1, 2), (4, 4, 4, 0, 1, 3), (5, 7.9, .1, 0, 1, 4)])
    builder = CanonicalTargetBuilder(source)
    assert not builder.removed
    result = builder.patch([4, 2, 0], 20)
    assert any(np.allclose(t['xyz'], [8, 0, 0]) for t in result.targets)


def test_geometry_guard_keeps_curved_tail():
    source = tree([(1, 0, 0, 0, 1, -1), (2, 16, 0, 0, 1, 1), (3, 16, 1.5, 0, 1, 2)])
    builder = CanonicalTargetBuilder(source)
    assert len(builder.decisions) == 1 and not builder.removed
    assert builder.decisions[0]['maximum_curve_deviation'] > .25


def test_guard_measures_added_error_separately_from_existing_sampling_error():
    source = tree([(1, 0, 0, 0, 1, -1), (2, 8, 0, 0, 1, 1),
                   (3, 8, 4, 0, 1, 2), (4, 12, 4, 0, 1, 3), (5, 12, 4.1, 0, 1, 4)])
    builder = CanonicalTargetBuilder(source)
    assert len(builder.removed) == 1
    decision = builder.decisions[0]
    assert decision['absolute_source_curve_deviation'] > .25
    assert decision['maximum_curve_deviation'] < .25
    result = builder.patch([8, 2, 0], 20)
    assert len(result.removed) == 1 and result.topology['passed']


def test_radius_guard_keeps_strong_taper():
    source = tree([(1, 0, 0, 0, 1, -1), (2, 16, 0, 0, 3, 1), (3, 16.1, 0, 0, 1, 2)])
    builder = CanonicalTargetBuilder(source)
    assert not builder.removed and builder.decisions[0]['maximum_radius_deviation'] > .25


def test_optional_grid_decisions_are_shared_across_overlapping_crops():
    builder = CanonicalTargetBuilder(straight())
    a, b = builder.patch([8, 0, 0], 20), builder.patch([12, 0, 0], 6)
    # Shared original arc samples have identical keep/drop decisions and coordinates.
    assert len(a.removed) == len(b.removed) == 1
    common = keys(a) & keys(b)
    assert ('node', 2) in common
    for key in common:
        ta = next(t for t in a.targets if tuple(t['reference_identity']) == key)
        tb = next(t for t in b.targets if tuple(t['reference_identity']) == key)
        assert ta['xyz'] == tb['xyz'] and ta['radius'] == tb['radius']


def test_forced_crop_boundary_overrides_optional_removal():
    builder = CanonicalTargetBuilder(straight())
    result = builder.patch([12, 0, 0], 4)
    assert not result.removed
    assert sorted(t['xyz'][0] for t in result.targets) == [8., 16.]
    assert all('crop_boundary' in t['roles'] for t in result.targets)


def test_crop_exit_and_reentry_remain_disconnected():
    source = tree([(1, -1, 0, 0, 1, -1), (2, 5, 0, 0, 1, 1),
                   (3, 5, 1, 0, 1, 2), (4, -1, 1, 0, 1, 3)])
    result = CanonicalTargetBuilder(source).patch([0, 0, 0], 2)
    assert result.topology['target']['components'] == 2


def test_empty_crop_stays_empty_and_capacity_never_truncates():
    builder = CanonicalTargetBuilder(straight())
    assert not builder.patch([100, 0, 0], 2).targets
    with pytest.raises(ValueError, match='overflow'):
        target_arrays(builder.patch([8, 0, 0], 20).targets, 2)


def test_subdivision_does_not_change_sample_positions_or_radius():
    a = straight()
    b = tree([(1, 0, 0, 0, 1, -1), (3, 4, 0, 0, 1, 1), (2, 16.1, 0, 0, 1, 3)])
    patches = [CanonicalTargetBuilder(t).patch([8, 0, 0], 20) for t in (a, b)]
    assert [t['xyz'] for t in patches[0].targets] == [t['xyz'] for t in patches[1].targets]


def test_zero_length_swc_edges_and_short_structural_chains():
    source = tree([(1, 0, 0, 0, 1, -1), (2, 0, 0, 0, 1, 1), (3, 16.1, 0, 0, 1, 2)])
    result = CanonicalTargetBuilder(source).patch([8, 0, 0], 20)
    assert len(result.removed) == 1 and result.topology['passed']
    tiny = CanonicalTargetBuilder(straight(.01)).patch([0, 0, 0], 2)
    assert len(tiny.targets) == 2 and not tiny.removed


@pytest.mark.parametrize('options', [{'step': 0}, {'endpoint_fraction': .5}, {'maximum_deviation': -1}, {'maximum_radius_deviation': np.nan}])
def test_invalid_policy_rejected(options):
    with pytest.raises(ValueError):
        CanonicalTargetBuilder(straight(), **options)
