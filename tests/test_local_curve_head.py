import numpy as np
import pytest

from pn2.supervision.local_curve import ProjectionIndex, distance_bands
from pn2.evaluation.local_curve import consolidate, score_points, summarize, geometric_fps


def test_projection_units_and_broad_support():
    points = np.array([[5., y, 0] for y in [0, 2, 4, 6, 7, 8, 20]])
    labels = ProjectionIndex([[[0, 0, 0], [10, 0, 0]]]).labels(points)
    np.testing.assert_allclose(labels['support'], [1, .75, .5, .25, .125, 0, 0])
    np.testing.assert_array_equal(labels['offset_mask'], [1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_allclose(labels['offset'][:4, 1], [0, -2, -4, -6])
    assert labels['source_distance'][-1] == 9  # declared censoring outside tube


def test_subdivision_does_not_create_ambiguity():
    points = np.array([[x, y, 0] for x in np.linspace(0, 20, 31) for y in [0, 3, 6]])
    full = ProjectionIndex([[[0, 0, 0], [20, 0, 0]]], fragment_length=2).labels(points)
    vertices = np.column_stack((np.arange(21), np.zeros((21, 2))))
    divided = ProjectionIndex(np.stack((vertices[:-1], vertices[1:]), axis=1), fragment_length=.4).labels(points)
    np.testing.assert_allclose(full['offset'], divided['offset'], atol=1e-6)
    np.testing.assert_array_equal(full['offset_mask'], divided['offset_mask'])
    assert not divided['ambiguous'].any()


def test_near_parallel_tie_mask_not_average():
    segments = [[[0, -1, 0], [10, -1, 0]], [[0, 1, 0], [10, 1, 0]]]
    labels = ProjectionIndex(segments).labels(np.array([[5, 0, 0], [5, .9, 0]]))
    assert labels['ambiguous'].tolist() == [True, False]
    assert labels['offset_mask'].tolist() == [False, True]
    assert abs(labels['projection'][0, 1]) == 1


def test_halo_avoids_crop_endpoint_attraction():
    # A box might end at x=10; full segments continue across it.
    p = np.array([[9, 5, 0.]])
    full = ProjectionIndex([[[0, 0, 0], [20, 20, 0]]]).labels(p)
    clipped = ProjectionIndex([[[0, 0, 0], [5, 5, 0]]]).labels(p)
    np.testing.assert_allclose(full['projection'][0], [7, 7, 0])
    assert not np.allclose(full['projection'], clipped['projection'])


def test_coincident_fragments_same_vector_no_topology_assertion():
    seg = [[[0, 0, 0], [10, 0, 0]]]*2
    labels = ProjectionIndex(seg).labels(np.array([[5, 1, 0.]]))
    assert labels['offset_mask'][0]
    np.testing.assert_allclose(labels['offset'][0], [0, -1, 0])


@pytest.mark.parametrize('separation,expected', [(.5, 1), (1.5, 2), (3., 2)])
def test_consolidation_close_branch_limit_is_explicit(separation, expected):
    assert len(consolidate([[0, 0, 0], [0, separation, 0]], [1., .9])) == expected


def test_consolidation_y_crossing_permutation_and_no_average():
    p = np.array([[0., 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0], [.1, 0, 0]])
    scores = np.array([1., .9, .8, .7, .6]); ids = np.array([5, 4, 3, 2, 1])
    kept = consolidate(p, scores, identities=ids)
    assert len(kept) == 4
    assert np.array_equal(p[kept][0], p[0])
    order = np.array([4, 2, 0, 3, 1])
    permuted = consolidate(p[order], scores[order], identities=ids[order])
    np.testing.assert_array_equal(p[kept], p[order][permuted])


def test_unknown_is_not_background_and_macro_is_neuron_based():
    ref = {'reference_runs': [], 'target_xyz': np.empty((0, 3)), 'target_roles': [], 'supervision': 'unknown'}
    result = score_points([[0, 0, 0]], ref)
    assert result == {'points': 1, 'known': False, 'curve_length': 0.}
    assert summarize([{'case_id': 'a', 'metrics': result}])['unknown_points'] == 1


def test_distance_band_boundaries_and_fps():
    np.testing.assert_array_equal(distance_bands([0, 2, 4, 6, 6.1, 8, 9]), [0, 1, 2, 2, 3, 4, 4])
    points = np.column_stack((np.arange(10), np.zeros((10, 2))))
    assert geometric_fps(points, 2).tolist() == [0, 9]


def test_head_offset_gradient_independent_of_support_and_bounded():
    torch = pytest.importorskip('torch')
    from pn2.models.local_curve_head import LocalCurveHead, curve_loss
    head = LocalCurveHead()
    p = torch.zeros(1, 3, 4)
    f = torch.randn(1, 3, 32, requires_grad=True)
    output = head(p, f, torch.zeros(1, 3), 64.)
    assert (output.offsets.norm(dim=-1) < 8).all()
    target = torch.ones(3, 3)
    support = torch.tensor([1., .5, .25]); mask = torch.ones(3, dtype=torch.bool)
    bands = torch.tensor([0, 1, 2])
    _, _, offset_loss = curve_loss(output, support, target, mask, bands)
    gradient = torch.autograd.grad(offset_loss, output.offsets, retain_graph=True)[0]
    with torch.no_grad():
        output.support_logits.fill_(-100)
    _, _, changed = curve_loss(output, support, target, mask, bands)
    torch.testing.assert_close(gradient, torch.autograd.grad(changed, output.offsets)[0], rtol=0, atol=0)
    head.eval()
    a = head(p, f, torch.zeros(1, 3), 64.).centers
    head.train()
    torch.testing.assert_close(a, head(p, f, torch.zeros(1, 3), 64.).centers, rtol=0, atol=0)
