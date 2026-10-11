import importlib.util
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pointneuron.data.medial_sampling import (
    MedialSamplingConfig, image_medial_axis, medial_distances,
    medial_probabilities, sample_medial_indices,
)
from pointneuron.data.training_cache import sample_indices_fixed
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler, symmetric_chamfer


def test_probability_mixture_and_extreme_distances():
    config = MedialSamplingConfig(alpha=100, uniform_fraction=0.25)
    probabilities = medial_probabilities(np.array([0., 10., 1e10]), config)
    np.testing.assert_allclose(probabilities, [0.75 + 0.25 / 3, 0.25 / 3, 0.25 / 3])
    np.testing.assert_allclose(medial_probabilities(np.full(3, np.inf), config), np.full(3, 1 / 3))


def test_distance_uses_physical_xyz_spacing():
    distances = medial_distances(np.array([[1, 0, 0], [0, 0, 1]]), np.zeros((1, 3)), (2, 3, 5))
    np.testing.assert_allclose(distances, [2, 5])


@pytest.mark.parametrize("settings", [
    {"alpha": float("nan")}, {"spacing_xyz": [1, 0, 1]},
    {"uniform_fraction": 1.1}, {"max_crop_voxels": 0}, {"dust_threshold": 1.5},
])
def test_invalid_config_rejected(settings):
    with pytest.raises(ValueError):
        MedialSamplingConfig(**settings)


def test_padding_preserves_all_foreground_and_empty_case():
    indices = np.array([4, 6, 8])
    selected = sample_medial_indices(indices, 10, 1, 7, np.random.default_rng(0))
    assert len(selected) == 7
    assert set(selected) == set(indices)
    assert len(sample_medial_indices(np.array([], dtype=np.int64), 10, 1, 7, np.random.default_rng(0))) == 0


def test_weighted_sampling_is_seeded_unique_and_stays_on_image():
    indices = np.arange(100)
    guide = np.array([[0., 0., 0.]])
    with patch("pointneuron.data.medial_sampling.image_medial_axis", return_value=guide):
        first = sample_indices_fixed(indices, 20, np.random.default_rng(2), strategy="medial", width=100, height=1)
        second = sample_indices_fixed(indices, 20, np.random.default_rng(2), strategy="medial", width=100, height=1)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 20
    assert set(first) <= set(indices)
    assert np.median(first) < 25  # strong near-axis preference while retaining exploration


def test_empty_guide_falls_back_but_extraction_errors_propagate():
    with patch("pointneuron.data.medial_sampling.image_medial_axis", return_value=np.empty((0, 3))):
        result = sample_medial_indices(np.arange(50), 50, 1, 10, np.random.default_rng(0))
        assert len(np.unique(result)) == 10
    with patch("pointneuron.data.medial_sampling.image_medial_axis", side_effect=RuntimeError("extraction failed")):
        with pytest.raises(RuntimeError, match="extraction failed"):
            sample_medial_indices(np.arange(50), 50, 1, 10, np.random.default_rng(0))


def test_duplicate_candidates_rejected():
    with pytest.raises(ValueError, match="unique"):
        sample_medial_indices(np.array([1, 1]), 10, 1, 1, np.random.default_rng(0))


@pytest.mark.skipif(importlib.util.find_spec("kimimaro") is None, reason="optional Kimimaro dependency")
def test_kimimaro_axis_frame_anisotropy_thin_branch_and_cap():
    mask = np.zeros((20, 40, 50), dtype=bool)  # ZYX volume storage
    mask[6:11, 14:19, 7:31] = True  # thick horizontal trunk
    mask[8, 16:30, 29] = True  # one-voxel thin branch
    indices = np.flatnonzero(mask.ravel())
    guide = image_medial_axis(indices, 50, 40, MedialSamplingConfig(spacing_xyz=(2, 3, 5)))
    assert len(guide) > 10
    assert np.all((guide.min(axis=0) >= [7, 14, 6]) & (guide.max(axis=0) <= [30, 29, 10]))
    assert np.min(np.linalg.norm(guide - [29, 29, 8], axis=1)) <= 2
    with pytest.raises(ValueError, match="cap"):
        image_medial_axis(indices, 50, 40, MedialSamplingConfig(max_crop_voxels=10))


@pytest.mark.skipif(importlib.util.find_spec("kimimaro") is None, reason="optional Kimimaro dependency")
def test_kimimaro_retains_separate_components_and_handles_dust():
    indices = np.concatenate([np.arange(10), np.arange(40, 50)])
    guide = image_medial_axis(indices, 100, 1, MedialSamplingConfig(dust_threshold=2))
    assert np.any(guide[:, 0] < 10) and np.any(guide[:, 0] >= 40)
    assert len(image_medial_axis(np.array([1]), 100, 1, MedialSamplingConfig(dust_threshold=8))) == 0


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_relaxation_selects_exact_unique_points_and_backpropagates(device):
    torch.manual_seed(4)
    sampler = AdaptiveMedialSampler().to(device)
    points = torch.rand(2, 24, 4, device=device, requires_grad=True)
    guide = torch.rand(2, 8, 3, device=device)
    result = sampler(points, guide, 6)
    expected = points.gather(1, result.indices[..., None].expand(-1, -1, 4))
    torch.testing.assert_close(result.points, expected)
    assert all(len(row.unique()) == 6 for row in result.indices)
    loss = symmetric_chamfer(result.points[..., :3], guide)
    loss.backward()
    assert torch.isfinite(sampler.log_alpha.grad) and sampler.log_alpha.grad.abs() > 0
    assert torch.isfinite(points.grad).all()
    optimizer = torch.optim.Adam(sampler.parameters(), lr=0.01)
    old_alpha = sampler.alpha.detach().clone()
    optimizer.step()
    assert sampler.alpha != old_alpha


def test_relaxation_eval_empty_guide_and_chamfer_contract():
    sampler = AdaptiveMedialSampler().eval()
    points = torch.rand(1, 10, 4)
    guide = torch.rand(1, 3, 3)
    first = sampler(points, guide, 4)
    second = sampler(points, guide, 4)
    torch.testing.assert_close(first.points, second.points)
    fallback = sampler(points, torch.empty(1, 0, 3), 4)
    torch.testing.assert_close(fallback.probabilities, torch.full((1, 10), 0.1))
    assert symmetric_chamfer(guide, guide) == 0
    with pytest.raises(ValueError):
        symmetric_chamfer(guide, torch.empty(1, 0, 3))
    with pytest.raises(ValueError):
        sampler(points, guide, 11)


def test_downstream_proposal_loss_reaches_sampler_without_chamfer():
    from pointneuron.models.dgcnn import DGCNNEncoder
    from pointneuron.models.proposal import SkeletonProposalHead

    torch.manual_seed(9)
    sampler = AdaptiveMedialSampler()
    encoder = DGCNNEncoder(edge_mlp_channels=((8,), (8,)), global_feature_dim=16, k=4)
    proposal = SkeletonProposalHead(in_channels=encoder.output_dim + 3, hidden_channels=(8,))
    selected = sampler(torch.rand(2, 24, 4), torch.rand(2, 5, 3), 12)
    output = proposal(selected.points, encoder(selected.points))
    loss = output.center_proposals.square().mean() + output.objectness_logits.square().mean()
    loss.backward()
    assert torch.isfinite(sampler.log_alpha.grad) and sampler.log_alpha.grad.abs() > 0
    assert torch.isfinite(proposal.mlp[-1].weight.grad).all()
