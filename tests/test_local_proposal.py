import pytest
import torch

from pointneuron.models.proposal import SkeletonProposalHead


@pytest.mark.parametrize('mode', ['local', 'normalized'])
def test_local_head_is_translation_equivariant_with_nonzero_learned_weights(mode):
    torch.manual_seed(7)
    points = torch.randint(0, 96, (2, 32, 4)).float()
    features = torch.randn(2, 32, 16)
    head = SkeletonProposalHead(in_channels=19, hidden_channels=(8,), coordinate_mode=mode).eval()
    shift = torch.tensor([512., -256., 1024.])
    moved = points.clone(); moved[..., :3] += shift
    with torch.no_grad():
        before, after = head(points, features), head(moved, features)
    torch.testing.assert_close(before.objectness_logits, after.objectness_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(before.radius, after.radius, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(before.offsets, after.offsets, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(before.center_proposals + shift, after.center_proposals)


def test_local_head_retains_voxel_offset_and_radius_units():
    points = torch.tensor([[[0., 0., 0., 100.], [96., 0., 0., 200.]]])
    features = torch.zeros(1, 2, 16)
    head = SkeletonProposalHead(in_channels=19, hidden_channels=(), coordinate_mode='local')
    head.initialize_conservative(radius=2.)
    head.mlp[-1].bias.data[3:6] = torch.tensor([1., -2., 3.])
    result = head(points, features)
    torch.testing.assert_close(result.offsets, torch.tensor([[[1., -2., 3.], [1., -2., 3.]]]))
    torch.testing.assert_close(result.radius, torch.full((1, 2, 1), 2.))
