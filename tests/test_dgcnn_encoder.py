import importlib.util
import unittest


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class DGCNNEncoderTests(unittest.TestCase):
    def test_encoder_output_shape(self) -> None:
        import torch

        from pointneuron.models.dgcnn import DGCNNEncoder

        points = torch.rand(2, 32, 4)
        encoder = DGCNNEncoder(edge_mlp_channels=((8,), (8,)), global_feature_dim=16, feature_dim=None, k=4)

        features = encoder(points)

        self.assertEqual(tuple(features.shape), (2, 32, 32))

    def test_encoder_optional_projection_shape(self) -> None:
        import torch

        from pointneuron.models.dgcnn import DGCNNEncoder

        points = torch.rand(2, 32, 4)
        encoder = DGCNNEncoder(edge_mlp_channels=((8,), (8,)), global_feature_dim=16, feature_dim=12, k=4)

        features = encoder(points)

        self.assertEqual(tuple(features.shape), (2, 32, 12))

    def test_proposal_head_shapes(self) -> None:
        import torch

        from pointneuron.models.proposal import SkeletonProposalHead

        points = torch.rand(2, 32, 4)
        features = torch.rand(2, 32, 16)
        proposal = SkeletonProposalHead(in_channels=16, hidden_channels=(8,))

        output = proposal(points, features)

        self.assertEqual(tuple(output.offsets.shape), (2, 32, 3))
        self.assertEqual(tuple(output.objectness_logits.shape), (2, 32, 2))
        self.assertEqual(tuple(output.radius.shape), (2, 32, 1))
        self.assertEqual(tuple(output.center_proposals.shape), (2, 32, 3))

    def test_proposal_targets_mark_near_skeleton_points_positive(self) -> None:
        import torch

        from pointneuron.models.proposal_loss import build_skeleton_proposal_targets

        points = torch.tensor([[[0.0, 0.0, 0.0, 10.0], [10.0, 0.0, 0.0, 10.0]]])
        skeleton_nodes = torch.tensor([[[1.0, 0.5, 0.0, 0.0, 2.0, -1.0]]])
        skeleton_mask = torch.tensor([[True]])

        targets = build_skeleton_proposal_targets(
            points,
            skeleton_nodes,
            skeleton_mask,
            positive_distance=1.0,
            radius_scale=0.0,
        )

        self.assertEqual(targets.objectness_labels.tolist(), [[1, 0]])
        self.assertEqual(targets.positive_mask.tolist(), [[True, False]])
        self.assertEqual(targets.matched_centers.tolist(), [[[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]])

    def test_proposal_loss_runs_without_positive_points(self) -> None:
        import torch

        from pointneuron.models.proposal import SkeletonProposalOutput
        from pointneuron.models.proposal_loss import build_skeleton_proposal_targets, skeleton_proposal_loss

        points = torch.tensor([[[0.0, 0.0, 0.0, 10.0], [10.0, 0.0, 0.0, 10.0]]])
        skeleton_nodes = torch.tensor([[[1.0, 100.0, 0.0, 0.0, 2.0, -1.0]]])
        skeleton_mask = torch.tensor([[True]])
        targets = build_skeleton_proposal_targets(points, skeleton_nodes, skeleton_mask, positive_distance=1.0)
        output = SkeletonProposalOutput(
            offsets=torch.zeros(1, 2, 3),
            objectness_logits=torch.zeros(1, 2, 2),
            radius=torch.ones(1, 2, 1),
            center_proposals=points[..., :3],
            raw=torch.zeros(1, 2, 6),
        )

        loss = skeleton_proposal_loss(output, targets, points)

        self.assertEqual(loss.positive_count, 0)
        self.assertTrue(torch.isfinite(loss.total))

    def test_knn_excludes_self_for_distinct_points(self) -> None:
        import torch

        from pointneuron.models.dgcnn import knn

        coords = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]])

        indices = knn(coords, k=1)

        self.assertEqual(indices.tolist(), [[[1], [0], [1]]])


if __name__ == "__main__":
    unittest.main()
