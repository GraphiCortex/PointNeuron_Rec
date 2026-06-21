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

    def test_knn_excludes_self_for_distinct_points(self) -> None:
        import torch

        from pointneuron.models.dgcnn import knn

        coords = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]])

        indices = knn(coords, k=1)

        self.assertEqual(indices.tolist(), [[[1], [0], [1]]])


if __name__ == "__main__":
    unittest.main()
