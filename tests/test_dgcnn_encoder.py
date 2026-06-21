import importlib.util
import unittest


class TrainProposalCliTests(unittest.TestCase):
    def test_train_proposal_defaults_match_paper_loss_weights(self) -> None:
        from scripts.train_proposal import build_arg_parser

        args = build_arg_parser().parse_args(["--split-file", "split.json"])

        self.assertEqual(args.offset_weight, 1.0)
        self.assertEqual(args.objectness_weight, 10.0)
        self.assertEqual(args.radius_weight, 1.0)
        self.assertIsNone(args.target_radius_floor)
        self.assertEqual(args.objectness_radius_floor, 3.0)
        self.assertEqual(args.radius_target_floor, 1.0)
        self.assertEqual(args.endpoint_loss_weight, 1.0)
        self.assertEqual(args.branch_loss_weight, 1.0)


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
        proposal = SkeletonProposalHead(in_channels=19, hidden_channels=(8,))

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

    def test_paper_skeleton_proposal_loss_runs(self) -> None:
        import torch

        from pointneuron.models.proposal import SkeletonProposalOutput
        from pointneuron.models.proposal_loss import paper_skeleton_proposal_loss

        points = torch.tensor([[[0.0, 0.0, 0.0, 10.0], [10.0, 0.0, 0.0, 10.0]]])
        skeleton_nodes = torch.tensor([[[1.0, 0.5, 0.0, 0.0, 2.0, -1.0]]])
        skeleton_mask = torch.tensor([[True]])
        output = SkeletonProposalOutput(
            offsets=torch.zeros(1, 2, 3),
            objectness_logits=torch.zeros(1, 2, 2),
            radius=torch.ones(1, 2, 1),
            center_proposals=points[..., :3],
            raw=torch.zeros(1, 2, 6),
        )

        loss = paper_skeleton_proposal_loss(output, skeleton_nodes, skeleton_mask, points)

        self.assertEqual(loss.positive_count, 1)
        self.assertTrue(torch.isfinite(loss.total))

    def test_target_radius_floor_relaxes_tiny_swc_radius(self) -> None:
        import torch

        from pointneuron.models.proposal import SkeletonProposalOutput
        from pointneuron.models.proposal_loss import build_skeleton_proposal_targets, paper_skeleton_proposal_loss

        points = torch.tensor([[[0.5, 0.0, 0.0, 10.0], [5.0, 0.0, 0.0, 10.0]]])
        skeleton_nodes = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.01, -1.0]]])
        skeleton_mask = torch.tensor([[True]])
        output = SkeletonProposalOutput(
            offsets=torch.zeros(1, 2, 3),
            objectness_logits=torch.zeros(1, 2, 2),
            radius=torch.ones(1, 2, 1),
            center_proposals=points[..., :3],
            raw=torch.zeros(1, 2, 6),
        )

        strict_loss = paper_skeleton_proposal_loss(output, skeleton_nodes, skeleton_mask, points)
        relaxed_loss = paper_skeleton_proposal_loss(output, skeleton_nodes, skeleton_mask, points, target_radius_floor=1.0)
        targets = build_skeleton_proposal_targets(
            points,
            skeleton_nodes,
            skeleton_mask,
            positive_distance=0.0,
            target_radius_floor=1.0,
        )

        self.assertEqual(strict_loss.positive_count, 0)
        self.assertEqual(relaxed_loss.positive_count, 1)
        self.assertEqual(targets.positive_mask.tolist(), [[True, False]])
        self.assertEqual(targets.matched_radius.tolist(), [[[1.0], [1.0]]])

    def test_objectness_and_radius_floors_can_differ(self) -> None:
        import torch

        from pointneuron.models.proposal import SkeletonProposalOutput
        from pointneuron.models.proposal_loss import build_skeleton_proposal_targets, paper_skeleton_proposal_loss

        points = torch.tensor([[[2.0, 0.0, 0.0, 10.0], [5.0, 0.0, 0.0, 10.0]]])
        skeleton_nodes = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.01, -1.0]]])
        skeleton_mask = torch.tensor([[True]])
        output = SkeletonProposalOutput(
            offsets=torch.zeros(1, 2, 3),
            objectness_logits=torch.zeros(1, 2, 2),
            radius=torch.ones(1, 2, 1),
            center_proposals=points[..., :3],
            raw=torch.zeros(1, 2, 6),
        )

        loss = paper_skeleton_proposal_loss(
            output,
            skeleton_nodes,
            skeleton_mask,
            points,
            objectness_radius_floor=3.0,
            radius_target_floor=1.0,
        )
        targets = build_skeleton_proposal_targets(
            points,
            skeleton_nodes,
            skeleton_mask,
            positive_distance=0.0,
            objectness_radius_floor=3.0,
            radius_target_floor=1.0,
        )

        self.assertEqual(loss.positive_count, 1)
        self.assertEqual(targets.positive_mask.tolist(), [[True, False]])
        self.assertEqual(targets.matched_radius.tolist(), [[[1.0], [1.0]]])

    def test_paper_skeleton_proposal_loss_defaults_match_paper_weights(self) -> None:
        import torch

        from pointneuron.models.proposal import SkeletonProposalOutput
        from pointneuron.models.proposal_loss import paper_skeleton_proposal_loss

        points = torch.tensor([[[0.0, 0.0, 0.0, 10.0], [10.0, 0.0, 0.0, 10.0]]])
        skeleton_nodes = torch.tensor([[[1.0, 0.5, 0.0, 0.0, 2.0, -1.0]]])
        skeleton_mask = torch.tensor([[True]])
        output = SkeletonProposalOutput(
            offsets=torch.zeros(1, 2, 3),
            objectness_logits=torch.zeros(1, 2, 2),
            radius=torch.ones(1, 2, 1),
            center_proposals=points[..., :3],
            raw=torch.zeros(1, 2, 6),
        )

        loss = paper_skeleton_proposal_loss(output, skeleton_nodes, skeleton_mask, points)

        expected = loss.offsets + 10.0 * loss.objectness + loss.radius
        self.assertTrue(torch.allclose(loss.total, expected))

    def test_skeleton_role_weights_emphasize_local_endpoints_and_branches(self) -> None:
        import torch

        from pointneuron.models.proposal_loss import skeleton_role_weights

        skeleton_nodes = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                [2.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [3.0, 2.0, 0.0, 0.0, 1.0, 2.0],
                [4.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            ]
        )

        weights = skeleton_role_weights(skeleton_nodes, endpoint_weight=5.0, branch_weight=2.0)

        self.assertEqual(weights.tolist(), [2.0, 1.0, 5.0, 5.0])

    def test_sphere_iou_detects_overlap(self) -> None:
        import torch

        from scripts.visualize_proposals import sphere_iou

        overlap = sphere_iou(torch.tensor([0.0, 0.0, 0.0]), 2.0, torch.tensor([1.0, 0.0, 0.0]), 2.0)
        separate = sphere_iou(torch.tensor([0.0, 0.0, 0.0]), 2.0, torch.tensor([8.0, 0.0, 0.0]), 2.0)

        self.assertGreater(overlap, 0.0)
        self.assertEqual(separate, 0.0)

    def test_knn_excludes_self_for_distinct_points(self) -> None:
        import torch

        from pointneuron.models.dgcnn import knn

        coords = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]])

        indices = knn(coords, k=1)

        self.assertEqual(indices.tolist(), [[[1], [0], [1]]])


if __name__ == "__main__":
    unittest.main()
