import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pointneuron.data.medial_sampling import MedialSamplingConfig
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.data.sampling_cache import (SamplingPoolConfig, build_image_pools, SamplingCacheDataset,
                                           collate_sampling_records, load_sampling_manifest)
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler, masked_chamfer
from pointneuron.models.sampling_pipeline import sampler_from_checkpoint, inference_sample
from scripts.pointneuron2.train_sampling import augment_batch, proposal_objective
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead


def test_bounded_pools_keep_image_coordinates_and_all_sparse_candidates():
    with patch("pointneuron.data.sampling_cache.image_medial_axis", return_value=np.arange(60).reshape(20, 3)):
        result = build_image_pools(np.array([2, 4, 6]), np.arange(30), 30, 1, SamplingPoolConfig(8, 5), MedialSamplingConfig(), 4)
    assert result["points"].shape == (8, 4)
    assert set(result["candidate_indices"]) == {2, 4, 6}
    assert len(result["image_guide_xyz"]) == 5
    np.testing.assert_array_equal(result["points"][:, 0], result["candidate_indices"])


def test_masked_guides_empty_fallback_and_padding_have_finite_gradients():
    sampler = AdaptiveMedialSampler()
    points = torch.rand(2, 12, 4)
    guide = torch.rand(2, 6, 3)
    mask = torch.tensor([[True, True, False, False, False, False], [False] * 6])
    result = sampler(points, guide, 4, mask)
    torch.testing.assert_close(result.probabilities[1], torch.full((12,), 1 / 12))
    loss = masked_chamfer(result.points[..., :3], guide, mask)
    loss.backward()
    assert torch.isfinite(sampler.log_alpha.grad)
    sampler.eval()
    one = sampler(points[:1], guide[:1, :2], 4, seed=2)
    padded = sampler(points[:1], guide[:1], 4, mask[:1], seed=2)
    torch.testing.assert_close(one.points, padded.points)


def test_eval_sampling_retains_seeded_exploration_and_checkpoint_roundtrip():
    sampler = AdaptiveMedialSampler()
    with torch.no_grad():
        sampler.log_alpha.add_(0.8)
    cfg = {"initial_alpha": 0.25, "uniform_fraction": 0.25, "temperature": 1., "spacing_xyz": [1., 1., 1.]}
    checkpoint = {"sampler": sampler.state_dict(), "sampling_pipeline": {"schema_version": 1, "strategy": "learned", "sampler_config": cfg}}
    restored = sampler_from_checkpoint(checkpoint)
    points = torch.rand(1, 20, 4); guide = torch.rand(1, 5, 3)
    torch.testing.assert_close(sampler.eval()(points, guide, 8, seed=7).points, restored(points, guide, 8, seed=7).points)
    assert not torch.equal(restored(points, guide, 8, seed=7).indices, restored(points, guide, 8, seed=8).indices)
    with pytest.raises(ValueError, match="weights"):
        sampler_from_checkpoint({"sampling_pipeline": checkpoint["sampling_pipeline"]})


def test_inference_and_training_eval_select_identical_cached_pool():
    guide = np.array([[2, 0, 0], [7, 0, 0]], dtype=np.float32)
    sampler = AdaptiveMedialSampler().eval()
    cfg = {"initial_alpha": .25, "uniform_fraction": .25, "temperature": 1., "spacing_xyz": [1., 1., 1.]}
    checkpoint = {"sampler": sampler.state_dict(), "sampling_pipeline": {
        "schema_version": 1, "volume_reader_version": VOLUME_READER_VERSION,
        "strategy": "learned", "pool_config": {"candidate_points": 12, "guide_points": 8},
        "medial_config": MedialSamplingConfig().to_dict(), "sampler_config": cfg, "selected_points": 4, "evaluation_seed": 3}}
    with patch("pointneuron.data.sampling_cache.image_medial_axis", return_value=guide):
        pools = build_image_pools(np.arange(20), np.arange(20), 20, 1, SamplingPoolConfig(12, 8), MedialSamplingConfig(), 2)
        expected = sampler(torch.tensor(pools["points"])[None], torch.tensor(guide)[None], 4, seed=5).points
        actual = inference_sample(checkpoint, sampler, np.arange(20), np.arange(20), 20, 1, 2)
    np.testing.assert_array_equal(actual, expected[0].detach().numpy())


def test_augmentation_moves_candidates_labels_and_image_guide_together():
    points = torch.tensor([[[1., 2., 3., 100.], [4., 2., 1., 110.]]])
    nodes = torch.tensor([[[1., 1., 2., 3., 1., -1.]]])
    guide = points[:, :1, :3].clone()
    result_points, result_nodes, result_guide = augment_batch(points, nodes, torch.ones(1, 1, dtype=torch.bool),
                                                            guide, torch.ones(1, 1, dtype=torch.bool), torch.ones(3))
    torch.testing.assert_close(result_points[:, :1, :3], result_nodes[:, :, 1:4])
    torch.testing.assert_close(result_guide, result_nodes[:, :, 1:4])
    torch.testing.assert_close(result_points[..., 3], points[..., 3])


def fixture_manifest(tmp_path):
    config = {"candidate_points": 8, "guide_points": 4}
    medial = MedialSamplingConfig().to_dict()
    partition = {"partitions": {"train": [{"sample_index": 0, "sample_id": "neuron-a"}],
                               "val": [{"sample_index": 1, "sample_id": "neuron-b"}],
                               "test": [{"sample_index": 2, "sample_id": "neuron-c"}]}}
    partition_path = tmp_path / "partition.json"; partition_path.write_text(json.dumps(partition))
    metadata = {"sampling_cache_schema": 1, "volume_reader_version": VOLUME_READER_VERSION,
                "sample_index": 0, "sample_id": "neuron-a", "partition": "train",
                "config": config, "medial_config": medial}
    record = tmp_path / "record.npz"
    np.savez(record, points=np.zeros((8, 4)), image_guide_xyz=np.ones((2, 3)), skeleton_nodes=np.zeros((0, 6)),
             edge_index=np.empty((0, 2), dtype=np.int64), metadata=np.array(json.dumps(metadata)))
    manifest = {"volume_reader_version": VOLUME_READER_VERSION, "config": config,
                "medial_config": medial, "partition_sha256": hashlib.sha256(partition_path.read_bytes()).hexdigest(),
                "records": [{"sample_index": 0, "sample_id": "neuron-a", "partition": "train", "path": "record.npz",
                             "sha256": hashlib.sha256(record.read_bytes()).hexdigest()}]}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    return path, partition_path, manifest, record


def test_cache_loading_collation_and_tamper_guards(tmp_path):
    path, partition_path, manifest, record = fixture_manifest(tmp_path)
    payload = load_sampling_manifest(path, partition_path)
    dataset = SamplingCacheDataset([payload["records"][0]["resolved_path"]])
    batch = collate_sampling_records([dataset[0]])
    assert batch["image_guide_mask"].all()
    assert batch["skeleton_mask"].shape == (1, 0)
    manifest["records"][0].update(partition="test", sample_index=2, sample_id="neuron-c")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reserved"):
        load_sampling_manifest(path, partition_path)
    path, partition_path, manifest, record = fixture_manifest(tmp_path)
    record.write_bytes(record.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        load_sampling_manifest(path, partition_path)


def test_unlabeled_patch_does_not_invent_background_supervision():
    recipe = json.loads((Path(__file__).resolve().parents[1] / "configs/pointneuron2/proposal_baseline.json").read_text())["training"]
    encoder = DGCNNEncoder(edge_mlp_channels=((8,),), global_feature_dim=8, k=3)
    proposal = SkeletonProposalHead(in_channels=encoder.output_dim + 3, hidden_channels=(8,))
    points = torch.rand(2, 12, 4, requires_grad=True)
    loss = proposal_objective(proposal(points, encoder(points)), points, torch.empty(2, 0, 6),
                              torch.empty(2, 0, dtype=torch.bool), recipe)
    loss.backward()
    assert loss == 0
    assert proposal.mlp[-1].weight.grad is None
    assert all(parameter.grad is None for parameter in encoder.parameters())
