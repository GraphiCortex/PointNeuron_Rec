import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pointneuron.data.curve_targets import curve_geometry, patch_targets, target_arrays
from pointneuron.data.query_cache import CACHE_SCHEMA, load_query_manifest, file_digest
from pointneuron.data.curve_targets import TARGET_SCHEMA
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.data.swc import SwcNode, SwcTree
from pointneuron.models.query_proposal import QueryProposalHead, QueryDecoderConfig, QueryProposalOutput
from pointneuron.models.query_loss import query_set_loss
from pointneuron.models.query_pipeline import (build_models, models_from_checkpoint, comparison_checkpoint,
                                               associate_encoder_features, forward_proposals)
from pointneuron.data.sampling_cache import build_image_pools, SamplingPoolConfig
from pointneuron.data.medial_sampling import MedialSamplingConfig
from scripts.pointneuron2.aggregate_query_proposals import image_inputs, predict_patch
from scripts.pointneuron2.train_query_comparison import apply_update, augment, state_digest, check_pairs, run_arm
from scripts.pointneuron2.audit_query_targets import curve_geometry as audited_geometry, patch_targets as audited_targets


@pytest.fixture(autouse=True, scope="module")
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_config():
    config = json.loads(Path("configs/pointneuron2/query_comparison.json").read_text())
    config["encoder_k"] = 4
    config["decoder"].update(query_count=8, model_dim=16, layers=1, heads=4, feedforward_dim=32, dropout=0.)
    return config


def make_output(centers):
    centers = torch.tensor(centers, dtype=torch.float32, requires_grad=True)
    b, q, _ = centers.shape
    return QueryProposalOutput(centers, torch.zeros(b, q, 2, requires_grad=True),
                               torch.full((b, q, 1), 2., requires_grad=True), centers / 48,
                               torch.zeros(b, q, 4))


def test_production_targets_match_independent_audit_and_preserve_provenance():
    tree = SwcTree(Path("synthetic.swc"), (
        SwcNode(1, 3, -10, 0, 0, 1, -1), SwcNode(2, 3, 10, 0, 0, 3, 1)))
    actual = patch_targets(curve_geometry(tree), [0, 0, 0], 2, 6)
    expected = audited_targets(audited_geometry(tree), [0, 0, 0], 2, 6)
    for a, b in zip(actual, expected):
        assert a["xyz"] == b["xyz"] and a["roles"] == b["roles"]
        assert a["reference_identity"][0] == "edge"
    nodes, roles = target_arrays(actual, 2)
    assert nodes.shape == (2, 6) and np.all((roles & 16) == 16)
    assert target_arrays([], 2)[0].shape == (0, 6)
    with pytest.raises(ValueError, match="overflow"):
        target_arrays(actual, 1)


def test_decoder_translation_and_memory_permutation_invariance():
    torch.manual_seed(2)
    head = QueryProposalHead(QueryDecoderConfig(6, 8, 16, 2, 4, 32, 0.)).eval()
    points, features = torch.rand(2, 12, 4), torch.rand(2, 12, 6)
    centers = torch.rand(2, 3)
    shift = torch.tensor([17., -9., 2.])
    moved = points.clone(); moved[..., :3] += shift
    with torch.no_grad():
        a, b = head(points, features, centers, 48), head(moved, features, centers + shift, 48)
        order = torch.randperm(12)
        permuted = head(points[:, order], features[:, order], centers, 48)
    torch.testing.assert_close(b.center_proposals, a.center_proposals + shift, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(a.objectness_logits, b.objectness_logits)
    torch.testing.assert_close(a.center_proposals, permuted.center_proposals)
    assert a.center_proposals.shape == (2, 8, 3) and torch.all(a.radius > 0)
    assert not hasattr(a, "offsets")


def test_assignment_is_unique_order_independent_and_excludes_unlabeled_gradients():
    output = make_output([[[1, 0, 0], [11, 0, 0], [40, 0, 0]], [[1, 0, 0], [2, 0, 0], [3, 0, 0]]])
    nodes = torch.tensor([[[1., 0, 0, 0, 1, -1], [2, 10, 0, 0, 1, -1]], [[float("nan")] * 6] * 2])
    mask = torch.tensor([[True, True], [False, False]])
    loss = query_set_loss(output, nodes, mask)
    reordered = query_set_loss(output, nodes[:, [1, 0]], mask)
    torch.testing.assert_close(loss.total, reordered.total)
    assert loss.assignments[0] == ((0, 1), (0, 1))
    assert loss.supervised_crops == 1 and loss.matched_count == 2
    loss.total.backward()
    for value in (output.center_proposals, output.objectness_logits, output.radius):
        assert torch.isfinite(value.grad).all() and value.grad[0].abs().sum() > 0
        assert value.grad[1].abs().sum() == 0


def test_unknown_batch_zero_loss_and_capacity_error():
    output = make_output([[[0, 0, 0], [1, 0, 0]]])
    zero = query_set_loss(output, torch.empty(1, 0, 6), torch.empty(1, 0, dtype=torch.bool))
    assert zero.supervised_crops == 0 and float(zero.total.detach()) == 0
    zero.total.backward()
    assert output.objectness_logits.grad.abs().sum() == 0
    with pytest.raises(ValueError, match="overflow"):
        query_set_loss(output, torch.zeros(1, 3, 6), torch.ones(1, 3, dtype=torch.bool))


def test_unlabeled_step_does_not_advance_existing_adam_state_or_weight_decay():
    parameter = torch.nn.Parameter(torch.tensor(2.))
    optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=.5)
    assert apply_update(SimpleNamespace(total=parameter.square(), supervised_crops=1), optimizer, [parameter])
    before, step = parameter.detach().clone(), optimizer.state[parameter]["step"].clone()
    assert not apply_update(SimpleNamespace(total=parameter * 0, supervised_crops=0), optimizer, [parameter])
    torch.testing.assert_close(parameter.detach(), before, atol=0, rtol=0)
    torch.testing.assert_close(optimizer.state[parameter]["step"], step, atol=0, rtol=0)


@pytest.mark.parametrize("arm", ["pointwise", "query"])
def test_checkpoint_roundtrip_native_prediction_and_graph_feature_dimensions(arm):
    config = small_config()
    encoder, head = build_models(config, arm, 0)
    other_encoder, _ = build_models(config, "query" if arm == "pointwise" else "pointwise", 0)
    assert state_digest(encoder) == state_digest(other_encoder)
    encoder.eval(); head.eval()
    points, centers = torch.rand(2, 12, 4), torch.rand(2, 3)
    with torch.no_grad():
        expected, attached = predict_patch(encoder, head, arm, points, centers, 48)
    stream = io.BytesIO()
    torch.save(comparison_checkpoint(encoder, head, config, arm, 0), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    e, h = models_from_checkpoint(checkpoint)
    with torch.no_grad():
        actual = forward_proposals(h, arm, points, e(points), centers, 48)
    for name in ("center_proposals", "objectness_logits", "radius"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    assert attached.features.shape[-1] == 1216
    checkpoint["schema"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported"):
        models_from_checkpoint(checkpoint)


def test_graph_adapter_attaches_to_geometry_not_query_slot_number():
    points = torch.tensor([[[0., 0, 0, 1], [10, 0, 0, 1], [20, 0, 0, 1]]])
    centers = torch.tensor([[[19., 0, 0], [1, 0, 0]]])
    features = torch.tensor([[[1.], [2.], [3.]]])
    result = associate_encoder_features(centers, points, features)
    assert result.features.tolist() == [[[3.], [1.]]]
    assert result.input_indices.tolist() == [[2, 0]]
    assert result.association_distances.tolist() == [[1., 1.]]


def test_image_only_inference_sampling_equals_existing_cached_pool():
    cfg = {"candidate_points": 12, "selected_points": 8, "evaluation_seed": 0}
    with patch("pointneuron.data.sampling_cache.image_medial_axis", return_value=np.empty((0, 3))):
        pool = build_image_pools(np.arange(20), np.arange(20), 20, 1, SamplingPoolConfig(12, None), MedialSamplingConfig(), 2)
    indices = torch.randperm(12, generator=torch.Generator().manual_seed(2))[:8]
    expected = torch.from_numpy(pool["points"])[indices]
    actual = image_inputs(np.arange(20), np.arange(20), 20, 1, {"input_config": cfg}, 2, "cpu")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_augmentation_moves_crop_center_and_targets_with_inputs():
    points = torch.tensor([[[1., 2, 3, 100], [5, 2, 1, 110]]])
    nodes = torch.tensor([[[1., 1, 2, 3, 1, -1]]])
    center = points[:, 0, :3].clone()
    p, n, c = augment(points, nodes, torch.ones(1, 1, dtype=torch.bool), center, 2)
    torch.testing.assert_close(p[:, 0, :3], n[:, 0, 1:4])
    torch.testing.assert_close(p[:, 0, :3], c)
    torch.testing.assert_close(p[..., 3], points[..., 3])


def test_paired_execution_gate_detects_different_inputs():
    one = {"scope": "fixed-training-crop execution check", "manifest_sha256": "a", "partition_sha256": "b",
           "initial_encoder_sha256": "c", "inputs_sha256": "d", "updates": 20}
    check_pairs([one, dict(one)])
    with pytest.raises(ValueError, match="Unmatched"):
        check_pairs([one, {**one, "inputs_sha256": "different"}])


def test_dropout_control_preserves_cache_config_and_weight_initialization():
    from scripts.pointneuron2.train_query_comparison import prepare_run_config
    base = small_config()
    base['decoder']['dropout'] = .1
    control = prepare_run_config(base, decoder_dropout=0., smoke=True)
    assert base['decoder']['dropout'] == .1
    assert control == {**base, 'decoder': {**base['decoder'], 'dropout': 0.}}
    for arm in ('pointwise', 'query'):
        old = build_models(base, arm, 0)
        new = build_models(control, arm, 0)
        for before, after in zip(old, new):
            for key, value in before.state_dict().items():
                torch.testing.assert_close(value, after.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='restricted'):
        prepare_run_config(base, decoder_dropout=0.)
    for invalid in (-.1, 1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite'):
            prepare_run_config(base, decoder_dropout=invalid, smoke=True)


def test_fit_diagnostic_distinguishes_geometry_from_confidence():
    from scripts.pointneuron2.train_query_comparison import inspect_fit, inspect_selection
    output = QueryProposalOutput(torch.zeros(1, 2, 3), torch.tensor([[[1., -1.], [1., -1.]]]),
                                 torch.ones(1, 2, 1), torch.zeros(1, 2, 3), torch.zeros(1, 2, 16))
    nodes = torch.tensor([[[1., 0, 0, 0, 1, -1]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert inspect_fit(output, nodes, mask)['target_coverage_6'] == 1.
    selected = inspect_selection(output, nodes, mask)
    assert selected['accepted_counts'] == [0] and selected['target_coverage_6'] == 0.


def query_cache_fixture(tmp_path):
    config = small_config()
    inputs = {"candidate_points": 12}
    partition = {"partitions": {"train": [{"sample_index": 0, "sample_id": "a"}],
                                "val": [{"sample_index": 1, "sample_id": "b"}],
                                "test": [{"sample_index": 2, "sample_id": "c"}]}}
    pp = tmp_path / "partition.json"; pp.write_text(json.dumps(partition))
    meta = {"sample_index": 0, "sample_id": "a", "partition": "train", "patch_index": 0,
            "target_schema": TARGET_SCHEMA, "target_spacing": config["target_spacing"],
            "target_capacity": config["target_capacity"], "config": inputs}
    record = tmp_path / "record.npz"
    np.savez(record, points=np.ones((12, 4)), target_nodes=np.array([[1, 0, 0, 0, 1, -1]]),
             reference_curve=np.zeros((1, 3)), target_roles=np.array([1], dtype=np.uint8),
             metadata=np.array(json.dumps(meta)))
    manifest = {"schema": CACHE_SCHEMA, "target_schema": TARGET_SCHEMA, "status": "completed",
                "volume_reader_version": VOLUME_READER_VERSION, "partition_sha256": file_digest(pp),
                "config": config, "input_config": inputs,
                "records": [{"path": record.name, "sha256": file_digest(record), "sample_index": 0,
                             "sample_id": "a", "partition": "train", "patch_index": 0, "target_count": 1}]}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    return path, pp, record, manifest


def test_derived_cache_hash_and_reserved_identity_guards(tmp_path):
    path, partition, record, manifest = query_cache_fixture(tmp_path)
    assert len(load_query_manifest(path, partition)["records"]) == 1
    record.write_bytes(record.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_query_manifest(path, partition)
    path, partition, record, manifest = query_cache_fixture(tmp_path)
    manifest["records"][0].update(partition="test", sample_index=2, sample_id="c")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Reserved"):
        load_query_manifest(path, partition)


def test_derived_cache_path_escape_is_rejected_before_access(tmp_path):
    path, partition, record, manifest = query_cache_fixture(tmp_path)
    manifest["records"][0]["path"] = "../outside.npz"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid"):
        load_query_manifest(path, partition)


def test_paired_epoch_loop_uses_same_batches_and_supervised_update_counts(tmp_path):
    config = small_config()
    config.update(selected_points=8, target_capacity=8, epochs=1, snapshot_epochs=[1])
    rows = []
    rng = np.random.default_rng(5)
    for i, role in enumerate(("train", "train", "val", "val")):
        path = tmp_path / f"{role}_{i}.npz"
        nodes = np.empty((0, 6), np.float32) if i == 1 else np.array([[1, 1, 1, 1, 1, -1]], np.float32)
        meta = {"seed": i, "patch_center": [0, 0, 0], "sample_id": f"sample-{i}", "partition": role}
        np.savez(path, points=rng.random((12, 4)).astype(np.float32) * 5, target_nodes=nodes,
                 reference_curve=nodes[:, 1:4], metadata=np.array(json.dumps(meta)))
        rows.append({"path": path.name, "resolved_path": str(path), "partition": role, "target_count": len(nodes)})
    manifest = {"scope": "internal matched target cache", "partition_sha256": "synthetic",
                "input_config": {"selected_points": 8}, "medial_config": {}, "records": rows}
    mp = tmp_path / "manifest.json"; mp.write_text(json.dumps(manifest))
    args = SimpleNamespace(device="cpu", smoke=False, manifest=mp, log_every=1)
    recipe = json.loads(Path(config["pointwise_recipe"]).read_text())["training"]
    reports = [run_arm(arm, args, manifest, config, recipe, tmp_path / arm) for arm in ("pointwise", "query")]
    check_pairs(reports)
    assert all(r["updates"] == 1 and r["epochs"][0]["train"]["supervised_crops"] == 1 for r in reports)
    assert all((tmp_path / arm / "epoch_001.pt").is_file() for arm in ("pointwise", "query"))
