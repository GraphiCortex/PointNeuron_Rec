from pathlib import Path

import numpy as np
import pytest
import torch

from pointneuron.data.centerline_field import crop_volume, segment_distance_field, weak_labels
from pointneuron.data.swc import SwcNode, SwcTree
from pointneuron.models.centerline_unet import CenterlineUNet, field_loss, field_loss_terms
from scripts.pointneuron2.train_centerline_fit import gate, load_records


def tree(rows):
    return SwcTree(Path("synthetic.swc"), tuple(SwcNode(i, 3, x, y, z, 1., p) for i, x, y, z, p in rows))


def test_xyz_crop_preserves_native_voxels_and_marks_padding_unknown():
    image = np.arange(3*4*5).reshape(3, 4, 5)
    cropped, valid, origin = crop_volume(image, [1, 2, 0], 2)
    assert origin.tolist() == [-1, 0, -2]
    assert cropped.shape == (5, 5, 5)
    assert cropped[3, 3, 4] == image[1, 3, 3]
    assert valid[3, 3, 4]
    assert not valid[:2].any()
    assert not valid[:, :, 0].any()
    assert np.count_nonzero(valid) == 3*4*4


def test_fractional_box_has_no_half_voxel_coordinate_shift():
    cropped, _, origin = crop_volume(np.ones((9, 9, 9)), [3.2, 4.5, 3.7], 2)
    assert origin.tolist() == [2, 3, 2]
    assert cropped.shape == (4, 4, 4)


def test_segment_crossing_with_both_endpoints_outside_and_subdivision_invariance():
    a = tree([(1, -10, 2, 3, -1), (2, 20, 2, 3, 1)])
    b = tree([(1, -10, 2, 3, -1), (3, 2.3, 2, 3, 1), (2, 20, 2, 3, 3)])
    da = segment_distance_field(a, [0, 0, 0], (7, 6, 9), 3.)
    db = segment_distance_field(b, [0, 0, 0], (7, 6, 9), 3.)
    np.testing.assert_allclose(da, db, atol=1e-6)
    assert np.all(da[3, 2] == 0)
    assert da[4, 3, 6] == pytest.approx(np.sqrt(2))
    assert np.isinf(da[0, 5, 1])


def test_isolated_root_and_zero_length_edge_have_correct_native_origin():
    t = tree([(1, 12, 23, 34, -1), (2, 12, 23, 34, 1)])
    distance = segment_distance_field(t, [10, 20, 30], (6, 6, 6), 2.)
    assert distance[4, 3, 2] == 0
    assert distance[4, 3, 3] == 1
    assert distance[4, 3, 5] == np.inf


def test_bright_distant_evidence_and_padding_excluded_but_dark_negative_kept():
    distance = np.array([[[0., 2., 4., np.inf, np.inf, 0.]]])
    raw = np.array([[[1., 2., 50., 100., 1., 100.]]])
    valid = np.array([[[True, True, True, True, True, False]]])
    target, known, unknown = weak_labels(raw, valid, distance, 10)
    assert target.tolist() == [[[True, True, False, False, False, False]]]
    assert known.tolist() == [[[True, True, True, False, True, False]]]
    assert unknown.tolist() == [[[False, False, False, True, False, False]]]


def test_no_trace_crop_is_unknown_including_dark_background():
    raw = np.ones((3, 3, 3))
    target, known, _ = weak_labels(raw, raw.astype(bool), np.full(raw.shape, np.inf), 10)
    assert not target.any() and not known.any()


def test_unknown_has_zero_direct_loss_gradient_and_known_classes_receive_gradient():
    logits = torch.tensor([0., 0., 0.], requires_grad=True)
    loss = field_loss(logits, torch.tensor([True, False, False]), torch.tensor([True, True, False]))
    loss.backward()
    assert logits.grad[0] < 0 and logits.grad[1] > 0 and logits.grad[2] == 0
    with pytest.raises(ValueError, match="positive and negative"):
        field_loss(logits, torch.tensor([True, False, False]), torch.zeros(3, dtype=torch.bool))


def test_original_loss_is_preserved_exactly():
    from torch.nn import functional as F
    logits = torch.tensor([-.7, .8, 1.2, -2., 9.], requires_grad=True)
    target = torch.tensor([True, True, False, False, False])
    known = torch.tensor([True, True, True, True, False])
    positive, negative = target & known, ~target & known
    p, y = logits.sigmoid()[known], target[known].float()
    old = .5*(F.softplus(-logits[positive]).mean()+F.softplus(logits[negative]).mean())
    old = old + (1 - (2*(p*y).sum()+1e-6)/(p.sum()+y.sum()+1e-6))
    torch.testing.assert_close(field_loss(logits, target, known), old, atol=0, rtol=0)


def test_topk_targets_hard_known_negatives_without_changing_positive_or_dice():
    logits = torch.tensor([.8, 8., 5., 2., -1., -3., -5., 99.], requires_grad=True)
    target = torch.tensor([True, False, False, False, False, False, False, False])
    known = torch.tensor([True, True, True, True, True, True, True, False])
    a = field_loss_terms(logits, target, known)
    b = field_loss_terms(logits, target, known, negative_aggregation="topk")
    assert a[3] == 6 and b[3] == 3
    torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
    torch.testing.assert_close(a[2], b[2], atol=0, rtol=0)
    assert b[1] > a[1]
    b[1].backward()
    assert (logits.grad[1:4] > 0).all()
    assert torch.count_nonzero(logits.grad[[0, 4, 5, 6, 7]]) == 0
    logits.grad = None
    field_loss(logits, target, known, negative_aggregation="topk").backward()
    assert logits.grad[7] == 0  # Unknown remains excluded from both loss terms.
    assert logits.grad[4] > 0   # Non-mined known negatives still receive Dice gradient.


def test_topk_uses_all_negatives_when_pool_is_smaller_than_three_positives():
    logits = torch.tensor([.5, 1., -.5, 2.])
    target = torch.tensor([True, True, False, False])
    known = torch.ones(4, dtype=torch.bool)
    torch.testing.assert_close(field_loss(logits, target, known, negative_aggregation="topk"),
                               field_loss(logits, target, known), atol=0, rtol=0)


def test_odd_size_single_crop_inference_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = CenterlineUNet(4).eval()
    x = torch.rand(2, 1, 17, 19, 21)
    with torch.no_grad():
        y = m(x)
        single = m(x[:1])
    assert y.shape == x.shape and torch.isfinite(y).all()
    torch.testing.assert_close(y[:1], single, atol=3e-6, rtol=3e-6)
    path = tmp_path / "model.pt"
    torch.save(m.state_dict(), path)
    restored = CenterlineUNet(4).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(x), y, atol=0, rtol=0)


def test_gate_cannot_hide_a_failed_crop_in_an_average():
    cfg = {"gate": {"minimum_crop_reference_coverage": .95, "minimum_crop_reference_proximity": .90,
                    "minimum_mean_known_dice": .80}}
    rows = [{"accepted_voxels": 10, "reference_coverage_6": 1., "reference_proximity_6": 1., "known_voxel_dice": .9} for _ in range(8)]
    assert gate(rows, cfg)
    rows[7] = {**rows[7], "reference_coverage_6": .9}
    assert not gate(rows, cfg)


def test_cache_rejects_reserved_identity_before_loading_files(tmp_path):
    import json
    from pointneuron.data.centerline_field import SCHEMA
    from pointneuron.data.query_cache import file_digest
    from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"partitions": {"train": [{"sample_index": 0, "sample_id": "a"}],
                        "val": [], "test": [{"sample_index": 2, "sample_id": "b"}]}}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "completed", "schema": SCHEMA,
                         "volume_reader_version": VOLUME_READER_VERSION, "partition_sha256": file_digest(partition),
                         "records": [{"partition": "train", "sample_index": 2, "sample_id": "b"}]}))
    with pytest.raises(ValueError, match="training neurons only"):
        load_records(manifest, partition)


def test_synthetic_runner_saves_final_result_and_reloads_checkpoint(tmp_path, monkeypatch):
    """Exercise save/evaluation/finalization without fitting the real experiment."""
    from argparse import Namespace
    import json
    from pointneuron.data.centerline_field import SCHEMA
    from pointneuron.data.query_cache import file_digest
    from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
    import scripts.pointneuron2.train_centerline_fit as runner

    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/pointneuron2/centerline_fit.json").read_text())
    cfg.update(crops=[[i, j] for i in range(4) for j in range(2)], base_channels=4, steps=1, eval_steps=[0, 1])
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"partitions": {"train": [{"sample_index": i, "sample_id": f"synthetic_{i}"} for i in range(4)], "val": [], "test": []}}))
    records = []
    zz, yy, xx = np.indices((7, 9, 11))
    distance = np.sqrt((zz-3.)**2 + (yy-4.)**2).astype(np.float32)
    valid = np.ones(distance.shape, dtype=bool)
    for i, j in cfg["crops"]:
        meta = {"field_schema": SCHEMA, "sample_index": i, "sample_id": f"synthetic_{i}",
                "patch_index": j, "partition": "train", "origin_xyz": [0, 0, 0], "threshold": .2, "image_scale": 1.}
        path = tmp_path / f"sample_{i}_{j}.npz"
        np.savez_compressed(path, image=np.exp(-distance).astype(np.float32), target=distance<=2,
                            known=valid, valid=valid, unknown_bright=~valid, distance=distance,
                            reference_local_xyz=np.array([[x, 4., 3.] for x in range(11)]), metadata=np.array(json.dumps(meta)))
        records.append({"path": path.name, "sha256": file_digest(path), **{k: meta[k] for k in ("sample_index", "sample_id", "patch_index", "partition")}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"status": "completed", "schema": SCHEMA, "config": cfg,
                                   "volume_reader_version": VOLUME_READER_VERSION,
                                   "partition_sha256": file_digest(partition), "records": records}))
    monkeypatch.setattr(runner, "inspection", lambda *args, **kwargs: None)
    output = tmp_path / "run"
    result = runner.run(Namespace(manifest=manifest, partition=partition, output_dir=output, device="cpu", preflight=False))
    assert result["status"] == "completed"
    assert result["checkpoint_reload_max_logit_delta"] == 0
    assert [r["step"] for r in result["evaluations"]] == [0, 1]
    assert all(len(r["crops"]) == 8 for r in result["evaluations"])
    assert len(list(output.glob("field_step_*.pt"))) == 2
    assert len(list(output.glob("step_0001/*.npz"))) == 8
    assert json.loads((output / "fit_report.json").read_text())["status"] == "completed"

    # A tiny paired control exercises actual save/evaluation/order comparisons
    # without launching either real eight-crop fit.
    control_config = tmp_path / "loss_control.json"
    control_config.write_text((Path(__file__).resolve().parents[1] / "configs/pointneuron2/centerline_loss_control.json").read_text())
    control_output = tmp_path / "control"
    controlled = runner.run(Namespace(manifest=manifest, partition=partition, output_dir=control_output,
        device="cpu", preflight=False, baseline_models=output, loss_config=control_config))
    comparison = json.loads((control_output / "loss_comparison.json").read_text())
    assert all(comparison["pairing_checks"].values())
    assert comparison["baseline_crop_order_recorded"] is True
    assert controlled["loss_spec"]["negative_aggregation"] == "topk"
    assert controlled["config"] == result["config"]

    from scripts.pointneuron2.centerline_loss_comparison import check_baseline, compare_final
    fresh = CenterlineUNet(4)
    with pytest.raises(ValueError, match="initialization differs"):
        check_baseline(output, manifest, cfg, torch.device("cpu"), fresh, control_config)
    torch.manual_seed(cfg["seed"])
    fresh = CenterlineUNet(4)
    with pytest.raises(ValueError, match="cache/config/device"):
        check_baseline(output, manifest, {**cfg, "threshold": .4}, torch.device("cpu"), fresh, control_config)
    pairing = check_baseline(output, manifest, cfg, torch.device("cpu"), fresh, control_config)
    controlled["crop_index_order"] = [7 if controlled["crop_index_order"][0] != 7 else 6]
    with pytest.raises(AssertionError, match="crop order"):
        compare_final(controlled, pairing)
