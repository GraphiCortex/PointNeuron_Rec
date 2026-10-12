from pathlib import Path

import numpy as np
import pytest
import torch

from pointneuron.data.centerline_field import crop_volume, segment_distance_field, weak_labels
from pointneuron.data.swc import SwcNode, SwcTree
from pointneuron.models.centerline_unet import CenterlineUNet, field_loss
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
