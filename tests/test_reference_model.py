"""Retained model interface checks on CPU; no training steps or real data."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pn2.reference.model import fresh_reference, predict_points, state_digest
from pn2.reference.forward_audit import audit_reference_forward


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cloud(batch=2, count=17):
    generator = torch.Generator().manual_seed(54)
    points = torch.rand(batch, count, 4, generator=generator)
    points[..., :3] = points[..., :3] * 37
    points[..., 3] = points[..., 3] * 4095
    return points


def test_nonzero_head_feature_flow_eval_batching_and_state_preservation():
    model = fresh_reference(2, conservative=False)
    # Nondefault buffers ensure this tests ordinary running-statistics inference.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean.copy_(torch.linspace(-.2, .2, len(module.running_mean)))
                module.running_var.fill_(1.3)
    points = cloud()
    before = state_digest(model)
    both = predict_points(model, points)
    assert both.center_proposals.shape == (2, 17, 3)
    assert both.objectness_logits.shape == (2, 17, 2)
    assert both.offsets.abs().max() > 1e-5
    assert model.encoder(points).std() > 0
    for b in range(2):
        one = predict_points(model, points[b:b + 1])
        for field in ("center_proposals", "radius", "objectness_logits"):
            torch.testing.assert_close(getattr(one, field)[0], getattr(both, field)[b], atol=2e-5, rtol=1e-5)
    assert state_digest(model) == before


def test_local_head_translation_point_order_and_exact_state_reload():
    model = fresh_reference(4, conservative=False)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean.fill_(.13)
                module.running_var.fill_(1.7)
    points = cloud(batch=1)
    original = predict_points(model, points)
    shift = torch.tensor([512., -256., 64.])
    shifted = points.clone(); shifted[..., :3] += shift
    translated = predict_points(model, shifted)
    torch.testing.assert_close(translated.center_proposals - shift, original.center_proposals, atol=6e-5, rtol=1e-5)
    torch.testing.assert_close(translated.objectness_logits, original.objectness_logits, atol=1e-5, rtol=1e-5)
    permutation = torch.randperm(17, generator=torch.Generator().manual_seed(12))
    shuffled = predict_points(model, points[:, permutation])
    torch.testing.assert_close(shuffled.objectness_logits, original.objectness_logits[:, permutation], atol=2e-5, rtol=1e-5)
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    restored = fresh_reference(99, conservative=False)
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True), strict=True)
    replay = predict_points(restored, points)
    assert state_digest(restored) == state_digest(model)
    torch.testing.assert_close(replay.raw, original.raw, rtol=0, atol=0)


@pytest.mark.parametrize("count", [2, 3, 20, 32])
def test_short_distinct_clouds_run_without_padding(count):
    model = fresh_reference(0)
    points = cloud(batch=1, count=count)
    output = predict_points(model, points)
    torch.testing.assert_close(output.center_proposals, points[..., :3], atol=0, rtol=0)
    torch.testing.assert_close(output.radius, torch.ones(1, count, 1))
    assert output.objectness_logits.shape == (1, count, 2)


def test_prediction_refuses_training_normalization_and_singletons():
    model = fresh_reference(0)
    with pytest.raises(ValueError, match="two input"):
        predict_points(model, cloud(batch=1, count=1))
    model.encoder.train()
    with pytest.raises(ValueError, match="explicit eval"):
        predict_points(model, cloud(batch=1))


def manifests(tmp_path):
    # One normal cloud, one empty mask, one singleton, one missing image.
    for name, values in [("ready", np.full((1, 2, 4), 100, np.uint8)),
                         ("empty", np.zeros((1, 2, 4), np.uint8)),
                         ("single", np.array([100, 0, 0, 0, 0, 0, 0, 0], np.uint8).reshape(1, 2, 4))]:
        head = b"raw_image_stack_by_hpengL" + (1).to_bytes(2, "little")
        head += b"".join(d.to_bytes(4, "little") for d in (4, 2, 1, 1))
        (tmp_path / f"{name}.v3draw").write_bytes(head + values.tobytes())
    cases = {"schema_version": 1, "cases": [{"case_id": name, "partition": "synthetic",
        "image_path": f"{name}.v3draw", "coordinates": {"mode": "native", "units": "voxel"}}
        for name in ("ready", "empty", "single", "missing")]}
    config = {"schema_version": 1, "tile_size_xyz": [97, 97, 97], "stride_xyz": [48, 48, 48]}
    clouds = {"schema_version": 1, "purpose": "repaired_pn1_input_control", "foreground": {
        "mode": "image_max_fraction", "fraction": .2}, "max_input_points": 512, "seed": 0}
    paths = []
    for name, data in [("cases", cases), ("tiling", config), ("clouds", clouds)]:
        path = tmp_path / f"{name}.json"; path.write_text(json.dumps(data), encoding="utf-8"); paths.append(path)
    return paths


def test_all_tiles_accounted_distinct_packets_failure_statuses_and_no_reference_reads(tmp_path, monkeypatch):
    cases, tiling, clouds = manifests(tmp_path)
    original = Path.read_bytes
    def guarded(path):
        assert path.suffix != ".swc"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    report = audit_reference_forward(cases, tiling, clouds, tmp_path / "out")
    assert report["status"] == "failed"
    assert report["model"]["trained"] is False
    assert report["model"]["state_unchanged"] is True
    assert len(report["cases"]) == 4
    ready, empty, singleton, missing = report["cases"]
    assert ready["status"] == "completed" and ready["model_processed_tiles"] == 1
    assert empty["status"] == "completed" and empty["tile_status_counts"] == {"empty_input": 1}
    assert singleton["status"] == "unsupported" and singleton["tile_status_counts"] == {"unsupported_singleton": 1}
    assert missing["status"] == "failed"
    for row in (ready, empty, singleton):
        assert row["tiles_accounted"] == row["tiles_expected"] == 1
    with np.load(tmp_path / "out/case_0000/tile_000000.npz") as packet:
        assert packet["points_xyzi"].shape == (8, 4)
        np.testing.assert_array_equal(packet["center_proposals"], packet["points_xyzi"][:, :3])
        assert len(np.unique(packet["selected_flat_indices"])) == 8
    with pytest.raises(FileExistsError):
        audit_reference_forward(cases, tiling, clouds, tmp_path / "out")


def test_cli_executes_cpu_control_and_returns_failure_for_retained_unsupported_cases(tmp_path):
    cases, tiling, clouds = manifests(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/audit_reference_forward.py"
    command = [sys.executable, str(script), "--cases", str(cases), "--tiling", str(tiling),
               "--clouds", str(clouds), "--output-dir", str(tmp_path / "cli")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert "no optimizer or reconstruction" in result.stdout
    obj = json.loads((tmp_path / "cli/report.json").read_text(encoding="utf-8"))
    assert obj["runtime"]["cpu_threads"] == 1
    assert obj["model"]["state_unchanged"] is True
