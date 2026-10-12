"""Tiny synthetic I/O and frozen-model fixture; no real research training."""
from argparse import Namespace
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from pointneuron.data.centerline_field import SCHEMA
from pointneuron.data.query_cache import file_digest
from pointneuron.data.vaa3d_raw import RAW_HEADER_KEY, VOLUME_READER_VERSION
from scripts.pointneuron2.train_centerline_fit import field_metrics
import scripts.pointneuron2.compare_field_sampling as runner

ROOT = Path(__file__).resolve().parents[1]


def write_raw(path, raw):
    z, y, x = raw.shape
    kind = raw.dtype.itemsize
    path.write_bytes(RAW_HEADER_KEY.encode() + b"L" + struct.pack("<h4i", kind, x, y, z, 1) + raw.tobytes())


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    zz, yy, xx = np.indices((9, 9, 9))
    distance = np.sqrt((zz-4.)**2 + (yy-4.)**2).astype(np.float32)
    target = distance <= 2
    raw = np.where(target, 200, 0).astype(np.uint8)
    raw[4, 4, :] = 20  # Dim centers below the original cutoff; must retain original intensity.
    valid = np.ones(raw.shape, dtype=bool)
    image = raw.astype(np.float32)/200
    logits = torch.from_numpy(np.where(target, 10., -10.).astype(np.float32))[None, None]
    calls = []

    class FixedSyntheticField(torch.nn.Module):
        def __init__(self, base_channels):
            super().__init__()
            self.register_buffer("logits", logits.clone())

        def forward(self, x):
            calls.append("inference")
            return self.logits.expand_as(x)

    monkeypatch.setattr(runner, "CenterlineUNet", FixedSyntheticField)
    monkeypatch.setattr(runner, "inspect_record", lambda *args: None)
    cache, models, data = [tmp_path / name for name in ("cache", "models", "data")]
    for directory in (cache, models, data):
        directory.mkdir()
    source = json.loads((ROOT / "configs/pointneuron2/centerline_fit.json").read_text())
    cfg = {**source, "steps": 1600, "eval_steps": [0, 400, 800, 1600]}
    partition = tmp_path / "partition.json"
    partition.write_text(json.dumps({"partitions": {
        "train": [{"sample_index": i, "sample_id": f"synthetic_{i}"} for i in (0, 32, 96, 162)],
        "val": [], "test": [{"sample_index": 99, "sample_id": "reserved"}]}}))
    rows, evaluated = [], []
    for sample, patch in source["crops"]:
        directory = data / f"synthetic_{sample}"
        directory.mkdir(exist_ok=True)
        raw_path, swc_path = directory / "image.v3draw", directory / "trace.swc"
        write_raw(raw_path, raw)
        swc_path.write_text("1 3 0 4 4 1 -1\n2 3 8 4 4 1 1\n")
        metadata = {"field_schema": SCHEMA, "sample_index": sample, "patch_index": patch,
                    "sample_id": directory.name, "partition": "train", "volume_sha256": file_digest(raw_path),
                    "swc_sha256": file_digest(swc_path), "patch_center": [4, 4, 4], "origin_xyz": [0, 0, 0],
                    "config": {"patch_radius": 4}, "image_scale": 200., "threshold": 40,
                    "foreground_count": int((raw > 40).sum())}
        path = cache / f"sample_{sample:04d}_patch_{patch:03d}.npz"
        reference = np.array([[x, 4., 4.] for x in range(9)])
        np.savez_compressed(path, image=image, valid=valid, target=target, known=valid,
                            unknown_bright=~valid, distance=distance, reference_local_xyz=reference,
                            metadata=np.array(json.dumps(metadata)))
        rows.append({k: metadata[k] for k in ("sample_index", "patch_index", "sample_id", "partition")})
        rows[-1].update(path=path.name, sha256=file_digest(path))
        probability = logits.sigmoid()[0, 0].numpy()
        evaluated.append({"sample_index": sample, "patch_index": patch,
                          **field_metrics(probability, {"valid": valid, "target": target, "known": valid,
                              "unknown_bright": ~valid, "distance": distance, "reference_local_xyz": reference})})
    manifest = cache / "manifest.json"
    manifest.write_text(json.dumps({"status": "completed", "schema": SCHEMA, "config": source,
        "records": rows, "volume_reader_version": VOLUME_READER_VERSION, "partition_sha256": file_digest(partition)}))
    loss_spec = {"schema": "pointneuron2-centerline-loss-control-v1", "negative_aggregation": "topk", "negative_to_positive_ratio": 3}
    fit = {"status": "completed", "final_gate_passed": True, "config": cfg, "torch_version": str(torch.__version__),
           "manifest_sha256": file_digest(manifest), "loss_spec": loss_spec, "checkpoint_reload_max_logit_delta": 0.,
           "duration_config_sha256": file_digest(ROOT / "configs/pointneuron2/centerline_duration.json"),
           "evaluations": [{"step": s, "crops": evaluated} for s in cfg["eval_steps"]]}
    (models / "fit_report.json").write_text(json.dumps(fit))
    torch.save({"schema": SCHEMA, "step": 1600, "config": cfg, "loss_spec": loss_spec,
                "manifest_sha256": file_digest(manifest), "model": FixedSyntheticField(8).state_dict()}, models / "field_step_1600.pt")
    args = Namespace(manifest=manifest, models=models, data_root=data, output_dir=tmp_path / "run",
                     config=ROOT / "configs/pointneuron2/field_sampling_comparison.json", partition=partition,
                     device="cpu", preflight=False)
    return args, calls, raw, rows


def test_complete_synthetic_comparison_freezes_all_inputs_before_swc_and_preserves_sources(fixture, monkeypatch):
    args, calls, raw, source_rows = fixture
    original = runner.matching_source

    def checked_source(directory, suffixes, sha):
        if suffixes == {".swc"}:
            assert len(calls) == 8
            assert len(list((args.output_dir / "inputs").glob("*.npz"))) == 8
            assert len(list((args.output_dir / "clouds").glob("*.npz"))) > 0
        return original(directory, suffixes, sha)

    monkeypatch.setattr(runner, "matching_source", checked_source)
    result = runner.run(args)
    assert result["status"] == "completed" and all(result["checks"].values())
    assert result["optimizer_updates"] == 0 and len(result["rows"]) == 640
    assert len(result["primary_guided_minus_coverage_pairs"]) == 40
    assert result["candidate_availability"][0]["below_threshold_additions"] == 9
    assert all(r["duplicate_count"] == 0 for r in result["rows"])
    for key in {(r["sample_index"], r["patch_index"], r["nominal_budget"], r["sampling_seed"]) for r in result["rows"]}:
        matched = [r for r in result["rows"] if r["group"] == "matched" and
                   (r["sample_index"], r["patch_index"], r["nominal_budget"], r["sampling_seed"]) == key]
        assert len({r["actual_unique_count"] for r in matched}) == 1
    assert all(file_digest(args.manifest.parent / r["path"]) == r["sha256"] for r in source_rows)
    assert (args.output_dir / "metrics.csv").exists()


def test_preflight_runs_one_forward_pass_and_produces_timing_only(fixture):
    args, calls, *_ = fixture
    args.preflight = True
    result = runner.run(args)
    assert result["status"] == "completed" and len(calls) == 1
    assert result["optimizer_updates"] == 0 and "primary_guided_minus_coverage_pairs" not in result
    assert result["representative_compute_estimate_seconds"] > 0
    assert (args.output_dir / "timing.json").exists() and not (args.output_dir / "comparison.json").exists()


def test_reserved_record_is_rejected_before_source_image_or_model_inference(fixture):
    args, calls, *_ = fixture
    manifest = json.loads(args.manifest.read_text())
    manifest["records"][0]["partition"] = "test"
    args.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Reserved"):
        runner.run(args)
    assert not calls and not args.output_dir.exists()


def test_uint16_source_intensity_is_verified_and_bad_cached_values_are_rejected(tmp_path):
    raw = np.arange(125, dtype=np.uint16).reshape(5, 5, 5)*30
    directory = tmp_path / "synthetic"
    directory.mkdir()
    path = directory / "image.v3draw"
    write_raw(path, raw)
    metadata = {"sample_id": "synthetic", "volume_sha256": file_digest(path), "image_scale": float(raw.max()),
                "patch_center": [2, 2, 2], "origin_xyz": [0, 0, 0], "config": {"patch_radius": 2}}
    valid = np.ones(raw.shape, dtype=bool)
    image = raw.astype(np.float32)/float(raw.max())
    restored, _ = runner.RawImages(tmp_path, {}).crop(metadata, image, valid)
    np.testing.assert_array_equal(restored, raw)
    image[0, 0, 0] = .5
    with pytest.raises(ValueError, match="Raw crop"):
        runner.RawImages(tmp_path, {}).crop(metadata, image, valid)
