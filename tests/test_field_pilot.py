"""Tiny CPU contracts: native labels, reserved data and complete pilot I/O."""
from argparse import Namespace
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from pointneuron.data.centerline_field import SCHEMA as FIELD_SCHEMA, segment_distance_field, weak_labels
from pointneuron.data.field_pilot import CACHE_SCHEMA, SCHEMA, NativeRecords, field_metrics, load_manifest, partition_contract, validate_record
from pointneuron.data.query_cache import file_digest
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import RAW_HEADER_KEY, VOLUME_READER_VERSION
from scripts.pointneuron2.field_sampling_metrics import reference_geometry
import scripts.pointneuron2.prepare_field_pilot as preparation
import scripts.pointneuron2.train_field_pilot as training
import scripts.pointneuron2.compare_field_pilot as comparison

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pointneuron2/field_pilot.json"
SAMPLING_CONFIG = ROOT / "configs/pointneuron2/field_sampling_comparison.json"


def write_raw(path, raw):
    z, y, x = raw.shape
    path.write_bytes(RAW_HEADER_KEY.encode()+b"L"+struct.pack("<h4i", raw.dtype.itemsize, x, y, z, 1)+raw.tobytes())


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(training, "inspection", lambda *a, **k: None)
    monkeypatch.setattr(comparison, "inspect_record", lambda *a, **k: None)
    cache, data = tmp_path/"cache", tmp_path/"data"
    cache.mkdir()
    data.mkdir()
    partition = tmp_path/"partition.json"
    partition.write_text(json.dumps({"partitions": {
        "train": [{"sample_index": 0, "sample_id": "train"}],
        "val": [{"sample_index": 1, "sample_id": "dev"}],
        "test": [{"sample_index": 2, "sample_id": "reserved"}]}}))
    cfg = json.loads(CONFIG.read_text())
    rows = []
    for sample, role, identity in ((0, "train", "train"), (1, "val", "dev")):
        directory = data/identity
        directory.mkdir()
        swc_path = directory/"trace.swc"
        swc_path.write_text("1 3 0 4 4 1 -1\n2 3 8 4 4 1 1\n")
        tree = parse_swc(swc_path)
        for patch in (0, 1):
            shape = (9, 9, 9)
            raw = np.zeros(shape, dtype=np.uint8)
            raw[4, 4, :] = 200
            # Bright off-reference structure is unknown, not a verified negative.
            raw[0, 0, :] = 100
            valid = np.ones(shape, dtype=bool)
            distance = segment_distance_field(tree, [0, 0, 0], shape, 6) if patch == 0 else np.full(shape, np.inf, dtype=np.float32)
            target, known, unknown = weak_labels(raw, valid, distance, 40)
            geometry = reference_geometry(tree, [0, 0, 0], shape)
            reference = geometry["chains"][0]["probe"]["points"] if patch == 0 else np.empty((0, 3))
            weights = geometry["chains"][0]["probe"]["weights"] if patch == 0 else np.empty(0)
            meta = {"sample_index": sample, "sample_id": identity, "patch_index": patch, "partition": role,
                    "field_schema": FIELD_SCHEMA, "pilot_schema": SCHEMA, "image_scale": 200., "threshold": 40,
                    "patch_center": [4, 4 if patch == 0 else 16, 4],
                    "origin_xyz": [0, 0 if patch == 0 else 12, 0], "config": {"patch_radius": 4},
                    "foreground_count": int((raw > 40).sum()), "swc_sha256": file_digest(swc_path)}
            path = cache/f"sample_{sample:04d}_patch_{patch:03d}.npz"
            np.savez_compressed(path, raw_image=raw, image=raw.astype(np.float32)/200, valid=valid, distance=distance,
                                target=target, known=known, unknown_bright=unknown, reference_local_xyz=reference,
                                reference_weights=weights, metadata=np.array(json.dumps(meta)))
            row = {k: meta[k] for k in ("sample_index", "sample_id", "patch_index", "partition")}
            row.update(path=path.name, sha256=file_digest(path), supervised=patch == 0, positive_voxels=int(target.sum()))
            rows.append(row)
    manifest = cache/"manifest.json"
    manifest.write_text(json.dumps({"schema": CACHE_SCHEMA, "status": "completed", "config": cfg,
        "config_sha256": file_digest(CONFIG), "partition_sha256": file_digest(partition),
        "volume_reader_version": VOLUME_READER_VERSION, "records": rows}))
    return Namespace(manifest=manifest, partition=partition, config=CONFIG, output_dir=tmp_path/"timing",
                     device="cpu", preflight=True, timing=None), data, rows


def test_no_trace_crop_is_not_background_and_missing_reference_metrics_remain_none(native_fixture):
    args, _, rows = native_fixture
    cache = NativeRecords(args.manifest, 1)
    record = cache.get(rows[1])
    assert not record["known"].any() and not rows[1]["supervised"]
    metrics = field_metrics(np.ones(record["image"].shape, dtype=np.float32), record)
    assert metrics["known_voxel_dice"] is None and metrics["reference_coverage_2"] is None
    assert metrics["reference_proximity_6"] is None and metrics["accepted_voxels"] == 729
    bad = dict(record)
    bad["known"] = record["valid"].copy() & ~record["unknown_bright"]
    with pytest.raises(ValueError, match="background"):
        validate_record(bad, rows[1])


def test_lru_bound_and_changed_cache_are_enforced(native_fixture):
    args, _, rows = native_fixture
    cache = NativeRecords(args.manifest, 1)
    cache.get(rows[0])
    cache.get(rows[1])
    assert len(cache.cache) == 1 and rows[1]["path"] in cache.cache
    path = args.manifest.parent/rows[0]["path"]
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="Changed"):
        cache.get(rows[0])


def test_reserved_identity_is_rejected_before_optimizer_or_output(native_fixture):
    args, _, _ = native_fixture
    manifest = json.loads(args.manifest.read_text())
    manifest["records"][0].update(partition="test", sample_index=2, sample_id="reserved")
    args.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Reserved"):
        training.run(args)
    assert not args.output_dir.exists()


def test_training_requires_matching_preflight_and_does_not_start_implicitly(native_fixture):
    args, _, _ = native_fixture
    args.preflight = False
    with pytest.raises(ValueError, match="preflight"):
        training.run(args)
    assert not args.output_dir.exists()
    args.preflight = True
    result = training.run(args)
    assert result["scope"] == "timing only; throwaway weights" and result["planned_optimizer_updates"] == 30
    assert result["throwaway_optimizer_updates"] == 4
    assert not list(args.output_dir.rglob("*.pt"))
    args.timing = args.output_dir/"timing.json"
    timing = json.loads(args.timing.read_text())
    timing["manifest_sha256"] = "wrong"
    args.timing.write_text(json.dumps(timing))
    args.preflight = False
    args.output_dir = args.output_dir.parent/"model"
    with pytest.raises(ValueError, match="Preflight differs"):
        training.run(args)
    assert not args.output_dir.exists()


def test_tiny_complete_training_reload_and_development_cloud_freezing(native_fixture, monkeypatch):
    args, data, source_rows = native_fixture
    training.run(args)
    args.timing = args.output_dir/"timing.json"
    args.output_dir = args.output_dir.parent/"model"
    args.preflight = False
    result = training.run(args)
    assert result["status"] == "completed" and result["optimizer_updates"] == 30
    assert set(result["crop_index_order"]) == {0}  # No unsupervised train crop optimized.
    assert [e["epoch"] for e in result["evaluations"]] == [0, 5, 15, 30]
    assert result["checkpoint_reload_max_logit_delta"] <= 1e-6
    assert result["evaluations"][-1]["val"]["crops"][1]["known_voxel_dice"] is None
    compare_args = Namespace(manifest=args.manifest, partition=args.partition, config=CONFIG,
                             sampling_config=SAMPLING_CONFIG, models=args.output_dir, data_root=data,
                             output_dir=args.output_dir.parent/"comparison", device="cpu")
    original_source = comparison.matching_source
    scoring_calls = []
    def checked_source(directory, suffixes, digest):
        assert len(list((compare_args.output_dir/"inputs").glob("*.npz"))) == 2
        assert len(list((compare_args.output_dir/"clouds").glob("*.npz"))) > 0
        scoring_calls.append(1)
        return original_source(directory, suffixes, digest)
    monkeypatch.setattr(comparison, "matching_source", checked_source)
    report = comparison.run(compare_args)
    assert report["status"] == "completed" and report["optimizer_updates"] == 0
    assert len(report["rows"]) == 160 and len(report["primary_guided_minus_coverage_pairs"]) == 10
    assert len(scoring_calls) == 2 and all(report["checks"].values())
    assert all(file_digest(args.manifest.parent/r["path"]) == r["sha256"] for r in source_rows)
    for path in (compare_args.output_dir/"clouds").glob("*.npz"):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            assert meta["partition"] == "val"
            np.testing.assert_array_equal(z["points"][:, 3], NativeRecords(args.manifest).get(source_rows[2])["raw_image"].ravel()[z["voxel_indices"]])


def test_preparation_labels_full_crossing_segments_even_without_inside_nodes(tmp_path, monkeypatch):
    data = tmp_path/"data"
    partition = tmp_path/"partition.json"
    partition.write_text(json.dumps({"partitions": {
        "train": [{"sample_index": 0, "sample_id": "train"}],
        "val": [{"sample_index": 1, "sample_id": "dev"}], "test": []}}))
    source = tmp_path/"source.json"
    source.write_text("{}")
    records = []
    for sample, role, identity in ((0, "train", "train"), (1, "val", "dev")):
        directory = data/identity
        directory.mkdir(parents=True)
        raw = np.zeros((17, 17, 17), dtype=np.uint16)
        raw[8, 8, :] = 3000
        image_path = directory/"image.v3draw"
        write_raw(image_path, raw)
        swc_path = directory/"trace.swc"
        swc_path.write_text("1 3 0 8 8 1 -1\n2 3 16 8 8 1 1\n")
        meta = {"sample_index": sample, "sample_id": identity, "patch_index": 0, "partition": role,
                "patch_center": [8, 8, 8], "config": {"patch_radius": 4}, "threshold": 600,
                "volume_sha256": file_digest(image_path), "swc_sha256": file_digest(swc_path), "foreground_count": 9}
        path = tmp_path/(identity+".npz")
        np.savez_compressed(path, metadata=np.array(json.dumps(meta)))
        records.append({**{k: meta[k] for k in ("sample_index", "sample_id", "patch_index", "partition")},
                        "resolved_path": str(path), "sha256": file_digest(path)})
    monkeypatch.setattr(preparation, "load_query_manifest", lambda *a: {"records": records})
    monkeypatch.setattr(preparation, "partition_contract", lambda path, **k: partition_contract(path))
    monkeypatch.setattr(preparation, "inspection", lambda *a, **k: None)
    args = Namespace(manifest=source, partition=partition, config=CONFIG, data_root=data, output_dir=tmp_path/"native")
    manifest = preparation.prepare(args)
    assert manifest["status"] == "completed" and all(r["supervised"] for r in manifest["records"])
    assert all(r["reference_length_voxels"] == pytest.approx(8) for r in manifest["records"])
    loaded, _ = load_manifest(args.output_dir/"manifest.json", partition, CONFIG)
    record = NativeRecords(args.output_dir/"manifest.json").get(loaded["records"][0])
    assert record["raw_image"].dtype == np.uint16 and record["raw_image"].max() == 3000
    assert record["target"].any() and len(record["reference_local_xyz"]) > 0
