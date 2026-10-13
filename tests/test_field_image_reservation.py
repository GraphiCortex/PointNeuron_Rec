"""Quota provenance and a cached-only paired run; no research training."""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from pointneuron.data.field_guided_sampling import candidate_pools, select_cloud, local_xyz
from pointneuron.data.field_image_reservation import image_reserved_cloud
from pointneuron.data.query_cache import file_digest
from pointneuron.data.swc import parse_swc
from scripts.pointneuron2 import compare_field_image_reservation as runner
from scripts.pointneuron2.field_sampling_metrics import reference_geometry, cloud_metrics, reference_distances
from test_field_pilot import native_fixture, CONFIG

ROOT = Path(__file__).resolve().parents[1]


def evidence():
    raw = np.zeros((5, 7, 15), dtype=np.uint16)
    raw[2, 3, :] = 4095
    valid = np.ones(raw.shape, dtype=bool)
    valid[0, 0, 0] = False
    probability = np.zeros(raw.shape, dtype=np.float32)
    probability[:, 3, :] = .6
    probability[0, 0, 0] = 1
    return raw, valid, probability


@pytest.mark.parametrize("n", [0, 1, 13, 29, 1000])
def test_reservation_fills_unique_disjoint_budget_with_native_features(n):
    raw, valid, p = evidence()
    before = [a.copy() for a in (raw, valid, p)]
    pools = candidate_pools(raw, valid, p, 800)
    cloud, mask = image_reserved_cloud(raw, valid, p, [10, 20, 30], 800, n, 7)
    assert cloud.actual_count == min(n, len(pools.expanded))
    assert mask.sum() == min(cloud.actual_count // 2, len(pools.foreground))
    assert np.isin(cloud.voxel_indices[mask], pools.foreground).all()
    assert np.isin(cloud.voxel_indices, pools.expanded).all()
    assert len(cloud.voxel_indices) == len(np.unique(cloud.voxel_indices))
    assert np.all(valid.ravel()[cloud.voxel_indices])
    np.testing.assert_array_equal(cloud.points[:, :3], local_xyz(cloud.voxel_indices, raw.shape) + [10, 20, 30])
    np.testing.assert_array_equal(cloud.points[:, 3], raw.ravel()[cloud.voxel_indices])
    np.testing.assert_array_equal(cloud.field_scores, p.ravel()[cloud.voxel_indices])
    assert not cloud.selection_role.any()  # Quota is not a learned-guide role.
    repeat, repeat_mask = image_reserved_cloud(raw, valid, p, [10, 20, 30], 800, n, 7)
    np.testing.assert_array_equal(repeat.points, cloud.points)
    np.testing.assert_array_equal(repeat_mask, mask)
    for actual, original in zip((raw, valid, p), before):
        np.testing.assert_array_equal(actual, original)


def test_score_ranking_cannot_change_quota_selection_when_admission_is_fixed():
    raw, valid, p = evidence()
    altered = p.copy()
    altered[p >= .5] = np.linspace(.501, .999, (p >= .5).sum(), dtype=np.float32)
    left, mask = image_reserved_cloud(raw, valid, p, [0, 0, 0], 800, 20, 8)
    right, other = image_reserved_cloud(raw, valid, altered, [0, 0, 0], 800, 20, 8)
    np.testing.assert_array_equal(left.voxel_indices, right.voxel_indices)
    np.testing.assert_array_equal(mask, other)


def test_empty_foreground_and_empty_union_have_explicit_shortage_behavior():
    raw, valid, p = evidence()
    cloud, mask = image_reserved_cloud(raw*0, valid, p, [0, 0, 0], 800, 50, 0)
    assert cloud.actual_count == 50 and not mask.any()
    cloud, mask = image_reserved_cloud(raw*0, valid, p*0, [0, 0, 0], 800, 50, 0)
    assert cloud.actual_count == 0 and cloud.points.shape == (0, 4) and not len(mask)
    with pytest.raises(ValueError, match="nonnegative integer"):
        image_reserved_cloud(raw, valid, p, [0, 0, 0], 800, True, 0)
    with pytest.raises(ValueError, match="native XYZ"):
        image_reserved_cloud(raw, valid, p, [.1, 0, 0], 800, 20, 0)


@pytest.fixture
def saved_fixture(native_fixture):
    original, data, records = native_fixture
    base = original.output_dir.parent
    models, source = base / "baseline/model", base / "baseline/comparison"
    (models / "checkpoints").mkdir(parents=True)
    (models / "development_fields").mkdir()
    for folder in ["inputs", "clouds"]:
        (source / folder).mkdir(parents=True)
    checkpoint = models / "checkpoints/field_epoch_030.pt"
    # Synthetic frozen contract: no model is created or trained by this fixture.
    checkpoint.write_bytes(b"synthetic cached-only contract checkpoint")
    fields, measured, protected = {}, [], {}
    for row in records:
        native = original.manifest.parent / row["path"]
        protected[str(native.resolve())] = row["sha256"]
        if row["partition"] != "val":
            continue
        with np.load(native) as z:
            raw, valid, image = (z[k] for k in ["raw_image", "valid", "image"])
            meta = json.loads(str(z["metadata"]))
        origin = np.asarray(meta["origin_xyz"])
        prob = np.zeros(raw.shape, dtype=np.float32)
        prob[4, 3:6, :] = .9
        pools = candidate_pools(raw, valid, prob, meta["threshold"])
        name = Path(row["path"]).name
        field_path = models / "development_fields" / name
        np.savez_compressed(field_path, probability=prob, origin_xyz=origin)
        fields[name] = file_digest(field_path)
        input_path = source / "inputs" / name
        np.savez_compressed(input_path, raw_image=raw, valid=valid, image=image, probability=prob,
                            origin_xyz=origin, intensity_threshold=meta["threshold"])
        protected[str(input_path.resolve())] = file_digest(input_path)
        geometry = reference_geometry(parse_swc(data / row["sample_id"] / "trace.swc"), origin, raw.shape)
        for budget in [512, 2048]:
            n = min(budget, len(pools.foreground))
            for seed in range(5):
                cloud = select_cloud(raw, valid, prob, origin, meta["threshold"], "expanded_random", n,
                                     seed + row["sample_index"]*10000 + row["patch_index"])
                path = source / "clouds" / f"sample_{row['sample_index']}_patch_{row['patch_index']}_n{n}_seed{seed}.npz"
                np.savez_compressed(path, points=cloud.points, voxel_indices=cloud.voxel_indices,
                                    field_scores=cloud.field_scores, selection_role=cloud.selection_role,
                                    metadata=np.array(json.dumps({"partition": "val", "field_fallback": False})))
                protected[str(path.resolve())] = file_digest(path)
                dist = reference_distances(local_xyz(cloud.voxel_indices, raw.shape), geometry["full_segments"])
                metrics = cloud_metrics(cloud, raw, origin, meta["threshold"], geometry, cloud.voxel_indices, dist)
                if metrics["reference_coverage_2"] is None:
                    metrics["reference_proximity_6"] = None
                measured.append({"sample_index": row["sample_index"], "patch_index": row["patch_index"], "nominal_budget": budget,
                    "sampling_seed": seed, "group": "matched", "arm": "expanded_random", "actual_unique_count": n,
                    "cloud_path": str(path.relative_to(source)), **metrics})
    fit = {"status": "completed", "config": json.loads(CONFIG.read_text()),
           "model_spec": {"kind": "centerline_unet_v1", "shared_initialization_sha256": "fixture"},
           "manifest_sha256": file_digest(original.manifest), "partition_sha256": file_digest(original.partition),
           "final_checkpoint_sha256": file_digest(checkpoint), "checkpoint_reload_max_logit_delta": 0,
           "evaluations": [{"epoch": 30}], "final_development_fields_sha256": fields}
    runner.write_json(models / "pilot_report.json", fit)
    saved = {"status": "completed", "config": runner.EXPECTED_CONFIG, "source_fit_sha256": file_digest(models / "pilot_report.json"),
             "source_checkpoint_sha256": file_digest(checkpoint), "manifest_sha256": fit["manifest_sha256"],
             "model_spec": fit["model_spec"], "optimizer_updates": 0, "reserved_test_records": 0,
             "saved_field_max_score_delta": 0, "protected_files_sha256": protected, "source_files_sha256": {}, "rows": measured}
    runner.write_json(source / "comparison.json", saved)
    return Namespace(action="check", manifest=original.manifest, partition=original.partition, config=CONFIG,
                     recipe=ROOT / "configs/pointneuron2/field_image_reservation.json", models=models,
                     source_comparison=source, data_root=data, output_dir=base / "quota"), records


def test_cached_only_run_freezes_before_reference_and_preserves_sources(saved_fixture, monkeypatch):
    args, records = saved_fixture
    before = {str(p): file_digest(p) for folder in [args.models, args.source_comparison] for p in folder.rglob("*") if p.is_file()}
    from pointneuron.models.centerline_unet import CenterlineUNet
    monkeypatch.setattr(CenterlineUNet, "__init__", lambda *a, **k: pytest.fail("A cached-only run must not create a model"))
    checked = runner.run(args)
    assert checked["matched_pairs"] == 20 and not args.output_dir.exists()
    args.action = "compare"
    monkeypatch.setattr(runner, "inspection", lambda *a, **k: None)
    original_matching = runner.matching_source
    def after_freeze(*a, **k):
        report = runner.read_json(args.output_dir / "comparison.json")
        assert report["frozen_rows"] == 20 and report["phase"].startswith("all new clouds frozen")
        assert len(list((args.output_dir / "clouds").glob("*.npz"))) == 10
        return original_matching(*a, **k)
    monkeypatch.setattr(runner, "matching_source", after_freeze)
    report = runner.run(args)
    assert report["status"] == "completed" and report["optimizer_updates"] == report["inference_passes"] == 0
    assert all(report["checks"].values()) and len(report["rows"]) == 40 and len(report["paired"]["pairs"]) == 20
    assert len(report["summaries"]) == 4
    for row in report["rows"]:
        assert row["actual_unique_count"] == 18
        if row["arm"] == runner.RESERVED:
            assert row["reserved_foreground_count"] == 9 and row["foreground_point_count"] >= 9
        if row["patch_index"] == 1:
            assert row["reference_coverage_2"] is None and row["reference_proximity_6"] is None
    assert all(file_digest(p) == digest for p, digest in before.items())
    with pytest.raises(FileExistsError):
        runner.run(args)


@pytest.mark.parametrize("corruption", ["field", "control", "quota", "reserved"])
def test_stale_inputs_and_changed_recipe_fail_before_output(saved_fixture, corruption):
    args, records = saved_fixture
    if corruption == "field":
        path = next((args.models / "development_fields").glob("*.npz"))
        path.write_bytes(path.read_bytes() + b"changed")
    elif corruption == "control":
        path = next((args.source_comparison / "clouds").glob("*.npz"))
        path.write_bytes(path.read_bytes() + b"changed")
    elif corruption == "quota":
        args.recipe = args.output_dir.parent / "changed_recipe.json"
        recipe = dict(runner.EXPECTED_RECIPE); recipe["foreground_fraction"] = .25
        runner.write_json(args.recipe, recipe)
    else:
        manifest = runner.read_json(args.manifest)
        manifest["records"][2].update(partition="test", sample_index=2, sample_id="reserved")
        runner.write_json(args.manifest, manifest)
    args.action = "compare"
    with pytest.raises((ValueError, AssertionError)):
        runner.run(args)
    assert not args.output_dir.exists()


def test_source_artifact_paths_cannot_escape_cache(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        runner.safe_path(tmp_path, "../outside.npz")
