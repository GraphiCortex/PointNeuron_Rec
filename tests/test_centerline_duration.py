"""Protect the duration-only comparison from hidden data/loss/settings changes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from pointneuron.data.centerline_field import SCHEMA
from pointneuron.data.query_cache import file_digest
from pointneuron.models.centerline_unet import CenterlineUNet
from scripts.pointneuron2.centerline_duration_comparison import (
    check_duration_baseline, compare_duration, duration_config, seeded_order, completed_fit_config,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def replay(tmp_path):
    source = json.loads((ROOT / "configs/pointneuron2/centerline_fit.json").read_text())
    recipe = ROOT / "configs/pointneuron2/centerline_duration.json"
    loss = ROOT / "configs/pointneuron2/centerline_loss_control.json"
    cfg = duration_config(source, recipe)
    spec = json.loads(loss.read_text())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"config": source}))
    digest = file_digest(manifest)
    torch.manual_seed(source["seed"])
    model = CenterlineUNet(source["base_channels"])
    rows = [{"sample_index": i, "patch_index": j, "reference_coverage_6": 1.,
             "reference_proximity_6": .94, "known_voxel_dice": .6,
             "mean_reference_tube_score": .9, "accepted_voxels": 30,
             "accepted_unknown_bright_voxels": 2} for i, j in source["crops"]]
    baseline = {"status": "completed", "config": source, "manifest_sha256": digest,
                "device": "cpu", "torch_version": str(torch.__version__), "loss_spec": spec,
                "checkpoint_reload_max_logit_delta": 0., "final_gate_passed": False,
                "crop_index_order": seeded_order(0, 400),
                "evaluations": [{"step": s, "crops": deepcopy(rows), "gate_passed": False}
                                for s in source["eval_steps"]]}
    (tmp_path / "fit_report.json").write_text(json.dumps(baseline))
    for step in (0, 400):
        torch.save({"schema": SCHEMA, "step": step, "config": source, "loss_spec": spec,
                    "manifest_sha256": digest, "model": model.state_dict()}, tmp_path / f"field_step_{step:04d}.pt")
    pairing = check_duration_baseline(tmp_path, manifest, source, cfg, torch.device("cpu"), model, loss, recipe)
    report = {"config": cfg, "manifest_sha256": digest, "loss_spec": spec,
              "crop_index_order": seeded_order(0, 1600),
              "evaluations": [{"step": s, "crops": deepcopy(rows), "gate_passed": False} for s in cfg["eval_steps"]]}
    for row in report["evaluations"][-1]["crops"]:
        row["known_voxel_dice"] = .82
    report["evaluations"][-1]["gate_passed"] = True
    return tmp_path, source, cfg, manifest, model, loss, recipe, pairing, report


def test_recipe_cannot_change_loss_labels_or_gate(tmp_path):
    cfg = json.loads((ROOT / "configs/pointneuron2/centerline_fit.json").read_text())
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"schema": "pointneuron2-centerline-duration-v1", "steps": 1600,
                                 "eval_steps": [0, 400, 800, 1600], "threshold": .4}))
    with pytest.raises(ValueError, match="approved"):
        duration_config(cfg, recipe)


@pytest.mark.parametrize("mutation", ["learning_rate", "loss", "initialization", "order"])
def test_baseline_pairing_rejects_hidden_changes(replay, mutation):
    path, source, cfg, manifest, model, loss, recipe, pairing, report = replay
    if mutation == "learning_rate":
        cfg = {**cfg, "learning_rate": .002}
    elif mutation == "initialization":
        with torch.no_grad():
            next(model.parameters()).add_(.1)
    else:
        baseline = pairing["baseline_report"]
        if mutation == "loss":
            baseline["loss_spec"] = {**baseline["loss_spec"], "negative_aggregation": "all"}
        else:
            baseline["crop_index_order"][0] = (baseline["crop_index_order"][0] + 1) % 8
        (path / "fit_report.json").write_text(json.dumps(baseline))
    with pytest.raises(ValueError):
        check_duration_baseline(path, manifest, source, cfg, torch.device("cpu"), model, loss, recipe)


def test_duration_effect_uses_common_checkpoint_and_reports_replay_drift(replay):
    *_, pairing, report = replay
    result = compare_duration(report, pairing)
    assert all(result["pairing_checks"].values())
    assert result["common_budget_metrics_match_within_tolerance"]
    assert result["summary"]["replay_1600"]["known_voxel_dice"] == pytest.approx(.82)
    assert result["crops"][0]["duration_delta_1600_minus_replay400"]["known_voxel_dice"] == pytest.approx(.22)
    report["evaluations"][1]["crops"][0]["known_voxel_dice"] = .65
    drift = compare_duration(report, pairing)
    assert not drift["common_budget_metrics_match_within_tolerance"]
    assert drift["crops"][0]["duration_delta_1600_minus_replay400"]["known_voxel_dice"] == pytest.approx(.17)


def test_read_only_audit_accepts_only_verified_budget_change(replay):
    _, source, cfg, _, _, _, recipe, _, report = replay
    assert completed_fit_config(source, {"config": source}, recipe) == source
    with pytest.raises(ValueError, match="approved duration"):
        completed_fit_config(source, report, recipe)
    report["duration_config_sha256"] = file_digest(recipe)
    assert completed_fit_config(source, report, recipe) == cfg
    report["config"] = {**cfg, "threshold": .4}
    with pytest.raises(ValueError, match="approved duration"):
        completed_fit_config(source, report, recipe)


def test_finalize_rejects_reordered_or_missing_crop_and_changed_baseline(replay):
    path, *_, pairing, report = replay
    wrong = deepcopy(report)
    wrong["crop_index_order"][0], wrong["crop_index_order"][1] = wrong["crop_index_order"][1], wrong["crop_index_order"][0]
    with pytest.raises(AssertionError, match="crop order"):
        compare_duration(wrong, pairing)
    wrong = deepcopy(report)
    wrong["evaluations"][1]["crops"].pop()
    with pytest.raises(AssertionError, match="crop identities"):
        compare_duration(wrong, pairing)
    (path / "fit_report.json").write_text("changed")
    with pytest.raises(AssertionError, match="baseline changed"):
        compare_duration(report, pairing)
