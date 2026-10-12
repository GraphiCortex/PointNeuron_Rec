"""Pairing checks and final comparison for the one-change centerline loss control."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pointneuron.data.centerline_field import SCHEMA
from pointneuron.data.query_cache import file_digest


def check_baseline(models, manifest_path, cfg, device, initialized_model, loss_config):
    loss_config = Path(loss_config)
    spec = json.loads(loss_config.read_text())
    expected = {"schema": "pointneuron2-centerline-loss-control-v1", "negative_aggregation": "topk",
                "negative_to_positive_ratio": 3}
    if spec != expected:
        raise ValueError("Only the frozen top-min(N,3P) negative BCE control is supported")
    models = Path(models)
    report_path = models / "fit_report.json"
    baseline = json.loads(report_path.read_text())
    if (baseline.get("status") != "completed" or baseline["config"] != cfg or
            baseline["manifest_sha256"] != file_digest(manifest_path) or baseline["device"] != str(device)):
        raise ValueError("Baseline cache/config/device mismatch")
    if baseline.get("loss_spec", {"negative_aggregation": "all"})["negative_aggregation"] != "all":
        raise ValueError("Baseline must use the original all-negative BCE")
    if baseline["torch_version"] != str(torch.__version__):
        raise ValueError("Use the same PyTorch version as the completed baseline")
    if baseline.get("checkpoint_reload_max_logit_delta", float("inf")) > 1e-6:
        raise ValueError("Baseline checkpoint reload did not pass")
    if [e["step"] for e in baseline["evaluations"]] != cfg["eval_steps"]:
        raise ValueError("Baseline evaluation budget mismatch")
    for e in baseline["evaluations"]:
        if [(r["sample_index"], r["patch_index"]) for r in e["crops"]] != [tuple(r) for r in cfg["crops"]]:
            raise ValueError("Baseline evaluation identities mismatch")
    paths = [report_path, models / "field_step_0000.pt", models / f"field_step_{cfg['steps']:04d}.pt"]
    for step, path in ((0, paths[1]), (cfg["steps"], paths[2])):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["schema"] != SCHEMA or checkpoint["step"] != step or checkpoint["config"] != cfg
                or checkpoint["manifest_sha256"] != baseline["manifest_sha256"]):
            raise ValueError("Baseline checkpoint contract mismatch")
        if step == 0:
            actual = initialized_model.state_dict()
            if actual.keys() != checkpoint["model"].keys() or any(
                    not torch.equal(actual[key].detach().cpu(), tensor) for key, tensor in checkpoint["model"].items()):
                raise ValueError("Fresh initialization differs from saved baseline step zero")
    files = {str(p.resolve()): file_digest(p) for p in paths}
    return {"loss_spec": spec, "loss_config_sha256": file_digest(loss_config),
            "baseline_files_sha256": files, "checks": {
                "identical_initial_model_tensors": True, "identical_cache_hash": True,
                "identical_config_and_declared_update_budget": True, "identical_device_and_torch": True,
                "identical_evaluation_steps_and_crop_identities": True}, "baseline_report": baseline}


def assert_baseline_unchanged(pairing):
    if any(file_digest(path) != digest for path, digest in pairing["baseline_files_sha256"].items()):
        raise AssertionError("Saved baseline changed during control")


def assert_initial_metrics(fresh_crops, pairing):
    for original, fresh in zip(pairing["baseline_report"]["evaluations"][0]["crops"], fresh_crops):
        for key in ("reference_coverage_6", "reference_proximity_6", "known_voxel_dice", "mean_reference_tube_score"):
            if abs(original[key] - fresh[key]) > 1e-5:
                raise AssertionError("Step-zero inference differs from baseline")
        if original["accepted_voxels"] != fresh["accepted_voxels"]:
            raise AssertionError("Step-zero accepted voxel count differs from baseline")


def compare_final(control, pairing):
    assert_baseline_unchanged(pairing)
    baseline = pairing["baseline_report"]
    if control["config"] != baseline["config"] or control["manifest_sha256"] != baseline["manifest_sha256"]:
        raise AssertionError("Control settings differ from baseline")
    if [e["step"] for e in control["evaluations"]] != [e["step"] for e in baseline["evaluations"]]:
        raise AssertionError("Control evaluation budget differs from baseline")
    for e in control["evaluations"]:
        if [(r["sample_index"], r["patch_index"]) for r in e["crops"]] != [tuple(r) for r in baseline["config"]["crops"]]:
            raise AssertionError("Control crop identities differ from baseline")
    expected_order, remaining = [], []
    rng = np.random.default_rng(control["config"]["seed"])
    for _ in range(control["config"]["steps"]):
        if not remaining:
            remaining = rng.permutation(8).tolist()
        expected_order.append(remaining.pop())
    if control["crop_index_order"] != expected_order:
        raise AssertionError("Control crop order differs from baseline's declared seeded algorithm")
    if "crop_index_order" in baseline and baseline["crop_index_order"] != expected_order:
        raise AssertionError("Recorded baseline crop order differs from control")
    assert_initial_metrics(control["evaluations"][0]["crops"], pairing)
    result = {"scope": "fixed training-crop loss-only control; not generalization or SWC quality",
              "pairing_checks": {**pairing["checks"], "step_zero_inference_matches": True, "baseline_files_unchanged": True,
                                 "control_crop_order_matches_declared_seeded_algorithm": True},
              "baseline_crop_order_recorded": "crop_index_order" in baseline,
              "baseline_files_sha256": pairing["baseline_files_sha256"], "loss_spec": pairing["loss_spec"],
              "final_step": control["config"]["steps"], "threshold": control["config"]["threshold"], "crops": [],
              "baseline_gate_passed": baseline["final_gate_passed"], "control_gate_passed": control["final_gate_passed"]}
    keys = ("reference_coverage_6", "reference_proximity_6", "known_voxel_dice", "accepted_voxels", "accepted_unknown_bright_voxels")
    for old, new in zip(baseline["evaluations"][-1]["crops"], control["evaluations"][-1]["crops"]):
        result["crops"].append({"sample_index": old["sample_index"], "patch_index": old["patch_index"],
            "baseline": {k: old[k] for k in keys}, "control": {k: new[k] for k in keys},
            "delta": {k: new[k]-old[k] for k in keys}})
    result["summary"] = {arm: {key: float(np.mean([r[arm][key] for r in result["crops"]]))
                                 for key in keys[:3]} for arm in ("baseline", "control")}
    result.update(status="completed", next_step="review paired scores and overlays; no automatic validation promotion")
    return result
