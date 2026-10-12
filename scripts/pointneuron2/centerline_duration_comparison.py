"""Duration-only replay of the completed hard-negative training-crop fit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pointneuron.data.centerline_field import SCHEMA
from pointneuron.data.query_cache import file_digest
from scripts.pointneuron2.centerline_loss_comparison import assert_baseline_unchanged, assert_initial_metrics


def duration_config(source, path):
    recipe = json.loads(Path(path).read_text())
    if recipe != {"schema": "pointneuron2-centerline-duration-v1", "steps": 1600,
                  "eval_steps": [0, 400, 800, 1600]}:
        raise ValueError("Only the approved 1600-update duration recipe is supported")
    if source["steps"] != 400 or source["eval_steps"] != [0, 100, 200, 400]:
        raise ValueError("Duration replay requires the completed 400-update cache protocol")
    return {**source, "steps": recipe["steps"], "eval_steps": recipe["eval_steps"]}


def seeded_order(seed, steps):
    rng, order, visits = np.random.default_rng(seed), [], []
    for _ in range(steps):
        if not order:
            order = rng.permutation(8).tolist()
        visits.append(order.pop())
    return visits


def completed_fit_config(source, report, recipe_path):
    """Allow the approved budget in read-only audits without relaxing cache checks."""
    if report["config"] == source:
        return source
    if (report["config"] != duration_config(source, recipe_path)
            or report.get("duration_config_sha256") != file_digest(recipe_path)):
        raise ValueError("Completed fit changed more than the approved duration recipe")
    return report["config"]


def check_duration_baseline(models, manifest_path, source_cfg, cfg, device, model, loss_config, recipe_path):
    if cfg != duration_config(source_cfg, recipe_path):
        raise ValueError("Duration control changed settings beyond the update/evaluation budget")
    spec = json.loads(Path(loss_config).read_text())
    expected = {"schema": "pointneuron2-centerline-loss-control-v1", "negative_aggregation": "topk",
                "negative_to_positive_ratio": 3}
    if spec != expected:
        raise ValueError("Duration replay requires the unchanged hard-negative loss")
    models = Path(models)
    report_path = models / "fit_report.json"
    baseline = json.loads(report_path.read_text())
    if (baseline.get("status") != "completed" or baseline["config"] != source_cfg
            or baseline["manifest_sha256"] != file_digest(manifest_path) or baseline["device"] != str(device)
            or baseline["torch_version"] != str(torch.__version__) or baseline.get("loss_spec") != spec):
        raise ValueError("Duration baseline cache/config/device/Torch/loss mismatch")
    if baseline.get("checkpoint_reload_max_logit_delta", float("inf")) > 1e-6:
        raise ValueError("Duration baseline checkpoint reload failed")
    if [e["step"] for e in baseline["evaluations"]] != source_cfg["eval_steps"]:
        raise ValueError("Duration baseline evaluation budget mismatch")
    identities = [tuple(c) for c in cfg["crops"]]
    for e in baseline["evaluations"]:
        if [(c["sample_index"], c["patch_index"]) for c in e["crops"]] != identities:
            raise ValueError("Duration baseline crop identity mismatch")
    if baseline.get("crop_index_order") != seeded_order(cfg["seed"], source_cfg["steps"]):
        raise ValueError("Duration baseline lacks matching recorded crop order")
    paths = [report_path, models / "field_step_0000.pt", models / "field_step_0400.pt"]
    for step, path in ((0, paths[1]), (400, paths[2])):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["schema"] != SCHEMA or checkpoint["step"] != step or checkpoint["config"] != source_cfg
                or checkpoint.get("loss_spec") != spec or checkpoint["manifest_sha256"] != baseline["manifest_sha256"]):
            raise ValueError("Duration baseline checkpoint contract mismatch")
        if step == 0:
            actual = model.state_dict()
            if actual.keys() != checkpoint["model"].keys() or any(
                    not torch.equal(actual[key].detach().cpu(), tensor) for key, tensor in checkpoint["model"].items()):
                raise ValueError("Duration replay initialization differs from saved step zero")
    return {"loss_spec": spec, "loss_config_sha256": file_digest(loss_config),
            "duration_config_sha256": file_digest(recipe_path),
            "baseline_files_sha256": {str(p.resolve()): file_digest(p) for p in paths},
            "checks": {"identical_initial_model_tensors": True, "identical_cache_hash": True,
                       "identical_training_settings_except_duration": True, "identical_device_torch_and_loss": True,
                       "baseline_recorded_crop_order_matches_seed": True}, "baseline_report": baseline}


def compare_duration(report, pairing):
    assert_baseline_unchanged(pairing)
    baseline = pairing["baseline_report"]
    old_cfg, cfg = baseline["config"], report["config"]
    if cfg != {**old_cfg, "steps": 1600, "eval_steps": [0, 400, 800, 1600]}:
        raise AssertionError("Duration replay changed settings beyond the approved budget")
    if report["manifest_sha256"] != baseline["manifest_sha256"] or report["loss_spec"] != baseline["loss_spec"]:
        raise AssertionError("Duration replay cache or loss changed")
    if [e["step"] for e in report["evaluations"]] != cfg["eval_steps"]:
        raise AssertionError("Duration replay evaluation budget changed")
    identities = [tuple(c) for c in cfg["crops"]]
    for e in report["evaluations"]:
        if [(c["sample_index"], c["patch_index"]) for c in e["crops"]] != identities:
            raise AssertionError("Duration replay crop identities changed")
    visits = report["crop_index_order"]
    if visits != seeded_order(cfg["seed"], cfg["steps"]) or visits[:400] != baseline["crop_index_order"]:
        raise AssertionError("Duration replay crop order differs from saved baseline prefix")
    assert_initial_metrics(report["evaluations"][0]["crops"], pairing)
    keys = ("reference_coverage_6", "reference_proximity_6", "known_voxel_dice",
            "accepted_voxels", "accepted_unknown_bright_voxels")
    arms = {"saved_400": baseline["evaluations"][-1],
            **{f"replay_{e['step']}": e for e in report["evaluations"] if e["step"]}}
    rows = []
    for i, (sample, patch) in enumerate(identities):
        values = {arm: {k: e["crops"][i][k] for k in keys} for arm, e in arms.items()}
        rows.append({"sample_index": sample, "patch_index": patch, **values,
                     "duration_delta_1600_minus_replay400": {
                         k: values["replay_1600"][k] - values["replay_400"][k] for k in keys},
                     "common_budget_delta_replay400_minus_saved400": {
                         k: values["replay_400"][k] - values["saved_400"][k] for k in keys}})
    # Observe common-budget reproducibility; do not silently claim identical trained weights.
    common_matches = all(
        abs(r["common_budget_delta_replay400_minus_saved400"][k]) <= (1e-5 if k in keys[:3] else 0)
        for r in rows for k in keys)
    return {"status": "completed", "scope": "duration-only fixed training-crop learnability; not validation or SWC quality",
            "pairing_checks": {**pairing["checks"], "step_zero_inference_matches": True,
                               "baseline_files_unchanged": True, "recorded_crop_order_prefix_matches": True},
            "common_budget_metrics_match_within_tolerance": common_matches,
            "common_budget_note": "Report any numerical drift; primary duration effect is replay1600 versus replay400 in one uninterrupted fit.",
            "baseline_files_sha256": pairing["baseline_files_sha256"], "crops": rows,
            "summary": {arm: {k: float(np.mean([r[arm][k] for r in rows])) for k in keys[:3]} for arm in arms},
            "gate_passed": {arm: e["gate_passed"] for arm, e in arms.items()},
            "next_step": "review final 1600-update gate and all overlays; no automatic extension or validation"}
