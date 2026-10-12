"""Explicit two-arm field comparison; training and sampling are separate actions."""
from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pointneuron.data.field_pilot import write_json
from pointneuron.data.query_cache import file_digest
from pointneuron.models.centerline_unet import CenterlineUNet
from pointneuron.models.centerline_attention import AttentionCenterlineUNet
from scripts.pointneuron2 import train_field_pilot as training
from scripts.pointneuron2 import compare_field_pilot as sampling

EXPECTED_RECIPE = {"schema": "pointneuron2-field-attention-v1", "arms": ["baseline", "attention"],
                   "attention_kind": "additive_decoder_conditioned_skip_gates", "gates": ["gate_b", "gate_a"],
                   "initial_gate_coefficient": .99, "shared_initialization": "saved_pilot_epoch_000_backbone",
                   "sampling_checkpoint_epoch": 30, "sampling_controls_unchanged": True}


def model_spec(arm, initial_digest):
    spec = {"kind": "centerline_unet_v1" if arm == "baseline" else "attention_centerline_unet_v1",
            "shared_initialization_sha256": initial_digest}
    if arm == "attention":
        spec["initial_gate_coefficient"] = .99
    return spec


def load_context(args):
    if json.loads(args.attention_config.read_text()) != EXPECTED_RECIPE:
        raise ValueError("Recipe differs from the agreed two-skip-gate comparison")
    contract_args = Namespace(manifest=args.manifest, config=args.config, partition=args.partition,
                              sampling_config=args.sampling_config, models=args.baseline_models, device="cpu")
    manifest, cfg, _, old, _, protected = sampling.load_contract(contract_args)
    initial_path = args.baseline_models / "checkpoints/field_epoch_000.pt"
    initial_digest = file_digest(initial_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if (old.get("initial_checkpoint_sha256") != initial_digest or initial.get("epoch") != 0
            or initial.get("optimizer_updates") != 0 or initial.get("config") != cfg
            or initial.get("manifest_sha256") != file_digest(args.manifest)
            or initial.get("partition_sha256") != file_digest(args.partition)):
        raise ValueError("Saved initial baseline does not match the completed pilot/cache")
    protected[str(initial_path.resolve())] = initial_digest
    extras = [Path(__file__), args.attention_config, initial_path,
              ROOT / "src/pointneuron/models/centerline_attention.py"]
    protected.update({str(p.resolve()): file_digest(p) for p in extras})
    return manifest, cfg, old, initial["model"], initial_digest, extras, protected


def initial_factory(arm, initial):
    def create(channels):
        if arm == "baseline":
            model = CenterlineUNet(channels)
            model.load_state_dict(initial, strict=True)
        elif arm == "attention":
            model = AttentionCenterlineUNet(channels, .99)
            model.load_shared_backbone(initial)
        else:
            raise ValueError("Unknown comparison arm")
        return model
    return create


def inference_factory(channels, spec):
    if spec["kind"] == "centerline_unet_v1":
        return CenterlineUNet(channels)
    if spec["kind"] == "attention_centerline_unet_v1" and spec.get("initial_gate_coefficient") == .99:
        return AttentionCenterlineUNet(channels, .99)
    raise ValueError("Unsupported explicit field architecture")


def differences(left, right, *, left_label="baseline", right_label="attention"):
    result = {}
    for role in ("train", "val"):
        a, b = left["evaluations"][-1][role], right["evaluations"][-1][role]
        if [(r["sample_index"], r["patch_index"]) for r in a["crops"]] != [(r["sample_index"], r["patch_index"]) for r in b["crops"]]:
            raise ValueError("Compared evaluation identities differ")
        result[role] = {left_label: a["neuron_macro_mean"], right_label: b["neuron_macro_mean"],
                        f"{right_label}_minus_{left_label}": {k: b["neuron_macro_mean"][k]-v
                            for k, v in a["neuron_macro_mean"].items() if v is not None and b["neuron_macro_mean"][k] is not None},
                        f"{left_label}_per_neuron": a["per_neuron"], f"{right_label}_per_neuron": b["per_neuron"]}
    return result


def verify_pair(args, old, initial_digest):
    reports = {}
    shared_states = {}
    for arm in ("baseline", "attention"):
        models = args.experiment_dir/arm/"model"
        report = json.loads((models/"pilot_report.json").read_text())
        if (report.get("status") != "completed" or report.get("model_spec") != model_spec(arm, initial_digest)
                or report.get("manifest_sha256") != file_digest(args.manifest)
                or report.get("config_sha256") != file_digest(args.config)
                or [e["epoch"] for e in report["evaluations"]] != [0, 5, 15, 30]):
            raise ValueError("Missing, stale or mismatched completed attention trial")
        # Full checkpoint/field hashes and final update count are checked before comparison too.
        check = Namespace(manifest=args.manifest, config=args.config, partition=args.partition,
                          sampling_config=args.sampling_config, models=models, device="cpu")
        sampling.load_contract(check, model_factory=inference_factory)
        cp = torch.load(models/"checkpoints/field_epoch_000.pt", map_location="cpu", weights_only=True)
        shared_states[arm] = {k: v for k, v in cp["model"].items() if not k.startswith(("gate_a.", "gate_b."))}
        if file_digest(models/"checkpoints/field_epoch_000.pt") != report["initial_checkpoint_sha256"]:
            raise ValueError("Trial initial checkpoint changed")
        reports[arm] = report
    a, b = reports["baseline"], reports["attention"]
    if (a["config"] != b["config"] or a["backend"] != b["backend"] or a["source_files_sha256"] != b["source_files_sha256"]
            or a["crop_index_order"] != b["crop_index_order"] or a["crop_index_order"] != old["crop_index_order"]
            or set(shared_states["baseline"]) != set(shared_states["attention"])
            or any(not torch.equal(v, shared_states["attention"][k]) for k, v in shared_states["baseline"].items())):
        raise ValueError("Two-arm settings, backend, crop order or initial shared tensors differ")
    return reports


def compare_saved_sampling(reports):
    a, b = reports["baseline"], reports["attention"]
    key = lambda r: (r["sample_index"], r["patch_index"], r["group"], r["nominal_budget"], r["sampling_seed"], r["arm"])
    lookup_a, lookup_b = {key(r): r for r in a["rows"]}, {key(r): r for r in b["rows"]}
    if lookup_a.keys() != lookup_b.keys() or a["config"] != b["config"]:
        raise ValueError("Sampling settings/identities changed between models")
    pairs = []
    for identity, left in sorted(lookup_a.items()):
        right = lookup_b[identity]
        equal = left["actual_unique_count"] == right["actual_unique_count"]
        if identity[2] == "matched" and not equal:
            raise AssertionError("Matched counts changed between field models")
        row = {k: left[k] for k in ("sample_index", "patch_index", "group", "nominal_budget", "sampling_seed", "arm")}
        row.update(baseline_unique_count=left["actual_unique_count"], attention_unique_count=right["actual_unique_count"],
                   equal_counts_between_models=equal)
        for metric in ("reference_coverage_2", "reference_proximity_6"):
            row[metric+"_delta"] = None if left[metric] is None or right[metric] is None else right[metric]-left[metric]
        l, r = left["worst_chain_coverage"]["2"], right["worst_chain_coverage"]["2"]
        row["worst_chain_coverage_2_delta"] = None if l is None or r is None else r-l
        pairs.append(row)
    return {"baseline_summaries": a["summaries"], "attention_summaries": b["summaries"],
            "attention_minus_baseline_pairs": pairs,
            "interpretation": "Matched rows have equal actual N across models; expanded rows may include availability differences. Inspect per-neuron/chain losses, not only means."}


def run(args):
    manifest, cfg, old, initial, digest, extras, protected = load_context(args)
    if args.action == "preflight":
        target = args.experiment_dir/"timing.json"
    elif args.action == "train":
        target = args.experiment_dir/"training_comparison.json"
    else:
        target = args.experiment_dir/"sampling_comparison.json"
    if target.exists():
        raise FileExistsError("Preserve earlier comparison outputs; choose a fresh experiment directory")
    output_name = {"preflight": "timing", "train": "model", "compare": "comparison"}[args.action]
    if any((args.experiment_dir/arm/output_name).exists() for arm in ("baseline", "attention")):
        raise FileExistsError("Preserve existing arm outputs; no partial comparison is restarted automatically")
    if args.action == "train":
        if not (args.experiment_dir/"timing.json").exists():
            raise ValueError("Run the two-arm preflight before training")
        # Validate both timing contracts before the first expensive fit starts.
        for arm in ("baseline", "attention"):
            timing_path = args.experiment_dir/arm/"timing/timing.json"
            timing = json.loads(timing_path.read_text())
            torch.set_num_threads(cfg["torch_threads"])
            backend = training.backend_record(torch.device(args.device))
            if (timing.get("status") != "completed" or timing.get("model_spec") != model_spec(arm, digest)
                    or timing.get("scope") != "timing only; throwaway weights"
                    or timing.get("planned_optimizer_updates") != cfg["epochs"]*sum(r["supervised"] for r in manifest["records"] if r["partition"] == "train")
                    or timing.get("manifest_sha256") != file_digest(args.manifest)
                    or timing.get("config_sha256") != file_digest(args.config)
                    or timing.get("source_files_sha256") != training.source_hashes(extras)
                    or timing.get("backend") != backend):
                raise ValueError("Two-arm preflight is stale; no fit started")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"schema": EXPECTED_RECIPE["schema"], "status": "running", "action": args.action,
              "recipe": EXPECTED_RECIPE, "shared_initialization_sha256": digest,
              "saved_baseline_report_sha256": file_digest(args.baseline_models/"pilot_report.json"),
              "manifest_sha256": file_digest(args.manifest), "arms": {}, "reserved_test_records": 0}
    write_json(target, report)
    try:
        if args.action in ("preflight", "train"):
            for arm in ("baseline", "attention"):
                print(f"\n=== {arm}: {'four throwaway timing updates' if args.action == 'preflight' else 'fresh 30-epoch fit'} ===", flush=True)
                out = args.experiment_dir/arm/("timing" if args.action == "preflight" else "model")
                trial_args = Namespace(manifest=args.manifest, partition=args.partition, config=args.config,
                                       output_dir=out, device=args.device, preflight=args.action == "preflight",
                                       timing=args.experiment_dir/arm/"timing/timing.json")
                report["arms"][arm] = training.run(trial_args, model_factory=initial_factory(arm, initial),
                                                 model_spec=model_spec(arm, digest), extra_sources=extras)
                write_json(target, report)
            if args.action == "preflight":
                total = sum(r["compute_estimate_seconds"] for r in report["arms"].values())
                report.update(estimated_total_compute_minutes=total/60, planning_minutes_with_io_margin=[total/60,total*1.5/60],
                              optimizer_updates_retained=0, throwaway_optimizer_updates=8)
                print(f"Two-arm total estimate: {total/60:.1f}-{total*1.5/60:.1f} minutes; training is a separate command.", flush=True)
            else:
                trial_reports = verify_pair(args, old, digest)
                report.update(final_field_metrics=differences(trial_reports["baseline"], trial_reports["attention"]),
                              fresh_baseline_minus_saved_reference=differences(old, trial_reports["baseline"],
                                  left_label="saved_baseline", right_label="fresh_baseline"),
                              checks={"shared_initial_backbone_exact": True, "same_training_crop_order": True,
                                      "same_training_recipe_and_backend": True, "checkpoint_reload_both_passed": True},
                              optimizer_updates=sum(r["optimizer_updates"] for r in trial_reports.values()))
        else:
            verify_pair(args, old, digest)
            sampling_reports = {}
            for arm in ("baseline", "attention"):
                print(f"\n=== {arm}: fixed epoch-30 sampling; zero updates ===", flush=True)
                trial_args = Namespace(manifest=args.manifest, config=args.config, partition=args.partition,
                                       sampling_config=args.sampling_config, models=args.experiment_dir/arm/"model",
                                       output_dir=args.experiment_dir/arm/"comparison", data_root=args.data_root, device=args.device)
                sampling_reports[arm] = sampling.run(trial_args, model_factory=inference_factory, extra_sources=extras)
                report["arms"][arm] = {"status": "completed", "report": str(trial_args.output_dir/"comparison.json"),
                                       "report_sha256": file_digest(trial_args.output_dir/"comparison.json")}
                write_json(target, report)
            # Image-only baseline points must be identical; their field-score metadata may differ.
            for row in sampling_reports["baseline"]["rows"]:
                if row["arm"] != "threshold_random":
                    continue
                import numpy as np
                with np.load(args.experiment_dir/"baseline/comparison"/row["cloud_path"], allow_pickle=False) as left:
                    with np.load(args.experiment_dir/"attention/comparison"/row["cloud_path"], allow_pickle=False) as right:
                        if not np.array_equal(left["points"], right["points"]):
                            raise AssertionError("Image-only reference cloud changed between models")
            report.update(sampling_metrics=compare_saved_sampling(sampling_reports), optimizer_updates=0,
                          checks={"field_training_pair_verified": True, "sampling_rules_unchanged": True,
                                  "image_only_reference_points_identical": True, "matched_counts_across_models": True})
        training.verify_unchanged(protected)
        report.update(status="completed", elapsed_seconds=time.perf_counter()-started,
                      saved_baseline_and_cache_unchanged=True,
                      next_step="Review per-neuron/branch changes; no automatic attention adoption, selector change or later-stage execution.")
        write_json(target, report)
        print(f"Status: completed; {'timing only, no checkpoint retained' if args.action == 'preflight' else 'paired execution checks passed; quality requires review'}\nreport: {target.resolve()}", flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(target, report)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("preflight", "train", "compare"), required=True)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--baseline-models", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT/"configs/pointneuron2/field_pilot.json")
    p.add_argument("--attention-config", type=Path, default=ROOT/"configs/pointneuron2/field_attention.json")
    p.add_argument("--sampling-config", type=Path, default=ROOT/"configs/pointneuron2/field_sampling_comparison.json")
    p.add_argument("--partition", type=Path, default=ROOT/"configs/pointneuron2/partition_seed0.json")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(p.parse_args())
