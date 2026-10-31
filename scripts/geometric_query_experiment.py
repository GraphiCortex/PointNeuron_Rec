"""Geometric-query candidate: CPU checks and explicit manual GPU timing/fit.

Same GN baseline parameters, clouds, targets, objective, optimizer and cap.
Only query initialization and position parameterization change. No GPU is
launched by cpucheck, and timing cannot automatically start a fit.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

import sampler_node_common as common
import groupnorm_node_experiment as gn
from sampler_node_experiment import launch_worker

ROOT = common.ROOT
CONFIG = ROOT / "configs/geometric_query_fit.json"
VARIANT_FILES = ("scripts/geometric_query_experiment.py", "src/pn2/models/geometric_queries.py")
QUERY_POLICY = {"selection": "native_fps_local_xyz_fixed_input_order", "count": "min_512_N",
    "initial_content": "learned_token_plus_gathered_projected_features_and_position",
    "center": "input_start_plus_unbounded_offset_times_64_voxel",
    "memory": "all_input_points", "public_local_centers": "crop_relative"}


def load_recipe(path):
    control = json.loads(Path(path).read_text())
    fields = {"schema_version", "baseline_config", "baseline_recipe_fingerprint", "baseline_report",
              "baseline_report_sha256", "query_policy", "cpu_preflight", "variant_source_sha256"}
    if (set(control) != fields or control["schema_version"] != 1 or control["query_policy"] != QUERY_POLICY
            or set(control["variant_source_sha256"]) != set(VARIANT_FILES)):
        raise ValueError("Only the declared geometric-query candidate is supported")
    for name, expected in control["variant_source_sha256"].items():
        if common.digest(ROOT / name) != expected:
            raise ValueError(f"Candidate source changed: {name}; update recipe and CPU check")
    baseline = gn.load_recipe(ROOT / control["baseline_config"])
    if common.fingerprint(baseline) != control["baseline_recipe_fingerprint"]:
        raise ValueError("GN baseline recipe changed")
    report = common.checked_json(control["baseline_report"], control["baseline_report_sha256"])
    if (report["status"] != "completed" or report["stage"] != "train" or report["updates"] != 2000
            or not report["normalization_consistency_passed"]
            or any(report.get(k) != v for k, v in gn.identity(baseline).items())):
        raise ValueError("Expected the completed, mode-consistent GN baseline")
    recipe = copy.deepcopy(baseline)
    recipe.update(scope="Geometric-query node learnability candidate against saved GN learned tokens",
        model_variant="pointnext_gn8_geometric_query_v1", cpu_preflight=control["cpu_preflight"],
        geometric_query_control=control)
    return recipe


def identity(recipe):
    return {"recipe_fingerprint": common.fingerprint(recipe),
        "code_identity": {**gn.identity(recipe)["code_identity"], **recipe["geometric_query_control"]["variant_source_sha256"]},
        "upstream_identity": gn.identity(recipe)["upstream_identity"]}


def make_model(recipe, device):
    from pn2.models.geometric_queries import GeometricQueryHead
    baseline = gn.load_recipe(ROOT / recipe["geometric_query_control"]["baseline_config"])
    model = gn.make_model(baseline, "cpu")
    before = common.state_digest(model.state_dict())
    rng = common.nested_digest(common.rng_state())
    names = tuple(name for name, _ in model.named_parameters())
    model.head = GeometricQueryHead(model.head)
    if (common.state_digest(model.state_dict()) != before or tuple(n for n, _ in model.named_parameters()) != names
            or common.nested_digest(common.rng_state()) != rng):
        raise ValueError("Geometric query construction changed fresh parameters/state/RNG")
    return model.to(device)


def require_cpu(recipe, path=None):
    report = json.loads((Path(path) if path else ROOT / recipe["cpu_preflight"]).read_text())
    if (report["status"] != "passed" or not report["normalization_consistency_passed"]
            or not report["checkpoint_reload_passed"] or len(report["records"]) != 4
            or len(report["synthetic_budget_checks"]) != 2
            or any(report.get(k) != v for k, v in identity(recipe).items())):
        raise ValueError("Geometric-query CPU preflight missing, failed or stale")


def synthetic_record(torch, recipe, count, device):
    import probe_pointnext_capacity as probe
    config, _ = probe.load(ROOT / recipe["encoder_config"])
    local, attributes = probe.cloud(torch, count, config, device)
    center = torch.tensor([[63.5]*3], device=device)
    native = local*64.+center[:, None]
    return {"local_xyz": local, "attributes": attributes,
            "points": torch.cat((native, attributes[..., 3:4]*762.), -1),
            "box_center": center, "target": native[0, :3].clone()}


def reload_probe(model, optimizer, recipe, step, device):
    import torch
    expected = gn.prefix_identity(model, optimizer)
    stream = io.BytesIO()
    torch.save(common.checkpoint_payload(model, optimizer, step, recipe, "pn2_sato"), stream)
    stream.seek(0)
    restored = make_model(recipe, device)
    restored_optimizer = common.make_optimizer(restored, recipe)
    loaded = torch.load(stream, map_location="cpu", weights_only=False)
    restored_step = common.restore_payload(loaded, restored, restored_optimizer, recipe, "pn2_sato")
    if restored_step != step or gn.prefix_identity(restored, restored_optimizer) != expected:
        raise ValueError("Geometric model/Adam/RNG reload differs")
    return restored, restored_optimizer


def cpu_check(recipe, folder):
    torch = common.runtime(recipe, "cpu")
    from pn2.models.node_objective import node_set_loss
    from pn2.evaluation.node_proposals import one_to_one
    model = make_model(recipe, "cpu")
    from pointneuron.models.pointnext_ops import gather_points
    baseline = gn.make_model(gn.load_recipe(ROOT / recipe["geometric_query_control"]["baseline_config"]), "cpu")
    if (common.state_digest(model.state_dict()) != common.state_digest(baseline.state_dict())
            or sum(p.numel() for p in model.parameters()) != 1688518):
        raise ValueError("Candidate/baseline initialization or parameter count differs")
    del baseline
    initial = common.state_digest(model.state_dict())
    records = common.load_tasks(recipe)["pn2_sato"]
    report = {"status": "running", "device": "cpu", "torch": torch.__version__, **identity(recipe),
        "parameter_count": sum(p.numel() for p in model.parameters()), "fresh_parameters_exactly_match_GN": True,
        "real_data_optimizer_updates": 0, "records": [], "synthetic_budget_checks": []}
    for record in records:
        data = common.tensors(record, torch, "cpu")
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.train(); training = model(data)
            model.eval(); inference = model(data)
        consistency = gn.prediction_consistency(training, inference)
        if not consistency["passed"]:
            raise ValueError("Geometric train/eval discrepancy")
        model.train(); prediction = model(data)
        loss = node_set_loss(prediction, data["target"], **recipe["loss"])
        loss.total.backward()
        if not loss.supervised or any(p.grad is None or not bool(p.grad.isfinite().all()) for p in model.parameters()):
            raise ValueError("Missing/nonfinite node-loss gradients")
        final = model.head.prediction[-1]
        if (bool((final.weight.grad[2] != 0).any()) or bool(final.bias.grad[2] != 0)
                or not bool((final.weight.grad[:2] != 0).any()) or not bool((final.weight.grad[3:] != 0).any())):
            raise ValueError("Unexpected radius/confidence/geometry gradients")
        if common.state_digest(model.state_dict()) != initial:
            raise ValueError("CPU real-data check changed state")
        indices = model.head.starting_indices((data["points"][..., :3]-data["box_center"][:, None])/64.)
        starts = gather_points(data["points"][..., :3], indices)[0].detach().numpy()
        if len(torch.unique(indices)) != 512:
            raise ValueError("Starting source indices repeat")
        centers = inference.center_proposals[0].detach().numpy()
        scores = inference.objectness_logits[0].softmax(-1)[:, 1].detach().numpy()
        metrics = {label: {str(t): one_to_one(record["target_xyz"], points, t)[0] for t in (2., 6.)}
            for label, points in (("raw_starts", starts), ("untrained_all_predictions", centers),
                                  ("untrained_accepted_predictions", centers[scores >= .5]))}
        meta = record["metadata"]
        packet = folder / f"{meta['alias']}_tile{meta['tile_id']}_untrained.npz"
        np.savez_compressed(packet, start_indices=indices[0].numpy(), starts=starts, centers=centers, scores=scores)
        row = {"alias": meta["alias"], "tile_id": meta["tile_id"], "targets": len(record["target_xyz"]),
            "points": 2048, "queries": 512, "consistency": consistency, "all_gradients_finite": True,
            "radius_row_gradient_zero": True, "state_unchanged": True, "metrics": metrics,
            "initial_loss": float(loss.total.detach()), "prediction_packet": packet.name,
            "assignment_mean_distance_voxel": float(torch.linalg.vector_norm(
                prediction.center_proposals[0, list(loss.prediction_indices)]-
                data["target"][list(loss.target_indices)], dim=-1).mean().detach()),
            "prediction_packet_sha256": common.digest(packet)}
        report["records"].append(row)
        print(f"CPU GEOMETRIC {row['alias']}/{row['tile_id']}: starts={metrics['raw_starts']['2.0']['matched']} actual_untrained={metrics['untrained_all_predictions']['2.0']['matched']}/{row['targets']} train/eval exact; finite gradients; no optimizer", flush=True)
    for count in (256, 4096):
        record = synthetic_record(torch, recipe, count, "cpu")
        with torch.no_grad():
            model.train(); a = model(record)
            model.eval(); b = model(record)
        check = gn.prediction_consistency(a, b)
        if not check["passed"] or a.center_proposals.shape != (1, min(count, 512), 3):
            raise ValueError("Synthetic point-budget interface failed")
        report["synthetic_budget_checks"].append({"points": count, "queries": min(count, 512), "consistency": check})
    if common.state_digest(model.state_dict()) != initial:
        raise ValueError("Untrained checks mutated state")
    # One entirely synthetic throwaway update exercises nonempty Adam state.
    fixture = synthetic_record(torch, recipe, 256, "cpu")
    optimizer = common.make_optimizer(model, recipe)
    common.update(model, optimizer, fixture, recipe)
    restored, _ = reload_probe(model, optimizer, recipe, 1, "cpu")
    with torch.no_grad():
        model.eval(); restored.eval()
        reload_outputs = gn.prediction_consistency(model(fixture), restored(fixture))
    if not reload_outputs["passed"]:
        raise ValueError("Reloaded CPU predictions differ")
    report.update(status="passed", normalization_consistency_passed=True, checkpoint_reload_passed=True,
        synthetic_throwaway_optimizer_updates=1, reload_outputs=reload_outputs,
        initial_state_sha256=recipe["initial_state_sha256"], gpu_executed=False)
    common.write(folder / "report.json", report)
    print(f"PASSED CPU geometric-query preflight: {folder / 'report.json'}", flush=True)


def require_timing(recipe, folder):
    if folder is None:
        raise ValueError("Training requires its own passed geometric-query --timing-run")
    report = json.loads((Path(folder)/"report.json").read_text())
    if (report.get("stage") != "timing" or report.get("status") != "completed" or report.get("updates") != 8
            or not report.get("checkpoint_reload_passed") or not report.get("normalization_consistency_passed")
            or any(report.get(k) != v for k, v in identity(recipe).items())):
        raise ValueError("Expected current-source passed geometric-query timing; GN timing is incompatible")
    return report


def baseline_comparison(recipe, terminal):
    c = recipe["geometric_query_control"]
    baseline = common.checked_json(c["baseline_report"], c["baseline_report_sha256"])
    previous = {(r["alias"], r["tile_id"]): r for r in baseline["evaluations"][-1]["records"]}
    rows = []
    for row in terminal["records"]:
        before = previous[(row["alias"], row["tile_id"])]["modes"]["ordinary_inference"]
        after = row["modes"]["ordinary_inference"]
        if before["target_count"] != after["target_count"]:
            raise ValueError("Geometric/GN target denominator differs")
        rows.append({"alias": row["alias"], "tile_id": row["tile_id"],
            "baseline_GN_ordinary_inference": before, "geometric_query_ordinary_inference": after})
    return {"baseline_learnability_gate": baseline["learnability_gate"],
        "geometric_query_learnability_gate": terminal["learnability_gate"], "records": rows,
        "scope": "Same four training boxes and GN recipe, geometric query mechanism differs; no generalization claim"}


def worker(recipe, args):
    require_cpu(recipe, args.cpu_preflight)
    timing = require_timing(recipe, args.timing_run) if args.stage == "train" else None
    torch = common.runtime(recipe, "cuda:0")
    records = [common.tensors(r, torch, "cuda:0") for r in common.load_tasks(recipe)["pn2_sato"]]
    model = make_model(recipe, "cuda:0")
    optimizer = common.make_optimizer(model, recipe)
    folder = Path(args.output)
    device = torch.cuda.get_device_name(0)
    if timing and (timing["device"] != device or timing["torch"] != torch.__version__):
        raise ValueError("Geometric timing device/runtime differs")
    cap = 8 if args.stage == "timing" else 2000
    report = {"status": "running", "stage": args.stage, "method": "pn2_sato", "device": device,
        "torch": torch.__version__, "cuda_build": torch.version.cuda, **identity(recipe),
        "parameter_count": sum(p.numel() for p in model.parameters()), "cap": cap, "updates": 0, "evaluations": []}
    print(f"{args.stage.upper()} GEOMETRIC: four fixed Sato TRAINING boxes; fresh GN parameters; all 2048 points in memory; cap={cap}; device={device}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    if args.stage == "train":
        common.save_checkpoint(folder/"checkpoint_step000000.pt", model, optimizer, 0, recipe, "pn2_sato")
    report["evaluations"].append(gn.evaluate(model, records, recipe, folder, 0))
    started, durations = time.perf_counter(), []
    with (folder/"updates.jsonl").open("x", encoding="utf-8") as journal:
        for step in range(1, cap+1):
            record = records[(step-1)%4]
            torch.cuda.synchronize(); one = time.perf_counter()
            metrics = common.update(model, optimizer, record, recipe)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-one
            durations.append(elapsed)
            row = {"step": step, "alias": record["metadata"]["alias"], "tile_id": record["metadata"]["tile_id"], "seconds": elapsed, **metrics}
            journal.write(json.dumps(row, allow_nan=False)+"\n"); journal.flush()
            report["updates"] = step
            if args.stage == "timing" or step%50 == 0:
                print(f"UPDATE GEOMETRIC {step}/{cap} {row['alias']}/{row['tile_id']} loss={row['loss']:.5f} class={row['classification']:.5f} center={row['center']:.5f} seconds={elapsed:.3f}", flush=True)
            if step == 8:
                report["prefix_identity"] = gn.prefix_identity(model, optimizer)
                if args.stage == "timing":
                    model, optimizer = reload_probe(model, optimizer, recipe, step, "cuda:0")
                    report["checkpoint_reload_passed"] = True
                elif report["prefix_identity"] != timing["prefix_identity"]:
                    common.write(folder/"prefix_mismatch.json", {"expected": timing["prefix_identity"], "actual": report["prefix_identity"]})
                    raise ValueError("First eight geometric updates do not replay passed timing")
                else:
                    print("PREFIX GEOMETRIC step=8 model/Adam/RNG exactly match passed timing", flush=True)
            if step in recipe["evaluation_steps"]:
                if args.stage == "train":
                    common.save_checkpoint(folder/f"checkpoint_step{step:06d}.pt", model, optimizer, step, recipe, "pn2_sato")
                report["evaluations"].append(gn.evaluate(model, records, recipe, folder, step))
    report.update(status="completed", update_seconds=durations,
        elapsed_after_initial_evaluation_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        normalization_consistency_passed=all(e["normalization_consistency"]["passed"] for e in report["evaluations"]),
        learnability_gate=report["evaluations"][-1]["learnability_gate"] if args.stage == "train" else None)
    if args.stage == "timing":
        mean = float(np.mean(durations[4:]))
        report["estimated_update_minutes"] = [mean*2000*.8/60, mean*2000*1.2/60]
        print(f"GEOMETRIC UPDATE-TIME ESTIMATE minutes={report['estimated_update_minutes']}; excludes evaluation/checkpoint overhead", flush=True)
    else:
        report["baseline_comparison"] = baseline_comparison(recipe, report["evaluations"][-1])
    common.write(folder/"worker_report.json", report)
    print(f"COMPLETED {args.stage} geometric consistency={report['normalization_consistency_passed']} learnability={report['learnability_gate']}", flush=True)


def controller(recipe, args):
    require_cpu(recipe, args.cpu_preflight)
    if args.action == "train":
        require_timing(recipe, args.timing_run)
    common.load_tasks(recipe)
    folder = Path(args.output).resolve(); folder.mkdir(parents=True, exist_ok=False)
    common.write(folder/"recipe.json", recipe)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--action", "_worker",
        "--stage", args.action, "--config", str(Path(args.config).resolve()), "--output", str(folder)]
    for name in ("timing_run", "cpu_preflight"):
        if getattr(args, name):
            command += ["--"+name.replace("_", "-"), str(Path(getattr(args, name)).resolve())]
    code, status = launch_worker(command, folder, recipe[f"{args.action}_worker_timeout_seconds"])
    path = folder/"worker_report.json"
    report = json.loads(path.read_text()) if path.exists() else {"status": "failed_missing_report", "stage": args.action, **identity(recipe)}
    if status != "completed":
        report["status"] = status
    elif (report.get("updates") != (8 if args.action == "timing" else 2000)
            or not report.get("normalization_consistency_passed")
            or (args.action == "timing" and not report.get("checkpoint_reload_passed"))):
        report["status"] = "failed_incomplete_control"
    report["exit_code"] = code
    common.write(folder/"report.json", report)
    print(f"FINISHED geometric {args.action} status={report['status']} report={folder/'report.json'}", flush=True)
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("cpucheck", "timing", "train", "_worker"), required=True)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-run")
    parser.add_argument("--cpu-preflight")
    parser.add_argument("--stage", choices=("timing", "train"))
    args = parser.parse_args()
    recipe = load_recipe(args.config)
    if args.action == "cpucheck":
        folder = Path(args.output).resolve(); folder.mkdir(parents=True, exist_ok=False)
        common.write(folder/"recipe.json", recipe); cpu_check(recipe, folder)
        return 0
    if args.action == "_worker":
        if args.stage is None:
            raise ValueError("Worker stage missing")
        worker(recipe, args); return 0
    return controller(recipe, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
