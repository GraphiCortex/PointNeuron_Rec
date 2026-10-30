"""One normalization-only GN8 control against the completed Sato/BN fit.

Fresh model; unchanged targets/clouds/query head/objective/optimizer/cap. CPU
verification runs here; GPU timing/fit remain explicit manual actions.
The original sampler runner/common files are never modified or monkey-patched.
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
from sampler_node_experiment import launch_worker, verify_upstream

ROOT = common.ROOT
CONFIG = ROOT / "configs/groupnorm_node_fit.json"
VARIANT_FILES = ("scripts/groupnorm_node_experiment.py", "src/pn2/models/groupnorm_encoder.py")


def load_recipe(path):
    control = json.loads(Path(path).read_text())
    fields = {"schema_version", "baseline_recipe", "baseline_recipe_fingerprint", "baseline_report",
              "baseline_report_sha256", "method", "normalization", "initial_state_sha256",
              "cpu_preflight", "variant_source_sha256"}
    if (set(control) != fields or control["schema_version"] != 1 or control["method"] != "pn2_sato"
            or control["normalization"] != {"kind": "GroupNorm", "groups": 8, "eps": 1e-5, "layers": 16}
            or set(control["variant_source_sha256"]) != set(VARIANT_FILES)):
        raise ValueError("Only the declared GN8 normalization control is supported")
    for name, expected in control["variant_source_sha256"].items():
        if common.digest(ROOT / name) != expected:
            raise ValueError(f"Variant source changed: {name}; update the prospective configuration and CPU check")
    baseline = common.load_recipe(ROOT / control["baseline_recipe"])
    if common.fingerprint(baseline) != control["baseline_recipe_fingerprint"]:
        raise ValueError("Baseline learning recipe changed")
    baseline_run = common.checked_json(control["baseline_report"], control["baseline_report_sha256"])
    if (baseline_run["status"] != "completed" or baseline_run["stage"] != "train"
            or baseline_run["recipe_fingerprint"] != common.fingerprint(baseline)
            or baseline_run["code_identity"] != common.code_identity()):
        raise ValueError("Completed baseline/source does not match")
    arm = next(r for r in baseline_run["records"] if r["method"] == "pn2_sato")
    if arm["status"] != "completed" or arm["updates"] != 2000:
        raise ValueError("Expected the completed 2000-update Sato baseline")
    recipe = copy.deepcopy(baseline)
    recipe.update(scope="GN8 normalization-only node learnability control against saved Sato/BN",
        model_variant="pointnext_s_native_groupnorm8_query_control_v1",
        methods=["pn2_sato"], normalization="groupnorm8_input_statistics_train_and_eval",
        diagnostic_normalization="same_groupnorm_computation_in_both_modes_exact_consistency_required",
        initial_state_sha256=control["initial_state_sha256"], cpu_preflight=control["cpu_preflight"],
        normalization_control=control)
    return recipe


def identity(recipe):
    return {"recipe_fingerprint": common.fingerprint(recipe),
            "code_identity": {**common.code_identity(), **recipe["normalization_control"]["variant_source_sha256"]},
            "upstream_identity": verify_upstream(recipe)}


def make_model(recipe, device):
    from pn2.models.groupnorm_encoder import replace_fresh_encoder_batchnorm
    baseline = common.load_recipe(ROOT / recipe["normalization_control"]["baseline_recipe"])
    model = common.make_model(baseline, "cpu")
    expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    manifest = replace_fresh_encoder_batchnorm(model.encoder)
    if (set(expected) != {n for n, _ in model.named_parameters()}
            or any(not parameter.detach().equal(expected[name]) for name, parameter in model.named_parameters())):
        raise ValueError("Normalizer replacement changed fresh parameter values or names")
    if common.audited_state_digest(model.state_dict()) != recipe["initial_state_sha256"]:
        raise ValueError("GN initialization differs from declared buffer-free baseline parameters")
    if list(model.buffers()):
        raise ValueError("Unexpected running/state buffers in the GN control")
    model.normalizer_manifest = manifest
    return model.to(device)


def prediction_consistency(training, inference):
    fields = ("center_proposals", "local_centers", "objectness_logits", "radius", "decoder_features")
    result = {}
    for name in fields:
        a, b = getattr(training, name).detach(), getattr(inference, name).detach()
        if not bool(a.isfinite().all()) or not bool(b.isfinite().all()):
            raise ValueError("Nonfinite consistency-check output")
        result[name] = {"exact": bool(a.equal(b)), "maximum_absolute_difference": float((a-b).abs().max())}
    return {"passed": all(r["exact"] for r in result.values()), "fields": result}


def require_cpu(recipe, path=None):
    report = json.loads((Path(path) if path else ROOT / recipe["cpu_preflight"]).read_text())
    if (report["status"] != "passed" or not report["normalization_consistency_passed"]
            or len(report["records"]) != 4 or len(report["synthetic_budget_checks"]) != 2
            or any(report.get(k) != v for k, v in identity(recipe).items())):
        raise ValueError("GN CPU check is missing, failed or stale")


def cpu_check(recipe, output):
    torch = common.runtime(recipe, "cpu")
    from pn2.models.node_objective import node_set_loss
    records = common.load_tasks(recipe)["pn2_sato"]
    model = make_model(recipe, "cpu")
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_hash = common.state_digest(initial)
    report = {"status": "running", "device": "cpu", "torch": torch.__version__, **identity(recipe),
              "normalizer_manifest": model.normalizer_manifest,
              "parameter_count": sum(p.numel() for p in model.parameters()), "records": [],
              "synthetic_budget_checks": [], "real_data_optimizer_updates": 0}
    for record in records:
        record = common.tensors(record, torch, "cpu")
        model.load_state_dict(initial, strict=True); model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.train(); training = model(record)
            model.eval(); inference = model(record)
        consistency = prediction_consistency(training, inference)
        if not consistency["passed"] or common.state_digest(model.state_dict()) != initial_hash:
            raise ValueError("Real-input train/eval computation or state differs")
        model.train()
        prediction = model(record)
        loss = node_set_loss(prediction, record["target"], **recipe["loss"])
        loss.total.backward()
        if not loss.supervised or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("Node-loss gradients failed")
        final = model.head.prediction[-1]
        if bool((final.weight.grad[2] != 0).any()) or bool(final.bias.grad[2] != 0):
            raise ValueError("Radius output has direct loss gradients")
        if common.state_digest(model.state_dict()) != initial_hash:
            raise ValueError("CPU check changed model state")
        row = {"alias": record["metadata"]["alias"], "tile_id": record["metadata"]["tile_id"],
               "targets": len(record["target"]), "loss": float(loss.total.detach()),
               "consistency": consistency, "all_parameter_gradients_finite": True,
               "radius_row_gradient_zero": True, "state_unchanged": True}
        report["records"].append(row)
        print(f"CPU GN8 {row['alias']}/{row['tile_id']}: train/eval exact; node gradients finite; no optimizer", flush=True)
    # Execute the whole model on two synthetic grid budgets; not extra data or fits.
    import probe_pointnext_capacity as probe
    encoder_recipe, _ = probe.load(ROOT / recipe["encoder_config"])
    for count in (256, 4096):
        local, attributes = probe.cloud(torch, count, encoder_recipe, "cpu")
        center = torch.tensor([[63.5, 63.5, 63.5]])
        native = local*64.+center[:, None]
        record = {"local_xyz": local, "attributes": attributes,
                  "points": torch.cat((native, attributes[..., 3:4]*762.), -1), "box_center": center}
        with torch.no_grad():
            model.train(); a = model(record)
            model.eval(); b = model(record)
        consistency = prediction_consistency(a, b)
        if not consistency["passed"] or common.state_digest(model.state_dict()) != initial_hash:
            raise ValueError("Synthetic budget consistency/state failed")
        report["synthetic_budget_checks"].append({"points": count, "consistency": consistency})
        print(f"CPU GN8 synthetic N={count}: train/eval exact; state unchanged", flush=True)
    report.update(status="passed", normalization_consistency_passed=True,
                  initial_state_sha256=recipe["initial_state_sha256"])
    common.write(output / "report.json", report)
    print(f"PASSED CPU GN8 preflight: {output / 'report.json'}", flush=True)


def evaluate(model, records, recipe, output, step):
    result = common.evaluate(model, records, recipe, output, step)
    path = output / result["prediction_packet"]
    rows = []
    with np.load(path, allow_pickle=False) as data:
        for i, record in enumerate(records):
            fields = {}
            for name in ("centers", "scores", "logits"):
                a, b = (data[f"box{i}_{mode}_{name}"] for mode in ("ordinary_inference", "current_crop_diagnostic"))
                fields[name] = {"exact": bool(np.array_equal(a, b)), "maximum_absolute_difference": float(np.abs(a-b).max())}
            rows.append({"alias": record["metadata"]["alias"], "tile_id": record["metadata"]["tile_id"], "fields": fields})
    consistency = {"passed": all(v["exact"] for row in rows for v in row["fields"].values()), "records": rows}
    common.write(output / f"normalization_consistency_step{step:06d}.json", consistency)
    if not consistency["passed"]:
        raise ValueError("GN train/eval consistency failed; stop without substituting a policy")
    result["normalization_consistency"] = consistency
    return result


def prefix_identity(model, optimizer):
    return {"model_state_sha256": common.state_digest(model.state_dict()),
            "optimizer_state_sha256": common.nested_digest(optimizer.state_dict()),
            "rng_sha256": common.nested_digest(common.rng_state())}


def require_timing(recipe, folder):
    if folder is None:
        raise ValueError("GN training needs its own passed --timing-run; BN timing is not compatible")
    report = json.loads((Path(folder) / "report.json").read_text())
    if (report["stage"] != "timing" or report["status"] != "completed"
            or any(report.get(k) != v for k, v in identity(recipe).items())
            or not report.get("checkpoint_reload_passed") or not report.get("normalization_consistency_passed")
            or report["updates"] != 8):
        raise ValueError("Expected completed current-source GN timing/reload/consistency")
    return report


def reload_probe(model, optimizer, recipe, step):
    import torch
    expected = prefix_identity(model, optimizer)
    stream = io.BytesIO()
    torch.save(common.checkpoint_payload(model, optimizer, step, recipe, "pn2_sato"), stream)
    stream.seek(0)
    restored = make_model(recipe, "cuda:0")
    restored_optimizer = common.make_optimizer(restored, recipe)
    restored_step = common.restore_payload(torch.load(stream, map_location="cpu", weights_only=False), restored,
                                          restored_optimizer, recipe, "pn2_sato")
    if restored_step != step or prefix_identity(restored, restored_optimizer) != expected:
        raise ValueError("Exact GN model/Adam/RNG reload failed")
    return restored, restored_optimizer


def baseline_comparison(recipe, terminal):
    control = recipe["normalization_control"]
    baseline = common.checked_json(control["baseline_report"], control["baseline_report_sha256"])
    arm = next(r for r in baseline["records"] if r["method"] == "pn2_sato")
    previous = {(r["alias"], r["tile_id"]): r for r in arm["evaluations"][-1]["records"]}
    rows = []
    for record in terminal["records"]:
        before = previous[(record["alias"], record["tile_id"])]["modes"]["ordinary_inference"]
        after = record["modes"]["ordinary_inference"]
        if before["target_count"] != after["target_count"]:
            raise ValueError("GN/baseline target denominator differs")
        rows.append({"alias": record["alias"], "tile_id": record["tile_id"],
                     "baseline_ordinary_inference": before, "groupnorm_ordinary_inference": after})
    return {"baseline_learnability_gate": arm["learnability_gate"],
            "groupnorm_learnability_gate": terminal["learnability_gate"], "records": rows,
            "scope": "Same four training boxes and recipe except normalization; no generalization claim"}


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
        raise ValueError("GN timing device/runtime differs")
    cap = 8 if args.stage == "timing" else 2000
    report = {"status": "running", "stage": args.stage, "method": "pn2_sato", "normalization": recipe["normalization"],
              "device": device, "torch": torch.__version__, "cuda_build": torch.version.cuda,
              **identity(recipe), "cap": cap, "updates": 0, "evaluations": []}
    print(f"{args.stage.upper()} GN8: four frozen Sato TRAINING boxes; fresh parameters; cap={cap}; device={device}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    if args.stage == "train":
        common.save_checkpoint(folder / "checkpoint_step000000.pt", model, optimizer, 0, recipe, "pn2_sato")
    report["evaluations"].append(evaluate(model, records, recipe, folder, 0))
    started, durations = time.perf_counter(), []
    with (folder / "updates.jsonl").open("x", encoding="utf-8") as journal:
        for step in range(1, cap+1):
            record = records[(step-1) % 4]
            torch.cuda.synchronize(); one = time.perf_counter()
            metrics = common.update(model, optimizer, record, recipe)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-one
            durations.append(elapsed)
            row = {"step": step, "alias": record["metadata"]["alias"], "tile_id": record["metadata"]["tile_id"],
                   "seconds": elapsed, **metrics}
            journal.write(json.dumps(row, allow_nan=False)+"\n"); journal.flush()
            report["updates"] = step
            if args.stage == "timing" or step % 50 == 0:
                print(f"UPDATE GN8 {step}/{cap} {row['alias']}/{row['tile_id']} loss={row['loss']:.5f} class={row['classification']:.5f} center={row['center']:.5f} seconds={elapsed:.3f}", flush=True)
            if step == 8:
                report["prefix_identity"] = prefix_identity(model, optimizer)
                if args.stage == "timing":
                    model, optimizer = reload_probe(model, optimizer, recipe, step)
                    report["checkpoint_reload_passed"] = True
                elif report["prefix_identity"] != timing["prefix_identity"]:
                    common.write(folder / "prefix_mismatch.json", {"expected": timing["prefix_identity"], "actual": report["prefix_identity"]})
                    raise ValueError("GN first eight updates do not replay GN timing; stop")
                else:
                    print("PREFIX GN8 step=8 model/Adam/RNG exactly match passed GN timing", flush=True)
            if step in recipe["evaluation_steps"]:
                if args.stage == "train":
                    common.save_checkpoint(folder / f"checkpoint_step{step:06d}.pt", model, optimizer, step, recipe, "pn2_sato")
                report["evaluations"].append(evaluate(model, records, recipe, folder, step))
    report.update(status="completed", update_seconds=durations, elapsed_after_initial_evaluation_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        normalization_consistency_passed=all(e["normalization_consistency"]["passed"] for e in report["evaluations"]),
        learnability_gate=report["evaluations"][-1]["learnability_gate"] if args.stage == "train" else None)
    if args.stage == "timing":
        mean = float(np.mean(durations[4:]))
        report["estimated_update_minutes"] = [mean*2000*.8/60, mean*2000*1.2/60]
        print(f"GN8 UPDATE-TIME ESTIMATE minutes={report['estimated_update_minutes']}; excludes evaluation/checkpoint overhead", flush=True)
    else:
        report["baseline_comparison"] = baseline_comparison(recipe, report["evaluations"][-1])
    common.write(folder / "worker_report.json", report)
    print(f"COMPLETED {args.stage} GN8 consistency={report['normalization_consistency_passed']} learnability={report['learnability_gate']}", flush=True)


def controller(recipe, args):
    require_cpu(recipe, args.cpu_preflight)
    if args.action == "train":
        require_timing(recipe, args.timing_run)
    common.load_tasks(recipe)
    folder = Path(args.output).resolve(); folder.mkdir(parents=True, exist_ok=False)
    common.write(folder / "recipe.json", recipe)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--action", "_worker",
               "--stage", args.action, "--config", str(Path(args.config).resolve()), "--output", str(folder)]
    for name in ("timing_run", "cpu_preflight"):
        if getattr(args, name):
            command += ["--"+name.replace("_", "-"), str(Path(getattr(args, name)).resolve())]
    code, status = launch_worker(command, folder, recipe[f"{args.action}_worker_timeout_seconds"])
    path = folder / "worker_report.json"
    report = json.loads(path.read_text()) if path.exists() else {"status": "failed_missing_report", "stage": args.action, **identity(recipe)}
    if status != "completed":
        report["status"] = status
    elif (report.get("updates") != (8 if args.action == "timing" else 2000)
            or not report.get("normalization_consistency_passed")
            or (args.action == "timing" and not report.get("checkpoint_reload_passed"))):
        report["status"] = "failed_incomplete_control"
    report["exit_code"] = code
    common.write(folder / "report.json", report)
    print(f"FINISHED GN8 {args.action} status={report['status']} report={folder / 'report.json'}", flush=True)
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
        common.write(folder / "recipe.json", recipe); cpu_check(recipe, folder)
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
