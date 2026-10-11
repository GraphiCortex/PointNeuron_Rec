"""Matched proposal training, or a bounded fixed-crop execution check."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from pointneuron.data.query_cache import QueryCacheDataset, collate_query_records, load_query_manifest, file_digest
from pointneuron.models.query_loss import query_set_loss, pointwise_comparison_loss
from pointneuron.models.query_pipeline import (build_models, forward_proposals, comparison_checkpoint,
                                               models_from_checkpoint, associate_encoder_features)


def state_digest(module):
    result = hashlib.sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def select_inputs(points, metadata, count, seed, device):
    selected = []
    for i, meta in enumerate(metadata):
        one_seed = seed + int(meta["seed"])
        generator = torch.Generator(device=device).manual_seed(one_seed)
        chosen = torch.randperm(points.shape[1], generator=generator, device=device)[:count]
        selected.append(points[i, chosen])
    return torch.stack(selected)


def augment(points, nodes, mask, centers, seed):
    points, nodes, centers = points.clone(), nodes.clone(), centers.clone()
    generator = torch.Generator(device=points.device).manual_seed(seed)
    for i in range(len(points)):
        pivot = points[i, :, :3].mean(0)
        angle = torch.rand((), device=points.device, generator=generator) * 2 * torch.pi
        c, s, z = angle.cos(), angle.sin(), angle.new_zeros(())
        rotation = torch.stack([torch.stack([c, -s, z]), torch.stack([s, c, z]),
                                torch.stack([z, z, angle.new_ones(())])])
        signs = torch.where(torch.rand(3, device=points.device, generator=generator) < .5, -1., 1.)
        rotation = torch.diag(signs) @ rotation
        points[i, :, :3] = (points[i, :, :3] - pivot) @ rotation.T + pivot
        nodes[i, mask[i], 1:4] = (nodes[i, mask[i], 1:4] - pivot) @ rotation.T + pivot
        centers[i] = (centers[i] - pivot) @ rotation.T + pivot
    return points, nodes, centers


def objective(arm, output, points, nodes, mask, config, recipe):
    if arm == "query":
        return query_set_loss(output, nodes, mask, **config["query_loss"])
    return pointwise_comparison_loss(output, points, nodes, mask, recipe)


def apply_update(loss, optimizer, parameters):
    optimizer.zero_grad(set_to_none=True)
    if not loss.supervised_crops:
        return False
    if not bool(torch.isfinite(loss.total)):
        raise FloatingPointError("Nonfinite supervised loss")
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
    optimizer.step()
    return True


def prepare_batch(batch, config, device, sample_seed, augmentation_seed=None):
    points, nodes, mask, centers = [batch[k].to(device) for k in
                                    ("points", "skeleton_nodes", "skeleton_mask", "patch_centers")]
    if augmentation_seed is not None:
        points, nodes, centers = augment(points, nodes, mask, centers, augmentation_seed)
    selected = select_inputs(points, batch["metadata"], config["selected_points"], sample_seed, device)
    return selected, nodes, mask, centers


def inspect_fit(output, nodes, mask):
    values, coverage = [], []
    for i in range(len(nodes)):
        if bool(mask[i].any()):
            distances = torch.cdist(nodes[i, mask[i], 1:4], output.center_proposals[i]).min(1).values
            values.append(float(distances.mean())); coverage.append(float((distances <= 6).float().mean()))
    return {"mean_target_distance": float(np.mean(values)), "target_coverage_6": float(np.mean(coverage)),
            "note": "all-slot target support on fitted training crops; excludes confidence and is not validation"}


def inspect_selection(output, nodes, mask):
    scores = output.objectness_logits.softmax(-1)[..., 1]
    accepted, coverage = [], []
    for i in range(len(nodes)):
        if not bool(mask[i].any()):
            continue
        keep = scores[i] >= .5
        accepted.append(int(keep.sum()))
        if bool(keep.any()):
            distances = torch.cdist(nodes[i, mask[i], 1:4], output.center_proposals[i, keep]).min(1).values
            coverage.append(float((distances <= 6).float().mean()))
        else:
            coverage.append(0.)
    return {"score_threshold": .5, "accepted_counts": accepted,
            "target_coverage_6": float(np.mean(coverage)),
            "note": "raw score filtering on fitted training crops; no suppression or validation"}


def prepare_run_config(base, *, seed=None, decoder_dropout=None, smoke=False):
    config = copy.deepcopy(base)
    if seed is not None:
        config["seed"] = seed
    if decoder_dropout is not None:
        if not smoke:
            raise ValueError("Decoder dropout override is restricted to the fixed-crop check")
        if isinstance(decoder_dropout, bool) or not math.isfinite(decoder_dropout) or not 0 <= decoder_dropout < 1:
            raise ValueError("Decoder dropout must be finite and in [0,1)")
        config["decoder"]["dropout"] = float(decoder_dropout)
    return config


def run_arm(arm, args, manifest, config, recipe, output_dir):
    encoder, head = build_models(config, arm, config["seed"])
    initial_encoder = state_digest(encoder)
    encoder, head = encoder.to(args.device), head.to(args.device)
    parameters = [*encoder.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    torch.manual_seed(config["seed"] + 4001)
    rows = {role: [r for r in manifest["records"] if r["partition"] == role] for role in ("train", "val")}
    report = {"arm": arm, "status": "running", "scope": "fixed-training-crop execution check" if args.smoke else "matched internal pilot",
              "device": args.device, "initial_encoder_sha256": initial_encoder,
              "manifest_sha256": file_digest(args.manifest), "partition_sha256": manifest["partition_sha256"],
              "parameter_count": sum(p.numel() for p in parameters), "updates": 0, "epochs": [],
              "config": config, "pointwise_recipe": recipe,
              "source_sha256": {str(p.relative_to(REPO_ROOT)): file_digest(p) for p in
                                sorted((REPO_ROOT / "src/pointneuron").rglob("*.py")) + [Path(__file__)]}}
    output_dir.mkdir()
    path = output_dir / "run.json"
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"{arm}: GPU={torch.cuda.get_device_name()} parameters={report['parameter_count']:,}", flush=True)
    started = time.perf_counter()
    pipeline = {"input_config": manifest["input_config"], "medial_config": manifest["medial_config"]}

    def save(epoch, filename):
        torch.save(comparison_checkpoint(encoder, head, config, arm, epoch,
                    scope=report["scope"], training_manifest_sha256=report["manifest_sha256"],
                    partition_sha256=report["partition_sha256"], input_pipeline=pipeline,
                    initial_encoder_sha256=initial_encoder), output_dir / filename)

    try:
        if args.smoke:
            chosen = [r for r in rows["train"] if r["target_count"] > 0][:2]
            if len(chosen) != 2:
                raise ValueError("Execution check needs two annotated training crops")
            batch = collate_query_records([QueryCacheDataset(chosen)[i] for i in range(2)])
            points, nodes, mask, centers = prepare_batch(batch, config, args.device, 0)
            report["crop_paths"] = [r["path"] for r in chosen]
            report["inputs_sha256"] = hashlib.sha256(points.cpu().numpy().tobytes()).hexdigest()
            encoder.eval(); head.eval()
            with torch.no_grad():
                first_output = forward_proposals(head, arm, points, encoder(points), centers, config["patch_radius"])
                report["initial_fit"] = inspect_fit(first_output, nodes, mask)
            times, losses = [], []
            for step in range(args.steps):
                tick = time.perf_counter()
                encoder.train(); head.train()
                output = forward_proposals(head, arm, points, encoder(points), centers, config["patch_radius"])
                loss = objective(arm, output, points, nodes, mask, config, recipe)
                report["updates"] += apply_update(loss, optimizer, parameters)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - tick)
                losses.append({"total": float(loss.total.detach()), "center": float(loss.center),
                               "objectness": float(loss.objectness), "radius": float(loss.radius)})
                if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
                    print(f"{arm} fixed-crop step={step+1}/{args.steps} loss={losses[-1]['total']:.4f} "
                          f"center={losses[-1]['center']:.4f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            encoder.eval(); head.eval()
            with torch.no_grad():
                final_output = forward_proposals(head, arm, points, encoder(points), centers, config["patch_radius"])
                report["final_fit"] = inspect_fit(final_output, nodes, mask)
                report["final_selection"] = inspect_selection(final_output, nodes, mask)
                adapter = associate_encoder_features(final_output.center_proposals, points, encoder(points))
                report["adapter_features_shape"] = list(adapter.features.shape)
            print(f"{arm} final eval: mean_target_distance={report['final_fit']['mean_target_distance']:.4f} "
                  f"coverage_6={report['final_fit']['target_coverage_6']:.4f} "
                  f"accepted_at_0.5={report['final_selection']['accepted_counts']} "
                  f"filtered_coverage_6={report['final_selection']['target_coverage_6']:.4f}", flush=True)
            save(0, "proposal.pt")
            restored_encoder, restored_head = models_from_checkpoint(torch.load(output_dir / "proposal.pt", map_location="cpu", weights_only=True), args.device)
            with torch.no_grad():
                restored = forward_proposals(restored_head, arm, points, restored_encoder(points), centers, config["patch_radius"])
            for name in ("center_proposals", "objectness_logits", "radius"):
                torch.testing.assert_close(getattr(restored, name), getattr(final_output, name), rtol=0, atol=0)
            report.update(losses=losses, roundtrip="exact", median_update_seconds=float(np.median(times[1:] or times)))
        else:
            if manifest.get("scope") != "internal matched target cache" or not rows["train"] or not rows["val"]:
                raise ValueError("Full paired training requires an untruncated train/validation target cache")
            loaders = {role: DataLoader(QueryCacheDataset(records), batch_size=config["batch_size"],
                          shuffle=(role == "train"), num_workers=0,
                          drop_last=(role == "train" and len(records) % config["batch_size"] == 1),
                          generator=torch.Generator().manual_seed(config["seed"] + 1001),
                          collate_fn=collate_query_records) for role, records in rows.items()}
            for epoch in range(1, config["epochs"] + 1):
                record = {"epoch": epoch}
                for role in ("train", "val"):
                    training = role == "train"
                    encoder.train(training); head.train(training)
                    labeled, count, total, updates = 0, 0, 0., 0
                    order, inputs = hashlib.sha256(), hashlib.sha256()
                    tick = time.perf_counter()
                    for j, batch in enumerate(loaders[role]):
                        points, nodes, mask, centers = prepare_batch(batch, config, args.device,
                            config["seed"] + 3000 + epoch * 100000 + j if training else 0,
                            config["seed"] + 2000 + epoch * 100000 + j if training and config["augment"] else None)
                        order.update(json.dumps(batch["paths"]).encode())
                        inputs.update(points.detach().cpu().numpy().tobytes())
                        count += len(points)
                        supervised = int(mask.any(1).sum())
                        if not supervised:
                            continue  # Neither optimizer nor BatchNorm advances on all-unknown batches.
                        with torch.set_grad_enabled(training):
                            output = forward_proposals(head, arm, points, encoder(points), centers, config["patch_radius"])
                            loss = objective(arm, output, points, nodes, mask, config, recipe)
                            if not bool(torch.isfinite(loss.total)):
                                raise FloatingPointError("Nonfinite objective")
                            if training:
                                updates += apply_update(loss, optimizer, parameters)
                        labeled += supervised; total += float(loss.total.detach()) * supervised
                        if j == 0 or (j + 1) % args.log_every == 0 or j + 1 == len(loaders[role]):
                            print(f"{arm} epoch={epoch}/{config['epochs']} {role} batch={j+1}/{len(loaders[role])} "
                                  f"loss={float(loss.total.detach()):.4f} center={float(loss.center):.4f} "
                                  f"elapsed={time.perf_counter()-tick:.1f}s", flush=True)
                    if not labeled:
                        raise ValueError(f"No supervised crops in {role}")
                    record[role] = {"loss": total / labeled, "records": count, "supervised_crops": labeled,
                                    "updates": updates, "batch_order_sha256": order.hexdigest(),
                                    "selected_inputs_sha256": inputs.hexdigest()}
                    report["updates"] += updates
                report["epochs"].append(record)
                if epoch in config["snapshot_epochs"] or epoch == config["epochs"]:
                    save(epoch, f"epoch_{epoch:03d}.pt")
                path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            save(config["epochs"], "proposal.pt")  # Fixed final epoch; no cross-objective best-loss selection.
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def check_pairs(reports):
    first = reports[0]
    for report in reports:
        for key in ("manifest_sha256", "partition_sha256", "initial_encoder_sha256"):
            if report[key] != first[key]:
                raise ValueError(f"Unmatched arms: {key}")
        if report["scope"] == "fixed-training-crop execution check":
            if report["inputs_sha256"] != first["inputs_sha256"] or report["updates"] != first["updates"]:
                raise ValueError("Unmatched execution-check inputs/updates")
        else:
            if len(report["epochs"]) != len(first["epochs"]):
                raise ValueError("Unmatched epoch counts")
            for actual, expected in zip(report["epochs"], first["epochs"]):
                for role in ("train", "val"):
                    for key in ("records", "supervised_crops", "updates", "batch_order_sha256", "selected_inputs_sha256"):
                        if actual[role][key] != expected[role][key]:
                            raise ValueError(f"Unmatched {role} {key} epoch={actual['epoch']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--partition", type=Path, default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--decoder-dropout", type=float, help="Fixed-crop control only; leave the cache and base config unchanged")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.log_every, args.cpu_threads) < 1:
        parser.error("Steps, log frequency and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    torch.set_num_threads(args.cpu_threads)
    manifest = load_query_manifest(args.manifest, args.partition)
    config = prepare_run_config(manifest["config"], seed=args.seed, decoder_dropout=args.decoder_dropout, smoke=args.smoke)
    recipe = json.loads((REPO_ROOT / config["pointwise_recipe"]).read_text(encoding="utf-8"))["training"]
    if config["batch_size"] < 2 or config["epochs"] < 1:
        raise ValueError("Training requires batch size >=2 and positive epochs")
    if not args.smoke and manifest["input_config"].get("evaluation_seed", 0) != 0:
        raise ValueError("This runner requires the declared zero evaluation seed")
    args.output_dir.mkdir(parents=True)
    print(f"Scope: {'fixed two-crop execution check' if args.smoke else 'paired training'}; "
          f"device={args.device}; decoder_dropout={config['decoder']['dropout']}; outputs={args.output_dir.resolve()}", flush=True)
    reports = []
    summary = {"status": "running", "config": config, "arms": []}
    try:
        for arm in ("pointwise", "query"):
            reports.append(run_arm(arm, args, manifest, config, recipe, args.output_dir / arm))
        check_pairs(reports)
        summary.update(status="completed", paired_checks="passed", arms=reports)
    except BaseException as exc:
        summary.update(status="failed_or_interrupted", error=str(exc), arms=reports)
        raise
    finally:
        (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Status: completed; paired checks: passed; report: {args.output_dir / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
