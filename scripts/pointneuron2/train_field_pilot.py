"""Separate timing/training commands; fixed epoch-30 Phase 1 field candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from pointneuron.data.field_pilot import SCHEMA, NativeRecords, field_metrics, load_manifest, macro_metrics, write_json
from pointneuron.data.query_cache import file_digest
from pointneuron.models.centerline_unet import CenterlineUNet, field_loss
from scripts.pointneuron2.prepare_centerline_fit import inspection


def source_hashes():
    paths = [Path(__file__), ROOT / "src/pointneuron/data/field_pilot.py",
             ROOT / "src/pointneuron/models/centerline_unet.py", ROOT / "src/pointneuron/data/centerline_field.py"]
    return {str(p.resolve()): file_digest(p) for p in paths}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def backend_record(device):
    return {"torch_version": str(torch.__version__), "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "training_repeatability": "seed/order/backend flags recorded; cross-run numerical identity not established"}


def verify_unchanged(protected):
    if any(file_digest(p) != digest for p, digest in protected.items()):
        raise AssertionError("Protected pilot source/cache changed")


def run(args):
    manifest, cfg = load_manifest(args.manifest, args.partition, args.config)
    torch.set_num_threads(cfg["torch_threads"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose the device explicitly")
    device = torch.device(args.device)
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory; earlier runs are immutable")
    train_rows = [r for r in manifest["records"] if r["partition"] == "train" and r["supervised"]]
    val_rows = [r for r in manifest["records"] if r["partition"] == "val"]
    all_train_rows = [r for r in manifest["records"] if r["partition"] == "train"]
    protected = {str(p.resolve()): file_digest(p) for p in (args.manifest, args.partition, args.config)}
    protected.update({str((args.manifest.parent/r["path"]).resolve()): r["sha256"] for r in manifest["records"]})
    protected.update(source_hashes())
    total_updates = cfg["epochs"] * len(train_rows)
    if not args.preflight:
        if not args.timing:
            raise ValueError("Run the visible preflight first and pass its timing.json")
        timing = json.loads(args.timing.read_text())
        if (timing.get("status") != "completed" or timing.get("scope") != "timing only; throwaway weights"
                or timing.get("manifest_sha256") != file_digest(args.manifest)
                or timing.get("config_sha256") != file_digest(args.config)
                or timing.get("source_files_sha256") != source_hashes()
                or timing.get("backend") != backend_record(device) or timing.get("planned_optimizer_updates") != total_updates):
            raise ValueError("Preflight differs from the requested cache/config/backend/source; rerun preflight")
        protected[str(args.timing.resolve())] = file_digest(args.timing)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    report_path = args.output_dir / ("timing.json" if args.preflight else "pilot_report.json")
    report = {"schema": SCHEMA, "status": "running", "config": cfg,
              "manifest_sha256": file_digest(args.manifest), "config_sha256": file_digest(args.config),
              "partition_sha256": file_digest(args.partition), "source_files_sha256": source_hashes(),
              "backend": backend_record(device), "planned_optimizer_updates": total_updates,
              "scope": "timing only; throwaway weights" if args.preflight else "train/development patch pilot; no reserved test or final SWC",
              "training_crops": len(all_train_rows), "supervised_training_crops": len(train_rows),
              "development_crops": len(val_rows), "reserved_test_records": 0, "evaluations": []}
    write_json(report_path, report)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.cuda.reset_peak_memory_stats(device)
    model = CenterlineUNet(cfg["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    cache = NativeRecords(args.manifest, cfg["cpu_cache_records"])
    print(f"Device: {report['backend']['device_name']} | native 97-voxel crop boxes | batch=1 float32", flush=True)
    print(f"Training {len(train_rows)} supervised of {len(all_train_rows)} crops; development {len(val_rows)} crops; "
          f"epochs={cfg['epochs']} updates={total_updates}; no augmentation", flush=True)
    print("Existing U-Net/GroupNorm and top min(N,3P) known-negative BCE + masked Dice; no PointNeXt/Transformer/GATv2 execution.", flush=True)
    print(f"Outputs: {args.output_dir.resolve()}", flush=True)

    def update(row):
        record = cache.get(row)
        if not row["supervised"]:
            raise AssertionError("Unsupervised crop entered the optimizer")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        image = torch.from_numpy(record["image"])[None, None].to(device)
        target = torch.from_numpy(record["target"])[None, None].to(device)
        known = torch.from_numpy(record["known"])[None, None].to(device)
        loss = field_loss(model(image), target, known, negative_aggregation=cfg["negative_aggregation"],
                          negative_to_positive_ratio=cfg["negative_to_positive_ratio"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite field loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())

    def inference(row):
        record = cache.get(row)
        model.eval()
        with torch.inference_mode():
            probability = model(torch.from_numpy(record["image"])[None, None].to(device)).sigmoid()[0, 0].cpu().numpy().copy()
        return probability, record

    def measure(row, probability, record):
        return {k: row[k] for k in ("sample_index", "sample_id", "patch_index", "partition")} | field_metrics(probability, record)

    try:
        if args.preflight:
            # Include disk decompression/CPU->GPU transfer in timed update costs.
            print("Four throwaway optimizer updates: two warm-up, two measured. No model checkpoint retained.", flush=True)
            for i in range(2):
                update(train_rows[i % len(train_rows)])
            cache.cache.clear()
            synchronize(device)
            measured_started = time.perf_counter()
            for i in (2, 3):
                update(train_rows[i % len(train_rows)])
            synchronize(device)
            seconds_update = (time.perf_counter()-measured_started)/2
            cache.cache.clear()
            measured_started = time.perf_counter()
            probability, record = inference(val_rows[0])
            measure(val_rows[0], probability, record)
            synchronize(device)
            seconds_eval = time.perf_counter()-measured_started
            crop_evals = len(cfg["snapshot_epochs"]) * (len(all_train_rows)+len(val_rows))
            compute_estimate = total_updates*seconds_update + crop_evals*seconds_eval
            report.update(status="completed", throwaway_optimizer_updates=4,
                          seconds_per_update_including_load=seconds_update, seconds_per_crop_eval=seconds_eval,
                          compute_estimate_seconds=compute_estimate,
                          approximate_minutes_with_io_margin=[compute_estimate/60, compute_estimate*1.5/60],
                          estimate_note="Representative updates/evaluation; checkpoints, final overlays, source verification and sustained GPU conditions add variability. Sampling comparison is separate.")
            print(f"Estimated training/evaluation compute: {compute_estimate/60:.1f} minutes; "
                  f"planning range with I/O margin: {compute_estimate/60:.1f}-{compute_estimate*1.5/60:.1f} minutes.", flush=True)
        else:
            checkpoint_dir = args.output_dir / "checkpoints"
            checkpoint_dir.mkdir()
            rng = np.random.default_rng(cfg["seed"])
            visits, update_count = [], 0
            initial_path = checkpoint_dir / "field_epoch_000.pt"

            def checkpoint(epoch):
                path = checkpoint_dir / f"field_epoch_{epoch:03d}.pt"
                torch.save({"schema": SCHEMA, "epoch": epoch, "optimizer_updates": update_count, "config": cfg,
                            "manifest_sha256": report["manifest_sha256"], "partition_sha256": report["partition_sha256"],
                            "source_files_sha256": report["source_files_sha256"], "backend": report["backend"],
                            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "torch_rng_state": torch.get_rng_state(), "crop_index_order": visits}, path)
                return path

            def evaluate(epoch):
                evaluation = {"epoch": epoch, "optimizer_updates": update_count}
                for role, rows in (("train", all_train_rows), ("val", val_rows)):
                    results = []
                    for number, row in enumerate(rows, 1):
                        probability, record = inference(row)
                        measured = measure(row, probability, record)
                        results.append(measured)
                        # Store full precision final development fields for exact threshold membership.
                        if epoch == cfg["checkpoint_for_sampling"] and role == "val":
                            output = args.output_dir / "development_fields"
                            output.mkdir(exist_ok=True)
                            stem = Path(row["path"]).stem
                            np.savez_compressed(output / (stem + ".npz"), probability=probability,
                                                origin_xyz=np.array(record["metadata"]["origin_xyz"]))
                            inspection(record["image"], record["distance"], record["valid"], record["unknown_bright"],
                                       f"Development neuron {row['sample_index']}, crop {row['patch_index']} | epoch {epoch}",
                                       output / (stem + ".png"), probability)
                        if number == 1 or number % 20 == 0 or number == len(rows):
                            print(f"eval epoch={epoch} role={role} crop={number}/{len(rows)} "
                                  f"neuron={row['sample_index']} patch={row['patch_index']} "
                                  f"cov2={measured['reference_coverage_2']} dice={measured['known_voxel_dice']}", flush=True)
                    evaluation[role] = {"crops": results, **macro_metrics(results)}
                    print(f"epoch={epoch} {role} neuron means: {evaluation[role]['neuron_macro_mean']}", flush=True)
                report["evaluations"].append(evaluation)
                checkpoint(epoch)
                write_json(report_path, report)

            evaluate(0)
            report["initial_checkpoint_sha256"] = file_digest(initial_path)
            report["epoch_losses"] = []
            for epoch in range(1, cfg["epochs"]+1):
                losses = []
                order = rng.permutation(len(train_rows)).tolist()
                for batch, i in enumerate(order, 1):
                    losses.append(update(train_rows[i]))
                    visits.append(i)
                    update_count += 1
                    if batch == 1 or batch % 20 == 0 or batch == len(order):
                        print(f"epoch={epoch}/{cfg['epochs']} train batch={batch}/{len(order)} "
                              f"updates={update_count}/{total_updates} loss={losses[-1]:.5f} "
                              f"elapsed={time.perf_counter()-started:.1f}s "
                              f"cuda_allocated={torch.cuda.memory_allocated(device)/2**20 if device.type == 'cuda' else 0:.1f}MiB", flush=True)
                report["epoch_losses"].append({"epoch": epoch, "mean_loss": float(np.mean(losses))})
                if epoch in cfg["snapshot_epochs"]:
                    evaluate(epoch)
                write_json(report_path, report)
            final_path = checkpoint_dir / f"field_epoch_{cfg['epochs']:03d}.pt"
            cp = torch.load(final_path, map_location=device, weights_only=True)
            reloaded = CenterlineUNet(cfg["base_channels"]).to(device).eval()
            reloaded.load_state_dict(cp["model"], strict=True)
            model.eval()
            # Compare development crops from every neuron, not just one fitted crop.
            checked, max_delta = set(), 0.
            for row in val_rows:
                if row["sample_index"] in checked:
                    continue
                record = cache.get(row)
                x = torch.from_numpy(record["image"])[None, None].to(device)
                with torch.inference_mode():
                    max_delta = max(max_delta, float((model(x)-reloaded(x)).abs().max()))
                checked.add(row["sample_index"])
            if max_delta > 1e-6 or update_count != total_updates:
                raise AssertionError("Checkpoint reload or optimizer accounting failed")
            report.update(status="completed", optimizer_updates=update_count, crop_index_order=visits,
                          checkpoint_reload_max_logit_delta=max_delta, reload_checked_development_neurons=sorted(checked),
                          final_checkpoint_sha256=file_digest(final_path),
                          final_development_fields_sha256={p.name: file_digest(p) for p in sorted((args.output_dir / "development_fields").glob("*.npz"))},
                          checkpoint_selection="fixed final epoch; no development winner selection",
                          next_step="Review train/development fitting and agreement, then run the separate fixed four-control comparison; no automatic promotion.")
        verify_unchanged(protected)
        report.update(elapsed_seconds=time.perf_counter()-started,
                      cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None,
                      bounded_cpu_cache_peak_array_mib=cache.peak_array_bytes/2**20,
                      execution_checks={"sources_and_cache_unchanged": True, "reserved_test_records_absent": True,
                                        "unsupervised_crops_excluded_from_optimizer": True})
        write_json(report_path, report)
        print(f"Status: completed; {'timing only; no checkpoint retained' if args.preflight else 'execution checks passed; quality requires review'}\nreport: {report_path.resolve()}", flush=True)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--partition", type=Path, default=ROOT / "configs/pointneuron2/partition_seed0.json")
    p.add_argument("--config", type=Path, default=ROOT / "configs/pointneuron2/field_pilot.json")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--timing", type=Path)
    p.add_argument("--preflight", action="store_true")
    run(p.parse_args())
