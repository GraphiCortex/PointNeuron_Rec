"""Jointly train the image-guided sampler and PN 1.0 encoder/proposal head."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pointneuron.data.sampling_cache import SamplingCacheDataset, collate_sampling_records, load_sampling_manifest
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler, masked_chamfer
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead
from pointneuron.models.proposal_loss import conservative_skeleton_proposal_loss


def proposal_objective(output, points, nodes, mask, recipe):
    losses = []
    for b in range(len(points)):
        one = type(output)(**{f.name: getattr(output, f.name)[b:b + 1] for f in fields(output)})
        if not mask[b].any():
            # No local SWC nodes is not a verified background annotation. Keep
            # the image-defined patch, but do not fabricate negative labels.
            losses.append(points[b].sum() * 0)
            continue
        losses.append(conservative_skeleton_proposal_loss(
            one, nodes[b:b + 1], mask[b:b + 1], points[b:b + 1],
            positive_distance=recipe["positive_distance"], radius_scale=recipe["radius_scale"],
            positive_class_weight=recipe["positive_class_weight"],
            objectness_weight=recipe["conservative_objectness_weight"], center_weight=recipe["conservative_center_weight"],
            radius_weight=recipe["conservative_radius_weight"], offset_regularization_weight=recipe["offset_regularization_weight"],
            non_worsen_weight=recipe["non_worsen_weight"], non_worsen_margin=recipe["non_worsen_margin"],
            center_beta=recipe["conservative_center_beta"], non_worsen_beta=recipe["non_worsen_beta"],
            objectness_radius_floor=recipe["objectness_radius_floor"], radius_target_floor=recipe["radius_target_floor"],
            radius_target_mode=recipe["radius_target_mode"], selection_radius_floor=recipe["selection_radius_floor"],
            selection_radius_ceiling=recipe["selection_radius_ceiling"],
        ).total)
    return torch.stack(losses).mean()


@contextmanager
def isolated_sampling_rng(seed, device):
    """Sampler draw count must not alter augmentation or model randomness."""
    device = torch.device(device)
    index = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == 'cuda' else None
    with torch.random.fork_rng(devices=[] if index is None else [index]):
        torch.random.default_generator.manual_seed(seed)
        if index is not None:
            torch.cuda.default_generators[index].manual_seed(seed)
        yield


def state_digest(*modules):
    digest = hashlib.sha256()
    for module in modules:
        for name, value in module.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sampling_objective(supervision, selected, guide, guide_mask, spacing, config, strategy):
    weight = config['chamfer_weight'] if strategy == 'learned' else 0.
    if weight == 0:
        return supervision, None
    scale = 2 * config['patch_radius'] * spacing.max()
    chamfer = masked_chamfer(selected[..., :3] * spacing / scale, guide * spacing / scale,
                             guide_mask, config.get('distance_chunk_size'))
    return supervision + weight * chamfer, chamfer


def augment_batch(points, nodes, node_mask, guide, guide_mask, spacing, generator=None):
    if not torch.allclose(spacing, spacing[0].expand_as(spacing)):
        raise ValueError("Voxel-coordinate rotation augmentation currently requires isotropic spacing")
    points, nodes, guide = points.clone(), nodes.clone(), guide.clone()
    for b in range(len(points)):
        pivot = points[b, :, :3].mean(0)
        angle = torch.rand((), device=points.device, generator=generator) * 2 * torch.pi
        c, s, zero = angle.cos(), angle.sin(), angle.new_zeros(())
        rotation = torch.stack([torch.stack([c, -s, zero]), torch.stack([s, c, zero]),
                                torch.stack([zero, zero, angle.new_ones(())])])
        signs = torch.where(torch.rand(3, device=points.device, generator=generator) < 0.5, -1., 1.)
        rotation = torch.diag(signs) @ rotation
        points[b, :, :3] = (points[b, :, :3] - pivot) @ rotation.T + pivot
        nodes[b, node_mask[b], 1:4] = (nodes[b, node_mask[b], 1:4] - pivot) @ rotation.T + pivot
        guide[b, guide_mask[b]] = (guide[b, guide_mask[b]] - pivot) @ rotation.T + pivot
    return points, nodes, guide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--partition", type=Path, default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--chamfer-weight", type=float, help="Override the guide loss; zero skips its computation")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--log-every", type=int, default=10, help="Print batch progress every N batches, plus first and last")
    parser.add_argument("--strategy", choices=["learned", "random"], default="learned")
    parser.add_argument('--proposal-coordinate-mode', choices=['raw', 'normalized', 'local'],
                        help='Explicit coordinate ablation on the same repaired image cache')
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    manifest = load_sampling_manifest(args.manifest, args.partition)
    if manifest.get("status") != "completed" or manifest.get("failures"):
        parser.error("Cache preparation must complete without recorded failures")
    config = dict(manifest["config"])
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.chamfer_weight is not None:
        config['chamfer_weight'] = args.chamfer_weight
    if not np.isfinite(config['chamfer_weight']) or config['chamfer_weight'] < 0:
        parser.error("Chamfer weight must be finite and nonnegative")
    if args.strategy == 'random':
        config['chamfer_weight'] = 0.
    if config["epochs"] < 1 or config["batch_size"] < 2 or (args.limit_train_batches is not None and args.limit_train_batches < 1):
        parser.error("Epochs/limits must be positive and training batch size >= 2")
    if not 2 <= config["selected_points"] <= config["candidate_points"]:
        parser.error("selected_points must be in [2, candidate_points]")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh run directory")
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    train_rows = [r for r in manifest["records"] if r["partition"] == "train"]
    val_rows = [r for r in manifest["records"] if r["partition"] == "val"]
    if len(train_rows) < 2 or not val_rows:
        parser.error("Need at least two train records and one validation record")
    loaders = {}
    for role, rows in (("train", train_rows), ("val", val_rows)):
        loaders[role] = DataLoader(SamplingCacheDataset([r["resolved_path"] for r in rows]),
                                  batch_size=config["batch_size"], shuffle=(role == "train"),
                                  drop_last=(role == "train" and len(rows) % config["batch_size"] == 1),
                                  generator=torch.Generator().manual_seed(config['seed'] + 1001),
                                  collate_fn=collate_sampling_records, num_workers=0)
    recipe = json.loads((REPO_ROOT / config["recipe"]).read_text(encoding="utf-8"))["training"]
    if args.proposal_coordinate_mode is not None:
        recipe['proposal_coordinate_mode'] = args.proposal_coordinate_mode
    encoder = DGCNNEncoder(k=recipe["k"]).to(args.device)
    proposal = SkeletonProposalHead(in_channels=encoder.output_dim + 3, coordinate_mode=recipe["proposal_coordinate_mode"]).to(args.device)
    if recipe["conservative_init"]:
        proposal.initialize_conservative()
    medial = manifest["medial_config"]
    sampler_config = {"initial_alpha": medial["alpha"], "uniform_fraction": medial["uniform_fraction"],
                      "temperature": config["temperature"], "spacing_xyz": medial["spacing_xyz"]}
    if config.get('distance_chunk_size') is not None:
        sampler_config['distance_chunk_size'] = config['distance_chunk_size']
    sampler = AdaptiveMedialSampler(**sampler_config).to(args.device)
    params = [{"params": list(encoder.parameters()) + list(proposal.parameters()), "lr": recipe["lr"]}]
    if args.strategy == "learned":
        params.append({"params": sampler.parameters(), "lr": config["sampler_lr"], "weight_decay": 0.})
    optimizer = torch.optim.AdamW(params, weight_decay=recipe["weight_decay"])
    spacing = torch.tensor(medial["spacing_xyz"], device=args.device)
    # Full precision for the surrogate gradient; AMP can be ablated separately.
    pipeline = {"schema_version": 1, "volume_reader_version": VOLUME_READER_VERSION, "strategy": args.strategy,
                "threshold_mode": config.get('threshold_mode', 'dtype_fraction'),
                "pool_config": {"candidate_points": config["candidate_points"], "guide_points": config["guide_points"]},
                "medial_config": medial, "sampler_config": sampler_config, "selected_points": config["selected_points"],
                "evaluation_seed": config["evaluation_seed"], "patch_radius": config["patch_radius"],
                "stride": config["stride"], "threshold_fraction": config["threshold_fraction"],
                "min_foreground_points": config["min_foreground_points"]}
    args.output_dir.mkdir(parents=True)
    source_files = sorted((REPO_ROOT / "src/pointneuron").rglob("*.py")) + [Path(__file__)]
    run = {"status": "running", "scope": "limited execution smoke" if args.limit_train_batches else "internal sampler training",
           "config": config, "recipe": recipe, "strategy": args.strategy, "device": args.device,
           "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
           "train_records": len(train_rows), "val_records": len(val_rows), "epochs": [],
           "initial_alpha": float(sampler.alpha.detach()), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
           "initial_model_sha256": state_digest(encoder, proposal),
           "random_streams": {"loader_seed": config['seed'] + 1001,
                              "augmentation_seed": "seed + 2000 + epoch*100000 + batch_index",
                              "sampling_seed": "seed + 3000 + epoch*100000 + batch_index"},
           "partition_sha256": manifest["partition_sha256"], "source_sha256": {
               str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    run_path = args.output_dir / "run.json"
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    guide_description = "full extracted guides" if config['guide_points'] is None else f"guide cap {config['guide_points']}"
    print(f"Device: {run['gpu'] or args.device} | strategy={args.strategy} | chamfer_weight={config['chamfer_weight']} | {guide_description}", flush=True)
    print(f"Training: {len(train_rows)} patches; validation: {len(val_rows)} patches; epochs: {config['epochs']}; "
          f"batch size: {config['batch_size']}; candidates/input: {config['candidate_points']}/{config['selected_points']}", flush=True)
    print(f"Live logs and checkpoint folder: {args.output_dir.resolve()}", flush=True)
    best = float("inf")
    started = time.perf_counter()
    try:
        for epoch in range(1, config["epochs"] + 1):
            epoch_result = {"epoch": epoch}
            for role in ("train", "val"):
                training = role == "train"
                encoder.train(training); proposal.train(training); sampler.train(training)
                total_sum = supervision_sum = chamfer_sum = 0.
                count = labeled_count = 0; nonzero_gradients = 0
                order_digest, augmentation_digest = hashlib.sha256(), hashlib.sha256()
                role_batches = len(loaders[role])
                if training and args.limit_train_batches:
                    role_batches = min(role_batches, args.limit_train_batches)
                role_started = time.perf_counter()
                with torch.set_grad_enabled(training):
                    for batch_index, batch in enumerate(loaders[role]):
                        if training and args.limit_train_batches and batch_index >= args.limit_train_batches:
                            break
                        points, nodes, node_mask, guide, guide_mask = [batch[key].to(args.device) for key in
                            ("points", "skeleton_nodes", "skeleton_mask", "image_guide_xyz", "image_guide_mask")]
                        if training and config["augment"]:
                            generator = torch.Generator(device=args.device).manual_seed(config['seed'] + 2000 + epoch * 100000 + batch_index)
                            points, nodes, guide = augment_batch(points, nodes, node_mask, guide, guide_mask, spacing, generator)
                        order_digest.update(json.dumps([m['seed'] for m in batch['metadata']]).encode())
                        augmentation_digest.update(points.detach().cpu().contiguous().numpy().tobytes())
                        with isolated_sampling_rng(config['seed'] + 3000 + epoch * 100000 + batch_index, args.device):
                            if args.strategy == "learned" and training:
                                selected = sampler(points, guide, config["selected_points"], guide_mask).points
                            else:
                                selected_records = []
                                for b, metadata in enumerate(batch["metadata"]):
                                    seed = metadata["seed"] + config["evaluation_seed"]
                                    if args.strategy == "learned":
                                        selected_records.append(sampler(points[b:b + 1], guide[b:b + 1], config["selected_points"],
                                                                       guide_mask[b:b + 1], seed=seed).points)
                                    else:
                                        generator = None if training else torch.Generator(device=args.device).manual_seed(seed)
                                        chosen = torch.randperm(points.shape[1], device=args.device, generator=generator)[:config["selected_points"]]
                                        selected_records.append(points[b:b + 1, chosen])
                                selected = torch.cat(selected_records)
                        output = proposal(selected, encoder(selected))
                        supervision = proposal_objective(output, selected, nodes, node_mask, recipe)
                        total, chamfer = sampling_objective(supervision, selected, guide, guide_mask, spacing, config, args.strategy)
                        if not torch.isfinite(total):
                            raise FloatingPointError("Nonfinite training/validation loss")
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                            # A zero loss through the straight-through points can
                            # still require grad. Do not advance Adam momentum on
                            # a wholly unannotated batch when the guide loss is off.
                            update = total.requires_grad and (bool(node_mask.any()) or config['chamfer_weight'] > 0)
                            if update:
                                total.backward()
                            if args.strategy == "learned" and sampler.log_alpha.grad is not None:
                                if not torch.isfinite(sampler.log_alpha.grad):
                                    raise FloatingPointError("Nonfinite sampler gradient")
                                nonzero_gradients += int(sampler.log_alpha.grad.abs() > 0)
                            if any(p.grad is not None and not torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group["params"]):
                                raise FloatingPointError("Nonfinite proposal/encoder gradient")
                            torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 10.)
                            if update:
                                optimizer.step()
                        n = len(points); count += n
                        labeled_count += int(node_mask.any(dim=1).sum())
                        total_sum += float(total.detach()) * n; supervision_sum += float(supervision.detach()) * n
                        if chamfer is not None:
                            chamfer_sum += float(chamfer.detach()) * n
                        step = batch_index + 1
                        if step == 1 or step % args.log_every == 0 or step == role_batches:
                            memory = (f" cuda_allocated={torch.cuda.memory_allocated() / 2**20:.1f}MiB"
                                      if args.device == 'cuda' else '')
                            print(f"epoch={epoch}/{config['epochs']} {role} batch={step}/{role_batches} "
                                  f"loss={float(total.detach()):.4f} proposal={float(supervision.detach()):.4f} "
                                  f"chamfer={float(chamfer.detach()) if chamfer is not None else 'off'} alpha={float(sampler.alpha.detach()):.6f} "
                                  f"elapsed={time.perf_counter() - role_started:.1f}s{memory}", flush=True)
                if not count:
                    raise ValueError("No records evaluated")
                if not labeled_count:
                    raise ValueError(f"No annotated reference nodes evaluated in {role}")
                epoch_result[role] = {"total": total_sum / count, "proposal": supervision_sum / labeled_count,
                                      "normalized_chamfer": chamfer_sum / count if config['chamfer_weight'] > 0 else None, "records": count,
                                      "batch_order_sha256": order_digest.hexdigest(),
                                      "augmented_candidates_sha256": augmentation_digest.hexdigest(),
                                      "annotated_records": labeled_count,
                                      "nonzero_sampler_gradient_batches": nonzero_gradients}
            epoch_result["alpha"] = float(sampler.alpha.detach())
            run["epochs"].append(epoch_result)
            # Select using the common proposal objective, not a different auxiliary-loss total.
            if epoch_result["val"]["proposal"] < best:
                best = epoch_result["val"]["proposal"]
                torch.save({"encoder": encoder.state_dict(), "proposal": proposal.state_dict(),
                            "sampler": sampler.state_dict() if args.strategy == "learned" else None,
                            "sampling_pipeline": pipeline, "args": recipe, "epoch": epoch, "val_loss": best,
                            "training_manifest_sha256": run["manifest_sha256"]}, args.output_dir / "proposal.pt")
            run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
            print(f"epoch={epoch} train_proposal={epoch_result['train']['proposal']:.4f} "
                  f"val_proposal={epoch_result['val']['proposal']:.4f} best_val_proposal={best:.4f} "
                  f"alpha={epoch_result['alpha']:.6f}", flush=True)
        run["status"] = "completed"
    except BaseException as exc:
        run.update(status="failed_or_interrupted", error=str(exc))
        raise
    finally:
        run["elapsed_seconds"] = time.perf_counter() - started
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"checkpoint: {args.output_dir / 'proposal.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
