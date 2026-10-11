"""Compare point representation on cached validation patches, using SWCs only as references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler
from pointneuron.models.sampling_pipeline import sampler_from_checkpoint


def reference_curve(nodes, step=1.):
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Reference resampling step must be positive")
    if not len(nodes):
        return np.empty((0, 3))
    pieces = [nodes[:, 1:4]]
    lookup = {int(node[0]): node[1:4] for node in nodes}
    for node in nodes:
        parent = lookup.get(int(node[5]))
        if parent is not None:
            child = node[1:4]
            intervals = max(1, int(np.ceil(np.linalg.norm(child - parent) / step)))
            pieces.append(np.linspace(parent, child, intervals + 1))
    return np.unique(np.concatenate(pieces), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--partition", type=Path, default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--distance", type=float, default=6.)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.distance) or args.distance <= 0:
        parser.error("Distance must be finite and positive")
    if args.output.exists():
        raise FileExistsError("Preserve earlier reports; choose a new output")
    manifest = load_sampling_manifest(args.manifest, args.partition)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if checkpoint.get("training_manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint and cached pools have different provenance")
    learned = sampler_from_checkpoint(checkpoint, args.device)
    if learned is None:
        raise ValueError("Expected a learned-sampler checkpoint")
    pipeline = checkpoint["sampling_pipeline"]
    fixed = AdaptiveMedialSampler(**pipeline["sampler_config"]).to(args.device).eval()
    spacing = np.asarray(pipeline["medial_config"]["spacing_xyz"])
    rows = []
    with torch.no_grad():
        for record in manifest["records"]:
            if record["partition"] != "val":
                continue
            with np.load(record["resolved_path"], allow_pickle=False) as cached:
                points = torch.tensor(cached["points"], device=args.device)[None]
                guide = torch.tensor(cached["image_guide_xyz"], device=args.device)[None]
                metadata = json.loads(str(cached["metadata"]))
                curve = reference_curve(cached["skeleton_nodes"] * np.array([1, *spacing, 1, 1]))
            count = pipeline["selected_points"]
            seed = metadata["seed"] + pipeline["evaluation_seed"]
            generator = torch.Generator(device=args.device).manual_seed(seed)
            random_indices = torch.randperm(points.shape[1], device=args.device, generator=generator)[:count]
            selections = {"random": points[:, random_indices],
                          "fixed_medial": fixed(points, guide, count, seed=seed).points,
                          "learned": learned(points, guide, count, seed=seed).points}
            row = {"sample_index": record["sample_index"], "patch_index": record["patch_index"],
                   "reference_points": len(curve), "metrics": {}}
            for name, selected in selections.items():
                xyz = selected[0, :, :3].cpu().numpy() * spacing
                if len(curve):
                    distances = cKDTree(curve).query(xyz)[0]
                    coverage = cKDTree(xyz).query(curve)[0]
                    row["metrics"][name] = {"local_reference_precision": float(np.mean(distances <= args.distance)),
                                              "local_reference_coverage": float(np.mean(coverage <= args.distance)),
                                              "mean_reference_distance": float(distances.mean())}
                else:
                    row["metrics"][name] = None
            row["learned_same_points_as_fixed"] = bool(torch.equal(selections["learned"], selections["fixed_medial"]))
            rows.append(row)
    if not rows:
        raise ValueError("No validation patches")
    output = {"scope": "local reference representation diagnostic, not full-neuron reconstruction evaluation",
              "reference_rule": "unique nodes plus step-1 interpolation of edges with both endpoints inside the cached cubic patch",
              "distance_threshold": args.distance, "spacing_xyz": spacing.tolist(),
              "validation_patches": len(rows), "patches_with_reference": sum(r["reference_points"] > 0 for r in rows),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": manifest_hash, "learned_alpha": float(learned.alpha.detach()), "patches": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
