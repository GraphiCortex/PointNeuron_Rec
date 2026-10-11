"""Bounded image-only sampling probe on a neuron in the reserved training split."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.medial_sampling import image_medial_axis, load_medial_config, medial_distances, medial_probabilities
from pointneuron.data.training_cache import flat_indices_to_xyz, indices_to_point_array, sample_indices_fixed, volume_data_array
from pointneuron.data.vaa3d_raw import read_volume
from scripts.aggregate_proposals import build_patch_specs, effective_threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--partition", default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json", type=Path)
    parser.add_argument("--config", default=REPO_ROOT / "configs/pointneuron2/medial_sampling.json", type=Path)
    parser.add_argument("--max-patches", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--patch-radius", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=REPO_ROOT / "tmp/pointneuron2/sampling_probe/sample_0000.json", type=Path)
    args = parser.parse_args()
    if args.max_patches <= 0 or args.max_points <= 0 or args.patch_radius <= 0:
        parser.error("Patch and point budgets must be positive")
    partition = json.loads(Path(args.partition).read_text(encoding="utf-8"))
    # Check both scan index and immutable sample identity before reading voxels.
    train = partition["partitions"]["train"]
    if args.sample_index not in {row["sample_index"] for row in train}:
        parser.error("The implementation probe is restricted to the training partition")
    sample = scan_gold166(args.root)[args.sample_index]
    if not any(row["sample_id"] == sample.sample_id and row["sample_index"] == args.sample_index for row in train):
        raise ValueError("Scanned neuron identity does not match the partition")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = load_medial_config(args.config)
    volume = read_volume(sample.volume_path)
    width, height, depth, _ = volume.dimensions
    data = volume_data_array(volume)
    threshold = effective_threshold(volume.header.datatype, 0, 0.2)
    specs = build_patch_specs(np.flatnonzero(data > threshold), data, width, height, depth,
                             args.patch_radius, args.patch_radius, threshold, 256, args.max_patches, "coverage")
    if not specs:
        raise ValueError("No foreground patches survived")
    rows = []
    for patch_index, spec in specs:
        candidates = spec["indices"]
        start = time.perf_counter()
        guide = image_medial_axis(candidates, width, height, config)
        extraction_seconds = time.perf_counter() - start
        candidate_xyz = flat_indices_to_xyz(candidates, width, height)
        rows.append({"patch_index": patch_index, "candidate_count": len(candidates), "guide_vertices": len(guide),
                     "extraction_seconds": extraction_seconds, "strategies": {}})
        for strategy in ("random", "spatial", "medial"):
            start = time.perf_counter()
            selected = sample_indices_fixed(candidates, args.max_points, np.random.default_rng(args.seed + patch_index),
                                            strategy=strategy, data=data, width=width, height=height, medial_config=config)
            seconds = time.perf_counter() - start
            selected_xyz = flat_indices_to_xyz(selected, width, height)
            to_guide = medial_distances(selected_xyz, guide, config.spacing_xyz)
            to_selected = medial_distances(guide, selected_xyz, config.spacing_xyz)
            rows[-1]["strategies"][strategy] = {
                "sampled_count": len(selected), "unique_count": len(np.unique(selected)),
                "all_on_foreground": bool(np.all(data[selected] > threshold)), "seconds": seconds,
                "mean_medial_distance": float(to_guide.mean()) if len(guide) else None,
                "guide_coverage_within_3": float(np.mean(to_selected <= 3)) if len(guide) else None,
            }
        # Gradient smoke uses a bounded, image-derived pool and guide. No SWC is loaded.
        import torch
        from pointneuron.models.adaptive_sampling import AdaptiveMedialSampler, symmetric_chamfer

        if len(guide) and config.alpha > 0 and config.uniform_fraction < 1:
            torch.manual_seed(args.seed + patch_index)
            pool = np.random.default_rng(args.seed).choice(candidates, min(len(candidates), 1024), replace=False)
            points = torch.from_numpy(indices_to_point_array(pool, data, width, height))[None]
            guide_subset = guide[np.linspace(0, len(guide) - 1, min(len(guide), 1024), dtype=int)]
            guide_tensor = torch.tensor(guide_subset, dtype=points.dtype)[None]
            sampler = AdaptiveMedialSampler(config.alpha, config.uniform_fraction, spacing_xyz=config.spacing_xyz)
            result = sampler(points, guide_tensor, min(64, len(pool)))
            spacing = torch.tensor(config.spacing_xyz, dtype=points.dtype)
            loss = symmetric_chamfer(result.points[..., :3] * spacing, guide_tensor * spacing)
            loss.backward()
            rows[-1]["gradient_smoke"] = {"chamfer": float(loss.detach()), "alpha_gradient": float(sampler.log_alpha.grad),
                                          "finite_gradient": bool(torch.isfinite(sampler.log_alpha.grad))}
        print(f"patch={patch_index} candidates={len(candidates)} guide={len(guide)}", flush=True)
    with Path(sample.volume_path).open("rb") as handle:
        volume_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    payload = {"sample_index": args.sample_index, "sample_id": sample.sample_id, "partition": "train",
               "scope": "image-only execution and guide-relative metrics; no reconstruction quality claim",
               "threshold_fraction": 0.2, "threshold": threshold, "patch_radius": args.patch_radius,
               "max_points": args.max_points, "seed": args.seed, "config": config.to_dict(),
               "kimimaro_version": version("kimimaro"), "torch_version": version("torch"),
               "volume_sha256": volume_sha,
               "partition_sha256": hashlib.sha256(Path(args.partition).read_bytes()).hexdigest(),
               "source_sha256": {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), REPO_ROOT / "src/pointneuron/data/medial_sampling.py",
                                               REPO_ROOT / "src/pointneuron/models/adaptive_sampling.py",
                                               REPO_ROOT / "src/pointneuron/data/training_cache.py",
                                               REPO_ROOT / "scripts/aggregate_proposals.py"]},
               "patches": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
