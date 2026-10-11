"""Image-only inference for versioned pointwise/query comparison checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
from pointneuron.data.query_cache import file_digest
from pointneuron.data.training_cache import indices_to_point_array, volume_data_array
from pointneuron.data.vaa3d_raw import read_volume
from pointneuron.models.query_pipeline import models_from_checkpoint, forward_proposals, associate_encoder_features
from scripts.aggregate_proposals import build_patch_specs, effective_threshold
from scripts.visualize_proposals import select_proposals


def image_inputs(indices, data, width, height, pipeline, seed, device):
    config = pipeline["input_config"]
    count = config["candidate_points"]
    rng = np.random.default_rng(seed)
    if len(indices) >= count:
        pool = np.sort(rng.choice(indices, count, replace=False))
    else:
        pool = np.sort(np.concatenate((indices, rng.choice(indices, count - len(indices)))))
    points = torch.from_numpy(indices_to_point_array(pool, data, width, height)).to(device)
    generator = torch.Generator(device=device).manual_seed(seed + config["evaluation_seed"])
    chosen = torch.randperm(len(points), generator=generator, device=device)[:config["selected_points"]]
    return points[chosen]


def predict_patch(encoder, head, arm, points, centers, radius):
    features = encoder(points)
    output = forward_proposals(head, arm, points, features, centers, radius)
    associated = associate_encoder_features(output.center_proposals, points, features)
    return output, associated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-patches", type=int, default=256)
    parser.add_argument("--partition", type=Path, default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new proposal output")
    if args.max_patches < 1:
        parser.error("Max patches must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    if file_digest(args.target_manifest) != checkpoint["provenance"]["training_manifest_sha256"]:
        raise ValueError("Target manifest differs from checkpoint provenance")
    if file_digest(args.partition) != checkpoint["provenance"]["partition_sha256"]:
        raise ValueError("Neuron partition differs from checkpoint")
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    allowed = {r["sample_index"]: r["sample_id"] for role in ("train", "val") for r in partition["partitions"][role]}
    if args.sample_index not in allowed:
        raise ValueError("Inference comparison uses train/validation neurons only; test is reserved")
    rows = [r for r in manifest["records"] if r["sample_index"] == args.sample_index]
    if not rows or any(r["sample_id"] != allowed[args.sample_index] for r in rows):
        raise ValueError("Neuron absent from checkpoint target cache")
    record_path = (args.target_manifest.parent / rows[0]["path"]).resolve()
    if not record_path.is_relative_to(args.target_manifest.parent.resolve()) or file_digest(record_path) != rows[0]["sha256"]:
        raise ValueError("Target record content/path mismatch")
    with np.load(record_path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))  # Identity/hashes only; labels never enter prediction.
    directory = (args.data_root / allowed[args.sample_index]).resolve()
    if not directory.is_relative_to(args.data_root.resolve()):
        raise ValueError("Image directory escapes data root")
    volumes = [p for p in sorted(directory.iterdir()) if p.suffix in (".v3dpbd", ".v3draw") and file_digest(p) == meta["volume_sha256"]]
    if not volumes:
        raise ValueError("Source image differs from checkpoint cache")
    encoder, head = models_from_checkpoint(checkpoint, args.device)
    config, pipeline = checkpoint["config"], checkpoint["provenance"]["input_pipeline"]
    inputs = pipeline["input_config"]
    if inputs["selected_points"] != config["selected_points"] or inputs["patch_radius"] != config["patch_radius"]:
        raise ValueError("Input and proposal contracts disagree")
    volume = read_volume(volumes[0]); data = volume_data_array(volume)
    width, height, depth, channels = volume.dimensions
    if channels != 1:
        raise ValueError("Expected single-channel image")
    threshold = effective_threshold(volume.header.datatype, 0, inputs["threshold_fraction"], data=data,
                                    mode=inputs["threshold_mode"])
    foreground = np.flatnonzero(data > threshold)
    specs = build_patch_specs(foreground, data, width, height, depth, inputs["patch_radius"], inputs["stride"],
                              threshold, inputs["min_foreground_points"], args.max_patches, "coverage")
    if not specs:
        raise ValueError("No image-defined inference patches")
    raw, local = [], []
    settings = config["downstream"]
    with torch.no_grad():
        for j, (patch_index, spec) in enumerate(specs):
            seed = inputs["seed"] + args.sample_index * 10000 + patch_index
            points = image_inputs(spec["indices"], data, width, height, pipeline, seed, args.device)[None]
            center = torch.tensor(spec["center"], dtype=points.dtype, device=args.device)[None]
            output, attached = predict_patch(encoder, head, checkpoint["arm"], points, center, config["patch_radius"])
            centers, radii = output.center_proposals[0], output.radius[0, :, 0]
            scores = output.objectness_logits[0].softmax(-1)[:, 1]
            keep = select_proposals(centers=centers, radii=radii, scores=scores, score_threshold=settings["local_score_threshold"],
                top_proposals=settings["local_top"], nms_mode=settings["local_nms_mode"],
                iou_threshold=settings["local_iou_threshold"], nms_radius=settings["local_nms_radius"])
            raw.append((centers.cpu(), radii.cpu(), scores.cpu(), torch.full((len(centers),), patch_index),
                        attached.association_distances[0].cpu()))
            local.append((centers[keep].cpu(), radii[keep].cpu(), scores[keep].cpu(), attached.features[0, keep].cpu(),
                          attached.association_distances[0, keep].cpu(), torch.full((len(keep),), patch_index)))
            print(f"{checkpoint['arm']} patches={j+1}/{len(specs)} raw={len(centers)} local={len(keep)}", flush=True)
    all_centers, all_radii, all_scores, all_features, all_distances, all_patches = [torch.cat([r[i] for r in local]) for i in range(6)]
    keep = select_proposals(centers=all_centers, radii=all_radii, scores=all_scores, score_threshold=0.,
        top_proposals=settings["global_top"], nms_mode="sphere", iou_threshold=settings["global_iou_threshold"],
        nms_radius=settings["global_nms_radius"])
    metadata = {"sample_index": args.sample_index, "sample_id": allowed[args.sample_index], "arm": checkpoint["arm"],
                "checkpoint_sha256": file_digest(args.checkpoint), "target_manifest_sha256": file_digest(args.target_manifest),
                "input_pipeline": pipeline, "downstream": settings, "patches": len(specs), "threshold": threshold,
                "volume_sha256": meta["volume_sha256"], "checkpoint_scope": checkpoint["provenance"]["scope"],
                "feature_policy": "nearest-input encoder feature, both arms", "volume_reader_version": checkpoint["volume_reader_version"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, centers=all_centers[keep].numpy(), radii=all_radii[keep].numpy(),
        scores=all_scores[keep].numpy(), features=all_features[keep].numpy(), patch_indices=all_patches[keep].numpy(),
        association_distances=all_distances[keep].numpy(), all_centers=all_centers.numpy(), all_radii=all_radii.numpy(),
        all_scores=all_scores.numpy(), all_features=all_features.numpy(), all_patch_indices=all_patches.numpy(),
        all_association_distances=all_distances.numpy(),
        raw_centers=torch.cat([r[0] for r in raw]).numpy(), raw_radii=torch.cat([r[1] for r in raw]).numpy(),
        raw_scores=torch.cat([r[2] for r in raw]).numpy(), raw_patch_indices=torch.cat([r[3] for r in raw]).numpy(),
        raw_association_distances=torch.cat([r[4] for r in raw]).numpy(),
        metadata=np.array(json.dumps(metadata)))
    print(f"proposals: {args.output.resolve()}; final={len(keep)}", flush=True)


if __name__ == "__main__":
    main()
