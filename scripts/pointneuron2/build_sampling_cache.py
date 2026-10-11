"""Cache bounded image-only candidates, full guides and separate SWC labels."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.medial_sampling import load_medial_config
from pointneuron.data.sampling_cache import SamplingPoolConfig, build_image_pools
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import skeleton_edge_index_from_array, skeleton_to_array, volume_data_array
from pointneuron.data.point_cloud import swc_to_skeleton_records
from pointneuron.data.vaa3d_raw import read_volume, VOLUME_READER_VERSION
from scripts.aggregate_proposals import build_patch_specs, effective_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pointneuron2/sampling_training.json")
    parser.add_argument("--partition", type=Path, default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--sample-index", action="append", type=int)
    parser.add_argument("--max-patches", type=int, help="Override per-neuron patch cap for an execution smoke")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_patches is not None:
        config["max_patches_per_neuron"] = args.max_patches
    pools = SamplingPoolConfig(config["candidate_points"], config["guide_points"])
    if min(config["patch_radius"], config["stride"], config["max_patches_per_neuron"], config["min_foreground_points"]) <= 0:
        parser.error("Patch sizes/counts must be positive")
    medial = load_medial_config(REPO_ROOT / config["medial_config"])
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    identities = {row["sample_index"]: (role, row["sample_id"])
                  for role, rows in partition["partitions"].items() for row in rows}
    requested = sorted(set(args.sample_index)) if args.sample_index else sorted(i for i, (role, _) in identities.items() if role in ("train", "val"))
    if not requested or any(i not in identities or identities[i][0] == "test" for i in requested):
        parser.error("Only neurons in the train/val partition may be cached; test is reserved")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; caches and earlier runs are immutable")
    samples = scan_gold166(args.root)
    for i in requested:
        if samples[i].sample_id != identities[i][1]:
            raise ValueError("Dataset scan identity differs from the frozen partition")
    args.output_dir.mkdir(parents=True)
    schema = 2 if pools.guide_points is None else 1
    manifest = {"status": "running", "schema_version": schema, "volume_reader_version": VOLUME_READER_VERSION,
                "config": config, "medial_config": medial.to_dict(),
                "partition_sha256": hashlib.sha256(args.partition.read_bytes()).hexdigest(),
                "kimimaro_version": version("kimimaro"), "requested_sample_indices": requested,
                "scope": "image-defined patches; SWC labels separate; no GT patch filtering", "records": [], "failures": []}
    manifest_path = args.output_dir / "manifest.json"
    for i in requested:
        try:
            sample = samples[i]
            volume = read_volume(sample.volume_path)
            swc = parse_swc(sample.swc_path)
            if not check_swc_in_volume(swc, volume.header).is_aligned:
                raise ValueError("Reference SWC has out-of-bounds nodes")
            skeleton = skeleton_to_array(swc_to_skeleton_records(swc))
            width, height, depth, channels = volume.dimensions
            if channels != 1:
                raise ValueError("Expected a single image channel")
            data = volume_data_array(volume)
            threshold = effective_threshold(volume.header.datatype, 0, config["threshold_fraction"], data=data,
                                            mode=config.get('threshold_mode', 'dtype_fraction'))
            if threshold >= float(data.max()):
                raise ValueError(f'Foreground cutoff {threshold} exceeds decoded signal range 0..{data.max()}. '
                                 'Use an image-derived intensity policy; storage datatype is not acquisition bit depth.')
            specs = build_patch_specs(np.flatnonzero(data > threshold), data, width, height, depth,
                                      config["patch_radius"], config["stride"], threshold,
                                      config["min_foreground_points"], config["max_patches_per_neuron"], "coverage")
            if not specs:
                raise ValueError("No foreground patches survived")
            with sample.volume_path.open("rb") as handle:
                image_sha = hashlib.file_digest(handle, "sha256").hexdigest()
            swc_sha = hashlib.sha256(sample.swc_path.read_bytes()).hexdigest()
            for patch_index, spec in specs:
                seed = config["seed"] + i * 10000 + patch_index
                arrays = build_image_pools(spec["indices"], data, width, height, pools, medial, seed)
                # Cubic image patch and label crop agree; labels never choose inputs.
                labels = skeleton[np.all(np.abs(skeleton[:, 1:4] - spec["center"]) <= config["patch_radius"], axis=1)]
                metadata = {"sampling_cache_schema": schema, "volume_reader_version": VOLUME_READER_VERSION,
                            "sample_index": i, "sample_id": sample.sample_id,
                            "partition": identities[i][0], "patch_index": patch_index, "seed": seed,
                            "config": config, "medial_config": medial.to_dict(), "threshold": threshold,
                            "patch_center": spec["center"].tolist(), "volume_sha256": image_sha, "swc_sha256": swc_sha,
                            "foreground_count": arrays.pop("foreground_count"), "full_guide_count": arrays.pop("full_guide_count")}
                path = args.output_dir / f"sample_{i:04d}_patch_{patch_index:03d}.npz"
                np.savez_compressed(path, **arrays, skeleton_nodes=labels,
                                    edge_index=skeleton_edge_index_from_array(labels), metadata=np.array(json.dumps(metadata)))
                manifest["records"].append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                             "sample_index": i, "sample_id": sample.sample_id, "partition": identities[i][0],
                                             "patch_index": patch_index, "guide_vertices": len(arrays["image_guide_xyz"]),
                                             "label_nodes": len(labels)})
            print(f"cached neuron={i} partition={identities[i][0]} patches={len(specs)}", flush=True)
        except Exception as exc:
            manifest["failures"].append({"sample_index": i, "error": str(exc), "error_type": type(exc).__name__})
            print(f"FAILED neuron={i}: {exc}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["status"] = "failed" if manifest["failures"] else "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
