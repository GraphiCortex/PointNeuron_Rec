"""Derive common curve targets from a corrected cache; never starts training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from pointneuron.data.curve_targets import TARGET_SCHEMA, curve_geometry, patch_targets, target_arrays
from pointneuron.data.query_cache import CACHE_SCHEMA, file_digest
from pointneuron.data.sampling_cache import load_sampling_manifest
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION


def build_cache(source_manifest, partition, data_root, output_dir, config, partitions=("train", "val"), limit=None):
    output_dir, data_root = Path(output_dir), Path(data_root).resolve()
    if output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve earlier caches")
    if not partitions or any(p not in ("train", "val") for p in partitions):
        raise ValueError("Only train/val targets may be prepared; test is reserved")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    source = load_sampling_manifest(source_manifest, partition)
    if source.get("status") != "completed" or source.get("failures"):
        raise ValueError("Corrected source cache must be complete")
    if config["patch_radius"] != source["config"]["patch_radius"]:
        raise ValueError("Patch radius differs from source cache")
    if config["selected_points"] != config["decoder"]["query_count"] or config["target_capacity"] != config["decoder"]["query_count"]:
        raise ValueError("The first comparison requires matched input/output capacities")
    selected = []
    for role in partitions:
        rows = [r for r in source["records"] if r["partition"] == role]
        selected.extend(rows if limit is None else rows[:limit])
    if not selected:
        raise ValueError("No records selected")
    sources, prepared = {}, []
    started = time.perf_counter()
    # Validate all selected geometry and capacity before creating output files.
    for row in selected:
        with np.load(row["resolved_path"], allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            key = (row["sample_index"], row["sample_id"])
            if key not in sources:
                directory = (data_root / row["sample_id"]).resolve()
                if not directory.is_relative_to(data_root):
                    raise ValueError("Source directory escapes data root")
                paths = [p for p in sorted(directory.glob("*.swc")) if file_digest(p) == meta["swc_sha256"]]
                if not paths:
                    raise ValueError("Source SWC hash differs from corrected cache")
                sources[key] = (curve_geometry(parse_swc(paths[0])), meta["swc_sha256"])
            geometry, source_hash = sources[key]
            if source_hash != meta["swc_sha256"]:
                raise ValueError("Source SWC changed across crops")
            targets = patch_targets(geometry, meta["patch_center"], config["patch_radius"], config["target_spacing"])
            nodes, roles = target_arrays(targets, config["target_capacity"])
            curve = patch_targets(geometry, meta["patch_center"], config["patch_radius"], 1.)
            reference = np.array([t["xyz"] for t in curve], dtype=np.float32).reshape(-1, 3)
            metadata = {**meta, "target_schema": TARGET_SCHEMA, "target_spacing": config["target_spacing"],
                        "target_capacity": config["target_capacity"], "source_cache_sha256": row["sha256"],
                        "annotation_policy": "reference-agreement; empty reference is unknown",
                        "target_identities": [t["reference_identity"] for t in targets]}
            prepared.append((row, z["points"].copy(), nodes, roles, reference, metadata))
    manifest = {"schema": CACHE_SCHEMA, "target_schema": TARGET_SCHEMA, "status": "running",
                "volume_reader_version": VOLUME_READER_VERSION, "config": config, "input_config": source["config"],
                "medial_config": source["medial_config"], "partition_sha256": file_digest(partition),
                "source_manifest_sha256": file_digest(source_manifest), "records": [],
                "scope": "limited execution cache" if limit else "internal matched target cache",
                "geometry_source_sha256": file_digest(REPO_ROOT / "src/pointneuron/data/curve_targets.py")}
    output_dir.mkdir(parents=True)
    path = output_dir / "manifest.json"
    try:
        for row, points, nodes, roles, curve, metadata in prepared:
            file = output_dir / row["path"]
            np.savez_compressed(file, points=points, target_nodes=nodes, target_roles=roles,
                                reference_curve=curve, metadata=np.array(json.dumps(metadata)))
            manifest["records"].append({k: row[k] for k in ("path", "sample_index", "sample_id", "partition", "patch_index")})
            manifest["records"][-1].update(sha256=file_digest(file), target_count=len(nodes),
                                          original_label_nodes=row["label_nodes"])
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed_or_interrupted", error=str(exc))
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for role in partitions:
        rows = [r for r in manifest["records"] if r["partition"] == role]
        print(f"{role}: crops={len(rows)} labeled={sum(r['target_count'] > 0 for r in rows)} "
              f"max_targets={max((r['target_count'] for r in rows), default=0)}", flush=True)
    print(f"manifest: {path.resolve()}; CPU preparation only", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pointneuron2/query_comparison.json")
    parser.add_argument("--partition", type=Path, default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--partitions", choices=("train", "val"), nargs="+", default=["train", "val"])
    parser.add_argument("--limit-per-partition", type=int)
    args = parser.parse_args()
    build_cache(args.manifest, args.partition, args.data_root, args.output_dir,
                json.loads(args.config.read_text(encoding="utf-8")), args.partitions, args.limit_per_partition)


if __name__ == "__main__":
    main()
