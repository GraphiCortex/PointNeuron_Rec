from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import PatchCacheConfig, build_patch_training_records, build_training_record
from pointneuron.data.vaa3d_raw import read_header


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cached PointNeuron training records from Gold166 samples.")
    parser.add_argument("--root", default="data/gold166", help="Path to Gold166 root.")
    parser.add_argument("--output-dir", default="tmp/training_cache", help="Directory for .npz cache files.")
    parser.add_argument("--sample-index", type=int, action="append", help="Build one scanned sample by index. May be repeated.")
    parser.add_argument("--max-samples", type=int, help="Build at most this many clean samples.")
    parser.add_argument("--threshold", type=int, default=0, help="Foreground threshold; voxels > threshold become points.")
    parser.add_argument("--threshold-fraction", type=float, help="Normalized foreground threshold in [0, 1], e.g. 0.2 from the paper.")
    parser.add_argument("--max-points", type=int, default=4096, help="Maximum sampled foreground points per record.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing cache records with matching metadata.")
    parser.add_argument("--patches-per-sample", type=int, default=0, help="Build this many local SWC-centered patch records per sample.")
    parser.add_argument("--patch-radius", type=int, default=96, help="Patch radius in voxels when --patches-per-sample is set.")
    parser.add_argument("--min-points", type=int, default=256, help="Minimum foreground points required to keep a patch.")
    parser.add_argument("--min-unique-fraction", type=float, default=0.0, help="Minimum unique foreground count as a fraction of --max-points for patch records.")
    parser.add_argument("--center-strategy", default="random", choices=["random", "topology", "foreground"], help="Patch center sampling strategy for patch records.")
    parser.add_argument("--endpoint-fraction", type=float, default=0.25, help="Fraction of patch centers reserved for SWC endpoints when --center-strategy topology.")
    parser.add_argument("--branch-fraction", type=float, default=0.10, help="Fraction of patch centers reserved for SWC branch nodes when --center-strategy topology.")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    if args.sample_index is not None:
        selected = [(index, samples[index]) for index in dict.fromkeys(args.sample_index)]
    else:
        selected = clean_indexed_samples(samples)
        if args.max_samples is not None:
            selected = selected[: args.max_samples]

    output_dir = Path(args.output_dir)
    records = []
    for ordinal, (sample_index, sample) in enumerate(selected):
        threshold = effective_threshold(sample, args.threshold, args.threshold_fraction)
        if args.patches_per_sample > 0:
            reused = []
            if args.resume:
                reused = cached_patch_summaries(
                    output_dir=output_dir,
                    sample_index=sample_index,
                    threshold=threshold,
                    max_points=args.max_points,
                    patch_radius=args.patch_radius,
                    threshold_fraction=args.threshold_fraction,
                    min_unique_fraction=args.min_unique_fraction,
                    center_strategy=args.center_strategy,
                    endpoint_fraction=args.endpoint_fraction,
                    branch_fraction=args.branch_fraction,
                )
            if reused:
                records.extend(reused)
                print(f"reused_patches index={sample_index} count={len(reused)}")
                continue

            try:
                patch_records = build_patch_training_records(
                    sample,
                    output_dir=output_dir,
                    sample_index=sample_index,
                    config=PatchCacheConfig(
                        patches_per_sample=args.patches_per_sample,
                        patch_radius=args.patch_radius,
                        max_points=args.max_points,
                        min_points=args.min_points,
                        min_unique_fraction=args.min_unique_fraction,
                        center_strategy=args.center_strategy,
                        endpoint_fraction=args.endpoint_fraction,
                        branch_fraction=args.branch_fraction,
                    ),
                    threshold=threshold,
                    seed=args.seed + ordinal,
                    threshold_fraction=args.threshold_fraction,
                )
            except ValueError as exc:
                print(f"skipped index={sample_index}: {exc}")
                continue
            for record in patch_records:
                records.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": record.sample_id,
                        "path": str(record.output_path),
                        "points": record.point_count,
                        "skeleton_nodes": record.skeleton_node_count,
                        "edges": record.edge_count,
                        "total_foreground_points": record.total_foreground_count,
                    }
                )
            print(f"cached_patches index={sample_index} count={len(patch_records)}")
            continue

        output_path = output_dir / f"sample_{sample_index:04d}.npz"
        if args.resume and output_path.exists():
            cached = cached_record_summary(
                output_path,
                sample_index=sample_index,
                threshold=threshold,
                max_points=args.max_points,
                threshold_fraction=args.threshold_fraction,
            )
            if cached is not None:
                records.append(cached)
                print(
                    f"reused index={sample_index} points={cached['points']} "
                    f"skeleton_nodes={cached['skeleton_nodes']} edges={cached['edges']} path={output_path}"
                )
                continue

        try:
            record = build_training_record(
                sample,
                output_path=output_path,
                threshold=threshold,
                max_points=args.max_points,
                seed=args.seed + ordinal,
                threshold_fraction=args.threshold_fraction,
            )
        except ValueError as exc:
            print(f"skipped index={sample_index}: {exc}")
            continue
        records.append(
            {
                "sample_index": sample_index,
                "sample_id": record.sample_id,
                "path": str(record.output_path),
                "points": record.point_count,
                "skeleton_nodes": record.skeleton_node_count,
                "edges": record.edge_count,
                "total_foreground_points": record.total_foreground_count,
            }
        )
        print(
            f"cached index={sample_index} points={record.point_count} "
            f"skeleton_nodes={record.skeleton_node_count} edges={record.edge_count} path={record.output_path}"
        )

    manifest_path = output_dir / "cache_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "root": args.root,
                "threshold": args.threshold if args.threshold_fraction is None else None,
                "threshold_fraction": args.threshold_fraction,
                "max_points": args.max_points,
                "patches_per_sample": args.patches_per_sample,
                "patch_radius": args.patch_radius if args.patches_per_sample > 0 else None,
                "min_unique_fraction": args.min_unique_fraction if args.patches_per_sample > 0 else None,
                "center_strategy": args.center_strategy if args.patches_per_sample > 0 else None,
                "endpoint_fraction": args.endpoint_fraction if args.patches_per_sample > 0 else None,
                "branch_fraction": args.branch_fraction if args.patches_per_sample > 0 else None,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"cache_manifest: {manifest_path}")
    print(f"records_built: {len(records)}")
    return 0


def clean_indexed_samples(samples):
    clean = []
    for index, sample in enumerate(samples):
        if sample.volume_path is None:
            continue
        header = read_header(sample.volume_path)
        swc = parse_swc(sample.swc_path)
        if check_swc_in_volume(swc, header).is_aligned:
            clean.append((index, sample))
    return clean


def cached_record_summary(
    path: Path,
    sample_index: int,
    threshold: int,
    max_points: int,
    threshold_fraction: float | None = None,
) -> dict[str, object] | None:
    try:
        import numpy as np

        record = np.load(path, allow_pickle=False)
        metadata = json.loads(str(record["metadata"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None

    if metadata.get("threshold") != threshold or metadata.get("max_points") != max_points:
        return None
    if metadata.get("threshold_fraction") != threshold_fraction:
        return None

    return {
        "sample_index": sample_index,
        "sample_id": metadata.get("sample_id", ""),
        "path": str(path),
        "points": int(record["points"].shape[0]),
        "skeleton_nodes": int(record["skeleton_nodes"].shape[0]),
        "edges": int(record["edge_index"].shape[0]),
        "total_foreground_points": int(metadata.get("total_foreground_count", 0)),
    }


def cached_patch_summaries(
    output_dir: Path,
    sample_index: int,
    threshold: int,
    max_points: int,
    patch_radius: int,
    threshold_fraction: float | None = None,
    min_unique_fraction: float = 0.0,
    center_strategy: str = "random",
    endpoint_fraction: float = 0.25,
    branch_fraction: float = 0.10,
) -> list[dict[str, object]]:
    paths = sorted(output_dir.glob(f"sample_{sample_index:04d}_patch_*.npz"))
    records = []
    for path in paths:
        summary = cached_record_summary(
            path,
            sample_index=sample_index,
            threshold=threshold,
            max_points=max_points,
            threshold_fraction=threshold_fraction,
        )
        if summary is None:
            return []
        try:
            import numpy as np

            record = np.load(path, allow_pickle=False)
            metadata = json.loads(str(record["metadata"]))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return []
        if metadata.get("patch_radius") != patch_radius:
            return []
        if metadata.get("min_unique_fraction", 0.0) != min_unique_fraction:
            return []
        if metadata.get("center_strategy", "random") != center_strategy:
            return []
        if metadata.get("endpoint_fraction", 0.25) != endpoint_fraction:
            return []
        if metadata.get("branch_fraction", 0.10) != branch_fraction:
            return []
        records.append(summary)
    return records


def effective_threshold(sample, threshold: int, threshold_fraction: float | None) -> int:
    if threshold_fraction is None:
        return threshold
    if threshold_fraction < 0.0 or threshold_fraction > 1.0:
        raise ValueError(f"--threshold-fraction must be in [0, 1], got {threshold_fraction}")
    if sample.volume_path is None:
        return threshold
    header = read_header(sample.volume_path)
    if header.datatype == 1:
        max_value = 255
    elif header.datatype == 2:
        max_value = 65535
    else:
        raise NotImplementedError(f"Vaa3D datatype {header.datatype} is not supported yet")
    return int(round(max_value * threshold_fraction))


if __name__ == "__main__":
    raise SystemExit(main())
