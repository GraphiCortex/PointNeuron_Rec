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
from pointneuron.data.training_cache import build_training_record
from pointneuron.data.vaa3d_raw import read_header


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cached PointNeuron training records from Gold166 samples.")
    parser.add_argument("--root", default="data/gold166", help="Path to Gold166 root.")
    parser.add_argument("--output-dir", default="tmp/training_cache", help="Directory for .npz cache files.")
    parser.add_argument("--sample-index", type=int, help="Build one scanned sample by index.")
    parser.add_argument("--max-samples", type=int, help="Build at most this many clean samples.")
    parser.add_argument("--threshold", type=int, default=0, help="Foreground threshold; voxels > threshold become points.")
    parser.add_argument("--max-points", type=int, default=4096, help="Maximum sampled foreground points per record.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    if args.sample_index is not None:
        selected = [(args.sample_index, samples[args.sample_index])]
    else:
        selected = clean_indexed_samples(samples)
        if args.max_samples is not None:
            selected = selected[: args.max_samples]

    output_dir = Path(args.output_dir)
    records = []
    for ordinal, (sample_index, sample) in enumerate(selected):
        output_path = output_dir / f"sample_{sample_index:04d}.npz"
        try:
            record = build_training_record(
                sample,
                output_path=output_path,
                threshold=args.threshold,
                max_points=args.max_points,
                seed=args.seed + ordinal,
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
                "threshold": args.threshold,
                "max_points": args.max_points,
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


if __name__ == "__main__":
    raise SystemExit(main())
