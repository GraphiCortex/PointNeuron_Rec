from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.point_cloud import point_cloud_stats, swc_to_skeleton_records, volume_to_point_cloud
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_header, read_volume


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a thresholded point cloud for one Gold166 sample.")
    parser.add_argument("--root", default="data/gold166", help="Path to Gold166 root.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index in the scanned Gold166 samples.")
    parser.add_argument("--threshold", type=int, default=0, help="Foreground threshold; voxels > threshold become points.")
    parser.add_argument("--max-points", type=int, help="Optional random downsample size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used with --max-points.")
    parser.add_argument("--output", help="Optional CSV path for exported sampled points.")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    sample = samples[args.sample_index]
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    header = read_header(sample.volume_path)
    swc = parse_swc(sample.swc_path)
    report = check_swc_in_volume(swc, header)
    if not report.is_aligned:
        print(f"sample_id: {sample.sample_id}")
        print(f"out_of_bounds_nodes: {len(report.out_of_bounds_node_ids)}")
        print("Refusing to build point cloud for misaligned sample. Pick a clean sample or handle the label policy first.")
        return 2

    volume = read_volume(sample.volume_path)
    point_cloud = volume_to_point_cloud(
        volume,
        threshold=args.threshold,
        max_points=args.max_points,
        seed=args.seed,
    )
    skeleton = swc_to_skeleton_records(swc)

    print(f"sample_id: {sample.sample_id}")
    print(f"volume_dimensions: {point_cloud.volume_dimensions}")
    print(f"threshold: {point_cloud.threshold}")
    for key, value in point_cloud_stats(point_cloud).items():
        print(f"{key}: {value}")
    print(f"skeleton_nodes: {len(skeleton)}")
    print(f"skeleton_edges: {sum(1 for node in skeleton if node.parent_id != -1)}")

    if args.output:
        write_point_csv(point_cloud, Path(args.output))
        print(f"output: {args.output}")

    return 0


def write_point_csv(point_cloud, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z", "intensity"])
        for point in point_cloud.points:
            writer.writerow([point.x, point.y, point.z, point.intensity])


if __name__ == "__main__":
    raise SystemExit(main())

