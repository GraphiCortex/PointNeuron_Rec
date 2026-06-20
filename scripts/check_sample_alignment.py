from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_header, read_volume, volume_stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether selected SWC nodes fit inside their raw volume.")
    parser.add_argument("--root", default="data/gold166", help="Path to Gold166 root.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index in the scanned Gold166 samples.")
    parser.add_argument("--all", action="store_true", help="Check every scanned Gold166 sample.")
    parser.add_argument("--decode-volume", action="store_true", help="Decode the raw volume and print intensity stats.")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    if args.all:
        failures = 0
        for index, sample in enumerate(samples):
            failures += 0 if check_sample(sample, index, decode_volume=False, verbose=False) else 1
        print(f"samples_checked: {len(samples)}")
        print(f"alignment_failures: {failures}")
        return 0 if failures == 0 else 2

    sample = samples[args.sample_index]
    return 0 if check_sample(sample, args.sample_index, decode_volume=args.decode_volume, verbose=True) else 2


def check_sample(sample, index: int, decode_volume: bool, verbose: bool) -> bool:
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    header = read_header(sample.volume_path)
    swc = parse_swc(sample.swc_path)
    xs = [node.x for node in swc.nodes]
    ys = [node.y for node in swc.nodes]
    zs = [node.z for node in swc.nodes]
    width, height, depth, channels = header.dimensions

    report = check_swc_in_volume(swc, header)
    out_of_bounds = report.out_of_bounds_node_ids

    if not verbose:
        if out_of_bounds:
            print(f"alignment_failed: index={index} sample_id={sample.sample_id} out_of_bounds_nodes={len(out_of_bounds)}")
        return report.is_aligned

    print(f"sample_id: {sample.sample_id}")
    print(f"volume_path: {sample.volume_path}")
    print(f"swc_path: {sample.swc_path}")
    print(f"swc_selection: {sample.swc_selection}")
    print(f"volume_dimensions: {header.dimensions}")
    print(f"swc_nodes: {len(swc.nodes)}")
    print(f"swc_roots: {swc.root_count}")
    print(f"swc_edges: {swc.edge_count}")
    print(f"swc_x_bounds: ({min(xs):.3f}, {max(xs):.3f})")
    print(f"swc_y_bounds: ({min(ys):.3f}, {max(ys):.3f})")
    print(f"swc_z_bounds: ({min(zs):.3f}, {max(zs):.3f})")
    print(f"out_of_bounds_nodes: {len(out_of_bounds)}")
    if out_of_bounds:
        print(f"out_of_bounds_node_ids: {out_of_bounds[:20]}")

    if decode_volume:
        volume = read_volume(sample.volume_path)
        stats = volume_stats(volume.data)
        for key, value in stats.items():
            print(f"volume_{key}: {value}")

    return report.is_aligned



if __name__ == "__main__":
    raise SystemExit(main())
