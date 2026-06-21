from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.vaa3d_raw import read_header, read_volume, volume_stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Vaa3D raw or packed volume files.")
    parser.add_argument("--path", help="Path to a .v3dpbd or .v3draw volume.")
    parser.add_argument("--manifest-root", default="data/gold166", help="Dataset root used when --sample-index is set.")
    parser.add_argument("--sample-index", type=int, help="Inspect a sample by index from the Gold166 scan.")
    parser.add_argument("--decode", action="store_true", help="Decode volume bytes and print intensity stats.")
    args = parser.parse_args()

    if args.path:
        volume_path = Path(args.path)
    elif args.sample_index is not None:
        samples = scan_gold166(args.manifest_root)
        sample = samples[args.sample_index]
        if sample.volume_path is None:
            raise ValueError(f"Sample {sample.sample_id} has no volume path")
        volume_path = sample.volume_path
        print(f"sample_id: {sample.sample_id}")
    else:
        parser.error("Provide either --path or --sample-index")

    header = read_header(volume_path)
    print(f"path: {volume_path}")
    print(f"format: {header.key}")
    print(f"datatype: {header.datatype}")
    print(f"dimensions: {header.dimensions}")
    print(f"voxel_count: {header.voxel_count}")

    if args.decode:
        volume = read_volume(volume_path)
        stats = volume_stats(volume.data, datatype=volume.header.datatype, endian=volume.header.endian)
        for key, value in stats.items():
            print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
