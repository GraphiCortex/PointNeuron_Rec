from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import manifest_summary, scan_gold166, write_manifest
from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_header


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate a Gold166 sample manifest.")
    parser.add_argument("--root", default="data/gold166", help="Path to the Gold166 dataset root.")
    parser.add_argument("--output", default="tmp/gold166_manifest.json", help="Manifest JSON output path.")
    parser.add_argument("--include-without-volume", action="store_true", help="Include SWC-only folders in the manifest.")
    parser.add_argument("--include-invalid-swc", action="store_true", help="Include samples even when every SWC is structurally invalid.")
    parser.add_argument("--require-aligned", action="store_true", help="Exclude samples whose selected SWC has nodes outside the raw volume bounds.")
    parser.add_argument("--validate-swc", action="store_true", help="Parse selected SWCs and report structural errors.")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    samples = scan_gold166(
        root,
        include_without_volume=args.include_without_volume,
        include_invalid_swc=args.include_invalid_swc,
    )
    if args.require_aligned:
        aligned_samples = []
        for sample in samples:
            if sample.volume_path is None:
                continue
            header = read_header(sample.volume_path)
            swc = parse_swc(sample.swc_path)
            if check_swc_in_volume(swc, header).is_aligned:
                aligned_samples.append(sample)
        samples = aligned_samples

    write_manifest(samples, root, output)

    print("Gold166 manifest written:", output)
    for key, value in manifest_summary(samples).items():
        print(f"{key}: {value}")

    if args.validate_swc:
        error_count = 0
        for sample in samples:
            try:
                tree = parse_swc(sample.swc_path)
                errors = tree.validate()
            except ValueError as exc:
                errors = [str(exc)]
            if errors:
                error_count += 1
                print(f"SWC validation failed for {sample.sample_id}: {'; '.join(errors)}")
        print(f"swc_validation_errors: {error_count}")
        if error_count:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
