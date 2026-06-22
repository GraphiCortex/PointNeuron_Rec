from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import volume_data_array
from pointneuron.data.vaa3d_raw import read_header, read_volume


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a paper-like subset before running PointNeuron skeleton experiments.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--include-regex", default="janelia|fly|fruitfly", help="Case-insensitive sample_id regex to include.")
    parser.add_argument("--exclude-regex", default="", help="Case-insensitive sample_id regex to exclude.")
    parser.add_argument("--max-samples", type=int, default=42, help="Maximum selected samples.")
    parser.add_argument("--max-width", type=int, default=384, help="Maximum volume width.")
    parser.add_argument("--max-height", type=int, default=384, help="Maximum volume height.")
    parser.add_argument("--max-depth", type=int, default=384, help="Maximum volume depth.")
    parser.add_argument("--threshold-fraction", type=float, default=0.2, help="Foreground threshold fraction used for optional foreground counts.")
    parser.add_argument("--count-foreground", action="store_true", help="Decode volumes and count foreground voxels. Slower but useful.")
    parser.add_argument("--min-foreground", type=int, default=1000, help="Minimum foreground count when --count-foreground is used.")
    parser.add_argument("--max-foreground", type=int, default=60000, help="Maximum foreground count when --count-foreground is used.")
    parser.add_argument("--output", default="tmp/paper_subset_indices.json", help="Output JSON path.")
    args = parser.parse_args()

    include = re.compile(args.include_regex, flags=re.IGNORECASE) if args.include_regex else None
    exclude = re.compile(args.exclude_regex, flags=re.IGNORECASE) if args.exclude_regex else None
    samples = scan_gold166(args.root)
    selected = []
    rejected = []
    for index, sample in enumerate(samples):
        reason = rejection_reason(index, sample, args, include, exclude)
        if reason is None:
            row = sample_row(index, sample, args)
            selected.append(row)
            if len(selected) >= args.max_samples:
                break
        else:
            rejected.append({"sample_index": index, "sample_id": sample.sample_id, "reason": reason})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": args.root,
        "include_regex": args.include_regex,
        "exclude_regex": args.exclude_regex,
        "threshold_fraction": args.threshold_fraction,
        "count_foreground": args.count_foreground,
        "selected_count": len(selected),
        "selected": selected,
        "rejected_preview": rejected[:100],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selected_count: {len(selected)}")
    print(f"output: {output}")
    if selected:
        indices = " ".join(f"--sample-index {row['sample_index']}" for row in selected)
        print("sample_index_args:")
        print(indices)
    return 0 if selected else 2


def rejection_reason(index: int, sample, args, include, exclude) -> str | None:
    if sample.volume_path is None:
        return "missing_volume"
    if include is not None and include.search(sample.sample_id) is None:
        return "include_regex"
    if exclude is not None and exclude.search(sample.sample_id) is not None:
        return "exclude_regex"
    try:
        header = read_header(sample.volume_path)
        swc = parse_swc(sample.swc_path)
    except Exception as exc:
        return f"read_error:{exc}"
    width, height, depth, channels = header.dimensions
    if channels != 1:
        return "multi_channel"
    if width > args.max_width or height > args.max_height or depth > args.max_depth:
        return "volume_too_large"
    if not check_swc_in_volume(swc, header).is_aligned:
        return "swc_not_aligned"
    if args.count_foreground:
        foreground = foreground_count(sample.volume_path, header.datatype, args.threshold_fraction)
        if foreground < args.min_foreground:
            return "too_little_foreground"
        if foreground > args.max_foreground:
            return "too_much_foreground"
    return None


def sample_row(index: int, sample, args) -> dict:
    header = read_header(sample.volume_path)
    row = {
        "sample_index": index,
        "sample_id": sample.sample_id,
        "volume_path": str(sample.volume_path),
        "swc_path": str(sample.swc_path),
        "dimensions": list(header.dimensions),
        "datatype": header.datatype,
    }
    if args.count_foreground:
        row["foreground_count"] = foreground_count(sample.volume_path, header.datatype, args.threshold_fraction)
    return row


def foreground_count(volume_path: Path, datatype: int, threshold_fraction: float) -> int:
    volume = read_volume(volume_path)
    data = volume_data_array(volume)
    if datatype == 1:
        threshold = int(round(255 * threshold_fraction))
    elif datatype == 2:
        threshold = int(round(65535 * threshold_fraction))
    else:
        raise NotImplementedError(f"Vaa3D datatype {datatype} is not supported")
    return int(np.count_nonzero(data > threshold))


if __name__ == "__main__":
    raise SystemExit(main())
