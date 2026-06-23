from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_header
from scripts.analyze_gold_domains import domain_family, source_group


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a clean stratified Gold166 subset for mixed-domain skeleton training.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--output", default="tmp/mixed_gold_subset_seed0.json", help="Output JSON selection.")
    parser.add_argument("--stratify-by", default="domain_family", choices=["domain_family", "source_group"], help="Grouping key for balancing.")
    parser.add_argument("--max-per-group", type=int, default=8, help="Maximum aligned samples selected from each group. 0 disables.")
    parser.add_argument("--min-swc-nodes", type=int, default=20, help="Reject tiny SWCs below this many nodes.")
    parser.add_argument("--max-volume-voxels", type=int, default=0, help="Reject volumes larger than this voxel count. 0 disables.")
    parser.add_argument("--include-groups", default="", help="Comma-separated group allowlist after stratification.")
    parser.add_argument("--exclude-groups", default="", help="Comma-separated group denylist after stratification.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for within-group selection.")
    args = parser.parse_args()

    include_groups = parse_group_set(args.include_groups)
    exclude_groups = parse_group_set(args.exclude_groups)
    rng = random.Random(args.seed)

    selected_by_group: dict[str, list[dict[str, Any]]] = {}
    rejected = []
    for index, sample in enumerate(scan_gold166(args.root)):
        row, reason = candidate_row(index, sample, args)
        group = str(row.get(args.stratify_by, ""))
        if reason is None and include_groups and group not in include_groups:
            reason = "not_in_include_groups"
        if reason is None and group in exclude_groups:
            reason = "excluded_group"
        if reason is not None:
            rejected.append(dict(row, rejection_reason=reason))
            continue
        selected_by_group.setdefault(group, []).append(row)

    selected = []
    for group, rows in sorted(selected_by_group.items()):
        rows = sorted(rows, key=lambda item: item["sample_index"])
        rng.shuffle(rows)
        if args.max_per_group > 0:
            rows = rows[: args.max_per_group]
        selected.extend(sorted(rows, key=lambda item: item["sample_index"]))
    selected.sort(key=lambda item: (item[args.stratify_by], item["sample_index"]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": args.root,
        "seed": args.seed,
        "stratify_by": args.stratify_by,
        "max_per_group": args.max_per_group,
        "min_swc_nodes": args.min_swc_nodes,
        "max_volume_voxels": args.max_volume_voxels,
        "selected_count": len(selected),
        "selected_by_group": summarize_groups(selected, args.stratify_by),
        "selected": selected,
        "rejected_preview": rejected[:200],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"selected_count: {len(selected)}")
    for group, count in payload["selected_by_group"].items():
        print(f"{group}: {count}")
    print(f"output: {output}")
    if selected:
        print("sample_index_args:")
        print(" ".join(f"--sample-index {row['sample_index']}" for row in selected))
    return 0 if selected else 2


def candidate_row(index: int, sample, args) -> tuple[dict[str, Any], str | None]:
    row: dict[str, Any] = {
        "sample_index": index,
        "sample_id": sample.sample_id,
        "source_group": source_group(sample.sample_id),
        "domain_family": domain_family(sample.sample_id),
        "swc_selection": sample.swc_selection,
        "has_pixel_size": sample.pixel_size_path is not None,
    }
    if sample.volume_path is None:
        return row, "missing_volume"

    try:
        header = read_header(sample.volume_path)
        swc = parse_swc(sample.swc_path)
        validation_errors = swc.validate()
    except Exception as exc:
        row["read_error"] = str(exc)
        return row, "read_error"

    width, height, depth, channels = header.dimensions
    row.update(
        {
            "volume_width": width,
            "volume_height": height,
            "volume_depth": depth,
            "volume_channels": channels,
            "volume_voxels": width * height * depth * channels,
            "volume_datatype": header.datatype,
            "swc_nodes": len(swc.nodes),
            "swc_edges": swc.edge_count,
            "swc_roots": swc.root_count,
            "swc_valid": not validation_errors,
            "volume_path": str(sample.volume_path),
            "swc_path": str(sample.swc_path),
        }
    )

    if channels != 1:
        return row, "multi_channel"
    if validation_errors:
        row["swc_validation_errors"] = validation_errors
        return row, "invalid_swc"
    if len(swc.nodes) < args.min_swc_nodes:
        return row, "too_few_swc_nodes"
    if args.max_volume_voxels > 0 and row["volume_voxels"] > args.max_volume_voxels:
        return row, "volume_too_large"

    alignment = check_swc_in_volume(swc, header)
    row["aligned"] = alignment.is_aligned
    row["out_of_bounds_nodes"] = len(alignment.out_of_bounds_node_ids)
    if not alignment.is_aligned:
        return row, "swc_not_aligned"
    return row, None


def parse_group_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        group = str(row[key])
        summary[group] = summary.get(group, 0) + 1
    return dict(sorted(summary.items()))


if __name__ == "__main__":
    raise SystemExit(main())
