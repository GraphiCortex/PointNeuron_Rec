from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


METRIC_KEYS = [
    "precision",
    "coverage",
    "terminal_coverage",
    "local_terminal_coverage",
    "terminal_patch_coverage",
    "branch_coverage",
    "mean_distance",
    "selected",
    "nodes",
    "terminals",
    "branch_nodes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize full-sample proposal aggregation quality.")
    parser.add_argument("--input-dir", default="tmp/connectivity_dataset/aggregations", help="Directory of *_proposals.npz files.")
    parser.add_argument("--csv-output", default="tmp/proposal_audit_summary.csv", help="Optional CSV summary path. Use empty string to skip.")
    parser.add_argument("--min-coverage", type=float, default=0.65, help="Coverage threshold for a usable sample.")
    parser.add_argument("--min-terminal-coverage", type=float, default=0.50, help="Terminal coverage threshold for a usable sample.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("*_proposals.npz"))
    if not paths:
        print(f"No proposal aggregation files found in {input_dir}")
        return 2

    rows = [load_row(path, args.min_coverage, args.min_terminal_coverage) for path in paths]
    write_csv(Path(args.csv_output), rows) if args.csv_output else None
    print_summary(rows, args.min_coverage, args.min_terminal_coverage)
    if args.csv_output:
        print(f"csv_output: {args.csv_output}")
    return 0


def load_row(path: Path, min_coverage: float, min_terminal_coverage: float) -> dict:
    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}
    metrics = metadata.get("metrics", {})
    row = {
        "path": str(path),
        "sample_index": metadata.get("sample_index"),
        "sample_id": metadata.get("sample_id"),
        "patches": metadata.get("patches"),
        "local_score_threshold": metadata.get("local_score_threshold"),
        "local_nms_mode": metadata.get("local_nms_mode"),
        "global_nms_mode": metadata.get("global_nms_mode"),
        "global_nms_radius": metadata.get("global_nms_radius"),
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    row["usable_for_connectivity"] = (
        as_float(row.get("coverage")) >= min_coverage
        and as_float(row.get("terminal_coverage")) >= min_terminal_coverage
    )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], min_coverage: float, min_terminal_coverage: float) -> None:
    usable = [row for row in rows if row["usable_for_connectivity"]]
    print(f"samples: {len(rows)}")
    print(f"usable_thresholds: coverage>={min_coverage:.2f} terminal_coverage>={min_terminal_coverage:.2f}")
    print(f"usable_for_connectivity: {len(usable)}/{len(rows)}")
    for key in ["precision", "coverage", "terminal_coverage", "branch_coverage", "mean_distance"]:
        values = [as_float(row.get(key)) for row in rows if row.get(key) is not None and np.isfinite(as_float(row.get(key)))]
        if values:
            print(
                f"{key}: mean={statistics.fmean(values):.4f} "
                f"median={statistics.median(values):.4f} min={min(values):.4f} max={max(values):.4f}"
            )
    print("samples_by_coverage:")
    for row in sorted(rows, key=lambda item: as_float(item.get("coverage"))):
        print(
            f"  sample_index={row.get('sample_index')} "
            f"coverage={as_float(row.get('coverage')):.4f} "
            f"terminal={as_float(row.get('terminal_coverage')):.4f} "
            f"precision={as_float(row.get('precision')):.4f} "
            f"selected={row.get('selected')} nodes={row.get('nodes')} "
            f"usable={row['usable_for_connectivity']} "
            f"sample_id={row.get('sample_id')}"
        )


def as_float(value) -> float:
    if value is None:
        return 0.0
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
