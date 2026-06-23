from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_header


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Gold166 domains and join them to an end-to-end run.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--run-root", help="Optional end-to-end output root containing summary.jsonl and sample folders.")
    parser.add_argument("--output-json", default="tmp/gold_domain_report.json")
    parser.add_argument("--output-csv", default="tmp/gold_domain_report.csv")
    args = parser.parse_args()

    root = Path(args.root)
    run_root = Path(args.run_root) if args.run_root else None
    run_states = load_run_states(run_root) if run_root else {}

    rows = []
    for sample_index, sample in enumerate(scan_gold166(root)):
        row = sample_row(sample_index, sample)
        if run_root:
            row.update(run_row(run_root, sample_index, run_states.get(sample_index)))
        rows.append(row)

    report = {
        "root": str(root),
        "run_root": str(run_root) if run_root else None,
        "summary": summarize(rows),
        "domains": summarize_by(rows, "domain_family"),
        "source_groups": summarize_by(rows, "source_group"),
        "samples": rows,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output_csv, rows)

    print_summary(report)
    print(f"json_output: {output_json}")
    print(f"csv_output: {output_csv}")
    return 0


def sample_row(sample_index: int, sample) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_index": sample_index,
        "sample_id": sample.sample_id,
        "source_group": source_group(sample.sample_id),
        "domain_family": domain_family(sample.sample_id),
        "swc_selection": sample.swc_selection,
        "alternate_swc_count": len(sample.alternate_swc_paths),
        "has_pixel_size": sample.pixel_size_path is not None,
        "volume_path": str(sample.volume_path) if sample.volume_path else "",
        "swc_path": str(sample.swc_path),
    }
    try:
        swc = parse_swc(sample.swc_path)
        row.update(
            {
                "swc_nodes": len(swc.nodes),
                "swc_edges": swc.edge_count,
                "swc_roots": swc.root_count,
                "swc_valid": not swc.validate(),
            }
        )
    except Exception as exc:
        row.update({"swc_nodes": 0, "swc_edges": 0, "swc_roots": 0, "swc_valid": False, "swc_error": str(exc)})
        swc = None

    if sample.volume_path is None:
        row.update({"has_volume": False, "aligned": False, "out_of_bounds_nodes": None})
        return row

    row["has_volume"] = True
    try:
        header = read_header(sample.volume_path)
        width, height, depth, channels = header.dimensions
        row.update(
            {
                "volume_width": width,
                "volume_height": height,
                "volume_depth": depth,
                "volume_channels": channels,
                "volume_voxels": width * height * depth * channels,
                "volume_datatype": header.datatype,
            }
        )
        if swc is not None:
            alignment = check_swc_in_volume(swc, header)
            row["aligned"] = alignment.is_aligned
            row["out_of_bounds_nodes"] = len(alignment.out_of_bounds_node_ids)
    except Exception as exc:
        row.update({"aligned": False, "out_of_bounds_nodes": None, "volume_error": str(exc)})
    return row


def load_run_states(run_root: Path | None) -> dict[int, dict[str, Any]]:
    if run_root is None:
        return {}
    jsonl = run_root / "summary.jsonl"
    if not jsonl.exists():
        return {}

    states: dict[int, dict[str, Any]] = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        sample_index = int(item["sample_index"])
        states[sample_index] = item
    return states


def run_row(run_root: Path, sample_index: int, state: dict[str, Any] | None) -> dict[str, Any]:
    sample_tag = f"sample_{sample_index:04d}"
    sample_dir = run_root / sample_tag
    proposal_path = sample_dir / f"{sample_tag}_proposals.npz"
    graph_path = sample_dir / f"{sample_tag}_geodesic_graph.npz"

    row: dict[str, Any] = {
        "run_status": "not_run",
        "proposal_exists": proposal_path.exists(),
        "graph_exists": graph_path.exists(),
    }
    if state is not None:
        row["run_status"] = "failed" if state.get("status") == "failed" else "success"
        if state.get("status") == "failed":
            row.update(
                {
                    "failure_error_type": state.get("error_type", ""),
                    "failure_returncode": state.get("returncode", ""),
                    "failure_error": state.get("error", ""),
                }
            )
        else:
            copy_keys(
                row,
                state,
                [
                    "reconstruction_nodes",
                    "reconstruction_roots",
                    "bridge_edges",
                    "reachable_edge_fraction",
                    "mean_edge_geodesic_distance",
                    "mean_snap_distance",
                    "foreground_threshold",
                    "foreground_threshold_was_adapted",
                    "foreground_cap_satisfied",
                ],
            )

    if proposal_path.exists():
        row.update(proposal_metrics(proposal_path))
    if graph_path.exists():
        row.update(graph_metrics(graph_path))
    return row


def proposal_metrics(path: Path) -> dict[str, Any]:
    try:
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata"]))
        metrics = metadata.get("metrics", {})
        return {
            "proposal_path": str(path),
            "proposal_selected": metrics.get("selected", int(payload["centers"].shape[0])),
            "proposal_precision": metrics.get("precision"),
            "proposal_coverage": metrics.get("coverage"),
            "proposal_terminal_coverage": metrics.get("terminal_coverage"),
            "proposal_branch_coverage": metrics.get("branch_coverage"),
            "proposal_mean_distance": metrics.get("mean_distance"),
            "proposal_patches": metadata.get("patches"),
            "proposal_threshold": metadata.get("threshold"),
        }
    except Exception as exc:
        return {"proposal_path": str(path), "proposal_error": str(exc)}


def graph_metrics(path: Path) -> dict[str, Any]:
    try:
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata"]))
        return {
            "graph_path": str(path),
            "graph_nodes": metadata.get("nodes"),
            "graph_edges": metadata.get("edges"),
            "graph_bridge_edges": metadata.get("bridge_edges"),
            "graph_reachable_edge_fraction": metadata.get("reachable_edge_fraction"),
            "graph_foreground_voxels": metadata.get("foreground_voxels"),
            "graph_foreground_threshold": metadata.get("foreground_threshold"),
            "graph_foreground_cap_satisfied": metadata.get("foreground_cap_satisfied"),
        }
    except Exception as exc:
        return {"graph_path": str(path), "graph_error": str(exc)}


def copy_keys(target: dict[str, Any], source: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        if key in source:
            target[key] = source[key]


def source_group(sample_id: str) -> str:
    first = sample_id.split("/", 1)[0].lower()
    for prefix in ("e_checked6_", "e_checked7_", "p_checked6_", "p_checked7_"):
        if first.startswith(prefix):
            return first[len(prefix) :]
    return first


def domain_family(sample_id: str) -> str:
    text = sample_id.lower()
    if "janelia" in text:
        return "janelia_fly"
    if "utokyo_fly" in text:
        return "utokyo_fly"
    if "fruitfly_larvae" in text:
        return "fruitfly_larvae"
    if "zebrafish" in text:
        return "zebrafish"
    if "chick" in text:
        return "chick"
    if "frog" in text:
        return "frog"
    if "human" in text:
        return "human"
    if "mouse" in text:
        return "mouse"
    return "other"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_rows(rows)


def summarize_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return [dict({key: name}, **summarize_rows(group)) for name, group in sorted(groups.items())]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("run_status") == "success"]
    failures = [row for row in rows if row.get("run_status") == "failed"]
    proposal_rows = [row for row in rows if row.get("proposal_exists")]
    return {
        "sample_count": len(rows),
        "aligned_count": sum(bool(row.get("aligned")) for row in rows),
        "out_of_bounds_count": sum((row.get("out_of_bounds_nodes") or 0) > 0 for row in rows),
        "run_success_count": len(successes),
        "run_failure_count": len(failures),
        "mean_proposal_coverage": mean(row.get("proposal_coverage") for row in proposal_rows),
        "mean_proposal_precision": mean(row.get("proposal_precision") for row in proposal_rows),
        "mean_proposal_terminal_coverage": mean(row.get("proposal_terminal_coverage") for row in proposal_rows),
        "mean_reachable_edge_fraction": mean(row.get("reachable_edge_fraction") for row in successes),
        "mean_bridge_edges": mean(row.get("bridge_edges") for row in successes),
        "zero_proposal_count": sum((row.get("proposal_selected") == 0) for row in rows),
        "low_coverage_count": sum(is_number(row.get("proposal_coverage")) and float(row["proposal_coverage"]) < 0.1 for row in rows),
        "low_reachable_count": sum(is_number(row.get("reachable_edge_fraction")) and float(row["reachable_edge_fraction"]) < 0.2 for row in rows),
        "large_bridge_count": sum(is_number(row.get("bridge_edges")) and int(row["bridge_edges"]) >= 50 for row in rows),
    }


def mean(values) -> float | None:
    numeric = [float(value) for value in values if is_number(value)]
    return float(sum(numeric) / len(numeric)) if numeric else None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(report: dict[str, Any]) -> None:
    print("overall:")
    for key, value in report["summary"].items():
        print(f"  {key}: {format_value(value)}")
    print("domains:")
    for row in report["domains"]:
        print(
            "  "
            f"{row['domain_family']}: samples={row['sample_count']} "
            f"success={row['run_success_count']} failed={row['run_failure_count']} "
            f"proposal_cov={format_value(row['mean_proposal_coverage'])} "
            f"reachable={format_value(row['mean_reachable_edge_fraction'])} "
            f"low_cov={row['low_coverage_count']} low_reach={row['low_reachable_count']}"
        )


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
