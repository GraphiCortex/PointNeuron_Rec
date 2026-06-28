from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build connectivity records from E2E graph outputs that pass topology eligibility checks."
    )
    parser.add_argument("--summary", action="append", required=True, help="E2E summary.json. Can be repeated.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--output-root", default="tmp/connectivity_guarded30_eligible")
    parser.add_argument("--min-edge-f1", type=float, default=0.65)
    parser.add_argument("--min-reachable", type=float, default=0.90)
    parser.add_argument("--max-bridges", type=int, default=5)
    parser.add_argument("--min-nodes", type=int, default=8)
    parser.add_argument("--target-mode", default="mst", choices=["mst", "knn", "mst_knn"])
    parser.add_argument("--include-score", action="store_true", help="Append proposal objectness score to node features.")
    parser.add_argument("--init-from-target", action="store_true", help="Build records using GT-induced adjacency as input adjacency.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    records_dir = output_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    accepted = []
    rejected = []
    seen_sample_indices: set[int] = set()
    for summary_path in [Path(path) for path in args.summary]:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        topology_by_sample = load_topology(summary_path)
        for row in summary.get("samples", []):
            sample_index = int(row["sample_index"])
            if sample_index in seen_sample_indices:
                rejected.append(reject(row, summary_path, "duplicate_sample_index"))
                continue
            seen_sample_indices.add(sample_index)

            topology = topology_by_sample.get(sample_index, {})
            decision = eligibility_decision(
                row=row,
                topology=topology,
                min_edge_f1=float(args.min_edge_f1),
                min_reachable=float(args.min_reachable),
                max_bridges=int(args.max_bridges),
                min_nodes=int(args.min_nodes),
            )
            if decision:
                rejected.append(reject(row, summary_path, decision, topology=topology))
                continue

            sample_tag = f"sample_{sample_index:04d}"
            record_path = records_dir / f"{sample_tag}_connectivity.npz"
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_connectivity_record.py"),
                "--init-graph",
                str(row["graph_path"]),
                "--use-ground-truth",
                "--root",
                args.root,
                "--sample-index",
                str(sample_index),
                "--target-mode",
                args.target_mode,
                "--output",
                str(record_path),
            ]
            if args.include_score:
                command.append("--include-score")
            if args.init_from_target:
                command.append("--init-from-target")

            if args.dry_run:
                print("would build:", " ".join(command))
            elif args.skip_existing and record_path.exists():
                print(f"skip existing: {record_path}")
            else:
                print("build:", record_path)
                subprocess.run(command, cwd=REPO_ROOT, check=True)

            accepted.append(
                {
                    "sample_index": sample_index,
                    "sample_tag": sample_tag,
                    "record": str(record_path),
                    "graph": str(row["graph_path"]),
                    "proposal": str(row["proposal_path"]),
                    "summary": str(summary_path),
                    "edge_f1": float(topology.get("edge_f1", 0.0)),
                    "bridge_edges": int(topology.get("bridge_edges", row.get("bridge_edges", 0))),
                    "reachable_edge_fraction": float(topology.get("reachable_edge_fraction", row.get("reachable_edge_fraction", 0.0))),
                    "nodes": int(topology.get("nodes", row.get("reconstruction_nodes", 0))),
                }
            )

    manifest = {
        "root": args.root,
        "output_root": str(output_root),
        "eligibility": {
            "min_edge_f1": float(args.min_edge_f1),
            "min_reachable": float(args.min_reachable),
            "max_bridges": int(args.max_bridges),
            "min_nodes": int(args.min_nodes),
            "target_mode": args.target_mode,
            "include_score": bool(args.include_score),
            "init_from_target": bool(args.init_from_target),
        },
        "records": accepted,
        "rejected": rejected,
        "counts": {
            "accepted": len(accepted),
            "rejected": len(rejected),
        },
    }
    manifest_path = output_root / "eligible_connectivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"accepted: {len(accepted)}")
    print(f"rejected: {len(rejected)}")
    print("rejection_reasons:")
    for reason, count in sorted(reason_counts(rejected).items()):
        print(f"  {reason}: {count}")
    print(f"manifest: {manifest_path}")
    return 0


def load_topology(summary_path: Path) -> dict[int, dict]:
    report_path = summary_path.with_name("topology_report.json")
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {int(row["sample_index"]): row for row in report.get("samples", [])}


def eligibility_decision(
    row: dict,
    topology: dict,
    min_edge_f1: float,
    min_reachable: float,
    max_bridges: int,
    min_nodes: int,
) -> str:
    if not bool(row.get("swc_valid", True)):
        return "invalid_swc"
    if int(row.get("reconstruction_roots", 1)) != 1:
        return "not_single_root"
    if not bool(row.get("foreground_cap_satisfied", True)):
        return "foreground_cap_not_satisfied"

    nodes = int(topology.get("nodes", row.get("reconstruction_nodes", 0)))
    edge_f1 = float(topology.get("edge_f1", 0.0))
    bridges = int(topology.get("bridge_edges", row.get("bridge_edges", 0)))
    reachable = float(topology.get("reachable_edge_fraction", row.get("reachable_edge_fraction", 0.0)))

    if nodes < int(min_nodes):
        return "too_few_nodes"
    if edge_f1 < float(min_edge_f1):
        return "low_edge_f1"
    if reachable < float(min_reachable):
        return "low_reachable"
    if bridges > int(max_bridges):
        return "too_many_bridges"
    return ""


def reject(row: dict, summary_path: Path, reason: str, topology: dict | None = None) -> dict:
    topology = topology or {}
    return {
        "sample_index": int(row["sample_index"]),
        "sample_tag": row.get("sample_tag", f"sample_{int(row['sample_index']):04d}"),
        "summary": str(summary_path),
        "reason": reason,
        "edge_f1": float(topology.get("edge_f1", 0.0)),
        "bridge_edges": int(topology.get("bridge_edges", row.get("bridge_edges", 0))),
        "reachable_edge_fraction": float(topology.get("reachable_edge_fraction", row.get("reachable_edge_fraction", 0.0))),
    }


def reason_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
