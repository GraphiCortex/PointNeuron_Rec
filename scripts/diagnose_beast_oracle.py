from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import SwcTree, parse_swc


DEFAULT_BEAST_SAMPLES = [7, 16, 22, 23, 24]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an oracle-node diagnosis on early hard samples to separate graph-stage limits from proposal failures."
    )
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--sample-index", type=int, action="append", help="Sample index. Can be repeated.")
    parser.add_argument("--output-root", default="tmp/beast_oracle_diagnosis")
    parser.add_argument("--max-nodes", type=int, default=128, help="Oracle graph-node budget.")
    parser.add_argument("--foreground-threshold", type=int, default=5)
    parser.add_argument("--max-foreground-voxels", type=int, default=350000)
    parser.add_argument("--candidate-k", type=int, default=6)
    parser.add_argument("--max-geodesic-ratio", type=float, default=12.0)
    args = parser.parse_args()

    sample_indices = sorted(dict.fromkeys(args.sample_index or DEFAULT_BEAST_SAMPLES))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = scan_gold166(args.root)
    rows = []
    failures = []
    for sample_index in sample_indices:
        sample_tag = f"sample_{sample_index:04d}"
        sample_dir = output_root / sample_tag
        sample_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = sample_dir / f"{sample_tag}_oracle_proposals.npz"
        graph_path = sample_dir / f"{sample_tag}_oracle_geodesic_graph.npz"
        try:
            sample = samples[sample_index]
            swc = parse_swc(sample.swc_path)
            write_oracle_proposals(
                path=proposal_path,
                swc=swc,
                sample_index=sample_index,
                sample_id=sample.sample_id,
                max_nodes=int(args.max_nodes),
                threshold=int(args.foreground_threshold),
            )
            run_command(
                [
                    sys.executable,
                    "scripts/initialize_geodesic_graph.py",
                    "--root",
                    args.root,
                    "--sample-index",
                    str(sample_index),
                    "--proposals",
                    str(proposal_path),
                    "--mode",
                    "mst",
                    "--nms-distance",
                    "0",
                    "--max-nodes",
                    "0",
                    "--selection-mode",
                    "score_nms",
                    "--min-proposal-score",
                    "0",
                    "--foreground-threshold",
                    str(args.foreground_threshold),
                    "--max-foreground-voxels",
                    str(args.max_foreground_voxels),
                    "--candidate-k",
                    str(args.candidate_k),
                    "--max-geodesic-ratio",
                    str(args.max_geodesic_ratio),
                    "--bridge-components",
                    "--bridge-allow-unreachable-fallback",
                    "--output",
                    str(graph_path),
                ]
            )
            metadata = load_graph_metadata(graph_path)
            row = {
                "sample_index": int(sample_index),
                "sample_tag": sample_tag,
                "initializer": "oracle_geodesic",
                "proposal_path": str(proposal_path),
                "graph_path": str(graph_path),
                "swc_path": "",
                "compare_html": "",
                "oracle_nodes": int(metadata["nodes"]),
                "foreground_threshold": metadata["foreground_threshold"],
                "foreground_threshold_was_adapted": metadata["foreground_threshold_was_adapted"],
                "foreground_cap_satisfied": metadata["foreground_cap_satisfied"],
                "eligible_candidate_pairs": metadata["eligible_candidate_pairs"],
                "bridge_edges": metadata["bridge_edges"],
                "reachable_edge_fraction": metadata["reachable_edge_fraction"],
                "mean_snap_distance": metadata["mean_snap_distance"],
            }
            rows.append(row)
            print(
                f"{sample_tag}: oracle_nodes={row['oracle_nodes']} "
                f"reachable={row['reachable_edge_fraction']:.4f} bridges={row['bridge_edges']} "
                f"cap_ok={row['foreground_cap_satisfied']}"
            )
        except Exception as exc:
            failure = {
                "sample_index": int(sample_index),
                "sample_tag": sample_tag,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"FAILED {sample_tag}: {failure['error_type']}: {failure['error']}")

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "diagnosis": "oracle_gt_nodes",
                "samples": rows,
                "failures": failures,
                "summary": {
                    "requested_sample_count": len(sample_indices),
                    "sample_count": len(rows),
                    "failure_count": len(failures),
                    "mean_reachable_edge_fraction": mean(row["reachable_edge_fraction"] for row in rows),
                    "mean_bridge_edges": mean(row["bridge_edges"] for row in rows),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    topology_csv = output_root / "topology_report.csv"
    topology_json = output_root / "topology_report.json"
    if rows:
        run_command(
            [
                sys.executable,
                "scripts/evaluate_geodesic_baseline.py",
                "--root",
                args.root,
                "--summary",
                str(summary_path),
                "--csv-output",
                str(topology_csv),
                "--json-output",
                str(topology_json),
            ]
        )
        print_verdict(topology_json)
    print(f"summary: {summary_path}")
    return 1 if failures else 0


def write_oracle_proposals(path: Path, swc: SwcTree, sample_index: int, sample_id: str, max_nodes: int, threshold: int) -> None:
    selected_nodes = select_oracle_nodes(swc, max_nodes=max_nodes)
    centers = np.array([[node.x, node.y, node.z] for node in selected_nodes], dtype=np.float32)
    radii = np.array([max(float(node.radius), 1.0) for node in selected_nodes], dtype=np.float32)
    scores = np.ones((centers.shape[0],), dtype=np.float32)
    features = np.zeros((centers.shape[0], 0), dtype=np.float32)
    metadata = {
        "sample_id": sample_id,
        "sample_index": int(sample_index),
        "threshold": int(threshold),
        "source": "gt_oracle_swc",
        "swc_path": str(swc.path),
        "max_nodes": int(max_nodes),
    }
    np.savez_compressed(
        path,
        centers=centers,
        radii=radii,
        scores=scores,
        features=features,
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )


def select_oracle_nodes(swc: SwcTree, max_nodes: int):
    nodes = list(swc.nodes)
    if max_nodes <= 0 or len(nodes) <= max_nodes:
        return nodes

    children: dict[int, list[int]] = {node.node_id: [] for node in nodes}
    by_id = {node.node_id: node for node in nodes}
    for node in nodes:
        if node.parent_id in children:
            children[node.parent_id].append(node.node_id)

    priority_ids = []
    for node in nodes:
        child_count = len(children.get(node.node_id, []))
        if node.parent_id == -1 or child_count == 0 or child_count > 1:
            priority_ids.append(node.node_id)
    priority_ids = unique(priority_ids)

    if len(priority_ids) >= max_nodes:
        coords = np.array([[by_id[node_id].x, by_id[node_id].y, by_id[node_id].z] for node_id in priority_ids], dtype=np.float32)
        selected_local = farthest_point_indices(coords, max_nodes)
        return [by_id[priority_ids[index]] for index in selected_local]

    selected_ids = list(priority_ids)
    selected_set = set(selected_ids)
    remaining_nodes = [node for node in nodes if node.node_id not in selected_set]
    if remaining_nodes:
        seed_coords = np.array([[by_id[node_id].x, by_id[node_id].y, by_id[node_id].z] for node_id in selected_ids], dtype=np.float32)
        remaining_coords = np.array([[node.x, node.y, node.z] for node in remaining_nodes], dtype=np.float32)
        selected_remaining = farthest_fill_indices(
            remaining_coords=remaining_coords,
            seed_coords=seed_coords,
            count=max_nodes - len(selected_ids),
        )
        selected_ids.extend(remaining_nodes[index].node_id for index in selected_remaining)
    return [by_id[node_id] for node_id in selected_ids[:max_nodes]]


def farthest_point_indices(coords: np.ndarray, count: int) -> list[int]:
    if coords.shape[0] <= count:
        return list(range(coords.shape[0]))
    center = coords.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.sum((coords - center) ** 2, axis=1)))
    selected = [first]
    min_distance2 = np.sum((coords - coords[first].reshape(1, 3)) ** 2, axis=1)
    min_distance2[first] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(min_distance2))
        selected.append(next_index)
        distance2 = np.sum((coords - coords[next_index].reshape(1, 3)) ** 2, axis=1)
        min_distance2 = np.minimum(min_distance2, distance2)
        min_distance2[selected] = -np.inf
    return selected


def farthest_fill_indices(remaining_coords: np.ndarray, seed_coords: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    if remaining_coords.shape[0] <= count:
        return list(range(remaining_coords.shape[0]))
    if seed_coords.size:
        diff = remaining_coords[:, None, :] - seed_coords[None, :, :]
        min_distance2 = np.sum(diff * diff, axis=2).min(axis=1)
    else:
        return farthest_point_indices(remaining_coords, count)
    selected = []
    for _ in range(count):
        next_index = int(np.argmax(min_distance2))
        selected.append(next_index)
        distance2 = np.sum((remaining_coords - remaining_coords[next_index].reshape(1, 3)) ** 2, axis=1)
        min_distance2 = np.minimum(min_distance2, distance2)
        min_distance2[selected] = -np.inf
    return selected


def unique(values: list[int]) -> list[int]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def load_graph_metadata(path: Path) -> dict:
    payload = np.load(path, allow_pickle=False)
    return json.loads(str(payload["metadata"]))


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def print_verdict(topology_json: Path) -> None:
    report = json.loads(topology_json.read_text(encoding="utf-8"))
    summary = report["summary"]
    f1 = float(summary["mean_edge_f1"])
    bridges = float(summary["mean_bridge_edges"])
    reachable = float(summary["mean_reachable_edge_fraction"])
    print("oracle_diagnosis:")
    print(f"  mean_edge_f1: {f1:.4f}")
    print(f"  mean_bridge_edges: {bridges:.4f}")
    print(f"  mean_reachable_edge_fraction: {reachable:.4f}")
    if f1 >= 0.75 and reachable >= 0.90 and bridges <= 10.0:
        print("  verdict: graph_stage_can_handle_good_nodes")
    else:
        print("  verdict: graph_stage_or_foreground_geodesic_is_a_pointneuron1_weakness")


def mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
