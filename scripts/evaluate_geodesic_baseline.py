from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import parse_swc
from pointneuron.graph.initialization import initialize_proposal_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate geodesic baseline topology against GT-induced proposal connectivity.")
    parser.add_argument("--summary", default="tmp/e2e_baseline_janelia/summary.json", help="End-to-end summary JSON.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--csv-output", default="tmp/e2e_baseline_janelia/topology_report.csv")
    parser.add_argument("--json-output", default="tmp/e2e_baseline_janelia/topology_report.json")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    samples = scan_gold166(args.root)
    rows = []
    for row in summary["samples"]:
        rows.append(evaluate_sample(row, samples))

    report = {
        "samples": rows,
        "summary": summarize(rows),
    }
    write_csv(Path(args.csv_output), rows)
    Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"samples: {len(rows)}")
    for key, value in report["summary"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("worst_edge_f1:")
    for row in sorted(rows, key=lambda item: item["edge_f1"])[:5]:
        print(
            f"  sample {row['sample_index']}: f1={row['edge_f1']:.4f} "
            f"precision={row['edge_precision']:.4f} recall={row['edge_recall']:.4f} "
            f"bridges={row['bridge_edges']} bridge_hit_rate={row['bridge_hit_rate']:.4f}"
        )
    print(f"csv_output: {args.csv_output}")
    print(f"json_output: {args.json_output}")
    return 0


def evaluate_sample(summary_row: dict, samples) -> dict:
    graph_path = Path(summary_row["graph_path"])
    graph = np.load(graph_path, allow_pickle=False)
    metadata = json.loads(str(graph["metadata"]))
    centers = graph["centers"].astype(np.float32, copy=False)
    predicted_edges = normalize_edges(graph["edges"].astype(np.int64, copy=False))
    bridge_edges = normalize_edges(graph["bridge_edges"].astype(np.int64, copy=False)) if "bridge_edges" in graph else set()
    reachable_flags = graph["edge_geodesic_reachable"].astype(bool, copy=False) if "edge_geodesic_reachable" in graph else np.ones((len(predicted_edges),), dtype=bool)
    edge_geodesic = graph["edge_tree_distance"].astype(np.float32, copy=False) if "edge_tree_distance" in graph else np.zeros((len(predicted_edges),), dtype=np.float32)
    edge_euclidean = graph["edge_euclidean_distance"].astype(np.float32, copy=False) if "edge_euclidean_distance" in graph else np.zeros((len(predicted_edges),), dtype=np.float32)

    sample = samples[int(summary_row["sample_index"])]
    gt = initialize_proposal_graph(centers, parse_swc(sample.swc_path), mode="mst")
    target_edges = normalize_edges(gt.edges)
    metrics = precision_recall_f1(predicted_edges, target_edges)

    traced_edges = predicted_edges - bridge_edges
    bridge_hits = len(bridge_edges & target_edges)
    traced_hits = len(traced_edges & target_edges)
    edge_rows = edge_stats(
        edges=graph["edges"].astype(np.int64, copy=False),
        bridge_edges=bridge_edges,
        target_edges=target_edges,
        reachable_flags=reachable_flags,
        edge_geodesic=edge_geodesic,
        edge_euclidean=edge_euclidean,
    )
    return {
        "sample_index": int(summary_row["sample_index"]),
        "nodes": int(centers.shape[0]),
        "predicted_edges": len(predicted_edges),
        "target_edges": len(target_edges),
        "true_positive_edges": len(predicted_edges & target_edges),
        "edge_precision": metrics["precision"],
        "edge_recall": metrics["recall"],
        "edge_f1": metrics["f1"],
        "bridge_edges": len(bridge_edges),
        "bridge_hits": bridge_hits,
        "bridge_hit_rate": bridge_hits / len(bridge_edges) if bridge_edges else 1.0,
        "traced_edges": len(traced_edges),
        "traced_hits": traced_hits,
        "traced_hit_rate": traced_hits / len(traced_edges) if traced_edges else 0.0,
        "reachable_edge_fraction": float(metadata["reachable_edge_fraction"]),
        "mean_bridge_euclidean": edge_rows["mean_bridge_euclidean"],
        "mean_bridge_geodesic": edge_rows["mean_bridge_geodesic"],
        "mean_traced_euclidean": edge_rows["mean_traced_euclidean"],
        "mean_traced_geodesic": edge_rows["mean_traced_geodesic"],
        "compare_html": summary_row["compare_html"],
    }


def edge_stats(
    edges: np.ndarray,
    bridge_edges: set[tuple[int, int]],
    target_edges: set[tuple[int, int]],
    reachable_flags: np.ndarray,
    edge_geodesic: np.ndarray,
    edge_euclidean: np.ndarray,
) -> dict[str, float]:
    bridge_euclidean = []
    bridge_geodesic = []
    traced_euclidean = []
    traced_geodesic = []
    for index, (left, right) in enumerate(edges.tolist()):
        edge = tuple(sorted((int(left), int(right))))
        if edge in bridge_edges:
            bridge_euclidean.append(float(edge_euclidean[index]))
            bridge_geodesic.append(float(edge_geodesic[index]))
        else:
            traced_euclidean.append(float(edge_euclidean[index]))
            traced_geodesic.append(float(edge_geodesic[index]))
    return {
        "mean_bridge_euclidean": mean(bridge_euclidean),
        "mean_bridge_geodesic": mean(bridge_geodesic),
        "mean_traced_euclidean": mean(traced_euclidean),
        "mean_traced_geodesic": mean(traced_geodesic),
    }


def normalize_edges(edges: np.ndarray) -> set[tuple[int, int]]:
    result = set()
    for left, right in edges.tolist():
        left = int(left)
        right = int(right)
        if left != right:
            result.add(tuple(sorted((left, right))))
    return result


def precision_recall_f1(predicted: set[tuple[int, int]], target: set[tuple[int, int]]) -> dict[str, float]:
    true_positive = len(predicted & target)
    false_positive = len(predicted - target)
    false_negative = len(target - predicted)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize(rows: list[dict]) -> dict:
    return {
        "sample_count": len(rows),
        "mean_edge_precision": mean([row["edge_precision"] for row in rows]),
        "mean_edge_recall": mean([row["edge_recall"] for row in rows]),
        "mean_edge_f1": mean([row["edge_f1"] for row in rows]),
        "mean_bridge_edges": mean([row["bridge_edges"] for row in rows]),
        "mean_bridge_hit_rate": mean([row["bridge_hit_rate"] for row in rows]),
        "mean_traced_hit_rate": mean([row["traced_hit_rate"] for row in rows]),
        "mean_reachable_edge_fraction": mean([row["reachable_edge_fraction"] for row in rows]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
