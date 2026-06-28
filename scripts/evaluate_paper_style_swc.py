from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166
from pointneuron.data.swc import SwcTree, parse_swc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated SWCs with paper-style spatial point-distance "
            "metrics. ESA/DSA/PDS are reported as local approximations unless "
            "the official Vaa3D metric plugin is used separately."
        )
    )
    parser.add_argument("--summary", action="append", required=True, help="End-to-end summary JSON. Can be repeated.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--distance-threshold", type=float, default=6.0, help="Distance threshold for precision/recall/F1.")
    parser.add_argument("--sample-step", type=float, default=1.0, help="Approximate spacing for SWC edge resampling.")
    parser.add_argument("--max-points", type=int, default=250000, help="Downsample each SWC point set above this count.")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Rows per nearest-distance chunk.")
    parser.add_argument("--csv-output", default="tmp/paper_style_eval/paper_style_eval.csv")
    parser.add_argument("--json-output", default="tmp/paper_style_eval/paper_style_eval.json")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    rows = []
    for summary_path in args.summary:
        payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        for summary_row in payload.get("samples", []):
            if not summary_row.get("swc_valid", True):
                continue
            rows.append(evaluate_row(summary_path, summary_row, samples, args))

    report = {
        "metric_note": (
            "precision/recall/F1 are point-distance SWC metrics. "
            "approx_esa/approx_dsa/approx_pds are local NumPy approximations "
            "of Vaa3D-style spatial discrepancy metrics, not official Vaa3D outputs."
        ),
        "distance_threshold": args.distance_threshold,
        "sample_step": args.sample_step,
        "summaries": args.summary,
        "samples": rows,
        "summary": summarize(rows),
    }
    write_csv(Path(args.csv_output), rows)
    Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"samples: {len(rows)}")
    print(f"distance_threshold: {args.distance_threshold:g}")
    print(f"sample_step: {args.sample_step:g}")
    for key, value in report["summary"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("worst_by_f1:")
    for row in sorted(rows, key=lambda item: item["f1"])[:5]:
        print(
            f"  sample {row['sample_index']}: f1={row['f1']:.4f} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
            f"approx_esa={row['approx_esa']:.4f} approx_pds={row['approx_pds']:.4f}"
        )
    print(f"csv_output: {args.csv_output}")
    print(f"json_output: {args.json_output}")
    return 0


def evaluate_row(summary_path: str, summary_row: dict, samples, args) -> dict:
    sample_index = int(summary_row["sample_index"])
    gt_swc = parse_swc(samples[sample_index].swc_path)
    pred_swc = parse_swc(summary_row["swc_path"])
    gt_points = resample_swc_points(gt_swc, args.sample_step)
    pred_points = resample_swc_points(pred_swc, args.sample_step)
    gt_points = deterministic_downsample(gt_points, args.max_points)
    pred_points = deterministic_downsample(pred_points, args.max_points)

    pred_to_gt = nearest_distances(pred_points, gt_points, args.chunk_size)
    gt_to_pred = nearest_distances(gt_points, pred_points, args.chunk_size)

    threshold = float(args.distance_threshold)
    precision = float(np.mean(pred_to_gt <= threshold)) if pred_to_gt.size else 0.0
    recall = float(np.mean(gt_to_pred <= threshold)) if gt_to_pred.size else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    approx_esa = float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0) if pred_to_gt.size and gt_to_pred.size else 0.0
    approx_dsa = float(max(pred_to_gt.mean() if pred_to_gt.size else 0.0, gt_to_pred.mean() if gt_to_pred.size else 0.0))
    approx_pds = float(((pred_to_gt > threshold).mean() + (gt_to_pred > threshold).mean()) / 2.0) if pred_to_gt.size and gt_to_pred.size else 0.0

    return {
        "summary": summary_path,
        "sample_index": sample_index,
        "sample_tag": summary_row.get("sample_tag", f"sample_{sample_index:04d}"),
        "gt_swc": str(samples[sample_index].swc_path),
        "pred_swc": str(summary_row["swc_path"]),
        "gt_points": int(gt_points.shape[0]),
        "pred_points": int(pred_points.shape[0]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "approx_esa": approx_esa,
        "approx_dsa": approx_dsa,
        "approx_pds": approx_pds,
        "pred_to_gt_mean": float(pred_to_gt.mean()) if pred_to_gt.size else 0.0,
        "gt_to_pred_mean": float(gt_to_pred.mean()) if gt_to_pred.size else 0.0,
        "pred_to_gt_p95": percentile(pred_to_gt, 95),
        "gt_to_pred_p95": percentile(gt_to_pred, 95),
        "bridge_edges": int(summary_row.get("bridge_edges", 0)),
        "reachable_edge_fraction": float(summary_row.get("reachable_edge_fraction", 0.0)),
    }


def resample_swc_points(swc: SwcTree, step: float) -> np.ndarray:
    nodes_by_id = {node.node_id: node for node in swc.nodes}
    points = []
    for node in swc.nodes:
        point = np.array([node.x, node.y, node.z], dtype=np.float32)
        if node.parent_id == -1 or node.parent_id not in nodes_by_id:
            points.append(point)
            continue
        parent = nodes_by_id[node.parent_id]
        parent_point = np.array([parent.x, parent.y, parent.z], dtype=np.float32)
        delta = point - parent_point
        distance = float(np.linalg.norm(delta))
        segments = max(1, int(np.ceil(distance / max(step, 1.0e-6))))
        for index in range(segments):
            fraction = index / segments
            points.append(parent_point + fraction * delta)
        points.append(point)
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(points).astype(np.float32, copy=False)


def deterministic_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[indices]


def nearest_distances(query: np.ndarray, reference: np.ndarray, chunk_size: int) -> np.ndarray:
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    distances = np.empty((query.shape[0],), dtype=np.float32)
    ref = reference.astype(np.float32, copy=False)
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        diff = query[start:end, None, :] - ref[None, :, :]
        distances[start:end] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return distances


def summarize(rows: list[dict]) -> dict:
    return {
        "sample_count": len(rows),
        "mean_precision": mean(row["precision"] for row in rows),
        "mean_recall": mean(row["recall"] for row in rows),
        "mean_f1": mean(row["f1"] for row in rows),
        "median_f1": median(row["f1"] for row in rows),
        "mean_approx_esa": mean(row["approx_esa"] for row in rows),
        "mean_approx_dsa": mean(row["approx_dsa"] for row in rows),
        "mean_approx_pds": mean(row["approx_pds"] for row in rows),
        "mean_pred_to_gt": mean(row["pred_to_gt_mean"] for row in rows),
        "mean_gt_to_pred": mean(row["gt_to_pred_mean"] for row in rows),
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
    return float(statistics.fmean(values)) if values else 0.0


def median(values) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
