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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained connectivity GAE on connectivity records.")
    parser.add_argument("--records", nargs="+", required=True, help="Connectivity record .npz files.")
    parser.add_argument("--checkpoint", required=True, help="Connectivity checkpoint from scripts/train_connectivity.py.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for threshold metrics.")
    parser.add_argument("--csv-output", help="Optional CSV report path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before evaluation.")
        return 2

    from pointneuron.models.connectivity import ConnectivityGAE

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but torch.cuda.is_available() is false.")
        return 2

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model = ConnectivityGAE(in_channels=int(checkpoint["input_dim"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    with torch.no_grad():
        for record_path in args.records:
            row = evaluate_record(
                Path(record_path),
                model=model,
                threshold=args.threshold,
                normalize=bool(train_args.get("normalize_node_features", False)),
                torch=torch,
                device=device,
            )
            rows.append(row)

    print(f"records: {len(rows)}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"threshold: {args.threshold}")
    for key in [
        "threshold_precision",
        "threshold_recall",
        "threshold_f1",
        "topk_precision",
        "topk_recall",
        "topk_f1",
        "init_topk_precision",
        "init_topk_recall",
        "init_topk_f1",
    ]:
        values = np.array([row[key] for row in rows], dtype=np.float32)
        print(f"{key}: mean={values.mean():.4f} median={np.median(values):.4f} min={values.min():.4f} max={values.max():.4f}")
    print("worst_by_topk_f1:")
    for row in sorted(rows, key=lambda item: item["topk_f1"])[:5]:
        print(
            f"  path={row['path']} nodes={row['nodes']} target_edges={row['target_edges']} "
            f"topk_f1={row['topk_f1']:.4f} init_topk_f1={row['init_topk_f1']:.4f} threshold_f1={row['threshold_f1']:.4f}"
        )

    if args.csv_output:
        write_csv(Path(args.csv_output), rows)
        print(f"csv_output: {args.csv_output}")
    return 0


def evaluate_record(path: Path, model, threshold: float, normalize: bool, torch, device: str) -> dict:
    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}
    node_features = payload["node_features"].astype(np.float32, copy=True)
    if normalize:
        mean = node_features.mean(axis=0, keepdims=True)
        std = node_features.std(axis=0, keepdims=True)
        node_features = (node_features - mean) / np.maximum(std, 1e-6)
    init_adjacency = payload["init_adjacency"].astype(np.float32, copy=False)
    init_edges = payload["init_edges"].astype(np.int64, copy=False) if "init_edges" in payload else np.zeros((0, 2), dtype=np.int64)
    target_adjacency = payload["target_adjacency"].astype(np.uint8, copy=False)

    output = model(
        torch.from_numpy(node_features).to(device),
        torch.from_numpy(init_adjacency).to(device),
    )
    probabilities = torch.sigmoid(output.adjacency_logits).detach().cpu().numpy()
    threshold_metrics = edge_metrics(probabilities, target_adjacency, threshold=threshold)
    topk_metrics = topk_edge_metrics(probabilities, target_adjacency)
    init_topk_metrics = candidate_topk_edge_metrics(probabilities, target_adjacency, init_edges)
    return {
        "path": str(path),
        "nodes": int(target_adjacency.shape[0]),
        "target_edges": int(np.triu(target_adjacency, k=1).sum()),
        "init_edge_f1": float(metadata.get("edge_f1", 0.0)),
        "threshold_precision": threshold_metrics["precision"],
        "threshold_recall": threshold_metrics["recall"],
        "threshold_f1": threshold_metrics["f1"],
        "topk_precision": topk_metrics["precision"],
        "topk_recall": topk_metrics["recall"],
        "topk_f1": topk_metrics["f1"],
        "init_topk_precision": init_topk_metrics["precision"],
        "init_topk_recall": init_topk_metrics["recall"],
        "init_topk_f1": init_topk_metrics["f1"],
    }


def edge_metrics(probabilities: np.ndarray, target_adjacency: np.ndarray, threshold: float) -> dict[str, float]:
    mask = np.triu(np.ones_like(target_adjacency, dtype=bool), k=1)
    predicted = (probabilities >= threshold) & mask
    target = target_adjacency.astype(bool) & mask
    return precision_recall_f1(predicted, target)


def topk_edge_metrics(probabilities: np.ndarray, target_adjacency: np.ndarray) -> dict[str, float]:
    mask = np.triu(np.ones_like(target_adjacency, dtype=bool), k=1)
    target = target_adjacency.astype(bool) & mask
    target_count = int(target.sum())
    if target_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    flat_probabilities = probabilities[mask]
    flat_target = target[mask]
    k = min(target_count, flat_probabilities.shape[0])
    order = np.argsort(-flat_probabilities)[:k]
    selected = np.zeros_like(flat_target, dtype=bool)
    selected[order] = True
    return precision_recall_f1(selected, flat_target)


def candidate_topk_edge_metrics(probabilities: np.ndarray, target_adjacency: np.ndarray, candidate_edges: np.ndarray) -> dict[str, float]:
    if candidate_edges.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    target = np.triu(target_adjacency.astype(bool), k=1)
    target_count = int(target.sum())
    if target_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    scored = []
    for left, right in candidate_edges.tolist():
        left = int(left)
        right = int(right)
        if left == right:
            continue
        if left > right:
            left, right = right, left
        scored.append((float(probabilities[left, right]), left, right))
    scored.sort(reverse=True, key=lambda item: item[0])
    k = min(target_count, len(scored))
    selected = np.zeros_like(target, dtype=bool)
    for _probability, left, right in scored[:k]:
        selected[left, right] = True
    return precision_recall_f1(selected, target)


def precision_recall_f1(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    true_positive = int(np.count_nonzero(predicted & target))
    false_positive = int(np.count_nonzero(predicted & ~target))
    false_negative = int(np.count_nonzero(~predicted & target))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "nodes",
        "target_edges",
        "init_edge_f1",
        "threshold_precision",
        "threshold_recall",
        "threshold_f1",
        "topk_precision",
        "topk_recall",
        "topk_f1",
        "init_topk_precision",
        "init_topk_recall",
        "init_topk_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
