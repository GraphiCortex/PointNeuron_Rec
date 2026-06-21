from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the PointNeuron connectivity graph auto-encoder.")
    parser.add_argument("--records", nargs="+", required=True, help="Connectivity record .npz files.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--positive-weight", type=float, help="Optional positive class weight. Defaults to per-graph imbalance.")
    parser.add_argument("--normalize-node-features", action="store_true", help="Standardize node features per graph before training.")
    parser.add_argument("--checkpoint", default="tmp/checkpoints/connectivity_gae.pt", help="Checkpoint path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before training.")
        return 2

    from pointneuron.models.connectivity import ConnectivityGAE, adjacency_reconstruction_loss

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but torch.cuda.is_available() is false.")
        return 2

    records = [load_record(Path(path), normalize=args.normalize_node_features, torch=torch, device=device) for path in args.records]
    if not records:
        print("No connectivity records were provided.")
        return 2

    input_dim = int(records[0]["node_features"].shape[1])
    for record in records:
        if int(record["node_features"].shape[1]) != input_dim:
            raise ValueError("All connectivity records must have the same node feature dimension")

    model = ConnectivityGAE(in_channels=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}")
    print(f"records: {len(records)}")
    print(f"node_feature_dim: {input_dim}")
    print(f"normalize_node_features: {args.normalize_node_features}")

    best_loss: float | None = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = RunningMetrics()
        for record in records:
            optimizer.zero_grad(set_to_none=True)
            output = model(record["node_features"], record["init_adjacency"])
            loss = adjacency_reconstruction_loss(
                output.adjacency_logits,
                record["target_adjacency"],
                positive_weight=args.positive_weight,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().item())
            epoch_metrics.update(edge_metrics(output.adjacency_logits.detach(), record["target_adjacency"], torch=torch))

        mean_loss = epoch_loss / len(records)
        if best_loss is None or mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "loss": mean_loss,
                    "input_dim": input_dim,
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 20) == 0:
            print(
                f"epoch={epoch} loss={mean_loss:.6f} "
                f"edge_precision={epoch_metrics.mean('precision'):.4f} "
                f"edge_recall={epoch_metrics.mean('recall'):.4f} "
                f"edge_f1={epoch_metrics.mean('f1'):.4f}"
            )

    print(f"best_epoch: {best_epoch}")
    print(f"best_loss: {best_loss:.6f}")
    print(f"checkpoint: {checkpoint_path}")
    return 0


def load_record(path: Path, normalize: bool, torch, device: str) -> dict:
    payload = np.load(path, allow_pickle=False)
    node_features = payload["node_features"].astype(np.float32, copy=False)
    if normalize:
        mean = node_features.mean(axis=0, keepdims=True)
        std = node_features.std(axis=0, keepdims=True)
        node_features = (node_features - mean) / np.maximum(std, 1e-6)
    return {
        "path": path,
        "node_features": torch.from_numpy(node_features).to(device),
        "init_adjacency": torch.from_numpy(payload["init_adjacency"].astype(np.float32, copy=False)).to(device),
        "target_adjacency": torch.from_numpy(payload["target_adjacency"].astype(np.float32, copy=False)).to(device),
    }


def edge_metrics(logits, target_adjacency, torch) -> dict[str, float]:
    target = target_adjacency.bool()
    mask = torch.triu(torch.ones_like(target, dtype=torch.bool), diagonal=1)
    predicted = (torch.sigmoid(logits) >= 0.5) & mask
    target = target & mask
    true_positive = int((predicted & target).sum().item())
    false_positive = int((predicted & ~target).sum().item())
    false_negative = int((~predicted & target).sum().item())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


class RunningMetrics:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.values[key].append(float(value))

    def mean(self, key: str) -> float:
        values = self.values[key]
        return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
