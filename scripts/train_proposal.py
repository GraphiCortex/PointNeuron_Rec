from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.torch_dataset import TrainingCacheDataset, collate_training_records, load_split_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the PointNeuron skeleton proposal module.")
    parser.add_argument("--split-file", required=True, help="Path to split JSON.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to train on.")
    parser.add_argument("--val-split", default="val", choices=["train", "val", "test"], help="Split to evaluate after each epoch.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation evaluation.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--k", type=int, default=20, help="kNN neighbors for EdgeConv.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--positive-distance", type=float, default=6.0, help="Voxel distance for positive proposals.")
    parser.add_argument("--radius-scale", type=float, default=1.5, help="Also mark positives inside radius * this value.")
    parser.add_argument("--positive-class-weight", type=float, default=8.0, help="Class weight for positive objectness.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count. Keep 0 on Windows at first.")
    parser.add_argument("--drop-last", action="store_true", help="Drop the final incomplete batch.")
    parser.add_argument("--limit-batches", type=int, help="Optional maximum batches per epoch for quick checks.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--checkpoint", help="Optional path to save a checkpoint after training.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device to run on.")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError:
        print("PyTorch is required. Install a CUDA-compatible torch build before training.")
        return 2

    from pointneuron.models.dgcnn import DGCNNEncoder
    from pointneuron.models.proposal import SkeletonProposalHead
    from pointneuron.models.proposal_loss import build_skeleton_proposal_targets, skeleton_proposal_loss

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but torch.cuda.is_available() is false.")
        return 2

    paths = load_split_paths(args.split_file, args.split)
    dataset = TrainingCacheDataset(paths)
    if len(dataset) == 0:
        print(f"No records found for split {args.split!r}.")
        return 2

    drop_last = args.drop_last or (args.batch_size > 1 and len(dataset) % args.batch_size == 1)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_training_records,
        pin_memory=(device == "cuda"),
        drop_last=drop_last,
    )
    val_loader = None
    val_dataset = None
    if not args.no_val:
        val_paths = load_split_paths(args.split_file, args.val_split)
        val_dataset = TrainingCacheDataset(val_paths)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_training_records,
                pin_memory=(device == "cuda"),
                drop_last=False,
            )

    encoder = DGCNNEncoder(k=args.k).to(device)
    proposal = SkeletonProposalHead(in_channels=encoder.output_dim).to(device)
    parameters = list(encoder.parameters()) + list(proposal.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device: {device}")
    print(f"records: {len(dataset)}")
    if val_dataset is not None:
        print(f"val_records: {len(val_dataset)}")
    print(f"batch_size: {args.batch_size}")
    print(f"drop_last: {drop_last}")
    print(f"amp: {use_amp}")
    print(f"k: {args.k}")
    print(f"geometric_feature_dim: {encoder.geometric_feature_dim}")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proposal.train()
        running = RunningAverages()

        for batch_index, batch in enumerate(train_loader, start=1):
            if args.limit_batches is not None and batch_index > args.limit_batches:
                break

            points = batch["points"].to(device, non_blocking=True)
            skeleton_nodes = batch["skeleton_nodes"].to(device, non_blocking=True)
            skeleton_mask = batch["skeleton_mask"].to(device, non_blocking=True)

            with torch.no_grad():
                targets = build_skeleton_proposal_targets(
                    points=points,
                    skeleton_nodes=skeleton_nodes,
                    skeleton_mask=skeleton_mask,
                    positive_distance=args.positive_distance,
                    radius_scale=args.radius_scale,
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                features = encoder(points)
                output = proposal(points, features)
                loss = skeleton_proposal_loss(
                    output=output,
                    targets=targets,
                    points=points,
                    positive_class_weight=args.positive_class_weight,
                )
            scaler.scale(loss.total).backward()
            scaler.step(optimizer)
            scaler.update()

            running.update(
                total=float(loss.total.detach().item()),
                objectness=float(loss.objectness.item()),
                center=float(loss.center.item()),
                radius=float(loss.radius.item()),
                positive_count=loss.positive_count,
                total_count=loss.total_count,
            )

        message = f"epoch={epoch} {format_metrics('train', running)}"
        if val_loader is not None:
            validation = evaluate(
                encoder=encoder,
                proposal=proposal,
                loader=val_loader,
                device=device,
                positive_distance=args.positive_distance,
                radius_scale=args.radius_scale,
                positive_class_weight=args.positive_class_weight,
                use_amp=use_amp,
                torch=torch,
                build_skeleton_proposal_targets=build_skeleton_proposal_targets,
                skeleton_proposal_loss=skeleton_proposal_loss,
            )
            message += f" {format_metrics('val', validation)}"
        if device == "cuda":
            message += f" cuda_reserved_mb={torch.cuda.memory_reserved() / 1024 / 1024:.2f}"
        print(message)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "proposal": proposal.state_dict(),
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"checkpoint: {checkpoint_path}")

    return 0


class RunningAverages:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0
        self.positive_count = 0
        self.total_count = 0

    def update(
        self,
        total: float,
        objectness: float,
        center: float,
        radius: float,
        positive_count: int,
        total_count: int,
    ) -> None:
        self.count += 1
        self.sums["total"] = self.sums.get("total", 0.0) + total
        self.sums["objectness"] = self.sums.get("objectness", 0.0) + objectness
        self.sums["center"] = self.sums.get("center", 0.0) + center
        self.sums["radius"] = self.sums.get("radius", 0.0) + radius
        self.positive_count += positive_count
        self.total_count += total_count

    def mean(self, name: str) -> float:
        if self.count == 0:
            return 0.0
        return self.sums.get(name, 0.0) / self.count


def evaluate(
    encoder,
    proposal,
    loader,
    device: str,
    positive_distance: float,
    radius_scale: float,
    positive_class_weight: float,
    use_amp: bool,
    torch,
    build_skeleton_proposal_targets,
    skeleton_proposal_loss,
) -> RunningAverages:
    encoder.eval()
    proposal.eval()
    running = RunningAverages()
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            skeleton_nodes = batch["skeleton_nodes"].to(device, non_blocking=True)
            skeleton_mask = batch["skeleton_mask"].to(device, non_blocking=True)
            targets = build_skeleton_proposal_targets(
                points=points,
                skeleton_nodes=skeleton_nodes,
                skeleton_mask=skeleton_mask,
                positive_distance=positive_distance,
                radius_scale=radius_scale,
            )
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                features = encoder(points)
                output = proposal(points, features)
                loss = skeleton_proposal_loss(
                    output=output,
                    targets=targets,
                    points=points,
                    positive_class_weight=positive_class_weight,
                )
            running.update(
                total=float(loss.total.detach().item()),
                objectness=float(loss.objectness.item()),
                center=float(loss.center.item()),
                radius=float(loss.radius.item()),
                positive_count=loss.positive_count,
                total_count=loss.total_count,
            )
    return running


def format_metrics(prefix: str, running: RunningAverages) -> str:
    return (
        f"{prefix}_loss={running.mean('total'):.4f} "
        f"{prefix}_objectness={running.mean('objectness'):.4f} "
        f"{prefix}_center={running.mean('center'):.6f} "
        f"{prefix}_radius={running.mean('radius'):.4f} "
        f"{prefix}_positives={running.positive_count}/{running.total_count}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
