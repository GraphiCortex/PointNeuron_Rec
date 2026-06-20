from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class TrainingCacheDataset:
    def __init__(self, record_paths: list[str | Path]):
        self.record_paths = [Path(path) for path in record_paths]

    def __len__(self) -> int:
        return len(self.record_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _torch()
        path = self.record_paths[index]
        record = np.load(path, allow_pickle=False)
        return {
            "points": torch.from_numpy(record["points"].astype(np.float32, copy=False)),
            "skeleton_nodes": torch.from_numpy(record["skeleton_nodes"].astype(np.float32, copy=False)),
            "edge_index": torch.from_numpy(record["edge_index"].astype(np.int64, copy=False)),
            "metadata": json.loads(str(record["metadata"])),
            "path": str(path),
        }


def load_split_paths(split_file: str | Path, split: str) -> list[str]:
    payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if split not in payload["splits"]:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(payload['splits'])}")
    return payload["splits"][split]


def collate_training_records(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    points = torch.stack([item["points"] for item in batch], dim=0)
    skeleton_lengths = torch.tensor([item["skeleton_nodes"].shape[0] for item in batch], dtype=torch.long)
    edge_lengths = torch.tensor([item["edge_index"].shape[0] for item in batch], dtype=torch.long)
    max_nodes = int(skeleton_lengths.max().item())
    max_edges = int(edge_lengths.max().item())

    skeleton_nodes = torch.zeros((len(batch), max_nodes, batch[0]["skeleton_nodes"].shape[1]), dtype=torch.float32)
    skeleton_mask = torch.zeros((len(batch), max_nodes), dtype=torch.bool)
    edge_index = torch.full((len(batch), max_edges, 2), fill_value=-1, dtype=torch.long)
    edge_mask = torch.zeros((len(batch), max_edges), dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        node_count = item["skeleton_nodes"].shape[0]
        edge_count = item["edge_index"].shape[0]
        skeleton_nodes[batch_index, :node_count] = item["skeleton_nodes"]
        skeleton_mask[batch_index, :node_count] = True
        edge_index[batch_index, :edge_count] = item["edge_index"]
        edge_mask[batch_index, :edge_count] = True

    return {
        "points": points,
        "skeleton_nodes": skeleton_nodes,
        "skeleton_mask": skeleton_mask,
        "edge_index": edge_index,
        "edge_mask": edge_mask,
        "metadata": [item["metadata"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for pointneuron.data.torch_dataset. "
            "Install a CUDA-compatible torch build before training."
        ) from exc
    return torch

