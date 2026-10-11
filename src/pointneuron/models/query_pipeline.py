"""Versioned PN2 proposal comparison checkpoint and graph-adapter contract."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from pointneuron.data.curve_targets import TARGET_SCHEMA
from pointneuron.data.vaa3d_raw import VOLUME_READER_VERSION
from pointneuron.models.dgcnn import DGCNNEncoder
from pointneuron.models.proposal import SkeletonProposalHead
from pointneuron.models.query_proposal import QueryDecoderConfig, QueryProposalHead

CHECKPOINT_SCHEMA = "pointneuron2-query-comparison-v1"


def build_models(config, arm, seed):
    if arm not in ("pointwise", "query"):
        raise ValueError("Unknown comparison arm")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        encoder = DGCNNEncoder(k=config["encoder_k"])
        torch.random.default_generator.manual_seed(seed + 101)
        if arm == "query":
            head_config = QueryDecoderConfig(**config["decoder"])
            if head_config.in_channels != encoder.output_dim:
                raise ValueError("Decoder does not match encoder feature size")
            head = QueryProposalHead(head_config)
        else:
            head = SkeletonProposalHead(in_channels=encoder.output_dim + 3, coordinate_mode="local")
            head.initialize_conservative()
    return encoder, head


def forward_proposals(head, arm, points, features, patch_centers, patch_radius):
    if arm == "query":
        return head(points, features, patch_centers, patch_radius)
    if arm == "pointwise":
        return head(points, features)
    raise ValueError("Unknown comparison arm")


@dataclass(frozen=True)
class GraphProposalFeatures:
    features: torch.Tensor
    input_indices: torch.Tensor
    association_distances: torch.Tensor


def associate_encoder_features(centers, points, features):
    if centers.ndim != 3 or centers.shape[-1] != 3 or points.ndim != 3 or points.shape[-1] != 4:
        raise ValueError("Expected batched proposal and input geometry")
    if features.ndim != 3 or features.shape[:2] != points.shape[:2] or len(centers) != len(points):
        raise ValueError("Feature/geometry shape mismatch")
    with torch.no_grad():
        distances, indices = torch.cdist(centers, points[..., :3]).min(-1)
    selected = features.gather(1, indices[..., None].expand(-1, -1, features.shape[-1]))
    return GraphProposalFeatures(selected, indices, distances)


def comparison_checkpoint(encoder, head, config, arm, epoch, **provenance):
    return {"schema": CHECKPOINT_SCHEMA, "target_schema": TARGET_SCHEMA,
            "volume_reader_version": VOLUME_READER_VERSION, "arm": arm, "config": config,
            "epoch": epoch, "encoder": encoder.state_dict(), "proposal": head.state_dict(),
            "provenance": provenance}


def models_from_checkpoint(checkpoint, device="cpu"):
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("target_schema") != TARGET_SCHEMA:
        raise ValueError("Unsupported proposal comparison checkpoint")
    if checkpoint.get("volume_reader_version") != VOLUME_READER_VERSION:
        raise ValueError("Checkpoint requires the corrected volume reader")
    config, arm = checkpoint["config"], checkpoint["arm"]
    encoder, head = build_models(config, arm, config["seed"])
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    head.load_state_dict(checkpoint["proposal"], strict=True)
    if any(not bool(torch.isfinite(t).all()) for t in [*encoder.state_dict().values(), *head.state_dict().values()]):
        raise ValueError("Nonfinite checkpoint weights")
    return encoder.to(device).eval(), head.to(device).eval()
