"""Explicit PointNeXt native prototype identity; legacy DGCNN factories unchanged."""
from __future__ import annotations

from dataclasses import asdict
import torch
from torch import nn

from pointneuron.models.pointnext_native import PointNextNativeConfig, PointNextNativeEncoder, normalize_native_inputs
from pointneuron.models.query_proposal import QueryDecoderConfig, QueryProposalHead
from pointneuron.data.curve_targets import TARGET_SCHEMA

CHECKPOINT_SCHEMA = "pointneuron2-pointnext-query-v1"


class PointNextQueryModel(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = PointNextNativeEncoder()
            torch.random.default_generator.manual_seed(seed+101)
            self.proposal = QueryProposalHead(QueryDecoderConfig(in_channels=self.encoder.output_dim, dropout=0.))

    def forward(self, points, patch_centers, image_scales, patch_radius=48.):
        local, attrs = normalize_native_inputs(points, patch_centers, image_scales, patch_radius)
        features = self.encoder(local, attrs)
        return self.proposal(points, features, patch_centers, patch_radius)

    def model_spec(self):
        return {"kind": CHECKPOINT_SCHEMA, "encoder": asdict(self.encoder.config),
                "decoder": asdict(self.proposal.config), "backend": "native_pytorch_float32"}


def model_checkpoint(model, *, seed, step, recipe, provenance):
    return {"schema": CHECKPOINT_SCHEMA, "target_schema": TARGET_SCHEMA, "model_spec": model.model_spec(),
            "seed": seed, "step": step, "recipe": recipe, "provenance": provenance, "model": model.state_dict()}


def load_model_checkpoint(checkpoint, device="cpu"):
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("target_schema") != TARGET_SCHEMA:
        raise ValueError("Expected an explicit native PointNeXt/query checkpoint")
    model = PointNextQueryModel(checkpoint["seed"])
    if checkpoint.get("model_spec") != model.model_spec():
        raise ValueError("PointNeXt/query checkpoint recipe or backend identity differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError("Nonfinite checkpoint weights/statistics")
    return model.to(device).eval()
