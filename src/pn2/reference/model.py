"""Thin adapter around retained PN1 operators; no checkpoint or training policy."""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from .dgcnn import DGCNNEncoder
from .proposal import SkeletonProposalHead


class ReferenceProposalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DGCNNEncoder(k=20)
        self.proposal = SkeletonProposalHead(in_channels=self.encoder.output_dim + 3, coordinate_mode="local")

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 4 or points.shape[1] < 2 or points.shape[0] < 1:
            raise ValueError("Reference model needs [B,N,4] with a nonempty batch and at least two input points")
        if not torch.isfinite(points).all():
            raise ValueError("Reference model input must be finite")
        return self.proposal(points, self.encoder(points))


def fresh_reference(seed: int, *, conservative: bool = True):
    """Fresh CPU parameters, no inherited weights or statistics. No optimizer.

    Conservative initialization matches the retained reference starting point.
    Tests also exercise a nonzero final layer to verify geometry and feature flow.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = ReferenceProposalModel().cpu()
        if conservative:
            model.proposal.initialize_conservative()
    return model.eval()


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict_points(model, points):
    """Ordinary evaluation only; return all pointwise proposals without filtering."""
    if any(module.training for module in model.modules()):
        raise ValueError("Prediction requires explicit eval() normalization; no per-crop policy substitution")
    if points.dtype != torch.float32 or points.device.type != "cpu":
        raise ValueError("This bounded execution interface expects CPU float32 points")
    with torch.inference_mode():
        return model(points)
