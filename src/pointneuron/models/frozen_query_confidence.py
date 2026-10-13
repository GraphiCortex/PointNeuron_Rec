"""Fixed-feature linear confidence diagnostic; no geometry or model updates."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "pointneuron2-frozen-query-confidence-v1"


class FrozenQueryConfidence(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        if weight.shape != (2, 128) or bias.shape != (2,):
            raise ValueError("Expected saved two-class confidence projection [2,128]")
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(128, 2)
        self.linear.load_state_dict({"weight": weight.detach().clone(), "bias": bias.detach().clone()}, strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in self.parameters()):
            raise ValueError("Nonfinite confidence initialization")

    def forward(self, features):
        if (features.shape != (2, 512, 128) or features.dtype != torch.float32
                or not bool(torch.isfinite(features).all()) or features.requires_grad):
            raise ValueError("Expected detached fixed float32 features [2,512,128]")
        return self.linear(features)


def fixed_confidence_loss(logits, labels, recipe):
    if logits.shape != (2, 512, 2) or labels.shape != (2, 512) or labels.dtype != torch.long:
        raise ValueError("Expected two fixed-crop label vectors")
    if bool(((labels != 0) & (labels != 1)).any()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Invalid fixed labels or logits")
    weight = logits.new_tensor([1., recipe["positive_class_weight"]])
    losses = torch.stack([F.cross_entropy(logits[i], labels[i], weight=weight) for i in range(2)])
    return recipe["confidence_weight"] * losses.mean(), losses


def confidence_checkpoint(head, *, step, recipe, provenance):
    return {"schema": SCHEMA, "step": step, "recipe": recipe, "provenance": provenance,
            "model": head.state_dict(), "model_spec": {"kind": "fixed_feature_linear_128_to_2", "trainable_parameters": 258}}


def load_confidence_checkpoint(checkpoint, device="cpu"):
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("model_spec") != {"kind": "fixed_feature_linear_128_to_2", "trainable_parameters": 258}:
        raise ValueError("Expected fixed-feature confidence diagnostic checkpoint")
    state = checkpoint["model"]
    head = FrozenQueryConfidence(state["linear.weight"], state["linear.bias"])
    head.load_state_dict(state, strict=True)
    return head.to(device).eval()
