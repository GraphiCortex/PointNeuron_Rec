"""Native per-crop GATv2 prototype and matched pairwise edge-head control.

Graph attention updates features. A separate symmetric head predicts independent
undirected edge logits. Neither scorer imports reference targets or SWCs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pointneuron.graph.proposal_graph import message_graph

SCHEMA = "pointneuron2-gatv2-connectivity-v1"
CONTROL_SCHEMA = "pointneuron2-pairwise-connectivity-v1"


@dataclass(frozen=True)
class ConnectivityConfig:
    mode: str = "gatv2"
    hidden_channels: int = 128
    heads: int = 4
    layers: int = 2
    seed: int = 0

    def __post_init__(self):
        if self.mode not in ("gatv2", "pairwise"):
            raise ValueError("Unknown connectivity scorer")
        for value in (self.hidden_channels, self.heads, self.layers):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("Positive integer architecture dimensions required")
        if self.hidden_channels % self.heads:
            raise ValueError("Hidden channels must divide into heads")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("Invalid model seed")


class NativeGATv2Block(nn.Module):
    def __init__(self, channels=128, heads=4, edge_channels=5):
        super().__init__()
        if channels < 1 or heads < 1 or channels % heads:
            raise ValueError("Invalid attention dimensions")
        self.channels, self.heads, self.head_channels = channels, heads, channels//heads
        self.source = nn.Linear(channels, channels, bias=False)
        self.receiver = nn.Linear(channels, channels, bias=False)
        self.edge = nn.Linear(edge_channels, channels, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, self.head_channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.norm = nn.LayerNorm(channels)
        for layer in (self.source, self.receiver, self.edge):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.attention)

    def aggregate(self, features, edges, attributes):
        """Source->receiver, per-receiver/head softmax; no dropout or BN."""
        n = len(features)
        if features.shape != (n, self.channels) or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Invalid GATv2 feature/edge shape")
        if attributes.shape != (len(edges), self.edge.in_features):
            raise ValueError("Invalid geometric edge attributes")
        source, receiver = edges.unbind(1)
        values = self.source(features).reshape(n, self.heads, self.head_channels)
        queries = self.receiver(features).reshape(n, self.heads, self.head_channels)
        relation = self.edge(attributes).reshape(-1, self.heads, self.head_channels)
        logits = (F.leaky_relu(queries[receiver]+values[source]+relation, negative_slope=.2)
                  * self.attention).sum(-1)
        index = receiver[:, None].expand(-1, self.heads)
        maxima = features.new_full((n, self.heads), -torch.inf)
        maxima.scatter_reduce_(0, index, logits.detach(), reduce="amax", include_self=True)
        unnormalized = torch.exp(logits-maxima[receiver])
        denominator = features.new_zeros((n, self.heads)).index_add(0, receiver, unnormalized)
        alpha = unnormalized/denominator[receiver].clamp_min(torch.finfo(features.dtype).tiny)
        aggregate = features.new_zeros((n, self.heads, self.head_channels))
        aggregate.index_add_(0, receiver, alpha[..., None]*values[source])
        return aggregate.reshape(n, self.channels)+self.bias, alpha

    def forward(self, features, edges, attributes):
        aggregate, _ = self.aggregate(features, edges, attributes)
        return F.relu(self.norm(features+aggregate))


@dataclass(frozen=True)
class ConnectivityScores:
    node_features: torch.Tensor
    edge_logits: torch.Tensor


def symmetric_edge_features(packet, hidden, edges):
    u, v = edges.unbind(1)
    delta = (packet.positions[u]-packet.positions[v])/8.
    confidence = packet.objectness_logits.softmax(-1)[:, 1:2]
    return torch.cat((hidden[u]+hidden[v], (hidden[u]-hidden[v]).abs(), delta.abs(),
                      torch.linalg.vector_norm(delta, dim=-1, keepdim=True),
                      (packet.radii[u]+packet.radii[v])/8., (packet.radii[u]-packet.radii[v]).abs()/8.,
                      confidence[u]+confidence[v], (confidence[u]-confidence[v]).abs()), dim=-1)


class GATv2Connectivity(nn.Module):
    def __init__(self, config=ConnectivityConfig()):
        super().__init__()
        self.config = config
        # Independent seeded blocks give GATv2 and the control identical shared
        # stem/head initial weights, without disturbing the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.stem = nn.Sequential(nn.Linear(133, config.hidden_channels), nn.ReLU())
            torch.random.default_generator.manual_seed(config.seed+101)
            self.blocks = nn.ModuleList([NativeGATv2Block(config.hidden_channels, config.heads)
                                        for _ in range(config.layers if config.mode == "gatv2" else 0)])
            torch.random.default_generator.manual_seed(config.seed+202)
            self.edge_head = nn.Sequential(nn.Linear(2*config.hidden_channels+8, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, packet, candidates):
        packet.validate()
        candidates.validate(packet)
        hidden = self.stem(packet.node_features())
        if len(hidden) and self.blocks:
            edges, attributes = message_graph(packet, candidates)
            for block in self.blocks:
                hidden = block(hidden, edges, attributes)
        edge_features = symmetric_edge_features(packet, hidden, candidates.edges)
        logits = self.edge_head(edge_features).squeeze(-1)
        if not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite connectivity outputs")
        return ConnectivityScores(hidden, logits)

    def model_spec(self):
        return {"schema": SCHEMA if self.config.mode == "gatv2" else CONTROL_SCHEMA,
                "config": asdict(self.config), "node_input_dim": 133, "edge_input_dim": 2*self.config.hidden_channels+8,
                "actual_gatv2_layers": len(self.blocks), "backend": "native_pytorch",
                "normalization": "per_node_layernorm_in_gatv2_blocks", "dropout": 0.}


def connectivity_checkpoint(model, *, provenance, step=0, optimizer=None, scheduler=None):
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Invalid update count")
    if scheduler is not None and optimizer is None:
        raise ValueError("Scheduler state requires an optimizer")
    return {"schema": model.model_spec()["schema"], "spec": model.model_spec(),
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "step": step, "provenance": deepcopy(dict(provenance)),
            "optimizer": None if optimizer is None else deepcopy(optimizer.state_dict()),
            "scheduler": None if scheduler is None else deepcopy(scheduler.state_dict()),
            "resumable_training_state": optimizer is not None,
            "rng": {"torch_cpu": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}}


def load_connectivity_checkpoint(checkpoint, *, expected_mode="gatv2", device="cpu"):
    expected_schema = SCHEMA if expected_mode == "gatv2" else CONTROL_SCHEMA if expected_mode == "pairwise" else None
    if expected_schema is None or checkpoint.get("schema") != expected_schema:
        raise ValueError("Expected explicit connectivity checkpoint identity")
    config = ConnectivityConfig(**checkpoint["spec"]["config"])
    if config.mode != expected_mode:
        raise ValueError("Connectivity scorer mode differs")
    model = GATv2Connectivity(config)
    if checkpoint["spec"] != model.model_spec():
        raise ValueError("Connectivity specification differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("Nonfinite connectivity checkpoint")
    return model.to(device).eval()
