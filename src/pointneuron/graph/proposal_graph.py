"""Reference-free, per-crop proposal packets and deterministic candidates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np
import torch


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode("ascii")
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class ProposalGraphInput:
    positions: torch.Tensor
    radii: torch.Tensor
    objectness_logits: torch.Tensor
    local_centers: torch.Tensor
    decoder_features: torch.Tensor
    node_ids: torch.Tensor
    provenance: Mapping

    def validate(self):
        n = len(self.positions)
        arrays = [(self.positions, (n, 3)), (self.radii, (n, 1)),
                  (self.objectness_logits, (n, 2)), (self.local_centers, (n, 3)),
                  (self.decoder_features, (n, 128))]
        for value, shape in arrays:
            if value.shape != shape or not value.is_floating_point():
                raise ValueError(f"Expected floating proposal field {shape}")
            if value.device != self.positions.device or value.dtype != self.positions.dtype:
                raise ValueError("Proposal fields must share device and dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite proposal field")
        if bool((self.radii <= 0).any()):
            raise ValueError("Proposal radii must be positive")
        if self.node_ids.shape != (n,) or self.node_ids.dtype != torch.long:
            raise ValueError("Expected int64 node IDs [N]")
        if self.node_ids.device != self.positions.device or len(torch.unique(self.node_ids)) != n:
            raise ValueError("Node IDs must be unique and share the proposal device")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Expected provenance metadata")

    def node_features(self):
        self.validate()
        logit_difference = self.objectness_logits[:, 1:2]-self.objectness_logits[:, :1]
        return torch.cat((self.decoder_features, self.local_centers, self.radii/8., logit_difference), dim=-1)


def from_query_output(output, *, crop_center, patch_radius, provenance, node_ids=None):
    """Adapt one actual query output without filtering, detach, or policy conversion.

Feature digest is verified. Checkpoint/cloud digests are recorded provenance;
the caller must verify their source files when constructing a real cache.
"""
    required = ("upstream_schema", "normalization_policy", "checkpoint_sha256",
                "cloud_sha256", "feature_sha256", "partition")
    if any(not isinstance(provenance.get(key), str) or not provenance[key] for key in required):
        raise ValueError("Missing upstream checkpoint/normalization/data provenance")
    for key in ("checkpoint_sha256", "cloud_sha256", "feature_sha256"):
        value = provenance[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Expected lowercase SHA256 provenance")
    fields = (output.center_proposals, output.radius, output.objectness_logits,
              output.local_centers, output.decoder_features)
    if any(value.ndim != 3 or len(value) != 1 for value in fields):
        raise ValueError("The graph adapter requires exactly one crop")
    if not math.isfinite(patch_radius) or patch_radius <= 0:
        raise ValueError("Invalid patch radius")
    xyz = fields[0][0]
    center = torch.as_tensor(crop_center, dtype=xyz.dtype, device=xyz.device)
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError("Invalid crop center")
    if not torch.allclose(fields[3][0], (xyz-center)/patch_radius, atol=1e-4, rtol=1e-5):
        raise ValueError("Local/native coordinates disagree")
    if tensor_digest(fields[4][0]) != provenance["feature_sha256"]:
        raise ValueError("Decoder feature digest mismatch")
    metadata = dict(provenance, crop_center=center.detach().cpu().tolist(), patch_radius=float(patch_radius))
    ids = torch.arange(len(xyz), device=xyz.device) if node_ids is None else node_ids
    result = ProposalGraphInput(*(value[0] for value in fields), ids, MappingProxyType(metadata))
    result.validate()
    return result


@dataclass(frozen=True)
class CandidateGraph:
    edges: torch.Tensor  # [E,2], each undirected pair once; no self loops.
    neighbors: int
    radius: float

    def validate(self, packet):
        if self.edges.ndim != 2 or self.edges.shape[1] != 2 or self.edges.dtype != torch.long:
            raise ValueError("Expected int64 candidate pairs [E,2]")
        if self.edges.device != packet.positions.device:
            raise ValueError("Candidate/proposal devices disagree")
        if len(self.edges):
            if bool((self.edges < 0).any()) or bool((self.edges >= len(packet.positions)).any()):
                raise ValueError("Candidate index out of range")
            if bool((self.edges[:, 0] == self.edges[:, 1]).any()):
                raise ValueError("Self messages are not reconstruction candidates")
            canonical = self.edges.sort(dim=1).values
            if len(torch.unique(canonical, dim=0)) != len(self.edges):
                raise ValueError("Duplicate undirected candidates")


def build_candidate_graph(positions, node_ids, *, neighbors=16, radius=8.):
    """CPU geometry indexing only; stable IDs settle exact-distance ties.

No labels or reference identities are accepted. Index selection is detached;
relative features are computed later from the original tensors for gradients.
Working distance storage is O(N), not an N-by-N matrix. All-pair radius edges
can themselves be dense; no hidden cap discards them.
"""
    if not isinstance(neighbors, int) or isinstance(neighbors, bool) or neighbors < 1:
        raise ValueError("Neighbor count must be a positive integer")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Candidate radius must be positive and finite")
    if positions.ndim != 2 or positions.shape[1] != 3 or not positions.is_floating_point():
        raise ValueError("Expected floating positions [N,3]")
    if node_ids.shape != (len(positions),) or node_ids.dtype != torch.long:
        raise ValueError("Invalid node IDs")
    xyz = positions.detach().cpu().double().numpy()
    ids = node_ids.detach().cpu().numpy()
    if not np.isfinite(xyz).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("Nonfinite coordinates or duplicate IDs")
    pairs = set()
    for i in np.argsort(ids):
        distance2 = np.sum((xyz-xyz[i])**2, axis=1)
        ordered = np.lexsort((ids, distance2))
        nearest = ordered[ordered != i][:neighbors]
        radial = np.flatnonzero((distance2 <= (radius+1e-6)**2) & (np.arange(len(ids)) != i))
        for j in np.union1d(nearest, radial):
            pairs.add((int(i), int(j)) if ids[i] < ids[j] else (int(j), int(i)))
    pairs = sorted(pairs, key=lambda pair: (ids[pair[0]], ids[pair[1]]))
    edges = torch.tensor(pairs, dtype=torch.long, device=positions.device).reshape(-1, 2)
    return CandidateGraph(edges, neighbors, float(radius))


def message_graph(packet, candidates):
    """Source->receiver edges and five signed geometric attributes."""
    candidates.validate(packet)
    ids_order = torch.argsort(packet.node_ids)
    self_edges = torch.stack((ids_order, ids_order), dim=1)
    edges = torch.cat((candidates.edges, candidates.edges.flip(1), self_edges), dim=0)
    source, receiver = edges.unbind(1)
    delta = (packet.positions[source]-packet.positions[receiver])/8.
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    radius_difference = (packet.radii[source]-packet.radii[receiver])/8.
    attributes = torch.cat((delta, distance, radius_difference), dim=-1)
    return edges, attributes
