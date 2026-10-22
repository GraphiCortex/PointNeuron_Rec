"""Strict, non-repairing SWC forest reader for the shared evaluator.

Coordinates and radii remain in the file's native frame. Declared XYZ spacing
is applied by geometry, once; anisotropic scaling does not infer a scalar
physical radius. An empty SWC is represented explicitly, not made into a tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


TYPE_POLICIES = ("nonnegative", "opaque_integer")


class SwcValidationError(ValueError):
    """The supplied artifact cannot be interpreted as an SWC forest."""


@dataclass(frozen=True)
class Morphology:
    path: Path | None
    ids: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    radii: np.ndarray
    parents: np.ndarray  # Row indices, -1 for roots; never external node IDs.

    @property
    def edges(self) -> np.ndarray:
        children = np.flatnonzero(self.parents >= 0)
        return np.column_stack((children, self.parents[children])).astype(np.int64, copy=False)

    @property
    def degrees(self) -> np.ndarray:
        result = np.zeros(self.node_count, dtype=np.int64)
        np.add.at(result, self.edges.ravel(), 1)
        return result

    @property
    def segments(self) -> np.ndarray:
        return self.xyz[self.edges]

    @property
    def isolated_points(self) -> np.ndarray:
        return self.xyz[self.degrees == 0]

    @property
    def total_length(self) -> float:
        return float(np.linalg.norm(self.segments[:, 1] - self.segments[:, 0], axis=1).sum())

    @property
    def node_count(self) -> int:
        return len(self.xyz)

    @property
    def component_count(self) -> int:
        return int(np.count_nonzero(self.parents == -1))


def empty_morphology() -> Morphology:
    return Morphology(None, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                      np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64),
                      np.empty(0, dtype=np.int64))


def _integer(token: str, where: str) -> int:
    # int(float(...)) silently accepts fractional IDs and loses large integers.
    if not re.fullmatch(r"[+-]?\d+", token):
        raise SwcValidationError(f"{where}: expected an integer, received {token!r}")
    value = int(token)
    if value < np.iinfo(np.int64).min or value > np.iinfo(np.int64).max:
        raise SwcValidationError(f"{where}: integer is outside int64 range")
    return value


def parse_swc(path: str | Path, *, type_policy: str = "nonnegative") -> Morphology:
    """Read unordered rows; reject malformed data rather than silently repair it.

    Positive node IDs and integer nonnegative type codes are required by default.
    Explicit opaque_integer mode preserves signed type metadata for geometry-only
    reference agreement; it does not assign anatomical meaning, remove nodes,
    or change any other validation. Output readers keep the strict default. Roots
    have parent -1; multiple roots and zero radii are allowed and reported.
    Inline comments are accepted. Coordinates may be negative but must be finite.
    """
    if type_policy not in TYPE_POLICIES:
        raise SwcValidationError(f"Unknown SWC type policy: {type_policy!r}")
    path = Path(path)
    ids, types, coordinates, radii, parent_ids = [], [], [], [], []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise SwcValidationError(f"{path}: SWC is not valid UTF-8 text") from exc
    for line_number, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{path}:{line_number}"
        parts = line.split()
        if len(parts) != 7:
            raise SwcValidationError(f"{where}: expected 7 columns, received {len(parts)}")
        node_id, node_type, parent_id = (_integer(parts[i], where) for i in (0, 1, 6))
        if node_id <= 0 or (type_policy == "nonnegative" and node_type < 0):
            raise SwcValidationError(f"{where}: node IDs must be positive and types nonnegative")
        try:
            x, y, z, radius = map(float, parts[2:6])
        except ValueError as exc:
            raise SwcValidationError(f"{where}: invalid coordinate or radius") from exc
        if not np.isfinite([x, y, z, radius]).all() or radius < 0:
            raise SwcValidationError(f"{where}: coordinates/radius must be finite; radius must be nonnegative")
        ids.append(node_id)
        types.append(node_type)
        coordinates.append((x, y, z))
        radii.append(radius)
        parent_ids.append(parent_id)
    if len(set(ids)) != len(ids):
        raise SwcValidationError(f"{path}: duplicate node IDs")
    index = {node_id: row for row, node_id in enumerate(ids)}
    parents = []
    for node_id, parent_id in zip(ids, parent_ids):
        if parent_id == node_id:
            raise SwcValidationError(f"{path}: node {node_id} is its own parent")
        if parent_id != -1 and parent_id not in index:
            raise SwcValidationError(f"{path}: node {node_id} has missing parent {parent_id}")
        parents.append(-1 if parent_id == -1 else index[parent_id])
    # Iterative parent traversal handles deep arbors without Python recursion.
    completed: set[int] = set()
    for start in range(len(ids)):
        current, visiting = start, set()
        while current != -1 and current not in completed:
            if current in visiting:
                raise SwcValidationError(f"{path}: cycle through node {ids[current]}")
            visiting.add(current)
            current = parents[current]
        completed.update(visiting)
    return Morphology(path, np.asarray(ids, dtype=np.int64), np.asarray(types, dtype=np.int64),
                      np.asarray(coordinates, dtype=np.float64).reshape((-1, 3)),
                      np.asarray(radii, dtype=np.float64), np.asarray(parents, dtype=np.int64))
