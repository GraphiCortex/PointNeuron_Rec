from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SwcNode:
    node_id: int
    node_type: int
    x: float
    y: float
    z: float
    radius: float
    parent_id: int


@dataclass(frozen=True)
class SwcTree:
    path: Path
    nodes: tuple[SwcNode, ...]

    @property
    def node_ids(self) -> set[int]:
        return {node.node_id for node in self.nodes}

    @property
    def root_count(self) -> int:
        return sum(1 for node in self.nodes if node.parent_id == -1)

    @property
    def edge_count(self) -> int:
        return sum(1 for node in self.nodes if node.parent_id != -1)

    @property
    def missing_parent_ids(self) -> set[int]:
        ids = self.node_ids
        return {
            node.parent_id
            for node in self.nodes
            if node.parent_id != -1 and node.parent_id not in ids
        }

    @property
    def duplicate_node_ids(self) -> set[int]:
        seen: set[int] = set()
        duplicates: set[int] = set()
        for node in self.nodes:
            if node.node_id in seen:
                duplicates.add(node.node_id)
            seen.add(node.node_id)
        return duplicates

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.nodes:
            errors.append("SWC has no nodes")
        if self.root_count < 1:
            errors.append("SWC has no root node")
        if self.duplicate_node_ids:
            errors.append(f"SWC has duplicate node ids: {sorted(self.duplicate_node_ids)[:10]}")
        if self.missing_parent_ids:
            errors.append(f"SWC has missing parent ids: {sorted(self.missing_parent_ids)[:10]}")
        return errors


def parse_swc(path: str | Path) -> SwcTree:
    swc_path = Path(path)
    nodes: list[SwcNode] = []

    for line_number, raw_line in enumerate(swc_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f"{swc_path}:{line_number}: expected 7 SWC columns, got {len(parts)}")

        try:
            nodes.append(
                SwcNode(
                    node_id=int(float(parts[0])),
                    node_type=int(float(parts[1])),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    radius=float(parts[5]),
                    parent_id=int(float(parts[6])),
                )
            )
        except ValueError as exc:
            raise ValueError(f"{swc_path}:{line_number}: invalid SWC row: {raw_line}") from exc

    return SwcTree(path=swc_path, nodes=tuple(nodes))

