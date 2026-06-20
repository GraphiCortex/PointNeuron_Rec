from __future__ import annotations

from dataclasses import dataclass

from pointneuron.data.swc import SwcTree
from pointneuron.data.vaa3d_raw import Vaa3dHeader


@dataclass(frozen=True)
class AlignmentReport:
    out_of_bounds_node_ids: tuple[int, ...]

    @property
    def is_aligned(self) -> bool:
        return not self.out_of_bounds_node_ids


def check_swc_in_volume(swc: SwcTree, header: Vaa3dHeader) -> AlignmentReport:
    width, height, depth, _channels = header.dimensions
    out_of_bounds = tuple(
        node.node_id
        for node in swc.nodes
        if not (0 <= node.x < width and 0 <= node.y < height and 0 <= node.z < depth)
    )
    return AlignmentReport(out_of_bounds_node_ids=out_of_bounds)

