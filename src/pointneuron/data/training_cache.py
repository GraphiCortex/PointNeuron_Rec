from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from pointneuron.data.alignment import check_swc_in_volume
from pointneuron.data.gold166 import Gold166Sample
from pointneuron.data.point_cloud import PointCloud, SkeletonRecord, swc_to_skeleton_records, volume_to_point_cloud
from pointneuron.data.swc import parse_swc
from pointneuron.data.vaa3d_raw import read_header, read_volume


@dataclass(frozen=True)
class CachedTrainingRecord:
    sample_id: str
    output_path: Path
    point_count: int
    skeleton_node_count: int
    edge_count: int
    total_foreground_count: int


def build_training_record(
    sample: Gold166Sample,
    output_path: str | Path,
    threshold: int = 0,
    max_points: int = 4096,
    seed: int = 0,
) -> CachedTrainingRecord:
    if sample.volume_path is None:
        raise ValueError(f"Sample has no volume: {sample.sample_id}")

    header = read_header(sample.volume_path)
    swc = parse_swc(sample.swc_path)
    alignment = check_swc_in_volume(swc, header)
    if not alignment.is_aligned:
        raise ValueError(
            f"{sample.sample_id}: selected SWC has {len(alignment.out_of_bounds_node_ids)} out-of-bounds nodes"
        )

    volume = read_volume(sample.volume_path)
    point_cloud = volume_to_point_cloud(volume, threshold=threshold, max_points=max_points, seed=seed)
    skeleton = swc_to_skeleton_records(swc)

    points_array = point_cloud_to_array(point_cloud)
    skeleton_array = skeleton_to_array(skeleton)
    edge_index = skeleton_edge_index(skeleton)
    metadata = {
        "sample_id": sample.sample_id,
        "volume_path": str(sample.volume_path),
        "swc_path": str(sample.swc_path),
        "swc_selection": sample.swc_selection,
        "volume_dimensions": point_cloud.volume_dimensions,
        "threshold": threshold,
        "max_points": max_points,
        "seed": seed,
        "total_foreground_count": point_cloud.total_foreground_count,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=points_array,
        skeleton_nodes=skeleton_array,
        edge_index=edge_index,
        metadata=np.array(json.dumps(metadata), dtype=np.str_),
    )

    return CachedTrainingRecord(
        sample_id=sample.sample_id,
        output_path=path,
        point_count=len(point_cloud.points),
        skeleton_node_count=len(skeleton),
        edge_count=len(edge_index),
        total_foreground_count=point_cloud.total_foreground_count,
    )


def point_cloud_to_array(point_cloud: PointCloud) -> np.ndarray:
    return np.array(
        [
            [point.x, point.y, point.z, point.intensity]
            for point in point_cloud.points
        ],
        dtype=np.float32,
    )


def skeleton_to_array(skeleton: tuple[SkeletonRecord, ...]) -> np.ndarray:
    return np.array(
        [
            [node.node_id, node.x, node.y, node.z, node.radius, node.parent_id]
            for node in skeleton
        ],
        dtype=np.float32,
    )


def skeleton_edge_index(skeleton: tuple[SkeletonRecord, ...]) -> np.ndarray:
    node_id_to_index = {node.node_id: index for index, node in enumerate(skeleton)}
    edges: list[tuple[int, int]] = []
    for child_index, node in enumerate(skeleton):
        if node.parent_id == -1:
            continue
        parent_index = node_id_to_index.get(node.parent_id)
        if parent_index is None:
            raise ValueError(f"Missing parent id {node.parent_id} for node {node.node_id}")
        edges.append((parent_index, child_index))
    return np.array(edges, dtype=np.int64)
