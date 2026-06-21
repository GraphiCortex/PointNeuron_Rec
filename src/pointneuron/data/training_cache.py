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


@dataclass(frozen=True)
class PatchCacheConfig:
    patches_per_sample: int = 8
    patch_radius: int = 96
    max_points: int = 2048
    min_points: int = 256


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


def build_patch_training_records(
    sample: Gold166Sample,
    output_dir: str | Path,
    sample_index: int,
    config: PatchCacheConfig,
    threshold: int = 0,
    seed: int = 0,
) -> list[CachedTrainingRecord]:
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
    skeleton = swc_to_skeleton_records(swc)
    skeleton_array = skeleton_to_array(skeleton)
    data = volume_data_array(volume)
    width, height, depth, channels = volume.dimensions
    if channels != 1:
        raise NotImplementedError(f"Expected a single-channel volume, got {channels} channels")

    rng = np.random.default_rng(seed)
    centers = choose_patch_centers(skeleton_array, config.patches_per_sample, rng)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[CachedTrainingRecord] = []

    for patch_index, center in enumerate(centers):
        selected_indices = patch_foreground_indices(
            data=data,
            width=width,
            height=height,
            depth=depth,
            center_xyz=center,
            radius=config.patch_radius,
            threshold=threshold,
        )
        if selected_indices.shape[0] < config.min_points:
            continue
        selected_indices = sample_indices_fixed(selected_indices, config.max_points, rng)
        points_array = indices_to_point_array(selected_indices, data, width, height)
        patch_skeleton_array = skeleton_nodes_in_patch(skeleton_array, center, config.patch_radius)
        if patch_skeleton_array.shape[0] == 0:
            continue
        patch_edge_index = skeleton_edge_index_from_array(patch_skeleton_array)

        metadata = {
            "sample_id": sample.sample_id,
            "sample_index": sample_index,
            "patch_index": patch_index,
            "volume_path": str(sample.volume_path),
            "swc_path": str(sample.swc_path),
            "swc_selection": sample.swc_selection,
            "volume_dimensions": volume.dimensions,
            "threshold": threshold,
            "max_points": config.max_points,
            "seed": seed,
            "patch_radius": config.patch_radius,
            "patch_center": [float(value) for value in center],
            "patch_foreground_count": int(selected_indices.shape[0]),
        }
        path = output_path / f"sample_{sample_index:04d}_patch_{patch_index:03d}.npz"
        np.savez_compressed(
            path,
            points=points_array,
            skeleton_nodes=patch_skeleton_array,
            edge_index=patch_edge_index,
            metadata=np.array(json.dumps(metadata), dtype=np.str_),
        )
        records.append(
            CachedTrainingRecord(
                sample_id=f"{sample.sample_id}#patch-{patch_index}",
                output_path=path,
                point_count=int(points_array.shape[0]),
                skeleton_node_count=int(patch_skeleton_array.shape[0]),
                edge_count=int(patch_edge_index.shape[0]),
                total_foreground_count=int(selected_indices.shape[0]),
            )
        )

    return records


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


def volume_data_array(volume) -> np.ndarray:
    from pointneuron.data.point_cloud import _volume_dtype

    return np.frombuffer(volume.data, dtype=_volume_dtype(volume))


def choose_patch_centers(skeleton_array: np.ndarray, patches_per_sample: int, rng: np.random.Generator) -> np.ndarray:
    node_xyz = skeleton_array[:, 1:4]
    if node_xyz.shape[0] <= patches_per_sample:
        return node_xyz
    indices = rng.choice(np.arange(node_xyz.shape[0]), size=patches_per_sample, replace=False)
    return node_xyz[np.sort(indices)]


def patch_foreground_indices(
    data: np.ndarray,
    width: int,
    height: int,
    depth: int,
    center_xyz: np.ndarray,
    radius: int,
    threshold: int,
) -> np.ndarray:
    cx, cy, cz = center_xyz
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(width, int(np.ceil(cx + radius + 1)))
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(height, int(np.ceil(cy + radius + 1)))
    z0 = max(0, int(np.floor(cz - radius)))
    z1 = min(depth, int(np.ceil(cz + radius + 1)))
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return np.empty((0,), dtype=np.int64)

    volume = data.reshape((depth, height, width))
    patch = volume[z0:z1, y0:y1, x0:x1]
    local_indices = np.flatnonzero(patch > threshold)
    if local_indices.size == 0:
        return np.empty((0,), dtype=np.int64)

    patch_width = x1 - x0
    patch_height = y1 - y0
    local_z = local_indices // (patch_width * patch_height)
    local_offset = local_indices - local_z * patch_width * patch_height
    local_y = local_offset // patch_width
    local_x = local_offset - local_y * patch_width
    global_x = local_x + x0
    global_y = local_y + y0
    global_z = local_z + z0
    return (global_z * width * height + global_y * width + global_x).astype(np.int64)


def sample_indices_fixed(indices: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    replace = indices.shape[0] < max_points
    selected = rng.choice(indices, size=max_points, replace=replace)
    return np.sort(selected.astype(np.int64))


def indices_to_point_array(indices: np.ndarray, data: np.ndarray, width: int, height: int) -> np.ndarray:
    plane_size = width * height
    z = indices // plane_size
    offset = indices - z * plane_size
    y = offset // width
    x = offset - y * width
    intensity = data[indices].astype(np.float32)
    return np.stack([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), intensity], axis=1)


def skeleton_nodes_in_patch(skeleton_array: np.ndarray, center_xyz: np.ndarray, radius: int) -> np.ndarray:
    distances = np.linalg.norm(skeleton_array[:, 1:4] - center_xyz.reshape(1, 3), axis=1)
    return skeleton_array[distances <= radius].astype(np.float32, copy=False)


def skeleton_edge_index_from_array(skeleton_array: np.ndarray) -> np.ndarray:
    node_id_to_index = {int(node[0]): index for index, node in enumerate(skeleton_array)}
    edges: list[tuple[int, int]] = []
    for child_index, node in enumerate(skeleton_array):
        parent_id = int(node[5])
        if parent_id == -1:
            continue
        parent_index = node_id_to_index.get(parent_id)
        if parent_index is not None:
            edges.append((parent_index, child_index))
    return np.array(edges, dtype=np.int64)
