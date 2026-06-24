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
    patch_stride: int | None = None
    max_points: int = 2048
    min_points: int = 256
    min_unique_fraction: float = 0.0
    center_strategy: str = "random"
    point_sample_strategy: str = "random"
    point_sample_cell_size: int = 8
    endpoint_fraction: float = 0.25
    branch_fraction: float = 0.10


def build_training_record(
    sample: Gold166Sample,
    output_path: str | Path,
    threshold: int = 0,
    max_points: int = 4096,
    seed: int = 0,
    threshold_fraction: float | None = None,
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
        "threshold_fraction": threshold_fraction,
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
    threshold_fraction: float | None = None,
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
    if config.center_strategy == "foreground":
        centers = choose_foreground_patch_centers(
            data=data,
            width=width,
            height=height,
            patches_per_sample=config.patches_per_sample,
            rng=rng,
            threshold=threshold,
        )
    elif config.center_strategy == "coverage":
        centers = choose_coverage_patch_centers(
            data=data,
            width=width,
            height=height,
            depth=depth,
            patches_per_sample=config.patches_per_sample,
            patch_radius=config.patch_radius,
            stride=config.patch_stride or config.patch_radius,
            min_points=config.min_points,
            threshold=threshold,
        )
    else:
        centers = choose_patch_centers(
            skeleton_array,
            config.patches_per_sample,
            rng,
            strategy=config.center_strategy,
            endpoint_fraction=config.endpoint_fraction,
            branch_fraction=config.branch_fraction,
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[CachedTrainingRecord] = []

    for patch_index, center in enumerate(centers):
        if len(records) >= config.patches_per_sample:
            break
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
        patch_foreground_count = int(selected_indices.shape[0])
        if patch_foreground_count < int(config.max_points * config.min_unique_fraction):
            continue
        selected_indices = sample_indices_fixed(
            selected_indices,
            config.max_points,
            rng,
            strategy=config.point_sample_strategy,
            data=data,
            width=width,
            height=height,
            cell_size=config.point_sample_cell_size,
        )
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
            "threshold_fraction": threshold_fraction,
            "max_points": config.max_points,
            "seed": seed,
            "patch_radius": config.patch_radius,
            "patch_stride": config.patch_stride or config.patch_radius,
            "min_unique_fraction": config.min_unique_fraction,
            "center_strategy": config.center_strategy,
            "point_sample_strategy": config.point_sample_strategy,
            "point_sample_cell_size": config.point_sample_cell_size,
            "endpoint_fraction": config.endpoint_fraction,
            "branch_fraction": config.branch_fraction,
            "patch_center": [float(value) for value in center],
            "patch_foreground_count": patch_foreground_count,
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
                total_foreground_count=patch_foreground_count,
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
    return np.array(edges, dtype=np.int64).reshape(-1, 2)


def volume_data_array(volume) -> np.ndarray:
    from pointneuron.data.point_cloud import _volume_dtype

    return np.frombuffer(volume.data, dtype=_volume_dtype(volume))


def choose_patch_centers(
    skeleton_array: np.ndarray,
    patches_per_sample: int,
    rng: np.random.Generator,
    strategy: str = "random",
    endpoint_fraction: float = 0.25,
    branch_fraction: float = 0.10,
) -> np.ndarray:
    node_xyz = skeleton_array[:, 1:4]
    if node_xyz.shape[0] <= patches_per_sample:
        return node_xyz
    if strategy == "topology":
        endpoint_indices, branch_indices = skeleton_role_indices(skeleton_array)
        selected: list[int] = []
        endpoint_count = min(len(endpoint_indices), int(round(patches_per_sample * endpoint_fraction)))
        branch_count = min(len(branch_indices), int(round(patches_per_sample * branch_fraction)))
        if endpoint_count > 0:
            selected.extend(rng.choice(endpoint_indices, size=endpoint_count, replace=False).tolist())
        if branch_count > 0:
            available_branches = np.array([index for index in branch_indices if index not in set(selected)], dtype=np.int64)
            if available_branches.size > 0:
                selected.extend(rng.choice(available_branches, size=min(branch_count, available_branches.size), replace=False).tolist())
        remaining_count = patches_per_sample - len(selected)
        if remaining_count > 0:
            selected_set = set(selected)
            available = np.array([index for index in range(node_xyz.shape[0]) if index not in selected_set], dtype=np.int64)
            selected.extend(rng.choice(available, size=remaining_count, replace=False).tolist())
        return node_xyz[np.sort(np.array(selected[:patches_per_sample], dtype=np.int64))]
    if strategy != "random":
        raise ValueError(f"Unknown patch center strategy {strategy!r}; expected 'random', 'topology', 'foreground', or 'coverage'")
    indices = rng.choice(np.arange(node_xyz.shape[0]), size=patches_per_sample, replace=False)
    return node_xyz[np.sort(indices)]


def choose_foreground_patch_centers(
    data: np.ndarray,
    width: int,
    height: int,
    patches_per_sample: int,
    rng: np.random.Generator,
    threshold: int,
) -> np.ndarray:
    foreground_indices = flat_foreground_indices(data, threshold)
    if foreground_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    replace = foreground_indices.size < patches_per_sample
    selected = rng.choice(foreground_indices, size=patches_per_sample, replace=replace)
    plane_size = width * height
    z = selected // plane_size
    offset = selected - z * plane_size
    y = offset // width
    x = offset - y * width
    return np.stack([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)], axis=1)


def choose_coverage_patch_centers(
    data: np.ndarray,
    width: int,
    height: int,
    depth: int,
    patches_per_sample: int,
    patch_radius: int,
    stride: int,
    min_points: int,
    threshold: int,
) -> np.ndarray:
    bounds = foreground_bounds(
        data=data,
        width=width,
        height=height,
        depth=depth,
        threshold=threshold,
    )
    if bounds is None:
        return np.zeros((0, 3), dtype=np.float32)
    mins, maxs = bounds
    axes = [np.arange(mins[axis], maxs[axis] + 1, stride, dtype=np.float32) for axis in range(3)]
    if any(axis.size == 0 for axis in axes):
        return np.zeros((0, 3), dtype=np.float32)

    candidates: list[tuple[np.ndarray, int]] = []
    for cx in axes[0]:
        for cy in axes[1]:
            for cz in axes[2]:
                center = np.array([cx, cy, cz], dtype=np.float32)
                candidates.append((center, 1))

    if not candidates:
        return np.zeros((0, 3), dtype=np.float32)
    candidate_count = max(patches_per_sample, patches_per_sample * 4)
    if len(candidates) <= candidate_count:
        return np.stack([center for center, _count in candidates], axis=0).astype(np.float32, copy=False)
    return select_spatially_diverse_centers(candidates, candidate_count, np.array([width, height, depth], dtype=np.float32))


def select_spatially_diverse_centers(
    candidates: list[tuple[np.ndarray, int]],
    count: int,
    volume_shape: np.ndarray,
) -> np.ndarray:
    centers = np.stack([center for center, _foreground_count in candidates], axis=0).astype(np.float32, copy=False)
    foreground_counts = np.array([foreground_count for _center, foreground_count in candidates], dtype=np.float32)
    normalized = centers / np.maximum(volume_shape.reshape(1, 3), 1.0)

    selected = [int(np.argmax(foreground_counts))]
    remaining = np.ones(len(candidates), dtype=bool)
    remaining[selected[0]] = False
    min_distances = np.linalg.norm(normalized - normalized[selected[0]], axis=1)

    while len(selected) < count and bool(remaining.any()):
        density = foreground_counts / max(float(foreground_counts.max()), 1.0)
        score = min_distances + 0.05 * density
        score[~remaining] = -1.0
        next_index = int(np.argmax(score))
        selected.append(next_index)
        remaining[next_index] = False
        distances = np.linalg.norm(normalized - normalized[next_index], axis=1)
        min_distances = np.minimum(min_distances, distances)

    selected_centers = centers[selected]
    order = np.lexsort((selected_centers[:, 2], selected_centers[:, 1], selected_centers[:, 0]))
    return selected_centers[order].astype(np.float32, copy=False)


def flat_foreground_indices(data: np.ndarray, threshold: int, max_exact_indices: int = 50_000_000) -> np.ndarray:
    chunks: list[np.ndarray] = []
    foreground_count = 0
    chunk_size = 16_000_000
    for start in range(0, data.shape[0], chunk_size):
        stop = min(data.shape[0], start + chunk_size)
        local = np.flatnonzero(data[start:stop] > threshold)
        if local.size == 0:
            continue
        foreground_count += int(local.size)
        if foreground_count > max_exact_indices:
            raise ValueError(
                f"Foreground index materialization would require more than {max_exact_indices} int64 entries; "
                "use --center-strategy coverage for large volumes."
            )
        chunks.append(local.astype(np.int64, copy=False) + start)
    if not chunks:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(chunks)


def foreground_bounds(
    data: np.ndarray,
    width: int,
    height: int,
    depth: int,
    threshold: int,
    chunk_depth: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    volume = data.reshape((depth, height, width))
    min_x = width
    min_y = height
    min_z = depth
    max_x = -1
    max_y = -1
    max_z = -1

    for z0 in range(0, depth, chunk_depth):
        z1 = min(depth, z0 + chunk_depth)
        mask = volume[z0:z1] > threshold
        if not bool(mask.any()):
            continue

        z_any = mask.reshape((z1 - z0), -1).any(axis=1)
        y_any = mask.any(axis=(0, 2))
        x_any = mask.any(axis=(0, 1))
        z_values = np.flatnonzero(z_any) + z0
        y_values = np.flatnonzero(y_any)
        x_values = np.flatnonzero(x_any)

        min_z = min(min_z, int(z_values[0]))
        max_z = max(max_z, int(z_values[-1]))
        min_y = min(min_y, int(y_values[0]))
        max_y = max(max_y, int(y_values[-1]))
        min_x = min(min_x, int(x_values[0]))
        max_x = max(max_x, int(x_values[-1]))

    if max_x < 0:
        return None
    return (
        np.array([min_x, min_y, min_z], dtype=np.float32),
        np.array([max_x, max_y, max_z], dtype=np.float32),
    )


def skeleton_role_indices(skeleton_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_ids = skeleton_array[:, 0].astype(np.int64, copy=False)
    parent_ids = skeleton_array[:, 5].astype(np.int64, copy=False)
    child_counts = {int(node_id): 0 for node_id in node_ids}
    for parent_id in parent_ids:
        if int(parent_id) in child_counts:
            child_counts[int(parent_id)] += 1
    endpoints = [
        index
        for index, (node_id, parent_id) in enumerate(zip(node_ids, parent_ids))
        if child_counts[int(node_id)] == 0 and int(parent_id) >= 0
    ]
    branches = [index for index, node_id in enumerate(node_ids) if child_counts[int(node_id)] > 1]
    return np.array(endpoints, dtype=np.int64), np.array(branches, dtype=np.int64)


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


def sample_indices_fixed(
    indices: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
    strategy: str = "random",
    data: np.ndarray | None = None,
    width: int | None = None,
    height: int | None = None,
    cell_size: int = 8,
) -> np.ndarray:
    if strategy == "random":
        replace = indices.shape[0] < max_points
        selected = rng.choice(indices, size=max_points, replace=replace)
        return np.sort(selected.astype(np.int64))
    if strategy == "spatial":
        if data is None or width is None or height is None:
            raise ValueError("spatial point sampling requires data, width, and height")
        return spatially_balanced_sample_indices(
            indices=indices,
            data=data,
            width=width,
            height=height,
            max_points=max_points,
            rng=rng,
            cell_size=cell_size,
        )
    raise ValueError(f"Unknown point sample strategy {strategy!r}; expected 'random' or 'spatial'")


def spatially_balanced_sample_indices(
    indices: np.ndarray,
    data: np.ndarray,
    width: int,
    height: int,
    max_points: int,
    rng: np.random.Generator,
    cell_size: int = 8,
) -> np.ndarray:
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if indices.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")

    representative_indices = brightest_cell_representatives(
        indices=indices.astype(np.int64, copy=False),
        data=data,
        width=width,
        height=height,
        cell_size=cell_size,
    )
    if representative_indices.shape[0] >= max_points:
        selected = spatially_diverse_indices(
            candidate_indices=representative_indices,
            data=data,
            width=width,
            height=height,
            count=max_points,
        )
        return np.sort(selected.astype(np.int64, copy=False))

    selected = representative_indices.astype(np.int64, copy=False).tolist()
    selected_set = set(int(index) for index in selected)
    remaining = np.array([int(index) for index in indices if int(index) not in selected_set], dtype=np.int64)
    needed = max_points - len(selected)
    if needed > 0:
        pool = remaining if remaining.shape[0] > 0 else indices.astype(np.int64, copy=False)
        fill = rng.choice(pool, size=needed, replace=pool.shape[0] < needed)
        selected.extend(int(index) for index in fill.tolist())
    return np.sort(np.array(selected, dtype=np.int64))


def brightest_cell_representatives(
    indices: np.ndarray,
    data: np.ndarray,
    width: int,
    height: int,
    cell_size: int,
) -> np.ndarray:
    coords = flat_indices_to_xyz(indices, width=width, height=height)
    cells = np.floor_divide(coords, int(cell_size)).astype(np.int64, copy=False)
    cells_per_x = max(1, (int(width) + cell_size - 1) // cell_size)
    cells_per_y = max(1, (int(height) + cell_size - 1) // cell_size)
    cell_codes = cells[:, 0] + cells[:, 1] * cells_per_x + cells[:, 2] * cells_per_x * cells_per_y
    intensities = data[indices]
    order = np.lexsort((-intensities.astype(np.float64, copy=False), cell_codes))
    ordered_codes = cell_codes[order]
    first_in_cell = np.empty((ordered_codes.shape[0],), dtype=bool)
    first_in_cell[0] = True
    first_in_cell[1:] = ordered_codes[1:] != ordered_codes[:-1]
    return indices[order[first_in_cell]]


def spatially_diverse_indices(
    candidate_indices: np.ndarray,
    data: np.ndarray,
    width: int,
    height: int,
    count: int,
) -> np.ndarray:
    coords = flat_indices_to_xyz(candidate_indices, width=width, height=height).astype(np.float32, copy=False)
    span = np.maximum(coords.max(axis=0) - coords.min(axis=0), 1.0)
    normalized = (coords - coords.min(axis=0)) / span
    intensities = data[candidate_indices].astype(np.float32, copy=False)
    intensity_score = intensities / max(float(intensities.max()), 1.0)

    selected = [int(np.argmax(intensity_score))]
    remaining = np.ones(candidate_indices.shape[0], dtype=bool)
    remaining[selected[0]] = False
    min_distances = np.linalg.norm(normalized - normalized[selected[0]], axis=1)

    while len(selected) < count and bool(remaining.any()):
        score = min_distances + 0.05 * intensity_score
        score[~remaining] = -1.0
        next_index = int(np.argmax(score))
        selected.append(next_index)
        remaining[next_index] = False
        distances = np.linalg.norm(normalized - normalized[next_index], axis=1)
        min_distances = np.minimum(min_distances, distances)
    return candidate_indices[np.array(selected, dtype=np.int64)]


def flat_indices_to_xyz(indices: np.ndarray, width: int, height: int) -> np.ndarray:
    plane_size = width * height
    z = indices // plane_size
    offset = indices - z * plane_size
    y = offset // width
    x = offset - y * width
    return np.stack([x, y, z], axis=1)


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
    return np.array(edges, dtype=np.int64).reshape(-1, 2)
