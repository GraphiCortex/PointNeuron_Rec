from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from pointneuron.data.swc import SwcTree
from pointneuron.data.vaa3d_raw import Vaa3dVolume


@dataclass(frozen=True)
class PointRecord:
    x: int
    y: int
    z: int
    intensity: int


@dataclass(frozen=True)
class PointCloud:
    points: tuple[PointRecord, ...]
    volume_dimensions: tuple[int, int, int, int]
    threshold: int
    total_foreground_count: int

    @property
    def count(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class SkeletonRecord:
    node_id: int
    x: float
    y: float
    z: float
    radius: float
    parent_id: int


def volume_to_point_cloud(
    volume: Vaa3dVolume,
    threshold: int = 0,
    max_points: int | None = None,
    seed: int = 0,
) -> PointCloud:
    width, height, depth, channels = volume.dimensions
    if channels != 1:
        raise NotImplementedError(f"Expected a single-channel volume, got {channels} channels")
    if threshold < 0 or threshold > 255:
        raise ValueError(f"Threshold must be in [0, 255], got {threshold}")

    if max_points is not None and max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")

    data = np.frombuffer(volume.data, dtype=np.uint8)
    foreground_mask = data > threshold
    total_foreground = int(np.count_nonzero(foreground_mask))

    if max_points is not None and total_foreground > max_points:
        selected_indices = _sample_foreground_indices(data, foreground_mask, total_foreground, max_points, seed, threshold)
    else:
        selected_indices = np.flatnonzero(foreground_mask)

    points = _indices_to_points(selected_indices, data, width, height)
    return PointCloud(
        points=points,
        volume_dimensions=volume.dimensions,
        threshold=threshold,
        total_foreground_count=total_foreground,
    )


def swc_to_skeleton_records(swc: SwcTree) -> tuple[SkeletonRecord, ...]:
    return tuple(
        SkeletonRecord(
            node_id=node.node_id,
            x=node.x,
            y=node.y,
            z=node.z,
            radius=node.radius,
            parent_id=node.parent_id,
        )
        for node in swc.nodes
    )


def point_cloud_stats(point_cloud: PointCloud) -> dict[str, float | int]:
    if not point_cloud.points:
        return {
            "total_foreground_points": point_cloud.total_foreground_count,
            "sampled_points": 0,
            "points": 0,
            "min_intensity": 0,
            "max_intensity": 0,
            "mean_intensity": 0.0,
        }
    intensities = [point.intensity for point in point_cloud.points]
    return {
        "total_foreground_points": point_cloud.total_foreground_count,
        "sampled_points": len(point_cloud.points),
        "points": len(point_cloud.points),
        "min_intensity": min(intensities),
        "max_intensity": max(intensities),
        "mean_intensity": sum(intensities) / len(intensities),
    }


def _sample_foreground_indices(
    data: np.ndarray,
    foreground_mask: np.ndarray,
    total_foreground: int,
    max_points: int,
    seed: int,
    threshold: int,
) -> np.ndarray:
    density = total_foreground / len(data)
    rng = random.Random(seed)

    if density >= 0.2:
        selected: set[int] = set()
        while len(selected) < max_points:
            index = rng.randrange(len(data))
            if int(data[index]) > threshold:
                selected.add(index)
        return np.array(sorted(selected), dtype=np.int64)

    foreground_indices = np.flatnonzero(foreground_mask)
    numpy_rng = np.random.default_rng(seed)
    return np.sort(numpy_rng.choice(foreground_indices, size=max_points, replace=False))


def _indices_to_points(indices: np.ndarray, data: np.ndarray, width: int, height: int) -> tuple[PointRecord, ...]:
    plane_size = width * height
    points: list[PointRecord] = []
    for index in indices.tolist():
        z = index // plane_size
        offset = index - z * plane_size
        y = offset // width
        x = offset - y * width
        points.append(PointRecord(x=x, y=y, z=z, intensity=int(data[index])))
    return tuple(points)
