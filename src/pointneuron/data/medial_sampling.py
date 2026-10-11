"""Image-only TEASAR guide and fixed-budget sampling for PointNeuron 2.0.

This preprocessing sampler is discrete. The trainable relaxation lives in
models/adaptive_sampling.py; neither accepts a reference SWC.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MedialSamplingConfig:
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    alpha: float = 0.25
    uniform_fraction: float = 0.25
    teasar_scale: float = 1.5
    teasar_const: float = 2.0
    dust_threshold: int = 8
    max_crop_voxels: int = 16_000_000

    def __post_init__(self) -> None:
        spacing = tuple(self.spacing_xyz)
        if len(spacing) != 3 or not all(np.isfinite(v) and v > 0 for v in spacing):
            raise ValueError("spacing_xyz must contain three finite positive values")
        object.__setattr__(self, "spacing_xyz", spacing)
        for name in ("alpha", "teasar_scale", "teasar_const"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.uniform_fraction) or not 0 <= self.uniform_fraction <= 1:
            raise ValueError("uniform_fraction must be in [0, 1]")
        for name, minimum in (("dust_threshold", 0), ("max_crop_voxels", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["spacing_xyz"] = list(self.spacing_xyz)
        return payload


def load_medial_config(path: str | Path | None) -> MedialSamplingConfig:
    return MedialSamplingConfig(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else MedialSamplingConfig()


def _xyz(indices: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.column_stack((indices % width, (indices // width) % height, indices // (width * height)))


def _validate_indices(indices: np.ndarray, width: int, height: int) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if width <= 0 or height <= 0 or np.any(indices < 0):
        raise ValueError("width/height must be positive and indices nonnegative")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("foreground candidate indices must be unique")
    return indices.astype(np.int64, copy=False)


def image_medial_axis(
    indices: np.ndarray, width: int, height: int,
    config: MedialSamplingConfig = MedialSamplingConfig(),
) -> np.ndarray:
    """Return image-derived guide vertices in global voxel XYZ coordinates."""
    return image_medial_skeleton(indices, width, height, config)[0]


def image_medial_skeleton(
    indices: np.ndarray, width: int, height: int,
    config: MedialSamplingConfig = MedialSamplingConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return image skeleton vertices in global voxel XYZ coordinates.

    Build a padded XYZ crop from *all* thresholded foreground candidates, not
    the sampled cloud. Retain disconnected components; no gap repair is applied.
    Memory cap violations and unavailable Kimimaro fail explicitly. If dust
    removal leaves no skeleton, return empty vertices for uniform fallback.
    """
    indices = _validate_indices(indices, width, height)
    if not len(indices):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)
    import kimimaro

    xyz = _xyz(indices, width, height)
    origin = xyz.min(axis=0) - 1
    shape = xyz.max(axis=0) - origin + 2
    crop_size = int(shape[0]) * int(shape[1]) * int(shape[2])
    if crop_size > config.max_crop_voxels:
        raise ValueError(f"Medial crop has {crop_size} voxels; cap is {config.max_crop_voxels}. Use smaller patches.")
    mask = np.zeros(tuple(shape), dtype=np.uint8, order="F")
    local = xyz - origin
    mask[tuple(local.T)] = 1
    skeletons = kimimaro.skeletonize(
        mask,
        teasar_params={"scale": config.teasar_scale, "const": config.teasar_const,
                      "pdrf_exponent": 4, "pdrf_scale": 100000},
        anisotropy=config.spacing_xyz, dust_threshold=config.dust_threshold,
        fix_branching=True, fix_borders=False, fill_holes=False,
        progress=False, parallel=1,
    )
    pieces, edge_pieces, offset = [], [], 0
    for key in sorted(skeletons):
        skeleton = skeletons[key]
        vertices = np.asarray(skeleton.vertices, dtype=np.float64)
        if not len(vertices):
            continue
        pieces.append(vertices)
        edge_pieces.append(np.asarray(skeleton.edges, dtype=np.int64).reshape(-1, 2) + offset)
        offset += len(vertices)
    if not pieces:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)
    # Kimimaro emits physical XYZ; convert only once to the pipeline voxel frame.
    return (np.concatenate(pieces) / np.asarray(config.spacing_xyz) + origin,
            np.concatenate(edge_pieces))


def medial_distances(xyz: np.ndarray, guide_xyz: np.ndarray, spacing_xyz=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Nearest-vertex distance in configured physical units (voxel units by default)."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    guide_xyz = np.asarray(guide_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or guide_xyz.ndim != 2 or guide_xyz.shape[1] != 3:
        raise ValueError("Candidates and guide must have shapes [N, 3] and [M, 3]")
    if not np.isfinite(xyz).all() or not np.isfinite(guide_xyz).all():
        raise ValueError("Candidates and guide must be finite")
    spacing = np.asarray(MedialSamplingConfig(spacing_xyz=spacing_xyz).spacing_xyz)
    if not len(guide_xyz):
        return np.full(len(xyz), np.inf)
    return cKDTree(guide_xyz * spacing).query(xyz * spacing, workers=1)[0]


def medial_probabilities(distances: np.ndarray, config: MedialSamplingConfig) -> np.ndarray:
    """Normalized exponential prior mixed with uniform exploration probability."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1 or np.any(distances < 0) or np.isnan(distances).any():
        raise ValueError("Distances must be nonnegative and not NaN")
    if not len(distances):
        return distances.copy()
    if not np.isfinite(distances).any() or config.alpha == 0:
        return np.full(len(distances), 1.0 / len(distances))
    finite_min = distances[np.isfinite(distances)].min()
    weights = np.exp(-config.alpha * (distances - finite_min))
    weights /= weights.sum()
    return (1 - config.uniform_fraction) * weights + config.uniform_fraction / len(weights)


def sample_medial_indices(
    indices: np.ndarray, width: int, height: int, max_points: int,
    rng: np.random.Generator, config: MedialSamplingConfig = MedialSamplingConfig(),
) -> np.ndarray:
    """Weighted Gumbel top-k without replacement; only pad when N < budget.

    Sorted return preserves the existing cache/inference index convention.
    Empty image skeletons use uniform sampling; other extraction errors propagate.
    """
    indices = _validate_indices(indices, width, height)
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points <= 0:
        raise ValueError("max_points must be a positive integer")
    if not len(indices):
        return indices.copy()
    if len(indices) <= max_points:
        padding = rng.choice(indices, size=max_points - len(indices), replace=True)
        return np.sort(np.concatenate((indices, padding)))
    guide = image_medial_axis(indices, width, height, config)
    probabilities = medial_probabilities(medial_distances(_xyz(indices, width, height), guide, config.spacing_xyz), config)
    with np.errstate(divide="ignore"):
        keys = np.log(probabilities) + rng.gumbel(size=len(indices))
    # If uniform_fraction=0 underflows probabilities to zero, ties still fill the
    # requested budget from distinct image candidates rather than duplicating.
    selected = np.argpartition(keys, -max_points)[-max_points:]
    return np.sort(indices[selected])
