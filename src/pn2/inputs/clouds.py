"""Explicit PN1 reference control: image-max mask and distinct uniform voxel draws.

This is a fallible reference input policy, not a universal PN2 segmentation.
No SWCs, guides, learned samplers or model decisions enter cloud preparation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .cases import read_object, require_keys
from .image import DecodedImage
from .tiling import Tile


@dataclass(frozen=True)
class ReferenceCloudConfig:
    threshold_fraction: float
    max_input_points: int
    seed: int

    def __post_init__(self):
        if (isinstance(self.threshold_fraction, bool) or not isinstance(self.threshold_fraction, (int, float))
                or not math.isfinite(self.threshold_fraction) or not 0 <= self.threshold_fraction <= 1):
            raise ValueError("threshold_fraction must be finite and in [0,1]")
        if type(self.max_input_points) is not int or self.max_input_points < 2:
            raise ValueError("max_input_points must be an integer >=2")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    @classmethod
    def load(cls, path: str | Path):
        data = read_object(Path(path))
        keys = {"schema_version", "purpose", "foreground", "max_input_points", "seed"}
        require_keys(data, keys, keys, "reference cloud config")
        if data["purpose"] != "repaired_pn1_input_control":
            raise ValueError("This configuration is explicitly a repaired PN1 input control")
        require_keys(data["foreground"], {"mode", "fraction"}, {"mode", "fraction"}, "foreground")
        if data["foreground"]["mode"] != "image_max_fraction":
            raise ValueError("Only the inherited image_max_fraction control is implemented here")
        return cls(data["foreground"]["fraction"], data["max_input_points"], data["seed"])

    def as_dict(self):
        return {"schema_version": 1, "purpose": "repaired_pn1_input_control",
                "foreground": {"mode": "image_max_fraction", "fraction": self.threshold_fraction},
                "max_input_points": self.max_input_points, "seed": self.seed}


@dataclass(frozen=True)
class TileCloud:
    points_xyzi: np.ndarray
    selected_flat_indices: np.ndarray  # Global source-image voxel indices.
    metadata: dict


class ReferenceCloudBuilder:
    """Compute one whole-image threshold, then draw each tile independently.

    The threshold rounding/comparison retains the repaired reference rule:
    round(maximum * fraction), followed by image > threshold. No dtype scaling.
    """

    def __init__(self, image: DecodedImage, config: ReferenceCloudConfig):
        self.image, self.config = image, config
        self.image_maximum = int(image.array_zyx.max())
        self.threshold = int(round(self.image_maximum * config.threshold_fraction))

    def build(self, tile: Tile) -> TileCloud:
        dimensions = self.image.dimensions_xyz
        if any(not 0 <= a < b <= dim for a, b, dim in zip(tile.start_xyz, tile.stop_xyz, dimensions)):
            raise ValueError("Tile bounds are outside the source image")
        values = self.image.array_zyx[tile.slices_zyx]
        foreground = np.flatnonzero(values.reshape(-1) > self.threshold)
        count = len(foreground)
        # Stable identity/box seed: independent of call order, batch grouping and numeric dataset index.
        material = json.dumps([self.config.seed, self.image.case.case_id,
                               tile.start_xyz, tile.stop_xyz], separators=(",", ":")).encode("utf-8")
        seed_hex = hashlib.sha256(material).hexdigest()
        rng = np.random.default_rng(int(seed_hex, 16))
        chosen = (np.sort(rng.choice(foreground, self.config.max_input_points, replace=False))
                  if count > self.config.max_input_points else foreground)
        z, y, x = np.unravel_index(chosen, values.shape)
        global_xyz = np.column_stack((x, y, z)).astype(np.int64) + np.asarray(tile.start_xyz)
        points = np.column_stack((global_xyz, values.reshape(-1)[chosen])).astype(np.float32)
        if not np.array_equal(points[:, :3].astype(np.int64), global_xyz):
            raise NotImplementedError("Native XYZ cannot be represented exactly by this float32 reference input")
        width, height, _ = dimensions
        flats = (global_xyz[:, 2] * height + global_xyz[:, 1]) * width + global_xyz[:, 0]
        status = "empty_mask" if not count else "singleton_cloud" if len(chosen) == 1 else "ready"
        return TileCloud(points, flats, {
            "tile_id": tile.tile_id, "start_xyz": list(tile.start_xyz), "stop_xyz": list(tile.stop_xyz),
            "shape_xyz": list(tile.shape_xyz), "cloud_status": status,
            "image_maximum": self.image_maximum, "threshold": self.threshold,
            "threshold_mode": "image_max_fraction", "comparison": "strictly_greater_than",
            "foreground_voxels": count, "selected_points": len(chosen),
            "distinct_selected_voxels": len(chosen), "max_input_points": self.config.max_input_points,
            "sampling": "uniform_without_replacement_directly_from_tile_foreground",
            "rng": "numpy.default_rng / PCG64", "seed_sha256": seed_hex,
            "seed_key": "config_seed, stable_case_id, start_xyz, stop_xyz",
            "coordinate_units": "native_voxel", "intensity_transform": "none", "padding": "none",
            "point_packet_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
            "source_voxel_indices_sha256": hashlib.sha256(flats.astype('<i8').tobytes()).hexdigest(),
        })
