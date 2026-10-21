"""Deterministic half-open voxel boxes covering a declared whole image."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from pathlib import Path

from .cases import read_object, require_keys


def positive_xyz(values, label: str) -> tuple[int, int, int]:
    if (not isinstance(values, (tuple, list)) or len(values) != 3
            or any(type(x) is not int or x <= 0 for x in values)):
        raise ValueError(f"{label} must contain three positive XYZ integers")
    return tuple(values)


@dataclass(frozen=True)
class TileConfig:
    tile_size_xyz: tuple[int, int, int]
    stride_xyz: tuple[int, int, int]

    def __post_init__(self):
        object.__setattr__(self, "tile_size_xyz", positive_xyz(self.tile_size_xyz, "tile_size_xyz"))
        object.__setattr__(self, "stride_xyz", positive_xyz(self.stride_xyz, "stride_xyz"))
        if any(step > size for step, size in zip(self.stride_xyz, self.tile_size_xyz)):
            raise ValueError("stride cannot exceed tile size: that would leave gaps")

    @classmethod
    def load(cls, path: str | Path):
        data = read_object(Path(path))
        keys = {"schema_version", "tile_size_xyz", "stride_xyz"}
        require_keys(data, keys, keys, "tiling config")
        return cls(data["tile_size_xyz"], data["stride_xyz"])


@dataclass(frozen=True)
class Tile:
    tile_id: int
    start_xyz: tuple[int, int, int]
    stop_xyz: tuple[int, int, int]

    @property
    def shape_xyz(self):
        return tuple(b - a for a, b in zip(self.start_xyz, self.stop_xyz))

    @property
    def center_xyz(self):
        return tuple((a + b - 1) / 2 for a, b in zip(self.start_xyz, self.stop_xyz))

    @property
    def slices_zyx(self):
        return tuple(slice(a, b) for a, b in zip(self.start_xyz[::-1], self.stop_xyz[::-1]))


@dataclass(frozen=True)
class _Axis:
    dimension: int
    size: int
    stride: int

    @property
    def last_start(self):
        return max(0, self.dimension - self.size)

    @property
    def count(self):
        return self.last_start // self.stride + 1 + int(self.last_start % self.stride != 0)

    def __iter__(self):
        yield from range(0, self.last_start + 1, self.stride)
        if self.last_start % self.stride:
            yield self.last_start


@dataclass(frozen=True)
class TileGrid:
    dimensions_xyz: tuple[int, int, int]
    config: TileConfig

    def __post_init__(self):
        object.__setattr__(self, "dimensions_xyz", positive_xyz(self.dimensions_xyz, "dimensions_xyz"))

    @property
    def axes(self):
        return tuple(_Axis(d, size, step) for d, size, step in
                     zip(self.dimensions_xyz, self.config.tile_size_xyz, self.config.stride_xyz))

    @property
    def tile_count(self):
        return prod(axis.count for axis in self.axes)

    def __iter__(self):
        # X varies fastest, matching native array indexing. No tile cap or padding.
        for index, zyx in enumerate(product(*self.axes[::-1])):
            start = zyx[::-1]
            stop = tuple(min(d, a + size) for d, a, size in
                         zip(self.dimensions_xyz, start, self.config.tile_size_xyz))
            yield Tile(index, start, stop)
