"""Explicit image cases, separate from reference-bearing training/evaluation lists."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ImageCase:
    case_id: str
    image_path: Path
    partition: str
    units: str = "voxel"
    spacing_xyz: tuple[float, float, float] = (1., 1., 1.)

    def __post_init__(self):
        name(self.case_id, "case_id")
        if not isinstance(self.partition, str) or self.partition not in {"training", "development", "synthetic"}:
            raise ValueError("only training/development/synthetic images are allowed at this stage")
        if (not isinstance(self.spacing_xyz, (tuple, list)) or len(self.spacing_xyz) != 3
                or any(isinstance(x, bool) or not isinstance(x, (float, int))
                       or not math.isfinite(x) or x <= 0 for x in self.spacing_xyz)):
            raise ValueError("spacing_xyz must contain three finite positive numbers")
        if self.units not in ("voxel", "um") or (self.units == "voxel" and tuple(self.spacing_xyz) != (1, 1, 1)):
            raise ValueError("use native voxel coordinates or declared um spacing")
        object.__setattr__(self, "image_path", Path(self.image_path).resolve())
        object.__setattr__(self, "spacing_xyz", tuple(float(x) for x in self.spacing_xyz))


def read_object(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")

    data = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError(f"{path}: expected schema_version=1 object")
    return data


def require_keys(data: dict, required: set[str], allowed: set[str], where: str):
    if not isinstance(data, dict) or not required <= data.keys() or data.keys() - allowed:
        raise ValueError(f"{where}: expected fields {sorted(required)}; allowed {sorted(allowed)}")


def name(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{where}: expected nonempty string without surrounding whitespace")
    return value


def load_image_cases(path: str | Path) -> tuple[ImageCase, ...]:
    """Load explicit training/development/synthetic images; never scan directories.

    Reference keys and arbitrary extra fields are rejected before image access.
    Partitions are declared metadata, not proof of unseen-data independence.
    """
    path = Path(path).resolve()
    data = read_object(path)
    require_keys(data, {"schema_version", "cases"}, {"schema_version", "cases"}, "image manifest")
    if not isinstance(data["cases"], list) or not data["cases"]:
        raise ValueError("cases must be a nonempty explicit list")
    result, seen = [], set()
    for row in data["cases"]:
        fields = {"case_id", "image_path", "partition", "coordinates"}
        require_keys(row, fields, fields, "image case")
        case_id = name(row["case_id"], "case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate image case {case_id!r}")
        seen.add(case_id)
        if not isinstance(row["partition"], str) or row["partition"] not in {"training", "development", "synthetic"}:
            raise ValueError(f"{case_id}: only training/development/synthetic images are allowed at this stage")
        coordinates = row["coordinates"]
        require_keys(coordinates, {"mode", "units"}, {"mode", "units", "spacing_xyz"}, "coordinates")
        if coordinates == {"mode": "native", "units": "voxel"}:
            units, spacing = "voxel", (1., 1., 1.)
        elif coordinates.get("mode") == "spacing" and coordinates.get("units") == "um":
            spacing = coordinates.get("spacing_xyz")
            if (not isinstance(spacing, list) or len(spacing) != 3
                    or any(isinstance(x, bool) or not isinstance(x, (float, int))
                           or not math.isfinite(x) or x <= 0 for x in spacing)):
                raise ValueError(f"{case_id}: spacing_xyz must contain three finite positive numbers")
            units, spacing = "um", tuple(float(x) for x in spacing)
        else:
            raise ValueError(f"{case_id}: use native/voxel or spacing/um coordinates")
        image = Path(name(row["image_path"], "image_path"))
        result.append(ImageCase(case_id, (path.parent / image).resolve(), row["partition"], units, spacing))
    return tuple(result)
