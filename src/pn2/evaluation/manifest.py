"""Explicit development-only cases and model-output manifests; no dataset scan."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


ANNOTATION_STATUSES = {"reference_agreement", "verified_complete", "incomplete", "unknown"}
OUTPUT_STATUSES = {"completed", "missing", "empty", "invalid", "timeout", "unsupported", "failed"}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    partition: str
    reference_swc: Path
    units: str
    spacing: tuple[float, float, float]
    annotation_status: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkManifest:
    path: Path
    dataset: dict[str, Any]
    cases: tuple[BenchmarkCase, ...]


@dataclass(frozen=True)
class PredictionEntry:
    case_id: str
    status: str
    swc_path: Path | None
    runtime_seconds: float | None = None
    peak_memory_bytes: int | None = None


@dataclass(frozen=True)
class PredictionManifest:
    path: Path
    model_id: str
    outputs: dict[str, PredictionEntry]


def _payload(path: Path) -> dict:
    def no_nonfinite(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=no_nonfinite,
                      object_pairs_hook=no_duplicate_keys)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError(f"{path}: expected an object with schema_version=1")
    return data


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{where}: expected a nonempty identifier without surrounding whitespace")
    return value


def _path(value: Any, manifest: Path, field: str) -> Path:
    value = _name(value, field)
    candidate = Path(value)
    return (manifest.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def load_benchmark(path: str | Path) -> BenchmarkManifest:
    """Load explicit IDs only. Reserved/test/training cases are rejected here."""
    path = Path(path).resolve()
    data = _payload(path)
    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object with a name")
    _name(dataset.get("name"), "dataset.name")
    rows = data.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cases must be a nonempty explicit list; dataset globbing is unsupported")
    cases, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each case must be an object")
        case_id = _name(row.get("case_id"), "case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate benchmark case {case_id!r}")
        seen.add(case_id)
        partition = row.get("partition")
        if partition not in {"development", "synthetic"}:
            raise ValueError(f"{case_id}: only development/synthetic partitions are allowed in milestone A")
        annotation_status = row.get("annotation_status")
        if annotation_status not in ANNOTATION_STATUSES:
            raise ValueError(f"{case_id}: annotation_status must be explicitly declared")
        coordinates = row.get("coordinates")
        if not isinstance(coordinates, dict):
            raise ValueError(f"{case_id}: coordinates must explicitly declare mode and units")
        mode, units = coordinates.get("mode"), coordinates.get("units")
        if mode == "native" and units == "voxel":
            if "spacing_xyz" in coordinates:
                raise ValueError(f"{case_id}: native coordinates cannot also specify spacing_xyz")
            spacing = (1., 1., 1.)
        elif mode == "spacing" and units == "um":
            values = coordinates.get("spacing_xyz")
            if (not isinstance(values, list) or len(values) != 3
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) or x <= 0 for x in values)):
                raise ValueError(f"{case_id}: spacing_xyz must contain three finite positive numbers")
            spacing = tuple(float(x) for x in values)
        else:
            raise ValueError(f"{case_id}: supported coordinates are native/voxel or spacing/um")
        cases.append(BenchmarkCase(case_id, partition,
            _path(row.get("reference_swc"), path, "reference_swc"), units, spacing,
            annotation_status, dict(row)))
    return BenchmarkManifest(path, dict(dataset), tuple(cases))


def load_predictions(path: str | Path, benchmark: BenchmarkManifest) -> PredictionManifest:
    path = Path(path).resolve()
    data = _payload(path)
    model_id = _name(data.get("model_id"), "model_id")
    rows = data.get("outputs")
    if not isinstance(rows, list):
        raise ValueError("outputs must be an explicit list; an empty list counts every case as missing")
    allowed, outputs = {case.case_id for case in benchmark.cases}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each output must be an object")
        case_id = _name(row.get("case_id"), "case_id")
        if case_id not in allowed:
            raise ValueError(f"Output {case_id!r} is outside the approved benchmark")
        if case_id in outputs:
            raise ValueError(f"Duplicate output for {case_id!r}")
        status = row.get("status")
        if status not in OUTPUT_STATUSES:
            raise ValueError(f"{case_id}: output status must be explicitly declared")
        swc_path = _path(row["swc_path"], path, "swc_path") if row.get("swc_path") is not None else None
        if status == "completed" and swc_path is None:
            raise ValueError(f"{case_id}: completed output requires swc_path")
        runtime = row.get("runtime_seconds")
        if runtime is not None and (isinstance(runtime, bool) or not isinstance(runtime, (int, float))
                                    or not math.isfinite(runtime) or runtime < 0):
            raise ValueError(f"{case_id}: runtime_seconds must be finite and nonnegative")
        memory = row.get("peak_memory_bytes")
        if memory is not None and (type(memory) is not int or memory < 0):
            raise ValueError(f"{case_id}: peak_memory_bytes must be a nonnegative integer")
        outputs[case_id] = PredictionEntry(case_id, status, swc_path,
                                           float(runtime) if runtime is not None else None, memory)
    return PredictionManifest(path, model_id, outputs)
