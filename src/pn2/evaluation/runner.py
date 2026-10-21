"""Whole-case evaluation with explicit failure accounting and no model calls."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import statistics

from .geometry import geometry_metrics, tolerance_key
from .manifest import BenchmarkCase, BenchmarkManifest, PredictionEntry, PredictionManifest
from .swc import SwcValidationError, empty_morphology, parse_swc


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
    except OSError:
        return None


def _reference(case: BenchmarkCase) -> tuple[dict, object | None]:
    info = {"path": str(case.reference_swc), "annotation_status": case.annotation_status,
            "status": None, "sha256": None}
    if case.annotation_status == "unknown":
        info.update(status="unknown_annotation", note="Not interpreted as empty background; reference not opened")
        return info, None
    info["sha256"] = _sha256(case.reference_swc)
    try:
        morphology = parse_swc(case.reference_swc)
    except FileNotFoundError as exc:
        info.update(status="missing_reference", error=str(exc))
        return info, None
    except SwcValidationError as exc:
        info.update(status="invalid_reference", error=str(exc))
        return info, None
    except OSError as exc:
        info.update(status="unreadable_reference", error=str(exc))
        return info, None
    info.update(node_count=morphology.node_count, component_count=morphology.component_count,
                isolated_node_count=len(morphology.isolated_points))
    if morphology.total_length <= 0:
        info["status"] = "unscorable_reference_no_curve"
        return info, None
    info["status"] = "scorable_reference_agreement"
    if case.annotation_status == "incomplete":
        info["note"] = "Scores compare with the observed reference; disagreement does not establish false biology"
    return info, morphology


def _prediction(entry: PredictionEntry | None) -> tuple[dict, object | None]:
    if entry is None:
        return {"reported_status": None, "status": "missing", "path": None, "sha256": None,
                "note": "No output row supplied; case retained in denominator"}, None
    info = {"reported_status": entry.status, "status": entry.status,
            "path": str(entry.swc_path) if entry.swc_path else None, "sha256": None,
            "runtime_seconds": entry.runtime_seconds, "peak_memory_bytes": entry.peak_memory_bytes}
    if entry.status != "completed":
        info["note"] = "Declared unsuccessful output is counted; optional artifact is not scored"
        return info, None
    info["sha256"] = _sha256(entry.swc_path)
    try:
        morphology = parse_swc(entry.swc_path)
    except FileNotFoundError as exc:
        info.update(status="missing_artifact", error=str(exc))
        return info, None
    except SwcValidationError as exc:
        info.update(status="invalid", error=str(exc))
        return info, None
    except OSError as exc:
        info.update(status="unreadable_artifact", error=str(exc))
        return info, None
    info.update(node_count=morphology.node_count, component_count=morphology.component_count,
                isolated_node_count=len(morphology.isolated_points))
    info["status"] = "empty" if not morphology.node_count else "segmentless" if morphology.total_length <= 0 else "completed"
    return info, morphology


def _zero_metrics(reference, spacing, tolerances, maximum_step, measured_empty: bool,
                  prediction=None) -> dict:
    result = geometry_metrics(reference, prediction if prediction is not None else empty_morphology(), spacing=spacing,
                              tolerances=tolerances, maximum_step=maximum_step)
    result["failure_policy"] = "No valid reconstructed curve; zero precision/recall/F1 retained in macro scores"
    if not measured_empty:
        # An unreadable/invalid/timed-out artifact has no measurable curve length.
        # Zero F1 is a scoring policy, not a fabricated absolute-error measurement.
        result["predicted_length"] = None
        for values in result["tolerances"].values():
            values["extra_length"] = None
            values["missed_length"] = None
        result["structural_support"] = None
        result["isolated"] = None
    return result


def _statistics(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def _macro(rows: list[dict], tolerances: tuple[float, ...]) -> list[dict]:
    result = []
    for units in sorted({row["units"] for row in rows}):
        declared = [row for row in rows if row["units"] == units]
        scored = [row for row in declared if row["geometry"] is not None]
        for tolerance in tolerances:
            key = tolerance_key(tolerance)
            metrics = [row["geometry"]["tolerances"][key] for row in scored]
            result.append({"units": units, "tolerance": tolerance,
                "declared_cases": len(declared), "scorable_reference_cases": len(scored),
                "unscorable_reference_cases": len(declared) - len(scored),
                "prediction_status_counts": dict(Counter(row["prediction"]["status"] for row in declared)),
                **{metric: _statistics([entry[metric] for entry in metrics if entry[metric] is not None])
                   for metric in ("precision", "recall", "f1", "extra_length", "missed_length")}})
    return result


def _paired(models: list[dict], tolerances: tuple[float, ...]) -> list[dict]:
    if len(models) < 2:
        return []
    baseline = models[0]
    left = {row["case_id"]: row for row in baseline["cases"]}
    result = []
    for candidate in models[1:]:
        rows = []
        for right in candidate["cases"]:
            previous = left[right["case_id"]]
            for tolerance in tolerances:
                key = tolerance_key(tolerance)
                scorable = previous["geometry"] is not None and right["geometry"] is not None
                rows.append({"case_id": right["case_id"], "units": right["units"], "tolerance": tolerance,
                    "scorable": scorable,
                    **{f"delta_{metric}": right["geometry"]["tolerances"][key][metric]
                       - previous["geometry"]["tolerances"][key][metric] if scorable else None
                       for metric in ("precision", "recall", "f1")}})
        summary = []
        for units in sorted({row["units"] for row in rows}):
            for tolerance in tolerances:
                selected = [row for row in rows if row["units"] == units and row["tolerance"] == tolerance]
                summary.append({"units": units, "tolerance": tolerance, "declared_pairs": len(selected),
                    **{f"delta_{metric}": _statistics([row[f"delta_{metric}"] for row in selected
                                                     if row[f"delta_{metric}"] is not None])
                       for metric in ("precision", "recall", "f1")}})
        result.append({"baseline": baseline["model_id"], "candidate": candidate["model_id"],
                       "direction": "candidate minus baseline", "cases": rows, "summary": summary})
    return result


def evaluate(benchmark: BenchmarkManifest, predictions: list[PredictionManifest], *,
             tolerances: tuple[float, ...] = (2., 6.), maximum_step: float = .5,
             check_convergence: bool = False) -> dict:
    """Evaluate only manifest paths. No image reads, model inference, or training.

    The first model is the paired baseline. Every declared case is emitted for
    every model; numeric means include all reference-positive failures as zero.
    Mixed units form separate groups rather than one physically misleading mean.
    """
    if not predictions or len({item.model_id for item in predictions}) != len(predictions):
        raise ValueError("Supply at least one prediction manifest with unique model IDs")
    if (not tolerances or len(set(tolerances)) != len(tolerances)
            or any(not math.isfinite(t) or t <= 0 for t in tolerances)
            or not math.isfinite(maximum_step) or maximum_step <= 0):
        raise ValueError("Tolerances and maximum_step must be finite positive numbers; tolerances must be unique")
    references = {case.case_id: _reference(case) for case in benchmark.cases}
    models = []
    for model in predictions:
        rows = []
        for case in benchmark.cases:
            reference_info, reference = references[case.case_id]
            prediction_info, prediction = _prediction(model.outputs.get(case.case_id))
            geometry, convergence = None, None
            if reference is not None:
                if prediction is not None and prediction.total_length > 0:
                    geometry = geometry_metrics(reference, prediction, spacing=case.spacing,
                                                tolerances=tolerances, maximum_step=maximum_step)
                    if check_convergence:
                        finer = geometry_metrics(reference, prediction, spacing=case.spacing,
                                                 tolerances=tolerances, maximum_step=maximum_step / 2)
                        convergence = {"maximum_step": maximum_step / 2, "geometry": finer,
                            "primary_result_retained": True,
                            "absolute_deltas": {key: {metric: abs(finer["tolerances"][key][metric] - values[metric])
                                for metric in ("precision", "recall", "f1")}
                                for key, values in geometry["tolerances"].items()}}
                else:
                    measured_empty = prediction is not None or prediction_info["status"] == "empty"
                    geometry = _zero_metrics(reference, case.spacing, tolerances, maximum_step, measured_empty, prediction)
            rows.append({"case_id": case.case_id, "partition": case.partition,
                "units": case.units, "spacing_xyz": list(case.spacing), "reference": reference_info,
                "prediction": prediction_info, "geometry": geometry, "convergence_check": convergence})
        models.append({"model_id": model.model_id, "manifest": str(model.path),
                       "manifest_sha256": _sha256(model.path), "cases": rows, "macro": _macro(rows, tolerances)})
    source_dir = Path(__file__).parent
    return {"schema_version": 1, "evaluator_version": "pn2-milestone-a-1", "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Explicit development/synthetic reference agreement; no test access or model execution",
        "benchmark": str(benchmark.path), "benchmark_sha256": _sha256(benchmark.path),
        "dataset": benchmark.dataset, "declared_case_count": len(benchmark.cases),
        "reference_status_counts": dict(Counter(value[0]["status"] for value in references.values())),
        "settings": {"tolerances": list(tolerances), "maximum_step": maximum_step,
                     "check_convergence": check_convergence, "curve_metric": "arclength-weighted midpoint quadrature",
                     "structural_metric": "endpoint/junction spatial support only; not topology accuracy",
                     "radius_metric": "not scored; anisotropic spacing does not define scalar physical radii",
                     "distance_maximum": "quadrature-sampled maximum; not an exact Hausdorff distance",
                     "failure_policy": "Reference-positive missing/empty/invalid/timeout/unsupported/failed output F1=0; unknown references unscorable"},
        "source_sha256": {name: _sha256(source_dir / name) for name in ("swc.py", "manifest.py", "geometry.py", "runner.py")},
        "models": models, "paired": _paired(models, tolerances)}


def csv_rows(report: dict) -> list[dict]:
    result = []
    for model in report["models"]:
        for row in model["cases"]:
            for tolerance in report["settings"]["tolerances"]:
                metrics = row["geometry"]["tolerances"][tolerance_key(tolerance)] if row["geometry"] else {}
                result.append({"model_id": model["model_id"], "case_id": row["case_id"],
                    "units": row["units"], "tolerance": tolerance,
                    "annotation_status": row["reference"]["annotation_status"],
                    "reference_status": row["reference"]["status"],
                    "prediction_status": row["prediction"]["status"],
                    **{name: metrics.get(name) for name in ("precision", "recall", "f1", "extra_length", "missed_length")}})
    return result
