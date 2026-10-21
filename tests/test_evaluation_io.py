"""Synthetic artifact and denominator tests; no dataset annotations are opened."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pn2.evaluation.manifest import load_benchmark, load_predictions
from pn2.evaluation.runner import csv_rows, evaluate
from pn2.evaluation.swc import SwcValidationError, parse_swc


def text(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def payload(path: Path, contents: dict) -> Path:
    path.write_text(json.dumps(contents), encoding="utf-8")
    return path


def line(path: Path) -> Path:
    return text(path, "1 1 0 0 0 0 -1\n2 3 10 0 0 1 1\n")


def benchmark(tmp_path: Path, names: list[str]) -> Path:
    line(tmp_path / "reference.swc")
    return payload(tmp_path / "benchmark.json", {"schema_version": 1, "dataset": {"name": "synthetic"},
        "cases": [{"case_id": name, "partition": "synthetic", "reference_swc": "reference.swc",
                   "coordinates": {"mode": "native", "units": "voxel"},
                   "annotation_status": "reference_agreement"} for name in names]})


def outputs(tmp_path: Path, rows: list[dict], name: str = "model") -> Path:
    return payload(tmp_path / f"{name}.json", {"schema_version": 1, "model_id": name, "outputs": rows})


def test_strict_reader_handles_unordered_forest_without_repair(tmp_path):
    path = text(tmp_path / "forest.swc", "# header\n20 3 1 0 0 0 10\n10 1 0 0 0 1 -1\n40 3 8 0 0 1 -1 # isolated root\n")
    morphology = parse_swc(path)
    assert morphology.node_count == 3
    assert morphology.component_count == 2
    assert morphology.parents.tolist() == [1, -1, -1]
    assert morphology.edges.tolist() == [[0, 1]]
    assert morphology.total_length == 1
    np.testing.assert_array_equal(morphology.isolated_points, [[8, 0, 0]])
    assert morphology.radii[0] == 0  # Zero is permitted; no clamping.


@pytest.mark.parametrize("rows", [
    "1.5 1 0 0 0 1 -1\n",  # Fractional ID must not be truncated.
    "1.0 1 0 0 0 1 -1\n",
    "1 1.5 0 0 0 1 -1\n",
    "1 1 0 0 0 1 -1.0\n",
    "0 1 0 0 0 1 -1\n",
    "1 -1 0 0 0 1 -1\n",
    "1 1 0 0 0 1 -1\n1 3 1 0 0 1 -1\n",
    "1 1 0 0 0 1 9\n",
    "1 1 0 0 0 1 1\n",
    "1 1 0 0 0 1 2\n2 3 1 0 0 1 1\n",
    "1 1 0 0 0 1 -1\n2 3 1 0 0 1 3\n3 3 2 0 0 1 2\n",
    "1 1 nan 0 0 1 -1\n",
    "1 1 0 inf 0 1 -1\n",
    "1 1 0 0 0 inf -1\n",
    "1 1 0 0 0 -0.1 -1\n",
    "1 1 0 0 0 1 -1 extra\n",
])
def test_invalid_swc_is_rejected(tmp_path, rows):
    with pytest.raises(SwcValidationError):
        parse_swc(text(tmp_path / "invalid.swc", rows))


def test_empty_swc_is_distinct_from_invalid_and_deep_tree_is_supported(tmp_path):
    empty = parse_swc(text(tmp_path / "empty.swc", "# no reconstruction\n"))
    assert empty.xyz.shape == (0, 3)
    assert empty.edges.shape == (0, 2)
    assert empty.component_count == empty.total_length == 0
    rows = [f"{i} 3 {i} 0 0 1 {i - 1 if i > 1 else -1}" for i in range(1, 3001)]
    morphology = parse_swc(text(tmp_path / "deep.swc", "\n".join(reversed(rows))))
    assert morphology.total_length == 2999


@pytest.mark.parametrize("partition", ["test", "reserved", "training", "validation", None])
def test_unapproved_partition_rejected_before_reference_access(tmp_path, partition):
    path = benchmark(tmp_path, ["protected"])
    data = json.loads(path.read_text())
    data["cases"][0].update(partition=partition, reference_swc="must_not_be_opened.swc")
    payload(path, data)
    with pytest.raises(ValueError, match="only development/synthetic"):
        load_benchmark(path)


def test_manifest_requires_explicit_units_and_unique_cases(tmp_path):
    path = benchmark(tmp_path, ["a"])
    data = json.loads(path.read_text())
    data["cases"][0]["coordinates"]["spacing_xyz"] = [1, 1, 2]
    payload(path, data)
    with pytest.raises(ValueError, match="cannot also specify"):
        load_benchmark(path)
    data["cases"][0]["coordinates"] = {"mode": "spacing", "units": "um", "spacing_xyz": [1, 0, 2]}
    payload(path, data)
    with pytest.raises(ValueError, match="finite positive"):
        load_benchmark(path)
    data["cases"][0]["coordinates"] = {"mode": "native", "units": "voxel"}
    data["cases"].append(dict(data["cases"][0]))
    payload(path, data)
    with pytest.raises(ValueError, match="Duplicate benchmark"):
        load_benchmark(path)


def test_output_manifest_does_not_enlarge_population(tmp_path):
    cases = load_benchmark(benchmark(tmp_path, ["allowed"]))
    path = outputs(tmp_path, [{"case_id": "unapproved", "status": "completed", "swc_path": "not_opened.swc"}])
    with pytest.raises(ValueError, match="outside the approved"):
        load_predictions(path, cases)
    path = outputs(tmp_path, [{"case_id": "allowed", "status": "timeout"}] * 2)
    with pytest.raises(ValueError, match="Duplicate output"):
        load_predictions(path, cases)


def test_every_positive_reference_failure_stays_in_macro_denominator(tmp_path):
    names = ["good", "missing_row", "missing_artifact", "empty", "invalid", "timeout",
             "unsupported", "failed", "segmentless", "unknown_annotation", "invalid_reference", "empty_reference"]
    path = benchmark(tmp_path, names)
    data = json.loads(path.read_text())
    data["cases"][9].update(annotation_status="unknown", reference_swc="not_opened.swc")
    data["cases"][10]["reference_swc"] = "invalid.swc"
    data["cases"][11]["reference_swc"] = "empty.swc"
    payload(path, data)
    line(tmp_path / "good.swc")
    text(tmp_path / "empty.swc", "")
    text(tmp_path / "invalid.swc", "1 1 0 0 0 1 2\n")
    text(tmp_path / "isolated.swc", "1 1 0 0 0 1 -1\n")
    rows = [{"case_id": "good", "status": "completed", "swc_path": "good.swc"},
            {"case_id": "missing_artifact", "status": "completed", "swc_path": "absent.swc"},
            {"case_id": "empty", "status": "completed", "swc_path": "empty.swc"},
            {"case_id": "invalid", "status": "completed", "swc_path": "invalid.swc"},
            {"case_id": "segmentless", "status": "completed", "swc_path": "isolated.swc"}]
    rows += [{"case_id": name, "status": name} for name in ("timeout", "unsupported", "failed")]
    cases = load_benchmark(path)
    report = evaluate(cases, [load_predictions(outputs(tmp_path, rows), cases)])
    model = report["models"][0]
    assert len(model["cases"]) == report["declared_case_count"] == 12
    for group in model["macro"]:
        assert group["scorable_reference_cases"] == 9
        assert group["unscorable_reference_cases"] == 3
        assert group["f1"]["mean"] == pytest.approx(1 / 9)
        assert group["f1"]["min"] == group["f1"]["median"] == 0
    indexed = {row["case_id"]: row for row in model["cases"]}
    assert indexed["missing_row"]["prediction"]["status"] == "missing"
    assert indexed["missing_artifact"]["prediction"]["status"] == "missing_artifact"
    assert indexed["segmentless"]["prediction"]["isolated_node_count"] == 1
    assert indexed["segmentless"]["geometry"]["isolated"]["predicted"] == 1
    assert indexed["invalid"]["geometry"]["tolerances"]["2"]["extra_length"] is None
    assert indexed["empty"]["geometry"]["tolerances"]["2"]["missed_length"] == 10
    assert indexed["unknown_annotation"]["reference"]["sha256"] is None
    assert indexed["unknown_annotation"]["geometry"] is None
    json.dumps(report, allow_nan=False)


def test_paired_models_include_missing_case_as_zero_and_keep_units_separate(tmp_path):
    path = benchmark(tmp_path, ["voxel_case", "physical_case"])
    data = json.loads(path.read_text())
    data["cases"][1]["coordinates"] = {"mode": "spacing", "units": "um", "spacing_xyz": [2, 1, 1]}
    data["cases"][1]["annotation_status"] = "incomplete"
    payload(path, data)
    cases = load_benchmark(path)
    good = [{"case_id": name, "status": "completed", "swc_path": "reference.swc"}
            for name in ("voxel_case", "physical_case")]
    report = evaluate(cases, [load_predictions(outputs(tmp_path, good, "baseline"), cases),
                             load_predictions(outputs(tmp_path, [], "candidate"), cases)], check_convergence=True)
    assert len(report["models"][0]["macro"]) == 4  # Two tolerances per declared unit group.
    physical = report["models"][0]["cases"][1]
    assert physical["geometry"]["reference_length"] == 20
    assert "observed reference" in physical["reference"]["note"]
    assert physical["convergence_check"]["primary_result_retained"] is True
    for row in report["paired"][0]["cases"]:
        assert row["delta_f1"] == -1
    for group in report["paired"][0]["summary"]:
        assert group["declared_pairs"] == group["delta_f1"]["count"] == 1


def test_cli_writes_explicit_artifacts_and_refuses_overwrite(tmp_path):
    path = benchmark(tmp_path, ["a"])
    model = outputs(tmp_path, [{"case_id": "a", "status": "completed", "swc_path": "reference.swc"}])
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_reconstructions.py"
    command = [sys.executable, str(script), "--benchmark", str(path), "--predictions", str(model),
               "--output", str(tmp_path / "report.json"), "--csv", str(tmp_path / "report.csv"), "--check-convergence"]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "no models executed" in first.stdout
    assert len((tmp_path / "report.csv").read_text().splitlines()) == 3
    saved = (tmp_path / "report.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "Preserve existing outputs" in second.stderr
    assert (tmp_path / "report.json").read_bytes() == saved


def test_noninteger_and_scientific_tolerances_keep_one_shared_key(tmp_path):
    path = benchmark(tmp_path, ["a"])
    cases = load_benchmark(path)
    row = [{"case_id": "a", "status": "completed", "swc_path": "reference.swc"}]
    report = evaluate(cases, [load_predictions(outputs(tmp_path, row, "baseline"), cases),
                             load_predictions(outputs(tmp_path, row, "candidate"), cases)],
                      tolerances=(1.23456789, 1000000., .000000123456789), check_convergence=True)
    assert len(csv_rows(report)) == 6
    assert all(group["f1"]["mean"] == 1 for group in report["models"][0]["macro"])
    assert all(row["delta_f1"] == 0 for row in report["paired"][0]["cases"])


def test_convergence_difference_is_reported_without_replacing_primary(tmp_path):
    path = benchmark(tmp_path, ["a"])
    cases = load_benchmark(path)
    text(tmp_path / "vertical.swc", "1 1 0 1 0 1 -1\n2 3 0 2 0 1 1\n")
    model = load_predictions(outputs(tmp_path, [{"case_id": "a", "status": "completed", "swc_path": "vertical.swc"}]), cases)
    report = evaluate(cases, [model], tolerances=(2.55,), maximum_step=.5, check_convergence=True)
    row = report["models"][0]["cases"][0]
    check = row["convergence_check"]
    assert check["primary_result_retained"] is True
    assert row["geometry"]["numerical"]["maximum_step"] == .5
    assert check["geometry"]["numerical"]["maximum_step"] == .25
    assert check["absolute_deltas"]["2.55"]["recall"] > 0
    assert "passed" not in check
