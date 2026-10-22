"""A source-image to saved-cloud to reference-evaluation integration control."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np


def test_two_separate_commands_preserve_source_and_frozen_clouds(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    z, y, x = np.indices((9, 13, 17))
    values = np.rint(200 * np.exp(-((z - 4)**2 + (y - 6)**2) / 8)).astype(np.uint8)
    payload = b"raw_image_stack_by_hpengL" + (1).to_bytes(2, "little")
    payload += b"".join(n.to_bytes(4, "little") for n in (17, 13, 9, 1)) + values.tobytes()
    image = tmp_path / "source.v3draw"
    image.write_bytes(payload)
    case = {"case_id": "synthetic-training-control", "partition": "training", "image_path": image.name,
            "coordinates": {"mode": "native", "units": "voxel"}}
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps({"schema_version": 1, "cases": [case]}))
    recipe = json.loads((repo / "configs/phase1_sampling_probe.json").read_text())
    recipe["seeds"] = [0, 1]
    recipe["tile_side_voxel"] = 8
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(recipe))
    output = tmp_path / "clouds"
    generation = subprocess.run([sys.executable, str(repo / "scripts/probe_phase1_sampling.py"),
        "--cases", str(manifest), "--recipe", str(config), "--output-dir", str(output)],
        capture_output=True, text=True)
    assert generation.returncode == 0, generation.stdout + generation.stderr
    # Create the reference only after generation succeeds: no reference could
    # have been used to choose its domain, point counts or source points.
    reference = tmp_path / "reference.swc"
    reference.write_text("1 3 0 6 4 1 -1\n2 3 16 6 4 1 1\n")
    inventory = tmp_path / "training.json"
    inventory.write_text(json.dumps({"schema_version": 1, "cases": [{**case,
        "reference_swc": reference.name, "annotation_status": "reference_agreement"}]}))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*.npz")}
    assert len(hashes) == 12
    evaluation = subprocess.run([sys.executable, str(repo / "scripts/evaluate_phase1_sampling.py"),
        "--clouds", str(output / "clouds.json"), "--training-inventory", str(inventory)],
        capture_output=True, text=True)
    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    result = json.loads((output / "evaluation/report.json").read_text())
    assert len(result["rows"]) == 12 and not result["failures"]
    assert result["decision"]["automatic_training_authorized"] is False
    assert result["decision"]["promising_for_replication"] is False  # no junction class
    for path, before in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert image.read_bytes() == payload
    assert (output / "evaluation/overlays/case_00/reference_cloud_overlay.png").is_file()
    assert "SCORE case_id=synthetic-training-control" in evaluation.stdout
    # Unsupported reference metadata must remain visible, not become a score
    # or a passed one-case denominator. Point preparation remains unchanged.
    reference.write_text("1 3 0 6 4 1 -1\n2 -73 16 6 4 1 1\n")
    second = tmp_path / "unscorable_control"
    shutil.copytree(output, second, ignore=shutil.ignore_patterns(
        "evaluation", "matplotlib_cache", "reference_cloud_overlay.png"))
    invalid = subprocess.run([sys.executable, str(repo / "scripts/evaluate_phase1_sampling.py"),
        "--clouds", str(second / "clouds.json"), "--training-inventory", str(inventory)],
        capture_output=True, text=True)
    assert invalid.returncode == 0, invalid.stdout + invalid.stderr
    recorded = json.loads((second / "evaluation/report.json").read_text())
    assert recorded["decision"]["outcome"] == "incomplete"
    assert recorded["decision"]["case_accounting"] == {"declared": 1, "scored": 0, "unscorable_reference": 1}
    assert not recorded["rows"] and not recorded["decision"]["promising_for_replication"]
    assert recorded["failures"][0]["reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert "UNSCORABLE case_id=synthetic-training-control" in invalid.stdout
    # A named, opt-in geometry interpretation retains the signed metadata;
    # strict reports and clouds remain immutable.
    adapted = second / "evaluation_opaque_integer"
    compatible = subprocess.run([sys.executable, str(repo / "scripts/evaluate_phase1_sampling.py"),
        "--clouds", str(second / "clouds.json"), "--training-inventory", str(inventory),
        "--reference-type-policy", "opaque_integer", "--output-dir", str(adapted)],
        capture_output=True, text=True)
    assert compatible.returncode == 0, compatible.stdout + compatible.stderr
    preserved = json.loads((adapted / "report.json").read_text())
    assert len(preserved["rows"]) == 12 and not preserved["failures"]
    assert preserved["references"][0]["type_counts"] == {"-73": 1, "3": 1}
    assert preserved["references"][0]["nodes_retained"] == 2
    assert preserved["reference_type_policy"] == "opaque_integer"
    # Type metadata cannot change any geometry score on this identical curve.
    assert [r["metrics"] for r in preserved["rows"]] == [r["metrics"] for r in result["rows"]]
    for path, before in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert json.loads((second / "evaluation/report.json").read_text()) == recorded
