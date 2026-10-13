"""CPU-only saved-weight inference audit; no optimizer or checkpoint changes.

Compare ordinary eval with current-crop BatchNorm statistics on temporary model
copies. The latter is a diagnostic, not a promoted deployment policy.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import torch
from torch import nn
from scripts.pointneuron2 import pointnext_query_experiment as experiment


def main(directory):
    torch.set_num_threads(2)
    args = experiment.parser().parse_args([
        "--action", "preflight", "--device", "cpu",
        "--experiment-dir", str(directory),
        "--native-manifest", str(ROOT / "tmp/pointneuron2/local_runs/field_pilot_20261012/cache/manifest.json"),
        "--field-models", str(ROOT / "tmp/pointneuron2/local_runs/field_attention_20261013/baseline/model"),
        "--data-root", str(ROOT.parent / "data/gold166"),
    ])
    recipe, manifest, records, protected = experiment.load_cache(args)
    fit_path = directory / "model/fit_report.json"
    checkpoint_path = directory / "model/step_2000.pt"
    fit = experiment.read_json(fit_path)
    if fit["status"] != "completed" or fit["optimizer_updates"] != 2000:
        raise ValueError("Expected completed fixed 2,000-update fit")
    protected.update(fit["source_files_sha256"])
    protected[str(fit_path.resolve())] = experiment.file_digest(fit_path)
    protected[str(checkpoint_path.resolve())] = fit["final_checkpoint_sha256"]
    experiment.protected_unchanged(protected)
    if fit["cache_manifest_sha256"] != experiment.file_digest(directory / "cache/manifest.json"):
        raise ValueError("Fit/cache identity differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    original = experiment.load_model_checkpoint(checkpoint).requires_grad_(False)
    rows = []
    for mode in ("ordinary_eval", "current_crop_batchnorm_diagnostic"):
        for record in records:
            model = copy.deepcopy(original).eval()
            normalizers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            if mode != "ordinary_eval":
                for module in normalizers:
                    module.train()
            with torch.no_grad():
                prediction = experiment.forward(model, record)
                loss = experiment.query_set_loss(prediction, record["nodes"], record["mask"], **recipe["query_loss"])
                accepted = prediction.objectness_logits.softmax(-1)[0, :, 1] >= .5
                support = experiment.measure_support(prediction.center_proposals[0, accepted].double().numpy(), record["geometry"])
            matching = experiment.inspect_matches(prediction, record["nodes"], record["mask"], loss)[0]
            coverage = support["reference_to_points"]["within_voxels"]["6"]
            passed = (matching["max_matched_distance"] <= 1
                      and matching["matched_accepted_within_6"] == matching["targets"]
                      and matching["unused_accepted"] == 0 and coverage >= .95
                      and bool(torch.isfinite(prediction.radius).all()) and bool((prediction.radius > 0).all()))
            # BatchNorm buffers may change only in the disposable copy. All parameters stay fixed.
            assert all(torch.equal(p, dict(original.named_parameters())[name]) for name, p in model.named_parameters())
            rows.append({"mode": mode, "crop": record["metadata"]["patch_index"],
                         "loss": float(loss.total), "matching": matching,
                         "selection": experiment.inspect_selection(prediction, record["nodes"], record["mask"]),
                         "accepted_curve_coverage_6": coverage,
                         "learnability_gate_passed": passed, "batchnorm_layers": len(normalizers)})
    experiment.protected_unchanged(protected)
    result = {"status": "completed", "device": "CPU", "optimizer_updates": 0,
              "checkpoint_writes": 0, "scope": "saved weights, temporary BN-mode comparison; not a deployment change",
              "protected_files_unchanged": True, "rows": rows}
    output = directory / "diagnostics/inference_normalization.json"
    if output.exists():
        raise FileExistsError("Preserve existing audit outputs")
    output.parent.mkdir(exist_ok=True)
    experiment.write_json(output, result)
    print(experiment.json.dumps(result, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        default=ROOT / "tmp/pointneuron2/local_runs/pointnext_query_20261013")
    main(parser.parse_args().experiment_dir)
