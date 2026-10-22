"""Evaluate already-saved input clouds; references never influence generation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.evaluation.cloud_support import CloudSupport
from pn2.evaluation.swc import TYPE_POLICIES, parse_swc
from pn2.inputs.cases import ImageCase, read_object
from pn2.inputs.image import decode_image
from pn2.inputs.sampling import partition_tiles, selection_count


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(rows, recipe):
    grouped = defaultdict(list)
    tolerance = str(float(recipe["evaluation"]["primary_tolerance_voxel"]))
    fields = ["curve_support", "worst_branch_support", "branch_p10_support",
              "longest_unsupported_run", "endpoint_support", "junction_support",
              "selected_point_reference_agreement"]
    for row in rows:
        grouped[(row["case_id"], row["cell_side_voxel"], row["method"])].append(row)
    summary = []
    for (case, cell, method), group in grouped.items():
        metrics = {}
        for field in fields:
            values = [g["metrics"]["tolerances"][tolerance][field] for g in group]
            metrics[field] = None if any(v is None for v in values) else {
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "minimum": min(values), "maximum": max(values)}
        summary.append({"case_id": case, "cell_side_voxel": cell, "method": method,
                        "seed_count": len(group), "unique_count": group[0]["unique_count"], "metrics": metrics})
    return summary


def decision(summary, rows, recipe):
    settings = recipe["evaluation"]
    lean, dense = max(recipe["cell_sides_voxel"]), min(recipe["cell_sides_voxel"])
    lookup = {(r["case_id"], r["cell_side_voxel"], r["method"]): r for r in summary}
    checks = []
    def record(case, cell, control, metric, difference, required, passed):
        checks.append({"case_id": case, "cell_side_voxel": cell, "control": control,
                       "metric": metric, "sato_minus_control": difference, "required": required, "passed": bool(passed)})
    for case in dict.fromkeys(r["case_id"] for r in summary):
        for cell in recipe["cell_sides_voxel"]:
            for control in ("uniform", "intensity"):
                a = lookup[(case, cell, "sato")]["metrics"]
                b = lookup[(case, cell, control)]["metrics"]
                gain = a["curve_support"]["mean"] - b["curve_support"]["mean"]
                bound = settings["minimum_mean_gain"] if cell == lean else -settings["maximum_local_coverage_drop"]
                record(case, cell, control, "curve_support", gain, f">= {bound}", gain >= bound)
                for field in ("worst_branch_support", "branch_p10_support", "endpoint_support", "junction_support"):
                    if a[field] is None or b[field] is None:
                        record(case, cell, control, field, None, "defined structural class", False)
                    else:
                        delta = a[field]["mean"] - b[field]["mean"]
                        bound = -settings["maximum_local_coverage_drop"]
                        record(case, cell, control, field, delta, f">= {bound}", delta >= bound)
                delta = a["longest_unsupported_run"]["mean"] - b["longest_unsupported_run"]["mean"]
                bound = settings["maximum_gap_increase_voxel"]
                record(case, cell, control, "longest_unsupported_run", delta, f"<= {bound}", delta <= bound)
    refinement = max((r["primary_refinement_absolute_change"] for r in rows), default=None)
    outside = max((r["metrics"]["outside_image_quadrature_samples"] for r in rows), default=0)
    return {"scope": "Promising for broader training-input replication only; no reconstruction/generalization verdict",
            "promising_for_replication": bool(checks) and all(c["passed"] for c in checks)
                and refinement is not None and refinement <= .01 and outside == 0,
            "maximum_refinement_change": refinement, "reference_outside_image_samples": outside,
            "coordinate_review": "manual diagnostic overlay review still required; no alignment fitted",
            "checks": checks, "automatic_training_authorized": False}


def overlay(folder, cloud_root, item, raw, morphology):
    os.environ.setdefault("MPLCONFIGDIR", str(folder / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 7), constrained_layout=True)
    lean = max(o["cell_side_voxel"] for o in item["outputs"])
    for column, method in enumerate(("reference", "uniform", "intensity", "sato")):
        selected = next((o for o in item["outputs"] if o["method"] == method
                         and o["seed"] == 0 and o["cell_side_voxel"] == lean), None)
        xyz = np.load(cloud_root / selected["path"])["xyz_native"] if selected else None
        for row, (projection, pair, labels) in enumerate(((raw.max(axis=0), (0, 1), ("X", "Y")),
                                                         (raw.max(axis=1), (0, 2), ("X", "Z")))):
            ax = axes[row, column]
            ax.imshow(np.log1p(projection.astype(float)), origin="lower", cmap="gray", aspect="equal")
            if xyz is not None:
                ax.scatter(xyz[:, pair[0]], xyz[:, pair[1]], s=.2, c="#ff58cb", alpha=.25, linewidths=0)
            for segment in morphology.segments:
                ax.plot(segment[:, pair[0]], segment[:, pair[1]], c="#64ed67", linewidth=.5, alpha=.8)
            ax.set(title=f"{method}; {'XY' if row == 0 else 'XZ'} projection",
                   xlabel=f"{labels[0]} native voxel", ylabel=f"{labels[1]} native voxel")
    fig.suptitle("Original image log1p display; green reference; magenta selected points\n"
                 "Seed 0 / lean density; projections do not establish 3D connectivity", fontsize=11)
    fig.savefig(folder / "reference_cloud_overlay.png", dpi=150)
    plt.close(fig)


def evaluate(cloud_manifest, inventory_path, *, reference_type_policy="nonnegative", output_dir=None):
    cloud_manifest, inventory_path = Path(cloud_manifest).resolve(), Path(inventory_path).resolve()
    root = cloud_manifest.parent
    if reference_type_policy not in TYPE_POLICIES:
        raise ValueError("Unknown reference type policy")
    if reference_type_policy != "nonnegative" and output_dir is None:
        raise ValueError("Geometry-only reference mode requires an explicit fresh output directory")
    target = Path(output_dir).resolve() if output_dir is not None else root / "evaluation"
    target.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    code_hashes = {"evaluation_script_sha256": digest(__file__),
                   "support_code_sha256": digest(source_root / "src/pn2/evaluation/cloud_support.py"),
                   "distance_code_sha256": digest(source_root / "src/pn2/evaluation/geometry.py"),
                   "swc_reader_sha256": digest(source_root / "src/pn2/evaluation/swc.py")}
    clouds, inventory = read_object(cloud_manifest), read_object(inventory_path)
    if digest(root / "recipe.json") != clouds["frozen_recipe_sha256"]:
        raise ValueError("Frozen comparison recipe changed before evaluation")
    recipe = json.loads((root / "recipe.json").read_text(encoding="utf-8"))
    settings = recipe["evaluation"]
    tolerance = str(float(settings["primary_tolerance_voxel"]))
    rows, references, failures = [], [], []
    started = time.perf_counter()
    for case_number, item in enumerate(clouds["cases"]):
        case = item["case_id"]
        matches = [c for c in inventory["cases"] if c["case_id"] == case and c["partition"] == "training"]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one explicit training reference for {case}")
        row = matches[0]
        if row["coordinates"] != {"mode": "native", "units": "voxel"}:
            raise ValueError("This comparison requires native-voxel reference coordinates")
        image_path = (inventory_path.parent / row["image_path"]).resolve()
        if image_path != Path(item["image"]["image_path"]).resolve() or digest(image_path) != item["image"]["source_sha256"]:
            raise ValueError("Image provenance changed; stop before reference scoring")
        decoded = decode_image(ImageCase(case, image_path, "training"))
        raw = decoded.array_zyx
        reference = (inventory_path.parent / row["reference_swc"]).resolve()
        try:
            morphology = parse_swc(reference, type_policy=reference_type_policy)
            metric = CloudSupport(morphology, settings["quadrature_step_voxel"], decoded.dimensions_xyz)
            refined = CloudSupport(morphology, settings["quadrature_step_voxel"] / 2, decoded.dimensions_xyz)
            if metric.index is None:
                raise ValueError("No positive reference curve length")
        except (OSError, ValueError) as error:
            failures.append({"case_id": case, "status": "unscorable_reference", "reason": str(error),
                             "reference_path": str(reference),
                             "reference_sha256": digest(reference) if reference.is_file() else None})
            print(f"UNSCORABLE case_id={case} reason={error}", flush=True)
            continue
        codes, counts = np.unique(morphology.types, return_counts=True)
        references.append({"case_id": case, "path": str(reference), "sha256": digest(reference),
                           "annotation_status": row["annotation_status"],
                           "type_policy": reference_type_policy,
                           "type_counts": {str(int(k)): int(v) for k, v in zip(codes, counts)},
                           "negative_type_rows": int(np.count_nonzero(morphology.types < 0)),
                           "nodes_retained": morphology.node_count,
                           "type_interpretation": "opaque metadata; no anatomical labels inferred; no nodes filtered"})
        case_rows = []
        for output in item["outputs"]:
            start = time.perf_counter()
            path = root / output["path"]
            if digest(path) != output["sha256"]:
                raise ValueError("Saved cloud changed before scoring")
            with np.load(path, allow_pickle=False) as packet:
                xyz, flat, intensity = packet["xyz_native"], packet["flat_indices"], packet["intensity"]
            expected = sum(selection_count(int(raw[tile].size), output["cell_side_voxel"])
                           for tile in partition_tiles(raw.shape, recipe["tile_side_voxel"]))
            if len(flat) != expected or len(np.unique(flat)) != expected or output["unique_count"] != expected:
                raise ValueError("Incorrect source-point count or duplication")
            if (flat < 0).any() or (flat >= raw.size).any():
                raise ValueError("Source voxel index outside image")
            np.testing.assert_array_equal(xyz, np.array(np.unravel_index(flat, raw.shape))[::-1].T)
            np.testing.assert_array_equal(intensity, raw.ravel()[flat])
            metrics = metric.measure(xyz, settings["tolerances_voxel"])
            finer = refined.measure(xyz, [settings["primary_tolerance_voxel"]], include_point_agreement=False)
            result = {"case_id": case, **output, "metrics": metrics,
                      "primary_refinement_absolute_change": abs(metrics["tolerances"][tolerance]["curve_support"]
                          - finer["tolerances"][tolerance]["curve_support"]),
                      "evaluation_seconds": time.perf_counter() - start}
            case_rows.append(result)
            print(f"SCORE case_id={case} {output['method']} cell={output['cell_side_voxel']} seed={output['seed']} "
                  f"support@2={metrics['tolerances'][tolerance]['curve_support']:.4f} "
                  f"seconds={result['evaluation_seconds']:.2f}", flush=True)
        rows.extend(case_rows)
        write(target / f"case_{case_number:02d}.json", {"case_id": case, "rows": case_rows})
        overlay_folder = target / "overlays" / f"case_{case_number:02d}"
        overlay_folder.mkdir(parents=True, exist_ok=False)
        overlay(overlay_folder, root, item, raw, morphology)
    summary = summarize(rows, recipe)
    verdict = decision(summary, rows, recipe)
    if failures or len(references) != len(clouds["cases"]):
        verdict["promising_for_replication"] = False
    verdict["outcome"] = "incomplete" if failures else (
        "promising_for_replication" if verdict["promising_for_replication"] else "criteria_not_met")
    verdict["case_accounting"] = {"declared": len(clouds["cases"]), "scored": len(references),
                                  "unscorable_reference": len(failures)}
    result = {"schema_version": 1, "cloud_manifest_sha256": digest(cloud_manifest),
              **code_hashes, "reference_type_policy": reference_type_policy,
              "recipe": recipe, "references": references, "failures": failures,
              "rows": rows, "summary": summary, "decision": verdict,
              "evaluation_seconds": time.perf_counter() - started}
    write(target / "report.json", result)
    print(f"COMPLETED input-support evaluation; promising={verdict['promising_for_replication']} "
          f"report={target / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds", required=True, type=Path)
    parser.add_argument("--training-inventory", required=True, type=Path)
    parser.add_argument("--reference-type-policy", choices=TYPE_POLICIES, default="nonnegative")
    parser.add_argument("--output-dir", type=Path, help="Fresh evaluation directory; required for opaque reference metadata")
    args = parser.parse_args()
    evaluate(args.clouds, args.training_inventory, reference_type_policy=args.reference_type_policy,
             output_dir=args.output_dir)
