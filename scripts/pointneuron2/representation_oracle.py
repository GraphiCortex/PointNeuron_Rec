"""Bounded training-reference graph/SWC feasibility check; no images or models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pointneuron.data.curve_targets import curve_geometry, patch_targets
from pointneuron.data.swc import parse_swc
from pointneuron.graph.representation_oracle import (
    candidate_report, clipped_reference_graph, export_and_verify, geometry_report,
    segments, target_graph, topology_comparison,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def overlay(reference, target, path, title, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, dimensions, labels in zip(axes, [(0, 1), (0, 2)], [("X", "Y"), ("X", "Z")]):
        axis.add_collection(LineCollection(segments(reference)[:, :, dimensions], colors="#a7acb2", linewidths=2,
                                           label="Original reference curve"))
        axis.add_collection(LineCollection(segments(target)[:, :, dimensions], colors="#147d92", linewidths=.9,
                                           label="Oracle target chords"))
        axis.scatter(target.xyz[:, dimensions[0]], target.xyz[:, dimensions[1]], s=7, c="#c35726", zorder=3,
                     label="Target nodes")
        axis.autoscale()
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel(f"{labels[0]} (native voxels)")
        axis.set_ylabel(f"{labels[1]} (native voxels)")
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    at2 = metric["tolerances"]["2.0"]
    figure.suptitle(f"{title}\nReference oracle only | arclength P/R at 2 voxels: "
                   f"{at2['precision']:.2%} / {at2['recall']:.2%}", fontsize=11)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def overview(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [f"{r['name'].split('_')[0].title()}\n{r['sample_index']}/"
             + ("+".join(map(str, r["patch_indices"])) if r["patch_indices"] is not None else "full") for r in rows]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].bar(x-.18, [r["geometry"]["tolerances"]["2.0"]["precision"]*100 for r in rows], .36, label="Precision")
    axes[0].bar(x+.18, [r["geometry"]["tolerances"]["2.0"]["recall"]*100 for r in rows], .36, label="Recall")
    axes[0].axhline(99, color="#a23c28", ls="--", lw=1, label="99% criterion")
    axes[0].set_title("Arclength geometry at 2 voxels")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8)
    for offset, key, label in [(-.18, "knn", "16-neighbour"), (.18, "knn_union_radius", "+ radius 8")]:
        axes[1].bar(x+offset, [r["candidates"][key]["required_edge_recall"]*100 for r in rows], .36, label=label)
    axes[1].set_title("Known-edge candidate recall")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    axes[2].bar(x, [r["topology"]["target"]["nodes"] for r in rows], color="#147d92")
    axes[2].axhline(128, color="#a23c28", ls="--", label="Historical 128 anchors (illustration)")
    axes[2].set_title("Target counts (no global truncation)")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.set_xticks(x, names, fontsize=7)
        axis.grid(axis="y", alpha=.15)
    figure.suptitle("Reference-only representation feasibility; no learned reconstruction", fontsize=12)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(args):
    started = time.perf_counter()
    config = load_json(args.config)
    if config["schema"] != "pointneuron2-representation-oracle-v1":
        raise ValueError("Unexpected configuration schema")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory; existing evidence is immutable")
    audit_path = REPO_ROOT/config["target_audit"]
    if digest(audit_path) != config["target_audit_sha256"]:
        raise ValueError("Target audit hash mismatch")
    audit = load_json(audit_path)
    partition_path = REPO_ROOT/config["partition"]
    if audit["partition_sha256"] != digest(partition_path) or audit["verification"] != "passed":
        raise ValueError("Audit/partition provenance mismatch")
    partition = load_json(partition_path)
    allowed = {r["sample_index"]: r["sample_id"] for r in partition["partitions"]["train"]}
    for role in ("val", "test"):
        if any(r["sample_index"] in allowed or r["sample_id"] in allowed.values()
               for r in partition["partitions"][role]):
            raise ValueError("Neuron partitions overlap")
    sources = {r["sample_index"]: r for r in audit["sources"]}
    records = {(r["sample_index"], r["patch_index"]): r for r in audit["records"]}
    protected_paths = [REPO_ROOT/"src/pointneuron/data/curve_targets.py", partition_path, audit_path]
    protected_paths.extend(sorted((REPO_ROOT/"src/pointneuron/models").glob("*.py")))
    code_paths = [Path(__file__).resolve(), REPO_ROOT/"src/pointneuron/graph/representation_oracle.py",
                  REPO_ROOT/"src/pointneuron/data/swc.py"]
    protected_paths.extend(code_paths)
    protected_paths.append(args.config.resolve())
    protected = {str(p.relative_to(REPO_ROOT)): digest(p) for p in protected_paths}
    loaded = {}
    for case in config["cases"]:
        index = case["sample_index"]
        if index not in allowed or index not in sources:
            raise ValueError("Case is outside the verified training partition")
        source = sources[index]
        path = (args.data_root/source["path"]).resolve()
        if not path.is_relative_to(args.data_root.resolve()) or path.parent != (args.data_root/allowed[index]).resolve():
            raise ValueError("Source path escapes its training-neuron directory")
        if digest(path) != source["sha256"]:
            raise ValueError("Source SWC hash mismatch")
        loaded[index] = (path, parse_swc(path))
    args.output_dir.mkdir(parents=True)
    snapshot_dir = args.output_dir/"source_snapshot"
    snapshot_dir.mkdir()
    for path in code_paths + [args.config.resolve()]:
        # Relative paths preserve the two different representation_oracle.py files.
        relative = path.relative_to(REPO_ROOT)
        snapshot = snapshot_dir/relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    rows, failures = [], []
    for case in config["cases"]:
        case_started = time.perf_counter()
        print(f"START {case['name']} (training reference only)", flush=True)
        try:
            source_path, tree = loaded[case["sample_index"]]
            per_crop = []
            component_sum = 0
            boxes = None
            if case["patch_indices"] is not None:
                boxes = []
                for patch in case["patch_indices"]:
                    record = records[(case["sample_index"], patch)]
                    if record["sample_id"] != allowed[case["sample_index"]]:
                        raise ValueError("Crop identity mismatch")
                    center = np.asarray(record["patch_center"])
                    count = len(patch_targets(curve_geometry(tree), center, config["patch_radius"], config["step_voxels"]))
                    expected = record["steps"][str(float(config["step_voxels"]))]["targets"]
                    if count != expected or count > config["query_capacity_per_crop"]:
                        raise ValueError("Target count differs from audit or exceeds query capacity")
                    per_crop.append({"patch_index": patch, "target_count": count, "center": center.tolist()})
                    boxes.append((center, config["patch_radius"]))
                    component_sum += topology_comparison(clipped_reference_graph(tree, [boxes[-1]]),
                                                        target_graph(tree, [boxes[-1]], config["step_voxels"]))["target"]["components"]
            reference = clipped_reference_graph(tree, boxes)
            target = target_graph(tree, boxes, config["step_voxels"])
            topo = topology_comparison(reference, target)
            shared = sum(c["target_count"] for c in per_crop)-len(target.xyz) if per_crop else 0
            joining = {"shared_target_identities": shared,
                       "per_crop_component_sum": component_sum if per_crop else None,
                       "component_reduction": component_sum-topo["target"]["components"] if per_crop else None,
                       "required": case.get("require_shared_boundary", False)}
            candidates = candidate_report(target, config["candidate_neighbors"], config["candidate_radius_voxels"])
            path = args.output_dir/f"{case['name']}.swc"
            serialization = export_and_verify(target, path, config["radius_floor_voxels"])
            serialized = clipped_reference_graph(parse_swc(path))
            metrics = geometry_report(reference, serialized, config["quadrature_maximum_step_voxels"],
                                      config["tolerances_voxels"])
            geometry_pass = all(value is not None and value >= config["minimum_precision_and_recall_at_2"]
                                for value in metrics["tolerances"]["2.0"].values())
            candidate_pass = candidates["knn_union_radius"]["missing_required_edges"] == 0
            gates = {"topology": bool(topo["passed"]), "candidate_recall": candidate_pass,
                     "geometry": geometry_pass, "swc_roundtrip": serialization["passed"],
                     "boundary_join_exercised": not joining["required"]
                         or (shared > 0 and joining["component_reduction"] > 0)}
            np.savez_compressed(args.output_dir/f"{case['name']}_graphs.npz",
                                reference_xyz=reference.xyz, reference_edges=np.array(reference.edges, dtype=int).reshape(-1, 2),
                                target_xyz=target.xyz, target_edges=np.array(target.edges, dtype=int).reshape(-1, 2))
            (args.output_dir/f"{case['name']}_identities.json").write_text(json.dumps(target.identities, indent=2)+"\n", encoding="utf-8")
            overlay(reference, target, args.output_dir/f"{case['name']}.png", case["name"], metrics)
            row = {**case, "source_sha256": digest(source_path), "per_crop": per_crop,
                   "topology": topo, "candidates": candidates, "geometry": metrics, "oracle_joining": joining,
                   "serialization": serialization, "gates": gates, "gate_passed": all(gates.values()),
                   "above_illustrative_128_anchor_budget": len(target.xyz) > config["illustrative_legacy_anchor_budget"],
                   "elapsed_seconds": time.perf_counter()-case_started}
            rows.append(row)
            print(f"DONE {case['name']}: nodes={len(target.xyz)} edges={len(target.edges)} "
                  f"P/R@2={metrics['tolerances']['2.0']['precision']:.4f}/{metrics['tolerances']['2.0']['recall']:.4f} "
                  f"gates={gates}", flush=True)
        except Exception as error:
            failures.append({"name": case["name"], "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {case['name']}: {error}", flush=True)
    if protected != {str(p.relative_to(REPO_ROOT)): digest(p) for p in protected_paths}:
        raise RuntimeError("Protected source/partition/audit changed during the diagnostic")
    for index, (path, _) in loaded.items():
        if digest(path) != sources[index]["sha256"]:
            raise RuntimeError("Source SWC changed during the diagnostic")
    if rows:
        overview(rows, args.output_dir/"overview.png")
    report = {"schema": config["schema"], "status": "completed" if not failures else "execution_failed",
              "scope": "three training neurons, four reference-only cases; no images/models/GPU/training/dev/test references",
              "distance_units": "native-voxel Euclidean; not physically calibrated",
              "config": config, "config_sha256": digest(args.config), "audit_sha256": digest(audit_path),
              "code_sha256": {str(p.relative_to(REPO_ROOT)): digest(p) for p in code_paths},
              "protected_sha256": protected, "preservation_checks_passed": True,
              "rows": rows, "failures": failures,
              "gate_passed": not failures and len(rows) == len(config["cases"]) and all(r["gate_passed"] for r in rows),
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": [
                  "Reference identity/order supplies perfect nodes, consolidation and edge labels; no deployable matcher or learned decoder is evaluated.",
                  "Candidate graphs use symmetrized kNN union radius, not an implemented image/tracer initializer.",
                  "Radius 8 should include consecutive spacing-8 targets by construction; recall alone does not validate edge discrimination.",
                  "Nonrequired candidates are not certified biological negatives, especially between annotation fragments.",
                  "The full-reference case bypasses crop coverage and does not apply a 512-node global cap.",
                  "Historical 128 anchors are illustrative only; no graph budget is imposed or promoted.",
                  "Geometry uses midpoint arclength quadrature, with exact point-to-segment distances at those samples.",
                  "No phase/reroot perturbation, noisy proposals, learned stitching, input quality or generalization claim."]}
    (args.output_dir/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Execution: {report['status']}; representation gate: {report['gate_passed']}; "
          f"elapsed {report['elapsed_seconds']:.2f}s; no training or inference", flush=True)
    print(f"Results: {args.output_dir/'summary.json'}", flush=True)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT/"configs/pointneuron2/representation_oracle.json")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent/"data/gold166")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
