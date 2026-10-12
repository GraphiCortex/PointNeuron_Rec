"""Reference-tube oracle, diagnostic only: unchanged extractor, same two crops.

This input contains ground truth. It is not model inference or a reconstruction
result and must never replace the learned model's measured performance.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from pointneuron.data.medial_sampling import MedialSamplingConfig, image_medial_skeleton
from pointneuron.data.query_cache import file_digest
from scripts.pointneuron2.train_centerline_fit import load_records
from scripts.pointneuron2.audit_field_centerpaths import CASES, reference_segments, measure_paths, plot_paths


def oracle_mask(record, radius):
    expected = (record["distance"] <= radius) & record["valid"]
    if not np.array_equal(record["target"], expected):
        raise ValueError("Cached labels disagree with the exact continuous-segment tube")
    return expected.copy()


def run(args):
    torch.set_num_threads(2)
    manifest, records = load_records(args.manifest, ROOT / "configs/pointneuron2/partition_seed0.json")
    records = {(r["row"]["sample_index"], r["row"]["patch_index"]): r for r in records}
    prior_path = args.learned_paths / "centerpaths.json"
    prior = json.loads(prior_path.read_text())
    if (prior["status"] != "completed" or prior["manifest_sha256"] != file_digest(args.manifest)
            or prior["cases"] != [list(c) for c in CASES] or prior["threshold"] != .5
            or prior["kimimaro_version"] != version("kimimaro")):
        raise ValueError("Prior center-path comparison does not match this diagnostic")
    protected = {str(prior_path.resolve()): file_digest(prior_path),
                 str(args.manifest.resolve()): file_digest(args.manifest), **prior["checkpoint_and_report_hashes"]}
    if any(file_digest(path) != digest for path, digest in protected.items()):
        raise ValueError("Previously recorded checkpoint/report/cache has changed")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh oracle directory")
    args.output_dir.mkdir(parents=True)
    report = {"status": "running", "scope": "ground-truth-input oracle on two selected training crops; NOT learned-model performance",
              "cases": [list(c) for c in CASES], "input": "cached exact SWC tube intersected with valid native voxels",
              "tube_radius_voxels": manifest["config"]["tube_radius_voxels"],
              "mask_probability_threshold": .5, "kimimaro_version": version("kimimaro"),
              "prior_paths_sha256": file_digest(prior_path), "manifest_sha256": file_digest(args.manifest),
              "primary_gate_changed": False, "rows": []}
    started = time.perf_counter()
    for case in CASES:
        r = records[case]
        expected = next(row for row in prior["rows"] if (row["sample_index"], row["patch_index"], row["arm"]) == (*case, "hard_negative"))
        medial = MedialSamplingConfig(**r["metadata"]["medial_config"])
        if medial.to_dict() != expected["medial_config"]:
            raise ValueError("Extractor settings differ from learned-path run")
        mask = oracle_mask(r, manifest["config"]["tube_radius_voxels"])
        print(f"Oracle CPU extraction: neuron={case[0]} crop={case[1]} reference_tube_voxels={mask.sum()}", flush=True)
        t = time.perf_counter()
        vertices, edges = image_medial_skeleton(np.flatnonzero(mask), mask.shape[2], mask.shape[1], medial)
        extraction_seconds = time.perf_counter()-t
        if len(vertices):
            if not np.isfinite(vertices).all() or (vertices < -1e-5).any() or (vertices > np.array(mask.shape[::-1])-1+1e-5).any():
                raise AssertionError("Oracle paths escape native image frame")
            rounded = np.rint(vertices).astype(int)
            if not np.all(mask[tuple(rounded[:, ::-1].T)]):
                raise AssertionError("Oracle path vertex lacks reference-tube support")
        if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
            raise AssertionError("Oracle graph has invalid edge indices")
        # Evaluation reference is only supplied to this scorer, never extractor.
        full_reference, clipped_reference, _ = reference_segments(r, args.data_root)
        stats = measure_paths(vertices, edges, full_reference, clipped_reference)
        learned_path = args.learned_paths / f"sample_{case[0]:04d}_patch_{case[1]:03d}_hard_negative.npz"
        protected[str(learned_path.resolve())] = file_digest(learned_path)
        with np.load(learned_path, allow_pickle=False) as z:
            learned_vertices, learned_edges = z["vertices_local_xyz"], z["edges"]
            if not np.array_equal(z["origin_xyz"], r["metadata"]["origin_xyz"]):
                raise ValueError("Saved learned paths use a different coordinate origin")
        learned_stats = measure_paths(learned_vertices, learned_edges, full_reference, clipped_reference)
        for key, value in learned_stats.items():
            if value != expected[key]:
                raise AssertionError(f"Shared path scorer changes the prior learned result: {key}")
        row = {"sample_index": case[0], "patch_index": case[1], "medial_config": medial.to_dict(),
               "oracle_input_voxels": int(mask.sum()), "oracle_extraction_seconds": extraction_seconds,
               "learned": learned_stats, "oracle": stats}
        report["rows"].append(row)
        np.savez_compressed(args.output_dir / f"sample_{case[0]:04d}_patch_{case[1]:03d}_oracle.npz",
                            vertices_local_xyz=vertices, edges=edges, origin_xyz=np.array(r["metadata"]["origin_xyz"]))
        plot_paths(r, [("hard_negative", learned_vertices, learned_edges), ("oracle", vertices, edges)],
                   args.output_dir / f"sample_{case[0]:04d}_patch_{case[1]:03d}_oracle_paths.png",
                   diagnostic_title="Diagnostic only: learned output versus ground-truth tube input")
        print(f"neuron={case[0]} crop={case[1]}: learned_proximity6={learned_stats['paths_to_reference']['within_voxels']['6']} "
              f"oracle_proximity6={stats['paths_to_reference']['within_voxels']['6']} "
              f"oracle_reference_coverage2={stats['reference_to_paths']['within_voxels']['2']} "
              f"oracle_length={stats['path_length_voxels']:.1f}", flush=True)
        (args.output_dir / "oracle.json").write_text(json.dumps(report, indent=2))
    if any(file_digest(path) != digest for path, digest in protected.items()):
        raise AssertionError("Protected model/data/path artifact changed")
    report.update(status="completed", learned_path_metrics_reproduced=True, protected_artifacts_unchanged=True,
                  protected_artifact_hashes=protected, elapsed_seconds=time.perf_counter()-started,
                  optimizer_steps=0, checkpoint_inference_passes=0)
    (args.output_dir / "oracle.json").write_text(json.dumps(report, indent=2))
    print(f"report: {args.output_dir / 'oracle.json'}", flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--learned-paths", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
