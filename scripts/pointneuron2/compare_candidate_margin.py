"""Fixed radius-eight/twelve CPU comparison; no threshold search or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from pointneuron.graph.proposal_graph import ProposalGraphInput, build_candidate_graph
from pointneuron.data.connectivity_targets import assign_edge_labels
from scripts.pointneuron2 import branched_query_experiment as base


def compare_prediction(saved, record, recipe, radii=(8., 12.)):
    xyz = torch.from_numpy(saved["centers"][0])
    local = (xyz-torch.tensor(record["meta"]["patch_center"], dtype=xyz.dtype))/48.
    packet = ProposalGraphInput(torch.from_numpy(saved["centers"][0]), torch.from_numpy(saved["radii"][0]),
        torch.from_numpy(saved["logits"][0]), local,
        torch.from_numpy(saved["decoder_features"][0]), torch.arange(512), {"scope": "fixed candidate comparison"})
    accepted = packet.objectness_logits[:, 1] >= packet.objectness_logits[:, 0]
    assignment = (saved["pred_indices"], saved["target_indices"])
    rows, first_pairs = [], None
    for radius in radii:
        start = time.perf_counter()
        graph = build_candidate_graph(packet.positions, packet.node_ids, neighbors=recipe["candidate_neighbors"], radius=radius)
        elapsed = time.perf_counter()-start
        labels = assign_edge_labels(packet, graph, record["nodes"][:, 1:4], record["target_edges"],
            record["target_components"], assignment, maximum_match_error=recipe["trusted_match_tolerance"])
        pairs = {tuple(map(int, p)) for p in graph.edges.numpy()}
        if first_pairs is not None and not first_pairs <= pairs:
            raise AssertionError("Larger radius lost candidate pairs")
        first_pairs = pairs
        positive = labels.labels == 1
        accepted_count = int((positive & accepted[graph.edges].all(-1)).sum())
        n = labels.statistics["required_target_edges"]
        rows.append({"radius": radius, "candidate_edges": len(graph.edges), "directed_messages_including_self": 2*len(graph.edges)+512,
            "cpu_candidate_construction_seconds": elapsed, "connectivity_supervision": labels.statistics,
            "edge_ceiling_after_node_cutoff": accepted_count/n if n else None})
    return rows


def run(args):
    spec = base.read(args.config)
    assert spec["candidate_radii_for_reporting"] == [8., 12.] and spec["candidate_neighbors"] == 16
    experiment = ROOT/spec["baseline_experiment"]
    report_path = experiment/"model/report.json"
    assert base.file_digest(report_path) == spec["baseline_report_sha256"]
    report = base.read(report_path)
    recipe = base.recipe_at(ROOT/spec["baseline_recipe"])
    assert report["recipe"] == recipe and report["status"] == "completed"
    base.verify_hashes(report["provenance"]["source_files_sha256"])
    manifest, records = base.load_records(argparse.Namespace(experiment_dir=experiment), recipe)
    if args.output_dir.exists():
        raise FileExistsError("Preserve prior comparison")
    if not args.output_dir.resolve().is_relative_to((ROOT/"tmp/pointneuron2").resolve()):
        raise ValueError("Generated outputs belong under ignored tmp")
    rows = []
    for evaluation in report["evaluations"]:
        for i, record in enumerate(records):
            path = experiment/f"model/step_{evaluation['step']:04d}_crop_{i}.npz"
            with np.load(path, allow_pickle=False) as z:
                saved = {k: z[k].copy() for k in z.files}
            comparison = compare_prediction(saved, record, recipe)
            assert comparison[0]["connectivity_supervision"] == evaluation["crops"][i]["connectivity_supervision"]
            edges = record["target_edges"]
            maximum_chord = float(np.linalg.norm(record["nodes"][edges[:, 0], 1:4].astype(float)-record["nodes"][edges[:, 1], 1:4], axis=1).max())
            row = {"step": evaluation["step"], "sample_index": record["identity"]["sample_index"],
                "patch_index": record["identity"]["patch_index"], "prediction_sha256": base.file_digest(path),
                "maximum_cached_target_chord": maximum_chord,
                "actual_chord_plus_two_endpoint_tolerances": maximum_chord+2*recipe["trusted_match_tolerance"],
                "arms": comparison}
            rows.append(row)
            if evaluation["step"] == 4000:
                print(json.dumps(row), flush=True)
    base.verify_hashes(report["provenance"]["source_files_sha256"])
    args.output_dir.mkdir(parents=True)
    result = {"status": "completed", "schema": "pointneuron2-candidate-margin-comparison-v1", "config": spec,
        "fit_report_sha256": base.file_digest(report_path), "cache_manifest_sha256": base.file_digest(experiment/"cache/manifest.json"),
        "review_source_sha256": base.file_digest(Path(__file__)), "rows": rows,
        "optimizer_updates": 0, "model_forwards": 0, "scope": "fixed training predictions; no dev/test, fitting or production policy change",
        "note": "Nominal radius 12 is derived from spacing 8 plus two endpoint errors of 2. Cached float32 chords can differ slightly; no exact floating-point guarantee is asserted."}
    base.write(args.output_dir/"summary.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT/"configs/pointneuron2/balanced_query_comparison.json")
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
