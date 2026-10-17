"""CPU execution checks for new connectivity; zero optimizer updates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT/"src"))

from pointneuron.data.connectivity_targets import assign_edge_labels, full_reference_components, masked_edge_loss
from pointneuron.data.swc import parse_swc
from pointneuron.graph.connectivity_decoder import decode_forest, export_decoded_swc
from pointneuron.graph.proposal_graph import ProposalGraphInput, build_candidate_graph, tensor_digest
from pointneuron.graph.representation_oracle import ReferenceGraph, clipped_reference_graph, target_graph, topology_comparison
from pointneuron.models.gatv2_connectivity import (
    ConnectivityConfig, GATv2Connectivity, connectivity_checkpoint, load_connectivity_checkpoint,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def synthetic_packet(n, seed):
    generator = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(n, 3, generator=generator)*96).requires_grad_()
    radii = (torch.rand(n, 1, generator=generator)+1).requires_grad_()
    logits = torch.randn(n, 2, generator=generator, requires_grad=True)
    features = torch.randn(n, 128, generator=generator, requires_grad=True)
    return ProposalGraphInput(xyz, radii, logits, (xyz-48)/48, features, torch.arange(n),
                              {"source": "synthetic_execution_only"})


def check_synthetic(config, output):
    rows = []
    initial_hashes = None
    for mode in ("gatv2", "pairwise"):
        packet = synthetic_packet(config["synthetic_queries"], config["seed"])
        candidates = build_candidate_graph(packet.positions, packet.node_ids,
                                            neighbors=config["candidate_neighbors"], radius=config["candidate_radius_voxels"])
        hashes = {name: tensor_digest(getattr(packet, name)) for name in
                  ("positions", "radii", "objectness_logits", "local_centers", "decoder_features")}
        hashes["candidate_edges"] = tensor_digest(candidates.edges)
        if initial_hashes is not None and initial_hashes != hashes:
            raise RuntimeError("Synthetic control inputs differ")
        initial_hashes = hashes
        model = GATv2Connectivity(ConnectivityConfig(mode=mode, seed=config["seed"]))
        start = time.perf_counter()
        model.train()
        scores = model(packet, candidates)
        # Alternating synthetic labels exercise both BCE branches, not a fit.
        labels = (torch.arange(len(candidates.edges)) % 2).to(packet.positions.dtype)
        loss = masked_edge_loss(scores.edge_logits, labels)
        loss.total.backward()
        gradients_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not gradients_finite:
            raise RuntimeError("Missing/nonfinite synthetic parameter gradients")
        model.eval()
        with torch.no_grad():
            evaluation = model(packet, candidates).edge_logits
        delta = float((scores.edge_logits.detach()-evaluation).abs().max())
        if delta != 0:
            raise RuntimeError("Train/eval connectivity mismatch")
        checkpoint = connectivity_checkpoint(model, provenance={"source": "synthetic_execution_only", "optimizer_updates": 0})
        path = output/f"{mode}_untrained.pt"
        torch.save(checkpoint, path)
        reloaded = load_connectivity_checkpoint(torch.load(path, map_location="cpu", weights_only=False), expected_mode=mode)
        with torch.no_grad():
            reload_delta = float((reloaded(packet, candidates).edge_logits-evaluation).abs().max())
        if reload_delta != 0:
            raise RuntimeError("Connectivity disk reload mismatch")
        rows.append({"mode": mode, "config": asdict(model.config), "nodes": len(packet.positions),
                     "candidate_edges": len(candidates.edges), "input_sha256": hashes,
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "parameter_gradients_finite": gradients_finite, "train_eval_max_logit_delta": delta,
                     "reload_max_logit_delta": reload_delta, "optimizer_updates": 0,
                     "elapsed_seconds_including_reload": time.perf_counter()-start})
        print(f"SYNTHETIC {mode}: N={len(packet.positions)} E={len(candidates.edges)} "
              f"parameters={rows[-1]['parameters']} gradients/reload passed; zero updates", flush=True)
    return rows


def run(args):
    started = time.perf_counter()
    config = load(args.config)
    if config["schema"] != "pointneuron2-connectivity-execution-check-v1":
        raise ValueError("Unexpected execution-check recipe")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    torch.set_num_threads(config["torch_threads"])
    recipe_path = REPO_ROOT/config["oracle_recipe"]
    oracle = load(recipe_path)
    previous_path = REPO_ROOT/config["oracle_summary"]
    review = load(REPO_ROOT/config["oracle_review"])
    if digest(previous_path) != review["raw_summary_sha256"]:
        raise ValueError("Visible oracle summary differs from reviewed evidence")
    previous = load(previous_path)
    if digest(recipe_path) != previous["config_sha256"] or not previous["gate_passed"]:
        raise ValueError("Representation recipe/result mismatch")
    for name, sha in previous["protected_sha256"].items():
        if digest(REPO_ROOT/name) != sha:
            raise ValueError(f"Protected upstream source changed: {name}")
    audit_path = REPO_ROOT/oracle["target_audit"]
    if digest(audit_path) != oracle["target_audit_sha256"]:
        raise ValueError("Target-audit hash mismatch")
    audit = load(audit_path)
    partition_path = REPO_ROOT/oracle["partition"]
    if digest(partition_path) != audit["partition_sha256"]:
        raise ValueError("Partition hash mismatch")
    partition = load(partition_path)
    train = {r["sample_index"]: r["sample_id"] for r in partition["partitions"]["train"]}
    for role in ("val", "test"):
        if any(r["sample_index"] in train or r["sample_id"] in train.values() for r in partition["partitions"][role]):
            raise ValueError("Neuron partitions overlap")
    sources = {r["sample_index"]: r for r in audit["sources"]}
    records = {(r["sample_index"], r["patch_index"]): r for r in audit["records"]}
    paths = [Path(__file__).resolve(), args.config.resolve(), REPO_ROOT/"src/pointneuron/graph/proposal_graph.py",
             REPO_ROOT/"src/pointneuron/graph/connectivity_decoder.py", REPO_ROOT/"src/pointneuron/data/connectivity_targets.py",
             REPO_ROOT/"src/pointneuron/models/gatv2_connectivity.py"]
    code_hashes = {str(p.relative_to(REPO_ROOT)): digest(p) for p in paths}
    args.output_dir.mkdir(parents=True)
    for path in paths:
        destination = args.output_dir/"source_snapshot"/path.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    synthetic = check_synthetic(config, args.output_dir)
    rows, source_hashes = [], {}
    for case in oracle["cases"]:
        index = case["sample_index"]
        if index not in train:
            raise ValueError("Oracle case outside training partition")
        path = (args.data_root/sources[index]["path"]).resolve()
        if not path.is_relative_to(args.data_root.resolve()) or path.parent != (args.data_root/train[index]).resolve():
            raise ValueError("Oracle source outside its training-neuron directory")
        if digest(path) != sources[index]["sha256"]:
            raise ValueError("Source SWC hash mismatch")
        source_hashes[str(path)] = digest(path)
        tree = parse_swc(path)
        boxes = None if case["patch_indices"] is None else [
            (np.asarray(records[(index, patch)]["patch_center"]), oracle["patch_radius"]) for patch in case["patch_indices"]]
        target = target_graph(tree, boxes, oracle["step_voxels"])
        fine_reference = clipped_reference_graph(tree, boxes)
        xyz = torch.tensor(target.xyz, dtype=torch.float64)
        radii = torch.tensor(target.radii, dtype=torch.float64)[:, None].clamp_min(1.)
        logits = torch.tensor([[0., 12.]]*len(xyz), dtype=torch.float64)
        center = xyz.mean(0)
        packet = ProposalGraphInput(xyz, radii, logits, (xyz-center)/48., torch.zeros(len(xyz), 128, dtype=xyz.dtype),
                                    torch.arange(len(xyz)), {"source": "TRAINING_REFERENCE_ORACLE_NOT_MODEL_OUTPUT"})
        candidates = build_candidate_graph(xyz, packet.node_ids, neighbors=config["candidate_neighbors"],
                                           radius=config["candidate_radius_voxels"])
        identity = list(range(len(xyz)))
        labels = assign_edge_labels(packet, candidates, target.xyz, target.edges,
                                    full_reference_components(tree, target.identities), (identity, identity),
                                    maximum_match_error=config["match_tolerance_voxels"])
        required = {tuple(sorted(pair)) for pair in target.edges}
        # Ideal forest scores are not negative training labels for unknown edges.
        edge_logits = torch.tensor([config["oracle_edge_logits"]["reference_edge"] if tuple(sorted(pair)) in required
                                    else config["oracle_edge_logits"]["other_candidate"] for pair in candidates.edges.tolist()], dtype=xyz.dtype)
        decoded = decode_forest(packet, candidates, edge_logits, node_cutoff=config["node_cutoff"], edge_cutoff=config["edge_cutoff"])
        restored_edges = {tuple(sorted(edge)) for edge in decoded.selected_edges}
        if restored_edges != required or labels.statistics["attainable_recall_all_target_edges"] != 1.:
            raise RuntimeError("New decoder/candidates lose a known oracle edge")
        actual = ReferenceGraph(target.identities, target.xyz, target.radii,
                                [(parent, child) for child, parent in enumerate(decoded.parent) if parent >= 0])
        topology_check = topology_comparison(fine_reference, actual)
        if not topology_check["passed"]:
            raise RuntimeError("Decoded oracle branch incidence changed")
        serialization = export_decoded_swc(packet, decoded, args.output_dir/f"{case['name']}.swc",
                                            description="REFERENCE-DEFINED NODE/EDGE SCORES ONLY; NOT LEARNED GATv2.")
        rows.append({"name": case["name"], "sample_index": index, "source_sha256": digest(path),
                     "node_source": "reference oracle", "edge_score_source": "reference oracle, not GATv2",
                     "target_nodes": len(xyz), "target_edges": len(required), "all_known_edges_preserved": restored_edges == required,
                     "edge_supervision": labels.statistics, "decoder": decoded.statistics,
                     "topology": topology_check, "serialization": serialization})
        print(f"ORACLE {case['name']}: nodes={len(xyz)} edges={len(required)} "
              f"components={decoded.statistics['components']} topology/export passed", flush=True)
    for name, sha in {**previous["protected_sha256"], **code_hashes}.items():
        if digest(REPO_ROOT/name) != sha:
            raise RuntimeError("Protected source changed during checks")
    if any(digest(path) != sha for path, sha in source_hashes.items()):
        raise RuntimeError("Source SWC changed during checks")
    report = {"schema": config["schema"], "status": "completed", "implementation_checks_passed": True,
              "scope": "CPU synthetic GATv2/pairwise checks and four training-reference oracle decodes",
              "optimizer_updates": 0, "upstream_model_forwards": 0, "gpu_workloads": 0,
              "config": config, "code_sha256": code_hashes, "protected_upstream_sha256": previous["protected_sha256"],
              "source_sha256": source_hashes, "source_preservation_passed": True,
              "torch_version": torch.__version__, "synthetic": synthetic, "oracle_rows": rows,
              "elapsed_seconds_excluding_imports": time.perf_counter()-started,
              "limitations": ["No GATv2/pairwise fitting or learned reconstruction evaluation.",
                              "Real-case nodes, matches and edge scores are reference-defined; GATv2 is not run on those cases.",
                              "Full-reference forest is a decoder stress test, not a 512-query crop.",
                              "Unknown-edge scores in oracle decoding do not become negative training labels.",
                              "Upstream normalization/confidence failures and automatic whole-neuron crop matching remain unresolved.",
                              "CPU formula/reload checks do not establish official library/kernel parity or GPU runtime."]}
    (args.output_dir/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"PASS: implementation checks only; zero optimizer updates; results {args.output_dir/'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT/"configs/pointneuron2/connectivity_check.json")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent/"data/gold166")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
