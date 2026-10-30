"""CPU-only real-box target audit and untrained encoder/head forward/backward.

Fresh weights, no optimizer, checkpoint, sampling/filter rerun or GPU access.
The target grid is an audited candidate, not a new adopted training policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.cloud_support import polyline_samples, longest_run
from pn2.evaluation.geometry import geometry_metrics
from pn2.supervision.curve_nodes import rectangular_curve_nodes, SCHEMA
from prepare_sampler_comparison import digest, checked_json, write


def target_audit(config, manifest, output):
    inventory = checked_json(config["training_inventory"], config["training_inventory_sha256"])
    evidence = checked_json(config["reference_evidence"], config["reference_evidence_sha256"])
    rows, seen, references = [], set(), {}
    for record in manifest["records"]:
        case_id, tile = record["case_id"], record["tile_id"]
        if (case_id, tile) in seen:
            continue
        seen.add((case_id, tile))
        source = next(c for c in inventory["cases"] if c["case_id"] == case_id)
        if source["partition"] != "training" or record["partition"] != "training":
            raise ValueError("This audit may read only explicitly declared training references")
        expected = next(c for c in evidence["cases"] if c["case_id"] == case_id)
        reference_path = (ROOT / config["training_inventory"]).parent / source["reference_swc"]
        if digest(reference_path) != expected["reference_sha256"]:
            raise ValueError("Frozen reference changed")
        if case_id not in references:
            references[case_id] = parse_swc(reference_path, type_policy="opaque_integer")
        morphology = references[case_id]
        box = record["box"]
        targets = rectangular_curve_nodes(morphology, box["start_xyz"], box["stop_xyz"],
                                          step=config["target_step_voxel"], capacity=config["query_count"])
        samples = [polyline_samples(run, .5) for run in targets.reference_runs]
        if not len(targets.xyz) or not samples:
            raise ValueError("A declared reference-positive task became empty")
        tree = cKDTree(targets.xyz)
        support = {}
        for tolerance in (2., 6.):
            distances = [tree.query(p)[0] for p, _ in samples]
            total = sum(float(w.sum()) for _, w in samples)
            covered = sum(float(w[d <= tolerance].sum()) for d, (_, w) in zip(distances, samples))
            support[str(tolerance)] = {"curve_to_target_node_support": covered / total,
                                       "longest_unsupported_run": max(longest_run(d > tolerance, w) for d, (_, w) in zip(distances, samples))}
        points, ref_edges, offset = [], [], 0
        for run in targets.reference_runs:
            points.extend(run)
            ref_edges.extend((offset+j, offset+j+1) for j in range(len(run)-1))
            offset += len(run)
        reference = SimpleNamespace(xyz=np.asarray(points), edges=np.asarray(ref_edges, dtype=np.int64).reshape(-1, 2))
        predicted = SimpleNamespace(xyz=targets.xyz, edges=targets.edges)
        curve = geometry_metrics(reference, predicted, tolerances=(2., 6.))
        nearest = tree.query(targets.xyz, k=2)[0][:, 1] if len(targets.xyz) > 1 else np.array([])
        roles = sorted(set().union(*targets.roles))
        path = output / f"{record['alias']}_tile{tile}_targets.npz"
        np.savez_compressed(path, xyz=targets.xyz, reference_radius=targets.radii,
                            edges=targets.edges)
        write(path.with_suffix(".json"), {"schema": SCHEMA, "identities": targets.identities,
                                         "roles": [sorted(r) for r in targets.roles],
                                         "reference_sha256": expected["reference_sha256"]})
        rows.append({"alias": record["alias"], "case_id": case_id, "tile_id": tile,
                     "reference_sha256": expected["reference_sha256"], "reference_components": morphology.component_count,
                     "box": box, "target_count": len(targets.xyz), "query_count": config["query_count"],
                     "target_edges": len(targets.edges), "positive_reference_runs": len(targets.reference_runs),
                     "role_counts": {r: sum(r in rr for rr in targets.roles) for r in roles},
                     "minimum_target_separation": float(nearest.min()) if len(nearest) else None,
                     "targets_with_another_within_0_25_voxel": int(np.count_nonzero(nearest <= .25)),
                     "targets_with_another_within_2_voxel": int(np.count_nonzero(nearest <= 2.)),
                     "reference_radius_range": [float(targets.radii.min()), float(targets.radii.max())],
                     "maximum_reference_to_target_chord_bound": targets.maximum_reference_to_chord_bound,
                     "maximum_parametric_deviation_bound": targets.maximum_parametric_deviation_bound,
                     "target_node_support": support, "target_polyline_geometry": curve,
                     "packet": path.name, "packet_sha256": digest(path)})
        print(f"TARGET {record['alias']}/{tile}: nodes={len(targets.xyz)}/{config['query_count']} near_pairs_nodes={rows[-1]['targets_with_another_within_0_25_voxel']} curve_bound={targets.maximum_reference_to_chord_bound:.4f}", flush=True)
        if digest(reference_path) != expected["reference_sha256"]:
            raise ValueError("Reference changed during audit")
    return rows


def model_check(config, manifest, output):
    import probe_pointnext_capacity as probe
    encoder_config, source = probe.load(ROOT / config["encoder_config"])
    if digest(ROOT / config["head_source"]) != config["head_source_sha256"]:
        raise ValueError("Pinned Transformer source changed")
    torch, Encoder, _ = probe.imports(source, "compact_indices_v1")
    from pointneuron.models.query_proposal import QueryProposalHead, QueryDecoderConfig
    torch.set_num_threads(config["cpu_threads"])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config["seed"])
                self.encoder = Encoder()
                torch.random.default_generator.manual_seed(config["seed"] + 101)
                self.head = QueryProposalHead(QueryDecoderConfig(in_channels=32, query_count=config["query_count"], dropout=0.))

        def forward(self, packet, center):
            local = torch.from_numpy(packet["local_xyz"])[None]
            attributes = torch.from_numpy(packet["attributes"])[None]
            xyz = torch.from_numpy(packet["native_xyz"].astype(np.float32))[None]
            raw = torch.from_numpy(packet["raw_intensity"].astype(np.float32))[None, :, None]
            points = torch.cat((xyz, raw), -1)
            features = self.encoder(local, attributes)
            return self.head(points, features, torch.tensor(center, dtype=torch.float32)[None], 64.)

    model = Model()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    h = hashlib.sha256()
    for key, value in initial.items():
        h.update(key.encode()); h.update(value.numpy().tobytes())
    rows = []
    folder = (ROOT / config["inputs"]).parent
    for record in manifest["records"]:
        packet_path = folder / record["packet"]
        if digest(packet_path) != record["packet_sha256"]:
            raise ValueError("Prepared input packet changed")
        with np.load(packet_path, allow_pickle=False) as stored:
            packet = {key: stored[key].copy() for key in stored.files}
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        model.eval()
        with torch.no_grad():
            before = model(packet, record["center_xyz"])
        model.train()
        training = model(packet, record["center_xyz"])
        objective = training.local_centers.square().mean() + training.objectness_logits.square().mean() + training.radius.square().mean()
        objective.backward()
        gradients = [p.grad for p in model.parameters()]
        if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("Missing or nonfinite model gradients")
        if not any(bool((g != 0).any()) for g in gradients):
            raise ValueError("Model gradient is identically zero")
        if any(not torch.equal(p, initial_parameters[k]) for k, p in model.named_parameters()):
            raise ValueError("Execution check changed a parameter without an optimizer")
        model.eval()
        with torch.no_grad():
            after = model(packet, record["center_xyz"])
        arrays = {}
        for name, result in (("eval_initial", before), ("train", training), ("eval_after_one_bn_update", after)):
            for key, tensor in (("centers", result.center_proposals), ("logits", result.objectness_logits), ("radius", result.radius)):
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("Nonfinite proposal output")
                arrays[f"{name}_{key}"] = tensor.detach().numpy()
        if list(after.center_proposals.shape) != [1, 512, 3] or list(after.decoder_features.shape) != [1, 512, 128]:
            raise ValueError("Proposal shape disagrees with the declared interface")
        predictions = output / f"{record['alias']}_tile{record['tile_id']}_{record['method']}_untrained.npz"
        np.savez_compressed(predictions, **arrays)
        rows.append({"alias": record["alias"], "tile_id": record["tile_id"], "method": record["method"],
                     "slots": record["slots"], "distinct_points": record["distinct_points"],
                     "parameter_count": sum(p.numel() for p in model.parameters()),
                     "all_parameter_gradients_finite": True, "parameters_unchanged": True,
                     "center_shape": list(after.center_proposals.shape), "decoder_feature_shape": list(after.decoder_features.shape),
                     "maximum_train_vs_eval_after_one_bn_center_delta_voxel": float((training.center_proposals-after.center_proposals).detach().abs().max()),
                     "maximum_train_vs_eval_after_one_bn_logit_delta": float((training.objectness_logits-after.objectness_logits).detach().abs().max()),
                     "seconds": time.perf_counter()-started,
                     "prediction_packet": predictions.name, "prediction_packet_sha256": digest(predictions)})
        print(f"CPU MODEL {record['alias']}/{record['tile_id']} {record['method']}: forward/backward finite; no optimizer; {rows[-1]['seconds']:.2f}s", flush=True)
    return {"torch": torch.__version__, "device": "cpu", "seed": config["seed"],
            "initial_state_sha256": h.hexdigest(), "encoder_source_sha256": encoder_config["source_sha256"],
            "interpolation_source_sha256": encoder_config["additional_source_sha256"],
            "head_source_sha256": config["head_source_sha256"], "records": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/sampler_interface_check.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if (config["query_count"] != 512 or config["target_step_voxel"] != 8.
            or config["cpu_threads"] != 1 or config["seed"] != 0):
        raise ValueError("Only the fixed CPU interface check is supported")
    manifest = checked_json(config["inputs"], config["inputs_sha256"])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", config)
    targets = target_audit(config, manifest, output)
    write(output / "targets_report.json", {"target_schema": SCHEMA, "records": targets})
    model = model_check(config, manifest, output)
    report = {"scope": config["scope"], "status": "execution_passed_training_recipe_unresolved",
              "script_sha256": digest(__file__), "target_source_sha256": digest(ROOT / "src/pn2/supervision/curve_nodes.py"),
              "config_sha256": digest(args.config), "target_records": targets, "model": model,
              "limitations": ["No optimizer or training; fresh parameters and untrained BN buffers",
                              "Eight-voxel unpruned targets are an audited candidate, not an adopted training recipe",
                              "No sampler superiority or learned reconstruction quality measured"]}
    write(output / "report.json", report)
    print(f"CPU interface complete; no fit: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
