"""Shared data/model/state handling for the fixed sampler node control."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_sampler_comparison import digest, checked_json, write
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.cloud_support import polyline_samples
from pn2.supervision.curve_nodes import rectangular_curve_nodes


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    expected = {"schema_version": 1, "methods": ["pn1_range_random", "pn2_sato"],
        "box_order": [["fruitfly32", 27], ["fruitfly32", 20], ["frog22", 2], ["frog22", 5]],
        "seed": 0, "query_count": 512, "input_slots": 2048, "batch_size": 1,
        "dtype": "float32", "normalization": "ordinary_batchnorm_saved_running_statistics_at_inference",
        "attention_backend": "math_sdpa_no_mha_fastpath", "deterministic_algorithms": True,
        "tf32": False, "dropout": 0., "augmentation": "none", "updates_per_arm": 2000,
        "evaluation_steps": [0, 8, 500, 1000, 2000], "timing_updates_per_arm": 8,
        "cutoff": .5, "gate_tolerance_voxel": 2., "gate_precision": .95, "gate_recall": .95,
        "cpu_threads": 1}
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise ValueError("Recipe differs from the fixed bounded comparison")
    if recipe["loss"] != dict(classification_weight=2., center_weight=3., center_beta_voxel=4., match_scale_voxel=6.):
        raise ValueError("Loss change requires a new prospective recipe")
    if recipe["optimizer"] != dict(kind="AdamW", learning_rate=.0001, weight_decay=0., betas=[.9, .999], eps=1e-8):
        raise ValueError("Optimizer change requires a new prospective recipe")
    if recipe["gradient_clip"] != 10.:
        raise ValueError("Gradient clipping differs")
    if digest(ROOT / "src/pn2/supervision/curve_nodes.py") != recipe["target_source_sha256"]:
        raise ValueError("Audited target source changed")
    return recipe


def code_identity():
    names = ["scripts/sampler_node_common.py", "scripts/sampler_node_experiment.py",
             "src/pn2/models/node_objective.py", "src/pn2/evaluation/node_proposals.py",
             "src/pn2/supervision/curve_nodes.py", "scripts/pointnext_compact_interpolation.py",
             "scripts/probe_pointnext_capacity.py", "scripts/prepare_sampler_comparison.py",
             "src/pn2/evaluation/swc.py", "src/pn2/evaluation/cloud_support.py"]
    return {name: digest(ROOT / name) for name in names}


def load_tasks(recipe):
    manifest = checked_json(recipe["inputs"], recipe["inputs_sha256"])
    audit = checked_json(recipe["target_audit"], recipe["target_audit_sha256"])
    inventory = checked_json(recipe["training_inventory"], recipe["training_inventory_sha256"])
    folder = (ROOT / recipe["inputs"]).parent
    target_folder = (ROOT / recipe["target_audit"]).parent
    groups, references = {}, {}
    for method in recipe["methods"]:
        groups[method] = []
        for alias, tile in recipe["box_order"]:
            rows = [r for r in manifest["records"] if r["method"] == method and r["alias"] == alias and r["tile_id"] == tile]
            if len(rows) != 1:
                raise ValueError("Missing/duplicate declared input row")
            row = rows[0]
            target = next(t for t in audit["target_records"] if t["case_id"] == row["case_id"] and t["tile_id"] == tile)
            if row["partition"] != "training" or row["slots"] != recipe["input_slots"]:
                raise ValueError("Unexpected partition/slots")
            packet_path, target_path = folder / row["packet"], target_folder / target["packet"]
            if digest(packet_path) != row["packet_sha256"] or digest(target_path) != target["packet_sha256"]:
                raise ValueError("Frozen input/target packet changed")
            with np.load(packet_path, allow_pickle=False) as data:
                packet = {k: data[k].copy() for k in data.files}
            with np.load(target_path, allow_pickle=False) as data:
                target_xyz = data["xyz"].copy()
            source = next(c for c in inventory["cases"] if c["case_id"] == row["case_id"])
            if source["partition"] != "training":
                raise ValueError("Only declared training references allowed")
            reference_path = (ROOT / recipe["training_inventory"]).parent / source["reference_swc"]
            if digest(reference_path) != target["reference_sha256"]:
                raise ValueError("Reference changed")
            if row["case_id"] not in references:
                references[row["case_id"]] = parse_swc(reference_path, type_policy="opaque_integer")
            rebuilt = rectangular_curve_nodes(references[row["case_id"]], row["box"]["start_xyz"], row["box"]["stop_xyz"])
            if not np.array_equal(rebuilt.xyz, target_xyz) or len(target_xyz) != target["target_count"]:
                raise ValueError("Target coordinates/counts differ from the completed audit")
            if len(packet["flat_indices"]) != row["slots"] or len(np.unique(packet["flat_indices"])) != row["distinct_points"]:
                raise ValueError("Frozen duplicate counts changed")
            groups[method].append({"metadata": row, "packet": packet, "target_xyz": target_xyz,
                "target_roles": rebuilt.roles, "curve_samples": [polyline_samples(run, .5) for run in rebuilt.reference_runs]})
    return groups


def runtime(recipe, device):
    # Must precede CUDA context creation for deterministic CUDA matmul.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.set_num_threads(recipe["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.random.default_generator.manual_seed(recipe["seed"])
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no automatic CPU fallback")
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(recipe["seed"])
    return torch


@contextmanager
def attention_math():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        yield


def state_digest(state):
    import torch
    h = hashlib.sha256()
    for key, value in state.items():
        value = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def nested_digest(value):
    import torch
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(b"tensor"); h.update(state_digest({"value": item}).encode())
        elif isinstance(item, np.ndarray):
            h.update(str((item.dtype, item.shape)).encode()); h.update(item.tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=str):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for entry in item:
                visit(entry)
        else:
            h.update(json.dumps(item, allow_nan=False).encode()); h.update(b";")
    visit(value)
    return h.hexdigest()


def audited_state_digest(state):
    # Preserve the preceding CPU audit's state-hash convention for exact identity.
    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_model(recipe, device):
    import probe_pointnext_capacity as probe
    encoder_recipe, source = probe.load(ROOT / recipe["encoder_config"])
    if encoder_recipe["interpolation_variant"] != "compact_indices_v1":
        raise ValueError("Expected compact interpolation")
    if digest(ROOT / recipe["head_source"]) != recipe["head_source_sha256"]:
        raise ValueError("Pinned head source changed")
    torch, Encoder, _ = probe.imports(source, "compact_indices_v1")
    from pointneuron.models.query_proposal import QueryProposalHead, QueryDecoderConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(recipe["seed"])
                self.encoder = Encoder()
                torch.random.default_generator.manual_seed(recipe["seed"]+101)
                self.head = QueryProposalHead(QueryDecoderConfig(in_channels=32, query_count=512, dropout=0.))

        def forward(self, record):
            with attention_math():
                features = self.encoder(record["local_xyz"], record["attributes"])
                return self.head(record["points"], features, record["box_center"], 64.)

    model = Model()
    if audited_state_digest(model.state_dict()) != recipe["initial_state_sha256"]:
        raise ValueError("Fresh model initialization differs from the completed CPU audit")
    return model.to(device)


def tensors(record, torch, device):
    packet, meta = record["packet"], record["metadata"]
    tensor = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32).copy()).to(device)
    return {**record, "local_xyz": tensor(packet["local_xyz"])[None],
        "attributes": tensor(packet["attributes"])[None],
        "points": tensor(np.column_stack((packet["native_xyz"], packet["raw_intensity"])))[None],
        "box_center": tensor(meta["center_xyz"])[None], "target": tensor(record["target_xyz"])}


def make_optimizer(model, recipe):
    import torch
    settings = recipe["optimizer"]
    return torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]), eps=settings["eps"],
        foreach=False, fused=False)


def update(model, optimizer, record, recipe):
    import torch
    from pn2.models.node_objective import node_set_loss
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(record)
    loss = node_set_loss(output, record["target"], **recipe["loss"])
    if not loss.supervised:
        raise ValueError("Declared four-box control cannot silently skip an unknown crop")
    loss.total.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError("Missing/nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite updated parameters")
    return {"loss": float(loss.total.detach()), "classification": float(loss.classification),
            "center": float(loss.center), "gradient_norm_before_clip": float(norm),
            "matched_targets": len(loss.prediction_indices)}


def current_crop_diagnostic(model, record):
    """Read current-crop behavior without changing buffers or the scored policy."""
    import torch
    mode = model.training
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    try:
        model.train()
        with torch.no_grad():
            return model(record)
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        model.train(mode)


def evaluate(model, records, recipe, output, step):
    import torch
    from pn2.evaluation.node_proposals import proposal_metrics
    from pn2.models.node_objective import node_set_loss
    model.eval()
    rows, arrays = [], {}
    before = state_digest(model.state_dict())
    before_rng = nested_digest(rng_state())
    for i, record in enumerate(records):
        with torch.no_grad():
            ordinary = model(record)
        diagnostic = current_crop_diagnostic(model, record)
        row = {"alias": record["metadata"]["alias"], "tile_id": record["metadata"]["tile_id"],
               "distinct_points": record["metadata"]["distinct_points"], "modes": {}}
        for name, prediction in (("ordinary_inference", ordinary), ("current_crop_diagnostic", diagnostic)):
            with torch.no_grad():
                loss = node_set_loss(prediction, record["target"], **recipe["loss"])
                centers = prediction.center_proposals[0].cpu().numpy()
                scores = prediction.objectness_logits[0].softmax(-1)[:, 1].cpu().numpy()
                logits = prediction.objectness_logits[0].cpu().numpy()
            metrics = proposal_metrics(centers, scores, record["target_xyz"], record["target_roles"],
                                       record["curve_samples"], cutoff=recipe["cutoff"])
            positive = np.zeros(len(scores), dtype=bool)
            positive[list(loss.prediction_indices)] = True
            metrics["assigned_positive_score_range"] = [float(scores[positive].min()), float(scores[positive].max())]
            metrics["strongest_assigned_unused_score"] = float(scores[~positive].max()) if (~positive).any() else None
            row["modes"][name] = metrics
            for key, array in (("centers", centers), ("scores", scores), ("logits", logits),
                               ("assignment_predictions", np.asarray(loss.prediction_indices)),
                               ("assignment_targets", np.asarray(loss.target_indices))):
                arrays[f"box{i}_{name}_{key}"] = array
        rows.append(row)
        main = row["modes"]["ordinary_inference"]["tolerances"]["2.0"]["accepted_nodes"]
        print(f"EVAL step={step} {row['alias']}/{row['tile_id']} matched={main['matched']}/{main['reference']} extras={main['extras']} P/R={main['precision']:.3f}/{main['recall']:.3f} gate={row['modes']['ordinary_inference']['learnability_gate']}", flush=True)
    if state_digest(model.state_dict()) != before:
        raise ValueError("Evaluation/normalization diagnostic changed model parameters or buffers")
    if nested_digest(rng_state()) != before_rng:
        raise ValueError("Evaluation/normalization diagnostic changed RNG state")
    predictions = output / f"predictions_step{step:06d}.npz"
    np.savez_compressed(predictions, **arrays)
    result = {"step": step, "records": rows,
        "learnability_gate": all(r["modes"]["ordinary_inference"]["learnability_gate"] for r in rows),
        "diagnostic_gate_not_used_for_decision": all(r["modes"]["current_crop_diagnostic"]["learnability_gate"] for r in rows),
        "model_state_sha256": before, "prediction_packet": predictions.name,
        "prediction_packet_sha256": digest(predictions)}
    write(output / f"evaluation_step{step:06d}.json", result)
    return result


def rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    import torch
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def checkpoint_payload(model, optimizer, step, recipe, method):
    return {"schema": "pn2-sampler-node-state-v1", "method": method, "step": step,
        "recipe": recipe, "recipe_fingerprint": fingerprint(recipe), "code_identity": code_identity(),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state(),
        "model_training": model.training}


def save_checkpoint(path, model, optimizer, step, recipe, method):
    import torch
    payload = checkpoint_payload(model, optimizer, step, recipe, method)
    # New immutable files avoid the historical Windows replace/reader lock failure.
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
        stream.flush(); os.fsync(stream.fileno())
    write(Path(path).with_suffix(".complete.json"), {"checkpoint": Path(path).name,
        "sha256": digest(path), "step": step, "recipe_fingerprint": fingerprint(recipe)})


def restore_checkpoint(path, model, optimizer, recipe, method):
    import torch
    complete = json.loads(Path(path).with_suffix(".complete.json").read_text())
    if complete["sha256"] != digest(path) or complete["recipe_fingerprint"] != fingerprint(recipe):
        raise ValueError("Incomplete, changed or incompatible checkpoint")
    # Only this run's locally written, hash-checked state files are supported.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return restore_payload(payload, model, optimizer, recipe, method)


def restore_payload(payload, model, optimizer, recipe, method):
    if (payload["schema"] != "pn2-sampler-node-state-v1" or payload["method"] != method
            or payload["recipe_fingerprint"] != fingerprint(recipe) or payload["code_identity"] != code_identity()):
        raise ValueError("Checkpoint identity differs")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    rng = payload["rng"]
    rng["torch_cpu"] = rng["torch_cpu"].cpu()
    if rng["torch_cuda"] is not None:
        rng["torch_cuda"] = [value.cpu() for value in rng["torch_cuda"]]
    restore_rng(rng)
    model.train(payload["model_training"])
    return payload["step"]
