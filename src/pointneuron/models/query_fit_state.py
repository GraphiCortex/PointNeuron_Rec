"""Resumable state for the bounded branched query experiment, not legacy fits."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch

from pointneuron.data.curve_targets import TARGET_SCHEMA
from pointneuron.models.pointnext_query_cropnorm import CHECKPOINT_SCHEMA, POLICY, load_cropnorm_checkpoint

FIT_SCHEMA = "pointneuron2-branched-query-fit-v1"


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"kind": state[0], "keys": state[1].tolist(), "position": state[2],
                      "has_gauss": state[3], "cached_gaussian": state[4]},
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["kind"], np.array(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_initialized() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the recorded CUDA RNG devices")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def fit_checkpoint(model, optimizer, *, step, recipe, provenance, backend):
    if (recipe.get("schema") != FIT_SCHEMA or recipe.get("normalization") != POLICY
            or not isinstance(step, int) or isinstance(step, bool) or not 0 <= step <= recipe["updates"]):
        raise ValueError("Invalid bounded-fit recipe/step")
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("The bounded fit uses AdamW without a scheduler")
    if sum(len(g["params"]) for g in optimizer.param_groups) != len(list(model.parameters())):
        raise ValueError("Checkpoint optimizer must cover all model parameters")
    inference = {"schema": CHECKPOINT_SCHEMA, "target_schema": TARGET_SCHEMA, "seed": recipe["seed"],
                 "step": step, "recipe": copy.deepcopy(recipe), "model_spec": model.model_spec(),
                 "model": cpu_snapshot(model.state_dict()), "provenance": copy.deepcopy(provenance)}
    return {"schema": FIT_SCHEMA, "step": step, "next_crop_index": step % len(recipe["training_crops"]),
            "recipe": copy.deepcopy(recipe), "provenance": copy.deepcopy(provenance), "backend": copy.deepcopy(backend),
            "inference_checkpoint": inference, "optimizer": cpu_snapshot(optimizer.state_dict()),
            "optimizer_kind": "AdamW", "scheduler": None, "rng": capture_rng(), "resumable": step < recipe["updates"]}


def restore_fit(checkpoint, *, recipe, provenance, backend, device):
    """Validate identity, restore all state and RNG; never expand the update cap."""
    if (checkpoint.get("schema") != FIT_SCHEMA or not checkpoint.get("resumable")
            or checkpoint.get("recipe") != recipe or checkpoint.get("provenance") != provenance
            or checkpoint.get("backend") != backend or checkpoint.get("optimizer_kind") != "AdamW"
            or checkpoint.get("scheduler") is not None):
        raise ValueError("Resume recipe, cache/source identity or backend differs")
    step = checkpoint.get("step")
    if (not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < recipe["updates"]
            or checkpoint.get("next_crop_index") != step % len(recipe["training_crops"])):
        raise ValueError("Invalid/completed resume step or crop cursor")
    inference = checkpoint["inference_checkpoint"]
    if inference["step"] != step or inference["recipe"] != recipe or inference["provenance"] != provenance:
        raise ValueError("Nested inference state differs from fit state")
    model = load_cropnorm_checkpoint(inference, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng"])
    return model, optimizer, step
