"""Continuation and decision checks; synthetic CPU updates are discarded."""
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from pointneuron.models.pointnext_query_cropnorm import PointNextCropNormQueryModel
from pointneuron.models.query_fit_state import fit_checkpoint, restore_fit
from pointneuron.models.query_loss import query_set_loss
from scripts.pointneuron2.branched_query_experiment import advancement_gates


@pytest.fixture
def recipe():
    torch.set_num_threads(2)
    return json.loads((ROOT / "configs/pointneuron2/branched_query_fit.json").read_text())


def synthetic_update(model, optimizer, recipe):
    draws = (random.random(), float(np.random.rand()))
    points = torch.rand(1, 256, 4)
    points[:, :, :3] *= 96
    points[:, :, 3] *= 255
    points[:, :, :3] += (draws[0] + draws[1]) * .01
    nodes = torch.tensor([[[1., 30., 40., 50., 1., -1.], [2., 35., 45., 55., 2., -1.]]])
    model.train(); optimizer.zero_grad(set_to_none=True)
    out = model(points, torch.tensor([[48., 48., 48.]]), torch.tensor([255.]), 48.)
    loss = query_set_loss(out, nodes, torch.ones(1, 2, dtype=torch.bool), **recipe["query_loss"])
    loss.total.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.total.detach()), draws


def test_single_thread_disk_resume_reproduces_next_full_model_update_and_rng(tmp_path, recipe):
    # Gather/scatter gradient accumulation can vary with CPU worker scheduling.
    # A single thread isolates checkpoint correctness from that runtime effect.
    torch.set_num_threads(1)
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    model = PointNextCropNormQueryModel(recipe["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    synthetic_update(model, optimizer, recipe)
    checkpoint = fit_checkpoint(model, optimizer, step=1, recipe=recipe, provenance={"cache": "synthetic"}, backend={"device": "cpu"})
    path = tmp_path / "state.pt"
    torch.save(checkpoint, path)
    expected, expected_draws = synthetic_update(model, optimizer, recipe)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    # The in-memory snapshot must not change when the original optimizer advances.
    for key, value in loaded["inference_checkpoint"]["model"].items():
        assert torch.equal(value, checkpoint["inference_checkpoint"]["model"][key])
    for key, state in loaded["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, checkpoint["optimizer"]["state"][key][name])
    restored, resumed_optimizer, step = restore_fit(loaded, recipe=recipe, provenance={"cache": "synthetic"}, backend={"device": "cpu"}, device="cpu")
    assert step == 1 and loaded["next_crop_index"] == 1
    actual, actual_draws = synthetic_update(restored, resumed_optimizer, recipe)
    assert actual == expected and actual_draws == expected_draws
    differences = {key: float((value-restored.state_dict()[key]).abs().max()) for key, value in model.state_dict().items()
                   if not torch.equal(value, restored.state_dict()[key])}
    assert not differences, sorted(differences.items(), key=lambda pair: -pair[1])[:15]
    a, b = optimizer.state_dict(), resumed_optimizer.state_dict()
    assert a["param_groups"] == b["param_groups"]
    for key, state in a["state"].items():
        for name, value in state.items():
            assert torch.equal(value, b["state"][key][name])


def test_resume_rejects_changed_experiment_and_completed_budget(recipe):
    model = PointNextCropNormQueryModel(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    provenance, backend = {"cache": "fixed"}, {"device": "cpu"}
    checkpoint = fit_checkpoint(model, optimizer, step=0, recipe=recipe, provenance=provenance, backend=backend)
    for key, replacement in (("recipe", dict(recipe, updates=8000)), ("provenance", {"cache": "other"}), ("backend", {"device": "cuda"})):
        kwargs = {"recipe": recipe, "provenance": provenance, "backend": backend, "device": "cpu"}
        kwargs[key] = replacement
        with pytest.raises(ValueError):
            restore_fit(checkpoint, **kwargs)
    malformed = copy.deepcopy(checkpoint); malformed["next_crop_index"] = 3
    with pytest.raises(ValueError):
        restore_fit(malformed, recipe=recipe, provenance=provenance, backend=backend, device="cpu")
    final = fit_checkpoint(model, optimizer, step=4000, recipe=recipe, provenance=provenance, backend=backend)
    assert not final["resumable"]
    with pytest.raises(ValueError):
        restore_fit(final, recipe=recipe, provenance=provenance, backend=backend, device="cpu")


def good_crop(targets=16):
    return {"targets": targets, "trusted_target_fraction": 1., "visible_junctions": 2, "trusted_visible_junctions": 2,
            "accepted_assigned": targets, "accepted_unassigned": 0, "accepted_curve_support_6": 1.,
            "connectivity_supervision": {"positive": 15, "negative": 20, "attainable_recall_all_target_edges": 1.}}


def test_gate_keeps_geometry_confidence_and_empty_supervision_distinct(recipe):
    row = good_crop()
    assert advancement_gates(row, recipe)["combined_learnability_gate_passed"]
    row["accepted_assigned"] = 0; row["accepted_curve_support_6"] = 0.
    result = advancement_gates(row, recipe)
    assert result["geometry_gate_passed"] and not result["selection_gate_passed"]
    row = good_crop(); row["trusted_visible_junctions"] = 1
    result = advancement_gates(row, recipe)
    assert not result["geometry_gate_passed"] and result["selection_gate_passed"]
    row = good_crop(); row["connectivity_supervision"]["positive"] = 0
    assert not advancement_gates(row, recipe)["combined_learnability_gate_passed"]
    row = good_crop(); row["accepted_curve_support_6"] = None
    assert not advancement_gates(row, recipe)["combined_learnability_gate_passed"]


def test_gate_does_not_relax_small_crop_extras_or_dense_target_fraction(recipe):
    row = good_crop(6); row["accepted_unassigned"] = 1
    assert not advancement_gates(row, recipe)["selection_gate_passed"]
    row = good_crop(418); row["accepted_unassigned"] = 20; row["trusted_target_fraction"] = 398/418
    assert advancement_gates(row, recipe)["combined_learnability_gate_passed"]
    row["trusted_target_fraction"] = 397/418
    assert not advancement_gates(row, recipe)["geometry_gate_passed"]
    row["trusted_target_fraction"] = 1.; row["accepted_unassigned"] = 21
    assert not advancement_gates(row, recipe)["selection_gate_passed"]
