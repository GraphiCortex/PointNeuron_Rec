"""Gate gradients and tiny complete paired experiment; no research GPU training."""
from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from pointneuron.models.centerline_unet import CenterlineUNet, field_loss
from pointneuron.models.centerline_attention import AttentionCenterlineUNet, SkipAttentionGate
from scripts.pointneuron2 import field_attention_experiment as experiment
from scripts.pointneuron2 import train_field_pilot as training
from scripts.pointneuron2 import compare_field_pilot as sampling
from test_field_pilot import native_fixture, CONFIG, SAMPLING_CONFIG

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT/"configs/pointneuron2/field_attention.json"


def test_gate_initialization_conditioning_and_gradient_flow():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    gate = SkipAttentionGate(8, 16, 4)
    skip = torch.randn(1, 8, 7, 9, 11, requires_grad=True)
    context = torch.randn(1, 16, 3, 4, 5, requires_grad=True)
    out = gate(skip, context)
    torch.testing.assert_close(out, skip*.99)
    out.square().mean().backward()
    assert gate.score.weight.grad.abs().sum() > 0 and gate.score.bias.grad.abs().sum() > 0
    with torch.no_grad():
        gate.score.weight.fill_(.1)
    gate.zero_grad()
    gate(skip, context).square().mean().backward()
    assert gate.skip_projection.weight.grad.abs().sum() > 0
    assert gate.context_projection.weight.grad.abs().sum() > 0 and context.grad.abs().sum() > 0


def test_shared_backbone_odd_shape_and_strict_reload():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    baseline = CenterlineUNet(8)
    gated = AttentionCenterlineUNet(8)
    gated.load_shared_backbone(baseline.state_dict())
    assert sum(p.numel() for p in baseline.parameters()) == 88673
    assert sum(p.numel() for p in gated.parameters()) == 89167
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, gated.state_dict()[key])
    x = torch.randn(1, 1, 9, 11, 13)
    y = gated(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    target = torch.zeros_like(y, dtype=torch.bool)
    target[:, :, 4, 4, :] = True
    field_loss(y, target, torch.ones_like(target), negative_aggregation="topk").backward()
    assert gated.gate_a.score.weight.grad.abs().sum() > 0
    assert gated.gate_b.score.weight.grad.abs().sum() > 0
    reloaded = AttentionCenterlineUNet(8).eval()
    reloaded.load_state_dict(gated.state_dict(), strict=True)
    gated.eval()
    with torch.no_grad():
        torch.testing.assert_close(gated(x), reloaded(x), rtol=0, atol=0)
    with pytest.raises(ValueError, match="exact ungated backbone"):
        gated.load_shared_backbone({})


def test_explicit_architecture_factory_rejects_unknown_specs():
    with pytest.raises(ValueError, match="Unsupported"):
        experiment.inference_factory(8, {"kind": "other"})
    assert isinstance(experiment.inference_factory(8, {"kind": "attention_centerline_unet_v1", "initial_gate_coefficient": .99}), AttentionCenterlineUNet)


def test_tiny_two_arm_training_sampling_and_saved_baseline_preservation(native_fixture):
    original_args, data, _ = native_fixture
    training.run(original_args)
    original_args.timing = original_args.output_dir/"timing.json"
    original_args.output_dir = original_args.output_dir.parent/"saved_model"
    original_args.preflight = False
    old = training.run(original_args)
    args = Namespace(action="preflight", experiment_dir=original_args.output_dir.parent/"attention_trial",
                     manifest=original_args.manifest, partition=original_args.partition, config=CONFIG,
                     attention_config=RECIPE, sampling_config=SAMPLING_CONFIG,
                     baseline_models=original_args.output_dir, data_root=data, device="cpu")
    timed = experiment.run(args)
    assert timed["status"] == "completed" and timed["throwaway_optimizer_updates"] == 8
    assert not list(args.experiment_dir.rglob("*.pt"))
    args.action = "train"
    trained = experiment.run(args)
    assert trained["status"] == "completed" and all(trained["checks"].values())
    assert trained["optimizer_updates"] == 60 and trained["saved_baseline_and_cache_unchanged"]
    assert "fresh_baseline_minus_saved_baseline" in trained["fresh_baseline_minus_saved_reference"]["val"]
    for arm in ("baseline", "attention"):
        assert trained["arms"][arm]["crop_index_order"] == old["crop_index_order"]
    # The ordinary runner must not silently load a gated checkpoint as a baseline.
    generic = Namespace(manifest=args.manifest, partition=args.partition, config=CONFIG,
                        sampling_config=SAMPLING_CONFIG, models=args.experiment_dir/"attention/model", device="cpu")
    with pytest.raises(ValueError, match="explicit attention"):
        sampling.load_contract(generic)
    args.action = "compare"
    compared = experiment.run(args)
    assert compared["status"] == "completed" and compared["optimizer_updates"] == 0
    assert all(compared["checks"].values()) and compared["saved_baseline_and_cache_unchanged"]
    matched = [r for r in compared["sampling_metrics"]["attention_minus_baseline_pairs"] if r["group"] == "matched"]
    assert matched and all(r["equal_counts_between_models"] for r in matched)
    with pytest.raises(FileExistsError, match="Preserve"):
        experiment.run(args)


def test_between_model_sampling_reports_flag_availability_differences():
    identity = {"sample_index": 1, "patch_index": 0, "group": "expanded", "nominal_budget": 2048,
                "sampling_seed": 0, "arm": "expanded_random"}
    a = {**identity, "actual_unique_count": 1000, "reference_coverage_2": .7, "reference_proximity_6": .9,
         "worst_chain_coverage": {"2": .5}}
    b = {**a, "actual_unique_count": 800, "reference_coverage_2": .8}
    result = experiment.compare_saved_sampling({"baseline": {"rows": [a], "summaries": [], "config": {}},
                                                "attention": {"rows": [b], "summaries": [], "config": {}}})
    assert not result["attention_minus_baseline_pairs"][0]["equal_counts_between_models"]
    a["group"] = b["group"] = "matched"
    with pytest.raises(AssertionError, match="Matched counts"):
        experiment.compare_saved_sampling({"baseline": {"rows": [a], "summaries": [], "config": {}},
                                            "attention": {"rows": [b], "summaries": [], "config": {}}})
