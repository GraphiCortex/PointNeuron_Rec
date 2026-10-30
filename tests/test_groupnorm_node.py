"""GN policy, fresh-state preservation and control failures; no dataset/GPU fit."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Use retained Torch runtime for GN component tests")
from pn2.models.groupnorm_encoder import replace_fresh_encoder_batchnorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sampler_node_common as common
import groupnorm_node_experiment as gn


def fresh_encoder(channels=32):
    return torch.nn.Sequential(*[torch.nn.BatchNorm1d(channels) for _ in range(16)])


def test_replacement_preserves_all_parameters_and_removes_all_bn_state():
    encoder = fresh_encoder()
    before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    rows = replace_fresh_encoder_batchnorm(encoder)
    assert len(rows) == 16
    assert all(isinstance(m, torch.nn.GroupNorm) and m.num_groups == 8 for m in encoder)
    assert all(p.equal(before[name]) for name, p in encoder.named_parameters())
    assert list(encoder.buffers()) == []
    assert all(r["channels_per_group"] == 4 and r["eps"] == 1e-5 for r in rows)


def test_replacement_rejects_a_trained_normalizer_before_mutating_any_layer():
    encoder = fresh_encoder()
    encoder[-1](torch.randn(1, 32, 4))
    with pytest.raises(ValueError, match="fresh"):
        replace_fresh_encoder_batchnorm(encoder)
    assert all(isinstance(m, torch.nn.BatchNorm1d) for m in encoder)


@pytest.mark.parametrize("encoder", [torch.nn.Sequential(torch.nn.BatchNorm1d(32)), fresh_encoder(30)])
def test_wrong_count_or_width_cannot_silently_change_policy(encoder):
    with pytest.raises(ValueError):
        replace_fresh_encoder_batchnorm(encoder)


@pytest.mark.parametrize("shape", [(1, 32, 1), (1, 32, 17), (1, 32, 9, 7)])
def test_groupnorm_train_eval_is_exact_and_independent_of_other_batch_samples(shape):
    layer = torch.nn.GroupNorm(8, 32)
    first = torch.randn(shape)
    second = torch.randn(shape)*11+7
    layer.train(); training = layer(first)
    layer.eval(); inference = layer(first)
    assert torch.equal(training, inference)
    together = layer(torch.cat((first, second), 0))[:1]
    assert torch.allclose(inference, together, atol=1e-6, rtol=1e-6)
    inference.square().mean().backward()
    assert torch.isfinite(layer.weight.grad).all()


def test_consistency_checker_detects_a_single_changed_output():
    fields = ("center_proposals", "local_centers", "objectness_logits", "radius", "decoder_features")
    a = SimpleNamespace(**{k: torch.ones(1, 2, 3) for k in fields})
    b = SimpleNamespace(**{k: value.clone() for k, value in vars(a).items()})
    assert gn.prediction_consistency(a, b)["passed"]
    b.objectness_logits[0, 0, 0] += 1e-4
    assert not gn.prediction_consistency(a, b)["passed"]


def test_new_checkpoint_metadata_binds_variant_code_through_recipe(tmp_path):
    model = torch.nn.Sequential(torch.nn.Conv1d(32, 32, 1), torch.nn.GroupNorm(8, 32))
    recipe = {"normalization_control": {"variant_source_sha256": {"fixture.py": "expected_hash"}},
              "optimizer": dict(learning_rate=.0001, weight_decay=0., betas=[.9, .999], eps=1e-8)}
    optimizer = common.make_optimizer(model, recipe)
    value = torch.randn(1, 32, 8)
    loss = model(value).square().mean(); loss.backward(); optimizer.step()
    expected = gn.prefix_identity(model, optimizer)
    path = tmp_path / "gn.pt"
    common.save_checkpoint(path, model, optimizer, 1, recipe, "pn2_sato")
    restored = copy.deepcopy(model)
    restored_optimizer = common.make_optimizer(restored, recipe)
    assert common.restore_checkpoint(path, restored, restored_optimizer, recipe, "pn2_sato") == 1
    assert gn.prefix_identity(restored, restored_optimizer) == expected
    changed = copy.deepcopy(recipe)
    changed["normalization_control"]["variant_source_sha256"]["fixture.py"] = "different_hash"
    with pytest.raises(ValueError):
        common.restore_checkpoint(path, restored, restored_optimizer, changed, "pn2_sato")


def test_evaluation_consistency_is_separate_from_node_learnability(monkeypatch, tmp_path):
    arrays = {}
    for mode in ("ordinary_inference", "current_crop_diagnostic"):
        arrays[f"box0_{mode}_centers"] = np.zeros((512, 3), dtype=np.float32)
        arrays[f"box0_{mode}_scores"] = np.zeros(512, dtype=np.float32)
        arrays[f"box0_{mode}_logits"] = np.zeros((512, 2), dtype=np.float32)
    np.savez(tmp_path / "predictions.npz", **arrays)
    monkeypatch.setattr(common, "evaluate", lambda *a: {"prediction_packet": "predictions.npz", "learnability_gate": False})
    records = [{"metadata": {"alias": "fixture", "tile_id": 0}}]
    result = gn.evaluate(None, records, {}, tmp_path, 0)
    assert result["normalization_consistency"]["passed"]
    assert result["learnability_gate"] is False


def test_mode_discrepancy_stops_instead_of_promoting_diagnostic(monkeypatch, tmp_path):
    arrays = {}
    for mode in ("ordinary_inference", "current_crop_diagnostic"):
        arrays[f"box0_{mode}_centers"] = np.zeros((512, 3), dtype=np.float32)
        arrays[f"box0_{mode}_scores"] = np.zeros(512, dtype=np.float32)
        arrays[f"box0_{mode}_logits"] = np.zeros((512, 2), dtype=np.float32)
    arrays["box0_current_crop_diagnostic_centers"][0, 0] = 1
    np.savez(tmp_path / "predictions.npz", **arrays)
    monkeypatch.setattr(common, "evaluate", lambda *a: {"prediction_packet": "predictions.npz", "learnability_gate": True})
    with pytest.raises(ValueError, match="consistency"):
        gn.evaluate(None, [{"metadata": {"alias": "fixture", "tile_id": 0}}], {}, tmp_path, 0)
    assert not json.loads((tmp_path / "normalization_consistency_step000000.json").read_text())["passed"]


def test_bn_timing_cannot_be_reused_as_gn_timing(monkeypatch, tmp_path):
    report = {"stage": "timing", "status": "completed", "recipe_fingerprint": "old_bn"}
    (tmp_path / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(gn, "identity", lambda r: {"recipe_fingerprint": "new_gn"})
    with pytest.raises(ValueError, match="GN timing"):
        gn.require_timing({}, tmp_path)


def test_missing_worker_report_is_an_execution_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(gn, "require_cpu", lambda *a: None)
    monkeypatch.setattr(common, "load_tasks", lambda *a: None)
    monkeypatch.setattr(gn, "identity", lambda *a: {"fixture": True})
    monkeypatch.setattr(gn, "launch_worker", lambda *a: (0, "completed"))
    args = SimpleNamespace(action="timing", output=str(tmp_path / "run"), config="fixture.json", timing_run=None, cpu_preflight=None)
    assert gn.controller({"timing_worker_timeout_seconds": 1}, args) == 1
    report = json.loads((Path(args.output) / "report.json").read_text())
    assert report["status"] == "failed_incomplete_control"


def test_comparison_pairs_boxes_by_identity_and_rejects_changed_targets(monkeypatch):
    def box(alias, count):
        return {"alias": alias, "tile_id": 1, "modes": {"ordinary_inference": {"target_count": count}}}
    baseline = {"records": [{"method": "pn2_sato", "learnability_gate": False,
                             "evaluations": [{"records": [box("a", 2), box("b", 3)]}]}]}
    monkeypatch.setattr(common, "checked_json", lambda *a: baseline)
    recipe = {"normalization_control": {"baseline_report": "fixture", "baseline_report_sha256": "fixture"}}
    terminal = {"records": [box("b", 3), box("a", 2)], "learnability_gate": False}
    result = gn.baseline_comparison(recipe, terminal)
    assert [r["alias"] for r in result["records"]] == ["b", "a"]
    assert not result["baseline_learnability_gate"] and not result["groupnorm_learnability_gate"]
    terminal["records"][0]["modes"]["ordinary_inference"]["target_count"] = 4
    with pytest.raises(ValueError, match="denominator"):
        gn.baseline_comparison(recipe, terminal)
