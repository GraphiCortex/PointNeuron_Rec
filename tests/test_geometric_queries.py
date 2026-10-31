"""Geometry/memory contracts and fail-closed run gates; no real-data fitting."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import sampler_node_common as common
import groupnorm_node_experiment as gn
import geometric_query_experiment as experiment
from pn2.models.geometric_queries import GeometricQueryHead


@pytest.fixture(scope="module", autouse=True)
def pinned_native_source():
    # Hash-check/load dependencies exactly as the experiment does.
    recipe = gn.load_recipe(gn.CONFIG)
    common.runtime(recipe, "cpu")
    model = gn.make_model(recipe, "cpu")
    del model


def fixture_head(count=8):
    from pointneuron.models.query_proposal import QueryDecoderConfig, QueryProposalHead
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        original = QueryProposalHead(QueryDecoderConfig(in_channels=4, query_count=count,
            model_dim=16, layers=1, heads=4, feedforward_dim=32, dropout=0.))
    return GeometricQueryHead(original)


def fixture_inputs(n=32, batch=1):
    g = torch.Generator().manual_seed(23)
    xyz = torch.randint(0, 64, (batch, n, 3), generator=g).float()
    points = torch.cat((xyz, torch.rand(batch, n, 1, generator=g)), -1)
    features = torch.randn(batch, n, 4, generator=g)
    return points, features, torch.ones(batch, 3)*31.5


def test_wrap_preserves_names_values_and_rng_without_extra_parameters():
    from pointneuron.models.query_proposal import QueryDecoderConfig, QueryProposalHead
    original = QueryProposalHead(QueryDecoderConfig(in_channels=4, query_count=8))
    before = common.state_digest(original.state_dict())
    names = tuple(n for n, _ in original.named_parameters())
    rng = torch.get_rng_state().clone()
    candidate = GeometricQueryHead(original)
    assert common.state_digest(candidate.state_dict()) == before
    assert tuple(n for n, _ in candidate.named_parameters()) == names
    assert torch.equal(torch.get_rng_state(), rng)


def test_zero_offsets_reproduce_reference_free_fps_starts_and_crop_local_semantics():
    from pointneuron.models.pointnext_ops import farthest_point_indices, gather_points
    model = fixture_head()
    points, features, center = fixture_inputs(batch=2)
    with torch.no_grad():
        model.prediction[-1].weight[3:].zero_(); model.prediction[-1].bias[3:].zero_()
    output = model(points, features, center, 64.)
    indices = farthest_point_indices((points[..., :3]-center[:, None])/64., 8)
    starts = gather_points(points[..., :3], indices)
    assert torch.equal(output.center_proposals, starts)
    assert torch.equal(output.local_centers, (starts-center[:, None])/64.)
    assert all(len(torch.unique(row)) == 8 for row in indices)


def test_offsets_are_unbounded_and_scale_uses_voxels():
    from pointneuron.models.pointnext_ops import gather_points
    model = fixture_head()
    points, features, center = fixture_inputs()
    with torch.no_grad():
        model.prediction[-1].weight[3:].zero_()
        model.prediction[-1].bias[3:] = torch.tensor([2., -3., 4.])
    output = model(points, features, center, 64.)
    starts = gather_points(points[..., :3], model.starting_indices((points[..., :3]-center[:, None])/64.))
    assert torch.equal(output.center_proposals-starts, torch.tensor([128., -192., 256.]).expand_as(starts))
    assert torch.equal(output.local_centers, (output.center_proposals-center[:, None])/64.)


def test_translation_and_joint_coordinate_scale_preserve_scores_and_local_geometry():
    model = fixture_head(); model.eval()
    points, features, center = fixture_inputs()
    shift = torch.tensor([128., -256., 64.])
    moved = points.clone(); moved[..., :3] += shift
    scaled = points.clone(); scaled[..., :3] *= 2.
    with common.attention_math(), torch.no_grad():
        a = model(points, features, center, 64.)
        b = model(moved, features, center+shift, 64.)
        c = model(scaled, features, center*2, 128.)
    assert torch.allclose(b.center_proposals, a.center_proposals+shift, atol=3e-5, rtol=0)
    assert torch.allclose(b.local_centers, a.local_centers, atol=1e-6, rtol=0)
    assert torch.equal(a.objectness_logits, b.objectness_logits)
    assert torch.equal(a.objectness_logits, c.objectness_logits)
    assert torch.equal(c.center_proposals, 2*a.center_proposals)


def test_full_memory_includes_unselected_points_with_gradients():
    model = fixture_head()
    points, features, center = fixture_inputs()
    features.requires_grad_()
    seen = []
    handle = model.decoder.register_forward_pre_hook(lambda m, args: seen.append(tuple(args[1].shape)))
    with common.attention_math():
        output = model(points, features, center, 64.)
        output.objectness_logits.square().sum().backward()
    handle.remove()
    indices = model.starting_indices((points[..., :3]-center[:, None])/64.)[0]
    unselected = torch.ones(32, dtype=torch.bool); unselected[indices] = False
    assert seen == [(1, 32, 16)]
    assert bool(features.grad[0, unselected].isfinite().all())
    assert bool((features.grad[0, unselected].abs().sum(-1) > 0).all())
    # Perturb only non-start features; the output must still depend on them.
    changed = features.detach().clone(); changed[0, unselected, 0] += 10.
    with common.attention_math(), torch.no_grad():
        after = model(points, changed, center, 64.)
    assert not torch.equal(output.objectness_logits, after.objectness_logits)


def test_query_content_uses_selected_features_position_and_learned_tokens():
    from pointneuron.models.pointnext_ops import gather_points
    model = fixture_head()
    points, features, center = fixture_inputs()
    memory = model.memory_projection(features)+model.position((points[..., :3]-center[:, None])/64.)
    idx = model.starting_indices((points[..., :3]-center[:, None])/64.)
    seen = []
    handle = model.decoder.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
    with common.attention_math():
        model(points, features, center, 64.)
    handle.remove()
    assert torch.equal(seen[0], model.queries.weight[None]+gather_points(memory, idx))


def test_short_and_duplicate_support_uses_distinct_indices_without_synthetic_locations():
    model = fixture_head(8)
    points, features, center = fixture_inputs(n=4)
    points[..., :3].zero_()
    idx = model.starting_indices((points[..., :3]-center[:, None])/64.)
    assert torch.equal(idx, torch.arange(4)[None])
    with common.attention_math():
        output = model(points, features, center, 64.)
    assert output.center_proposals.shape == (1, 4, 3)
    assert bool(torch.isfinite(output.objectness_logits).all())


@pytest.mark.parametrize("scale", [0., -1., float("nan")])
def test_invalid_scale_fails(scale):
    with pytest.raises(ValueError, match="radius"):
        fixture_head()(*fixture_inputs(), scale)


def test_candidate_factory_matches_gn_initial_state():
    recipe = experiment.load_recipe(experiment.CONFIG)
    candidate = experiment.make_model(recipe, "cpu")
    baseline = gn.make_model(gn.load_recipe(gn.CONFIG), "cpu")
    assert common.state_digest(candidate.state_dict()) == common.state_digest(baseline.state_dict())
    assert sum(p.numel() for p in candidate.parameters()) == 1688518


def test_altered_query_policy_cannot_reuse_the_candidate_config(tmp_path):
    control = json.loads(experiment.CONFIG.read_text())
    control["query_policy"]["memory"] = "only_selected_points"
    path = tmp_path/"changed.json"; path.write_text(json.dumps(control))
    with pytest.raises(ValueError, match="declared"):
        experiment.load_recipe(path)


def test_old_gn_timing_is_incompatible(monkeypatch, tmp_path):
    (tmp_path/"report.json").write_text(json.dumps({"stage": "timing", "status": "completed", "updates": 8,
        "checkpoint_reload_passed": True, "normalization_consistency_passed": True, "recipe_fingerprint": "old_GN"}))
    monkeypatch.setattr(experiment, "identity", lambda r: {"recipe_fingerprint": "new_geometric"})
    with pytest.raises(ValueError, match="incompatible"):
        experiment.require_timing({}, tmp_path)


def test_changed_candidate_code_is_rejected_by_cpu_gate(monkeypatch, tmp_path):
    report = {"status": "passed", "normalization_consistency_passed": True, "checkpoint_reload_passed": True,
        "records": [0]*4, "synthetic_budget_checks": [0]*2, "recipe_fingerprint": "stale"}
    path = tmp_path/"report.json"; path.write_text(json.dumps(report))
    monkeypatch.setattr(experiment, "identity", lambda r: {"recipe_fingerprint": "current"})
    with pytest.raises(ValueError, match="stale"):
        experiment.require_cpu({}, path)


def test_missing_worker_report_is_not_success(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "require_cpu", lambda *a: None)
    monkeypatch.setattr(common, "load_tasks", lambda *a: None)
    monkeypatch.setattr(experiment, "identity", lambda *a: {"fixture": True})
    monkeypatch.setattr(experiment, "launch_worker", lambda *a: (0, "completed"))
    args = SimpleNamespace(action="timing", output=str(tmp_path/"run"), config="fixture.json", timing_run=None, cpu_preflight=None)
    assert experiment.controller({"timing_worker_timeout_seconds": 1}, args) == 1
    assert json.loads((Path(args.output)/"report.json").read_text())["status"] == "failed_incomplete_control"


def test_checkpoint_recipe_rejects_wrong_query_semantics(tmp_path):
    model = fixture_head()
    recipe = {"optimizer": dict(learning_rate=.0001, weight_decay=0., betas=[.9,.999], eps=1e-8),
              "geometric_query_control": {"query_policy": experiment.QUERY_POLICY}}
    optimizer = common.make_optimizer(model, recipe)
    points, features, center = fixture_inputs()
    with common.attention_math():
        loss = model(points, features, center, 64.).objectness_logits.square().mean()
        loss.backward(); optimizer.step()
    path = tmp_path/"state.pt"
    common.save_checkpoint(path, model, optimizer, 1, recipe, "pn2_sato")
    expected = gn.prefix_identity(model, optimizer)
    restored = copy.deepcopy(model); restored_optimizer = common.make_optimizer(restored, recipe)
    assert common.restore_checkpoint(path, restored, restored_optimizer, recipe, "pn2_sato") == 1
    assert gn.prefix_identity(restored, restored_optimizer) == expected
    wrong = copy.deepcopy(recipe); wrong["geometric_query_control"]["query_policy"]["center"] = "crop_center"
    with pytest.raises(ValueError, match="incompatible"):
        common.restore_checkpoint(path, restored, restored_optimizer, wrong, "pn2_sato")
