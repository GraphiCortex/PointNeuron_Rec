"""Mathematical contract checks; no dataset training or GPU execution."""
from itertools import permutations
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from torch.nn import functional as F
from pn2.models.consistent_node_objective import consistent_node_set_loss
from pn2.models.node_objective import node_set_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import check_consistent_node_objective as candidate
import sampler_node_common as common
from pn2.training import node_fit_state as state


def output(xyz, logits):
    return SimpleNamespace(center_proposals=torch.tensor([xyz], dtype=torch.float32, requires_grad=True),
        objectness_logits=torch.tensor([logits], dtype=torch.float32, requires_grad=True))


def explicit(out, target, indices):
    q = out.center_proposals.shape[1]
    labels = torch.zeros(q, dtype=torch.long); labels[list(indices)] = 1
    ce = F.cross_entropy(out.objectness_logits[0], labels)
    distances = torch.linalg.vector_norm(out.center_proposals[0, list(indices)]-target, dim=-1)/6
    geometry = F.smooth_l1_loss(distances, torch.zeros_like(distances), beta=4/6, reduction='sum')/q
    return 2*ce+3*geometry


@pytest.mark.parametrize('seed', range(6))
def test_assignment_minimizes_actual_loss_against_all_injections(seed):
    torch.manual_seed(seed)
    out = SimpleNamespace(center_proposals=torch.randn(1, 4, 3)*9,
                          objectness_logits=torch.randn(1, 4, 2)*3)
    target = torch.randn(2, 3)*9
    loss = consistent_node_set_loss(out, target)
    alternatives = [explicit(out, target, p) for p in permutations(range(4), 2)]
    assert torch.allclose(loss.total, torch.stack(alternatives).min(), atol=2e-6)
    by_target = [None]*2
    for p, t in zip(loss.prediction_indices, loss.target_indices): by_target[t] = p
    assert torch.allclose(loss.total, explicit(out, target, by_target), atol=2e-6)


@pytest.mark.parametrize('m', [1, 3, 7])
def test_uninformative_classifier_optimum_tracks_slot_prevalence(m):
    q = 8
    target = torch.tensor([[float(i), 0, 0] for i in range(m)])
    xyz = torch.tensor([[[float(i), 0, 0] for i in range(q)]])
    score = torch.tensor(0., requires_grad=True)
    logits = torch.stack((torch.zeros(q), score.expand(q)), -1)[None]
    out = SimpleNamespace(center_proposals=xyz, objectness_logits=logits)
    new = consistent_node_set_loss(out, target)
    derivative = torch.autograd.grad(new.total, score, retain_graph=True)[0]
    assert float(derivative) == pytest.approx(2*(.5-m/q), abs=1e-6)
    old = node_set_loss(out, target)
    assert abs(float(torch.autograd.grad(old.total, score)[0])) < 1e-6
    # This checks class weighting, not whether a real model learns useful scores.


def test_unknown_abstains_and_does_not_turn_missing_reference_into_background():
    out = output([[0, 0, 0]], [[0, 3]])
    loss = consistent_node_set_loss(out, torch.empty(0, 3))
    assert not loss.supervised and loss.total.item() == 0
    loss.total.backward()
    assert out.objectness_logits.grad.count_nonzero() == 0
    assert out.center_proposals.grad.count_nonzero() == 0


def test_finite_geometry_and_classifier_gradients_and_coincident_targets():
    out = output([[1, 0, 0], [3, 0, 0], [8, 0, 0]], [[0, 1], [0, 2], [0, -2]])
    target = torch.tensor([[0., 0, 0], [0., 0, 0]])
    loss = consistent_node_set_loss(out, target);loss.total.backward()
    assert len(set(loss.prediction_indices)) == 2
    for value in (out.center_proposals.grad, out.objectness_logits.grad):
        assert torch.isfinite(value).all() and value.abs().sum() > 0


def test_translation_and_target_permutation_do_not_change_loss():
    out = output([[1, 2, 0], [8, 0, 0], [3, 4, 5]], [[0, 1], [0, -1], [1, 0]])
    target = torch.tensor([[1., 1, 0], [7., 0, 0]])
    first = consistent_node_set_loss(out, target)
    shifted = SimpleNamespace(center_proposals=out.center_proposals+100,
                              objectness_logits=out.objectness_logits)
    second = consistent_node_set_loss(shifted, target.flip(0)+100)
    assert torch.allclose(first.total, second.total, atol=1e-6)


def test_full_occupancy_and_exact_zero_distance_have_finite_gradients():
    out = output([[1, 0, 0], [2, 0, 0]], [[0, 0], [0, 0]])
    loss = consistent_node_set_loss(out, out.center_proposals[0].detach().clone())
    loss.total.backward()
    assert torch.isfinite(out.center_proposals.grad).all()
    assert loss.center.item() == 0 and len(loss.prediction_indices) == 2


def test_overflow_nonfinite_and_wrong_dtype_are_rejected():
    out = output([[0, 0, 0]], [[0, 0]])
    with pytest.raises(ValueError, match='overflow'):
        consistent_node_set_loss(out, torch.zeros(2, 3))
    with pytest.raises(ValueError, match='float32'):
        consistent_node_set_loss(out, torch.zeros(1, 3, dtype=torch.float64))
    with pytest.raises(FloatingPointError):
        consistent_node_set_loss(out, torch.full((1, 3), float('nan')))


def test_local_curve_does_not_determine_full_chain_sample_phase():
    result = candidate.target_phase_counterexample()
    assert result['same_local_curve'] and not result['same_target_nodes']
    assert not result['active_targets_changed']


@pytest.mark.parametrize('method', ['baseline', 'consistent'])
def test_next_adam_update_and_rng_reload_are_exact_for_both_objectives(method):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.linear = torch.nn.Linear(3, 5)
        def forward(self, x):
            values = self.linear(x+torch.rand_like(x)*.01)
            return SimpleNamespace(center_proposals=values[:, :, :3], objectness_logits=values[:, :, 3:])
    torch.manual_seed(71)
    model = Tiny();opt = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=0., foreach=False)
    record = torch.tensor([[[1., 0, 0], [3., 0, 0], [8., 0, 0]]])
    target = torch.tensor([[0., 0, 0], [4., 0, 0]])
    recipe = {'loss': {}, 'gradient_clip': 10.}
    class Batch(dict):
        # The model receives this fixture while the step reads its targets.
        pass
    original_forward = model.forward
    model.forward = lambda batch: original_forward(batch['inputs'])
    batch = Batch(inputs=record, target=target)
    candidate.step(model, opt, batch, recipe, method)
    bindings = {'maximum_updates': 2}
    buffer = io.BytesIO();torch.save(state.capture(model, opt, bindings, 1), buffer)
    expected_loss = candidate.step(model, opt, batch, recipe, method)
    expected = (common.state_digest(model.state_dict()), common.nested_digest(opt.state_dict()),
                common.nested_digest(common.rng_state()))
    buffer.seek(0);payload = torch.load(buffer, weights_only=False)
    state.restore(payload, model, opt, bindings)
    actual_loss = candidate.step(model, opt, batch, recipe, method)
    actual = (common.state_digest(model.state_dict()), common.nested_digest(opt.state_dict()),
              common.nested_digest(common.rng_state()))
    assert actual == expected and actual_loss == expected_loss


def test_gpu_gate_rejects_a_stale_cpu_report(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(candidate, 'ROOT', tmp_path)
    (tmp_path/'report.json').write_text(json.dumps({'status': 'passed', 'config_fingerprint': 'stale'}))
    with pytest.raises(ValueError, match='CPU audit'):
        candidate.require_cpu({'cpu_report': 'report.json'})
