"""Analytic loss/metrics and exact synthetic checkpoint continuation; no dataset/GPU."""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Use retained Torch runtime for learned-component tests")
from pn2.models.node_objective import node_set_loss
from pn2.evaluation.node_proposals import one_to_one, proposal_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sampler_node_common as common


def output(points, logits):
    return SimpleNamespace(center_proposals=torch.tensor([points], dtype=torch.float32, requires_grad=True),
                           objectness_logits=torch.tensor([logits], dtype=torch.float32, requires_grad=True),
                           radius=torch.full((1, len(points), 1), float("nan"), requires_grad=True))


def test_geometry_matching_and_analytic_balanced_loss_ignore_radius():
    prediction = output([[0, 0, 0], [10, 0, 0]], [[2., 0.], [0., 2.]])
    target = torch.tensor([[.5, 0, 0]], dtype=torch.float32)
    loss = node_set_loss(prediction, target)
    assert loss.prediction_indices == (0,)
    expected_class = torch.log1p(torch.exp(torch.tensor(2.)))
    assert float(loss.classification) == pytest.approx(float(expected_class))
    assert float(loss.center) == pytest.approx(.5**2/(2*4))
    assert float(loss.total.detach()) == pytest.approx(2*float(expected_class)+3*.5**2/8)
    loss.total.backward()
    assert prediction.radius.grad is None
    assert torch.isfinite(prediction.center_proposals.grad).all()
    assert torch.isfinite(prediction.objectness_logits.grad).all()


def test_negative_group_size_does_not_change_identical_mean_class_loss():
    target = torch.zeros(1, 3)
    a = node_set_loss(output([[0, 0, 0], [10, 0, 0]], [[0, 1], [1, 0]]), target)
    b = node_set_loss(output([[0, 0, 0], [10, 0, 0], [20, 0, 0]], [[0, 1], [1, 0], [1, 0]]), target)
    assert float(a.total.detach()) == pytest.approx(float(b.total.detach()))


def test_exact_center_match_has_finite_backward_and_target_order_is_irrelevant():
    prediction = output([[0, 0, 0], [10, 0, 0], [100, 0, 0]], [[0, 1], [0, 1], [1, 0]])
    target = torch.tensor([[0., 0, 0], [10., 0, 0]])
    a = node_set_loss(prediction, target)
    b = node_set_loss(prediction, target.flip(0))
    assert float(a.total.detach()) == pytest.approx(float(b.total.detach()))
    a.total.backward()
    assert torch.isfinite(prediction.center_proposals.grad).all()


def test_common_translation_does_not_change_loss():
    target = torch.tensor([[.25, 0, 0]])
    a = node_set_loss(output([[0, 0, 0], [10, 0, 0]], [[0, 1], [1, 0]]), target)
    b = node_set_loss(output([[100, 100, 100], [110, 100, 100]], [[0, 1], [1, 0]]), target+100)
    assert float(a.total.detach()) == pytest.approx(float(b.total.detach()))


def test_empty_reference_abstains_with_zero_gradients_instead_of_background_labels():
    prediction = output([[0, 0, 0], [1, 0, 0]], [[0, 1], [0, 1]])
    result = node_set_loss(prediction, torch.empty(0, 3))
    assert not result.supervised and result.prediction_indices == ()
    result.total.backward()
    assert torch.count_nonzero(prediction.objectness_logits.grad) == 0


def test_overflow_nonfinite_geometry_and_invalid_loss_scale_are_rejected():
    prediction = output([[0, 0, 0]], [[0, 1]])
    with pytest.raises(ValueError, match="overflow"):
        node_set_loss(prediction, torch.zeros(2, 3))
    with pytest.raises(FloatingPointError):
        node_set_loss(prediction, torch.full((1, 3), float("nan")))
    with pytest.raises(ValueError):
        node_set_loss(prediction, torch.zeros(1, 3), center_weight=float("inf"))


def test_maximum_matching_avoids_greedy_conflict_and_counts_duplicates_as_extras():
    reference = np.array([[0, 0, 0], [1, 0, 0]])
    predicted = np.array([[.5, 0, 0], [-.25, 0, 0]])
    metrics, mask = one_to_one(reference, predicted, .6)
    assert metrics["matched"] == 2 and mask.all()
    metrics, _ = one_to_one(reference[:1], np.zeros((2, 3)), 0.)
    assert metrics["matched"] == 1 and metrics["extras"] == 1 and metrics["precision"] == .5


def test_a_single_proposal_cannot_count_as_two_close_targets():
    target = np.array([[0, 0, 0], [.02, 0, 0]])
    metrics = proposal_metrics(np.zeros((1, 3)), np.array([.8]), target, [set(), set()], [])
    row = metrics["tolerances"]["2.0"]["accepted_nodes"]
    assert row["matched"] == 1 and row["recall"] == .5
    assert not metrics["learnability_gate"]


def test_cutoff_is_fixed_empty_acceptance_is_zero_and_curve_is_length_weighted():
    target = np.array([[0, 0, 0]])
    samples = [(np.array([[0, 0, 0], [10, 0, 0]]), np.array([1., 3.]))]
    metrics = proposal_metrics(target, np.array([.5]), target, [{"leaf"}], samples)
    assert metrics["accepted_count"] == 1
    assert metrics["tolerances"]["2.0"]["curve_to_accepted_node_support"] == .25
    assert metrics["tolerances"]["2.0"]["longest_unsupported_run"] == 3
    assert metrics["learnability_gate"]
    empty = proposal_metrics(target, np.array([.499]), target, [{"leaf"}], samples)
    assert empty["tolerances"]["2.0"]["accepted_nodes"]["precision"] == 0
    assert empty["tolerances"]["2.0"]["accepted_nodes"]["recall"] == 0
    assert empty["tolerances"]["2.0"]["longest_unsupported_run"] == 4


def test_current_crop_diagnostic_restores_batchnorm_buffers_and_mode():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2)).eval()
    before = common.state_digest(model.state_dict())
    common.current_crop_diagnostic(model, torch.tensor([[1., 2, 3], [4., 5, 6]]))
    assert common.state_digest(model.state_dict()) == before and not model.training


def test_current_crop_diagnostic_restores_buffers_even_when_forward_fails():
    class Failing(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.bn = torch.nn.BatchNorm1d(3)
        def forward(self, value):
            self.bn(value); raise RuntimeError("fixture failure")
    model = Failing().eval()
    before = common.state_digest(model.state_dict())
    with pytest.raises(RuntimeError, match="fixture failure"):
        common.current_crop_diagnostic(model, torch.ones(2, 3))
    assert common.state_digest(model.state_dict()) == before and not model.training


def test_checkpoint_reproduces_next_dropout_adam_update_and_all_rng_streams(tmp_path):
    torch.manual_seed(17); np.random.seed(18); random.seed(19)
    def build():
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.25), torch.nn.Linear(5, 2))
        return model, torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0., foreach=False)
    def step(model, optimizer):
        value = torch.rand(4, 3) + float(np.random.rand()) + random.random()
        optimizer.zero_grad(set_to_none=True)
        loss = model(value).square().mean(); loss.backward(); optimizer.step()
        return float(loss.detach())
    model, optimizer = build(); step(model, optimizer)
    recipe = {"fixture": "synthetic_only"}
    path = tmp_path / "state.pt"
    common.save_checkpoint(path, model, optimizer, 1, recipe, "fixture")
    expected_loss = step(model, optimizer)
    expected_model = common.state_digest(model.state_dict())
    expected_adam = common.nested_digest(optimizer.state_dict())
    expected_rng = common.nested_digest(common.rng_state())
    restored, restored_optimizer = build()
    assert common.restore_checkpoint(path, restored, restored_optimizer, recipe, "fixture") == 1
    assert step(restored, restored_optimizer) == expected_loss
    assert common.state_digest(restored.state_dict()) == expected_model
    assert common.nested_digest(restored_optimizer.state_dict()) == expected_adam
    assert common.nested_digest(common.rng_state()) == expected_rng
    with pytest.raises(FileExistsError):
        common.save_checkpoint(path, restored, restored_optimizer, 2, recipe, "fixture")
    with path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checkpoint"):
        common.restore_checkpoint(path, restored, restored_optimizer, recipe, "fixture")
