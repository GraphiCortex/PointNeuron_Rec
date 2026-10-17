"""Check the agreed objective, preserved geometry and full-state continuation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from pointneuron.models.hard_negative_query_loss import POLICY, classification_term, hard_negative_query_loss
from pointneuron.models.balanced_query_loss import balanced_query_loss, POLICY as BALANCED_POLICY
from pointneuron.models.pointnext_query_cropnorm import PointNextCropNormQueryModel
from pointneuron.models.query_fit_state import fit_checkpoint, restore_fit
from scripts.pointneuron2 import hard_negative_query_experiment as runner


def recipe():
    value = json.loads((ROOT/"configs/pointneuron2/branched_query_fit.json").read_text())
    return dict(value, classification_policy=POLICY)


@pytest.mark.parametrize("p", [6, 12, 16, 418])
def test_actual_logit_gradients_equal_the_proposed_coefficients(p):
    torch.set_num_threads(2)
    logits = torch.randn(512, 2, generator=torch.Generator().manual_seed(13), dtype=torch.float64, requires_grad=True)
    positive = torch.arange(512) < p
    loss, hard = classification_term(logits, positive)
    negative = np.arange(p, 512)
    margin = (logits[:, 1]-logits[:, 0]).detach().numpy()
    expected_hard = negative[np.lexsort((negative, -margin[negative]))[:min(p, 512-p)]]
    assert set(hard.tolist()) == set(expected_hard.tolist())
    expected_weights = torch.zeros(512, dtype=torch.float64)
    expected_weights[positive] = .5/p
    expected_weights[~positive] = .25/(512-p)
    expected_weights[expected_hard] += .25/len(expected_hard)
    loss.backward()
    target = positive.to(torch.float64)
    expected = expected_weights*(logits.detach().softmax(-1)[:, 1]-target)
    torch.testing.assert_close(logits.grad[:, 1], expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(logits.grad[:, 0], -expected, rtol=1e-12, atol=1e-12)
    assert bool((logits.grad[~positive, 1] > 0).all())  # Every unmatched query stays supervised.
    if p == 418:
        ref = .5*(-logits.log_softmax(-1)[positive, 1].mean()-logits.log_softmax(-1)[~positive, 0].mean())
        reference_gradient = torch.autograd.grad(ref, logits)[0]
        assert torch.equal(loss.detach(), ref.detach()) and torch.equal(logits.grad, reference_gradient)


def test_exact_score_ties_use_query_index_and_high_scoring_positives_are_excluded():
    logits = torch.zeros(8, 2, requires_grad=True)
    positive = torch.tensor([False, True, False, False, False, False, True, False])
    with torch.no_grad():
        logits[positive, 1] = 100
    _, hard = classification_term(logits, positive)
    assert hard.tolist() == [0, 2]


@pytest.mark.parametrize("p", [6, 418])
def test_assignment_and_geometry_gradients_match_balanced_objective(p):
    g = torch.Generator().manual_seed(9)
    out = SimpleNamespace(center_proposals=torch.rand(1,512,3,generator=g,requires_grad=True),
        radius=(torch.rand(1,512,1,generator=g)+1).requires_grad_(),
        objectness_logits=torch.randn(1,512,2,generator=g,requires_grad=True))
    nodes = torch.zeros(1,p,6); nodes[:,:,1:4] = torch.rand(1,p,3,generator=g); nodes[:,:,4] = 1
    mask = torch.ones(1,p,dtype=torch.bool)
    old = balanced_query_loss(out,nodes,mask)
    new = hard_negative_query_loss(out,nodes,mask)
    assert old.assignments == new.assignments
    assert torch.equal(old.center,new.center) and torch.equal(old.radius,new.radius)
    variables = (out.center_proposals,out.radius,out.objectness_logits)
    a = torch.autograd.grad(old.total,variables,retain_graph=True)
    b = torch.autograd.grad(new.total,variables)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    assert torch.equal(a[2],b[2]) == (p == 418)


def test_unknown_crops_are_skipped_and_all_positive_crop_is_supervised():
    out = SimpleNamespace(center_proposals=torch.zeros(2,4,3,requires_grad=True),
        radius=torch.ones(2,4,1,requires_grad=True), objectness_logits=torch.zeros(2,4,2,requires_grad=True))
    nodes = torch.zeros(2,4,6); nodes[:,:,4] = 1
    mask = torch.tensor([[False]*4,[True]*4])
    result = hard_negative_query_loss(out,nodes,mask)
    assert result.supervised_crops == 1 and result.matched_count == 4
    assert abs(float(result.objectness)-np.log(2)) < 1e-6
    result.total.backward()
    assert out.objectness_logits.grad[0].abs().sum() == 0
    empty = hard_negative_query_loss(out,nodes,torch.zeros_like(mask))
    assert empty.supervised_crops == 0 and float(empty.total.detach()) == 0
    with pytest.raises(ValueError):
        classification_term(out.objectness_logits[0],torch.zeros(4,dtype=torch.bool))


def test_full_model_update_and_disk_resume_are_exact_on_single_thread_cpu(tmp_path):
    torch.set_num_threads(1)
    r = recipe()
    points = np.random.default_rng(7).uniform(0,96,(256,4)).astype(np.float32)
    points[:,3] *= 255/96
    record = {"points": points, "nodes": np.array([[1,40,40,40,1,-1],[2,48,48,48,1,-1]],dtype=np.float32),
        "meta": {"patch_center": [48.,48.,48.], "image_scale": 255.}}
    model = PointNextCropNormQueryModel(0)
    optimizer = torch.optim.AdamW(model.parameters(),lr=r["learning_rate"],weight_decay=r["weight_decay"])
    metrics = runner.update(model,optimizer,record,r,"cpu")
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert len(optimizer.state) == len(list(model.parameters())) == 123
    cp = fit_checkpoint(model,optimizer,step=1,recipe=r,provenance={"test":"hard-negative"},backend={"device":"cpu"})
    path = tmp_path/"step_0001.pt"; torch.save(cp,path)
    loaded = torch.load(path,map_location="cpu",weights_only=True)
    old_recipe = dict(r,classification_policy=BALANCED_POLICY)
    with pytest.raises(ValueError):
        restore_fit(loaded,recipe=old_recipe,provenance=cp["provenance"],backend=cp["backend"],device="cpu")
    runner.update(model,optimizer,record,r,"cpu")
    restored, restored_optimizer, step = restore_fit(loaded,recipe=r,provenance=cp["provenance"],backend=cp["backend"],device="cpu")
    runner.update(restored,restored_optimizer,record,r,"cpu")
    assert step == 1 and all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    assert {int(s["step"]) for s in restored_optimizer.state.values()} == {2}
    report = {}; runner.record_gradient_diagnostic(report,metrics,r)
    assert report["gradient_clipping_this_segment"]["updates_observed"] == 1


def test_checkpoint_is_preserved_before_evaluation_failure(tmp_path,monkeypatch):
    r = recipe(); model = PointNextCropNormQueryModel(0)
    optimizer = torch.optim.AdamW(model.parameters(),lr=r["learning_rate"],weight_decay=r["weight_decay"])
    def fail(*args):
        raise RuntimeError("synthetic evaluation failure")
    monkeypatch.setattr(runner,"evaluate",fail)
    report = {"evaluations": []}
    with pytest.raises(RuntimeError,match="synthetic evaluation failure"):
        runner.snapshot(model,optimizer,[],r,"cpu",tmp_path,0,{"test":"snapshot"},{"device":"cpu"},report)
    saved = torch.load(tmp_path/"step_0000.pt",map_location="cpu",weights_only=True)
    assert saved["experiment_schema"] == runner.SCHEMA and saved["recipe"]["classification_policy"] == POLICY
    assert json.loads((tmp_path/"report.json").read_text())["last_checkpoint_step"] == 0
