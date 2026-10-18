"""Four-crop balanced head loss, frozen-state invariance and exact continuation."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"),str(ROOT)]
from pointneuron.models.four_crop_frozen_confidence import FourCropFrozenConfidence, stage_loss
from scripts.pointneuron2 import four_crop_frozen_confidence as runner
from scripts.pointneuron2.reference_gaps import segment_probe


def recipe():
    return json.loads((ROOT/"configs/pointneuron2/four_crop_frozen_confidence.json").read_text())


def cache():
    labels = torch.zeros(4,512,dtype=torch.long)
    for i,p in enumerate((6,12,16,418)): labels[i,:p] = 1
    features = torch.zeros(4,512,128); features[:,:,0] = 2*labels.float()-1
    return {"features":features,"labels":labels,"centers":torch.zeros(4,512,3),"radii":torch.ones(4,512,1),
        "initial_weight":torch.zeros(2,128),"initial_bias":torch.zeros(2)}


def test_balanced_loss_is_mean_of_four_crop_class_means_and_reaches_all_parameters():
    torch.set_num_threads(2)
    data = cache(); head = runner.make_head(data)
    logits = head(data["features"]); logits.retain_grad()
    loss,per_crop = stage_loss(logits,data["labels"])
    expected = torch.stack([.5*(F.cross_entropy(logits[i,data["labels"][i] == 1],data["labels"][i,data["labels"][i] == 1])
        + F.cross_entropy(logits[i,data["labels"][i] == 0],data["labels"][i,data["labels"][i] == 0])) for i in range(4)])
    # Independent CE reductions differ by float32 summation rounding.
    torch.testing.assert_close(per_crop,expected,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(loss,2*expected.mean(),rtol=1e-6,atol=1e-7)
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    assert sum(p.numel() for p in head.parameters()) == 258 and set(dict(head.named_parameters())) == {"linear.weight","linear.bias"}
    # The sum of positive/negative used-logit gradient magnitudes matches in each crop.
    for i in range(4):
        positive = data["labels"][i] == 1
        torch.testing.assert_close(logits.grad[i,positive,1].abs().sum(),logits.grad[i,~positive,1].abs().sum())


def test_synthetic_fit_cannot_mutate_cached_features_geometry_or_initial_rows():
    data = cache(); before = runner.fingerprints(data); head = runner.make_head(data)
    r = dict(recipe(),learning_rate=.1)
    optimizer = torch.optim.AdamW(head.parameters(),lr=.1,weight_decay=r["weight_decay"])
    for _ in range(20): runner.update(head,optimizer,data,r)
    assert torch.equal((head(data["features"])[...,1] >= head(data["features"])[...,0]).long(),data["labels"])
    assert runner.fingerprints(data) == before
    assert all(not value.requires_grad for value in data.values())
    with pytest.raises(ValueError,match="detached"):
        head(data["features"].clone().requires_grad_(True))
    with pytest.raises(ValueError):
        stage_loss(head(data["features"]),torch.zeros(4,512,dtype=torch.long))


def test_disk_resume_reproduces_next_stage_update_and_refuses_other_recipe_or_terminal(tmp_path):
    torch.set_num_threads(1)
    data = cache(); head = runner.make_head(data); r = recipe()
    optimizer = torch.optim.AdamW(head.parameters(),lr=r["learning_rate"],weight_decay=r["weight_decay"])
    for _ in range(3): runner.update(head,optimizer,data,r)
    cp = runner.checkpoint(head,optimizer,3,r,{"test":"stage"})
    path = tmp_path/"step_0003.pt"; torch.save(cp,path)
    loaded = torch.load(path,map_location="cpu",weights_only=True)
    runner.update(head,optimizer,data,r)
    restored,other,step = runner.restore(loaded,r,{"test":"stage"},resume=True)
    runner.update(restored,other,data,r)
    assert step == 3 and all(torch.equal(v,restored.state_dict()[k]) for k,v in head.state_dict().items())
    assert {int(s["step"]) for s in other.state.values()} == {4}
    for change in ({"updates":8000},{"learning_rate":.001}):
        with pytest.raises(ValueError): runner.restore(loaded,{**r,**change},{"test":"stage"},resume=True)
    for altered in ({**loaded,"resumable":False},{**loaded,"purpose":"check"},{**loaded,"step":4000}):
        with pytest.raises(ValueError): runner.restore(altered,r,{"test":"stage"},resume=True)
    with pytest.raises(ValueError): runner.checkpoint(head,optimizer,4001,r,{})


def test_snapshot_preserves_complete_state_before_metric_failure(tmp_path,monkeypatch):
    data = cache(); head = runner.make_head(data); r = recipe()
    optimizer = torch.optim.AdamW(head.parameters(),lr=r["learning_rate"],weight_decay=r["weight_decay"])
    def fail(*args): raise RuntimeError("synthetic metrics failure")
    monkeypatch.setattr(runner,"evaluate",fail)
    report = {"evaluations":[]}
    with pytest.raises(RuntimeError,match="synthetic metrics failure"):
        runner.snapshot(head,optimizer,0,r,{"test":"snapshot"},report,tmp_path,data,{},[])
    cp = torch.load(tmp_path/"step_0000.pt",map_location="cpu",weights_only=True)
    assert cp["step"] == 0 and cp["optimizer_state"]["state"] == {}
    assert json.loads((tmp_path/"report.json").read_text())["last_checkpoint_step"] == 0


def test_strict_label_gate_does_not_claim_geometry_success():
    data = cache(); head = FourCropFrozenConfidence(torch.tensor([[-1.]+[0.]*127,[1.]+[0.]*127]),torch.zeros(2))
    records,rows = [],[]
    for i,p in enumerate((6,12,16,418)):
        data[f"candidate_edges_{i}"] = torch.tensor([[0,1]])
        data[f"edge_labels_{i}"] = torch.tensor([1.])
        segments = np.array([[[0.,0.,0.],[1.,0.,0.]]])
        geometry = {"full_segments":segments,"critical_points":np.array([[0.,0.,0.]]),
            "chains":[{"length_voxels":1.,"probe":segment_probe(segments)}]}
        records.append({"target_edges":np.array([[0,1]]),"geometry":geometry})
        rows.append({"sample_index":i,"patch_index":0,"source_snapshot_row":{
            "connectivity_supervision":{"trusted_matches":0},"trusted_visible_junctions":0}})
    result = runner.evaluate(head,data,{"records":rows},records,0)
    assert result["strict_fixed_label_gate"] and all(r["frozen_trusted_targets"] == 0 for r in result["crops"])
