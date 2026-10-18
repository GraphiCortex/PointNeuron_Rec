"""Known feasible/infeasible cases and the actual hidden-feature boundary."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pointneuron2 import audit_four_crop_confidence_capacity as audit


def verifier(x,y):
    def verify(w,b):
        logits = F.linear(torch.tensor(x,dtype=torch.float32),torch.tensor(np.stack((np.zeros_like(w),w)),dtype=torch.float32),torch.tensor([0,b],dtype=torch.float32))
        score = logits[:,1]-logits[:,0]
        margin = torch.tensor(2*np.asarray(y)-1)*score
        return {"finite":bool(torch.isfinite(margin).all()),"label_errors":int(((score >= 0).numpy() != y).sum()),"minimum_signed_margin":float(margin.min())}
    return verify


@pytest.mark.parametrize("tiny",[False,True])
def test_separable_data_and_small_nonconstant_feature_are_retained(tiny):
    x = np.array([[1,0],[1,1],[1,2],[1,3]],dtype=np.float32)
    if tiny: x[:,1] *= 1e-12
    y = np.array([0,0,1,1])
    result = audit.linear_capacity(x,y,verifier(x,y),seconds=5)
    assert result["conclusion"] == "strict_separator_verified_in_float32_classifier_arithmetic"
    assert result["active_feature_columns"] == 1
    assert result["float32_verification"]["label_errors"] == 0


@pytest.mark.parametrize("xor",[False,True])
def test_opposing_identical_points_and_xor_need_numerical_certificate(xor):
    x = np.array([[0,0],[0,1],[1,0],[1,1]]) if xor else np.ones((2,2))
    y = np.array([0,1,1,0]) if xor else np.array([0,1])
    result = audit.linear_capacity(x,y,verifier(x,y),seconds=5)
    assert result["conclusion"] == "numerical_infeasibility_certificate_within_tolerance"
    assert result["certificate_max_residual"] <= 1e-7
    assert result["norm_bounded_margin_upper_bound_numerical"] <= 1e-7


def test_timeout_is_unresolved_and_does_not_call_witness_verifier(monkeypatch):
    monkeypatch.setattr(audit,"linprog",lambda *a,**kw:SimpleNamespace(success=False,status=1,message="test time limit"))
    result = audit.linear_capacity(np.array([[0.],[1.]]),np.array([0,1]),lambda *a:pytest.fail("No witness exists"))
    assert result["conclusion"] == "unresolved" and result["solver_calls"] == 1


def test_float32_failure_does_not_become_a_separability_claim():
    result = audit.linear_capacity(np.array([[0.],[1.]]),np.array([0,1]),
        lambda *a:{"finite":True,"label_errors":1,"minimum_signed_margin":0.})
    assert result["primal_status"] == 0 and result["conclusion"] == "unresolved"


def test_nonfinite_float32_witness_is_inconclusive():
    result = audit.linear_capacity(np.array([[0.],[1.]]),np.array([0,1]),
        lambda *a:{"finite":False,"label_errors":None,"minimum_signed_margin":None})
    assert result["primal_status"] == 0 and result["conclusion"] == "unresolved"


def test_actual_classifier_input_is_post_hidden_relu():
    state = {"proposal.prediction.0.weight":torch.eye(2),"proposal.prediction.0.bias":torch.zeros(2),
        "proposal.prediction.2.weight":torch.ones(6,2),"proposal.prediction.2.bias":torch.arange(6.)}
    hidden,raw = audit.reconstruct_inputs(torch.tensor([[[-2.,3.]]]),state)
    assert hidden.tolist() == [[[0.,3.]]]
    assert torch.equal(raw,torch.arange(6.)[None,None]+3)
