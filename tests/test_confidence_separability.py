"""Known geometry checks for the numerical feasibility diagnostic."""
import numpy as np
from scripts.pointneuron2.audit_confidence_separability import separability


def test_known_separable_features_have_a_verified_margin():
    result=separability(np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]]),np.array([0,0,1,1]))
    assert result["numerical_conclusion"]=="strict_linear_separator_verified"
    assert result["minimum_float64_signed_margin"]>=1-1e-6
    assert result["float32_label_errors"]==0


def test_identical_opposite_labels_have_a_checked_infeasibility_witness():
    result=separability(np.array([[1.,2.],[1.,2.]]),np.array([0,1]))
    assert result["numerical_conclusion"]=="infeasibility_certificate_verified_within_tolerance"
    assert result["certificate_max_residual"]<=1e-7
    assert len(result["witness_rows"])==2


def test_xor_is_not_affinely_separable_despite_distinct_features():
    result=separability(np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]),np.array([0,1,1,0]))
    assert result["numerical_conclusion"]=="infeasibility_certificate_verified_within_tolerance"
