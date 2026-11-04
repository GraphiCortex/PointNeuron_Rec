import importlib.util
from pathlib import Path
import sys
import pytest


def test_execution_prefix_requires_exact_loss_and_gradient_identity():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    s=importlib.util.spec_from_file_location('resident_fixture',scripts/'run_broad_curve_resident.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    a=[{'loss':.5,'gradient_norm':1.2,'support_loss':.3,'offset_loss':.2}]
    m.exact_prefix(a,[dict(a[0])])
    with pytest.raises(ValueError,match='changed saved'):m.exact_prefix(a,[{**a[0],'gradient_norm':1.200001}])
