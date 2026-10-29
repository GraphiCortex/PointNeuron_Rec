import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pn2.evaluation.swc import Morphology
from pn2.inputs import vaa3d
from pn2.inputs.sampling import probabilities, ranked_selection

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pn1_comparison", ROOT / "scripts/compare_pn1_input_evidence.py")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
comparison = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)


def baseline():
    config = json.loads((ROOT / "configs/pn1_input_comparison.json").read_text())
    if any(not (ROOT.parent / name).is_file() for name in config["pn1_source_sha256"]):
        pytest.skip("Pinned comparison requires the local PN1 main checkout")
    return comparison.pn1_sources(config)


def test_exact_pn1_threshold_function_and_replacement_semantics():
    helper, _ = baseline()
    assert helper["effective_threshold"](1, 0, .2) == 51
    assert helper["effective_threshold"](2, 0, .2) == 13107
    ids = helper["sample_indices_fixed"](np.arange(3), 8, np.random.default_rng(0))
    assert len(ids) == 8 and len(np.unique(ids)) <= 3


def test_legacy_decoder_discrepancies_on_format_defined_fixtures():
    _, old = baseline()
    pbd8 = bytes([0, 10, 33, 2])
    assert vaa3d.decode_pbd8(pbd8, 2) == bytes([10, 12])
    assert old.decode_pbd8(pbd8, 2) == bytes([10, 10])
    pbd16 = bytes([0, 10, 0, 32, 128])
    assert np.frombuffer(vaa3d.decode_pbd16(pbd16, 2), dtype="<u2").tolist() == [10, 14]
    assert np.frombuffer(old.decode_pbd16(pbd16, 2), dtype="<u2").tolist() == [10, 6]


def test_duplicate_slots_do_not_inflate_mass_or_point_information():
    common = np.array([[[5, 10, 15]]], dtype=np.uint16)
    ids = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    window = {"xyz": [1., 0., 0.], "tangent": [1., 0., 0.], "background": 0.,
              "arc_weight": 1., "width_class": "thin", "contrast_class": "dim",
              "domains": [{"available_positive_contrast_mass": 30.} for _ in range(4)]}
    result = comparison.score(ids, common.ravel()[ids], common, common.shape, [window], [],
                              {"radii_voxel": [1., 2., 4., 8.], "half_length_voxel": 2.})
    assert result["unique_count"] == 3 and result["duplicate_slots"] == 5
    assert result["contrast_fractions"] == [[1., 1., 1., 1.]]
    assert result["local_selected_point_counts"] == [[3, 3, 3, 3]]


def test_frozen_pps_prefix_is_same_lower_budget_selection():
    p, _ = probabilities(np.arange(1000), .1)
    large = ranked_selection(p, 512, 0, 1)
    small = ranked_selection(p, 128, 0, 1)
    assert np.array_equal(large[:128], small)


def test_curve_exit_and_reentry_do_not_merge_unsupported_runs():
    xyz = np.array([[0, 0, 0], [10, 0, 0], [0, 0, 0]], dtype=float)
    m = Morphology(None, np.arange(1, 4), np.ones(3, int), xyz, np.ones(3), np.array([-1, 0, 1]))
    runs = comparison.clipped_curve_samples(m, np.array([0, 0, 0]), np.array([4, 1, 1]))
    assert len(runs) == 2
    measured = comparison.geometry(np.empty((0, 3)), runs)
    assert measured["2.0"]["curve_support"] == 0
    assert measured["2.0"]["longest_unsupported_run"] == 3.5
