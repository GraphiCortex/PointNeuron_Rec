"""Opaque reference metadata cannot weaken geometry or default output validation."""
import numpy as np
import pytest

from pn2.evaluation.swc import SwcValidationError, parse_swc


def test_signed_reference_types_preserved_verbatim_with_every_node(tmp_path):
    source = tmp_path / "reference.swc"
    payload = b"# custom metadata\n9 -73 1.25 -2 3 0 4\n4 77 0 0 0 1 -1\n"
    source.write_bytes(payload)
    with pytest.raises(SwcValidationError):
        parse_swc(source)
    morphology = parse_swc(source, type_policy="opaque_integer")
    assert morphology.ids.tolist() == [9, 4]
    assert morphology.types.tolist() == [-73, 77]
    assert morphology.parents.tolist() == [1, -1]
    np.testing.assert_array_equal(morphology.xyz, [[1.25, -2, 3], [0, 0, 0]])
    np.testing.assert_array_equal(morphology.radii, [0, 1])
    assert source.read_bytes() == payload


@pytest.mark.parametrize("rows", [
    "1 -7.3 0 0 0 1 -1\n",  # Integer syntax remains mandatory.
    "1 -9223372036854775809 0 0 0 1 -1\n",  # No integer overflow.
    "0 -73 0 0 0 1 -1\n",  # Positive node IDs remain mandatory.
    "1 -73 nan 0 0 1 -1\n",
    "1 -73 0 0 0 -1 -1\n",
    "1 -73 0 0 0 1 2\n",  # Missing parent.
    "1 -73 0 0 0 1 1\n",  # Self-parent.
    "1 -73 0 0 0 1 2\n2 77 1 0 0 1 1\n",  # Cycle.
    "1 -73 0 0 0 1 -1\n1 77 1 0 0 1 -1\n",  # Duplicate ID.
])
def test_opaque_metadata_keeps_all_other_validation(tmp_path, rows):
    source = tmp_path / "invalid.swc"
    source.write_text(rows)
    with pytest.raises(SwcValidationError):
        parse_swc(source, type_policy="opaque_integer")


def test_unknown_policy_rejected_before_read(tmp_path):
    with pytest.raises(SwcValidationError, match="Unknown SWC type policy"):
        parse_swc(tmp_path / "missing.swc", type_policy="auto")
