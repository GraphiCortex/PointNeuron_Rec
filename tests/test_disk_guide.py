"""Exact operator parity, including seams, reflect edges and sampling effects."""
import numpy as np
import pytest

from pn2.inputs.disk_guide import disk_sato_guide
from pn2.inputs.sampling import probabilities, ranked_selection, sato_guide


@pytest.mark.parametrize("shape,side,sigmas", [
    ((7, 9, 11), 4, [1., 2., 4.]),
    ((1, 5, 13), 3, [1., 4.]),
    ((17, 15, 19), 8, [.8, 1.01, 2.]),
    ((9, 11, 13), 32, [1., 2., 4.]),
])
def test_random_guide_and_selected_indices_are_exact(tmp_path, shape, side, sigmas):
    image = np.random.default_rng(23).random(shape, dtype=np.float32)
    source = tmp_path / "normalized.npy"
    np.save(source, image)
    target, metadata = disk_sato_guide(source, sigmas, tmp_path / "work", block_side=side)
    reference = sato_guide(image, sigmas)
    actual = np.load(target)
    np.testing.assert_array_equal(actual, reference)
    for seed in range(5):
        a, _ = probabilities(reference, .1)
        b, _ = probabilities(actual, .1)
        np.testing.assert_array_equal(ranked_selection(a, min(50, image.size), seed, 0),
                                      ranked_selection(b, min(50, image.size), seed, 0))
    assert metadata["backend"] == "disk_separable_v1"
    assert list(target.parent.iterdir()) == [target]
    np.testing.assert_array_equal(np.load(source), image)


@pytest.mark.parametrize("kind", ["constant", "boundary_impulses", "seam_tube"])
def test_structured_fixtures_all_boundaries_and_two_block_sizes(tmp_path, kind):
    shape = (17, 19, 21)
    image = np.zeros(shape, dtype=np.float32)
    if kind == "constant":
        image[:] = .5
    elif kind == "boundary_impulses":
        image[0, 0, 0] = 1
        image[-1, -1, -1] = .7
        image[0, 8, 8] = .4
    else:
        z, y, x = np.indices(shape)
        image = np.exp(-((z-8)**2+(y-8)**2)/8).astype(np.float32)
    source = tmp_path / "normalized.npy"
    np.save(source, image)
    reference = sato_guide(image, [1., 2., 4.])
    for side in [7, 16]:
        target, _ = disk_sato_guide(source, [1., 2., 4.], tmp_path / f"work{side}", block_side=side)
        np.testing.assert_array_equal(np.load(target), reference)


def test_invalid_inputs_and_existing_work_directory_stop(tmp_path):
    source = tmp_path / "input.npy"
    np.save(source, np.zeros((3, 4, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="float32"):
        disk_sato_guide(source, [1.], tmp_path / "bad")
    np.save(source, np.zeros((3, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="scales"):
        disk_sato_guide(source, [float("nan")], tmp_path / "bad")
    tmp_path.joinpath("existing").mkdir()
    with pytest.raises(FileExistsError):
        disk_sato_guide(source, [1.], tmp_path / "existing")
    protected = tmp_path / "protected.npy"
    protected.write_bytes(b"existing artifact")
    with pytest.raises(FileExistsError):
        disk_sato_guide(source, [1.], tmp_path / "unused", output_path=protected)
    assert protected.read_bytes() == b"existing artifact"
