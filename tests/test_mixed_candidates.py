import numpy as np
import pytest

from pointneuron.data.mixed_candidates import mixed_candidate_indices


def test_bright_allocation_survives_a_large_permissive_population():
    bright = np.arange(20)
    permissive = np.arange(10000)
    chosen = mixed_candidate_indices(bright, permissive, 32, 4)
    assert len(chosen) == len(np.unique(chosen)) == 32
    assert np.isin(chosen, bright).sum() == 16
    assert np.isin(chosen, permissive).all()
    np.testing.assert_array_equal(chosen, mixed_candidate_indices(bright, permissive, 32, 4))


def test_overlap_is_counted_once_and_bright_voxels_outside_noise_are_eligible():
    bright = np.arange(10)
    permissive = np.arange(5, 15)
    chosen = mixed_candidate_indices(bright, permissive, 15, 3)
    np.testing.assert_array_equal(chosen, np.arange(15))


def test_shortage_transfers_slots_without_padding_a_large_union():
    chosen = mixed_candidate_indices(np.arange(2), np.arange(20), 10, 5)
    assert len(np.unique(chosen)) == 10 and np.isin(chosen, [0, 1]).sum() == 2
    chosen = mixed_candidate_indices(np.arange(20), np.arange(2), 10, 5)
    assert len(np.unique(chosen)) == 10


def test_small_union_is_preserved_before_padding_and_empty_stays_empty():
    chosen = mixed_candidate_indices(np.array([1]), np.array([1, 5, 8]), 10, 0)
    np.testing.assert_array_equal(np.unique(chosen), [1, 5, 8])
    assert len(chosen) == 10
    empty = np.empty(0, dtype=int)
    assert len(mixed_candidate_indices(empty, empty, 10, 0)) == 0
    assert len(np.unique(mixed_candidate_indices(empty, np.arange(20), 10, 0))) == 10


@pytest.mark.parametrize('fraction', [-.1, 1.1, np.nan])
def test_invalid_fractions_are_rejected(fraction):
    with pytest.raises(ValueError):
        mixed_candidate_indices(np.arange(5), np.arange(10), 8, 0, fraction)


def test_invalid_indices_and_budgets_are_rejected():
    for indices in (np.array([1, 1]), np.array([-1]), np.array([1.5])):
        with pytest.raises(ValueError):
            mixed_candidate_indices(indices, np.arange(10), 8, 0)
    with pytest.raises(ValueError):
        mixed_candidate_indices(np.arange(5), np.arange(10), 0, 0)
