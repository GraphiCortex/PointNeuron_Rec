import numpy as np
import pytest

from pointneuron.data.foreground import fractional_intensity_threshold
from scripts.aggregate_proposals import effective_threshold


def test_uint16_storage_of_12_bit_signal_uses_actual_image_range():
    image = np.array([0, 100, 1000, 4095], dtype=np.uint16)
    assert fractional_intensity_threshold(2, .2, mode='image_max_fraction', data=image) == 819
    assert fractional_intensity_threshold(2, .2) == 13107
    assert effective_threshold(2, 0, .2, mode='image_max_fraction', data=image) == 819


def test_legacy_uint8_threshold_preserved_and_source_intensities_untouched():
    image = np.array([0, 12, 255], dtype=np.uint8)
    before = image.copy()
    assert fractional_intensity_threshold(1, .2, mode='image_max_fraction', data=image) == 51
    np.testing.assert_array_equal(image, before)
    assert effective_threshold(1, 7, None) == 7


@pytest.mark.parametrize('image', [[], [float('nan')], [float('inf')], [-1, 2]])
def test_invalid_image_range_rejected(image):
    with pytest.raises(ValueError):
        fractional_intensity_threshold(2, .2, mode='image_max_fraction', data=image)


@pytest.mark.parametrize('fraction', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        fractional_intensity_threshold(1, fraction)
