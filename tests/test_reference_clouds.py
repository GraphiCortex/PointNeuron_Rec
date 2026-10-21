"""Image-derived reference controls; no torch, annotations or learned model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pn2.inputs.cases import ImageCase
from pn2.inputs.clouds import ReferenceCloudBuilder, ReferenceCloudConfig
from pn2.inputs.image import DecodedImage
from pn2.inputs.tiling import Tile, TileConfig, TileGrid


def image(values, identity="synthetic"):
    return DecodedImage(ImageCase(identity, Path("unused_image"), "synthetic"), np.asarray(values), {})


def test_threshold_uses_entire_image_signal_not_storage_dtype_or_tile_maximum():
    values = np.zeros((1, 2, 8), dtype=np.uint16)
    values[0, 0] = [4095, 820, 819, 100, 1000, 500, 0, 0]
    builder = ReferenceCloudBuilder(image(values), ReferenceCloudConfig(.2, 512, 0))
    assert builder.threshold == 819  # Not 0.2*65535 or per-tile 0.2*1000.
    first = builder.build(Tile(0, (0, 0, 0), (4, 2, 1)))
    second = builder.build(Tile(1, (4, 0, 0), (8, 2, 1)))
    assert first.metadata["foreground_voxels"] == 2
    assert second.metadata["threshold"] == 819
    assert second.points_xyzi.tolist() == [[4., 0., 0., 1000.]]
    assert second.metadata["cloud_status"] == "singleton_cloud"


@pytest.mark.parametrize("count", [0, 1, 2, 7])
def test_empty_and_short_inputs_preserve_actual_distinct_points_without_padding(count):
    values = np.zeros((1, 1, 8), dtype=np.uint8)
    values.reshape(-1)[:count] = 100
    cloud = ReferenceCloudBuilder(image(values), ReferenceCloudConfig(.2, 512, 0)).build(
        Tile(0, (0, 0, 0), (8, 1, 1)))
    assert cloud.points_xyzi.shape == (count, 4)
    assert cloud.selected_flat_indices.tolist() == list(range(count))
    assert cloud.metadata["selected_points"] == cloud.metadata["distinct_selected_voxels"] == count
    assert cloud.metadata["cloud_status"] == ("empty_mask" if count == 0 else "singleton_cloud" if count == 1 else "ready")


def test_global_coordinates_intensity_and_source_indices_survive_noncontiguous_crop():
    z, y, x = np.indices((4, 7, 9))
    values = (100 * z + 10 * y + x + 1).astype(np.uint16)
    tile = Tile(19, (3, 2, 1), (8, 6, 4))
    cloud = ReferenceCloudBuilder(image(values), ReferenceCloudConfig(0, 512, 1)).build(tile)
    xyz = cloud.points_xyzi[:, :3].astype(int)
    np.testing.assert_array_equal(cloud.points_xyzi[:, 3], values[xyz[:, 2], xyz[:, 1], xyz[:, 0]])
    np.testing.assert_array_equal(cloud.selected_flat_indices, (xyz[:, 2] * 7 + xyz[:, 1]) * 9 + xyz[:, 0])
    assert xyz.min(axis=0).tolist() == [3, 2, 1]
    assert xyz.max(axis=0).tolist() == [7, 5, 3]
    assert len(np.unique(cloud.selected_flat_indices)) == 60


def test_sampling_is_without_replacement_reproducible_and_call_order_independent():
    values = np.arange(1, 1 + 10 * 12 * 14, dtype=np.uint16).reshape(10, 12, 14)
    config = ReferenceCloudConfig(.2, 23, 17)
    builder = ReferenceCloudBuilder(image(values), config)
    tiles = list(TileGrid((14, 12, 10), TileConfig((9, 8, 7), (5, 5, 5))))
    a = builder.build(tiles[0]); builder.build(tiles[-1]); b = builder.build(tiles[0])
    np.testing.assert_array_equal(a.points_xyzi, b.points_xyzi)
    assert a.metadata == b.metadata
    assert a.metadata["selected_points"] == 23
    assert len(np.unique(a.selected_flat_indices)) == 23
    other = ReferenceCloudBuilder(image(values), ReferenceCloudConfig(.2, 23, 18)).build(tiles[0])
    assert not np.array_equal(other.selected_flat_indices, a.selected_flat_indices)


def test_empty_mask_from_high_cutoff_is_reported_without_lowering_threshold():
    builder = ReferenceCloudBuilder(image(np.full((1, 2, 3), 100, np.uint8)), ReferenceCloudConfig(1, 512, 0))
    cloud = builder.build(Tile(0, (0, 0, 0), (3, 2, 1)))
    assert cloud.metadata["threshold"] == 100
    assert cloud.metadata["cloud_status"] == "empty_mask"


@pytest.mark.parametrize("fraction,budget,seed", [(float("nan"), 2, 0), (-.1, 2, 0),
    (1.1, 2, 0), (True, 2, 0), (.2, 1, 0), (.2, 2., 0), (.2, 2, -1), (.2, 2, True)])
def test_invalid_cloud_config_is_rejected(fraction, budget, seed):
    with pytest.raises(ValueError):
        ReferenceCloudConfig(fraction, budget, seed)


def test_config_is_explicit_control_and_rejects_reference_or_sampler_fields(tmp_path):
    path = tmp_path / "config.json"
    data = ReferenceCloudConfig(.2, 512, 0).as_dict()
    path.write_text(json.dumps(data), encoding="utf-8")
    assert ReferenceCloudConfig.load(path) == ReferenceCloudConfig(.2, 512, 0)
    data["reference_swc"] = "must_not_be_opened.swc"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        ReferenceCloudConfig.load(path)


def test_invalid_crop_is_rejected():
    builder = ReferenceCloudBuilder(image(np.zeros((2, 3, 4), np.uint8)), ReferenceCloudConfig(.2, 512, 0))
    with pytest.raises(ValueError, match="outside"):
        builder.build(Tile(0, (0, 0, 0), (5, 3, 2)))
