"""Independent format/geometry fixtures; no real images, annotations or models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pn2.inputs import vaa3d
from pn2.inputs.audit import audit_inputs
from pn2.inputs.cases import ImageCase, load_image_cases
from pn2.inputs.image import decode_image
from pn2.inputs.tiling import TileConfig, TileGrid


def header(dimensions=(5, 3, 2, 1), datatype=2, endian="L", compressed=False):
    # Format-defined bytes, independent of production header parsing.
    key = b"v3d_volume_pkbitdf_encod" if compressed else b"raw_image_stack_by_hpeng"
    order = "little" if endian == "L" else "big"
    return (key + endian.encode("ascii") + datatype.to_bytes(2, order)
            + b"".join(d.to_bytes(4, order) for d in dimensions))


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def raw(path, values, endian="L"):
    depth, height, width = values.shape
    datatype = values.dtype.itemsize
    dtype = np.dtype("u1" if datatype == 1 else "<u2" if endian == "L" else ">u2")
    path.write_bytes(header((width, height, depth, 1), datatype, endian) + values.astype(dtype).tobytes())
    return path


def case(path, name="synthetic"):
    return ImageCase(name, path, "synthetic")


def fixture_manifests(tmp_path, rows=None):
    if rows is None:
        raw(tmp_path / "image.v3draw", np.arange(30, dtype=np.uint8).reshape(2, 3, 5))
        rows = [{"case_id": "synthetic", "image_path": "image.v3draw", "partition": "synthetic",
                 "coordinates": {"mode": "native", "units": "voxel"}}]
    manifest = write_json(tmp_path / "images.json", {"schema_version": 1, "cases": rows})
    config = write_json(tmp_path / "tiling.json", {"schema_version": 1,
                         "tile_size_xyz": [3, 2, 2], "stride_xyz": [2, 1, 2]})
    return manifest, config


@pytest.mark.parametrize("endian", ["L", "B"])
@pytest.mark.parametrize("datatype", [1, 2])
def test_raw_xyz_axes_byte_order_intensity_and_exact_source_provenance(tmp_path, endian, datatype):
    z, y, x = np.indices((2, 3, 5))
    values = (100 * z + 10 * y + x).astype(np.uint8 if datatype == 1 else np.uint16)
    path = raw(tmp_path / "image.v3draw", values, endian)
    decoded = decode_image(ImageCase("asymmetric", path, "synthetic", "um", (2., 3., 4.)))
    np.testing.assert_array_equal(decoded.array_zyx, values)
    assert decoded.dimensions_xyz == (5, 3, 2)
    assert decoded.array_zyx[1, 2, 4] == 124
    assert decoded.provenance["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert decoded.provenance["decoded_bytes"] == 30 * datatype
    assert decoded.provenance["spacing_xyz"] == [2., 3., 4.]
    assert decoded.provenance["native_coordinates_units"] == "voxel"
    assert decoded.provenance["declared_metric_units"] == "um"
    assert decoded.provenance["intensity_transform"] == decoded.provenance["spatial_resampling"] == "none"
    with pytest.raises(ValueError):
        decoded.array_zyx[0, 0, 0] = 99
    # New loader agrees with retained loader while hashing the actual byte snapshot.
    assert decoded.provenance["decoded_sha256"] == hashlib.sha256(vaa3d.read_volume(path).data).hexdigest()


def test_pbd8_all_packed_bytes_partial_groups_and_carry():
    # Vaa3D low-two-bits-first codes are 0,+1,+2,-1. Include the next run's carry.
    for packed in range(256):
        for length in (1, 2, 3, 4):
            expected = [100]
            for index in range(length):
                code = (packed // 4 ** index) % 4
                expected.append((expected[-1] + (0, 1, 2, -1)[code]) % 256)
            expected.append((expected[-1] + 2) % 256)
            stream = bytes([0, 100, 32 + length, packed, 33, 2])
            assert vaa3d.decode_pbd8(stream, length + 2) == bytes(expected)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 94, 95])
def test_pbd8_full_run_boundaries_and_unsigned_wrap(length):
    codes = ([1, 2, 3, 0] * 24)[:length]
    packed = [sum(code * 4 ** j for j, code in enumerate(codes[i:i + 4])) for i in range(0, length, 4)]
    expected = [255]
    for code in codes:
        expected.append((expected[-1] + (0, 1, 2, -1)[code]) % 256)
    assert vaa3d.decode_pbd8(bytes([0, 255, 32 + length] + packed), length + 1) == bytes(expected)


def test_pbd8_literal_repeat_and_loader_dispatch(tmp_path):
    path = tmp_path / "compressed.v3dpbd"
    path.write_bytes(header((7, 1, 1, 1), 1, compressed=True)
                     + bytes([0, 10, 36, 0b00111001, 129, 42]))
    expected = np.array([10, 11, 13, 12, 12, 42, 42], dtype=np.uint8).reshape(1, 1, 7)
    np.testing.assert_array_equal(decode_image(case(path)).array_zyx, expected)


@pytest.mark.parametrize("order", ["little", "big"])
@pytest.mark.parametrize("length", [1, 2, 3, 8, 47, 48])
def test_pbd16_independent_code_mapping_and_byte_boundaries(tmp_path, order, length):
    codes = ([0, 1, 2, 3, 4, 5, 6, 7] * 6)[:length]
    bits = "".join(format(code, "03b") for code in codes)
    bits += "0" * ((-len(bits)) % 8)
    packed = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    expected = [65535]
    for code in codes:
        expected.append((expected[-1] + (0, 1, 2, 3, 4, -1, -2, -3)[code]) % 65536)
    stream = bytes([0]) + (65535).to_bytes(2, order) + bytes([31 + length]) + packed
    decoded = vaa3d.decode_pbd16(stream, length + 1, order)
    assert decoded == b"".join(value.to_bytes(2, order) for value in expected)
    path = tmp_path / "compressed.v3dpbd"
    path.write_bytes(header((length + 1, 1, 1, 1), 2, "L" if order == "little" else "B", True) + stream)
    assert decode_image(case(path)).array_zyx.reshape(-1).tolist() == expected


@pytest.mark.parametrize("control", [80, 182, 183, 222])
def test_pbd16_unverified_extensions_remain_explicitly_unsupported(tmp_path, control):
    path = tmp_path / "unsupported.v3dpbd"
    path.write_bytes(header((1, 1, 1, 1), 2, compressed=True) + bytes([control, 0]))
    with pytest.raises(NotImplementedError, match="verified decoder"):
        decode_image(case(path))


@pytest.mark.parametrize("stream,expected", [(b"", 1), (bytes([0]), 1),
    (bytes([0, 10, 33]), 2), (bytes([128]), 1), (bytes([0, 10, 0]), 1)])
def test_corrupt_pbd8_streams_fail(stream, expected):
    with pytest.raises(ValueError):
        vaa3d.decode_pbd8(stream, expected)


@pytest.mark.parametrize("payload,error", [
    (b"short", ValueError),
    (b"x" * 43, NotImplementedError),
    (header((1, 1, 1, 2), 1) + b"12", NotImplementedError),
    (header((1, 1, 1, 1), 4) + b"1234", NotImplementedError),
    (header((0, 1, 1, 1), 1), ValueError),
    (header((1, 1, 1, 1), 1) + b"12", ValueError),
    (header((1, 1, 1, 1), 2) + b"1", ValueError),
])
def test_bad_or_unsupported_images_are_not_silently_reinterpreted(tmp_path, payload, error):
    path = tmp_path / "input.bin"
    path.write_bytes(payload)
    with pytest.raises(error):
        decode_image(case(path))


@pytest.mark.parametrize("dimensions,size,stride", [
    ((1, 1, 1), (97, 97, 97), (48, 48, 48)),
    ((5, 3, 2), (97, 97, 97), (48, 48, 48)),
    ((11, 7, 5), (4, 4, 4), (3, 2, 4)),
    ((10, 4, 4), (4, 4, 4), (3, 4, 4)),
    ((9, 5, 3), (1, 1, 1), (1, 1, 1)),
    ((11, 13, 17), (4, 5, 6), (4, 5, 6)),
])
def test_all_voxels_and_final_boundary_are_covered_without_padding(dimensions, size, stride):
    grid = TileGrid(dimensions, TileConfig(size, stride))
    cover = np.zeros(dimensions[::-1], dtype=np.int32)
    tiles = list(grid)
    assert len(tiles) == grid.tile_count
    assert tiles == list(grid)
    assert [tile.tile_id for tile in tiles] == list(range(len(tiles)))
    assert len({tile.start_xyz for tile in tiles}) == len(tiles)
    for tile in tiles:
        cover[tile.slices_zyx] += 1
        assert all(0 <= a < b <= dim for a, b, dim in zip(tile.start_xyz, tile.stop_xyz, dimensions))
        assert tile.shape_xyz == tuple(min(dim, s) for dim, s in zip(dimensions, size))
    assert cover.min() >= 1
    assert tiles[0].start_xyz == (0, 0, 0)
    assert tiles[-1].stop_xyz == dimensions


def test_tiling_x_fastest_and_native_coordinate_transform():
    grid = TileGrid((8, 8, 8), TileConfig((4, 4, 4), (3, 3, 3)))
    tiles = list(grid)
    assert [tile.start_xyz for tile in tiles[:4]] == [(0, 0, 0), (3, 0, 0), (4, 0, 0), (0, 3, 0)]
    tile = tiles[-1]
    z, y, x = np.indices((8, 8, 8))
    values = 100 * z + 10 * y + x
    cropped = values[tile.slices_zyx]
    for lz, ly, lx in np.ndindex(cropped.shape):
        gx, gy, gz = np.array([lx, ly, lz]) + tile.start_xyz
        assert cropped[lz, ly, lx] == 100 * gz + 10 * gy + gx
    assert TileGrid((97, 97, 97), TileConfig((97, 97, 97), (48, 48, 48))).__iter__().__next__().center_xyz == (48., 48., 48.)


@pytest.mark.parametrize("size,stride", [((0, 4, 4), (1, 1, 1)), ((4, 4, 4), (5, 4, 4)),
    ((4., 4, 4), (1, 1, 1)), ((True, 4, 4), (1, 1, 1)), ((4, 4, 4), (0, 1, 1))])
def test_gapped_or_invalid_tiling_is_rejected(size, stride):
    with pytest.raises(ValueError):
        TileConfig(size, stride)


def test_count_is_known_without_constructing_volume_or_tile_list():
    assert TileGrid((10**9, 1, 1), TileConfig((1, 1, 1), (1, 1, 1))).tile_count == 10**9


@pytest.mark.parametrize("updates", [{"partition": []}, {"spacing_xyz": (0, 1, 1)},
    {"spacing_xyz": (float("nan"), 1, 1)}, {"spacing_xyz": (True, 1, 1)},
    {"units": "voxel", "spacing_xyz": (2, 1, 1)}, {"units": "arbitrary"}])
def test_direct_image_api_also_validates_case_coordinates(tmp_path, updates):
    arguments = {"case_id": "a", "image_path": tmp_path / "a.bin", "partition": "synthetic"}
    arguments.update(updates)
    with pytest.raises(ValueError):
        ImageCase(**arguments)


def test_physical_manifest_preserves_native_boxes_and_does_not_resample(tmp_path):
    manifest, _ = fixture_manifests(tmp_path)
    data = json.loads(manifest.read_text())
    data["cases"][0]["coordinates"] = {"mode": "spacing", "units": "um", "spacing_xyz": [2, 3, 4]}
    write_json(manifest, data)
    image_case = load_image_cases(manifest)[0]
    assert image_case.spacing_xyz == (2., 3., 4.)
    image = decode_image(image_case)
    assert image.dimensions_xyz == (5, 3, 2)
    assert image.array_zyx.shape == (2, 3, 5)
    assert image.provenance["spatial_resampling"] == "none"


def test_duplicate_and_nonfinite_manifest_fields_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1,"cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_image_cases(path)
    path.write_text('{"schema_version":1,"cases":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite JSON value"):
        load_image_cases(path)


@pytest.mark.parametrize("partition", ["reserved", "test", "validation"])
def test_reserved_roles_rejected_before_image_access(tmp_path, partition):
    rows = [{"case_id": "protected", "image_path": "never_opened.bin", "partition": partition,
             "coordinates": {"mode": "native", "units": "voxel"}}]
    manifest, _ = fixture_manifests(tmp_path, rows)
    with pytest.raises(ValueError, match="only training/development/synthetic"):
        load_image_cases(manifest)


def test_reference_keys_and_hidden_tile_caps_are_rejected_before_output_creation(tmp_path):
    manifest, config = fixture_manifests(tmp_path)
    data = json.loads(manifest.read_text())
    data["cases"][0]["reference_swc"] = "not_to_be_read.swc"
    write_json(manifest, data)
    with pytest.raises(ValueError, match="allowed"):
        audit_inputs(manifest, config, tmp_path / "no_output")
    assert not (tmp_path / "no_output").exists()
    del data["cases"][0]["reference_swc"]
    write_json(manifest, data)
    tiling = json.loads(config.read_text())
    tiling["max_tiles"] = 8
    write_json(config, tiling)
    with pytest.raises(ValueError, match="allowed"):
        audit_inputs(manifest, config, tmp_path / "no_output")
    assert not (tmp_path / "no_output").exists()


def test_signal_inventory_keeps_all_cases_and_does_not_open_annotations(tmp_path, monkeypatch):
    raw(tmp_path / "zero.v3draw", np.zeros((1, 2, 3), dtype=np.uint8))
    raw(tmp_path / "signal.v3draw", np.array([0, 0, 1, 2, 0, 0], dtype=np.uint8).reshape(1, 2, 3))
    (tmp_path / "unsupported.v3dpbd").write_bytes(header((1, 1, 1, 1), 2, compressed=True) + bytes([80, 0]))
    (tmp_path / "bad.bin").write_bytes(b"broken")
    (tmp_path / "unrelated.swc").write_text("malformed reference", encoding="utf-8")
    original_read = Path.read_bytes
    def guarded_read(path):
        assert path.suffix != ".swc", "Input stage must never open an annotation"
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    rows = [{"case_id": name, "image_path": name + suffix, "partition": "synthetic",
             "coordinates": {"mode": "native", "units": "voxel"}}
            for name, suffix in [("zero", ".v3draw"), ("signal", ".v3draw"),
                                 ("missing", ".v3draw"), ("unsupported", ".v3dpbd"), ("bad", ".bin")]]
    manifest, config = fixture_manifests(tmp_path, rows)
    report = audit_inputs(manifest, config, tmp_path / "audit")
    assert report["status"] == "failed"
    assert report["case_status_counts"] == {"decoded_and_tiled": 2, "failed": 2, "unsupported": 1}
    assert len(report["cases"]) == 5
    assert report["cases"][0]["tile_signal_counts"] == {"all_zero": 1}
    assert report["cases"][1]["tile_signal_counts"] == {"nonzero_signal": 1}
    assert all(row["model_processed_tiles"] == 0 for row in report["cases"])
    assert report["settings"]["threshold"] is None
    tile = json.loads((tmp_path / "audit/case_0001_tiles.jsonl").read_text())
    assert tile["nonzero_voxels"] == 2
    assert tile["global_xyz_origin"] == [0, 0, 0]
    assert tile["shape_xyz"] == [3, 2, 1]
    json.dumps(report, allow_nan=False)


def test_cli_cpu_execution_artifacts_and_overwrite_refusal(tmp_path):
    manifest, config = fixture_manifests(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/audit_image_inputs.py"
    command = [sys.executable, str(script), "--cases", str(manifest), "--tiling", str(config),
               "--output-dir", str(tmp_path / "audit")]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "no foreground selection or model executed" in first.stdout
    report_path = tmp_path / "audit/report.json"
    saved = report_path.read_bytes()
    report = json.loads(saved)
    row = report["cases"][0]
    assert row["tiles_written"] == row["tiles_expected"] == 4
    assert len((tmp_path / "audit" / row["tiles_file"]).read_text().splitlines()) == 4
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert report_path.read_bytes() == saved


def test_input_module_imports_do_not_load_models_or_reference_evaluation():
    source = Path(__file__).resolve().parents[1] / "src"
    program = (f"import sys; sys.path.insert(0,{str(source)!r}); import pn2.inputs.audit; "
               "assert 'torch' not in sys.modules; assert 'pn2.evaluation.swc' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
