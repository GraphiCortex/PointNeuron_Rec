"""Read-only repeated-value/capacity accounting; no clouds, guides or references."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.inputs.cases import load_image_cases, read_object
from pn2.inputs.image import decode_image
from pn2.inputs.sampling import partition_tiles, selection_count


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def describe(values):
    """Count exact integer values; payload estimates exclude common metadata."""
    values = np.asarray(values)
    if (values.ndim != 3 or not values.size or values.dtype.kind != "u"
            or values.dtype.itemsize not in (1, 2) or values.size > 2**32):
        raise ValueError("Expected nonempty uint8/uint16 ZYX box fitting uint32 local indices")
    flat = values.ravel(order="C")
    histogram = np.bincount(flat.astype(np.int64), minlength=2**(8*values.dtype.itemsize))
    default = int(histogram.argmax())  # Smallest value wins a modal tie.
    exceptions = int(values.size - histogram[default])
    dense = int(values.nbytes)
    sparse = int(values.dtype.itemsize + exceptions*(4 + values.dtype.itemsize))
    return {"voxels": int(values.size), "dtype": values.dtype.str,
            "mode_value": default, "mode_count": int(histogram[default]),
            "exception_count": exceptions, "raw_zero_count": int(histogram[0]),
            "constant": exceptions == 0, "dense_payload_bytes": dense,
            "default_exception_payload_bytes": sparse,
            "smaller_payload_bytes": min(dense, sparse),
            "preferred_payload": "default_exceptions" if sparse < dense else "dense"}


def roundtrip(values, row):
    """In-memory representation proof, not a production serializer or model input."""
    flat = values.ravel(order="C")
    indices = np.flatnonzero(flat != row["mode_value"]).astype(np.uint32)
    exceptional_values = flat[indices]
    restored = np.full(flat.shape, row["mode_value"], dtype=values.dtype)
    restored[indices] = exceptional_values
    if len(indices) != row["exception_count"] or not np.array_equal(restored.reshape(values.shape), values):
        raise AssertionError("Default/exception reconstruction changed pixel values or layout")
    if len(indices)*4 + exceptional_values.nbytes + values.dtype.itemsize != row["default_exception_payload_bytes"]:
        raise AssertionError("Payload arithmetic disagrees with the declared array layout")


def self_check():
    bright_dark = np.full((2, 3, 5), 100, dtype=np.uint16)
    bright_dark[0, 0, 1], bright_dark[1, 2, 4] = 0, 65535
    cases = [
        ("constant_nonzero", np.full((2, 3, 5), 7, dtype=np.uint8), 7, 0, "default_exceptions"),
        ("bright_and_dark", bright_dark, 100, 2, "default_exceptions"),
        ("mode_tie", np.array([[[9, 2, 9, 2]]], dtype=np.uint8), 2, 2, "dense"),
        ("all_unique", np.arange(120, dtype=np.uint16).reshape(4, 5, 6), 0, 119, "dense"),
        ("noncontiguous_big_endian", bright_dark.astype(">u2")[:, ::-1, ::2], 100, 1, "default_exceptions"),
    ]
    results = []
    for name, values, expected_default, expected_count, expected_storage in cases:
        before = values.tobytes()
        row = describe(values)
        if (row["mode_value"], row["exception_count"], row["preferred_payload"]) != (
                expected_default, expected_count, expected_storage):
            raise AssertionError(f"Wrong accounting for {name}: {row}")
        roundtrip(values, row)
        if values.tobytes() != before:
            raise AssertionError("Fixture input was modified")
        results.append({"fixture": name, "passed": True, **row})
    for bad in (np.empty((0, 2, 3), dtype=np.uint8), np.ones((2, 3), dtype=np.uint8),
                np.ones((2, 3, 4), dtype=np.float32)):
        try:
            describe(bad)
        except ValueError:
            continue
        raise AssertionError("Invalid input was accepted")
    return {"roundtrip_fixtures": results, "invalid_inputs_rejected": 3, "passed": True}


def audit_case(case, expected, recipe):
    started = time.perf_counter()
    decoded = decode_image(case)
    raw = decoded.array_zyx
    if (decoded.provenance["source_sha256"] != expected["source_sha256"]
            or decoded.provenance["decoded_sha256"] != expected["decoded_sha256"]
            or case.image_path != Path(expected["image_path"]).resolve()):
        raise ValueError("Source differs from the frozen sampling provenance")
    decoded_seconds = time.perf_counter() - started
    tiles = []
    for index, tile in enumerate(partition_tiles(raw.shape, recipe["tile_side_voxel"])):
        values = raw[tile]
        row = describe(values)
        # Verify each real box too, with bounded per-box temporary arrays.
        roundtrip(values, row)
        row.update({"tile_id": index, "start_xyz": [s.start for s in tile[::-1]],
                    "stop_xyz": [s.stop for s in tile[::-1]],
                    "exceptions_alone_by_probe_budget": {str(d): {
                        "point_quota": selection_count(int(values.size), d),
                        "fits_without_context": row["exception_count"] <= selection_count(int(values.size), d),
                        "headroom_before_context": selection_count(int(values.size), d) - row["exception_count"]}
                        for d in recipe["cell_sides_voxel"]}})
        tiles.append(row)
    if sum(t["voxels"] for t in tiles) != raw.size:
        raise AssertionError("Box accounting does not cover the image")
    if hashlib.sha256(memoryview(raw).cast("B")).hexdigest() != decoded.provenance["decoded_sha256"]:
        raise AssertionError("Decoded source values changed during accounting")
    if digest(case.image_path) != decoded.provenance["source_sha256"]:
        raise ValueError("Source file changed during accounting")
    summary = {"boxes": len(tiles), "constant_boxes": sum(t["constant"] for t in tiles),
               "preferred_payload_counts": dict(Counter(t["preferred_payload"] for t in tiles)),
               "voxels": int(raw.size), "exceptions": sum(t["exception_count"] for t in tiles),
               "dense_payload_bytes": sum(t["dense_payload_bytes"] for t in tiles),
               "smaller_payload_bytes": sum(t["smaller_payload_bytes"] for t in tiles),
               "boxes_where_exceptions_alone_fit": {str(d): sum(
                   t["exceptions_alone_by_probe_budget"][str(d)]["fits_without_context"] for t in tiles)
                   for d in recipe["cell_sides_voxel"]}, "all_box_roundtrips_exact": True}
    return {"case_id": case.case_id, "image": decoded.provenance, "summary": summary,
            "decode_seconds": decoded_seconds, "total_seconds": time.perf_counter()-started, "tiles": tiles}


def run(args):
    checks = self_check()
    print("CHECK passed: five exact roundtrip fixtures; three invalid inputs rejected", flush=True)
    cases = [c for path in args.cases for c in load_image_cases(path)]
    if (len(cases) != 6 or any(c.partition != "training" or c.units != "voxel" for c in cases)
            or len({c.case_id for c in cases}) != len(cases)
            or len({c.image_path for c in cases}) != len(cases)):
        raise ValueError("Use exactly six unique explicit native-voxel training images")
    expected = {}
    for path in args.frozen_clouds:
        for row in read_object(path)["cases"]:
            if row["case_id"] in expected:
                raise ValueError("Duplicate frozen source identity")
            expected[row["case_id"]] = row["image"]
    if set(expected) != {c.case_id for c in cases}:
        raise ValueError("Case cohort must exactly match frozen source identities")
    recipe = read_object(args.recipe)
    if recipe["tile_side_voxel"] != 128 or recipe["cell_sides_voxel"] != [4, 8]:
        raise ValueError("This accounting check preserves the reviewed 128-box and 4/8 budgets")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    provenance = {"script_sha256": digest(__file__), "recipe_sha256": digest(args.recipe),
                  "manifests": {str(p): digest(p) for p in [*args.cases, *args.frozen_clouds]},
                  "input_code_sha256": {str(p.relative_to(root)): digest(p) for p in [
                      root/"src/pn2/inputs/image.py", root/"src/pn2/inputs/vaa3d.py",
                      root/"src/pn2/inputs/cases.py", root/"src/pn2/inputs/sampling.py"]}}
    write(args.output_dir/"launch.json", {"schema_version": 1, **provenance, "synthetic_checks": checks})
    started, rows = time.perf_counter(), []
    for index, case in enumerate(cases):
        print(f"READ {index+1}/{len(cases)} {case.case_id}", flush=True)
        row = audit_case(case, expected[case.case_id], recipe)
        rows.append(row)
        write(args.output_dir/f"case_{index:02d}.json", row)
        s = row["summary"]
        print(f"COUNT {index+1}/{len(cases)} boxes={s['boxes']} exceptions={s['exceptions']} "
              f"fit4/8={s['boxes_where_exceptions_alone_fit']} "
              f"smaller/dense={s['smaller_payload_bytes']}/{s['dense_payload_bytes']} "
              f"seconds={row['total_seconds']:.2f}", flush=True)
        gc.collect()
    if any(p == "torch" or p.startswith("pn2.evaluation") or p.startswith("skimage") for p in sys.modules):
        raise AssertionError("Unexpected model, reference evaluator or filter import")
    memory = psutil.Process().memory_info()
    result = {"schema_version": 1, "scope": "Read-only six-source accounting; no SWCs, guides, new clouds, model or training",
              **provenance, "recipe_comparators": {k: recipe[k] for k in ("tile_side_voxel", "cell_sides_voxel")},
              "payload_layout": "Dense raw voxel values versus one typed default value + uint32 local C-order indices + typed exception values. Common domain/type metadata, Python object/allocator overhead and compression excluded. Not a serialized format or process-RAM prediction.",
              "capacity_scope": "Exceptions alone fitting an old quota is a lower bound; excludes context, feature memory and all learned-model constraints",
              "interpretation": "Modal values/exceptions are numerical descriptions, not background/foreground. Exact source reconstruction does not establish finite-cloud sufficiency or reconstruction quality.",
              "synthetic_checks": checks, "cases": rows, "total_seconds": time.perf_counter()-started,
              "process_memory": {"peak_working_set_bytes": getattr(memory, "peak_wset", None),
                                 "peak_pagefile_bytes": getattr(memory, "peak_pagefile", None)},
              "completed": True, "automatic_cloud_generation": False, "automatic_training_authorized": False}
    write(args.output_dir/"report.json", result)
    print(f"COMPLETED accounting: {len(rows)} sources; report={args.output_dir/'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--frozen-clouds", type=Path, nargs="+", required=True,
                        help="Read image provenance only, never point arrays")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
