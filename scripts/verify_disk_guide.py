"""CPU replay against one saved guide and all its Sato selections; no SWC access."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.inputs.cases import ImageCase, read_object
from pn2.inputs.disk_guide import disk_sato_guide, read_block
from pn2.inputs.image import decode_image
from pn2.inputs.sampling import (normalized_image, partition_tiles, probabilities,
                                 ranked_selection, selection_count, source_indices)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def memory_snapshot():
    info = psutil.Process().memory_info()
    return {key: int(getattr(info, key)) for key in
            ("rss", "peak_wset", "private", "peak_pagefile")}


def run(args):
    root = args.clouds.resolve().parent
    baseline = read_object(args.clouds)
    item = baseline["cases"][args.case_index]
    if digest(root / "recipe.json") != baseline["frozen_recipe_sha256"]:
        raise ValueError("Baseline recipe changed")
    recipe = read_object(root / "recipe.json")
    original = root / f"case_{args.case_index:02d}" / "sato_guide.npy"
    if digest(original) != item["guide_sha256"]:
        raise ValueError("Baseline guide changed")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    for path in [Path(__file__), Path(__file__).resolve().parents[1] / "src/pn2/inputs/disk_guide.py"]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    code_hashes = {"script_sha256": digest(snapshot / "verify_disk_guide.py"),
                   "backend_code_sha256": digest(snapshot / "disk_guide.py")}
    started = time.perf_counter()
    memory_samples = {"before_decode": memory_snapshot()}
    decoded = decode_image(ImageCase(item["case_id"], Path(item["image"]["image_path"]), "training"))
    if decoded.provenance != item["image"]:
        raise ValueError("Decoded source provenance changed")
    normalized, transform = normalized_image(decoded.array_zyx)
    if transform != item["normalization"]:
        raise ValueError("Normalization changed")
    shape = normalized.shape
    source = directory / "normalized.npy"
    np.save(source, normalized, allow_pickle=False)
    del normalized, decoded
    memory_samples["after_normalization_save"] = memory_snapshot()
    prepared = time.perf_counter()
    target, execution = disk_sato_guide(source, recipe["sigmas_voxel"], directory / "filter",
                                       block_side=args.block_side, progress=lambda s: print(s, flush=True))
    filtered = time.perf_counter()
    memory_samples["after_filter"] = memory_snapshot()
    different = 0
    maximum_error = 0.
    for core in partition_tiles(shape, args.block_side):
        a, b = read_block(original, core), read_block(target, core)
        different += int(np.count_nonzero(a != b))
        maximum_error = max(maximum_error, float(np.max(np.abs(a-b))))
    print(f"GUIDE parity unequal_voxels={different} max_absolute_error={maximum_error}", flush=True)
    selections = []
    for seed in recipe["seeds"]:
        collected = {cell: [] for cell in recipe["cell_sides_voxel"]}
        for tile_id, tile in enumerate(partition_tiles(shape, recipe["tile_side_voxel"])):
            values = read_block(target, tile)
            probability, _ = probabilities(values, recipe["uniform_mass"])
            counts = {cell: selection_count(values.size, cell) for cell in collected}
            selected = ranked_selection(probability, max(counts.values()), seed, tile_id)
            for cell, count in counts.items():
                flat, _ = source_indices(selected[:count], tile, shape)
                collected[cell].append(flat)
        for cell, chunks in collected.items():
            old = next(o for o in item["outputs"] if o["method"] == "sato" and
                       o["seed"] == seed and o["cell_side_voxel"] == cell)
            saved = root / old["path"]
            if digest(saved) != old["sha256"]:
                raise ValueError("Baseline cloud changed")
            with np.load(saved, allow_pickle=False) as packet:
                expected = packet["flat_indices"]
            actual = np.concatenate(chunks)
            equal = np.array_equal(expected, actual)
            selections.append({"seed": seed, "cell_side_voxel": cell, "count": len(actual),
                               "exact_ordered_source_indices": equal, "baseline_cloud_sha256": old["sha256"]})
        print(f"SELECTION parity seed={seed} dense/lean exact={all(s['exact_ordered_source_indices'] for s in selections)}", flush=True)
    info = psutil.Process().memory_info()
    memory_samples["after_sampling"] = memory_snapshot()
    report = {"schema_version": 1, "scope": "Implementation parity only; no reference, biological score, GPU or training",
              "case_id": item["case_id"], "versions": {p: importlib.metadata.version(p) for p in
                  ("numpy", "scipy", "scikit-image", "psutil")},
              "baseline_manifest_sha256": digest(args.clouds), **code_hashes,
              "code_hash_capture": "source snapshots before decoding/filtering",
              "source_image_sha256": item["image"]["source_sha256"], "execution": execution,
              "normalization": transform, "shape_zyx": list(shape), "unequal_guide_voxels": different,
              "maximum_absolute_guide_error": maximum_error,
              "baseline_guide_sha256": digest(original), "disk_guide_sha256": digest(target),
              "selections": selections, "prepare_seconds": prepared-started,
              "filter_seconds": filtered-prepared, "total_seconds": time.perf_counter()-started,
              "process_peak_working_set_bytes": int(info.peak_wset),
              "process_peak_pagefile_bytes": int(info.peak_pagefile),
              "memory_samples": memory_samples,
              "baseline_cumulative_peak_working_set_bytes": item["process_memory"]["bytes"],
              "memory_comparison_limit": "Fresh replay process includes decoding/normalization/filtering/selection. Historical per-case peaks were cumulative in a two-case process; this is not a controlled filter-only memory benchmark or a product minimum specification.",
              "passed": different == 0 and all(s["exact_ordered_source_indices"] for s in selections)}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"FINISHED parity passed={report['passed']} peak={info.peak_wset/1024**3:.3f} GiB report={directory/'report.json'}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds", required=True, type=Path)
    parser.add_argument("--case-index", required=True, type=int, choices=[0, 1])
    parser.add_argument("--block-side", type=int, default=64)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())
