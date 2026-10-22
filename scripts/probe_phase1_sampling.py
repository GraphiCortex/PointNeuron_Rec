"""Generate frozen reference-free clouds for the three-arm CPU input probe."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.inputs.cases import load_image_cases, read_object, require_keys
from pn2.inputs.image import decode_image
from pn2.inputs.disk_guide import disk_sato_guide, read_block
from pn2.inputs.sampling import (normalized_image, partition_tiles, probabilities,
                                 ranked_selection, sato_guide, selection_count, source_indices)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def write(path, data):
    # New files in a fresh run directory: no frequently replaced Windows status file.
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_recipe(path):
    recipe = read_object(Path(path))
    keys = {"schema_version", "sigmas_voxel", "tile_side_voxel", "cell_sides_voxel",
            "uniform_mass", "seeds", "methods", "evaluation"}
    require_keys(recipe, keys, keys, "probe recipe")
    if recipe["methods"] != ["uniform", "intensity", "sato"]:
        raise ValueError("This approved probe requires exactly the three declared controls")
    for key in ("cell_sides_voxel", "seeds"):
        values = recipe[key]
        if (not isinstance(values, list) or not values or len(set(values)) != len(values)
                or any(type(v) is not int or v < (0 if key == "seeds" else 1) for v in values)):
            raise ValueError(f"Invalid {key}")
    if type(recipe["tile_side_voxel"]) is not int or recipe["tile_side_voxel"] <= 0:
        raise ValueError("Invalid tile side")
    if not 0 < recipe["uniform_mass"] <= 1:
        raise ValueError("Invalid uniform mass")
    return recipe


def peak_memory():
    info = psutil.Process().memory_info()
    return {"bytes": int(getattr(info, "peak_wset", info.rss)),
            "kind": "process_peak_working_set" if hasattr(info, "peak_wset") else "current_RSS_not_peak",
            "current_RSS_bytes": int(info.rss),
            "peak_pagefile_bytes": int(info.peak_pagefile) if hasattr(info, "peak_pagefile") else None,
            "current_private_bytes": int(info.private) if hasattr(info, "private") else None}


def generate(cases_path, recipe_path, output_dir, *, guide_backend="whole", block_side=64):
    if guide_backend not in {"whole", "disk"}:
        raise ValueError("Unknown guide backend")
    if type(block_side) is not int or block_side <= 0:
        raise ValueError("block_side must be a positive integer")
    cases, recipe = load_image_cases(cases_path), load_recipe(recipe_path)
    if any(c.partition != "training" or c.units != "voxel" for c in cases):
        raise ValueError("This bounded native-voxel probe accepts explicit training images only")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write(output_dir / "recipe.json", recipe)
    report = {"schema_version": 1, "scope": "Input-support probe; no SWCs, model or training in generation",
              "cases_manifest_sha256": sha256(cases_path), "recipe_sha256": sha256(recipe_path),
              "frozen_recipe_sha256": sha256(output_dir / "recipe.json"),
              "versions": {p: importlib.metadata.version(p) for p in
                           ("numpy", "scipy", "scikit-image", "psutil")},
              "source_code_sha256": sha256(Path(__file__).resolve().parents[1] / "src/pn2/inputs/sampling.py"),
              "generation_script_sha256": sha256(__file__),
              "guide_backend": guide_backend,
              "disk_guide_code_sha256": sha256(Path(__file__).resolve().parents[1] / "src/pn2/inputs/disk_guide.py")
                  if guide_backend == "disk" else None,
              "cases": []}
    began = time.perf_counter()
    for case_number, case in enumerate(cases):
        folder = output_dir / f"case_{case_number:02d}"
        folder.mkdir()
        start = time.perf_counter()
        decoded = decode_image(case)
        raw = decoded.array_zyx
        decode_seconds = time.perf_counter() - start
        normalized, transform = normalized_image(raw)
        normalized_path = folder / "normalized.npy"
        guide_path = folder / "sato_guide.npy"
        if guide_backend == "disk":
            np.save(normalized_path, normalized, allow_pickle=False)
            normalized = None
        print(f"GUIDE start case={case.case_id} XYZ={decoded.dimensions_xyz} CPU only", flush=True)
        start = time.perf_counter()
        execution = {"backend": "whole_skimage_0.25.2"}
        if guide_backend == "disk":
            _, execution = disk_sato_guide(normalized_path, recipe["sigmas_voxel"],
                folder / "guide_work", block_side=block_side,
                progress=lambda s: print(s, flush=True), output_path=guide_path)
            guide_seconds = time.perf_counter() - start
            execution["timing_scope"] = "disk filtering including intermediate and guide IO; excludes input normalization/save"
            (folder / "guide_work").rmdir()
            guide = None
        else:
            guide = sato_guide(normalized, recipe["sigmas_voxel"])
            guide_seconds = time.perf_counter() - start
            execution["timing_scope"] = "in-memory filtering only; excludes guide save"
            np.save(guide_path, guide, allow_pickle=False)
        if guide_backend == "disk":
            guide_range = [float("inf"), 0.]
            for core in partition_tiles(raw.shape, block_side):
                block = read_block(guide_path, core)
                guide_range = [min(guide_range[0], float(block.min())), max(guide_range[1], float(block.max()))]
            del block
        else:
            guide_range = [float(guide.min()), float(guide.max())]
        tiles = list(partition_tiles(raw.shape, recipe["tile_side_voxel"]))
        item = {"case_id": case.case_id, "image": decoded.provenance,
                "normalization": transform, "decode_seconds": decode_seconds,
                "guide_seconds": guide_seconds, "guide_sha256": sha256(guide_path),
                "guide_execution": execution, "guide_range": guide_range,
                "tile_count": len(tiles), "tiles": [
                    {"start_xyz": [s.start for s in tile][::-1], "stop_xyz": [s.stop for s in tile][::-1]}
                    for tile in tiles], "outputs": []}
        print(f"GUIDE done seconds={guide_seconds:.2f} tiles={len(tiles)}", flush=True)
        for seed in recipe["seeds"]:
            start = time.perf_counter()
            selected = defaultdict(list)
            fallbacks = defaultdict(int)
            for tile_id, tile in enumerate(tiles):
                fields = {"uniform": np.ones(raw[tile].size),
                          "intensity": read_block(normalized_path, tile) if guide_backend == "disk" else normalized[tile],
                          "sato": read_block(guide_path, tile) if guide_backend == "disk" else guide[tile]}
                counts = {d: selection_count(int(raw[tile].size), d) for d in recipe["cell_sides_voxel"]}
                for method, values in fields.items():
                    probability, fallback = probabilities(values, recipe["uniform_mass"])
                    fallbacks[method] += int(fallback)
                    ranking = ranked_selection(probability, max(counts.values()), seed, tile_id)
                    for cell, count in counts.items():
                        flat, _ = source_indices(ranking[:count], tile, raw.shape)
                        selected[(method, cell)].append(flat)
            selection_seconds = time.perf_counter() - start
            for (method, cell), chunks in selected.items():
                flat = np.concatenate(chunks)
                if len(np.unique(flat)) != len(flat):
                    raise AssertionError("Duplicate source voxels; stop without padding or filtering")
                xyz = np.array(np.unravel_index(flat, raw.shape))[::-1].T
                target = folder / f"{method}_cell{cell}_seed{seed}.npz"
                np.savez_compressed(target, flat_indices=flat, xyz_native=xyz, intensity=raw.ravel()[flat])
                item["outputs"].append({"method": method, "cell_side_voxel": cell, "seed": seed,
                    "unique_count": len(flat), "zero_response_tiles": fallbacks[method],
                    "path": str(target.relative_to(output_dir)), "sha256": sha256(target)})
            print(f"SELECT seed={seed} six count-matched clouds seconds={selection_seconds:.2f}", flush=True)
            item.setdefault("selection_seconds_by_seed", {})[str(seed)] = selection_seconds
        item["process_memory"] = peak_memory()
        write(folder / "provenance.json", item)
        report["cases"].append(item)
        if guide_backend == "disk":
            normalized_path.unlink()
        del decoded, raw, normalized, guide, selected, chunks, fields, values
    report["total_seconds"] = time.perf_counter() - began
    report["process_memory"] = peak_memory()
    write(output_dir / "clouds.json", report)
    print(f"COMPLETED reference-free cloud generation: {output_dir / 'clouds.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--guide-backend", choices=["whole", "disk"], default="whole")
    parser.add_argument("--block-side", type=int, default=64)
    args = parser.parse_args()
    generate(args.cases, args.recipe, args.output_dir, guide_backend=args.guide_backend,
             block_side=args.block_side)
