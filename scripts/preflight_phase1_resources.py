"""Header-only resource planning for an explicit Phase 1 case list; no filtering."""
from __future__ import annotations

import argparse
import hashlib
import json
from math import prod
from pathlib import Path
import sys

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.inputs.cases import load_image_cases, read_object
from pn2.inputs.sampling import partition_tiles, selection_count
from pn2.inputs.vaa3d import HEADER_SIZE, read_header
from probe_phase1_sampling import load_recipe


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(cases_path, recipe_path, baseline_path, output_path):
    cases = load_image_cases(cases_path)
    recipe = load_recipe(recipe_path)
    if any(c.partition != "training" or c.units != "voxel" for c in cases):
        raise ValueError("This qualification uses explicit native-voxel training cases")
    baseline = read_object(baseline_path)
    baseline_recipe = Path(baseline_path).resolve().parent / "recipe.json"
    if digest(baseline_recipe) != baseline["frozen_recipe_sha256"]:
        raise ValueError("Baseline recipe changed")
    if read_object(baseline_recipe) != recipe:
        raise ValueError("Resource projection requires the unchanged comparison recipe")
    # Per-case Windows peaks are cumulative within that process. These are
    # deliberately labelled proxies, not isolated per-case peaks or guarantees.
    observations = []
    for item in baseline["cases"]:
        memory = item["process_memory"]
        if memory["kind"] != "process_peak_working_set":
            raise ValueError("Expected measured Windows process peak working sets")
        voxels = prod(item["image"]["dimensions_xyzc"])
        observations.append({"case_id": item["case_id"], "voxels": voxels,
                             "peak_bytes_per_voxel_proxy": memory["bytes"] / voxels,
                             "guide_seconds_per_voxel": item["guide_seconds"] / voxels})
    rate = max(o["peak_bytes_per_voxel_proxy"] for o in observations)
    time_rates = [o["guide_seconds_per_voxel"] for o in observations]
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(Path(output_path).resolve().parent.anchor)
    rows = []
    for case in cases:
        header = read_header(case.image_path)
        if header.dimensions[3] != 1:
            raise ValueError("Expected scalar images")
        with case.image_path.open("rb") as stream:
            header_hash = hashlib.sha256(stream.read(HEADER_SIZE)).hexdigest()
        voxels = header.voxel_count
        tiles = list(partition_tiles(header.dimensions[:3][::-1], recipe["tile_side_voxel"]))
        counts = {str(cell): sum(selection_count(prod(s.stop-s.start for s in tile), cell)
                                for tile in tiles) for cell in recipe["cell_sides_voxel"]}
        peak_proxy = int(voxels * rate)
        # An explicit planning reserve, unrelated to any scientific pass gate.
        planned = int(1.25 * peak_proxy + 1024**3)
        points = sum(counts.values()) * len(recipe["seeds"]) * len(recipe["methods"])
        rows.append({"case_id": case.case_id, "image_path": str(case.image_path),
                     "source_header_sha256": header_hash,
                     "source_bytes": case.image_path.stat().st_size,
                     "dimensions_xyzc": list(header.dimensions), "datatype": header.datatype,
                     "decoded_bytes": voxels * header.bytes_per_voxel, "voxels": voxels,
                     "tile_count": len(tiles), "points_per_cloud_by_cell_side": counts,
                     "cloud_count": len(recipe["seeds"])*len(recipe["methods"])*len(counts),
                     "linear_peak_proxy_bytes": peak_proxy,
                     "planning_bytes_with_reserve": planned,
                     "planning_exceeds_installed_RAM": planned > memory.total,
                     "planning_exceeds_available_RAM": planned > memory.available,
                     "guide_seconds_linear_projection": [voxels*min(time_rates), voxels*max(time_rates)],
                     "guide_and_uncompressed_packet_bytes": 4*voxels + points*(32+header.bytes_per_voxel)})
    report = {"schema_version": 1,
              "scope": "43-byte source headers and existing run metadata only; no references, decode, filtering, sampling or training",
              "cases_sha256": digest(cases_path), "recipe_sha256": digest(recipe_path),
              "baseline_sha256": digest(baseline_path), "script_sha256": digest(__file__),
              "memory_snapshot": {"total_bytes": memory.total, "available_bytes": memory.available},
              "disk_free_bytes": disk.free, "baseline_observations": observations,
              "projection_limitations": "Linear planning heuristics from two smaller volumes, not measured requirements or bounds. Peaks are cumulative. Working set excludes paged-out allocations. Shape, allocator, concurrent use and reference complexity can change actual costs. Time covers guide only. Disk estimate excludes temporary/evaluation artifacts and compression overhead.",
              "planning_reserve": "1.25 times maximum observed peak-per-voxel proxy plus 1 GiB; resource planning only",
              "cases": rows,
              "launch_recommended_by_resource_check": all(r["planning_bytes_with_reserve"] <= memory.available for r in rows),
              "automatic_launch": False}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(f"{row['case_id']}: {row['voxels']/1e6:.2f} Mvox; "
              f"peak proxy {row['linear_peak_proxy_bytes']/1024**3:.2f} GiB; "
              f"planning {row['planning_bytes_with_reserve']/1024**3:.2f} GiB", flush=True)
    print(f"RESOURCE launch_recommended={report['launch_recommended_by_resource_check']}; report={output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    preflight(args.cases, args.recipe, args.baseline, args.output)
