"""Matched input-component comparison, not a PN1/PN2 model benchmark.

Seven fixed preparation arms, two requested budgets, four training boxes and
five seeds. PN1 pure selection helpers/decoder are reused from pinned main.
SWCs/windows enter scoring only. No learned model, crop scheduler or GPU.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pn2.inputs.cases import ImageCase
from pn2.inputs.image import decode_image
from pn2.evaluation.swc import parse_swc
from pn2.evaluation.cloud_support import branch_paths, polyline_samples, longest_run
from audit_real_branch_evidence import summaries
from validate_real_branch_counts import point_stats


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write("\n")


def pn1_sources(config):
    main = ROOT.parent
    for name, expected in config["pn1_source_sha256"].items():
        if digest(main / name) != expected:
            raise ValueError(f"PN1 source changed: {name}")
    # Compile these exact source functions, avoiding model imports or CLI scans.
    names = {"sample_indices_fixed", "spatially_balanced_sample_indices", "brightest_cell_representatives",
             "spatially_diverse_indices", "flat_indices_to_xyz"}
    tree = ast.parse((main / "src/pointneuron/data/training_cache.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("PN1 helper definitions missing")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_pn1_helpers", "exec"), namespace)
    tree = ast.parse((main / "scripts/aggregate_proposals.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "effective_threshold")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_pn1_threshold", "exec"), namespace)
    path = main / "src/pointneuron/data/vaa3d_raw.py"
    spec = importlib.util.spec_from_file_location("pinned_pn1_decoder", path)
    reader = importlib.util.module_from_spec(spec); sys.modules[spec.name] = reader
    sys.dont_write_bytecode = True
    spec.loader.exec_module(reader)
    return namespace, reader


def clipped_curve_samples(morphology, low, high):
    runs = []
    for path in branch_paths(morphology):
        points, weights = polyline_samples(morphology.xyz[path], .5)
        inside = ((points >= low-.5) & (points < high-.5)).all(1)
        ids = np.flatnonzero(inside)
        if len(ids):
            for part in np.split(ids, np.flatnonzero(np.diff(ids) > 1)+1):
                runs.append((points[part], weights[part]))
    return runs


def geometry(points, runs):
    tree = cKDTree(points)
    distances = [tree.query(p)[0] for p, _ in runs]
    total = sum(float(w.sum()) for _, w in runs)
    result = {}
    for radius in (2., 6.):
        supported = sum(float(w[d <= radius].sum()) for d, (_, w) in zip(distances, runs))
        result[str(radius)] = {"curve_support": supported/total if total else None,
                              "longest_unsupported_run": max((longest_run(d > radius, w) for d, (_, w) in zip(distances, runs)), default=None)}
    return result


def fidelity(legacy, verified):
    changed, absolute, maximum = 0, 0, 0
    for start in range(0, legacy.size, 1048576):
        a = legacy.ravel()[start:start+1048576].astype(np.int32)
        b = verified.ravel()[start:start+1048576].astype(np.int32)
        error = np.abs(a-b)
        changed += int(np.count_nonzero(error)); absolute += int(error.sum()); maximum = max(maximum, int(error.max()))
    return {"voxels": int(legacy.size), "changed_voxel_fraction": changed/legacy.size,
            "mean_absolute_intensity_error": absolute/legacy.size, "maximum_absolute_intensity_error": maximum,
            "pn1_range": [int(legacy.min()), int(legacy.max())], "verified_range": [int(verified.min()), int(verified.max())]}


def score(ids, values, common, shape, windows, runs, config):
    unique, first = np.unique(ids, return_index=True)
    points = np.array(np.unravel_index(unique, shape))[::-1].T
    actual = common.ravel()[unique]
    stats = point_stats(points, actual, windows, config["radii_voxel"], config["half_length_voxel"])
    fractions = [[row["retained_mass_fraction"] for row in w] for w in stats]
    error = np.abs(values[first].astype(float)-actual.astype(float))
    return {"unique_count": len(unique), "duplicate_slots": len(ids)-len(unique),
            "selected_attribute_MAE_against_verified": float(error.mean()) if len(error) else None,
            "selected_attribute_exact_fraction": float(np.mean(error == 0)) if len(error) else None,
            "contrast_fractions": fractions,
            "contrast_strata": summaries([{"metadata": w} for w in windows], fractions, config["radii_voxel"]),
            "curve_metrics": geometry(points, runs),
            "local_selected_point_counts": [[row["selected_points"] for row in w] for w in stats]}


def run(config_path, output):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    prior_path = ROOT / config["window_report"]
    if digest(prior_path) != config["window_report_sha256"]:
        raise ValueError("Fixed measurement windows changed")
    prior = json.loads(prior_path.read_text())
    fixed = json.loads((prior_path.parent / "config.json").read_text())
    assert config["seeds"] == fixed["seeds"] == [0, 1, 2, 3, 4]
    assert config["radii_voxel"] == fixed["radii_voxel"] and config["half_length_voxel"] == fixed["half_length_voxel"]
    assert config["primary_slots"] == 2048 and config["curve_tolerances_voxel"] == [2., 6.]
    assert config["curve_quadrature_step_voxel"] == .5
    helper, reader = pn1_sources(config)
    directory = Path(output).resolve(); directory.mkdir(parents=True, exist_ok=False)
    write(directory / "config.json", config)
    inventory_path = ROOT / "configs/training_inventory.json"
    inventory = json.loads(inventory_path.read_text())["cases"]
    report = {"scope": "Input-component evidence comparison on fixed training boxes, not learned reconstruction or published PN1 validation",
              "config_sha256": digest(config_path), "script_sha256": digest(__file__), "cases": []}
    started = time.perf_counter()
    for case in prior["cases"]:
        alias = case["alias"]; folder = directory / alias; folder.mkdir()
        record = next(c for c in inventory if c["case_id"] == case["case_id"])
        assert record["partition"] == "training"
        image_path = (inventory_path.parent / record["image_path"]).resolve()
        print(f"DECODE {alias}: compare PN1 and verified readers on identical source bytes", flush=True)
        decoded = decode_image(ImageCase(case["case_id"], image_path, "training"))
        common = decoded.array_zyx
        assert decoded.provenance["source_sha256"] == case["source_sha256"]
        old = reader.read_volume(image_path)
        dtype = np.dtype("u1" if old.header.datatype == 1 else "<u2" if old.header.endian == "L" else ">u2")
        legacy = np.frombuffer(old.data, dtype=dtype).reshape(common.shape)
        assert old.dimensions[:3] == common.shape[::-1]
        fn = helper["effective_threshold"]
        storage_threshold = fn(old.header.datatype, 0, config["threshold_fraction"])
        range_threshold = int(round(float(common.max())*config["threshold_fraction"]))
        ref = (inventory_path.parent / record["reference_swc"]).resolve()
        assert digest(ref) == case["reference_sha256"]
        morphology = parse_swc(ref, type_policy="opaque_integer")
        spec = next(s for s in fixed["cases"] if s["alias"] == alias)
        manifest = ROOT / spec["frozen_run"] / "clouds.json"
        assert digest(manifest) == spec["manifest_sha256"]
        item = next(i for i in json.loads(manifest.read_text())["cases"] if i["case_id"] == case["case_id"])
        record_out = {"alias": alias, "case_id": case["case_id"], "source_sha256": case["source_sha256"],
                      "verified_decode_sha256": decoded.provenance["decoded_sha256"],
                      "pn1_decode_sha256": hashlib.sha256(old.data).hexdigest(), "decoder_comparison": fidelity(legacy, common),
                      "datatype_threshold": storage_threshold, "observed_max_threshold": range_threshold, "boxes": []}
        print(f"RANGE {alias} PN1={legacy.min()}..{legacy.max()} verified={common.min()}..{common.max()} thresholds={storage_threshold}/{range_threshold}", flush=True)
        for box in case["boxes"]:
            low, high = np.array(box["tile"]["start_xyz"]), np.array(box["tile"]["stop_xyz"])
            slices = tuple(slice(int(a), int(b)) for a, b in zip(low[::-1], high[::-1]))
            zyx = np.indices(common[slices].shape).reshape(3, -1)+low[::-1, None]
            box_ids = np.ravel_multi_index(tuple(zyx), common.shape)
            pools = {"pn1_native": box_ids[legacy.ravel()[box_ids] > storage_threshold],
                     "pn1_decode_only": box_ids[common.ravel()[box_ids] > storage_threshold],
                     "pn1_range_random": box_ids[common.ravel()[box_ids] > range_threshold]}
            pools["pn1_range_spatial"] = pools["pn1_range_random"]
            windows = box["windows"]; runs = clipped_curve_samples(morphology, low, high)
            dense = next(a["point_count"] for a in box["arms"])
            box_out = {"tile_id": box["tile_id"], "tile": box["tile"], "windows": len(windows),
                       "reference_length_in_box_quadrature": sum(float(w.sum()) for _, w in runs),
                       "pools": {}, "arms": []}
            for name, ids in pools.items():
                frame = legacy if name == "pn1_native" else common
                result = score(ids, frame.ravel()[ids], common, common.shape, windows, runs, config)
                box_out["pools"][name] = {"eligible_count": len(ids), "native_minimum_256_would_be_met": len(ids) >= 256,
                                          "candidate_contrast_strata": result["contrast_strata"], "candidate_curve_metrics": result["curve_metrics"]}
            for budget in (config["primary_slots"], dense):
                for seed in config["seeds"]:
                    for name in config["arms"]:
                        if name.startswith("pn1_"):
                            pool = pools[name]; frame = legacy if name == "pn1_native" else common
                            ids = helper["sample_indices_fixed"](pool, budget, np.random.default_rng(seed),
                                strategy="spatial" if name.endswith("spatial") else "random", data=frame.ravel(),
                                width=common.shape[2], height=common.shape[1], cell_size=8) if len(pool) else np.empty(0, dtype=np.int64)
                            values = frame.ravel()[ids]
                        else:
                            method = name.removeprefix("pn2_")
                            packet = next(p for p in item["outputs"] if p["method"] == method and p["cell_side_voxel"] == 4 and p["seed"] == seed)
                            path = manifest.parent / packet["path"]
                            assert digest(path) == packet["sha256"]
                            with np.load(path, allow_pickle=False) as data:
                                inside = ((data["xyz_native"] >= low) & (data["xyz_native"] < high)).all(1)
                                ids, values = data["flat_indices"][inside][:budget], data["intensity"][inside][:budget]
                            assert len(ids) == budget and np.array_equal(values, common.ravel()[ids])
                        result = score(ids, values, common, common.shape, windows, runs, config)
                        if name.startswith("pn2_") and budget == dense:
                            original = next(a for a in box["arms"] if a["method"] == name.removeprefix("pn2_") and a["seed"] == seed)
                            assert result["contrast_fractions"] == original["input_retained_fraction"]
                        np.savez_compressed(folder / f"tile{box['tile_id']}_{name}_slots{budget}_seed{seed}.npz",
                                            flat_indices=ids, stored_intensity=values, verified_intensity=common.ravel()[ids])
                        box_out["arms"].append({"method": name, "seed": seed, "requested_slots": budget,
                                                "actual_slots": len(ids), "status": "no_eligible_voxels" if not len(ids) else "sampled",
                                                **result})
                print(f"SCORED {alias} tile={box['tile_id']} slots={budget} seven arms/five seeds", flush=True)
            assert len(box_out["arms"]) == 70
            record_out["boxes"].append(box_out)
        write(folder / "report.json", record_out); report["cases"].append(record_out)
    report["seconds"] = time.perf_counter()-started
    write(directory / "report.json", report)
    print(f"COMPLETED input comparison seconds={report['seconds']:.1f} report={directory / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pn1_input_comparison.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
