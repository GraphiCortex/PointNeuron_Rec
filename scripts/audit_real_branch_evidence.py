"""Fixed real-cloud evidence audit: two training sources/four boxes, CPU only.

No guide generation, resampling, encoder weights, learned inference or optimizer.
Native FPS/ball operators inspect geometric access only; fine skips still exist.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from pn2.evaluation.branch_evidence import arc_windows, measure, profile_window, source_priority_order
from pn2.evaluation.cloud_support import branch_paths
from pn2.evaluation.swc import parse_swc
from pn2.inputs.cases import ImageCase
from pn2.inputs.image import decode_image
from pn2.inputs.sampling import selection_count


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_ops():
    import torch
    source = ROOT.parent / "pointneuron2/src/pointneuron/models/pointnext_ops.py"
    expected = "968fd3af88446d92c2dc3295eb462362794910f829ee755a61474b1b3fb7886d"
    if digest(source) != expected:
        raise ValueError("Historical native operators changed")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("pinned_native_ops", source)
    ops = importlib.util.module_from_spec(spec); spec.loader.exec_module(ops)
    torch.set_num_threads(1)
    return torch, ops


def box_mask(flat_ids, start, stop, full_shape):
    coords = np.array(np.unravel_index(flat_ids, full_shape))[::-1].T
    local = coords-np.asarray(start)
    shape = tuple((np.asarray(stop)-start)[::-1])
    mask = np.zeros(shape, bool)
    mask[tuple(local[:, ::-1].T)] = True
    return mask


def trace(flat, xyz, start, stop, full_shape, torch, ops, sensitivity=False):
    order = source_priority_order(flat)
    center = (np.asarray(start)+np.asarray(stop)-1)/2
    original_positions = torch.tensor((xyz-center)/64, dtype=torch.float32)[None]
    positions = original_positions[:, order]
    source_ids = flat[order]
    arrays = {"input_flat_ids": source_ids}
    stages, first_mask, sensitivity_mask = [], None, None
    for i in range(4):
        before = time.perf_counter()
        fps = ops.farthest_point_indices(positions, positions.shape[1]//4)
        query = ops.gather_points(positions, fps)
        radius = .1*2**i
        neighbors = ops.ball_query_indices(positions, query, radius, 32, 256)
        members = neighbors[0].numpy()
        # Integer source grid gives no exact distance ties at these radii.
        eligible = cKDTree(positions[0].numpy()).query_ball_point(query[0].numpy(), radius, return_length=True)
        distinct = np.array([len(np.unique(row)) for row in members])
        assert np.all(eligible >= distinct) and np.all(distinct == np.minimum(eligible, 32))
        grouped = source_ids[members]
        centroids = source_ids[fps[0].numpy()]
        arrays[f"stage{i+1}_neighbor_flat_ids"] = grouped
        arrays[f"stage{i+1}_centroid_flat_ids"] = centroids
        stages.append({"stage": i+1, "support_count": int(positions.shape[1]), "centroid_count": int(query.shape[1]),
                       "radius_voxel": 64*radius, "K": 32, "mean_eligible_members": float(eligible.mean()),
                       "mean_distinct_selected_members": float(distinct.mean()),
                       "fraction_neighborhoods_truncated": float(np.mean(eligible > 32)),
                       "fraction_neighborhood_slots_repeated": float(np.mean((32-distinct)/32)),
                       "seconds": time.perf_counter()-before})
        if i == 0:
            first_mask = box_mask(np.unique(grouped), start, stop, full_shape)
            stages[-1]["fraction_input_points_in_any_selected_ball"] = float(first_mask.sum()/len(flat))
            if sensitivity:
                other = ops.ball_query_indices(original_positions, query, radius, 32, 256)[0].numpy()
                other_ids = flat[other]
                sensitivity_mask = box_mask(np.unique(other_ids), start, stop, full_shape)
                arrays["same_queries_race_order_neighbor_flat_ids"] = other_ids
                stages[-1]["race_vs_hash_same_queries_changed_fraction"] = float(np.mean([
                    set(a.tolist()) != set(b.tolist()) for a, b in zip(grouped, other_ids)]))
        print(f"TRACE stage={i+1} N={positions.shape[1]}->{query.shape[1]} truncated={stages[-1]['fraction_neighborhoods_truncated']:.3f} seconds={stages[-1]['seconds']:.2f}", flush=True)
        positions, source_ids = query, centroids
    return first_mask, sensitivity_mask, stages, arrays


def weighted(values, weights):
    valid = np.array([v is not None for v in values])
    return float(np.average(np.asarray(values, object)[valid].astype(float), weights=np.asarray(weights)[valid])) if valid.any() else None


def scores(windows, mask):
    return [[row["retained_mass_fraction"] for row in measure(w, mask)] for w in windows]


def summaries(windows, values, radii):
    result = []
    for width in ("all", "thin", "intermediate", "broad", "unresolved"):
        for contrast in ("all", "dim", "middle", "bright", "tied", "nonpositive_or_undefined"):
            chosen = [i for i, w in enumerate(windows) if
                      (width == "all" or w["metadata"]["width_class"] == width) and
                      (contrast == "all" or w["metadata"]["contrast_class"] == contrast)]
            if not chosen:
                continue
            weights = [windows[i]["metadata"]["arc_weight"] for i in chosen]
            for r, radius in enumerate(radii):
                v = [values[i][r] for i in chosen]
                result.append({"width_class": width, "contrast_class": contrast, "radius_voxel": radius,
                               "window_count": len(chosen), "defined_windows": sum(x is not None for x in v),
                               "arc_weighted_retained_fraction": weighted(v, weights)})
    return result


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    if config["seeds"] != [0, 1, 2, 3, 4] or config["cell_side_voxel"] != 4:
        raise ValueError("This bounded comparison fixes the five seeds and dense quota")
    directory = Path(output).resolve(); directory.mkdir(parents=True, exist_ok=False)
    write(directory / "config.json", config)
    torch, ops = load_ops()
    inventory_path = ROOT / "configs/training_inventory.json"
    inventory = json.loads(inventory_path.read_text())["cases"]
    report = {"scope": "Training-only local image evidence and geometric access; no learned feature/reconstruction verdict",
              "config_sha256": digest(config_path), "script_sha256": digest(__file__),
              "metric_source_sha256": digest(ROOT / "src/pn2/evaluation/branch_evidence.py"),
              "torch": torch.__version__, "cases": []}
    started = time.perf_counter()
    for spec in config["cases"]:
        name = spec["alias"]; folder = directory / name; folder.mkdir()
        manifest_path = ROOT / spec["frozen_run"] / "clouds.json"
        if digest(manifest_path) != spec["manifest_sha256"]:
            raise ValueError("Frozen cloud manifest changed")
        item = next(v for v in json.loads(manifest_path.read_text())["cases"] if v["case_id"] == spec["case_id"])
        case = next(v for v in inventory if v["case_id"] == spec["case_id"])
        if case["partition"] != "training" or case["coordinates"] != {"mode": "native", "units": "voxel"}:
            raise ValueError("This audit accepts declared native-voxel training sources only")
        print(f"DECODE {name}: existing training source; no filtering", flush=True)
        decoded = decode_image(ImageCase(case["case_id"], (inventory_path.parent / case["image_path"]).resolve(), "training"))
        raw = decoded.array_zyx
        if decoded.provenance["source_sha256"] != item["image"]["source_sha256"] or decoded.provenance["decoded_sha256"] != item["image"]["decoded_sha256"]:
            raise ValueError("Source image changed")
        reference_path = (inventory_path.parent / case["reference_swc"]).resolve()
        morphology = parse_swc(reference_path, type_policy="opaque_integer")
        if digest(reference_path) != spec["reference_sha256"] or morphology.component_count != 1:
            raise ValueError("Reference changed or leaves the declared single-component scope")
        samples = []
        for path_id, path in enumerate(branch_paths(morphology)):
            for sample in arc_windows(morphology.xyz[path], config["arc_step_voxel"]):
                samples.append(dict(sample, path_id=path_id))
        prepared = []
        for tile_id in spec["tile_ids"]:
            tile = item["tiles"][tile_id]; windows = []
            low, high = np.array(tile["start_xyz"])-.5, np.array(tile["stop_xyz"])-.5
            for sample in samples:
                if np.all(sample["xyz"] >= low) and np.all(sample["xyz"] < high):
                    w = profile_window(raw, sample["xyz"], sample["tangent"], tile["start_xyz"], tile["stop_xyz"],
                                       radii=config["radii_voxel"], half_length=config["half_length_voxel"],
                                       shell=config["background_shell_voxel"], bin_width=config["profile_bin_voxel"])
                    width = w["metadata"]["apparent_width_voxel"]
                    w["metadata"].update(path_id=sample["path_id"], arc_coordinate=sample["arc"], arc_weight=sample["weight"],
                        width_class="unresolved" if width is None else "thin" if width <= config["width_bins_voxel"]["thin_maximum"]
                        else "broad" if width >= config["width_bins_voxel"]["broad_minimum"] else "intermediate")
                    windows.append(w)
            prepared.append({"tile_id": tile_id, "tile": tile, "windows": windows, "arms": []})
        positive = [w["metadata"]["inner_peak_contrast"] for b in prepared for w in b["windows"]
                    if w["metadata"]["inner_peak_contrast"] is not None and w["metadata"]["inner_peak_contrast"] > 0]
        cutoffs = np.quantile(positive, [1/3, 2/3]).tolist() if positive else [None, None]
        for box in prepared:
            for w in box["windows"]:
                contrast = w["metadata"]["inner_peak_contrast"]
                group = "nonpositive_or_undefined" if contrast is None or contrast <= 0 else "tied" if cutoffs[0] == cutoffs[1] else (
                    "dim" if contrast <= cutoffs[0] else "bright" if contrast >= cutoffs[1] else "middle")
                w["metadata"]["contrast_class"] = group
        for packet in item["outputs"]:
            if packet["cell_side_voxel"] != 4 or packet["seed"] not in config["seeds"]:
                continue
            packet_path = manifest_path.parent / packet["path"]
            if digest(packet_path) != packet["sha256"]:
                raise ValueError("Frozen cloud changed")
            with np.load(packet_path, allow_pickle=False) as data:
                flat, xyz, intensity = data["flat_indices"], data["xyz_native"], data["intensity"]
            for box in prepared:
                tile = box["tile"]; low, high = np.array(tile["start_xyz"]), np.array(tile["stop_xyz"])
                keep = ((xyz >= low) & (xyz < high)).all(axis=1)
                ids, points = flat[keep], xyz[keep]
                expected = selection_count(int(np.prod(high-low)), 4)
                if (len(ids) != expected or len(np.unique(ids)) != expected
                        or not np.array_equal(intensity[keep], raw.ravel()[ids])
                        or not np.array_equal(np.ravel_multi_index(tuple(points[:, ::-1].T), raw.shape), ids)):
                    raise ValueError("Frozen box count/coordinate/intensity mismatch")
                mask = box_mask(ids, low, high, raw.shape)
                input_values = scores(box["windows"], mask)
                arm = {"method": packet["method"], "seed": packet["seed"], "point_count": expected,
                       "source_packet_sha256": packet["sha256"], "input_retained_fraction": input_values,
                       "input_summary": summaries(box["windows"], input_values, config["radii_voxel"])}
                if packet["seed"] == 0:
                    sensitivity = name == "frog22" and box["tile_id"] == 5 and packet["method"] == "sato"
                    reached, alternate, stages, arrays = trace(ids, points, low, high, raw.shape, torch, ops, sensitivity)
                    np.savez_compressed(folder / f"tile{box['tile_id']}_{packet['method']}_trace.npz", **arrays)
                    accessible = scores(box["windows"], reached)
                    conditional = [[b/a if a is not None and a > 0 and b is not None else None for a, b in zip(x, y)]
                                   for x, y in zip(input_values, accessible)]
                    arm.update(stages=stages, first_group_retained_fraction=accessible,
                               first_group_summary=summaries(box["windows"], accessible, config["radii_voxel"]),
                               first_group_conditional_summary=summaries(box["windows"], conditional, config["radii_voxel"]))
                    if alternate is not None:
                        arm["same_queries_race_order_summary"] = summaries(box["windows"], scores(box["windows"], alternate), config["radii_voxel"])
                box["arms"].append(arm)
                print(f"SCORED {name} tile={box['tile_id']} {packet['method']} seed={packet['seed']} N={expected} windows={len(box['windows'])}", flush=True)
        saved_boxes = []
        for box in prepared:
            assert len(box["arms"]) == 3*len(config["seeds"])
            assert len({(a["method"], a["seed"]) for a in box["arms"]}) == len(box["arms"])
            windows = []
            for w in box["windows"]:
                windows.append(dict(w["metadata"], domains=[{k: v for k, v in d.items() if k not in ("ids", "mass")}
                    | {"available_positive_contrast_mass": float(d["mass"].sum()) if d["mass"] is not None else None} for d in w["domains"]]))
            saved_boxes.append({"tile_id": box["tile_id"], "tile": box["tile"], "windows": windows, "arms": box["arms"]})
        record = {"alias": name, "case_id": case["case_id"], "partition": "training",
                  "source_sha256": item["image"]["source_sha256"], "reference_sha256": digest(reference_path),
                  "reference_type_policy": "opaque_integer", "reference_components": 1,
                  "contrast_terciles_raw_units": cutoffs, "boxes": saved_boxes}
        write(folder / "report.json", record); report["cases"].append(record)
    report["seconds"] = time.perf_counter()-started
    write(directory / "report.json", report)
    print(f"COMPLETED four-box audit seconds={report['seconds']:.1f} report={directory / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/real_branch_evidence.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
