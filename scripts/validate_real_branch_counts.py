"""Independently check saved contrast fractions from point attributes; add counts.

Reads frozen clouds and saved reference-window metadata only. No source decode,
filter, resampling, hierarchy replay, GPU, neural weights or optimizer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def point_stats(points, intensity, windows, radii, half_length):
    tree = cKDTree(points)
    result = []
    for window in windows:
        center = np.asarray(window["xyz"])
        radius = float(np.hypot(max(radii), half_length))
        margin = 64*np.finfo(float).eps*(1+np.linalg.norm(center)+radius)
        ids = tree.query_ball_point(center, radius+margin)
        offset = points[ids]-center
        axial = offset@np.asarray(window["tangent"])
        radial = np.sqrt(np.maximum(0., np.einsum("ij,ij->i", offset, offset)-axial**2))
        rows = []
        for r, domain in zip(radii, window["domains"]):
            chosen = (np.abs(axial) <= half_length) & (radial <= r)
            values = intensity[ids][chosen].astype(float)
            background = window["background"]
            contrast = values-background if background is not None else None
            mass = float(np.maximum(contrast, 0).sum()) if contrast is not None else None
            total = domain["available_positive_contrast_mass"]
            rows.append({"radius": r, "selected_points": len(values),
                         "points_above_background": int(np.count_nonzero(contrast > 0)) if contrast is not None else None,
                         "selected_intensity_median": float(np.median(values)) if len(values) else None,
                         "selected_signed_contrast_median": float(np.median(contrast)) if len(values) and contrast is not None else None,
                         "retained_positive_contrast_mass": mass,
                         "retained_mass_fraction": mass/total if total is not None and total > 0 else None})
        result.append(rows)
    return result


def validate(report_path):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    config = json.loads((report_path.parent / "config.json").read_text())
    result = {"scope": "Independent point-attribute mass validation and local counts; no new clouds or hierarchy runs",
              "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(), "boxes": []}
    maximum = 0.
    for case in report["cases"]:
        spec = next(s for s in config["cases"] if s["alias"] == case["alias"])
        frozen = ROOT / spec["frozen_run"]
        item = next(i for i in json.loads((frozen / "clouds.json").read_text())["cases"] if i["case_id"] == case["case_id"])
        for box in case["boxes"]:
            states = []
            low, high = np.asarray(box["tile"]["start_xyz"]), np.asarray(box["tile"]["stop_xyz"])
            for arm in box["arms"]:
                packet = next(o for o in item["outputs"] if o["method"] == arm["method"] and o["seed"] == arm["seed"] and o["cell_side_voxel"] == 4)
                path = frozen / packet["path"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == arm["source_packet_sha256"]
                with np.load(path, allow_pickle=False) as data:
                    keep = ((data["xyz_native"] >= low) & (data["xyz_native"] < high)).all(1)
                    points, values, flat = data["xyz_native"][keep], data["intensity"][keep], data["flat_indices"][keep]
                assert len(flat) == arm["point_count"]
                inputs = [("input", points, values, arm["input_retained_fraction"])]
                if arm["seed"] == 0:
                    trace = report_path.parent / case["alias"] / f"tile{box['tile_id']}_{arm['method']}_trace.npz"
                    with np.load(trace, allow_pickle=False) as data:
                        names = [("first_group", "stage1_neighbor_flat_ids", arm["first_group_retained_fraction"])]
                        if "same_queries_race_order_neighbor_flat_ids" in data:
                            names.append(("same_queries_race_order", "same_queries_race_order_neighbor_flat_ids", None))
                        for label, key, expected in names:
                            selected = np.isin(flat, np.unique(data[key]))
                            inputs.append((label, points[selected], values[selected], expected))
                for label, positions, attrs, expected in inputs:
                    stats = point_stats(positions, attrs, box["windows"], config["radii_voxel"], config["half_length_voxel"])
                    if expected is not None:
                        for old, rows in zip(expected, stats):
                            for a, row in zip(old, rows):
                                b = row["retained_mass_fraction"]
                                assert (a is None) == (b is None)
                                if a is not None:
                                    maximum = max(maximum, abs(a-b))
                                    assert abs(a-b) <= 1e-9
                    states.append({"method": arm["method"], "seed": arm["seed"], "state": label, "windows": stats})
            result["boxes"].append({"alias": case["alias"], "tile_id": box["tile_id"], "states": states})
    result["maximum_mass_fraction_discrepancy"] = maximum
    result["status"] = "passed"
    target = report_path.parent / "point_counts.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(f"PASSED point-attribute cross-check; max fraction discrepancy={maximum:.3g}; counts={target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    validate(args.report)
