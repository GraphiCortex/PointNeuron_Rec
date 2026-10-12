"""Fixed-field center-path diagnostic on two pre-agreed training crops.

Short frozen checkpoint inference, then existing CPU TEASAR extraction. No
optimizer, reference-guided extraction, parameter search, SWC promotion or
validation. Ground truth enters only the subsequent geometric measurements.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from pointneuron.data.centerline_field import SCHEMA
from pointneuron.data.curve_targets import curve_geometry, clip_parameters
from pointneuron.data.medial_sampling import MedialSamplingConfig, image_medial_skeleton
from pointneuron.data.query_cache import file_digest
from pointneuron.data.swc import parse_swc
from pointneuron.models.centerline_unet import CenterlineUNet
from scripts.pointneuron2.reference_gaps import segment_probe
from scripts.pointneuron2.train_centerline_fit import load_records, field_metrics

CASES = ((162, 2), (96, 3))


def nearest_segments(points, segments):
    """Exact distance to segment interiors; degenerate segments represent points."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
    if not np.isfinite(points).all() or not np.isfinite(segments).all():
        raise ValueError("Nonfinite path geometry")
    result = np.full(len(points), np.inf)
    if not len(segments):
        return result
    start = segments[:, 0]
    delta = segments[:, 1] - start
    lengths2 = np.einsum("ij,ij->i", delta, delta)
    for i in range(0, len(points), 64):
        difference = points[i:i+64, None, :] - start
        dot = np.einsum("nmi,mi->nm", difference, delta)
        fraction = np.divide(dot, lengths2, out=np.zeros_like(dot), where=lengths2 > 0)
        residual = difference - np.clip(fraction, 0, 1)[..., None] * delta
        result[i:i+64] = np.sqrt(np.einsum("nmi,nmi->nm", residual, residual).min(axis=1))
    return result


def distance_summary(distances, weights):
    if not len(weights) or weights.sum() <= 0:
        return None
    order = np.argsort(distances)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    finite = lambda value: float(value) if np.isfinite(value) else None
    quantile = lambda q: finite(distances[order[min(np.searchsorted(cumulative, q), len(order)-1)]])
    return {"length_weighted_mean_voxels": finite(np.average(distances, weights=weights)),
            "length_weighted_median_voxels": quantile(.5), "length_weighted_p95_voxels": quantile(.95),
            "within_voxels": {str(t): float(weights[distances <= t].sum()/weights.sum()) for t in (1, 2, 4, 6)}}


def reference_segments(record, data_root):
    """Read the original hashed SWC only after extraction; preserve all its edges."""
    meta = record["metadata"]
    directory = (data_root / meta["sample_id"]).resolve()
    if not directory.is_relative_to(data_root.resolve()):
        raise ValueError("Reference path escapes dataset")
    matches = [p for p in directory.glob("*.swc") if file_digest(p) == meta["swc_sha256"]]
    if len(matches) != 1:
        raise ValueError("Original reference hash cannot be recovered uniquely")
    tree = parse_swc(matches[0])
    nodes, children, _, _ = curve_geometry(tree)
    origin = np.asarray(meta["origin_xyz"])
    lower = np.asarray(meta["patch_center"]) - meta["config"]["patch_radius"]
    upper = np.asarray(meta["patch_center"]) + meta["config"]["patch_radius"]
    full, clipped = [], []
    isolated = []
    for n in nodes.values():
        b = np.array([n.x, n.y, n.z])
        if n.parent_id == -1:
            if not children[n.node_id]:
                full.append([b-origin, b-origin])
                if np.all((b >= lower) & (b <= upper)):
                    isolated.append(b-origin)
            continue
        p = nodes[n.parent_id]
        a = np.array([p.x, p.y, p.z])
        full.append([a-origin, b-origin])
        interval = clip_parameters(a, b, lower, upper)
        if interval is not None:
            clipped.append([a+interval[0]*(b-a)-origin, a+interval[1]*(b-a)-origin])
    return np.asarray(full).reshape(-1, 2, 3), np.asarray(clipped).reshape(-1, 2, 3), np.asarray(isolated).reshape(-1, 3)


def plot_paths(record, arms, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    image = record["image"]
    scale = float(np.quantile(image[record["valid"]], .995)) or float(image.max())
    display = np.clip(image / max(scale, 1e-6), 0, 1)
    reference = record["reference_local_xyz"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for column, (arm, vertices, edges) in enumerate(arms):
        for row, (axis, coordinates, plane) in enumerate(((0, (0, 1), "XY"), (1, (0, 2), "XZ"))):
            ax = axes[row, column]
            ax.imshow(display.max(axis=axis), cmap="gray", origin="lower", vmin=0, vmax=1)
            if len(edges):
                ax.add_collection(LineCollection(vertices[edges][:, :, coordinates], colors="orange", linewidths=1))
            ax.scatter(reference[:, coordinates[0]], reference[:, coordinates[1]], s=2, color="cyan")
            title = "Original loss" if arm == "original" else "Hard-negative loss"
            ax.set_title(f"{title}: {plane}")
            ax.set_xlabel("local X voxel")
            ax.set_ylabel("local Y / Z voxel")
    meta = record["metadata"]
    fig.suptitle(f"Training neuron {meta['sample_index']}, crop {meta['patch_index']} | fixed threshold 0.5\n"
                 "Orange: extracted paths; cyan: reference | fixed TEASAR settings\n"
                 "Image contrast for display only; projections do not establish connectivity")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def run(args):
    torch.set_num_threads(2)
    manifest, records = load_records(args.manifest, ROOT / "configs/pointneuron2/partition_seed0.json")
    records = {(r["row"]["sample_index"], r["row"]["patch_index"]): r for r in records}
    cfg = manifest["config"]
    files, models, reports = {}, {}, {}
    for arm, folder in (("original", args.baseline), ("hard_negative", args.control)):
        path = folder / "field_step_0400.pt"
        report_path = folder / "fit_report.json"
        files[str(path.resolve())], files[str(report_path.resolve())] = file_digest(path), file_digest(report_path)
        fit = json.loads(report_path.read_text())
        cp = torch.load(path, map_location=args.device, weights_only=True)
        if (fit["status"] != "completed" or fit["config"] != cfg or cp["schema"] != SCHEMA or cp["step"] != 400
                or cp["config"] != cfg or cp["manifest_sha256"] != file_digest(args.manifest)
                or fit["manifest_sha256"] != cp["manifest_sha256"] or cfg["threshold"] != .5):
            raise ValueError("Frozen checkpoint/report pairing mismatch")
        m = CenterlineUNet(cfg["base_channels"]).to(args.device).eval()
        m.load_state_dict(cp["model"], strict=True)
        models[arm], reports[arm] = m, fit
    if args.output_dir.exists():
        raise FileExistsError("Use a new diagnostic directory")
    args.output_dir.mkdir(parents=True)
    summary = {"status": "running", "scope": "two selected training-crop center-path diagnostics; no optimizer, validation or SWC promotion",
               "cases": [list(c) for c in CASES], "checkpoint_and_report_hashes": files,
               "manifest_sha256": file_digest(args.manifest), "threshold": .5, "kimimaro_version": version("kimimaro"),
               "probe_step_voxels": 1., "reference_usage": "measurement only after GT-free extraction",
               "reference_scope": "full-forest segments for predicted proximity; patch-clipped segments for coverage",
               "rows": []}
    started = time.perf_counter()
    for case in CASES:
        r = records[case]
        medial = MedialSamplingConfig(**r["metadata"]["medial_config"])
        if medial.spacing_xyz != (1., 1., 1.):
            raise ValueError("This native-voxel diagnostic requires the recorded unit spacing")
        plot_arms, extracted = [], {}
        for arm, m in models.items():
            with torch.no_grad():
                probability = m(torch.from_numpy(r["image"])[None, None].to(args.device)).sigmoid()[0, 0].cpu().numpy()
            measured = field_metrics(probability, r)
            original = next(x for x in reports[arm]["evaluations"][-1]["crops"] if (x["sample_index"], x["patch_index"]) == case)
            if any(abs(measured[k]-original[k]) > 1e-5 for k in ("reference_coverage_6", "reference_proximity_6", "known_voxel_dice")):
                raise AssertionError("Frozen field does not reproduce saved inference")
            accepted = (probability >= .5) & r["valid"]
            print(f"Extracting CPU paths: neuron={case[0]} crop={case[1]} arm={arm} voxels={accepted.sum()}", flush=True)
            t = time.perf_counter()
            vertices, edges = image_medial_skeleton(np.flatnonzero(accepted), accepted.shape[2], accepted.shape[1], medial)
            extraction_seconds = time.perf_counter()-t
            if (not np.isfinite(vertices).all() or np.any(vertices < -1e-5)
                    or np.any(vertices > np.array(accepted.shape[::-1])-1+1e-5)):
                raise AssertionError("Extracted paths escape native crop frame")
            if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
                raise AssertionError("Invalid extracted edge indices")
            if len(vertices):
                rounded = np.rint(vertices).astype(int)
                if not np.all(accepted[tuple(rounded[:, ::-1].T)]):
                    raise AssertionError("Extracted vertices lack accepted voxel support")
            plot_arms.append((arm, vertices, edges))
            extracted[arm] = (vertices, edges, measured, extraction_seconds)
        # Reference access happens after both arms have already been extracted.
        full_reference, clipped_reference, isolated_reference = reference_segments(r, args.data_root)
        reference_probe = segment_probe(clipped_reference)
        for arm, (vertices, edges, measured, extraction_seconds) in extracted.items():
            degree = np.bincount(edges.ravel(), minlength=len(vertices))
            segments = vertices[edges]
            points_as_segments = np.repeat(vertices[degree == 0, None, :], 2, axis=1)
            prediction_geometry = np.concatenate((segments, points_as_segments))
            prediction_probe = segment_probe(segments)
            reference_distances = nearest_segments(reference_probe["points"], prediction_geometry)
            prediction_distances = nearest_segments(prediction_probe["points"], full_reference)
            adj = coo_matrix((np.ones(len(edges)*2), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                             shape=(len(vertices), len(vertices))).tocsr()
            components = int(connected_components(adj, directed=False, return_labels=False)) if len(vertices) else 0
            row = {"sample_index": case[0], "patch_index": case[1], "arm": arm, "medial_config": medial.to_dict(),
                   "field": measured, "extraction_seconds": extraction_seconds,
                   "vertices": len(vertices), "edges": len(edges), "components": components,
                   "endpoints": int((degree == 1).sum()), "branch_vertices": int((degree >= 3).sum()),
                   "path_length_voxels": float(prediction_probe["weights"].sum()),
                   "reference_length_voxels": float(reference_probe["weights"].sum()),
                   "reference_to_paths": distance_summary(reference_distances, reference_probe["weights"]),
                   "paths_to_reference": distance_summary(prediction_distances, prediction_probe["weights"]),
                   "isolated_reference_points": len(isolated_reference)}
            summary["rows"].append(row)
            np.savez_compressed(args.output_dir / f"sample_{case[0]:04d}_patch_{case[1]:03d}_{arm}.npz",
                                vertices_local_xyz=vertices, edges=edges, origin_xyz=np.array(r["metadata"]["origin_xyz"]))
            print(f"neuron={case[0]} crop={case[1]} arm={arm}: components={components} path_length={row['path_length_voxels']:.1f} "
                  f"coverage6={row['reference_to_paths']['within_voxels']['6'] if row['reference_to_paths'] else None} "
                  f"proximity6={row['paths_to_reference']['within_voxels']['6'] if row['paths_to_reference'] else None}", flush=True)
        plot_paths(r, plot_arms, args.output_dir / f"sample_{case[0]:04d}_patch_{case[1]:03d}_paths.png")
        (args.output_dir / "centerpaths.json").write_text(json.dumps(summary, indent=2))
    if any(file_digest(p) != digest for p, digest in files.items()):
        raise AssertionError("Checkpoint or original reports changed")
    summary.update(status="completed", checkpoints_unchanged=True, inference_reproduction_checks="passed",
                   elapsed_seconds=time.perf_counter()-started, primary_gate_changed=False)
    (args.output_dir / "centerpaths.json").write_text(json.dumps(summary, indent=2))
    print(f"report: {args.output_dir / 'centerpaths.json'}", flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(p.parse_args())
