"""Training-only sizing of proposed SWC curve targets; no model/cache changes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from pointneuron.data.swc import SwcTree, parse_swc


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve_geometry(tree: SwcTree):
    """Split a validated forest at roots, leaves and branch junctions."""
    errors = tree.validate()
    if errors:
        raise ValueError("; ".join(errors))
    nodes = {n.node_id: n for n in tree.nodes}
    children = {i: [] for i in nodes}
    for n in tree.nodes:
        if not np.isfinite([n.x, n.y, n.z, n.radius]).all() or n.radius < 0:
            raise ValueError("Nonfinite geometry or negative radius")
        if n.parent_id != -1:
            children[n.parent_id].append(n.node_id)
    reachable, pending = set(), [i for i, n in nodes.items() if n.parent_id == -1]
    while pending:
        i = pending.pop()
        if i in reachable:
            raise ValueError("Cycle in reference")
        reachable.add(i)
        pending.extend(children[i])
    if len(reachable) != len(nodes):
        raise ValueError("Reference contains an unrooted cycle")
    critical = {i for i, n in nodes.items()
                if n.parent_id == -1 or len(children[i]) != 1}
    chains, visited = [], set()
    for start in sorted(critical):
        for child in sorted(children[start]):
            ids = [start]
            current = child
            while True:
                edge = (ids[-1], current)
                if edge in visited:
                    raise ValueError("Repeated edge or cycle")
                visited.add(edge)
                ids.append(current)
                if current in critical:
                    break
                current = children[current][0]
            xyz = np.array([[nodes[i].x, nodes[i].y, nodes[i].z] for i in ids])
            radii = np.array([nodes[i].radius for i in ids])
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
            chains.append((ids, xyz, radii, arc))
    if len(visited) != tree.edge_count:
        raise ValueError("Forest contains an unvisited cycle")
    return nodes, children, critical, chains


def clip_parameters(a, b, lower, upper):
    entry, leave = 0., 1.
    for axis, delta in enumerate(b - a):
        if abs(delta) < 1e-12:
            if not lower[axis] <= a[axis] <= upper[axis]:
                return None
        else:
            bounds = sorted(((lower[axis] - a[axis]) / delta,
                             (upper[axis] - a[axis]) / delta))
            entry, leave = max(entry, bounds[0]), min(leave, bounds[1])
    return (entry, leave) if entry <= leave else None


def patch_targets(geometry, center, radius, step):
    """Keep structural nodes, full-chain arc samples and crop intersections.

    Phase is anchored at the full-reference chain start, before cropping.
    Keys preserve distinct coincident reference nodes; no spatial NMS occurs.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Invalid patch center")
    if not np.isfinite([radius, step]).all() or min(radius, step) <= 0:
        raise ValueError("Radius and step must be positive and finite")
    lower, upper = center - radius, center + radius
    nodes, children, critical, chains = geometry
    targets = {}

    def add(key, xyz, r, role):
        if np.all((xyz >= lower - 1e-8) & (xyz <= upper + 1e-8)):
            if key not in targets:
                targets[key] = {"xyz": xyz.tolist(), "radius": float(r), "roles": set()}
            targets[key]["roles"].add(role)

    def edge_key(parent, child, t):
        if abs(t) < 1e-8:
            return ("node", parent)
        if abs(t - 1) < 1e-8:
            return ("node", child)
        return ("edge", parent, child, round(float(t), 8))

    for i in sorted(critical):
        n = nodes[i]
        xyz = np.array([n.x, n.y, n.z])
        if n.parent_id == -1:
            add(("node", i), xyz, n.radius, "root")
        if not children[i]:
            add(("node", i), xyz, n.radius, "leaf")
        if len(children[i]) > 1:
            add(("node", i), xyz, n.radius, "junction")

    for ids, xyz, radii, arc in chains:
        if np.any(xyz.max(axis=0) < lower) or np.any(xyz.min(axis=0) > upper):
            continue
        distances = np.arange(step, arc[-1] - 1e-8, step)
        indices = np.minimum(np.searchsorted(arc, distances, side="right") - 1, len(ids) - 2)
        lengths = arc[indices + 1] - arc[indices]
        if np.any(lengths <= 0):
            raise ValueError("Invalid interpolation interval")
        fractions = (distances - arc[indices]) / lengths
        positions = xyz[indices] + fractions[:, None] * (xyz[indices + 1] - xyz[indices])
        inside = np.all((positions >= lower - 1e-8) & (positions <= upper + 1e-8), axis=1)
        for k in np.flatnonzero(inside):
            j, t = indices[k], fractions[k]
            add(edge_key(ids[j], ids[j + 1], t), positions[k],
                radii[j] + t * (radii[j + 1] - radii[j]), "arc_sample")
        possible = np.all((np.maximum(xyz[:-1], xyz[1:]) >= lower) &
                          (np.minimum(xyz[:-1], xyz[1:]) <= upper), axis=1)
        for j in np.flatnonzero(possible):
            interval = clip_parameters(xyz[j], xyz[j + 1], lower, upper)
            if interval is None or interval == (0., 1.):
                continue
            # Only a box intersection is forced; internal SWC vertices are not.
            for t in set(interval):
                p = xyz[j] + t * (xyz[j + 1] - xyz[j])
                if np.any(np.isclose(p, lower, atol=1e-8, rtol=0)) or np.any(np.isclose(p, upper, atol=1e-8, rtol=0)):
                    add(edge_key(ids[j], ids[j + 1], t), p,
                        radii[j] + t * (radii[j + 1] - radii[j]), "crop_boundary")
    return list(targets.values())


def statistics(values):
    return {"min": int(min(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": int(max(values)),
            "zero": int(np.sum(np.asarray(values) == 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--partition", type=Path, default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=float, nargs="+", default=[4., 6., 8.])
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; existing evidence is immutable")
    if any(not np.isfinite(s) or s <= 0 for s in args.steps):
        parser.error("Steps must be positive and finite")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest.get("failures"):
        raise ValueError("Incomplete source cache")
    if manifest["volume_reader_version"] != "vaa3d-pbd-v2":
        raise ValueError("Expected corrected source cache")
    if manifest["partition_sha256"] != digest(args.partition):
        raise ValueError("Partition hash mismatch")
    allowed = {r["sample_index"]: r["sample_id"] for r in partition["partitions"]["train"]}
    for role in ("val", "test"):
        if any(r["sample_index"] in allowed or r["sample_id"] in allowed.values()
               for r in partition["partitions"][role]):
            raise ValueError("Crossed neuron partitions")
    rows = [r for r in manifest["records"] if r["partition"] == "train"]
    if not rows:
        raise ValueError("No training records")
    sources, records, seen = {}, [], set()
    for row in rows:
        index = row["sample_index"]
        if allowed.get(index) != row["sample_id"]:
            raise ValueError("Training identity mismatch")
        path = (args.manifest.parent / row["path"]).resolve()
        if not path.is_relative_to(args.manifest.parent.resolve()) or path in seen:
            raise ValueError("Invalid or duplicate cache path")
        seen.add(path)
        if digest(path) != row["sha256"]:
            raise ValueError("Training cache content mismatch")
        with np.load(path, allow_pickle=False) as cached:
            meta = json.loads(str(cached["metadata"]))
            for key in ("sample_index", "sample_id", "partition", "patch_index"):
                if meta[key] != row[key]:
                    raise ValueError("Cached identity mismatch")
            if meta["config"] != manifest["config"] or meta["medial_config"] != manifest["medial_config"]:
                raise ValueError("Cached settings mismatch")
            if meta["volume_reader_version"] != manifest["volume_reader_version"]:
                raise ValueError("Cached decoder version mismatch")
            labels = cached["skeleton_nodes"]
            if len(labels) != row["label_nodes"]:
                raise ValueError("Cached label count mismatch")
            if index not in sources:
                directory = (args.data_root / row["sample_id"]).resolve()
                if not directory.is_relative_to(args.data_root.resolve()):
                    raise ValueError("Source directory escapes data root")
                matches = [p for p in sorted(directory.glob("*.swc")) if digest(p) == meta["swc_sha256"]]
                if not matches:
                    raise ValueError("No matching source SWC in training neuron directory")
                tree = parse_swc(matches[0])
                sources[index] = (curve_geometry(tree), meta["swc_sha256"], matches[0], tree)
            geometry, source_hash, source_path, tree = sources[index]
            if source_hash != meta["swc_sha256"]:
                raise ValueError("Training source changes across crops")
            center, radius = np.asarray(meta["patch_center"]), meta["config"]["patch_radius"]
            expected = [n for n in tree.nodes if np.all(np.abs(np.array([n.x, n.y, n.z]) - center) <= radius)]
            expected_array = np.array([[n.node_id, n.x, n.y, n.z, n.radius, n.parent_id] for n in expected], dtype=np.float32).reshape(-1, 6)
            if labels.shape != expected_array.shape or not np.array_equal(labels, expected_array):
                raise ValueError("Cache labels differ from the verified full SWC crop")
            counts = {}
            for step in args.steps:
                targets = patch_targets(geometry, center, radius, step)
                roles = Counter(role for t in targets for role in t["roles"])
                counts[str(step)] = {"targets": len(targets), "roles": dict(roles)}
            records.append({"path": row["path"], "sha256": row["sha256"], "sample_index": index,
                            "sample_id": row["sample_id"], "patch_index": row["patch_index"],
                            "patch_center": center.tolist(), "original_label_nodes": len(labels), "steps": counts})
    summaries = {}
    for step in args.steps:
        values = [r["steps"][str(step)]["targets"] for r in records]
        summaries[str(step)] = {**statistics(values),
            "over_capacity": {str(q): sum(v > q for v in values) for q in (128, 256, 512, 768, 1024)},
            "previously_vertex_empty_now_with_curve": sum(r["original_label_nodes"] == 0 and r["steps"][str(step)]["targets"] > 0 for r in records)}
    report = {"status": "completed", "verification": "passed", "scope": "training only; no source-image/model reads, training, inference or cache writes",
              "manifest_sha256": digest(args.manifest), "partition_sha256": digest(args.partition),
              "script_sha256": digest(__file__), "training_neurons": len(sources), "training_crops": len(records),
              "distance_units": "voxel-coordinate Euclidean distance; no claim of physical calibration",
              "spacing_summaries": summaries, "original_label_counts": statistics([r["original_label_nodes"] for r in records]),
              "sources": [{"sample_index": i, "path": p.relative_to(args.data_root.resolve()).as_posix(),
                           "sha256": h, "nodes": len(t.nodes), "roots": t.root_count,
                           "chains": len(g[3])} for i, (g, h, p, t) in sorted(sources.items())],
              "records": records, "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["Reference agreement, not complete biological ground truth.",
                              "Roots and leaves may be annotation fragment ends.",
                              "Counts use only the 262 cached training crops, not all future inference crops.",
                              "Coincident distinct reference nodes are retained; role counts overlap."]}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified training neurons/crops: {len(sources)}/{len(records)}")
    print("Original SWC vertex counts:", report["original_label_counts"])
    for step, values in summaries.items():
        print(f"step={step} voxels:", values)
    print(f"Results: {args.output_dir / 'summary.json'}")
    print(f"Elapsed: {report['elapsed_seconds']:.1f}s; no training or inference")


if __name__ == "__main__":
    main()
