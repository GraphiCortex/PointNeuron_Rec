from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.torch_dataset import load_split_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect cached point-cloud patches and SWC target geometry.")
    parser.add_argument("--split-file", required=True, help="Path to split JSON.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to inspect.")
    parser.add_argument("--record-indices", nargs="*", type=int, help="Record indices within the split. Defaults to all.")
    parser.add_argument("--positive-distance", type=float, default=6.0, help="Distance threshold used for target sanity stats.")
    parser.add_argument("--coverage-distance", type=float, default=8.0, help="Skeleton coverage distance used for target sanity stats.")
    args = parser.parse_args()

    paths = load_split_paths(args.split_file, args.split)
    if args.record_indices:
        selected_indices = args.record_indices
    else:
        selected_indices = list(range(len(paths)))

    for record_index in selected_indices:
        if record_index < 0 or record_index >= len(paths):
            print(f"index={record_index} out_of_range split_size={len(paths)}")
            continue
        summary = summarize_record(
            record_index=record_index,
            path=Path(paths[record_index]),
            positive_distance=args.positive_distance,
            coverage_distance=args.coverage_distance,
        )
        print_summary(summary)
    return 0


def summarize_record(record_index: int, path: Path, positive_distance: float, coverage_distance: float) -> dict[str, object]:
    record = np.load(path, allow_pickle=False)
    points = record["points"].astype(np.float32, copy=False)
    skeleton_nodes = record["skeleton_nodes"].astype(np.float32, copy=False)
    edge_index = record["edge_index"].astype(np.int64, copy=False)
    metadata = json.loads(str(record["metadata"]))

    point_xyz = points[:, :3]
    point_intensity = points[:, 3]
    node_xyz = skeleton_nodes[:, 1:4]
    node_radius = np.maximum(skeleton_nodes[:, 4], 0.0)
    unique_xyz = np.unique(point_xyz.astype(np.int64, copy=False), axis=0)

    point_to_node, nearest_node_indices = nearest_distances(point_xyz, node_xyz)
    node_to_point, _ = nearest_distances(node_xyz, point_xyz)
    nearest_radius = node_radius[nearest_node_indices]
    radius_positive_mask = point_to_node <= nearest_radius
    distance_positive_mask = point_to_node <= positive_distance
    relaxed_positive_mask = point_to_node <= np.maximum(nearest_radius, positive_distance)

    patch_center = np.array(metadata.get("patch_center", [np.nan, np.nan, np.nan]), dtype=np.float32)
    if np.isfinite(patch_center).all():
        point_center_distance = np.linalg.norm(point_xyz - patch_center.reshape(1, 3), axis=1)
        node_center_distance = np.linalg.norm(node_xyz - patch_center.reshape(1, 3), axis=1)
    else:
        point_center_distance = np.array([], dtype=np.float32)
        node_center_distance = np.array([], dtype=np.float32)

    return {
        "record_index": record_index,
        "path": str(path),
        "sample_id": metadata.get("sample_id", ""),
        "patch_index": metadata.get("patch_index"),
        "threshold": metadata.get("threshold"),
        "threshold_fraction": metadata.get("threshold_fraction"),
        "patch_radius": metadata.get("patch_radius"),
        "patch_center": metadata.get("patch_center"),
        "patch_foreground_count": metadata.get("patch_foreground_count"),
        "max_points": metadata.get("max_points"),
        "points": int(points.shape[0]),
        "unique_points": int(unique_xyz.shape[0]),
        "duplicate_fraction": 1.0 - unique_xyz.shape[0] / points.shape[0] if points.shape[0] else 0.0,
        "skeleton_nodes": int(skeleton_nodes.shape[0]),
        "edges": int(edge_index.shape[0]),
        "point_bbox_extent": quantized_extent(point_xyz),
        "node_bbox_extent": quantized_extent(node_xyz),
        "intensity": quantiles(point_intensity),
        "node_radius": quantiles(node_radius),
        "node_radius_le_1": int((node_radius <= 1.0).sum()),
        "point_to_swc": quantiles(point_to_node),
        "swc_to_point": quantiles(node_to_point),
        "points_inside_swc_radius": count_fraction(radius_positive_mask),
        f"points_within_{positive_distance:g}": count_fraction(distance_positive_mask),
        f"points_within_max_radius_or_{positive_distance:g}": count_fraction(relaxed_positive_mask),
        f"swc_nodes_within_{coverage_distance:g}_of_point": count_fraction(node_to_point <= coverage_distance),
        "point_distance_from_patch_center": quantiles(point_center_distance),
        "node_distance_from_patch_center": quantiles(node_center_distance),
    }


def nearest_distances(query: np.ndarray, reference: np.ndarray, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    distances = []
    indices = []
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        diff = chunk[:, None, :] - reference[None, :, :]
        chunk_distances = np.linalg.norm(diff, axis=2)
        chunk_indices = np.argmin(chunk_distances, axis=1)
        distances.append(chunk_distances[np.arange(chunk.shape[0]), chunk_indices])
        indices.append(chunk_indices)
    return np.concatenate(distances), np.concatenate(indices)


def quantized_extent(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    extent = values.max(axis=0) - values.min(axis=0)
    return [round(float(value), 3) for value in extent]


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    qs = np.quantile(values.astype(np.float64, copy=False), [0.0, 0.5, 0.9, 0.99, 1.0])
    return {
        "min": round(float(qs[0]), 4),
        "p50": round(float(qs[1]), 4),
        "p90": round(float(qs[2]), 4),
        "p99": round(float(qs[3]), 4),
        "max": round(float(qs[4]), 4),
    }


def count_fraction(mask: np.ndarray) -> dict[str, float | int]:
    count = int(mask.sum())
    total = int(mask.size)
    return {"count": count, "total": total, "fraction": round(count / total if total else 0.0, 4)}


def print_summary(summary: dict[str, object]) -> None:
    print(
        f"index={summary['record_index']} patch={summary['patch_index']} "
        f"points={summary['points']} unique={summary['unique_points']} "
        f"dup={summary['duplicate_fraction']:.3f} nodes={summary['skeleton_nodes']} "
        f"foreground={summary['patch_foreground_count']} sample_id={summary['sample_id']}"
    )
    for key in (
        "intensity",
        "node_radius",
        "point_to_swc",
        "swc_to_point",
        "point_distance_from_patch_center",
        "node_distance_from_patch_center",
    ):
        print(f"  {key}: {summary[key]}")
    print(
        f"  bbox point={summary['point_bbox_extent']} node={summary['node_bbox_extent']} "
        f"node_radius_le_1={summary['node_radius_le_1']}"
    )
    for key in summary:
        if key.startswith("points_inside") or key.startswith("points_within") or key.startswith("swc_nodes_within"):
            print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    raise SystemExit(main())
