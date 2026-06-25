from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter cached proposal-training records to patches whose input points can reasonably supervise the SWC target."
    )
    parser.add_argument(
        "--cache-manifest",
        action="append",
        required=True,
        help="Path to a cache_manifest.json. May be repeated to merge candidate caches.",
    )
    parser.add_argument("--output", required=True, help="Output filtered cache manifest.")
    parser.add_argument("--hit-distance", type=float, default=8.0, help="Distance threshold for point/SWC hit fractions.")
    parser.add_argument(
        "--min-point-hit",
        type=float,
        default=0.10,
        help="Keep records with at least this fraction of sampled points within --hit-distance of the SWC.",
    )
    parser.add_argument(
        "--min-swc-hit",
        type=float,
        default=0.10,
        help="Keep records with at least this fraction of SWC nodes within --hit-distance of sampled points.",
    )
    parser.add_argument(
        "--max-mean-point-distance",
        type=float,
        help="Optional maximum mean distance from sampled points to the nearest SWC node.",
    )
    parser.add_argument(
        "--max-records-per-sample",
        type=int,
        help="Optional cap after filtering; keeps the strongest records per sample_index.",
    )
    parser.add_argument(
        "--sort-key",
        default="quality_score",
        choices=["quality_score", "point_hit", "swc_hit", "mean_point_distance"],
        help="Ranking key used with --max-records-per-sample.",
    )
    args = parser.parse_args()

    manifests = [Path(path) for path in args.cache_manifest]
    records = load_records(manifests)
    enriched = [summarize_record(record, hit_distance=args.hit_distance) for record in records]
    kept = [
        row
        for row in enriched
        if row["quality"]["point_hit"] >= args.min_point_hit
        and row["quality"]["swc_hit"] >= args.min_swc_hit
        and (
            args.max_mean_point_distance is None
            or row["quality"]["mean_point_distance"] <= args.max_mean_point_distance
        )
    ]
    if args.max_records_per_sample is not None:
        kept = cap_records_per_sample(kept, args.max_records_per_sample, args.sort_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source_manifests": [str(path) for path in manifests],
                "quality_filter": {
                    "hit_distance": args.hit_distance,
                    "min_point_hit": args.min_point_hit,
                    "min_swc_hit": args.min_swc_hit,
                    "max_mean_point_distance": args.max_mean_point_distance,
                    "max_records_per_sample": args.max_records_per_sample,
                    "sort_key": args.sort_key,
                },
                "records": [row["record"] for row in kept],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"source_records: {len(enriched)}")
    print(f"kept_records: {len(kept)}")
    print(f"output: {output_path}")
    print()
    print_group("all_source", enriched)
    print_group("all_kept", kept)
    for domain in sorted({row["domain"] for row in enriched}):
        print_group(f"{domain}_source", [row for row in enriched if row["domain"] == domain])
        print_group(f"{domain}_kept", [row for row in kept if row["domain"] == domain])
    print_worst(enriched, count=8)
    return 0


def load_records(manifests: list[Path]) -> list[dict[str, object]]:
    by_path: dict[str, dict[str, object]] = {}
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in payload["records"]:
            path = str(record["path"])
            if path not in by_path:
                by_path[path] = dict(record)
    return list(by_path.values())


def summarize_record(record: dict[str, object], hit_distance: float) -> dict[str, object]:
    path = Path(str(record["path"]))
    data = np.load(path, allow_pickle=False)
    points = data["points"].astype(np.float32, copy=False)
    skeleton_nodes = data["skeleton_nodes"].astype(np.float32, copy=False)
    metadata = json.loads(str(data["metadata"]))

    point_xyz = points[:, :3]
    node_xyz = skeleton_nodes[:, 1:4]
    point_to_swc = nearest_distances(point_xyz, node_xyz)
    swc_to_point = nearest_distances(node_xyz, point_xyz)
    point_hit = float((point_to_swc <= hit_distance).mean()) if point_to_swc.size else 0.0
    swc_hit = float((swc_to_point <= hit_distance).mean()) if swc_to_point.size else 0.0
    mean_point_distance = float(point_to_swc.mean()) if point_to_swc.size else float("inf")
    mean_swc_distance = float(swc_to_point.mean()) if swc_to_point.size else float("inf")
    quality_score = point_hit * swc_hit

    enriched_record = dict(record)
    enriched_record["quality"] = {
        "point_hit": round(point_hit, 6),
        "swc_hit": round(swc_hit, 6),
        "mean_point_distance": round(mean_point_distance, 6),
        "mean_swc_distance": round(mean_swc_distance, 6),
        "quality_score": round(quality_score, 6),
    }
    return {
        "record": enriched_record,
        "domain": domain_name(str(metadata.get("sample_id", ""))),
        "sample_index": int(metadata.get("sample_index", record.get("sample_index", -1))),
        "path": str(path),
        "quality": enriched_record["quality"],
    }


def nearest_distances(query: np.ndarray, reference: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    distances = []
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        diff = chunk[:, None, :] - reference[None, :, :]
        distances.append(np.linalg.norm(diff, axis=2).min(axis=1))
    return np.concatenate(distances).astype(np.float32, copy=False)


def cap_records_per_sample(
    rows: list[dict[str, object]],
    max_records_per_sample: int,
    sort_key: str,
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_index"]), []).append(row)

    reverse = sort_key != "mean_point_distance"
    capped = []
    for sample_rows in grouped.values():
        ranked = sorted(sample_rows, key=lambda row: float(row["quality"][sort_key]), reverse=reverse)
        capped.extend(ranked[:max_records_per_sample])
    return sorted(capped, key=lambda row: (int(row["sample_index"]), str(row["path"])))


def domain_name(sample_id: str) -> str:
    lower = sample_id.lower()
    if "janelia" in lower or "flylight" in lower:
        return "janelia_like"
    return "other_gold"


def print_group(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        print(f"{name}: n=0")
        return
    samples = sorted({int(row["sample_index"]) for row in rows})
    print(f"{name}: n={len(rows)} samples={len(samples)}")
    for key in ["point_hit", "swc_hit", "mean_point_distance", "mean_swc_distance", "quality_score"]:
        values = [float(row["quality"][key]) for row in rows]
        print(
            f"  {key}: mean={statistics.fmean(values):.4f} "
            f"p10={quantile(values, 0.10):.4f} p50={quantile(values, 0.50):.4f} p90={quantile(values, 0.90):.4f}"
        )


def print_worst(rows: list[dict[str, object]], count: int) -> None:
    print("worst_source_records:")
    for row in sorted(rows, key=lambda item: float(item["quality"]["quality_score"]))[:count]:
        quality = row["quality"]
        print(
            f"  sample {row['sample_index']}: point_hit={quality['point_hit']:.4f} "
            f"swc_hit={quality['swc_hit']:.4f} mean_point_distance={quality['mean_point_distance']:.2f} "
            f"path={row['path']}"
        )


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.array(values, dtype=np.float64), q))


if __name__ == "__main__":
    raise SystemExit(main())
