from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.swc import parse_swc


BASELINE_CONNECTIVITY = {
    "foreground_threshold": 5,
    "max_foreground_voxels": 350000,
    "candidate_k": 6,
    "max_geodesic_ratio": 12.0,
    "bridge_components": True,
    "bridge_allow_unreachable_fallback": True,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen PointNeuron baseline end to end.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--sample-index", type=int, action="append", help="Sample index. Can be repeated.")
    parser.add_argument("--sample-range", help="Inclusive range like 100-109.")
    parser.add_argument("--checkpoint", default="tmp/checkpoints/proposal_paper_skeleton_aug_50e.pt")
    parser.add_argument("--output-root", default="tmp/e2e_baseline_janelia")
    parser.add_argument("--existing-proposal-dir", default="tmp/paper_skeleton_eval_50e_heldout")
    parser.add_argument("--force-proposals", action="store_true", help="Regenerate proposals even if an existing proposal file is present.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first sample failure instead of recording it and continuing.")
    parser.add_argument("--initializer", default="geodesic", choices=["geodesic", "image_supported"], help="Connectivity graph initializer.")
    parser.add_argument("--proposal-threshold-fraction", type=float, default=0.2, help="Normalized foreground threshold used by proposal aggregation.")
    parser.add_argument("--proposal-score-threshold", type=float, default=0.5, help="Objectness threshold for local proposal selection.")
    parser.add_argument("--proposal-top-per-patch", type=int, default=512, help="Maximum local proposals retained per patch before global NMS.")
    parser.add_argument("--proposal-global-top", type=int, default=4096, help="Maximum aggregated proposals after global NMS.")
    parser.add_argument("--proposal-local-nms-radius", type=float, default=8.0, help="Local proposal NMS radius.")
    parser.add_argument("--proposal-global-nms-radius", type=float, default=12.0, help="Global proposal NMS radius.")
    parser.add_argument("--point-sample-strategy", default="random", choices=["random", "spatial"], help="How foreground points are sampled inside each proposal patch.")
    parser.add_argument("--point-sample-cell-size", type=int, default=8, help="Voxel cell size for --point-sample-strategy spatial.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--render-points", type=int, default=12000)
    args = parser.parse_args()

    sample_indices = requested_sample_indices(args.sample_index, args.sample_range)
    if not sample_indices:
        raise ValueError("Provide --sample-index or --sample-range")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / ("summary.json" if args.initializer == "geodesic" else f"summary_{args.initializer}.json")
    summary_jsonl_path = output_root / ("summary.jsonl" if args.initializer == "geodesic" else f"summary_{args.initializer}.jsonl")
    rows = []
    failures = []
    for sample_index in sample_indices:
        try:
            row = run_sample(
                sample_index=sample_index,
                root=args.root,
                checkpoint=Path(args.checkpoint),
                output_root=output_root,
                existing_proposal_dir=Path(args.existing_proposal_dir),
                force_proposals=args.force_proposals,
                initializer=args.initializer,
                proposal_threshold_fraction=args.proposal_threshold_fraction,
                proposal_score_threshold=args.proposal_score_threshold,
                proposal_top_per_patch=args.proposal_top_per_patch,
                proposal_global_top=args.proposal_global_top,
                proposal_local_nms_radius=args.proposal_local_nms_radius,
                proposal_global_nms_radius=args.proposal_global_nms_radius,
                point_sample_strategy=args.point_sample_strategy,
                point_sample_cell_size=args.point_sample_cell_size,
                device=args.device,
                render_points=args.render_points,
            )
        except Exception as exc:
            failure = sample_failure(sample_index, exc)
            failures.append(failure)
            append_jsonl(summary_jsonl_path, failure)
            write_summary(summary_path, rows, failures, requested_count=len(sample_indices))
            print(f"FAILED {failure['sample_tag']}: {failure['error_type']}: {failure['error']}")
            if args.fail_fast:
                raise
            continue
        rows.append(row)
        append_jsonl(summary_jsonl_path, row)
        write_summary(summary_path, rows, failures, requested_count=len(sample_indices))

    write_summary(summary_path, rows, failures, requested_count=len(sample_indices))
    print(f"requested_samples: {len(sample_indices)}")
    print(f"succeeded_samples: {len(rows)}")
    print(f"failed_samples: {len(failures)}")
    print(f"mean_reachable_edge_fraction: {mean([row['reachable_edge_fraction'] for row in rows]):.4f}")
    print(f"mean_bridge_edges: {mean([row['bridge_edges'] for row in rows]):.4f}")
    print(f"mean_reconstruction_nodes: {mean([row['reconstruction_nodes'] for row in rows]):.1f}")
    print(f"summary: {summary_path}")
    return 1 if failures else 0


def requested_sample_indices(sample_indices: list[int] | None, sample_range: str | None) -> list[int]:
    values: list[int] = []
    if sample_indices:
        values.extend(sample_indices)
    if sample_range:
        start_text, end_text = sample_range.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Invalid --sample-range: {sample_range}")
        values.extend(range(start, end + 1))
    return sorted(dict.fromkeys(values))


def run_sample(
    sample_index: int,
    root: str,
    checkpoint: Path,
    output_root: Path,
    existing_proposal_dir: Path,
    force_proposals: bool,
    initializer: str,
    proposal_threshold_fraction: float,
    proposal_score_threshold: float,
    proposal_top_per_patch: int,
    proposal_global_top: int,
    proposal_local_nms_radius: float,
    proposal_global_nms_radius: float,
    point_sample_strategy: str,
    point_sample_cell_size: int,
    device: str,
    render_points: int,
) -> dict:
    sample_tag = f"sample_{sample_index:04d}"
    sample_dir = output_root / sample_tag
    sample_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = sample_dir / f"{sample_tag}_proposals.npz"
    existing_proposal = existing_proposal_dir / f"{sample_tag}_proposals.npz"
    if existing_proposal.exists() and not force_proposals:
        proposal_path = existing_proposal
    elif proposal_path.exists() and not force_proposals:
        print(f"reuse proposal: {proposal_path}")
    else:
        proposal_html = sample_dir / f"{sample_tag}_proposals.html"
        run_command(
            [
                sys.executable,
                "scripts/aggregate_proposals.py",
                "--root",
                root,
                "--sample-index",
                str(sample_index),
                "--checkpoint",
                str(checkpoint),
                "--threshold-fraction",
                str(proposal_threshold_fraction),
                "--patch-radius",
                "96",
                "--stride",
                "96",
                "--max-patches",
                "256",
                "--patch-selection",
                "coverage",
                "--score-threshold",
                str(proposal_score_threshold),
                "--top-proposals-per-patch",
                str(proposal_top_per_patch),
                "--point-sample-strategy",
                point_sample_strategy,
                "--point-sample-cell-size",
                str(point_sample_cell_size),
                "--local-nms-mode",
                "sphere",
                "--global-nms-mode",
                "sphere",
                "--local-nms-radius",
                str(proposal_local_nms_radius),
                "--global-nms-radius",
                str(proposal_global_nms_radius),
                "--global-top-proposals",
                str(proposal_global_top),
                "--render-points",
                str(render_points),
                "--device",
                device,
                "--output",
                str(proposal_path),
                "--html-output",
                str(proposal_html),
            ]
        )

    graph_path = sample_dir / f"{sample_tag}_{initializer}_graph.npz"
    swc_path = sample_dir / f"{sample_tag}_{initializer}.swc"
    compare_html = sample_dir / f"{sample_tag}_{initializer}_compare.html"
    if graph_path.exists():
        print(f"reuse graph: {graph_path}")
    else:
        if initializer == "geodesic":
            run_command(
                [
                    sys.executable,
                    "scripts/initialize_geodesic_graph.py",
                    "--root",
                    root,
                    "--sample-index",
                    str(sample_index),
                    "--proposals",
                    str(proposal_path),
                    "--mode",
                    "mst",
                    "--nms-distance",
                    "12",
                    "--max-nodes",
                    "256",
                    "--foreground-threshold",
                    str(BASELINE_CONNECTIVITY["foreground_threshold"]),
                    "--max-foreground-voxels",
                    str(BASELINE_CONNECTIVITY["max_foreground_voxels"]),
                    "--candidate-k",
                    str(BASELINE_CONNECTIVITY["candidate_k"]),
                    "--max-geodesic-ratio",
                    str(BASELINE_CONNECTIVITY["max_geodesic_ratio"]),
                    "--bridge-components",
                    "--bridge-allow-unreachable-fallback",
                    "--output",
                    str(graph_path),
                ]
            )
        elif initializer == "image_supported":
            run_command(
                [
                    sys.executable,
                    "scripts/initialize_image_supported_graph.py",
                    "--root",
                    root,
                    "--sample-index",
                    str(sample_index),
                    "--proposals",
                    str(proposal_path),
                    "--mode",
                    "mst",
                    "--nms-distance",
                    "12",
                    "--max-nodes",
                    "256",
                    "--foreground-threshold",
                    str(BASELINE_CONNECTIVITY["foreground_threshold"]),
                    "--sample-step",
                    "2",
                    "--empty-penalty",
                    "8",
                    "--output",
                    str(graph_path),
                ]
            )
        else:
            raise ValueError(f"Unknown initializer: {initializer}")
    if swc_path.exists():
        print(f"reuse swc: {swc_path}")
    else:
        run_command([sys.executable, "scripts/generate_swc_from_graph.py", "--graph", str(graph_path), "--output", str(swc_path)])
    if compare_html.exists():
        print(f"reuse html: {compare_html}")
    else:
        run_command(
            [
                sys.executable,
                "scripts/visualize_reconstruction.py",
                "--root",
                root,
                "--sample-index",
                str(sample_index),
                "--reconstruction-swc",
                str(swc_path),
                "--render-points",
                str(render_points),
                "--output",
                str(compare_html),
            ]
        )

    graph_metadata = load_graph_metadata(graph_path)
    reconstruction = parse_swc(swc_path)
    row = {
        "sample_index": sample_index,
        "sample_tag": sample_tag,
        "initializer": initializer,
        "proposal_path": str(proposal_path),
        "graph_path": str(graph_path),
        "swc_path": str(swc_path),
        "compare_html": str(compare_html),
        "reconstruction_nodes": len(reconstruction.nodes),
        "reconstruction_roots": reconstruction.root_count,
        "reconstruction_edges": reconstruction.edge_count,
        "swc_valid": not reconstruction.validate(),
        "foreground_threshold": graph_metadata["foreground_threshold"],
        "requested_foreground_threshold": graph_metadata.get("requested_foreground_threshold", graph_metadata["foreground_threshold"]),
        "foreground_threshold_was_adapted": graph_metadata.get("foreground_threshold_was_adapted", False),
        "max_foreground_voxels": graph_metadata.get("max_foreground_voxels", 0),
        "foreground_cap_satisfied": graph_metadata.get("foreground_cap_satisfied", True),
        "candidate_k": graph_metadata.get("candidate_k", 0),
        "max_geodesic_ratio": graph_metadata.get("max_geodesic_ratio", 0.0),
        "bridge_edges": graph_metadata.get("bridge_edges", 0),
        "reachable_edge_fraction": graph_metadata.get("reachable_edge_fraction", 1.0),
        "components": graph_metadata["components"],
        "mean_edge_geodesic_distance": graph_metadata.get("mean_edge_geodesic_distance", graph_metadata.get("mean_edge_weight", 0.0)),
        "mean_snap_distance": graph_metadata.get("mean_snap_distance", 0.0),
    }
    print(
        f"{sample_tag}: roots={row['reconstruction_roots']} nodes={row['reconstruction_nodes']} "
        f"reachable={row['reachable_edge_fraction']:.4f} bridges={row['bridge_edges']} html={compare_html}"
    )
    return row


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def sample_failure(sample_index: int, exc: Exception) -> dict:
    failure = {
        "status": "failed",
        "sample_index": int(sample_index),
        "sample_tag": f"sample_{sample_index:04d}",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if isinstance(exc, subprocess.CalledProcessError):
        failure["returncode"] = int(exc.returncode)
        failure["command"] = " ".join(str(part) for part in exc.cmd)
    return failure


def load_graph_metadata(path: Path) -> dict:
    import numpy as np

    payload = np.load(path, allow_pickle=False)
    return json.loads(str(payload["metadata"]))


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary(path: Path, rows: list[dict], failures: list[dict], requested_count: int) -> None:
    payload = {
        "baseline_connectivity": BASELINE_CONNECTIVITY,
        "samples": rows,
        "failures": failures,
        "summary": {
            "requested_sample_count": int(requested_count),
            "sample_count": len(rows),
            "success_count": len(rows),
            "failure_count": len(failures),
            "mean_reachable_edge_fraction": mean([row["reachable_edge_fraction"] for row in rows]),
            "mean_bridge_edges": mean([row["bridge_edges"] for row in rows]),
            "mean_reconstruction_nodes": mean([row["reconstruction_nodes"] for row in rows]),
            "all_swc_valid": all(row["swc_valid"] for row in rows),
            "all_single_root": all(row["reconstruction_roots"] == 1 for row in rows),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
