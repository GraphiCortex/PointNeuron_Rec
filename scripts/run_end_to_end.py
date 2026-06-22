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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--render-points", type=int, default=12000)
    args = parser.parse_args()

    sample_indices = requested_sample_indices(args.sample_index, args.sample_range)
    if not sample_indices:
        raise ValueError("Provide --sample-index or --sample-range")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index in sample_indices:
        row = run_sample(
            sample_index=sample_index,
            root=args.root,
            checkpoint=Path(args.checkpoint),
            output_root=output_root,
            existing_proposal_dir=Path(args.existing_proposal_dir),
            force_proposals=args.force_proposals,
            device=args.device,
            render_points=args.render_points,
        )
        rows.append(row)
        append_jsonl(output_root / "summary.jsonl", row)

    write_summary(output_root / "summary.json", rows)
    print(f"samples: {len(rows)}")
    print(f"mean_reachable_edge_fraction: {mean([row['reachable_edge_fraction'] for row in rows]):.4f}")
    print(f"mean_bridge_edges: {mean([row['bridge_edges'] for row in rows]):.4f}")
    print(f"mean_reconstruction_nodes: {mean([row['reconstruction_nodes'] for row in rows]):.1f}")
    print(f"summary: {output_root / 'summary.json'}")
    return 0


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
                "0.2",
                "--patch-radius",
                "96",
                "--stride",
                "96",
                "--max-patches",
                "256",
                "--patch-selection",
                "coverage",
                "--score-threshold",
                "0.5",
                "--top-proposals-per-patch",
                "512",
                "--local-nms-mode",
                "sphere",
                "--global-nms-mode",
                "sphere",
                "--local-nms-radius",
                "8",
                "--global-nms-radius",
                "12",
                "--global-top-proposals",
                "4096",
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

    graph_path = sample_dir / f"{sample_tag}_geodesic_graph.npz"
    swc_path = sample_dir / f"{sample_tag}.swc"
    compare_html = sample_dir / f"{sample_tag}_compare.html"
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
    run_command([sys.executable, "scripts/generate_swc_from_graph.py", "--graph", str(graph_path), "--output", str(swc_path)])
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
        "proposal_path": str(proposal_path),
        "graph_path": str(graph_path),
        "swc_path": str(swc_path),
        "compare_html": str(compare_html),
        "reconstruction_nodes": len(reconstruction.nodes),
        "reconstruction_roots": reconstruction.root_count,
        "reconstruction_edges": reconstruction.edge_count,
        "swc_valid": not reconstruction.validate(),
        "foreground_threshold": graph_metadata["foreground_threshold"],
        "candidate_k": graph_metadata["candidate_k"],
        "max_geodesic_ratio": graph_metadata["max_geodesic_ratio"],
        "bridge_edges": graph_metadata["bridge_edges"],
        "reachable_edge_fraction": graph_metadata["reachable_edge_fraction"],
        "components": graph_metadata["components"],
        "mean_edge_geodesic_distance": graph_metadata["mean_edge_geodesic_distance"],
        "mean_snap_distance": graph_metadata["mean_snap_distance"],
    }
    print(
        f"{sample_tag}: roots={row['reconstruction_roots']} nodes={row['reconstruction_nodes']} "
        f"reachable={row['reachable_edge_fraction']:.4f} bridges={row['bridge_edges']} html={compare_html}"
    )
    return row


def run_command(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_graph_metadata(path: Path) -> dict:
    import numpy as np

    payload = np.load(path, allow_pickle=False)
    return json.loads(str(payload["metadata"]))


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary(path: Path, rows: list[dict]) -> None:
    payload = {
        "baseline_connectivity": BASELINE_CONNECTIVITY,
        "samples": rows,
        "summary": {
            "sample_count": len(rows),
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
