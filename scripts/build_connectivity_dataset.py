from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.data.gold166 import scan_gold166


def main() -> int:
    parser = argparse.ArgumentParser(description="Build full-sample PointNeuron connectivity records for Gold166 samples.")
    parser.add_argument("--root", default="data/gold166", help="Gold166 root.")
    parser.add_argument("--split-file", help="Sample/cache split JSON. sample_XXXX paths are converted to Gold166 indices.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to build when --split-file is used.")
    parser.add_argument("--sample-index", type=int, action="append", help="Explicit sample index. May be repeated.")
    parser.add_argument("--max-samples", type=int, help="Optional cap for smoke/batched runs.")
    parser.add_argument("--checkpoint", required=True, help="Trained proposal checkpoint.")
    parser.add_argument("--app2-exe", help="Path to Vaa3D-x.exe. Required when APP2 output is missing unless --skip-app2 is set.")
    parser.add_argument("--skip-app2", action="store_true", help="Do not run APP2; require existing APP2 SWCs.")
    parser.add_argument("--app2-timeout-seconds", type=float, default=600.0, help="Maximum seconds to wait for APP2 per sample before marking it failed. 0 disables.")
    parser.add_argument("--output-root", default="tmp/connectivity_dataset", help="Root for generated aggregations, APP2 SWCs, graphs, records, and manifest.")
    parser.add_argument("--threshold-fraction", type=float, default=0.2)
    parser.add_argument("--patch-radius", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-patches", type=int, default=128)
    parser.add_argument("--patch-selection", default="coverage", choices=["coverage", "density"])
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--top-proposals-per-patch", type=int, default=256)
    parser.add_argument("--local-nms-mode", default="distance", choices=["sphere", "distance"])
    parser.add_argument("--local-nms-radius", type=float, default=10.0)
    parser.add_argument("--global-nms-mode", default="distance", choices=["sphere", "distance"])
    parser.add_argument("--global-nms-radius", type=float, default=18.0)
    parser.add_argument("--global-top-proposals", type=int, default=768)
    parser.add_argument("--max-center-foreground-distance", type=float, default=4.0)
    parser.add_argument("--foreground-support-radius", type=float, default=8.0)
    parser.add_argument("--min-foreground-support", type=int, default=12)
    parser.add_argument("--max-proposal-init-distance", type=float, default=20.0)
    parser.add_argument("--adaptive-tree-base-distance", type=float, default=16.0)
    parser.add_argument("--adaptive-tree-min-distance", type=float, default=8.0)
    parser.add_argument("--adaptive-tree-max-distance", type=float, default=28.0)
    parser.add_argument("--adaptive-tree-density-radius", type=float, default=32.0)
    parser.add_argument("--init-mode", default="mst_knn", choices=["mst", "knn", "mst_knn"])
    parser.add_argument("--init-knn", type=int, default=3)
    parser.add_argument("--init-max-tree-distance", type=float, default=100.0)
    parser.add_argument("--target-mode", default="mst", choices=["mst", "knn", "mst_knn"])
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing outputs.")
    parser.add_argument("--no-html", action="store_true", help="Skip aggregation HTML visualizations.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    args = parser.parse_args()

    samples = scan_gold166(args.root)
    sample_indices = selected_sample_indices(args)
    if args.max_samples is not None:
        sample_indices = sample_indices[: args.max_samples]
    if not sample_indices:
        print("No sample indices selected.")
        return 2

    output_root = Path(args.output_root)
    aggregations_dir = output_root / "aggregations"
    app2_dir = output_root / "app2"
    graphs_dir = output_root / "graphs"
    records_dir = output_root / "records"
    visualizations_dir = output_root / "visualizations"
    logs_dir = output_root / "logs"
    for directory in (aggregations_dir, app2_dir, graphs_dir, records_dir, visualizations_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []
    for ordinal, sample_index in enumerate(sample_indices, start=1):
        sample = samples[sample_index]
        prefix = f"sample_{sample_index:04d}"
        aggregation_path = aggregations_dir / f"{prefix}_proposals.npz"
        app2_path = app2_dir / f"{prefix}_app2.swc"
        graph_path = graphs_dir / f"{prefix}_adaptive_mstknn_graph.npz"
        record_path = records_dir / f"{prefix}_connectivity.npz"
        html_path = visualizations_dir / f"{prefix}_proposals.html"
        sample_logs_dir = logs_dir / prefix
        sample_logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{ordinal}/{len(sample_indices)}] sample_index={sample_index} sample_id={sample.sample_id}", flush=True)
        try:
            run_step(
                output=aggregation_path,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                log_path=sample_logs_dir / "aggregate_proposals.log",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "aggregate_proposals.py"),
                    "--root",
                    args.root,
                    "--sample-index",
                    str(sample_index),
                    "--checkpoint",
                    args.checkpoint,
                    "--threshold-fraction",
                    str(args.threshold_fraction),
                    "--patch-radius",
                    str(args.patch_radius),
                    "--stride",
                    str(args.stride),
                    "--max-patches",
                    str(args.max_patches),
                    "--patch-selection",
                    args.patch_selection,
                    "--max-points",
                    str(args.max_points),
                    "--batch-size",
                    str(args.batch_size),
                    "--score-threshold",
                    str(args.score_threshold),
                    "--top-proposals-per-patch",
                    str(args.top_proposals_per_patch),
                    "--local-nms-mode",
                    args.local_nms_mode,
                    "--local-nms-radius",
                    str(args.local_nms_radius),
                    "--global-nms-mode",
                    args.global_nms_mode,
                    "--global-nms-radius",
                    str(args.global_nms_radius),
                    "--global-top-proposals",
                    str(args.global_top_proposals),
                    "--max-center-foreground-distance",
                    str(args.max_center_foreground_distance),
                    "--foreground-support-radius",
                    str(args.foreground_support_radius),
                    "--min-foreground-support",
                    str(args.min_foreground_support),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                    "--output",
                    str(aggregation_path),
                    "--html-output",
                    "" if args.no_html else str(html_path),
                ],
            )

            if not app2_path.exists() or not args.skip_existing:
                if args.skip_app2:
                    raise FileNotFoundError(f"Missing APP2 SWC and --skip-app2 was set: {app2_path}")
                if not args.app2_exe:
                    raise ValueError("--app2-exe is required when APP2 output is missing")
                run_app2(
                    app2_exe=Path(args.app2_exe),
                    volume_path=sample.volume_path,
                    output_path=app2_path,
                    dry_run=args.dry_run,
                    log_path=sample_logs_dir / "app2.log",
                    timeout_seconds=args.app2_timeout_seconds,
                )
            if not args.dry_run:
                require_output(app2_path, "APP2 SWC")

            run_step(
                output=graph_path,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                log_path=sample_logs_dir / "initialize_proposal_graph.log",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "initialize_proposal_graph.py"),
                    "--proposals",
                    str(aggregation_path),
                    "--init-swc",
                    str(app2_path),
                    "--mode",
                    args.init_mode,
                    "--knn",
                    str(args.init_knn),
                    "--max-tree-distance",
                    str(args.init_max_tree_distance),
                    "--max-proposal-init-distance",
                    str(args.max_proposal_init_distance),
                    "--adaptive-tree-nms",
                    "--adaptive-tree-base-distance",
                    str(args.adaptive_tree_base_distance),
                    "--adaptive-tree-min-distance",
                    str(args.adaptive_tree_min_distance),
                    "--adaptive-tree-max-distance",
                    str(args.adaptive_tree_max_distance),
                    "--adaptive-tree-density-radius",
                    str(args.adaptive_tree_density_radius),
                    "--output",
                    str(graph_path),
                ],
            )

            run_step(
                output=record_path,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                log_path=sample_logs_dir / "build_connectivity_record.log",
                command=[
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "build_connectivity_record.py"),
                    "--init-graph",
                    str(graph_path),
                    "--use-ground-truth",
                    "--root",
                    args.root,
                    "--sample-index",
                    str(sample_index),
                    "--target-mode",
                    args.target_mode,
                    "--output",
                    str(record_path),
                ],
            )

            if record_path.exists():
                metadata = npz_metadata(record_path)
                records.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": sample.sample_id,
                        "record": str(record_path),
                        "aggregation": str(aggregation_path),
                        "app2_swc": str(app2_path),
                        "graph": str(graph_path),
                        "nodes": metadata.get("nodes"),
                        "init_edges": metadata.get("init_edges"),
                        "target_edges": metadata.get("target_edges"),
                        "candidate_recall": metadata.get("edge_recall"),
                        "logs": str(sample_logs_dir),
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "sample_index": sample_index,
                    "sample_id": sample.sample_id,
                    "error": str(exc),
                    "logs": str(sample_logs_dir),
                }
            )
            print(f"FAILED sample_index={sample_index}: {exc}", flush=True)

    manifest = {
        "root": args.root,
        "split_file": args.split_file,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "output_root": str(output_root),
        "records": records,
        "failures": failures,
        "counts": {
            "requested": len(sample_indices),
            "records": len(records),
            "failures": len(failures),
        },
    }
    manifest_path = output_root / f"{args.split}_connectivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print(f"records: {len(records)}")
    print(f"failures: {len(failures)}")
    return 1 if failures else 0


def selected_sample_indices(args: argparse.Namespace) -> list[int]:
    if args.sample_index:
        return list(dict.fromkeys(args.sample_index))
    if not args.split_file:
        raise ValueError("Provide --split-file or at least one --sample-index")
    payload = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    paths = payload["splits"][args.split]
    indices = []
    for path in paths:
        match = re.search(r"sample_(\d+)\.npz$", str(path).replace("\\", "/"))
        if match is None:
            raise ValueError(f"Could not parse sample index from split path: {path}")
        indices.append(int(match.group(1)))
    return indices


def run_step(output: Path, skip_existing: bool, dry_run: bool, log_path: Path, command: list[str]) -> None:
    if skip_existing and output.exists():
        require_output(output, "existing step output")
        print(f"skip existing: {output}", flush=True)
        return
    print("run:", command_preview(command), flush=True)
    if dry_run:
        return
    run_logged(command=command, cwd=REPO_ROOT, log_path=log_path)
    require_output(output, "step output")


def run_app2(
    app2_exe: Path,
    volume_path: Path | None,
    output_path: Path,
    dry_run: bool,
    log_path: Path,
    timeout_seconds: float,
) -> None:
    if volume_path is None:
        raise ValueError("Sample has no volume path for APP2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    app2_exe = app2_exe.resolve()
    volume_path = volume_path.resolve()
    output_path = output_path.resolve()
    command = [
        str(app2_exe),
        "/x",
        "vn2",
        "/f",
        "app2",
        "/i",
        str(volume_path),
        "/o",
        str(output_path),
        "/p",
        "NULL",
        "0",
        "AUTO",
        "0",
        "0",
        "0",
        "0",
        "5",
        "1",
        "0",
    ]
    print("run:", command_preview(command), flush=True)
    if dry_run:
        return
    timeout = float(timeout_seconds) if timeout_seconds and timeout_seconds > 0.0 else None
    run_logged(command=command, cwd=app2_exe.parent, log_path=log_path, timeout=timeout)
    require_output(output_path, "APP2 SWC")


def run_logged(command: list[str], cwd: Path, log_path: Path, timeout: float | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        log_payload = {
            "command": command,
            "cwd": str(cwd),
            "timeout_seconds": timeout,
            "timed_out": True,
            "stdout": normalize_timeout_stream(exc.stdout),
            "stderr": normalize_timeout_stream(exc.stderr),
        }
        log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
        raise TimeoutError(f"Command timed out after {timeout:.1f} seconds; see {log_path}") from exc
    log_payload = {
        "command": command,
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "timed_out": False,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}; see {log_path}")


def normalize_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def require_output(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"Empty {label}: {path}")


def npz_metadata(path: Path) -> dict:
    payload = np.load(path, allow_pickle=False)
    if "metadata" not in payload:
        return {}
    return json.loads(str(payload["metadata"]))


def command_preview(command: list[str]) -> str:
    return " ".join(quote_part(part) for part in command)


def quote_part(part: str) -> str:
    if " " in part or "\t" in part:
        return f'"{part}"'
    return part


if __name__ == "__main__":
    raise SystemExit(main())
