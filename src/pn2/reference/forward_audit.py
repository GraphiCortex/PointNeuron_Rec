"""Image-to-proposal execution control with fresh untrained PN1-style parameters."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from pn2.inputs.cases import load_image_cases
from pn2.inputs.clouds import ReferenceCloudBuilder, ReferenceCloudConfig
from pn2.inputs.image import decode_image
from pn2.inputs.tiling import TileConfig, TileGrid
from .model import fresh_reference, predict_points, state_digest


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_reference_forward(cases_path, tiling_path, clouds_path, output_dir):
    cases_path, tiling_path, clouds_path, output_dir = (Path(p).resolve() for p in
        (cases_path, tiling_path, clouds_path, output_dir))
    cases, tiling = load_image_cases(cases_path), TileConfig.load(tiling_path)
    clouds = ReferenceCloudConfig.load(clouds_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    model = fresh_reference(clouds.seed)
    initial = state_digest(model)
    source_root = Path(__file__).parents[1]
    report = {"schema_version": 1, "stage": "fresh_untrained_reference_forward", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "device": "CPU",
        "scope": "Execution only; no learned checkpoint, targets, optimization, filtering, graph or SWC",
        "model": {"identity": "retained_local_PN1_DGCNN_pointwise_local_head", "trained": False,
                  "seed": clouds.seed, "initialization": "conservative_identity_head",
                  "normalization": "ordinary_eval_running_BatchNorm_statistics_fresh_defaults",
                  "initial_state_sha256": initial, "proposal_filter": None},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "cpu_threads": torch.get_num_threads()},
        "manifests": {name: {"path": str(path), "sha256": _sha(path)} for name, path in
                      (("cases", cases_path), ("tiling", tiling_path), ("clouds", clouds_path))},
        "cloud_config": clouds.as_dict(), "tiling": {"tile_size_xyz": list(tiling.tile_size_xyz),
                                                        "stride_xyz": list(tiling.stride_xyz)},
        "source_sha256": {str(path.relative_to(source_root)): _sha(path)
                          for folder in (source_root / "inputs", source_root / "reference")
                          for path in sorted(folder.glob("*.py"))},
        "cases": []}
    for case_index, case in enumerate(cases):
        began = time.perf_counter()
        row = {"case_id": case.case_id, "partition": case.partition, "status": "running",
               "image_path": str(case.image_path), "tiles_expected": None, "tiles_accounted": 0,
               "model_processed_tiles": 0, "pointwise_proposals": 0, "model_forward_seconds": 0.}
        counts, totals = Counter(), Counter()
        image, builder, cloud, points, output = None, None, None, None, None
        try:
            image = decode_image(case)
            row["image"] = image.provenance
            builder = ReferenceCloudBuilder(image, clouds)
            grid = TileGrid(image.dimensions_xyz, tiling)
            row["tiles_expected"] = grid.tile_count
            directory = output_dir / f"case_{case_index:04d}"
            directory.mkdir()
            row["tiles_file"] = str((directory / "tiles.jsonl").relative_to(output_dir))
            with (directory / "tiles.jsonl").open("x", encoding="utf-8") as handle:
                for tile in grid:
                    cloud = builder.build(tile)
                    item = dict(cloud.metadata)
                    arrays = {"points_xyzi": cloud.points_xyzi, "selected_flat_indices": cloud.selected_flat_indices}
                    if item["cloud_status"] == "ready":
                        points = torch.from_numpy(cloud.points_xyzi)[None]
                        start = time.perf_counter()
                        output = predict_points(model, points)
                        row["model_forward_seconds"] += time.perf_counter() - start
                        for field in ("center_proposals", "offsets", "radius", "objectness_logits"):
                            tensor = getattr(output, field)
                            if not torch.isfinite(tensor).all():
                                raise FloatingPointError(f"Nonfinite reference output: {field}")
                            arrays[field] = tensor[0].numpy()
                        item["model_status"] = "forward_completed"
                        row["model_processed_tiles"] += 1
                        row["pointwise_proposals"] += len(cloud.points_xyzi)
                    elif item["cloud_status"] == "singleton_cloud":
                        item["model_status"] = "unsupported_singleton"
                    else:
                        item["model_status"] = "empty_input"
                    item["packet"] = f"tile_{tile.tile_id:06d}.npz"
                    np.savez_compressed(directory / item["packet"], **arrays)
                    handle.write(json.dumps(item, allow_nan=False) + "\n")
                    counts[item["model_status"]] += 1
                    totals["foreground_voxels"] += item["foreground_voxels"]
                    totals["selected_points"] += item["selected_points"]
                    row["tiles_accounted"] += 1
            if row["tiles_accounted"] != row["tiles_expected"]:
                raise RuntimeError("Reference forward did not account for every tile")
            row["status"] = "unsupported" if counts["unsupported_singleton"] else "completed"
        except NotImplementedError as exc:
            row.update(status="unsupported", error_type=type(exc).__name__, error=str(exc))
        except (OSError, ValueError, MemoryError, FloatingPointError) as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        finally:
            del image, builder, cloud, points, output
        row.update(tile_status_counts=dict(counts), overlapping_tile_totals=dict(totals),
                   elapsed_seconds=time.perf_counter() - began)
        report["cases"].append(row)
        print(f"REFERENCE_INPUT case={case.case_id} status={row['status']} "
              f"tiles={row['tiles_accounted']}/{row['tiles_expected']} forward={row['model_processed_tiles']}", flush=True)
    final = state_digest(model)
    report["model"].update(final_state_sha256=final, state_unchanged=(initial == final))
    if final != initial:
        raise RuntimeError("Read-only forward altered model parameters or normalization buffers")
    report["status"] = "completed" if all(row["status"] == "completed" for row in report["cases"]) else "failed"
    report["case_status_counts"] = dict(Counter(row["status"] for row in report["cases"]))
    with (output_dir / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report
