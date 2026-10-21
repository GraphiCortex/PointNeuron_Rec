"""CPU-only image decoding and complete tile inventory. No point/model decisions."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .cases import load_image_cases
from .image import decode_image
from .tiling import TileConfig, TileGrid


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_inputs(cases_path: str | Path, config_path: str | Path, output_dir: str | Path):
    """Write all tile boxes to JSONL in a fresh directory; keep every input case.

    Empty/nonzero describe raw voxel values only, not foreground or neuron
    presence. Geometric tile coverage cannot establish branch retention.
    Images are decoded in memory one at a time. No CUDA/PyTorch is imported.
    """
    cases_path, config_path, output_dir = (Path(p).resolve() for p in (cases_path, config_path, output_dir))
    cases, config = load_image_cases(cases_path), TileConfig.load(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running", "stage": "decode_and_tile_inventory",
        "created_utc": datetime.now(timezone.utc).isoformat(), "device": "CPU",
        "scope": "Image-only input audit; no reference reads, foreground rule, sampling, model or reconstruction",
        "cases_manifest": str(cases_path), "cases_manifest_sha256": _digest(cases_path),
        "tiling_config": str(config_path), "tiling_config_sha256": _digest(config_path),
        "settings": {"tile_size_xyz": list(config.tile_size_xyz), "stride_xyz": list(config.stride_xyz),
                     "bounds": "start inclusive, stop exclusive", "tile_order": "X fastest, then Y, then Z",
                     "last_boundary": "final tile anchored at image edge", "tile_cap": None,
                     "threshold": None, "nonzero_interpretation": "raw signal only; not certified foreground"},
        "limitations": ["Whole decoded image is held in memory; streaming/out-of-core decoding is not implemented",
                        "Partition roles are caller declarations, not independently verified dataset identities",
                        "Spacing is declared metadata; calibration and radius conventions are not certified"],
        "source_sha256": {p.name: _digest(p) for p in Path(__file__).parent.glob("*.py")},
        "cases": []}
    for index, case in enumerate(cases):
        began = time.perf_counter()
        row = {"case_id": case.case_id, "partition": case.partition, "image_path": str(case.image_path),
               "status": "running", "tiles_written": 0, "tiles_expected": None,
               "model_processed_tiles": 0, "tile_signal_counts": {}}
        counts = Counter()
        image, values = None, None
        try:
            image = decode_image(case)
            row["image"] = image.provenance
            grid = TileGrid(image.dimensions_xyz, config)
            row["tiles_expected"] = grid.tile_count
            tile_path = output_dir / f"case_{index:04d}_tiles.jsonl"
            row["tiles_file"] = tile_path.name
            with tile_path.open("x", encoding="utf-8") as handle:
                for tile in grid:
                    values = image.array_zyx[tile.slices_zyx]
                    nonzero = int(np.count_nonzero(values))
                    signal = "nonzero_signal" if nonzero else "all_zero"
                    counts[signal] += 1
                    item = {"tile_id": tile.tile_id, "start_xyz": list(tile.start_xyz),
                        "stop_xyz": list(tile.stop_xyz), "shape_xyz": list(tile.shape_xyz),
                        "center_xyz": list(tile.center_xyz), "global_xyz_origin": list(tile.start_xyz),
                        "voxel_count": int(values.size), "nonzero_voxels": nonzero,
                        "minimum": int(values.min()), "maximum": int(values.max()),
                        "signal_status": signal, "status": "inventoried"}
                    handle.write(json.dumps(item, allow_nan=False) + "\n")
                    row["tiles_written"] += 1
            if row["tiles_written"] != row["tiles_expected"]:
                raise RuntimeError("Tile inventory is incomplete")
            row.update(status="decoded_and_tiled", geometric_coverage="complete_by_grid_construction")
        except NotImplementedError as exc:
            row.update(status="unsupported", error_type=type(exc).__name__, error=str(exc))
        except (OSError, ValueError, MemoryError) as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        finally:
            # Release the last tile view and image even after a partial inventory failure.
            # Otherwise the next image can be decoded while the failed image is still resident.
            del values, image
        row.update(elapsed_seconds=time.perf_counter() - began, tile_signal_counts=dict(counts))
        report["cases"].append(row)
        print(f"INPUT case={case.case_id} status={row['status']} tiles={row['tiles_written']}/{row['tiles_expected']}", flush=True)
    report["case_status_counts"] = dict(Counter(row["status"] for row in report["cases"]))
    report["status"] = "completed" if all(row["status"] == "decoded_and_tiled" for row in report["cases"]) else "failed"
    with (output_dir / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report
