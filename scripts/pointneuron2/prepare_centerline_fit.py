"""Prepare eight native image crops, weak labels and visible alignment overlays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pointneuron.data.centerline_field import SCHEMA, crop_volume, segment_distance_field, weak_labels
from pointneuron.data.query_cache import file_digest, load_query_manifest
from pointneuron.data.swc import parse_swc
from pointneuron.data.training_cache import volume_data_array
from pointneuron.data.vaa3d_raw import read_volume, VOLUME_READER_VERSION


def inspection(image, distance, valid, unknown, title, output, probability=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_scale = float(np.quantile(image[valid], .995))
    if display_scale <= 0:  # Sparse images can be >99.5% zero.
        display_scale = float(image[valid].max())
    display_scale = max(1e-6, display_scale)
    display = np.clip(image / display_scale, 0, 1)
    fig, axes = plt.subplots(1, 3 if probability is None else 4, figsize=(14 if probability is None else 18, 4))
    for ax, axis, label in zip(axes[:2], (0, 1), ("XY maximum projection", "XZ maximum projection")):
        ax.imshow(display.max(axis=axis), cmap="gray", vmin=0, vmax=1, origin="lower")
        yy, xx = np.nonzero(((distance <= .9) & valid).any(axis=axis))
        ax.scatter(xx, yy, s=1, color="red", alpha=.8)
        ax.set_title(label + "\nred = reference centerline")
    # Projection is illustrative; different depths can overlap in this panel.
    panel = np.zeros((*image.shape[1:], 3), dtype=float)
    panel[:] = display.max(axis=0)[..., None]
    panel[unknown.any(axis=0)] = [0, .8, 1]
    axes[2].imshow(panel, origin="lower")
    axes[2].set_title("XY weak-label uncertainty\ncyan = bright, distant, excluded")
    if probability is not None:
        axes[3].imshow(np.where(valid, probability, 0).max(axis=0), cmap="magma", vmin=0, vmax=1, origin="lower")
        axes[3].set_title("Predicted field XY\nfixed probability scale 0–1")
    for ax in axes:
        ax.set_xlabel("local X voxel")
        ax.set_ylabel("local Y / Z voxel")
    fig.suptitle(title + f"\nImage display white level {display_scale:.4f} (contrast only); network values unchanged")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def prepare(args):
    cfg = json.loads(args.config.read_text())
    source = load_query_manifest(args.manifest, args.partition)
    wanted = [tuple(x) for x in cfg["crops"]]
    if len(wanted) != 8 or len(set(wanted)) != 8 or len({x[0] for x in wanted}) != 4:
        raise ValueError("Fit contract requires eight unique crops from four training neurons")
    rows = {(r["sample_index"], r["patch_index"]): r for r in source["records"]}
    if any(x not in rows or rows[x]["partition"] != "train" for x in wanted):
        raise ValueError("Fixed learnability crops must be training-only; no validation/test")
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh cache directory")
    manifest = {"schema": SCHEMA, "status": "running", "scope": "fixed-training-crop learnability only",
                "volume_reader_version": VOLUME_READER_VERSION, "config": cfg,
                "config_sha256": file_digest(args.config), "partition_sha256": file_digest(args.partition),
                "source_manifest_sha256": file_digest(args.manifest), "records": []}
    args.output_dir.mkdir(parents=True)
    previous, volume, tree = None, None, None
    for identity in wanted:
        row = rows[identity]
        with np.load(row["resolved_path"], allow_pickle=False) as source_record:
            meta = json.loads(str(source_record["metadata"]))
            reference = source_record["reference_curve"].copy()
        directory = (args.data_root / row["sample_id"]).resolve()
        if not directory.is_relative_to(args.data_root.resolve()):
            raise ValueError("Source path escapes dataset")
        if previous != row["sample_id"]:
            volumes = [p for p in sorted(directory.iterdir()) if p.suffix in (".v3dpbd", ".v3draw")
                       and file_digest(p) == meta["volume_sha256"]]
            swcs = [p for p in sorted(directory.glob("*.swc")) if file_digest(p) == meta["swc_sha256"]]
            if len(volumes) != 1 or len(swcs) != 1:
                raise ValueError("Cannot uniquely recover original hashed image/reference")
            decoded = read_volume(volumes[0])
            x, y, z, channels = decoded.dimensions
            if channels != 1:
                raise ValueError("Only single-channel images are supported")
            volume = volume_data_array(decoded).reshape(z, y, x)
            tree = parse_swc(swcs[0])
            previous = row["sample_id"]
        raw, valid, origin = crop_volume(volume, meta["patch_center"], meta["config"]["patch_radius"])
        distance = segment_distance_field(tree, origin, raw.shape, cfg["uncertainty_distance_voxels"])
        target, known, unknown = weak_labels(raw, valid, distance, meta["threshold"],
                                            cfg["tube_radius_voxels"], cfg["uncertainty_distance_voxels"])
        if not target.any() or not known.any():
            raise ValueError("Requested fixed crop lacks usable native-grid trace labels")
        # Preserve all image values. Normalize by observed full-image maximum,
        # not storage bit depth and not the previous foreground threshold.
        scale = float(volume.max())
        if scale <= 0:
            raise ValueError("Empty source image")
        image = raw.astype(np.float32) / scale
        relative = reference - origin
        inside = np.all((relative >= 0) & (relative <= np.array(raw.shape[::-1])-1), axis=1)
        relative = relative[inside]
        if not len(relative):
            raise ValueError("No in-bounds continuous reference samples")
        record_meta = {**meta, "field_schema": SCHEMA, "origin_xyz": origin.tolist(),
                       "shape_zyx": list(raw.shape), "image_scale": scale,
                       "coordinate_units": "native acquisition voxels, unrescaled",
                       "annotation_policy": "tube-positive; far bright unknown; dark/near-trace weak negatives"}
        name = f"sample_{identity[0]:04d}_patch_{identity[1]:03d}"
        path = args.output_dir / (name + ".npz")
        np.savez_compressed(path, image=image, target=target, known=known, valid=valid,
                            unknown_bright=unknown, distance=distance, reference_local_xyz=relative,
                            metadata=np.array(json.dumps(record_meta)))
        centerline = (distance <= .9) & valid
        report = {"path": path.name, "sha256": file_digest(path), "sample_index": identity[0],
                  "sample_id": row["sample_id"], "patch_index": identity[1], "partition": "train",
                  "valid_voxels": int(valid.sum()), "positive_voxels": int(target.sum()),
                  "unknown_bright_voxels": int(unknown.sum()),
                  "reference_above_old_cutoff_fraction": float((raw[centerline] > meta["threshold"]).mean()),
                  "reference_mean_intensity_normalized": float(image[centerline].mean()),
                  "volume_sha256": meta["volume_sha256"], "swc_sha256": meta["swc_sha256"]}
        manifest["records"].append(report)
        inspection(image, distance, valid, unknown, f"Training neuron {identity[0]}, crop {identity[1]} | native XYZ/ZYX, no coordinate adjustment",
                   args.output_dir / "inspection" / (name + ".png"))
        print(f"prepared neuron={identity[0]} crop={identity[1]} shape={raw.shape} positives={target.sum()} "
              f"bright_unknown={unknown.sum()} reference_above_old_cutoff={report['reference_above_old_cutoff_fraction']:.3f}", flush=True)
    manifest["status"] = "completed"
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {args.output_dir / 'manifest.json'}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pointneuron2/centerline_fit.json")
    parser.add_argument("--partition", type=Path, default=ROOT / "configs/pointneuron2/partition_seed0.json")
    prepare(parser.parse_args())
