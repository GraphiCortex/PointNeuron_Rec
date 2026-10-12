"""Render two saved training-crop field checks; no inference or model changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    report = json.loads((args.models / "fit_report.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    if report["status"] != "completed" or report["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Completed fit/cache mismatch")
    if report["config"] != {**manifest["config"], "steps": 1600, "eval_steps": [0, 400, 800, 1600]}:
        raise ValueError("Expected the approved duration-only fit")
    if [e["step"] for e in report["evaluations"]] != [0, 400, 800, 1600]:
        raise ValueError("Missing saved evaluation")
    records = {(r["sample_index"], r["patch_index"]): r for r in manifest["records"]}
    # Illustrative established cases; all eight final overlays accompany the report.
    examples = ((0, 1), (96, 3))
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for row, identity in enumerate(examples):
        record = records[identity]
        path = (args.manifest.parent / record["path"]).resolve()
        if (record["partition"] != "train" or not path.is_relative_to(args.manifest.parent.resolve())
                or digest(path) != record["sha256"]):
            raise ValueError("Changed or invalid training record")
        with np.load(path, allow_pickle=False) as z:
            valid = z["valid"]
            reference = (z["distance"] <= .9) & valid
        yy, xx = np.nonzero(reference.any(axis=0))
        for col, step in enumerate((400, 1600)):
            with np.load(args.models / f"step_{step:04d}" / path.name, allow_pickle=False) as z:
                probability = z["probability"].astype(float)
            if (probability.shape != valid.shape or not np.isfinite(probability).all()
                    or probability.min() < 0 or probability.max() > 1):
                raise ValueError("Invalid saved probability field")
            ax = axes[row, col]
            plot = ax.imshow(np.where(valid, probability, 0).max(axis=0), cmap="magma",
                             vmin=0, vmax=1, origin="lower")
            ax.scatter(xx, yy, color="cyan", s=1)
            measured = next(e for e in report["evaluations"] if e["step"] == step)
            crop = next(c for c in measured["crops"] if (c["sample_index"], c["patch_index"]) == identity)
            ax.set_title(f"Replay update {step} | crop {identity[0]}/{identity[1]}\nTube Dice {crop['known_voxel_dice']:.3f}")
            ax.set_xlabel("local X voxel")
            ax.set_ylabel("local Y voxel")
    fig.colorbar(plot, ax=axes, label="Predicted probability, XY maximum projection", shrink=.75)
    fig.suptitle("One training run: 400 versus 1600 updates\nCyan: reference; fixed 0–1 scale; projections do not establish connectivity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
