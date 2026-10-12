"""Plot two illustrative saved field pairs at identical probability scales."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    comparison = json.loads((args.control / "loss_comparison.json").read_text())
    original = json.loads((args.baseline / "fit_report.json").read_text())
    controlled = json.loads((args.control / "fit_report.json").read_text())
    if original["config"] != controlled["config"] or original["manifest_sha256"] != controlled["manifest_sha256"]:
        raise ValueError("Reports do not share inputs and settings")
    if not all(comparison["pairing_checks"].values()):
        raise ValueError("Paired checks did not pass")
    # Chosen to illustrate corrected off-reference support and remaining crowding,
    # not to estimate generalization. All eight crop metrics are in the report.
    examples = ((0, 1), (96, 3))
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for row, (sample, patch) in enumerate(examples):
        name = f"sample_{sample:04d}_patch_{patch:03d}.npz"
        with np.load(args.manifest.parent / name, allow_pickle=False) as z:
            valid, reference = z["valid"], (z["distance"] <= .9) & z["valid"]
        yy, xx = np.nonzero(reference.any(axis=0))
        for column, models in enumerate((args.baseline, args.control)):
            with np.load(models / "step_0400" / name, allow_pickle=False) as z:
                probability = np.where(valid, z["probability"].astype(float), 0)
            ax = axes[row, column]
            plot = ax.imshow(probability.max(axis=0), cmap="magma", vmin=0, vmax=1, origin="lower")
            ax.scatter(xx, yy, color="cyan", s=1, label="reference")
            arm = "Original loss" if column == 0 else "Hard-negative loss"
            ax.set_title(f"{arm} | neuron {sample}, crop {patch}")
            ax.set_xlabel("local X voxel")
            ax.set_ylabel("local Y voxel")
    fig.colorbar(plot, ax=axes, label="Predicted probability, XY maximum projection", shrink=.75)
    fig.suptitle("Same training crops, initialization and 400 updates\nCyan: reference trace; identical 0–1 display scale; projections are not connectivity results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--control", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    main(p.parse_args())
