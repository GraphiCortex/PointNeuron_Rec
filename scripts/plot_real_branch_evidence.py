"""Plot the saved audit only; no image decode, model or rescore."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def metric(arm, key, width="all", contrast="all", radius=4.):
    rows = [r for r in arm[key] if r["width_class"] == width and r["contrast_class"] == contrast
            and r["radius_voxel"] == radius]
    return rows[0]["arc_weighted_retained_fraction"] if rows else None


def plot(path):
    root = Path(path).resolve().parent
    os.environ.setdefault("MPLCONFIGDIR", str(root / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(Path(path).read_text())
    methods = ("uniform", "intensity", "sato")
    colors = ("#64748b", "#15803d", "#7c3aed")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for row, case in enumerate(report["cases"]):
        windows = [w for b in case["boxes"] for w in b["windows"]]
        resolved = [w for w in windows if w["apparent_width_voxel"] is not None
                    and w["inner_peak_contrast"] is not None and w["inner_peak_contrast"] > 0]
        ax = axes[row, 0]
        for box, marker in zip(case["boxes"], ("o", "^")):
            points = [w for w in box["windows"] if w in resolved]
            ax.scatter([w["apparent_width_voxel"] for w in points], [w["inner_peak_contrast"] for w in points],
                       s=20, alpha=.6, marker=marker, label=f"Box {box['tile_id']}")
        ax.axvline(3, color="#94a3b8", ls="--", lw=1)
        ax.axvline(5, color="#94a3b8", ls="--", lw=1)
        for cutoff in case["contrast_terciles_raw_units"]:
            if cutoff is not None and cutoff > 0:
                ax.axhline(cutoff, color="#94a3b8", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("Apparent profile width (native voxels)")
        ax.set_ylabel("Inner peak minus local background (raw units)")
        thin_dim = sum(w["width_class"] == "thin" and w["contrast_class"] == "dim" for w in windows)
        ax.set_title(f"{case['alias']}: {len(windows)} windows; thin/dim n={thin_dim}\n{len(windows)-len(resolved)} unresolved/nonpositive profiles")
        ax.legend(fontsize=8)
        for column, key in ((1, "input_summary"), (2, "first_group_conditional_summary")):
            ax = axes[row, column]
            for j, (method, color) in enumerate(zip(methods, colors)):
                means, lower, upper = [], [], []
                for box in case["boxes"]:
                    arms = [a for a in box["arms"] if a["method"] == method
                            and (column == 1 or a["seed"] == 0)]
                    values = [metric(a, key) for a in arms]
                    values = [100*v for v in values if v is not None]
                    means.append(np.mean(values) if values else np.nan)
                    lower.append(np.mean(values)-min(values) if values else 0)
                    upper.append(max(values)-np.mean(values) if values else 0)
                x = np.arange(len(case["boxes"]))+(j-1)*.24
                ax.bar(x, means, width=.22, color=color, label=method,
                       yerr=np.array([lower, upper]), capsize=3)
            ax.set_xticks(np.arange(len(case["boxes"])), [f"Box {b['tile_id']}" for b in case["boxes"]])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Arc-weighted local fraction (%)")
            ax.set_title("Input positive contrast mass retained, R=4\nMean + range over five seeds" if column == 1 else
                         "Selected mass reached by first-stage balls, R=4\nConditional on input mass; seed zero")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.15)
    fig.suptitle("Frozen real clouds: image evidence and geometric access\nFully image-derived input selection; training-only diagnostic", fontsize=15)
    fig.supxlabel("Apparent width is not anatomy. Positive mass includes noise. Fine-level skips retain access to all input points.", fontsize=10)
    target = root / "summary.png"
    if target.exists():
        raise FileExistsError(target)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    plot(args.report)
