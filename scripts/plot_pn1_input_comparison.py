"""Render the saved matched 2048-slot comparison; no decode or rescore."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def plot(path):
    path = Path(path).resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads(path.read_text())
    methods = ["pn1_native", "pn1_decode_only", "pn1_range_random", "pn1_range_spatial",
               "pn2_uniform", "pn2_intensity", "pn2_sato"]
    labels = ["PN1\nlegacy", "PN1\n+decode", "PN1\nrange R", "PN1\nrange S",
              "PN2\nuniform", "PN2\nintensity", "PN2\nSato"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (case, box) in zip(axes.ravel(), [(c, b) for c in report["cases"] for b in c["boxes"]]):
        series = [[], []]; errors = [[], []]; actual_labels = []
        for method, label in zip(methods, labels):
            arms = [a for a in box["arms"] if a["method"] == method and a["requested_slots"] == 2048]
            assert len(arms) == 5
            star = any(a["unique_count"] < 2048 for a in arms)
            actual_labels.append(label+("*" if star else ""))
            curve = [a["curve_metrics"]["2.0"]["curve_support"]*100 for a in arms]
            mass = [next(s["arc_weighted_retained_fraction"] for s in a["contrast_strata"]
                         if s["width_class"] == "all" and s["contrast_class"] == "all" and s["radius_voxel"] == 4)*100 for a in arms]
            for i, values in enumerate((curve, mass)):
                mean = float(np.mean(values)); series[i].append(mean)
                errors[i].append([mean-min(values), max(values)-mean])
        x = np.arange(len(methods))
        for i, (color, label) in enumerate((("#2563eb", "Traced-curve proximity within 2 voxels"),
                                          ("#ea580c", "Positive contrast mass retained, radius 4"))):
            bars = ax.bar(x+(i-.5)*.36, series[i], width=.34, color=color,
                          yerr=np.asarray(errors[i]).T, capsize=2, label=label)
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in series[i]], fontsize=8, padding=3)
        ax.set_xticks(x, actual_labels, fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Local input-evidence metric (%)")
        ax.set_title(f"{case['alias']} / box {box['tile_id']} — same 2,048 requested slots")
        ax.grid(axis="y", alpha=.15)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .045), ncol=2, frameon=False)
    fig.suptitle("PN1 versus PN2 input components: means and ranges over five seeds\nSame source files, boxes and measurement windows; no model training", fontsize=15)
    fig.text(.5, .018, "* Fewer distinct voxels because the PN1 pool is empty or draws repeat. R=random; S=spatial. Input support is not reconstruction accuracy.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .1, 1, .93])
    target = path.parent / "comparison_2048.png"
    if target.exists():
        raise FileExistsError(target)
    fig.savefig(target, dpi=150); plt.close(fig)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    plot(args.report)
