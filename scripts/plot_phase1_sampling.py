"""Plot fixed-probe summary with all seed values and explicit case accounting."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing figure")
    os.environ.setdefault("MPLCONFIGDIR", str(args.report.parent / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads(args.report.read_text(encoding="utf-8"))
    cases = list(dict.fromkeys(r["case_id"] for r in report["summary"]))
    if len(cases) != 1:
        raise ValueError("This figure layout describes the one-scorable-case diagnostic")
    tolerance = str(float(report["recipe"]["evaluation"]["primary_tolerance_voxel"]))
    methods, colors = ["uniform", "intensity", "sato"], ["#7c8795", "#e7a23b", "#7e57bd"]
    panels = [(8, "curve_support", "Lean: traced-curve support", True),
              (4, "curve_support", "Dense: traced-curve support", True),
              (8, "worst_branch_support", "Lean: worst-branch support", True),
              (8, "longest_unsupported_run", "Lean: longest unsupported run", False)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (cell, field, title, percentage) in zip(axes.flat, panels):
        for i, (method, color) in enumerate(zip(methods, colors)):
            values = [r["metrics"]["tolerances"][tolerance][field] for r in report["rows"]
                      if r["method"] == method and r["cell_side_voxel"] == cell]
            values = np.asarray(values) * (100 if percentage else 1)
            ax.bar(i, values.mean(), color=color, alpha=.75)
            ax.scatter(i + np.linspace(-.12, .12, len(values)), values, color="#202124", s=18, zorder=3)
        ax.set_xticks(range(3), methods)
        ax.set(title=title, ylabel="Percent within 2 native voxels" if percentage else "Native voxel arclength")
        if percentage:
            ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Series021 input-point support: means and all five seeds\n"
                 "1 of 2 declared training images scorable; reconstruction untested", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
