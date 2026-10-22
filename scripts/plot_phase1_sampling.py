"""Plot fixed-probe summary with all seed values and explicit case accounting."""
import argparse
import json
import os
from pathlib import Path
import textwrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-id", help="Explicit case for a report containing multiple scored images")
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
    case_id = args.case_id if args.case_id is not None else (cases[0] if len(cases) == 1 else None)
    if case_id not in cases:
        raise ValueError("Select an explicit scored --case-id for a multi-image report")
    tolerance = str(float(report["recipe"]["evaluation"]["primary_tolerance_voxel"]))
    methods, colors = ["uniform", "intensity", "sato"], ["#7c8795", "#e7a23b", "#7e57bd"]
    lean, dense = max(report["recipe"]["cell_sides_voxel"]), min(report["recipe"]["cell_sides_voxel"])
    panels = [(lean, "curve_support", "Lean: traced-curve support", True),
              (dense, "curve_support", "Dense: traced-curve support", True),
              (lean, "worst_branch_support", "Lean: worst-branch support", True),
              (lean, "longest_unsupported_run", "Lean: longest unsupported run", False)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (cell, field, title, percentage) in zip(axes.flat, panels):
        for i, (method, color) in enumerate(zip(methods, colors)):
            values = [r["metrics"]["tolerances"][tolerance][field] for r in report["rows"]
                      if r["case_id"] == case_id and r["method"] == method and r["cell_side_voxel"] == cell]
            values = np.asarray(values) * (100 if percentage else 1)
            ax.bar(i, values.mean(), color=color, alpha=.75)
            ax.scatter(i + np.linspace(-.12, .12, len(values)), values, color="#202124", s=18, zorder=3)
        ax.set_xticks(range(3), methods)
        ax.set(title=title, ylabel=f"Percent within {float(tolerance):g} native voxels" if percentage else "Native voxel arclength")
        if percentage:
            ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    declared = len(report["references"]) + len(report["failures"])
    title = case_id.rsplit("/", 1)[-1] + " input-point support: means and seed values"
    fig.suptitle("\n".join(textwrap.wrap(title, 85)) +
                 f"\n{len(cases)} of {declared} declared training images scorable; reconstruction untested", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
