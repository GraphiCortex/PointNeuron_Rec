"""Inspect explicit original images before segmentation; no SWCs or model imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.inputs.cases import load_image_cases
from pn2.inputs.image import decode_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory; preserved outputs")
    args = parser.parse_args()
    cases = load_image_cases(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / "matplotlib_cache"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = {"scope": "Raw image inspection; no segmentation, point selection, reference or model",
              "display": "Full native range, linear and log1p display; no clipping or source transformation",
              "cases": []}
    for number, case in enumerate(cases):
        image = decode_image(case)
        values = image.array_zyx
        width, height, depth = image.dimensions_xyz
        xy, xz, yz = values.max(axis=0), values.max(axis=1), values.max(axis=2)
        maximum = int(values.max())
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        views = [(xy, "XY maximum projection / linear", "X", "Y", False),
                 (xy, "XY maximum projection / log1p", "X", "Y", True),
                 (xz, "XZ maximum projection / log1p", "X", "Z", True),
                 (yz, "YZ maximum projection / log1p", "Y", "Z", True),
                 (values[depth // 2], f"XY source slice z={depth // 2} / log1p", "X", "Y", True)]
        for ax, (view, title, horizontal, vertical, logarithmic) in zip(axes.flat, views):
            display = np.log1p(view.astype(float)) if logarithmic else view
            limit = np.log1p(maximum) if logarithmic else maximum
            ax.imshow(display, cmap="gray", vmin=0, vmax=max(float(limit), 1.), origin="lower", aspect="equal")
            ax.set(title=title, xlabel=f"{horizontal} (native voxel)", ylabel=f"{vertical} (native voxel)")
            ax.tick_params(labelsize=8)
        hist, edges = np.histogram(values, bins=256, range=(0, max(maximum + 1, 1)))
        ax = axes.flat[-1]
        ax.stairs(hist, edges)
        ax.set_yscale("log")
        ax.set(title="Raw intensity histogram (all voxels)", xlabel="Native intensity", ylabel="Voxel count / log scale")
        fig.suptitle("\n".join(textwrap.wrap(case.case_id, 95)) +
                     f"\nXYZ={width} x {height} x {depth}; native {values.dtype}; max={maximum}; no mask", fontsize=10)
        figure = args.output_dir / f"raw_{number:02d}.png"
        fig.savefig(figure, dpi=160)
        plt.close(fig)
        report["cases"].append({"case_id": case.case_id, "image": image.provenance,
            "figure": figure.name, "minimum": int(values.min()), "maximum": maximum,
            "mean": float(values.mean()), "zero_fraction": float(np.count_nonzero(values == 0) / values.size),
            "quantiles": {str(q): float(v) for q, v in zip([.5, .9, .99, .999], np.quantile(values, [.5, .9, .99, .999]))}})
        print(f"RAW case={case.case_id} XYZ={image.dimensions_xyz} range={int(values.min())}..{maximum}", flush=True)
        del image, values, xy, xz, yz, views, view, display
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Raw inspection: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
