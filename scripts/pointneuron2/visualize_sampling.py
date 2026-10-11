"""Project cached image candidates, image guides and SWC reference labels for inspection."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from pointneuron.data.sampling_cache import load_sampling_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--partition", type=Path, default=REPO_ROOT / "configs/pointneuron2/partition_seed0.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new figure path")
    manifest = load_sampling_manifest(args.manifest, args.partition)
    selected = []
    for role in ("train", "val"):
        selected.append(next(row for row in manifest["records"] if row["partition"] == role and row["label_nodes"] > 1))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for r, row in enumerate(selected):
        with np.load(row["resolved_path"]) as cached:
            points, guide, nodes = cached["points"], cached["image_guide_xyz"], cached["skeleton_nodes"]
        lookup = {int(node[0]): node[1:4] for node in nodes}
        for c, (a, b) in enumerate(((0, 1), (0, 2))):
            ax = axes[r, c]
            ax.scatter(points[:, a], points[:, b], color="#999999", s=3, alpha=.4, label="Image candidate pool")
            ax.scatter(guide[:, a], guide[:, b], color="#e59519", s=6, alpha=.8, label="Image medial guide")
            for node in nodes:
                parent = lookup.get(int(node[5]))
                if parent is not None:
                    ax.plot([parent[a], node[a + 1]], [parent[b], node[b + 1]], color="#d43136", lw=1.2)
            ax.scatter(nodes[:, a + 1], nodes[:, b + 1], color="#d43136", s=7, label="Reference SWC labels")
            ax.set_title(f"{row['partition']} neuron {row['sample_index']}, patch {row['patch_index']} — {'XY' if c == 0 else 'XZ'}")
            ax.set_xlabel("X (voxels)"); ax.set_ylabel("Y (voxels)" if c == 0 else "Z (voxels)")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(fontsize=7)
    fig.suptitle("Bounded image pools and local reference labels\nTwo annotated patches selected for inspection; projections do not verify full 3D alignment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
