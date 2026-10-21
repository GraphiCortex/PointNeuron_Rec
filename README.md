# PN2

This is the home for all fresh PN2 development, in its own Git worktree on
branch `pointneuron2`. It is a sibling of the historical `pointneuron2` folder.
The original PN1 checkout remains on `main`.

Start with the [revised proposal](docs/revised_proposal.md) and
[evidence audit](docs/research_reset_evidence.md).

## Layout

| Folder | Purpose |
|---|---|
| `src/` | New implementation, introduced through reviewed changes |
| `scripts/` | New commands and experiment entry points |
| `configs/` | New benchmark manifests and reviewed run settings |
| `tests/` | Checks for the new implementation and evaluator |
| `docs/` | Active proposal, design decisions and readable results |
| `tmp/` | Ignored generated caches, runs, checkpoints and logs |

All new work goes here. The sibling historical worktree's sources, reports and scripts are
retained reference material; reusing them will be explicit. No old experiment
or checkpoint is adopted automatically.

## Active direction: original raw images to PN2

Scope clarified 22 October 2026: PN1 informs conceptual design. Rebuilding,
training or fixing a PN1 reference is not a prerequisite for the product.

```text
Original raw image -> automatic PN2 point preparation -> PointNeXt
-> Transformer proposals -> consolidation -> GATv2 -> graph/SWC
```

Start with the [PN2 Phase 1 design draft and original-image views](docs/pn2_phase1_design.md).
The verified decoder, coordinate/tiling interface, source inventories and common
SWC evaluator remain useful infrastructure. Two original training files have now
been inspected directly, without segmentation, reference content or model input
selection. No PN2 foreground method, guide or sampler is promoted by those views.

The PN1-only cloud/model detour has been removed from active source. Reports
and pushed commit `48519ff` preserve it. Its DGCNN singleton issue is not the
next PN2 task. No PN1 cutoff, weights or point budget become PN2 defaults.

The next decision is the proposed soft structural guide and finite-budget cloud
construction, before implementing it. Keep execution, representation retention,
training learnability and genuine reconstruction evidence distinct.

## Working directory and runtime

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
```

Reuse the retained environment at
`..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe`. It has not been moved or
reinstalled. Original images, PN1 assets and the historical evidence archive
remain outside this fresh folder. New generated outputs belong in this folder's ignored `tmp/`.

The parent collaboration instructions still apply: discuss substantive changes,
use visible commands for agreed GPU runs, and distinguish execution from
reconstruction quality. No training or inference is launched by this setup.

To evaluate saved reconstructions, use the [evaluation command](scripts/README.md).
The command reads explicit development/synthetic SWC paths and never runs a
model. The base package uses NumPy and SciPy. Raw inspection uses the existing
Matplotlib runtime. No model library is imported by image/evaluation commands.
