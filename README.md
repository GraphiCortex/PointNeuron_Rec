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

Milestone A has started: the shared SWC evaluator, an explicit seven-neuron
development manifest and the repaired PN1-style reference specification are
implemented. Read the [benchmark contract](docs/benchmark_contract.md),
[reference specification](docs/baseline_reference.md) and
[verification record](docs/evaluator_verification.md).

The first [image-only input interface](docs/image_input_interface.md) now reuses
the verified decoder and inventories complete covering tiles. It has passed
synthetic checks and one real training-image CPU compatibility check. It does
not yet select model points or produce reconstructions.

This establishes measurement infrastructure. The fresh PN1 reference and PN2
candidate still need reviewed recipes and automatic whole-volume integration
before training. Structural correctness is not inferred from curve geometry.

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
model. The standalone package uses NumPy and SciPy; the retained model library
is not imported.
