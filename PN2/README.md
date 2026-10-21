# PN2

This is the home for all fresh PN2 development. It lives inside the existing
`pointneuron2` Git worktree and uses that branch. The original PN1 checkout
remains on `main`.

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

All new work goes here. The parent worktree's sources, reports and scripts are
retained reference material; reusing them will be explicit. No old experiment
or checkpoint is adopted automatically.

The next milestone is the shared whole-neuron evaluator, source/exposure
manifest and repaired PN1-style reference specification. No new training
command or model implementation has been created by this folder setup.

## Working directory and runtime

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\PN2"
```

Reuse the retained environment at
`..\tmp\pointneuron2\sampling_env\Scripts\python.exe`. It has not been moved or
reinstalled. Original images, PN1 assets and the historical evidence archive
remain outside this fresh folder. New generated outputs belong in `PN2/tmp/`.

The parent collaboration instructions still apply: discuss substantive changes,
use visible commands for agreed GPU runs, and distinguish execution from
reconstruction quality. No training or inference is launched by this setup.
