# Ready: approved 1,600-update duration-only field check

12 October 2026. The user approved this bounded follow-up to the
[ideal-input oracle](centerpath_oracle_results.md). The oracle shows useful
geometry is feasible for two selected crops; it is not learned performance.
This new run tests whether the current recipe needs more optimization.
**The full 1,600-update fit has not been launched during preparation.**

## Exactly what changes

Freshly replay the completed hard-negative recipe for **1,600 rather than 400
updates**, with saved checks at **0 / 400 / 800 / 1,600**. Update 1,600 is the
primary endpoint; update 400 is the common-budget comparison. The saved
100/200 checks from the earlier run remain historical observations.

The [duration recipe](../../configs/pointneuron2/centerline_duration.json)
contains only the approved update/evaluation budget. The immutable prepared
cache's original config is retained; the trainer records the effective config
in the new report and checkpoints.

Everything else is fixed:

- Eight training crops: 0/1, 0/2, 32/0, 32/3, 96/0, 96/3, 162/1, 162/2.
- Same hashed 97³ native image inputs, reference tubes and unknown masks.
- Same 88,673-parameter U-Net with GroupNorm, base channels 8, seed 0.
- Same hard-negative BCE using top min(N,3P) known negatives; same positive BCE and Dice.
- Batch 1, no augmentation, AdamW LR 0.001 / weight decay 0.0001, gradient clip 10.
- Same CUDA/PyTorch version, probability threshold 0.5 and final gate.
- No development-validation or reserved-test neurons, crop merging or final SWC run.

This is a **fresh replay**, not a resume from the old 400-update checkpoint,
which lacks optimizer state. One optimizer is created and kept for the full
new fit, including across the 400/800 evaluation snapshots. AdamW moments are
not reset at those checks. No architecture, learning-rate schedule or label
change is bundled with the duration test.

## Pairing and interpretation

The launcher reuses the completed hard-negative model folder. Before updating,
the trainer verifies cache/config/device/Torch/loss, checkpoint contracts,
exact initial parameter tensors and all eight initial inference measurements.
It verifies that the baseline's recorded crop order matches the seeded
algorithm, then records the new actual order and requires its first 400
indices to match the completed run.

The original report and initial/final checkpoints are hashed and checked for
preservation. `duration_comparison.json` reports each crop and mean geometry
at saved 400, replay 400, replay 800 and replay 1,600, including both common-budget
drift and the within-replay 400-to-1,600 effect. Matching settings does not
automatically prove identical trained numerical results; the common-budget
metrics are checked and their observed differences are saved explicitly.

The final gate remains: every crop has nonempty predictions, >=95% reference
sample coverage within six voxels and >=90% predicted-voxel reference proximity
within six voxels; mean known-region hard Dice >=0.80. The existing hard-negative
400-update run fails it (mean Dice 0.6084, one crop proximity 0.6998).

If the final gate fails, stop and inspect the trajectory and all crop overlays.
Do not automatically extend training, tune the threshold or launch validation.
A pass only establishes fit on selected training crops; it does not establish
generalization, accurate SWC topology or improvement over PN 1.0.

## User-visible local command

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_centerline.ps1 -Action duration
```

The default device is the **RTX 3070 Ti Laptop GPU**. Live losses print every
20 updates, followed by all eight crop measurements at each saved check.
Budget roughly **7–9 minutes** at the measured current speed, including imports
and plotting. The bounded preflight measured 0.253 seconds per update and
estimated **415 seconds** for updates and crop evaluation, before plotting and
startup; peak allocated GPU memory was **636 MiB**. This is slower than the
earlier run used for the initial 3–5 minute estimate, so the current measurement
is the more useful planning figure.

Outputs are under a fresh ignored directory:

```text
tmp/pointneuron2/local_runs/centerline_duration_<timestamp>_<PID>/
  console.log
  model/
    fit_report.json
    duration_comparison.json
    field_step_0000.pt / 0400.pt / 0800.pt / 1600.pt
    step_0000/ / step_0400/ / step_0800/ / step_1600/
```

Step folders contain eight probability arrays and eight readable overlays.
Prior results remain in their existing directories. The comparison report
prints both pairing status and measured metrics; an execution pass is distinct
from the learnability gate.

Defaults use the prepared cache at
`tmp/pointneuron2/local_runs/centerline_prepare_20261012/cache/manifest.json`
and completed hard-negative baseline at
`tmp/pointneuron2/local_runs/centerline_loss_control_20261012_203344_14488/model`.
Alternative paths may be supplied with `-Manifest` and `-BaselineModels`,
subject to the same checks.

`-Action duration-preflight` performs only four throwaway updates with no
trained checkpoint retained. It checks initial inference against the baseline
and estimates the selected 1,600-update budget. This action does not start the
full fit.

## Preparation checks

The real-cache CUDA preflight passed exact initial-parameter matching, all
eight initial inference comparisons, recorded baseline crop order and original
file preservation. It performed four throwaway updates and retained no trained
checkpoint. [Saved timing and pairing checks](centerline_duration_preflight.json).

**22 focused tests passed in 28.99 seconds.** They cover hidden loss/learning-rate/initialization/order changes,
missing crop records, protected-file mutations and explicit common-budget
numerical drift. The existing synthetic original/loss-control runner verifies
that its save/evaluation/checkpoint flow still works. Read-only field audits
recognize the approved duration config and its hash without weakening the
immutable input-cache checks.
