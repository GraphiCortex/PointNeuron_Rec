# Approved pooled LayerNorm comparison

12 October 2026. The query-only normalization comparison is implemented and
checked, ready for the user's visible GPU terminal. The assistant has not
launched the 2,000-update GPU fit.

The [completed BatchNorm fit and fixed-weight audit](query_extended_fit_results.md)
localized most of the two-crop placement gap to `encoder.global_mlp[1]`.
That model fits closely with training batch statistics, but ordinary inference
changes its placements and accepted set. The approved comparison tests whether
replacing only this pooled normalization preserves a close fit during inference.

## What changes

`DGCNNEncoder` accepts an optional `pooled_normalization` setting. Its default
remains `batchnorm`; old checkpoints and existing configurations use the same
default model. The control records `pooled_normalization: layernorm` in its
checkpoint configuration, so reloads and image-only inference reconstruct the
correct model. Exactly one encoder BatchNorm is replaced. EdgeConv BatchNorm,
the query decoder, geometry adapter and losses are unchanged.

The control trains **only the query model**, fresh from initialization, for
**2,000 updates** on the same first two annotated crops from neuron 0. It reuses
the completed query BatchNorm baseline instead of retraining either baseline
arm. Query dropout stays zero. Selected inputs, targets, seed, unchanged-layer
initial weights, learning rate, optimizer, loss, score cutoff and milestone
budget stay matched. No cache preparation or augmentation is added.

Preflight verifies the saved baseline's status, pairing, cache/partition,
device, update budget, milestone hashes and unchanged source outside the three
files that implement this control. It reproduces the baseline's original encoder
initialization hash. Every encoder tensor retained by the replacement, including
the affine weight/bias, and every query-head tensor initializes exactly equally.
The removed tensors are only the pooled BatchNorm's running mean, running
variance and batch counter. Trainable parameter count and feature width do not
change. Pairing checks after fitting verify the actual selected inputs too.

LayerNorm changes feature scaling and needs a fresh fit. Success is not implied
by the fixed-weight BatchNorm diagnostic. Remaining EdgeConv BatchNorm may still
cause a smaller training/inference gap. This is a two-crop learnability check,
not validation or whole-neuron reconstruction.

## Run in your terminal

Expected duration is roughly **1–2 minutes** on the NVIDIA RTX 3070 Ti Laptop
GPU, based on the preceding query model's approximately 40 ms update time.
Startup, checkpoint writes and single-crop checks add variable overhead.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_query.ps1 `
  -Action normalization-smoke `
  -Steps 2000 `
  -EvalSteps 500,1000,2000 `
  -LogEvery 50 `
  -BaselineModels ".\tmp\pointneuron2\local_runs\query_smoke_20261012_005258_4464\models" `
  -Manifest ".\tmp\pointneuron2\local_runs\query_implementation_20261011\cache\manifest.json" `
  -Device cuda
```

Do not add a dropout override: this action uses the saved baseline's zero
dropout and changes only pooled normalization. The baseline must have the same
device, update budget and milestones; incompatible runs fail the preflight.

The runner prints a fresh directory under
`tmp/pointneuron2/local_runs/query_normalization-smoke_<timestamp>_<PID>`.
Live logs appear every 50 updates, with paired-crop and single-crop evaluation
at each milestone. No broader training or reconstruction starts automatically.

## Outputs and judgment

The printed run directory contains `console.log`, `launch.json`, and
`models/normalization_comparison.json`. The comparison retains the saved
BatchNorm trajectory, its source path/hash, exact initialization checks,
the control settings and matched-input/update checks.

`models/query` contains the control's `run.json`, checkpoints and raw predictions
at 500/1,000/2,000, plus the fixed final `proposal.pt`. Final reload must reproduce
centers, logits and radii exactly. Existing baseline artifacts are read only.

Each control milestone records ordinary inference geometry, score-filtered
support, accepted slot counts and one-to-one matched errors/counts. It separates
accepted matched slots within six voxels from extra accepted unmatched slots.
The same metrics are also recorded when crops are evaluated one at a time,
alongside maximum center/score differences from evaluating the two crops together.
Minor floating-point differences are reported rather than forced to zero.

Judge whether ordinary inference retains accurate positions and a distinct,
confident proposal for each target, with few extra accepted slots, and whether
single-crop evaluation preserves that behavior. Passing execution or matching
checks does not establish that the normalization improved the fit. We inspect
these measured results together before deciding on the broader pilot.

## Completed preparation checks

**64 targeted tests passed in 16.56 seconds.** They include the default encoder
and optional projection, unchanged initialization, LayerNorm independence from
companion crops/training mode, old/new checkpoint reloads, observation-state
preservation, an actual short matched fit, and rejection of mismatched inputs,
budgets, devices, milestones and tampered snapshot files. Existing query,
target, local-proposal and DGCNN tests also pass.

The actual PowerShell command completed a **four-update CPU** control against
the preceding saved four-update CPU baseline. Initialization checks, selected
inputs, milestone/update matching and exact final checkpoint reload all passed.
Single-crop and paired-crop centers differed by at most 0.000031 voxels at the
checks. Four updates verify execution only; the model is not expected to fit
the targets in this tiny preparation run.

```text
CPU check: tmp/pointneuron2/local_runs/query_normalization-smoke_20261012_012217_26572/models/
Real 2,000-update baseline preflight: tmp/pointneuron2/local_runs/query_layernorm_preflight_20261012/preflight.json
```

The real 2,000-update CUDA baseline also passes metadata, snapshot/hash and CPU
initialization preflight without training or neural GPU forward passes.

Visible sources:
[dgcnn.py](../../src/pointneuron/models/dgcnn.py),
[query_pipeline.py](../../src/pointneuron/models/query_pipeline.py),
[compare_query_normalization.py](../../scripts/pointneuron2/compare_query_normalization.py),
[shared fit/snapshot runner](../../scripts/pointneuron2/train_query_comparison.py),
[PowerShell runner](../../scripts/pointneuron2/run_query.ps1),
[regression tests](../../tests/test_query_normalization.py).
