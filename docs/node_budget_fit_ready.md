# Matched point-budget study: bounded fits ready for manual launch

2 November 2026. Both integrated GPU timing arms passed. The evaluation-only
profile cache, final fit/resume launcher and release checks are complete.
**No long fit has been launched.** This is a concrete controlled node-proposal
study within the intended PN2 architecture, not final graph/SWC qualification.

## Integrated timing result

The user completed `tmp/node_budget_integration_timing_20261102_123748` in
**13m56s**. Both budgets made eight throwaway updates from the actual shared
training schedule, then inferred/scored all 415 development boxes across seven
neurons, including all 264 reference-empty boxes. Evaluation preserved the
model, Adam and RNG values; reload recovered the exact next schedule row.

| Measurement | 2048 points | 8192 points |
| --- | ---: | ---: |
| Warm update including loading/transfer | 0.6543 s | 1.5901 s |
| Full development pass with core metrics | 219.9 s | 572.9 s |
| Projected 9200 updates | 1.672 h | 4.064 h |
| Four development passes | 0.244 h | 0.637 h |
| Terminal training pass, extrapolated | 0.289 h | 0.753 h |
| Combined projection | **2.206 h** | **5.453 h** |
| Peak CUDA allocated | 239.0 MiB | 629.0 MiB |

The paired projection is **7.659 hours**. Keep **10–14 hours as a provisional
combined-run allowance**, not an upper bound. Eight updates and one development
pass do not measure sustained laptop performance, every target-count case or
terminal training scoring. The extra profile scoring and checkpoint IO add
work; source filtering and data preparation are already complete.

Saved report/journal identities, first-eight schedule rows, initial parameter
equality, prediction-file hashes and aggregate metric arithmetic were checked
on CPU. No quality claim follows from the eight-update development scores.
There is no need to repeat either timing command.

## Thin/dim diagnostics are fixed before the study

CPU preparation took **582.4 seconds** for 45,948 arclength windows and 27,573
reference targets. Sampling/model inputs/losses never read these profiles.
The source-image profile convention remains radii 1/2/4/8, axial half-length 2,
background shell 8–12 and radial bins 0.5 native voxels. Curve profiles are
measured every four voxels; coverage uses fine 0.5-voxel quadrature so the
four-voxel profile locations do not alias the two-voxel support tolerance.

The saved training-only, arclength-weighted 25th-percentile cutoffs are:

- Apparent width: **3.0002057613 native voxels**.
- Inner peak contrast / whole-source intensity range: **0.0836601307**.

These are evaluation characterizations, not new sampling thresholds or model
loss parameters. They are not calibrated anatomical caliber. Profiles use the
available source-image context, which can be clipped near volume boundaries.
Unresolved half-height/contrast profiles and ambiguous/zero reference tangents
remain unresolved. Structural junctions with multiple incident chains are not
assigned an invented single tangent. Node profiles follow their actual reference
chain identities; no spatial welding or SWC-radius caliber assumption is used.

Training nodes: 1150 apparent thin/dim, 3664 thin/bright, 3149 thick/dim,
9785 thick/bright and 2895 unresolved. Development nodes: 684, 2713, 764,
2280 and 489 respectively. These are tiled reference identities, not unique
whole-neuron biological node counts. Per-neuron denominators remain visible.

Report per-stratum reference-node recall and curve support, with the target
grid's own support baseline. Precision remains global: unmatched proposals
have no known reference branch class. Stratum/structural matches can be
ambiguous near coincident identities. Unresolved nodes stay in the primary
global metrics. None of these labels selects training crops, input points,
queries or accepted proposals. Application inference does not require SWCs
or this evaluation cache.

All 2379 profile packets were checked; training cutoffs were independently
recomputed. CPU replay of all **830 saved prediction packets** preserved core
node and curve totals while adding these diagnostics. This replay is metric
verification, not a new learned evaluation or GPU run.

## Exact study and stopping rules

Two independent fresh models, same GN8 native PointNeXt/geometric-query
Transformer, 512 proposal slots and 1,688,518 parameters. Retain the shared
targets and deterministic exposure schedule, XYZ/intensity features and nominal
2048/8192 input budgets; the single 7650-voxel clipped box stays unpadded.

Each arm receives **20 epochs / 9200 total updates**, batch one, 460 supervised
training boxes per epoch across 33 neurons. The 1504 reference-empty training
boxes are unknown and receive no negative-only loss. All 415 development boxes
are inferred at epochs 0/5/10/20; all 1964 training boxes are scored terminally.
The seven development neurons are repeatedly exposed development data. No
reserved source file or development gradient is used.

Same fresh initialization, AdamW LR 0.0001, zero weight decay, balanced node
classification/center objective, clipping 10, deterministic FP32/math attention
and cutoff 0.5. No augmentation, scheduler, radius loss, topology loss or GATv2
training is added. The first eight updates must match the passed integrated
timing's model/Adam/RNG prefix exactly.

Use the fixed epoch-20 state for the primary verdict. Intermediate development
scores do not select a checkpoint or trigger an extension. The prospective
90% training precision/recall and 80% development precision/recall/minimum
neuron F1 60% remain engineering screening diagnostics. A scientific gate
failure is reported separately from successful execution and does not suppress
the other prespecified arm. Nonfinite values, mismatched data/code, overflow,
timeout or invalid state stop execution without automatic retry.

## Launch

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_budget_fit.ps1 -Action train
```

This launches 2048, then a fresh 8192 arm, sequentially on the RTX 3070 Ti Laptop
GPU. Output defaults to `tmp/node_budget_fit_train_<timestamp>` with separate
`points2048`/`points8192` folders. The frozen recipe is
`configs/node_budget_fit.json`. Worker stop limits are **6 hours at 2048** and
**12 hours at 8192**; they are separate from the fixed 9200-update scientific
cap and the provisional 10–14-hour planning allowance. Keep the laptop plugged
in and awake; keep the frozen source/configuration/caches unchanged during the
run. Logs show evaluation, update progress and completed checkpoint paths.

This is the first broader paired training study. It may succeed or fail the
screening criteria; neither outcome alone establishes universal reconstruction
quality. Per-neuron results, unknown outputs and geometry/discretization
diagnostics are needed to interpret it. Connectivity, consolidation, radii and
final SWC topology still require separate evidence.

## Checkpoints and manual recovery

Save full model/Adam/RNG/data-cursor state **each completed epoch (460 updates)**
and at initialization, using new immutable files and completion hashes.
Checkpoints are saved before scheduled evaluations. No optimizer-state reset
or weights-only warm start is supported. Resume checks the budget, data,
profile, schedule, code bindings, finite Adam moments and matching update
clocks. An interruption can discard work since the last completed checkpoint;
replaying from that state preserves the specified trajectory/cap.

Recovery uses a fresh run folder. Replace the example checkpoint with an actual
completed path printed by the run:

```powershell
& .\scripts\run_node_budget_fit.ps1 -Action resume -Budget 8192 `
  -Checkpoint ".\tmp\YOUR_RUN\points8192\checkpoint_step004600.pt"
```

The total cap remains 9200, including the restored updates. Recovery from a
complete step-9200 checkpoint performs terminal scoring with **zero additional
optimizer updates**. A resume command selects one arm; it does not automatically
start another arm. For a fresh single arm, use `-Action train -Budget 8192`.

The new fit controller stops its owned worker process tree on timeout/Ctrl+C,
including Windows virtual-environment redirectors. A real CPU child-process
test verifies descendant termination. This addresses the inherited launcher's
parent-only kill behavior without modifying the old timing/model code.

## Verification and evidence

**30 tests pass**, including data/label separation, full-grid metrics, profile
quadrature/undefined handling, exact stochastic Adam continuation, invalid
resume rejection, terminal scoring recovery without further updates, both-arm
execution despite scientific gate failures, and owned-tree termination.
PowerShell parsing and all launch gates passed in the actual retained Torch
runtime. No package was installed or changed, and no long GPU fit was run.

- [Integrated timing review](evidence/node_budget_integrated_timing_results_20261102.json)
- [Fit-ready evidence, profile audit and tests](evidence/node_budget_fit_ready_20261102.json)
- Profiles: `tmp/node_budget_profiles_20261102/manifest.json`
- Profile review: `tmp/node_budget_profile_review_20261102/report.json`
- [Data/core integration](node_budget_data_integration.md)
- [Paired-study specification](node_budget_comparison.md)
