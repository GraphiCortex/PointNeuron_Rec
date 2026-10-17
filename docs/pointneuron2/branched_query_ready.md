# Four-crop proposal learnability fit prepared

**Evaluation repair, 17 October:** the first fit failed during its 500-update
dense-crop evaluation on an empty cached probe array. The
[repair and restart report](branched_query_evaluation_repair.md) records the exact
cause, ten passing tests and the preserved partial evidence. No step-500 state
was saved; the runner now checkpoints before evaluation. The launcher defaults
to `branched_query_restart_20261017`, whose byte-identical cache and repaired CPU
check are ready. GPU timing and the fresh retry remain user-launched.

**Original GPU timing reviewed, 17 October:** the user completed the eight throwaway
updates on the RTX 3070 Ti Laptop GPU. The source/cache/snapshot hashes match;
initial weights match the CPU seed-zero check; saved prediction reload delta is
0.0. No trained timing checkpoint remains. Estimated fit time is **76–114
minutes**, plus checkpoint IO/system variability, from one timed four-crop
cycle. PyTorch peak allocated/reserved memory is **93.1/118 MiB**; this excludes
GPU context and other memory outside PyTorch tracking. The eight-update gates
fail as expected this early; no learnability conclusion follows. That first fit
subsequently failed as described above; repeat timing for the repaired runner. See the
[saved timing review](branched_query_ready_seed0/timing_review.json).

17 October 2026. The user agreed to the bounded experiment proposed in the
[upstream audit](upstream_connectivity_results.md). The separate trainer,
configuration, exact frozen-cloud cache, CPU checks and visible launch commands
are prepared. CPU preparation used no real-neuron optimizer updates or GPU
workloads. The user subsequently completed timing and the interrupted first
fit described above. The repaired retry has not started. CPU unit tests use
discarded synthetic updates to verify continuation.

This experiment trains PointNeXt and the Transformer together. Phase 1 stays
frozen; no field inference or manual edits are needed. GATv2 is not trained by
this runner. Its purpose is to establish usable branched proposals before a
matched GATv2-versus-pairwise comparison, not to claim generalization or full
reconstruction quality.

## Fixed inputs and settings

| Training neuron / crop | Exact distinct cloud points | Targets | Visible junctions |
| --- | ---: | ---: | ---: |
| 0 / 0 | 848 | 6 | 0 |
| 0 / 1 | 2,048 | 12 | 0 |
| 92 / 4 | 1,219 | 16 | 2 |
| 86 / 3 | 2,048 | 418 | 77 |

The clouds, voxel indices, targets and reference adjacency match the preceding
audit exactly. No inputs are regenerated or padded; no targets are truncated.
Only the existing training caches and three training SWCs are involved. The
seven development neurons and reserved test data are not used.

Fresh seed-zero initialization; native PointNeXt-S prototype and existing
512-query Transformer; explicit batch-one current-crop normalization in both
train and eval; float32; no augmentation/dropout. The previous fitted checkpoint
and confidence-only replay head are not initialization sources. The diagnostic
normalization policy remains explicitly identified, not a promoted default.

AdamW LR .0001, weight decay .0001, gradient clip 10, unchanged query loss:
positive class weight 8, objectness/center/radius weights 2/3/.2, radius floor 1,
center beta 4 and matching scale 6. Cycle in the table order for **4,000 updates
maximum**, 1,000 crop appearances each. Evaluate all four crops and save full
state at 0/500/1,000/2,000/4,000. No scheduler or automatic early success/extension.

The visible recipe is [branched_query_fit.json](../../configs/pointneuron2/branched_query_fit.json).
Each snapshot records geometry, assigned/accepted/unassigned counts, radii,
continuous-curve point support, trusted junctions, known/unknown candidate-edge
counts and attainable edge recall before and after the .5 node cutoff. Exact
predictions, decoder features, assignments and candidate labels remain in tmp.

## Advancement and stopping

Every crop must meet all of these agreed training-learnability criteria:

- At least 95% of targets have trusted one-to-one matches within two voxels.
- At least 95% of all target edges have trusted endpoints in the reference-free
  kNN-16 union radius-eight candidates; positive and negative supervision exist.
- All visible junctions have trusted matches.
- At least 95% of assigned positives are accepted at the fixed .5 node cutoff.
- Accepted unassigned queries <= floor(5% of target count): 0/0/0/20 by crop.
- Accepted-proposal curve support within six voxels is at least 95%.

Report geometry and selection gates separately. The before-cutoff edge criterion
does not guarantee the same recall after node filtering; the latter is reported
explicitly. Extras are negatives under the current set objective, not automatically
biological false positives. None of these gates validates learned connectivity,
physical calibration, generalization, whole-neuron coverage or PN1 superiority.

Stop on nonfinite values/gradients, target overflow or violated source/data
contracts. At 4,000, stop and interpret the result. Do not extend the run, lower
cutoffs, replace difficult crops, change losses or begin GATv2 training
automatically. Geometry failure sends us back to the target/assignment/loss or
encoder question; selection failure remains explicit. A combined pass enables
preparation of a common frozen-upstream GATv2-versus-pairwise comparison.

## Verified locally

The real-crop CPU check completes in **22.99 seconds excluding imports** with
zero optimizer updates. Four fresh-model evaluations fail the learning gates,
as expected before training. All saved prediction fields reload with maximum
absolute delta **0.0**; inference leaves model state unchanged. Source/cache
hashes and saved source snapshots are verified. PowerShell launcher syntax
and its real prepare/check actions pass.

**Six targeted tests pass.** They cover full-model forward/backward, disk checkpoint snapshot
independence, AdamW state, Python/NumPy/Torch RNG, crop cursor, source/recipe/backend
identity rejection, completed-budget resume refusal, distinct geometry/selection
gates, small/dense crop thresholds, and existing train/eval normalization controls.
See the [readiness evidence](branched_query_ready_seed0/summary.json).

The first two-thread continuation test reproduced loss/random draws but was
not bitwise identical after an update (largest observed parameter difference
about 1.9e-6). Restricting that checkpoint test to one CPU thread reproduces
parameters and optimizer state exactly. This isolates a repeatable checkpoint
state check from parallel arithmetic variation; it does not prove bitwise
two-thread or GPU continuation. The real recipe stays at two CPU helper threads.
No GPU determinism setting or model policy was changed to make the test pass.
Earlier failing test XML is preserved under tmp alongside the verified results.

## Visible user-launched GPU commands

Run the following **timing action first** from PowerShell:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_branched_query.ps1 -Action timing
```

This performs eight throwaway updates on the local NVIDIA GeForce RTX 3070 Ti
Laptop GPU: one warm-up cycle and one timed cycle over all four crops, followed
by evaluation and checkpoint-state/reload checks. It retains no trained
checkpoint. It prints a duration range and peak allocated/reserved memory.
TF32 and cuDNN benchmarking are off. The completed first timing action estimates
76–114 minutes. The estimate includes measured update and evaluation costs,
with a 1.5x planning upper range, and excludes checkpoint IO/system variation.

Inspect its report before the full fit. The separate full-fit command is:

```powershell
& .\scripts\pointneuron2\run_branched_query.ps1 -Action train
```

The trainer requires a completed timing report with matching cache, source,
recipe and backend identities. Training starts again from fresh seed-zero
weights; the timing updates do not contribute to its 4,000-update budget.
The console prints progress every 50 updates plus four-crop snapshot results.
Runtime execution can pass while the combined learnability gate fails.

Default output root:
`C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\branched_query_restart_20261017`

Cache is in `cache/`; CPU evidence in `cpu_check/`; GPU estimate in
`timing/report.json`; fit snapshots/report in `model/`. Timestamped action
console logs sit alongside the experiment directory, inside ignored local_runs.
Existing action outputs are refused rather than overwritten. For an independent
repeat, supply a fresh `-ExperimentDirectory` under `tmp/pointneuron2`, then run
prepare/check/timing/train separately. No automatic launch chain is supplied.

## Resume only within the original cap

Checkpoints have a separate `pointneuron2-branched-query-fit-v1` container. They
save full model, AdamW, Python/NumPy/Torch CPU and initialized CUDA RNG states,
crop cursor, recipe, source/cache identity and backend. No scheduler is used.
The nested `inference_checkpoint` retains the explicit crop-normalization schema;
legacy loaders must not be given the outer training container.

For an interrupted run, explicitly select a saved checkpoint, for example:

```powershell
& .\scripts\pointneuron2\run_branched_query.ps1 -Action resume `
  -Checkpoint .\tmp\pointneuron2\local_runs\branched_query_restart_20261017\model\step_1000.pt
```

This writes a separate `resume_step_1000/` folder, restores optimizer/RNG/cursor,
and finishes at 4,000 total updates. Resume also accepts a checkpoint from an
earlier resume folder. A completed 4,000-update checkpoint cannot resume into
additional fitting. Source/backend changes are refused; bitwise GPU continuation
is not established by the CPU tests.

Source: [trainer](../../scripts/pointneuron2/branched_query_experiment.py),
[PowerShell launcher](../../scripts/pointneuron2/run_branched_query.ps1),
[checkpoint state](../../src/pointneuron/models/query_fit_state.py),
[new tests](../../tests/test_branched_query_fit.py).
