# Capped joint learnability fit with saved proposal Adam state

20 October 2026. The user agreed to prepare this fit after the [passed GPU timing
and mixed automatic-output review](joint_saved_adam_timing_results.md).
**Historical preparation status:** prepared and locally checked before GPU launch.
The subsequent fit stopped on Windows JSON replacement after 148 updates; see
the [I/O repair and checkpoint-eight recovery](joint_saved_adam_fit_io_recovery.md).

The decision question is whether the existing PointNeXt/Transformer/GATv2
formulation can learn clean **automatic per-crop reconstruction** on four fixed
training cases. The eight-step timing result established short execution
readiness, not reconstruction quality. Passing this longer fit would establish
bounded training-set learnability, not generalization, whole-neuron reconstruction
or improvement over PN 1.0.

## Fixed experiment

| Training neuron/crop | Input points | Targets | Reference edges | Crop components |
|---|---:|---:|---:|---:|
| 0/0 | 848 | 6 | 5 | 1 |
| 0/1 | 2,048 | 12 | 10 | 2 |
| 92/4 | 1,219 | 16 | 15 | 1 |
| 86/3 | 2,048 | 418 | 404 | 14 |

Phase 1 remains frozen. Start from the original pinned balanced proposal and
conditional GAT checkpoints, not discarded timing weights. Restore the proposal
Adam moments and counters at 4,000; start GAT Adam fresh at zero. This is a new
joint stage with **maximum 4,000 updates**, cyclic crop order, batch one, seed
zero, no augmentation or scheduler, LR 0.0001, weight decay 0.0001 and one global
gradient clip at 10. Proposal/GAT counters finish at 8,000/4,000, while the new
stage counter finishes at 4,000. No source checkpoint is edited or extended.

The visible [recipe](../../configs/pointneuron2/joint_saved_adam_fit.json) retains
the tested joint loss, Hungarian assignments, two-voxel edge-supervision trust
mask and unknown-label handling. Unmatched confidence labels mean absence from
the unique target-node assignment, not verified image background. All 512 queries
reach GAT; candidates are symmetrized 16NN union radius eight. Final nodes use
the original inclusive 0.5 confidence cutoff **after** graph scoring; inherit
scored edges without rebuilding candidates. Edge cutoff remains 0.5, followed
by the existing reference-free forest decoder. Explicit current-crop normalization
and zero dropout remain experimental policies. No stage, head or topology loss
is added or replaced; no development/test neurons are evaluated.

## Success and stopping rules

The unchanged automatic gate must pass on **every crop at terminal update 4,000**:

- All assigned target queries accepted, zero unmatched acceptances.
- Exact target adjacency with no missing/nonreference decoded edges and correct
  crop-forest component count; all reference junctions retained within two voxels
  and with correct incidence.
- Continuous-reference arclength precision **and** recall at two voxels at least
  99%, plus valid SWC serialization and coordinate/radius round trip.

Report assignment/placement, node acceptance, candidate coverage, scoring,
forest selection, continuous geometry and SWC validity separately. A missing
direct target pair is not automatically a disconnected anatomical path; some
unmatched nodes can be curve subdivisions. Those qualifications do not relax
this declared target-representation learnability gate. Loss reduction, a
component pass, valid SWCs alone, or a favorable intermediate crop does not pass
the terminal experiment.

Stop at the cap. A completed fit that fails the terminal gate returns to the
representation/interface/objective discussion; do not extend the fit, lower
cutoffs or select nodes with reference labels to call it successful. Execution
stops on the existing missing-edge-class guard, nonfinite values, source/cache
changes, OOM or replay mismatch. No automatic retries or budget reductions.

## Checkpoints, diagnostics and reproducibility

The new isolated runner reuses the tested timing update function. It writes full
incoming active-crop losses, labels, assignments and positions to
`active_update.json` before loss/backward/optimizer phases. Completed per-update
counts, losses, gradient norms and cursors append to `updates.jsonl`; the main
report stays compact. A training failure preserves the last active diagnostics,
incoming prediction packet where available, traceback and preceding checkpoints.
Per-update reporting avoids rewriting a growing full training history each step.

Save complete model/optimizer/RNG/config/source/backend/next-crop state **before**
scheduled evaluation at 0/500/1,000/2,000/4,000. An additional verification
checkpoint is saved at eight before checking the first eight GPU updates against
the passed timing output. Float predictions must match at relative/absolute
1e-5; assignments, candidate membership, labels, node/edge cutoff decisions and
decoded forest decisions must match exactly. A replay failure stops before
update nine. This checks the retained trainer's start; it is not a new quality
gate or additional optimizer budget.

Checkpoint validation covers parameter names/order, optimizer options and moment
shape/dtype/device/finiteness, both counter histories, configuration, data/source
provenance, backend and crop cursor. A verified initial prefix is required for
checkpoints beyond eight. Resume starts from the saved new-stage step and ends
at 4,000; it does not restart the budget. At an evaluation milestone, resume
evaluates the restored state again before advancing, preserving metrics that
may have failed after state was saved. Resume from the verification checkpoint
reruns the prefix comparison before proceeding. Check-only, timing-only and
terminal checkpoints cannot resume this fit. Each resume uses a fresh numbered
output folder, preserving earlier outputs.

Terminal reload verifies model/optimizer equality and exact four-crop predictions.
This trainer has a new fit schema; the old fresh-Adam launcher is unchanged and
must not be used for this initialization.

## Completed local checks and visible GPU command

Fifteen targeted tests pass. Synthetic disk-resume tests reproduce the next crop,
Torch/NumPy/Python RNG draws, parameters and optimizer state exactly; tests also
reject changed caps/backend/data/cursors/names/moments and terminal/check-only
resume, verify checkpoint survival after metric failure, and reject prefix
cutoff changes even within float tolerance. The real-model CPU check performs
**zero optimizer updates**, validates four baseline SWCs and reproduces weights,
mixed optimizer state and predictions exactly after disk reload. No trained
check-only checkpoint remains. These checks do not claim a GPU full fit or GPU
resume has executed. See [readiness measurements and provenance](joint_saved_adam_fit_ready_seed0/validation.json).

Launch the retained fit yourself with the new command:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_joint_saved_adam_fit.ps1 -Action train
```

Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU. Passed timing estimates **51-77
minutes of update time**, plus snapshot evaluation/checkpoint overhead and
possible candidate-density changes. Default output:
`tmp/pointneuron2/local_runs/joint_saved_adam_fit_20261020`.
Generated checkpoints, predictions, SWCs, logs and reports remain ignored.
Existing train/check action folders cannot be overwritten. Progress prints every
50 updates; all four automatic outputs are evaluated at the declared snapshots.
The final `model/report.json` distinguishes completed execution from its
`automatic_gate` result.

If interrupted after an intermediate fit checkpoint exists, explicit resume is
available, for example:

```powershell
& .\scripts\pointneuron2\run_joint_saved_adam_fit.ps1 -Action resume -Checkpoint ".\tmp\pointneuron2\local_runs\joint_saved_adam_fit_20261020\model\step_0500.pt"
```

Use resume to recover a saved state after interruption/system/metric failure;
a deterministic supervision/policy stop needs interpretation, not an automatic
retry. The example checkpoint does not exist until the user-launched fit creates
it. The original preparation created only `check/`. After the I/O repair,
current-source CPU checking uses `check_io_repair_v1/`; the recovery report above
gives the existing checkpoint and current command.

Visible implementation:
[joint_saved_adam_fit.py](../../scripts/pointneuron2/joint_saved_adam_fit.py) and
[PowerShell launcher](../../scripts/pointneuron2/run_joint_saved_adam_fit.ps1).
