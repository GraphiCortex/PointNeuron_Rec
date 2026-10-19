# Partial supervision: two short GPU controls ready for user launch

20 October 2026. Implements the [discussed policy and bounded comparison](joint_partial_supervision_proposal.md).
**43 targeted tests and real-model zero-update CPU checking pass. GPU comparison
has not been launched.** The historical guarded joint trainer is unchanged.

Run this new command, rather than the old train/resume command:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_joint_partial_stability.ps1 -Action compare
```

The matching CPU prerequisite is already complete. This command runs both arms
sequentially on the NVIDIA GeForce RTX 3070 Ti Laptop GPU, each importing the
same pinned update-1,000 weights, Adam moments/counters and CUDA RNG state.

| Fixed TRAINING crops | 0/0, 0/1, 92/4, 86/3 |
|---|---|
| Phase 1 / architecture | Frozen clouds; existing PointNeXt/Transformer/GATv2 |
| Initial Adam clocks | Proposal 5,000; GAT 1,000 |
| Arm A / Arm B LR | 0.0001 / 0.00001 |
| Maximum new updates | 64 per arm, 128 total; 16 exposures per crop per completed arm |
| Other optimizer settings | Existing AdamW, weight decay 0.0001, global clip ten |
| Queries / candidate graph | All 512; existing symmetrized 16NN union radius eight |
| Node/edge decisions | Existing inclusive 0.5 cutoffs and forest decoder |
| Reference placement trust | Two voxels, unchanged |

Planning expectation is **several minutes plus setup, probing, export and disk
overhead**, based on the preceding fit. The new controller itself has not been
GPU-timed; this is not a measured duration guarantee. The hard update budgets
do not change with wall time. Both arms are diagnostic forks, not resumptions
certified under the old 4,000-update protocol. No long-fit or resume action exists.

## What changes

The isolated [partial loss](../../src/pointneuron/models/partial_joint_loss.py)
uses the existing mean of whichever known edge classes are present. Positive-only
and negative-only batches are valid; unknowns remain masked. With no known edges,
only proposal loss is backpropagated and GAT gradients remain `None`. The runner
checks that GAT weights, moments and counters actually stay unchanged, including
when existing Adam moments are nonzero. Proposal and GAT clocks are recorded
separately. Both arms use this policy; only their LR differs.

The [new controller](../../scripts/pointneuron2/joint_partial_stability.py) imports
the pinned parent under an explicit new diagnostic schema. It validates hashes,
model specifications, parameter order, all Adam slots/options and original
initial cursors before changing LR. It does not relax the old trainer's resume
validation or edit any prior checkpoint. Diagnostic checkpoints are marked
nonresumable and saved before fallible start/endpoint graph evaluation.

## Required output and stopping behavior

Arm A must reproduce the historical start predictions and completed steps
1,001–1,013, then the saved incoming failure at 1,014, before interpreting this
comparison. Float tolerance is 1e-5; matching/candidates and recorded label
statistics must agree. A mismatch stops execution. At the verified former stop,
the new available-class loss permits the positive-only update.

All-crop probes every four updates record continuous mean/95th-percentile/max
matched distance, trust fractions, acceptance/extras and available supervision.
The runner checks that probes preserve parameters, buffers, modes and RNG.
Incoming diagnostics retain within-cycle fluctuations too.

An arm stops for persistent geometry degradation if any crop's mean matched
distance exceeds four voxels at three consecutive cycle probes. Nonfinite data
or states, unexpected gradients, altered sources or replay mismatch are execution
failures. There are no automatic retries or extensions.

At start and endpoint, export the actual automatic graph and SWC. Endpoints must
retain the declared node/edge/junction/component measures and valid SWCs, with
at most 0.005 decline in P@2 or R@2 relative to the common start. Stability also
requires at least 90% trusted matched centers per crop over the last eight cycle
probes **and** last eight incoming occurrences. These are diagnostic criteria;
the full automatic gate remains unchanged and separately reported.

The combined report recommends discussion of current LR, reduced LR, or neither
under the predeclared rules. It never starts a longer fit. If neither qualifies,
stop policy tuning and revisit geometry/objective coupling. If one qualifies,
that is local evidence supporting a future learnability trial, not proof of
generalization, full reconstruction success or superiority to PN 1.0.

## Saved outputs and completed checks

Default ignored root: `tmp/pointneuron2/local_runs/joint_partial_stability_20261020`.
The CPU check is in `check/`. The user-launched run will create
`comparison/current_lr/` and `comparison/reduced_lr/`, each with reports, incoming
diagnostics, cycle probes, nonresumable checkpoints, NPZ predictions and SWCs.
The combined result is `comparison/report.json`; existing action folders are
never overwritten. The PowerShell launcher retains full console logs.

Tests cover all four label cases, missing annotations, nonzero-momentum GAT
inactivity, active-gradient errors, independent checkpoint tensors, exact RNG/
next-crop restoration, probe mutation detection, replay decisions and cap/
reconstruction criteria. A synthetic complete 64-update controller test reproduces
a preserved prefix and former guard packet, traverses a proposal-only update,
and ends with 64 proposal updates but only 63 GAT updates. This verifies controller
behavior; it is not a real-image learnability result.

The real-model CPU check validates both imported initializations, eight baseline
automatic SWCs, exact disk-reload predictions and probe noninterference, with
zero optimizer updates and no CUDA initialization. CPU predictions differ
slightly from historical GPU predictions; their cross-backend equality is not
claimed. The GPU command performs its own same-backend historical replay checks.
See [validation and provenance](joint_partial_stability_ready_seed0/validation.json).

Visible [recipe](../../configs/pointneuron2/joint_partial_stability.json),
[launcher](../../scripts/pointneuron2/run_joint_partial_stability.ps1) and
[tests](../../tests/test_joint_partial_stability.py) are saved in the source tree.
