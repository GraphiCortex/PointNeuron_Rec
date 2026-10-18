# Frozen four-crop confidence: execution passes, declared learnability gate fails

18 October 2026. The user-launched CPU stage completes all 4,000 updates in
42.03 seconds of reported runner time (about 55 seconds including launch).
Only the 258 confidence parameters train. **No crop passes the strict fixed-label
gate at any saved snapshot.** Keep the balanced full-model fit as the incomplete
reference; do not promote this head or extend the run automatically.

## Terminal result against the source head

All counts use the declared .5 cutoff and the same frozen labels. “Extras” means
unmatched queries under the saved assignment, not necessarily image background.

| Training crop | Accepted assigned, before → after | Extras, before → after | Final strict gate |
|---|---:|---:|---|
| 0/0 | 6/6 → 6/6 | 11 → 4 | Fail |
| 0/1 | 12/12 → 12/12 | 32 → 28 | Fail |
| 92/4 | 16/16 → 16/16 | 17 → 8 | Fail |
| 86/3 | 411/418 → 409/418 | 4 → 4 | Fail |
| Total | 445/452 → 443/452 | 64 → 44 | Fail |

The new stage reduces total label errors from 71 to 53, but misses nine positives
and accepts 44 extras at its terminal checkpoint. This is partial fixed-training
selection improvement, with a dense-crop regression. It does not satisfy the
[declared criterion](four_crop_frozen_confidence_ready.md) of all 452 positives
and zero extras simultaneously. There is no undeclared intermediate winner.

## Loss and selection trajectory

Each pair below is accepted assigned plus extras. Geometry and labels stay fixed.

| New stage update | Declared optimization loss | 0/0 | 0/1 | 92/4 | 86/3 |
|---:|---:|---:|---:|---:|---:|
| 0 | .203534 | 6+11 | 12+32 | 16+17 | 411+4 |
| 100 | .188852 | 6+10 | 12+32 | 16+16 | 411+4 |
| 500 | .164266 | 6+9 | 12+31 | 16+12 | 411+4 |
| 1,000 | .152511 | 6+6 | 12+29 | 16+10 | 411+4 |
| 2,000 | .139659 | 6+5 | 12+29 | 16+8 | 408+4 |
| 4,000 | .122960 | 6+4 | 12+28 | 16+8 | 409+4 |

Loss falls 39.59%. Every crop's balanced CE decreases, but that does not guarantee
every query improves or that hard decisions improve on every crop. The strongest
crop-0/1 negative increases from .994686 to .999345 while its negative count falls.
At the end, the minimum positive / maximum negative probabilities are:

| Crop | Lowest positive | Highest negative |
|---|---:|---:|
| 0/0 | .973717 | .983862 |
| 0/1 | .898938 | .999345 |
| 92/4 | .931017 | .984495 |
| 86/3 | .008964 | .899961 |

There is score overlap within every crop. Even choosing separate scalar cutoffs
per crop could not cleanly separate their current scores. No cutoff is changed.

Read-only CPU backward checks at saved snapshots give objective-gradient L2 norm
.183030 → .010074. Neither this nor falling loss certifies convergence, an
optimum, a regularization cause or that more updates would reach the gate.
No review optimizer updates occur. The gradient is for the declared data loss;
AdamW's decoupled decay is not included in that gradient.

## What this control establishes

The [capacity oracle](four_crop_confidence_capacity_results.md) already verifies
one shared affine separator on these exact hidden features. The present stage
fails to learn a zero-error separator under its declared loss, AdamW settings,
initialization and finite budget, even with fixed features and fixed assignments.
Assignment movement is therefore **not necessary for failure under this recipe**.
It may still affect joint training; this experiment does not isolate its causal
contribution to the earlier balanced or hard-negative runs.

Do not infer an unavoidable encoder/head capacity limit, a broken PointNeXt
architecture, or that staged learning can never work. Conditioning, finite
optimization and objective/regularization tradeoffs remain possible explanations.
The stage uses a fresh optimizer and all-four-crop updates rather than cyclic
single-crop joint updates, so it is not a matched-compute continuation or a
single-factor causal comparison with joint training.

The selection improvement does not establish reconstruction improvement. Dense
accepted curve support within six voxels changes 99.684% → 99.502%, and required
R8 edge availability after confidence changes 371/404 → 366/404
(91.832% → 90.594%). These are frozen-geometry support/availability diagnostics,
not trained GATv2 predictions or reconstructed SWC accuracy. Trusted dense targets
remain 404/418 and trusted junctions 76/77. Even perfect assignment membership
would accept the 14 poorly placed assigned proposals.

## Integrity and decision

The review replays all six saved checkpoints on CPU and gets **zero difference**
from every saved score tensor. The terminal checkpoint has optimizer cursor
4,000, exactly 258 parameters and the recorded schema/recipe/provenance. Frozen
tensor fingerprints and raw feature bytes match preparation and the oracle;
50 current source hashes and 85 protected source hashes match. The runner also
reports unchanged cached features/labels/geometry and zero terminal reload delta.
No review training, GPU work, development/test evaluation or upstream model
forward occurs. Six small CPU backward checks evaluate gradients only.

Stop this confidence-fit series at its declared cap. Keep the completed balanced
model, rejected hard-negative comparison, oracle and this failed control as
separate evidence. Do not add an automatic longer run, LR sweep, larger head,
new threshold or GATv2 fit.

The next design discussion should settle the proposal-to-graph responsibility:
which proposals should survive, how duplicates should be handled, and whether
confidence should mean assignment membership or usable geometric support.
Assignment membership alone does not certify accurate, nonredundant,
branch-covering nodes. Assignment-based set prediction can still be a valid
formulation when its geometry and selection objectives work together; this
distinction is not proof of a defective loss or matcher. The failed control does
not identify the cause of its errors, and another tuning run would be an
insufficient answer to the broader design question. This is a recommendation
for discussion, not an implemented loss, matcher or architecture change.

Evidence: [snapshot and integrity summary](four_crop_frozen_confidence_results_seed0/summary.json).
Raw reports/checkpoints/scores remain under ignored
`tmp/pointneuron2/local_runs/four_crop_frozen_confidence_20261018/model`.
