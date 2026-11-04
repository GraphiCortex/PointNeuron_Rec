# Matched forests and frozen-checkpoint transfer — diagnostic protocol

The user parked the Transformer hypothesis and requested no-learning forests,
development inference, two-voxel input support and wrong-join diagnostics before
any sixteen-box decision. This adds measurements only. The original fit/variant
failures, sources, gates and proposals are immutable. No training or criterion
change is authorized. Timed work is launched locally by the user.

## Existing input evidence, no rerun

Reuse `tmp/curve_formulation_audit_20261103/report.json` and `boxes.jsonl`.
These already contain 2/6/8-voxel curve-to-input support, 2048/8192 budgets and
fixed apparent-width/contrast strata. Copy the two-voxel summaries and the exact
training frog alias-22 box-5 row with their hashes; do not re-filter or re-audit.
This is proximity to input coordinates, not proof of informative image features
or a hard ceiling for offset predictions/connecting edges.

## Fixed no-learning comparison

On the four fitted boxes and on all 151 reference-present development boxes
from seven neurons (415 total grid boxes; 264 unknown boxes excluded), compare:

1. Frozen update-2000 dense predictions, score >=.25 and original radius-1
   consolidation. No minimum-support filter, cap, threshold sweep or changes.
2. All cached 2048 input points, unit scores and source-ID tie breaking, same
   radius-1 consolidation. No offsets and no learned scores.
3. Same input points ranked by cached Sato values, same radius-1 consolidation,
   first 512 retained points. No offsets or score cutoff.

Counts differ and must be disclosed. These distinguish the full raw-input
baseline from a fixed sparse ranking baseline; they do not isolate each effect
of learned confidence versus offsets or equalize graph sizes. The 8192-point
budget is input-support evidence only, not another trained-model input or sweep.

All use the unchanged image-only scaffold: k=6, straight-edge cached-Sato cost
`L*(1+4*(1-mean_normalized_Sato))`, maximum .5-voxel samples, minimum spanning
forest, no new gap bridges, image-guide roots, type 0/radius 1 placeholders.
Candidate k=4/6/10 coverage remains diagnostic and selects nothing. Graphs/SWCs
are saved before reference scoring. Fit dense graphs are reused, not rebuilt.
Score all policies with the same clipped segment scorer at 2/6 voxels. Pool
directed arclength within neuron and macro average P/R/F1; expose every box and
neuron and retained counts. The learned coordinates were fitted only on the
four training boxes. Development has already been inspected repeatedly, so
this is transfer to untrained neurons, not an untouched final test.

## Conservative wrong-join diagnostic, fixed before results

Use the full supplied reference forest for scoring only. For each forest edge:

- Project both endpoints to continuous reference segments within two voxels.
  Alternatives within .25 distance that imply reference-path separation >.5
  voxel or different components make a projection ambiguous. Remove artificial
  subdivision-endpoint alternatives when an incident direction reduces distance.
  Coincident disconnected branches remain ambiguous; never weld fragments.
- Off-reference endpoints, ambiguous projection, distinct traced components,
  degenerate predicted edges, or reference paths leaving the scored box are
  distinct unscorable/unknown categories. Report their counts and lengths.
  Different components are not automatically known false joins.
- For unambiguous same-component endpoints with an in-box traced path, let D
  be continuous reference arclength between projections and L the predicted
  edge length. Flag a **reference-path shortcut** if `D > 2*L + 4 voxels`.
  Report all D/L and excess values, flag counts/lengths and scorable denominator.

This is a reference-conditioned shortcut diagnostic, not exhaustive wrong-join
ground truth. It can miss short cross-branch joins and cannot certify biological
correctness where annotations are incomplete. Geometry disagreement is reported
separately. No threshold is selected from results; no flagged edge is removed.

## Execution, stopping and decision

Two explicit CPU-only stages use fresh timestamped PN2/tmp folders. Fit controls
have a 30-minute cap. Development has a two-hour overall cap (including a child
CPU-only Torch forward stage); inference timing is unmeasured, so no completion
estimate is asserted. It is a conservative cap, not permission for extensions.
Use the existing CPU NumPy/SciPy runtime as parent and retained Torch runtime
for CPU forward only; hide CUDA before subprocess/model imports. No optimizer,
backward, RNG trajectory restore, fine tuning or model-state mutation.

Before development, load only model weights from the hash-bound completed
update-2000 checkpoint. Recompute four saved training predictions on CPU and
require coordinate max error <=.001 voxel, score error <=.0001 and identical
retained source-index sets versus saved GPU predictions. Log differences; stop
if this numerical parity check fails. Exact GPU bit replay is not asserted on
CPU. Model state digest must remain unchanged after inference. Save every
prediction packet and a final complete receipt. No incomplete packet set can
be scored as completed development evidence.

Stop on source/input/checkpoint/parity mismatch, nonfinite output, wall limit or
scoring failure. Preserve partial outputs; no retry, fallback to GPU, extension,
automatic sixteen-box preparation or checkpoint selection. User reviews both
fit and transfer measurements before discussing a new rung. Any future
downstream criterion is explicitly post-hoc relative to the original failed
gate and prospective only for a new experiment, with numeric floors fixed
before that experiment and evaluation on untrained neurons. No new numeric
acceptance floors are adopted by this diagnostic protocol.
