# Consistent node objective: implemented research candidate

3 November 2026. The user authorized addressing suspected causes of the failed
broader fits. This candidate changes the **training objective and assignment**;
it does not replace any intended architecture stage. It is implemented and
CPU-verified, not promoted as a successful reconstruction method. GPU timing
and learned comparison have not run.

## What prompted the change

The completed [paired point-budget study](node_budget_fit_results.md) failed its
training and development screening at both budgets after 20 epochs. This is a
bounded result, not proof of convergence or architectural impossibility.

The baseline averages positive CE and negative CE separately, then averages
those groups. With M targets and Q=512 queries, a negative query's weight relative
to a positive is M/(Q-M). Across the 460 supervised training boxes this ranges
from **0.00392 to 4.50538**. The objective therefore changes its class weighting
with target count. For uninformative shared scores, its optimum is p=0.5 at
every target count; ordinary per-query CE instead has optimum p=M/Q. This is a
mathematical property, not proof that it caused the observed excess predictions.

The baseline matcher uses `3*distance/6 - p_used`, whereas its differentiated
loss uses separately balanced log probabilities and voxel-unit Huber geometry.
Approximate assignment is common, including in DETR; this mismatch is not an
implementation bug in Hungarian matching. The candidate explicitly chooses an
assignment that minimizes its own objective to remove this additional heuristic.

Every update in the last five epochs triggered clipping at 10 in both baseline
arms. Median pre-clip norms were about 154 (2048) and 191 (8192). Persistent
clipping is a diagnostic, not evidence that clipping is wrong. Reducing a loss
scale does not imply proportionately smaller Adam updates or better learning.

## Exact candidate

Let s=6 voxels, b=4 voxels, wc=2 and wg=3, retaining existing constants. Define
`h(q,t) = smooth_L1(||x_q-y_t||/s, beta=b/s)`.

```text
L = [wc * sum(all Q query classification losses)
     + wg * sum(matched query spatial penalties h)] / Q

Hungarian cost(q,t) = wc * [log p_unused(q) - log p_used(q)] + wg * h(q,t)
```

The all-unused classification loss is constant with respect to assignment.
Replacing one unused label by a used label gives exactly the classification
increment above. Thus the selected assignment minimizes this same objective
over all one-to-one injections. Tests compare against exhaustive enumeration.

All queries share one denominator. Dense boxes have more supervised geometry
terms instead of each target receiving a smaller share of a fixed per-box mean.
The Huber transition stays at four voxels; changing units divides its magnitude
by six. Loss values are **not numerically comparable with baseline losses**.
This changes class weighting, geometry weighting and assignment together as one
coherent objective; it is not an ablation isolating each mechanism.

Implementation: [consistent_node_objective.py](../src/pn2/models/consistent_node_objective.py).
The frozen [baseline objective](../src/pn2/models/node_objective.py) is untouched.
No count penalty, count head, output cap, proximity relabeling or cutoff tuning
is added. Ordinary CE does not guarantee calibrated scores or correct counts.

## Kept fixed and unresolved

- Original image-derived 2048-point clouds and normalization; no filter/cache regeneration.
- GN8 PointNeXt, geometric-query Transformer, 512 queries and unrestricted offsets.
- Existing target packets, including structural identities and whole-chain sampling.
- Cutoff 0.5 and two-/six-voxel reference/continuous-curve metrics.
- Unknown boxes abstain. Unused-slot labels do not certify biological background.
- Radius and GATv2 remain outside this node-only control.

A constructed counterexample confirms that two identical local curve segments
can inherit different interior nodes/counts from whole-chain sampling outside
the box. This establishes a possible target-identifiability problem, not its
contribution to these cohort errors. The audit records the two node sets. No
target correction or new graph representation is silently adopted. A target
change requires a separate explained comparison so it cannot obscure this one.

Other unresolved hypotheses include acquisition differences, encoder support,
annotation completeness and insufficient training duration. This candidate does
not claim to fix them.

## Completed CPU evidence

[Saved evidence](evidence/consistent_node_objective_cpu_20261103.json) binds the
candidate sources/configuration and completed checkpoint. The CPU audit:

- Replayed assignment on all 460 positive training boxes using frozen predictions.
- Found 33.55% mean changed assignments, averaging each box's changed fraction.
- Compared both full-model gradients on four deterministic training-only target-count strata.
- Verified exact train/eval predictions and unchanged model state.
- Performed **zero optimizer updates** and **no CUDA initialization**.

| Targets in diagnostic box | Baseline gradient norm | Candidate gradient norm |
|---|---:|---:|
| 8 | 515.59 | 14.40 |
| 32 | 246.05 | 13.30 |
| 96 | 112.45 | 11.83 |
| 180 | 67.51 | 6.72 |

These four boxes come from two training acquisitions and are not a
representative accuracy benchmark. Three candidate gradients still exceed the
existing clip limit. Norm changes verify changed optimization scale, not better
conditioning, useful Adam updates or learned improvement. Frozen-prediction
replay changes assignments only; it cannot improve saved output geometry/scores.
The replay reconstructs log probabilities from saved softmax scores, clipping
only at float32 probability limits. It does not recover bitwise original logits;
the four full-model gradient checks use actual current forward outputs instead.

**50 CPU tests passed**, covering exhaustive assignments, sparse/dense class
weighting, unknown abstention, coincident targets, finite/zero-distance gradients,
overflow, target permutation, the phase counterexample, exact next Adam/RNG
serialization, and existing node/model/state contracts.

## Manual GPU timing only

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_consistent_node_objective.ps1 -Action timing
```

This uses the RTX 3070 Ti Laptop GPU in two fresh worker processes, each with a
600-second hard limit. Each arm independently loads identical terminal 2048
weights **and saved Adam/RNG state at update 9200**. Eight throwaway updates
cycle through the four named training strata; two further executions check one
exact next-update reload. Current passed CPU evidence is required first. No
trained checkpoint is retained and there is no full-fit/resume action.

This is a new throwaway objective comparison, not a continuation of the old
study's cap. Timing requires identical initial model/Adam/RNG states across arms
and exact next-update reload per arm; a failure stops without retry. Runtime is
not yet measured. Small-output/dense-box execution success is not an accuracy
gate. Artifacts go to ignored `tmp/consistent_objective_timing_<timestamp>`.

## What would decide adoption

After successful timing, discuss a separately capped matched continuation,
starting both objectives from the same completed checkpoint and optimizer
state. The unchanged-objective arm controls for additional training duration.
Neither the old failed gates nor its 9200-update cap is rewritten. A full-run
command was intentionally absent until its protocol and measured cost were agreed.
The user subsequently ran GPU timing successfully and agreed to the separate
[five-epoch paired continuation](consistent_node_fit_ready.md). That new driver
does not modify this timing-only launcher or the original study.

The candidate must improve actual training precision **and recall** against the
matched continuation, with development F1 improving without lower recall, and
continuous-curve support disclosed. Sparse and dense boxes must be reported
separately. Fewer accepted predictions, lower candidate loss or smaller gradients
alone do not qualify it. Preserve the original cutoff and score all development
boxes; do not mask unknown outputs or use reference counts at inference.

If it suppresses sparse positives, remains inaccurate, or only changes loss
scale, do not promote it or automatically extend it. Saved Adam moments were
learned under the old objective; the timing is a warm-start feasibility check,
not evidence that those moments suit the new objective. Resetting Adam would
be a different, explicitly controlled choice. No result here establishes
convergence, final graph quality or SWC improvement.

Reference implementation consulted: [official DETR criterion](https://github.com/facebookresearch/detr/blob/main/models/detr.py)
and [matcher](https://github.com/facebookresearch/detr/blob/main/models/matcher.py).
DETR uses weighted classification and an approximate probability matching cost;
this node objective is a PN2 research candidate, not a DETR reproduction.
