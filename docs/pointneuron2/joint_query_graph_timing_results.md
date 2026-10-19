# Joint timing stops before the second crop cycle

**Follow-up:** the agreed [four-update optimizer-state comparison](proposal_optimizer_state_results.md)
now passes the crop-0 criterion with saved AdamW state and fails with fresh state.
It tests proposal-only optimization; the current joint fit remains stopped.

19 October 2026. **Do not launch the 4,000-update joint fit with this recipe.**
The user-launched GPU timing action completed four throwaway updates and stopped
before update five on training crop 0/0. The declared requirement for both known
positive and known negative graph labels was no longer satisfied. No retained
joint training checkpoint, timing estimate or completed readiness result exists.

## What the GPU evidence establishes

The warm-start proposals reproduce the historical GPU predictions exactly on
all four crops: zero coordinate/logit differences and zero node-decision changes.
This resolves the earlier historical-replay check on the actual GPU backend.
It does not explain the smaller CPU/GPU numerical differences.

The four completed updates take 1.094 / 0.612 / 0.622 / 0.607 seconds. The
exception is raised in the supervision guard, before the fifth backward pass or
optimizer step; it is not an OOM exception. The eight-update timing and reload
check remain incomplete, so these four timings are not a training-duration or
memory-readiness estimate.

Initially crop 0/0 has six trusted assigned nodes, five known positive edges,
eight known negative edges and 13,568 unknown candidates. The exact failing
GPU counts and positions were **not saved**. We cannot claim that all six GPU
nodes became untrusted, or identify the missing GPU class from the exception
alone. The timing runner retains metrics only at successful completion, which
left an avoidable gap in this failure's evidence. Its `optimizer_updates: 0`
field must not be read as zero executed throwaway updates: the console records
four. The original report and logs are preserved without editing.

## One bounded CPU mechanism check and its matched control

The visible [diagnostic script](../../scripts/pointneuron2/audit_joint_timing_stop.py)
loads the same pinned weights, clouds, targets and recipe. Each arm makes only
four throwaway CPU updates, in the same cyclic order, then discards the model.
There is no GPU execution, development/test evaluation or retained trained
checkpoint. Source/cache hashes verify before and after. The proposal-only arm
uses the same fresh AdamW settings and clipping, omitting graph loss; graph
weights stay unchanged. CPU results are not exact GPU-trajectory replays.
The proposal-only arm records the graph-label guard but does not require graph
classes for its updates, since it trains no graph objective.

Crop 0/0 is probed after each completed update, including updates on other crops:

| Updates completed | Joint trusted nodes / 6 | Proposal-only trusted nodes / 6 | Joint maximum matched error, voxels | Proposal-only maximum matched error, voxels |
|---:|---:|---:|---:|---:|
| 0 | 6 | 6 | 1.162 | 1.162 |
| 1 | 3 | 0 | 3.035 | 3.251 |
| 2 | 1 | 0 | 5.221 | 5.579 |
| 3 | 0 | 0 | 6.348 | 6.720 |
| 4 | 0 | 0 | 6.892 | 7.237 |

In the joint CPU arm, both edge classes are already empty after update two;
the normal cyclic runner encounters that crop again at update five and would
stop there. Proposal centers have moved outside the fixed two-voxel trust mask,
so their incident labels become **unknown**, not background or known negatives.
The proposal-only arm loses all trusted crop-0 matches after one update even
though its node/geometry/radius objective remains fully supervised. Joint
preclip gradient norms are 472.81 / 458.28 / 791.04 / 1,040.64; proposal-only
norms are 467.17 / 638.84 / 619.95 / 1,024.94. Every update is clipped at 10.
Clipping parameter-gradient norm does not establish preservation of proposal
positions after an AdamW update.

This identifies a mechanism for disappearing graph supervision and establishes
that graph loss is **not necessary** for the early CPU geometry drift. It does
not establish that graph loss is harmless, that fresh Adam state is the sole
cause, or that four unstable updates predict every later learning trajectory.
The present shared-LR, fresh-optimizer restart has failed its declared stop rule
and cannot be treated as ready for a full fit.

## Decision before another substantial run

Retain the stop, two-voxel trust mask, 0.5 output cutoffs, automatic gates and
unknown-label handling. Do not turn this into a successful timing result by
removing the guard, extending training, or using reference-selected nodes.
The zero-update readiness checks proved wiring, gradients and reload behavior;
they did not prove optimizer-step stability.

The smallest next comparison is **fresh versus restored proposal optimizer
state**, with the proposal-only objective, identical source weights, LR,
clipping, crop order and four-update budget. The completed balanced proposal
checkpoint contains its AdamW state; a state-preserving control is feasible.
This directly tests the restart hypothesis before adjusting architecture or
starting a learning-rate search. It has not been implemented or launched here.

Before that control, declare preservation of all six trusted crop-0 nodes after
one complete four-crop cycle, and report changes on every crop. If restored state
preserves geometry while fresh state does not, optimizer restart is implicated
in this bounded task; a revised joint initialization still needs its own short
stability check. If both drift, that hypothesis is insufficient: stop and examine
cross-crop gradients/update scale before a full joint fit. Even a stable short
control establishes neither joint learnability nor reconstruction improvement.

A future runner revision should persist the attempted/completed step, crop,
edge-label counts, trusted-match distances, loss terms and gradient norms on
each timing update and at the failure. That is instrumentation, not permission
to change the current training policy. No runtime model, loss, recipe or launcher
has been modified by this review.

## Saved evidence

- Original GPU report:
  `tmp/pointneuron2/local_runs/joint_query_graph_ready_20261019/timing/report.json`.
- Joint CPU mechanism check:
  `tmp/pointneuron2/local_runs/joint_timing_stop_cpu_20261019/report.json`.
- Proposal-only CPU control:
  `tmp/pointneuron2/local_runs/joint_timing_stop_proposal_only_cpu_20261019/report.json`.
- [Visible measurements and provenance](joint_query_graph_timing_results_seed0/summary.json).

No new broad-data experiment, node head, threshold policy, production promotion,
training extension or retained training is authorized by this failed timing run.
