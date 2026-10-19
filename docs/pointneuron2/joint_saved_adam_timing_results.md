# GPU timing passes readiness; automatic reconstruction remains unsolved

**Agreed follow-up:** the [capped retained fit is now prepared](joint_saved_adam_fit_ready.md)
with saved proposal Adam, fresh GAT Adam and mixed-state checkpoint/resume.
Local checks pass; no GPU fit has been launched as part of preparation.

20 October 2026. The user-launched saved-proposal-Adam timing action completes
eight throwaway GPU updates and passes its declared short stability/reload gate.
**All four automatic reconstruction gates still fail.** The timing result clears
an execution prerequisite for a learning experiment; it is not a successful
reconstruction result or evidence of improvement over PN 1.0.

## Verified execution and resource evidence

- Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU, torch 2.12.1+cu130; deterministic
  policy and TF32 disabled as declared. Same four TRAINING crops; no broader data.
- Historical GPU proposal replay is exact on all crops: zero saved coordinate
  and confidence-logit differences. The initial trained weights are unchanged.
- Eight updates complete. Proposal Adam counters advance 4,000 to 4,008; GAT
  Adam starts fresh and advances zero to eight. Loss/mask/cutoff policies stay
  fixed; both known edge classes remain available on every crop at the end.
- Crop 0/0 retains all six trusted matches after updates four and eight, and at
  every intermediate probe. Dense trusted counts are
  **404 -> 414 -> 395 -> 365 -> 415 -> 416 -> 409 -> 403 -> 414**.
  Short stability does not mean that all intermediate geometry is unchanged.
- Recorded disk reload reproduces weights, optimizer state and predictions
  exactly; prediction delta is zero. The temporary checkpoint is deleted.
  No retained trained checkpoint or full fit exists.
- PyTorch peak allocation is **299,551,232 bytes = 0.279 GiB**; peak reservation
  is **406,847,488 bytes = 0.379 GiB**, on a reported approximately 8 GiB device.
  These measure the allocator during the eight updates/probes, not total GPU
  process usage or a guarantee for later candidate densities. Reload memory is
  not included in that recorded peak scope.
- Second-cycle update times are 0.671 / 0.787 / 0.759 / 0.874 seconds. The
  reported **51.5-77.3 minutes for 4,000 updates** is a planning range for update
  time, including incoming diagnostics/report writes. It excludes all-four-crop
  post-update probes, snapshot evaluation/checkpoint IO and future candidate
  density changes. The full timing action takes 32.15 seconds after setup;
  the user-visible command takes approximately 46 seconds including startup.

## Automatic output quality: mixed, with sparse regressions

Continuous reference-arclength precision/recall within two voxels:

| Training crop | Precision: stage 0 -> 8 | Recall: stage 0 -> 8 | Accepted assigned + unmatched: stage 0 -> 8 |
|---|---:|---:|---:|
| 0/0 | 59.89% -> 41.89% | 94.87% -> 79.52% | 6+11 -> 6+10 |
| 0/1 | 49.94% -> 46.63% | 100% -> 97.82% | 12+32 -> 12+32 |
| 92/4 | 63.69% -> 67.70% | 78.91% -> 88.15% | 16+17 -> 16+16 |
| 86/3 | 97.60% -> 97.93% | 94.12% -> 94.76% | 411+4 -> 409+3 |

Unmatched means outside the unique target-node assignment, not necessarily image
background: some such points can be legitimate curve subdivisions. The declared
learnability gate still requires the agreed unique target representation, and
geometry/topology measurements remain essential alongside those counts.

Keeping all six crop-0 matched proposals within two voxels does not ensure good
curve reconstruction: their displacement, unmatched-node paths and selected
connections can still reduce continuous geometry quality. Its five required
direct pairs remain candidates and all five pass the edge cutoff, but the forest
selects only two; three lose out during cycle resolution among extra-node paths.
The eleven-to-ten reduction in unmatched acceptances does not fix that output.
This is why proposal placement/label stability cannot substitute for the automatic
graph and arclength gates.

| Training crop | Required direct edges retained: 0 -> 8 | Output components: 0 -> 8 | Reference crop components |
|---|---:|---:|---:|
| 0/0 | 2/5 -> 2/5 | 1 -> 1 | 1 |
| 0/1 | 5/10 -> 4/10 | 4 -> 3 | 2 |
| 92/4 | 7/15 -> 8/15 | 3 -> 1 | 1 |
| 86/3 | 380/404 -> 375/404 | 27 -> 23 | 14 |

The dense crop improves arclength precision/recall and component count, while
rejecting two more assigned nodes, retaining fewer required direct edges and
increasing retained-junction degree errors from seven to eight. Nine assigned
dense queries remain rejected at stage eight. Conversely, a missing direct pair
can sometimes be replaced by a path through unmatched nodes; it is not by itself
proof of a broken anatomical connection. The report keeps direct edge identity,
continuous geometry and crop-forest topology separate. Eight steps, only two
updates per crop, do not establish a learning trend or an architectural verdict.

## Independent saved-output review

The new [read-only CPU reviewer](../../scripts/pointneuron2/review_joint_saved_adam_timing.py)
performs **no neural forwards, optimizer updates or GPU execution**. It verifies
source/cache/report hashes, the eight attempted/completed optimizer cursors,
active-crop label classes, the stability gate and update-time calculation.
For both snapshots on all four crops, it independently:

1. Recomputes automatic 0.5 selection from saved logits; verifies that retained
   candidates/scores are inherited from the full 512-query graph without rebuilding.
2. Recomputes decoded topology, missing-edge decomposition and continuous geometry
   from saved predictions and frozen references. Every checked metric matches
   the report exactly; saved evaluation-only labels match.
3. Parses all eight SWCs, validates trees, and verifies node/component/edge
   identities plus positions/radii against the decoded packet within 1e-6.

This independently verifies saved postprocessing/metrics and serialization. The
exact learned-weight and optimizer reload result is the instrumented GPU run's
recorded evidence; the reviewer cannot independently replay discarded weights.
No temporary trained checkpoint remains. Existing sources/cache hashes match.

## Recommended next decision

Prepare the bounded **4,000-new-update joint learnability fit** using saved
proposal Adam state and fresh GAT Adam state, not the old fresh-state trainer.
Start from the pinned original source models/optimizer history, not the discarded
timing weights. Keep the four training crops, LR 0.0001, existing balanced losses,
trust mask, candidate policy, normalization and final thresholds. Retain the
declared automatic node/graph/SWC and at least 99% arclength precision/recall gate
on every crop. Snapshot evaluation remains 0/500/1,000/2,000/4,000, with model,
mixed-history optimizer, RNG and stage/crop cursors persisted before metrics.

This is a proposal for the next substantive implementation, **not a prepared
or launched full fit**. The trainer needs explicit mixed-counter checkpoint and
resume validation: proposal counter 4,000 + new-stage step; GAT counter new-stage
step. The new stage must cap at 4,000, without silently extending the source fit.
Provide a visible GPU command for the user after that implementation is reviewed
and verified. The timing launcher has no train/resume action.

The decision question is whether the unchanged learned node/edge formulation can
fit clean **automatic** reconstruction on these cases. Passing would establish
bounded training-set learnability, not generalization. A completed fit that fails
the declared terminal gates should stop this control and return to the interface,
representation and objective discussion; do not lower cutoffs, use reference
selection, extend training or add another isolated tweak to call it a pass.
Execution stops on the existing missing-edge-class guard, nonfinite values,
source changes or OOM. The eight-step quality regressions are retained as evidence
and the longer bounded fit must earn its result; timing readiness does not promise
that the result will be favorable.

## Artifacts

- GPU report: `tmp/pointneuron2/local_runs/joint_saved_adam_timing_20261019/timing/report.json`.
- Read-only review: `tmp/pointneuron2/local_runs/joint_saved_adam_timing_review_20261020/summary.json`.
- [Visible measurements and verification provenance](joint_saved_adam_timing_results_seed0/summary.json).

No runtime model, loss, launcher, cutoff, cache or source checkpoint is modified
by this review. Only a saved-output reviewer, readable results and provenance are
added. Commits/pushes record that review; no additional GPU experiment is run.
