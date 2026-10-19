# Proposed partial-supervision contract and one short stability comparison

20 October 2026. For discussion after the [joint-fit supervision stop](joint_saved_adam_fit_stop_results.md).
**Historical design proposal, now accepted for implementation.** The isolated
controller and CPU checks are complete; see the [ready GPU command](joint_partial_stability_ready.md).
The historical trainer is unchanged and the GPU comparison has not been launched.

The destination remains an automatic per-crop reconstruction model that passes
the original four-crop learnability gate, followed by broader evaluation and a
matched PN 1.0 comparison. A short control below only decides whether the current
update policy warrants a longer trial. It cannot establish reconstruction success.

## Proposed training contract

Keep the frozen Phase 1 clouds, PointNeXt, Transformer, GATv2, ordinary Hungarian
matching, two-voxel trustworthy-placement limit, candidate graph, normalization,
0.5 inference cutoffs and reference-free forest decoder. No new head or topology
loss is introduced. Missing local annotation remains unknown, not background.

| Known edge labels in the current crop | Proposal loss | Edge loss | GAT Adam state/weights |
|---|---|---|---|
| Positives and negatives | Unchanged | Mean of positive and negative class means | Update normally |
| Positives only | Unchanged | Mean positive loss | Update normally |
| Negatives only | Unchanged | Mean negative loss | Update normally |
| No known edges, but supervised proposal targets | Unchanged | Omit the edge term entirely | Do not update |

Unknown edges never enter either class mean. This uses the existing
`masked_edge_loss` normalization for available classes: a single present class
has full edge-term weight, rather than half weight or an invented absent-class
term. The edge coefficient remains one. This is an explicit extension of the
joint training policy; it does not reinterpret old results as a successful fit.

When there are no known edges, backpropagate **proposal loss only**. GAT parameter
gradients must remain `None` for that update. Do not attach a zero-valued graph
term: Adam momentum and weight decay could otherwise move graph weights despite
zero graph gradient. Clear gradients with `set_to_none=True` before every update.
Validate finite gradients for every active parameter; missing proposal gradients
remain errors. On a supervised-edge update, missing GAT gradients remain errors.

Record proposal-update and GAT-update counters separately, together with the
edge-label counts and reason for every omitted term. A graph counter no longer
has to equal the number of proposal updates. A new controller/checkpoint schema
must express this policy; the old guarded trainer and historical checkpoints
remain untouched. Starting from an old checkpoint requires a documented,
validated import, not a fresh optimizer or a broad provenance bypass.

An unannotated crop is a different case: it does not become a negative example
and cannot be treated as the supervised proposal-only row in this table. The
four cases below all have local supervision. Invalid targets, nonfinite values,
data/source changes and unexpected absent gradients remain execution failures.

## One bounded comparison, if this contract is accepted

Question: **with the same valid partial-supervision policy, does a smaller
optimizer step reduce geometry fluctuations while preserving automatic output?**
This tests step-size sensitivity, not whether GAT caused the drift or whether a
new architecture is needed. One-tenth LR is a deliberately separated comparison,
not an inferred optimum or the start of an automatic LR search.

| Setting | Arm A | Arm B |
|---|---|---|
| Initial weights, Adam moments/counters, RNG | Exact saved joint update 1,000 | Identical copy |
| Partial-supervision policy | Table above | Identical |
| Learning rate, proposal and GAT | 0.0001 | 0.00001 |
| New proposal updates | At most 64 | At most 64 |
| Crop order | 0/0, 0/1, 92/4, 86/3 cyclically | Identical |
| Other optimizer settings / global clipping | Existing settings / ten | Identical |

Restore proposal Adam at 5,000 and GAT Adam at 1,000. Change only LR after exact
state restoration for B; do not reset either optimizer's history. Each completed
arm sees each crop 16 times. These are diagnostic forks ending at historical
stage 1,064, not continuations certified under the original contract. Neither
fork may automatically extend or overwrite the original run.

Pinned source: `tmp/pointneuron2/local_runs/joint_saved_adam_fit_20261020/resume_step_0008/step_1000.pt`,
SHA256 `c096a0906cc6d69f1c0b525ff105a87b207008383faa09e03f06500710b9e414`.
Its hash was rechecked while preparing this proposal.

For A, incoming predictions, loss components, supervision statistics and Adam
cursors for steps 1,001–1,013 must reproduce the preserved trajectory before
interpreting the comparison. At 1,014, the incoming crop-0/1 packet must reproduce
the saved failure within declared float tolerance (1e-5), with identical matching,
candidates and label counts. This is a reproduction check, not a quality pass.
The new policy then permits the known-positive-only loss. A mismatch stops the
comparison; it is not evidence for an LR effect.

Probe all four crops without gradients after each complete four-update cycle.
Save mean/95th-percentile/max matched distance, fraction of matched centers within
two voxels, node acceptance/extras, known-edge counts and optimizer activity.
Probe continuous errors even when edge labels disappear. Probes must not advance
RNG or normalization state; implementation must verify that rather than assume
it. Keep per-update incoming diagnostics too, so the cycle endpoint cannot hide
large within-cycle excursions.

Export and assess the actual automatic graph/SWC at fork start and end, and
before any early stop. There is no reference selection of model inputs or nodes.
Report assignment-conditioned adjacency separately from continuous morphology.

## Declared diagnostic criteria and stopping rules

These are proposed engineering criteria chosen before running the forks; they
are not calibrated biological thresholds or replacements for the original full
reconstruction gate.

1. **Execution:** complete the cap with finite states and correct active/inactive
   optimizer behavior. Missing edge classes alone no longer stop a crop.
2. **Persistent geometry degradation:** stop an arm early if any crop has mean
   matched distance greater than four voxels at three consecutive full-cycle
   probes. This is twice the unchanged edge-trust radius. Record the stop; do
   not roll back, lower LR or continue automatically. A single crossing of the
   two-voxel trust boundary is recorded, not an execution failure.
3. **Stability qualification:** in every one of the final eight cycle probes,
   each crop must retain at least 90% of matched centers within two voxels.
   Additionally, each crop's final eight incoming training occurrences must
   meet that same fraction. This requires 6/6, 11/12, 15/16 and 377/418 centers,
   respectively. Report continuous distances alongside this binary diagnostic.
4. **Preserve automatic output:** at the fork endpoint, versus the common
   update-1,000 start, no crop may gain unmatched accepted nodes or nonreference
   decoded edges, lose accepted assigned nodes or correct decoded reference
   edges, lose retained reference junctions, gain junction-incidence errors or
   cross-reference-component merges, increase absolute component-count error,
   or acquire an invalid SWC. Arclength P@2 and R@2 may each decline by at
   most 0.005 (half a percentage point) as a declared diagnostic tolerance.

The 90% stability qualification and 0.005 tolerance do **not** permit promoting
an imperfect reconstruction. The original full gate still requires all assigned
targets accepted, zero extras, exact reference adjacency/junction behavior,
correct crop forests, valid SWC and P@2/R@2 at least 99% on every crop. It remains
reported separately and unchanged.

If B alone meets both stability qualification and output preservation, a smaller
LR is supported for this local state. If A alone meets them, retain the original
LR for consideration. If both meet them, prefer A unless B improves at least one
crop's reconstruction metrics without worsening another; smaller steps alone
are not a quality improvement. If neither meets them, stop optimizer-policy
tuning and return to geometry parameterization/objective coupling. No additional
LR grid, longer fit, new head, wider trust mask or broad-data run follows
automatically. If an arm stops early, do not claim an equal-exposure quality
comparison; report the failure under the common cap.

Passing this control only justifies discussing one capped learnability fit with
the original automatic criteria. It does not prove generalization or PN 1.0
superiority, and does not by itself authorize that fit. One checkpoint and seed
give local optimization evidence only.

## Implementation and user-visible execution boundary

If accepted, implement an isolated controller plus meaningful CPU tests for the
four label cases, inactive GAT weight/moment/counter preservation, restoration,
probe noninterference and identical pre-stop replay. Preserve the old trainer and
reports. Keep the new script/config/report visible and generated fork states in
ignored `tmp/pointneuron2` directories. Prepare a command that the user launches
on the RTX 3070 Ti Laptop GPU; no automatic launch. Previous measured throughput
suggests minutes rather than an hour for 128 total updates, but all-crop probes
and export overhead are unmeasured and must be stated with the command.
