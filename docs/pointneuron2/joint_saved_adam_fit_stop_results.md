# Joint fit: progress, geometry fluctuations and a supervision-policy stop

20 October 2026. The user-launched recovery completed new-stage update **1,013**
and stopped during loss construction at **1,014**, before backward or optimizer
step. Latest full checkpoint: update 1,000. **Do not resume unchanged.** This is
an early stop under the declared protocol, not a completed 4,000-update negative
result and not proof that the architecture is incapable of learning.

Only this read-only saved-output audit and report are added. No model, trainer,
loss, threshold, configuration or training state is modified. No new fit runs.

## Established evidence

The Windows repair worked through this run. Restored update-eight predictions
again matched timing exactly. All **140 repeated updates 9–148** have identical
saved losses, supervision statistics, gradient norms and Adam cursors to the
original interrupted trajectory. The new failure is unrelated to JSON writing.

At update 1,014, training crop 0/1 has all 12 assigned proposals accepted but only
two assigned centers within the two-voxel edge-label trust limit:

| Quantity | Saved/recomputed value |
|---|---:|
| Matched centers within two voxels | 2/12 |
| Mean/max matched distance | 2.444 / 2.853 voxels |
| Known positive / known negative candidate edges | 1 / 0 |
| Unknown candidate edges | 5,266 |
| Required reference edges with trusted endpoints | 1/10 |
| Missing candidates among those trusted reference edges | 0 |
| Accepted assigned / unmatched proposals | 12 / 19 |
| Proposal loss | 2.46220 |
| Available-class edge loss | approximately 5.52e-11 |

The one positive edge and absence of known negatives trigger the explicit
`joint_loss` guard. Arrays and losses are finite. Unknown edges were not relabeled.
The input point cloud did not change. This is a loss of **trusted edge
supervision under moving geometry**, not evidence of an empty neuron or missing
candidate generation for the trusted pair.

All 12 target-to-query identities are unchanged from the update-1,000 snapshot.
Their mean displacement is [-0.837, -0.432, -1.170] voxels, with 0.436-voxel RMS
residual around that mean. Assignment churn is therefore not required for this
particular failure. This comparison spans 13 optimizer updates; it cannot assign
the displacement to one intervening crop or to the graph gradient.

The pre-update history already fluctuates: crop 0/1 mean matched distance is
0.915 at update 1,006, 0.583 at 1,010 and 2.444 at the failed 1,014. Crop 0/0's
trusted count falls from six at 1,005 to three at 1,009. All 1,005 completed
updates in the recovered invocation used global clipping at ten; median
preclip norms by crop are roughly 282–334. Clipping frequency alone does not
identify an instability cause or prove exploding gradients. Optimizer dynamics,
crop interference and objective interaction remain explanations to distinguish.

## Automatic reconstruction at the last evaluated checkpoint

Arclength P/R are measured within two voxels of the continuous reference. These
are automatic outputs from the same four **training crops**, without reference
selection of graph nodes.

| Crop | Accepted assigned + extras | Required edges retained | Nonreference edges | P@2 | R@2 | Full gate |
|---|---:|---:|---:|---:|---:|---|
| 0/0 | 6/6 + 1 | 5/5 | 1 | 94.15% | 100% | Fail |
| 0/1 | 12/12 + 18 | 6/10 | 20 | 51.07% | 92.71% | Fail |
| 92/4 | 16/16 + 5 | 13/15 | 7 | 84.19% | 100% | Fail |
| 86/3 | 418/418 + 0 | 399/404 | 5 | 99.71% | 99.16% | Fail |

This preserves useful progress. The dense crop meets the geometry thresholds
and exact node-membership criterion at this snapshot. Nevertheless, its 14
components and valid SWC do not make its topology correct: five same-tree
nonadjacent edges replace five required edges and one junction has an incidence
error. Every required edge survives node selection, exists as a candidate and
scores above 0.5. The score ordering makes the forest decoder reject five true
edges while preventing cycles. This is evidence about scorer/decoder agreement,
not evidence that the decoder should permit cycles or that all missed edges
correspond to missing anatomical branches.

For crop 0/1, all four missing required pairs are also lost during forest
selection despite surviving candidates and edge cutoff. Extras and their edge
context remain a separate problem from the trust-mask stop. Low average
classification/edge loss is insufficient evidence of a clean automatic graph.

These within-trajectory improvements do not establish a causal benefit of joint
training over a matched proposal-only continuation, development generalization,
whole-neuron quality or superiority to PN 1.0.

## Interpretation and recommended direction for discussion

The immediate stopping condition is partly our experimental policy: requiring
**both** edge classes in **every** crop/update. The existing lower-level
`masked_edge_loss` already computes finite loss from the classes that exist.
Preserving unknown labels does not mathematically require that guard. I made
the joint guard more restrictive than needed for general partial-supervision
training. Removing it would be a substantive training-policy change, not a
reconstruction improvement or a way to declare this control passed.

Retain the architecture provisionally. Before another fit, agree the contract
for known-positive-only, known-negative-only and entirely unknown edge batches:
which losses and optimizer states advance, which terms are skipped, and what
continuous geometry evidence triggers a stability stop. Keep unknown labels
masked and preserve the original automatic reconstruction criteria. The
all-unknown case also needs explicit handling of Adam momentum and weight decay;
zero gradient does not necessarily mean unchanged graph parameters.

Do not replace PointNeXt/GATv2, add a node head, lower inference cutoffs, enlarge
the trust radius or start broader training on the strength of this result.
Neither confidence failure nor edge-ranking errors have been resolved. A further
controlled experiment should test a declared training-policy hypothesis against
this preserved trajectory, with fixed data/exposure budget and reconstruction
metrics. Its purpose must be learning a stable automatic per-crop model, not
merely making the guard stop throwing. No such change or run is implemented by
this report.

## Verification

The [saved-output reviewer](../../scripts/pointneuron2/review_joint_saved_adam_fit_stop.py)
independently verifies all 12 prediction/SWC pairs at stages eight, 500 and 1,000:
file hashes, actual node selection, inherited candidate/score mapping, decoded
graph, geometry metrics, junction gate and SWC serialization. It also recomputes
the failed packet's label statistics and loss from recorded assignments, checks
the journal/cursors and validates the checkpoint hash. No model forward,
optimizer update or CUDA initialization is performed.

Measurements and provenance: [summary.json](joint_saved_adam_fit_stop_results_seed0/summary.json).
Full raw review stays under ignored
`tmp/pointneuron2/local_runs/joint_saved_adam_fit_stop_review_20261020/`.
