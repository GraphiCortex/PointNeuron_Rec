# Current frozen proposals cannot support the branched connectivity fit

17 October 2026. The declared CPU audit completed in 26.90 seconds excluding
imports, with **zero optimizer updates**, two field forwards and four
PointNeXt/Transformer forwards. Both branched cases fail the predeclared gate.
Do not proceed to a frozen-feature GATv2-versus-pairwise fit using this checkpoint
and these branched crops: all their candidate edge labels are unknown.

This is a limit of the current two-crop checkpoint, not evidence against the
architecture. It was fitted on two junction-free crops of neuron 0, not the
branched examples. The earlier reference oracle and connectivity execution
checks remain useful; they answer different questions.

## Where information is lost

| Training neuron / crop | Distinct cloud points | Targets | Assigned within 2 voxels | Visible junctions found | Usable positive candidates / required edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 0, fitted control | 848 | 6 | 6 / 6 | 0 / 0 | 5 / 5 |
| 0 / 1, fitted control | 2,048 | 12 | 12 / 12 | 0 / 0 | 10 / 10 |
| 92 / 4, modest branched | 1,219 | 16 | **0 / 16** | **0 / 2** | **0 / 15** |
| 86 / 3, dense branched | 2,048 | 418 | **0 / 418** | **0 / 77** | **0 / 404** |

These are existing Hungarian assignments qualified by the two-voxel placement
rule. In the dense crop, two targets have *some* query within two voxels and 19
within six; none has a trusted assigned query. These measurements must not be
conflated. The modest crop has no query within even six voxels of any target.
Mean assigned error is 24.83 voxels for the modest crop and 71.72 for the dense.

Input clouds retain substantial annotated curve support:

| Neuron / crop | Foreground pool @6 | Expanded pool @6 | Selected cloud @6 | All 512 proposals @6 |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | 81.40% | 100% | 100% | 100% |
| 0 / 1 | 95.65% | 100% | 100% | 100% |
| 92 / 4 | 76.83% | 94.74% | 94.74% | **0%** |
| 86 / 3 | 100% | 100% | 98.72% | **3.86%** |

Support is length-weighted nearest-point proximity to the reference curve,
using the existing one-voxel probes. It is not edge recall or topology quality.
Six-voxel proximity alone does not certify enough input evidence for every
branch. The modest crop still lacks some pool support; the dense crop loses
1.28 percentage points through sampling. Those losses remain real, but do not
account for the far larger proposal loss here. No target is truncated: 418
dense-crop targets fit within the 512-query capacity.

The candidate graphs have 87,201, 37,421, 39,204 and 36,835 undirected pairs,
respectively. Nearby unused/collapsed proposals make the radius-union graph much
denser than the earlier random synthetic case's 4,807 pairs. This matters for
future scorer timing and memory; no cap or candidate-policy change is introduced.

The fitted controls supply 5 positive / 1 negative and 10 positive / 56 negative
edge labels. The branched cases supply **zero positives and zero negatives**;
all 39,204 / 36,835 pairs are unknown because endpoints lack trusted placements.
All-unknown crops must skip the optimizer. An apparent zero edge loss there
would be absence of supervision, not learning.

## Confidence remains separate

This audit uses the original query checkpoint's classifier and explicit
current-crop normalization. It does not overlay the later confidence replay
head on new crops or call that replay an integrated reconstruction checkpoint.

On the fitted controls, the original head accepts 2/6 assigned targets plus
three unassigned queries, and 12/12 plus two unassigned queries. Even a perfect
edge classifier under the .5 node cutoff could retain only 1/5 and 10/10 required
edges. These reproduce the saved original-head results exactly.

The subsequent confidence-only replay improved those controls to 5/6 and 12/12
with no set-label extras. That useful result is preserved. Updating a classifier
on fixed features cannot move the branched proposals into position; resuming
the proposed 4,000-update confidence replay would not resolve this decision.
It remains deferred. Unassigned queries are set-label extras, not automatically
proven biological false positives.

## Recommended bounded next experiment, for discussion

The current checkpoint shows placement learnability on two fitted crops under
a diagnosed normalization policy. It does not show branched reconstruction.
This audit rules out **this frozen checkpoint for the declared branched fit**;
it does not rule out frozen upstream models in general or establish joint
training as superior to staged training.

I recommend one capped **branched proposal learnability experiment** before
spending updates on connectivity. Train the existing PointNeXt/Transformer on
the four frozen training clouds above, keeping the field fixed and using no
manual intervention. This is not a Phase 1 reimplementation. It tests whether
the current variable-cardinality targets/loss can learn both modest and dense
branching as well as the simple controls.

Proposed settings, **not an implemented or launched run**:

- Fresh seed-zero initialization; preserve the old checkpoint as evidence.
  Use the explicit batch-one current-crop normalization candidate consistently
  in training and inference, without promoting it as a production default.
- Retain 512 queries, spacing-eight targets, the original query loss, AdamW
  LR .0001, weight decay .0001, gradient clip 10, float32 and no augmentation.
  The classifier-only LR .001 result does not justify transferring that LR to
  the full model. No joint edge loss or invented GT anchors are introduced.
- Fixed cyclic order over the four crops; maximum 4,000 updates (1,000 crop
  appearances each), evaluations at 0/500/1,000/2,000/4,000. Save optimizer/RNG
  state for genuine resumability if subsequently agreed.
- Before launch, prepare reviewed local RTX 3070 Ti timing/memory estimates
  and a visible user-launched command with ignored output paths. Neither
  timing updates nor training are launched by this report.

Require the audit's >=95% trusted targets, >=95% attainable candidate-edge
recall and all visible junctions on **every crop**. Also require >=95% of
assigned positives accepted at .5, unassigned accepted queries <= floor(5% of
target count), and >=95% reference curve support at six voxels after filtering.
These are proposed training learnability criteria, not generalization or
product-quality standards. Report every metric even when the combined gate fails.

**Stopping rules:** stop on nonfinite values, target truncation, accidental
unknown-as-background treatment or violated coordinate/data contracts. Stop
and interpret at the fixed budget; do not extend updates, lower thresholds,
substitute easier crops or change losses automatically. If geometry fails,
revisit targets/assignment/loss or encoder behavior before graph training. If
geometry passes but confidence fails, identify that selection-objective failure
without masking it with graph scores. If the combined gate passes, freeze that
common upstream checkpoint and compare GATv2 against the pairwise control using
identical features, candidates, labels and budgets. Their edge and decoded
forest results then address whether graph context earns its complexity. A tie
would favor the simpler control for discussion, not silently remove GATv2.

The destination is a valid, matched connectivity comparison on actual
image-derived proposals. Whole-neuron automatic crop matching, preserved held-out
testing and a fresh neuron-split PN1 baseline are still required before claiming
end-to-end reconstruction improvement.

## Evidence and verification

The two fitted controls exactly match earlier saved CPU per-crop-normalization
predictions for positions, radii and logits (all maximum deltas **0.0**).
Independent NumPy recomputation verifies assignments, candidate pairs, positive
labels and confidence-limited edge ceilings from saved packets. Existing
connectivity and protected upstream sources retain prior hashes. All saved
upstream model state tensors are unchanged.

Only the four selected training image NPZs and three training SWCs were opened.
No development/test image caches, source volumes or SWCs were read. The original
control clouds came from CUDA/TF32 field preparation; new branched fields ran
on CPU. No backend-equivalence claim is made. The field previously saw the wider
training partition; the query model saw only the two neuron-0 crops.

The first invocation stopped during provenance inventory, before any forward
or output directory, on a generated PyTorch filename. Restricting the inventory
to existing project sources fixed it. The completed invocation is the sole
prediction audit; no fit/checkpoint is hidden by the correction.

- [Fixed protocol and CPU command](upstream_connectivity_audit.md)
- [Complete visible summary](upstream_connectivity_seed0/summary.json)
- [Independent evidence review](upstream_connectivity_seed0/review.json)
- [Modest branched inspection](upstream_connectivity_seed0/sample_0092_patch_004.png)
- [Dense branched inspection](upstream_connectivity_seed0/sample_0086_patch_003.png)
- [CPU audit runner](../../scripts/pointneuron2/audit_upstream_connectivity.py)
- [Pinned configuration](../../configs/pointneuron2/upstream_connectivity_audit.json)

Raw clouds, feature packets, edge labels and source snapshots stay in ignored
`tmp/pointneuron2/local_runs/upstream_connectivity_20261017`. Raw summary SHA256:
`34b058ea6beee9985696507adb6b76e49512a8dee3cdaec6098b475755599478`.
