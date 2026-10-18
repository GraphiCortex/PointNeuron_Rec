# Assignment and confidence: design audit before another fit

18 October 2026. Read-only review after rejection of the hard-negative variant.
No model/loss/matcher/inference policy is changed. No training run or replacement
architecture is prepared. The balanced fit remains the stronger incomplete
reference. This note separates the current contract, observed label changes and
the next question worth answering.

## What the implementation actually asks the model to do

The query head produces 512 centers, radii and two-class confidence logits from
the image-derived PointNeXt memory. Three Transformer decoder layers produce
128-dimensional query features. A shared linear/ReLU hidden layer feeds a final
six-output linear layer: two confidence logits, radius and three coordinates.
Changing confidence gradients therefore also changes features used by geometry.
The query ID alone is not the classifier's input; features depend on the crop.

The assignment is recomputed each supervised update. In
[query_loss.py](../../src/pointneuron/models/query_loss.py), the per-query/target
cost is:

`-p_used + 3 * center_distance / 6 + 0.2 * absolute_radius_error / 6`.

Hungarian matching assigns every cached target to a distinct query. An assigned
query receives confidence label one **regardless of its placement error**. Every
other query receives label zero, even if it lies on an annotated neurite. That
label means "not chosen as a unique representative of this target set", not
"image background" or "geometrically incorrect". Confidence is not directly
supervised as probability of passing the two-voxel trust test.

On an individual candidate/target cost, the entire probability term spans at
most one cost unit, equivalent to two voxels of center cost with other terms
fixed. Global one-to-one coupling makes assignment changes more complicated
than comparing one candidate pair. This coefficient fact is not evidence that
confidence is dominating the observed matches.

The center/radius losses supervise matched queries. Unmatched centers receive
no direct center target or repulsion term in this loss. All target nodes have
the same center/radius weighting: junction identity/adjacency is not an input to
the proposal loss. The explicit all-junction preservation gate therefore checks
an outcome that is only indirectly encouraged by these pointwise losses.
These are design tradeoffs, not proofs that set prediction is ill-posed.

## Query reassignment and binary-label changes differ

The earlier reports correctly counted changed target-to-query maps. That is not
the number of positive/negative label flips. The new
[saved-assignment audit](../../scripts/pointneuron2/audit_query_assignment_contract.py)
checks both, with prediction/report hashes pinned to the completed paired review.
It reads all five snapshots of both fits, producing 32 crop/interval rows.
No predictions are recomputed or relabeled on disk.

Between updates 2,000 and 4,000:

| Fit / crop | Targets changing matched query | Previous positives replaced | Binary labels flipping, among 512 | Previous positives retained |
|---|---:|---:|---:|---:|
| Balanced 0/0 | 5/6 | 5/6 | 10 | 1/6 |
| Hard-negative 0/0 | 6/6 | 6/6 | 12 | 0/6 |
| Balanced 0/1 | 7/12 | 7/12 | 14 | 5/12 |
| Hard-negative 0/1 | 10/12 | 10/12 | 20 | 2/12 |
| Balanced 92/4 | 9/16 | 9/16 | 18 | 7/16 |
| Hard-negative 92/4 | 14/16 | 14/16 | 28 | 2/16 |
| Balanced 86/3 | 83/418 | 30/418 | 60 | 388/418 |
| Hard-negative 86/3 | 267/418 | 62/418 | 124 | 356/418 |

Every positive leaving is balanced by a negative entering, because target count
is fixed. Thus 62 replacements imply 124 binary flips, not 267 or 62 flips.
In the hard-negative dense crop, another 205 queries stay positive but switch
their matched target. In the balanced dense crop that count is 53.

The sparse crops have fewer flips among all 512 queries, but they replace a
large fraction of the rare positives. Both denominators matter. The hard fit
has more replacements on all four crops in this interval; that is an observed
association, not an isolated explanation for its lower quality.

Output sets do not require stable query IDs to be correct. A query can move to
a different target as the model learns while the final geometric set improves.
These endpoints do not reveal per-update switching, how long a query was
positive, or whether switching preceded or followed a geometry change. Therefore
the audit does not justify freezing assignments in ordinary training, removing
confidence from matching, or declaring unstable labels the diagnosed cause.

## Consequences at the graph interface

The existing graph labels accept only trusted assigned endpoints. Unmatched or
poorly placed endpoints produce unknown edge labels, not supervised edge
negatives. The decoder thresholds nodes and preserves accepted isolated nodes.
Consequently, confidence's current assignment-membership target, the geometric
trust test and the final node-set responsibility are related but distinct.
Passing a poor node set to GATv2 does not supply a supervised cleanup mechanism.

This is a reason to make the interface explicit before changing objectives.
It is not a decision to make confidence a foreground detector or give GATv2 a
new node-deletion role. Those changes would require an agreed design and matched
comparison. Fully automatic reconstruction remains the goal.

## Recommended next question, not a prepared fit

Before another joint GPU experiment, ask:

**Can the existing final confidence layer represent the frozen four-crop
assignment labels at all, using the balanced model's actual classifier inputs?**

The previous linear-separability result concerned a different two-crop model and
feature cache. It does not answer this four-crop question. The current final
classifier receives the 128-dimensional **post-hidden-layer ReLU features**, not
the saved raw Transformer decoder features directly. A control must reconstruct
those actual inputs from saved decoder features and checkpoint head weights,
then verify that they reproduce saved logits before any capacity conclusion.

The smallest proposed check is a CPU linear-separability feasibility calculation
over the fixed 2,048 query features and 452 positive assignment labels. Freeze
all model parameters, geometry and assignments; do not rerun image/PointNeXt/
Transformer inference, train a head or tune a threshold. A candidate separator
must be verified through float32 classifier arithmetic at cutoff .5 on every
cached label. Solver tolerance, feature scaling and numerical margin must be
reported; ambiguous numerical results remain inconclusive.

- If a verified separator exists, the fixed features permit the current linear
  confidence responsibility on these examples. Joint learning/optimization
  remains the unresolved question; this does not prove assignment switching is
  the cause or authorize another joint fit automatically.
- If robust separation cannot be established, do not assume another CE weighting
  will solve the unchanged frozen features. Review feature/head capacity and the
  unique-set target contract, while distinguishing numerical uncertainty from
  evidence of practical nonseparability. This would not prove the jointly trained
  architecture incapable, because it can change its features.

This is a training-label capacity oracle, not automatic inference, a replacement
classifier to deploy, generalization or reconstruction improvement. It has a
specific decision purpose: distinguish feature/head representability from joint
learning before spending another GPU fit. The check is proposed for discussion;
no implementation or solver run is prepared by this note.

The broader milestone is unchanged: useful, selected skeleton nodes followed by
learned connectivity and matched PN 1.0/PN 2.0 reconstruction evaluation. No
architecture departure or additional loss sweep follows from this audit.

Evidence: [all saved label-transition counts](query_assignment_contract_review_seed0/summary.json).
