# Proposed comparison: concentrate part of the negative loss

18 October 2026. This records the design specification. The user subsequently
agreed to the [isolated implementation and CPU preparation](hard_negative_query_ready.md).
GPU timing and the retained fit subsequently completed; the
[paired result](hard_negative_query_results.md) fails the predeclared comparison.
No further fit is prepared. Existing baseline model/loss sources, sampler,
inference policy and architecture remain unchanged. The completed balanced fit
remains stopped at 4,000.

## Responsibility and hypothesis

Retain the current interface: the Transformer proposes a compact skeleton node
set; confidence selects unique set representatives; GATv2 predicts connectivity.
Confidence is not redefined as biological foreground. No suppression/merging,
manual masking or new graph node-rejection stage is introduced.

The balanced fit improves ranking, but has 11/32/17 accepted unmatched queries
on its smaller training crops. Its negative mean averages over 506/500/496
queries. Hypothesis: concentrating part of that negative contribution on the
highest-scoring unmatched queries can reduce extras at the fixed .5 cutoff,
without losing matched positives or trusted placement. This is an optimization
hypothesis, not a diagnosis that all residual errors arise from loss dilution.

## Exact candidate objective

For one supervised crop, let P be the number of Hungarian-matched queries and N
the number of unmatched queries. Here Q=512 and N=Q-P. Define:

- A: mean positive cross-entropy over all P matched queries.
- B: mean negative cross-entropy over all N unmatched queries.
- H: mean negative cross-entropy over the K highest-scoring unmatched queries,
  where **K=min(P,N)**. Negative CE increases monotonically with the used score.

The current objective is `C_balanced = 0.5 A + 0.5 B`.

The proposed objective is **`C_candidate = 0.5 A + 0.25 B + 0.25 H`**.

Thus positive/negative coefficient mass remains .5/.5, and all negatives retain
supervision. Half of the negative mass is redistributed to the difficult subset.
K uses a fixed one-hard-negative-per-positive rule, capped by available negatives;
it is not selected from a sweep or a desired final result. This choice and the
equal split are experimental design choices, not established optima.

Recompute the hard subset each update after the unchanged Hungarian assignment.
Only unmatched queries are eligible. Resolve equal scores by increasing query
index; selection indices are detached, and gradients pass through the gathered
CE values. Matched queries never enter the negative set. The target count is
already known to the training loss; it is not an inference input or a top-k
inference rule. Keep all 512 proposals available at inference, with cutoff .5.

If P=Q, use the positive mean alone, as in the current loss. Missing local SWC
annotation remains unknown and is skipped, never treated as a background-only
crop. The proposed rule does not establish how to supervise independently
verified empty images; that is outside these four fixed crops.

Retain outer classification coefficient 2, center coefficient 3 and radius
coefficient .2. No geometry loss, assignment cost, model or shared-feature policy
changes. Equal coefficient mass does not imply equal gradient magnitude or
equal influence relative to geometry. Increased hard-negative gradients can
compete with positive/geometry learning and can increase gradient clipping.

## Algebraic check on saved scores

The [read-only audit](../../scripts/pointneuron2/audit_hard_negative_objective.py)
uses the completed balanced fit's final scores and fixed saved assignments.
It computes proposed CE and analytic derivatives with respect to the score
margin; it performs no model forwards, neural backpropagation or updates.

| Crop | P / N | K | Current accepted extras in hard subset | Hard-negative derivative multiplier at fixed scores |
|---|---:|---:|---:|---:|
| 0/0 | 6 / 506 | 6 | 6 of 11 | 42.67 times |
| 0/1 | 12 / 500 | 12 | 12 of 32 | 21.33 times |
| 92/4 | 16 / 496 | 16 | 16 of 17 | 16 times |
| 86/3 | 418 / 94 | 94 | 4 of 4 | 1 time |

Non-selected negatives receive half their current per-query coefficient; the
positive coefficient is unchanged. The proposed retrospective classification
losses are .598/.980/.528/.158 versus .042/.137/.070/.158. Those increases are
imposed by the formula, not measured learning or improved predictions.

For the dense crop K=N, so **the candidate formula equals the current formula
for every possible output with this target count**, not just these saved scores.
This isolates the direct change to sparse-crop supervision. Shared parameters
still mean the dense crop's learned predictions may change. No claim of an
unchanged dense training trajectory is made.

These checks establish that the proposed loss targets the intended saved errors.
They do not establish linear separability, attainable joint optimization, clean
selection, generalization or reconstruction. See [saved algebraic evidence](hard_negative_objective_proposal_seed0/summary.json).

## Matched experiment if this design is accepted

Use the completed balanced fit as the primary comparator, not the older weighted
CE fit. Start fresh from its exact seed-zero initialization; do not continue its
terminal weights with a fresh optimizer. Copy the exact same frozen cache:
neuron 0/crops 0,1, neuron 92/crop 4 and neuron 86/crop 3.

Keep the PointNeXt/Transformer architecture, per-crop normalization, 512 queries,
LR .0001, AdamW weight decay .0001, clip 10, float32, batch one, no augmentation,
crop order and 4,000-update cap. Keep snapshots at 0/500/1,000/2,000/4,000.
Primary candidates remain kNN16 union radius eight; radius twelve is reported
separately on both models. Do not fit GATv2 or read development/test data.

Before a retained fit: check loss/gradient identities on fixed assignments,
empty/single-class handling, hard-set exclusion/tie behavior, dense-crop equality,
checkpoint policy identity and a discarded synthetic full-model update. Then
prepare visible, separate GPU timing and training commands. Device is the RTX
3070 Ti Laptop GPU. Prior fits took about 18–20 minutes; allow a fresh measured
estimate before committing to duration. Generated outputs belong in a new
ignored `tmp/pointneuron2/local_runs` directory; sources/reports stay visible.

## Decisions at the fixed cap

The hypothesis is supported on these four training crops only if **every crop
passes the original selection criteria together**:

- Accepted assigned queries: 6/6, 12/12, 16/16 and at least 398/418 at .5.
- Accepted unmatched queries: at most 0, 0, 0 and 20, respectively.
- Accepted curve support within six voxels at least 95% on every crop.

Also require no regression in trusted-target counts (6/12/16/404) or trusted
junction counts (0/0/2/76) relative to the completed balanced fit. Report mean/max
center error, radius error, accepted-and-trusted count, score ranking, assignment
changes, and required-edge availability both before and after confidence.
Inspect radius/geometry tradeoffs explicitly; selection improvement alone does
not establish a uniformly better model.

That is an objective-comparison decision, not the full proposal milestone.
The original combined gates still require all visible junctions, at least 95%
trusted placement and candidate recall, plus known positive/negative edges.
In particular, retaining 76/77 dense junctions would preserve the comparator but
still fail the full milestone. Passing old gates also does not automatically
authorize connectivity training: inspect the node-filtered graph interface first.
No success on these training crops establishes a matched PN 1.0 improvement.

If extras fall but any selection criterion remains unmet, record partial evidence,
not success. If positives or geometry deteriorate, the redistribution has failed
its proposed tradeoff. If assignment changes remain large, do not infer from
snapshots alone that churn caused the result. Harder negative penalties may also
encourage assignment changes because matching itself uses confidence.

Stop at 4,000; stop earlier on nonfinite values or provenance/checkpoint failure.
No extension, threshold search, K/coefficient sweep, automatic sampler change or
automatic graph training. If this comparison fails, return to the assignment and
unique-set representation contract before proposing another loss variant.
