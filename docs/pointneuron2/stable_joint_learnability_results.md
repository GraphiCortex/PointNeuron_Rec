# Stable joint fit: stable geometry, two complete crop passes, residual selection and target-identity errors

20 October 2026. **Completed at the agreed cap; learnability gate fails.**
The user-run GPU job completed 3,000 new updates from historical joint stage
1,000 to stage 4,000, at LR 0.00001, in 4,259 report seconds (71.0 minutes).
No extension, threshold change, checkpoint selection or new objective follows.

This is useful training-set progress, with a narrower failure than before.
All 452 assigned targets are accepted and within two voxels at the terminal
snapshot. All targets also remain trusted at every recorded all-crop probe.
Two crops meet the complete automatic reconstruction criterion. The two remaining
failures have different meanings: a real extra spur, and exact adjacency errors
involving almost coincident endpoint samples.

## Terminal automatic output

| Training neuron/crop | Accepted targets | Extras | Correct required edges | Nonreference edges | Precision / recall within 2 voxels | Full gate |
|---|---:|---:|---:|---:|---:|---|
| 0/0 | 6/6 | 0 | 5/5 | 0 | 100% / 100% | Pass |
| 0/1 | 12/12 | 1 | 10/10 | 1 | 93.73% / 100% | Fail |
| 92/4 | 16/16 | 0 | 15/15 | 0 | 100% / 100% | Pass |
| 86/3 | 418/418 | 0 | 402/404 | 2 | 99.96% / 99.71% | Fail |

Totals are 452 assigned targets, one extra node, 432/434 required edges and
three nonreference decoded edges. Aggregation is descriptive; the original
criterion requires every crop to pass. All four component counts match their
reference crop forests (1, 2, 1, 14). All 79 reference junctions are retained,
trusted and have the expected degree. Matching those counts alone does not
establish exact adjacency. All terminal SWCs validate and serialize the decoded
graphs faithfully. Radii are serialized with the existing floor of one voxel;
this experiment has no independent radius-accuracy acceptance criterion.

## What the residual errors actually are

**Crop 0/1: an extra proposal becomes a spur.** Query 98 has confidence
0.636887 and is 6.335 voxels from the nearest target node. It connects to the
proposal assigned to target 1 through an edge of length 6.333 voxels. Continuous
curve precision confirms that this is extra reconstructed structure, rather
than merely a harmless additional sample. All reference edges remain present.

That edge's training label is unknown, because it touches an unmatched query.
This follows the declared masking policy; it was not relabeled as background.
Proposal confidence is responsible for suppressing the unmatched node. GAT's
current edge objective supplies no direct rejection target for this edge. This
is a real limitation of the division of responsibilities, rather than evidence
that the graph model learned to reject arbitrary spurious nodes.

Confidence ordering has nevertheless improved: the highest unmatched probability
across the four terminal crops is 0.636887; the lowest assigned probability is
0.902079. A separating cutoff therefore exists **on these frozen training
predictions**. This differs from earlier snapshots with overlapping scores.
No alternative cutoff is applied, validated or promoted. The specified 0.5
decision still fails, and a post-hoc calibration observation proves neither
generalization nor perfect optimization of the existing classifier.

**Crop 86/3: two endpoint/interior-sample order reversals.** Both missing
required edges are available candidates and score above the unchanged edge
cutoff. Higher-scoring known-negative alternatives enter the forest first,
making the true edges cyclic. The decoder follows its documented score order;
there is no evidence here of an implementation error in Kruskal decoding.

| Zero-based target identities | Reference path | Selected path | Reference leaf/sample separation |
|---|---|---|---:|
| 248, 249, 57 | 248 — 249 — 57 | 248 — 57 — 249 | 0.164490 voxels |
| 405, 406, 129 | 405 — 406 — 129 | 405 — 129 — 406 | 0.143616 voxels |

Targets 57 and 129 are reference leaves; 249 and 406 are interior arc samples.
Each leaf exchanges its degree-one role with its adjacent degree-two sample.
The predicted short edges measure 0.145636 and 0.050555 voxels respectively.
The conflicting negative/positive edge logits are 3.231430/0.964904 and
3.538374/1.679111. These are genuine exact-label errors, but describing them
simply as major wrong branch connections would overstate their anatomical size.

A reference-conditioned diagnostic contracts just those two common short edges
in the reference and prediction. The entire resulting dense graph edge sets
then agree. This explains the discrepancy; it is **not** a deployable correction,
an adopted equivalence metric, or a pass of the original gate. It also does not
establish biological accuracy at subvoxel scales.

The target generator deliberately preserves structural nodes and full-chain arc
samples as distinct identities, including nearly coincident ones
([implementation](../../src/pointneuron/data/curve_targets.py)). Here an arc
sample lands extremely close to a leaf. This is evidence to review that target
contract before attributing the residual entirely to encoder/GAT capacity.
Spatially merging arbitrary nearby targets would be unsafe: separate branches
can lie close together. Any proposed canonicalization should use reference
chain identity, preserve structural nodes, and be agreed before implementation.

## Trajectory and integrity

| Historical stage | Full crop passes | Extras on 0/1 | Correct dense edges |
|---|---:|---:|---:|
| 1,000 (imported) | 0/4 | 18 | 399/404 |
| 1,064 (exact replay) | 1/4 | 18 | 402/404 |
| 1,500 | 1/4 | 15 | 402/404 |
| 2,000 | 3/4 | 12 | 404/404 |
| 3,000 | 2/4 | 3 | 399/404 |
| 4,000 (terminal) | 2/4 | 1 | 402/404 |

The dense crop passed fully at 2,000 and subsequently lost exact adjacency.
No saved stage passes all four crops. Selecting different checkpoints per crop
would not establish one successful automatic model. Stability of placement
does not imply monotonic edge accuracy or convergence of the selection losses.

The CPU audit verifies all 24 saved prediction/SWC pairs, inclusive 0.5 node
decisions, inherited candidates/scores, transferred training labels, graph and
continuous-geometry metrics, original gates, and SWC serialization. It checks
all six saved checkpoint states and the 3,000-entry update ledger. Initial
weights/moments/RNG match the parent; stage 1,064 model/Adam/RNG exactly match
the preserved short control. All 3,000 updates have both known edge classes, so
this particular run does not exercise an actual all-unknown GAT skip.

All recorded all-crop probes retain 100% trusted matches. Terminal mean matched
distances are 0.0412, 0.0522, 0.0366 and 0.0560 voxels. The declared final-window
stability check passes. Runtime reports exact final checkpoint prediction
reload (delta zero); the separate CPU audit checks saved arrays and state and
does not rerun model inference. Protected source/artifact hashes remain intact.
The terminal checkpoint is intentionally nonresumable.

## Research decision

Execution and long-run placement stability succeeded. Complete four-crop
training-set learnability under the fixed output contract did not. This does
not justify claiming that the overall architecture is broken, or that its
components generalize. It also does not justify automatically repeating the fit.

**Recommendation before another GPU run:** discuss the target representation
and output contract. First establish how near-endpoint arc samples should be
represented and how anatomical connectivity should be assessed independently
of arbitrary sample identity. A small CPU audit of the existing training target
cache can quantify this pattern without fitting a new model. Separately decide
how node-confidence calibration and unknown edges touching extras should be
handled. Those decisions can inform one prospective, bounded comparison;
neither a new loss nor a target rewrite is implemented by this review.

Do not discard the current model: it supplies a stable reference and saved
outputs for checking any proposed change. Whole-neuron crop merging, unseen
reconstruction quality, matched PN 1.0 superiority, and product readiness remain
unestablished. These four heavily inspected training crops cannot answer them.

## Evidence and reproduction

- [Original prospective protocol](stable_joint_learnability_ready.md).
- [Verified results/provenance](stable_joint_learnability_results_seed0/summary.json)
  and [reference-conditioned residual interpretation](stable_joint_learnability_results_seed0/residual_interpretation.json).
- [Saved-output audit](../../scripts/pointneuron2/review_stable_joint_learnability.py)
  and [residual inspection](../../scripts/pointneuron2/inspect_stable_joint_residuals.py).

Run both scripts with the local sampling environment's Python from this worktree.
They perform no model forward, optimizer update or CUDA initialization. Their
ignored audit outputs refuse overwrite. Raw fitting evidence remains under
`tmp/pointneuron2/local_runs/stable_joint_learnability_20261020/model/`.
Terminal checkpoint SHA-256:
`4c18cc7cdee82c331864641f9ea90204370d5fb2788f3258809b37c8d64666fa`.
