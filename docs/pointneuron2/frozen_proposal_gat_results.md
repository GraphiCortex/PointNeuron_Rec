# GATv2 learns all four fixed graphs with oracle-selected learned proposals

Subsequent integration result, 19 October: the agreed
[automatic-selection comparison](gat_automatic_selection_results.md) is complete
without retraining and exposes the node-selection gap. The conditional success
reported here remains valid; no further fit or production promotion follows.

18 October 2026. The user-launched 4,000-update connectivity fit **passes the
declared graph gate on every training crop**. The first evaluated passing
snapshot is update 500; updates 1,000, 2,000 and 4,000 also pass. Both independent
edge classification and actual forest decoding are correct. This is conditional
training-set learnability, not automatic image-to-SWC reconstruction.

## Terminal result

| Training crop | Retained nodes | Required edges recovered | Wrong decoded edges | Components, predicted / reference | Degree / junction-incidence errors | Gate |
|---|---:|---:|---:|---:|---:|---|
| 0/0 | 6 | 5/5 | 0 | 1/1 | 0/0 | Pass |
| 0/1 | 12 | 10/10 | 0 | 2/2 | 0/0 | Pass |
| 92/4 | 16 | 15/15 | 0 | 1/1 | 0/0 | Pass |
| 86/3 | 418 | 404/404 | 0 | 14/14 | 0/0 | Pass |

All **434 required candidate edges** are accepted and all **3,735 known negative
candidates** rejected at the fixed .5 cutoff. No unknown labels occur in these
four prepared graphs. The decoder has no wrong eligible edges to hide through
cycle rejection at the passing snapshots. Every undirected reference edge is
recovered; matching degree/component totals alone was not the success test.

The 14 dense components are the crop-visible reference forest. They do not mean
the biological neuron should consist of 14 globally disconnected pieces.

## What was actually trained

The [agreed protocol](frozen_proposal_gat_ready.md) uses the completed balanced
proposal model's saved coordinates, radii, decoder features and confidence
values. Reference assignment membership selects the nodes, in original query-ID
order. The scorer does not receive reference positions, adjacency, node roles
or target-order indices as inputs. Reference correspondence supplies training
edge labels, including marginally misplaced assigned nodes; the ordinary
two-voxel placement mask is deliberately absent from this diagnostic only.

Only the existing native-PyTorch GATv2 prototype trains: 119,041 parameters,
two layers, four heads, 128 channels, fresh seed zero. AdamW LR .001, weight
decay .0001, clip ten, cyclic one-crop updates, 1,000 updates per crop. No
upstream forward/training, threshold change, new images, development/test
evaluation, automatic node cleanup or crop merging occurs. The scorer sees
original learned confidence, while decoding retains all oracle-selected nodes.

GPU timing/reload passed before the separate retained fit. The fit reports
302.94 seconds (5.05 minutes), versus about 5m28s including launcher/startup.
There were zero clipped updates; maximum preclip norm was 4.225 against clip ten.
Checkpoint reload on the training GPU has zero score difference. It stops at
the declared 4,000-update cap and the terminal checkpoint is non-resumable.

## Trajectory and geometry

| Update | Crops passing exact graph gate | Crops passing independent edge classification | Smallest signed edge logit margin across all crops |
|---:|---:|---:|---:|
| 0 | 0/4 | 0/4 | −.8494 |
| 500 | 4/4 | 4/4 | 1.0844 |
| 1,000 | 4/4 | 4/4 | 4.0040 |
| 2,000 | 4/4 | 4/4 | 6.0525 |
| 4,000 | 4/4 | 4/4 | 8.0849 |

The evaluation cadence cannot establish the first individual update that passed.
Update 500 is the first **saved/evaluated** success. The terminal model remains
the declared result; no intermediate checkpoint is selected retrospectively.
Large training margins do not establish calibration or microscopy robustness.

The smaller crops have 100% arclength precision and recall within two voxels.
Dense geometry is **98.77% precision / 97.98% recall** at two voxels; all four
reach 100% at six voxels. These exactly reproduce the correct-connection
feasibility result on the same fixed learned coordinates. Connectivity learning
does not move the remaining placement errors or improve radius estimates.
Metrics use continuous reference segments clipped to the target crop, with
the declared arclength quadrature at maximum .5-voxel intervals.

All 20 exported SWCs preserve the corresponding decoded nodes, undirected
connections, components, coordinates and floored radii. Root orientation and
generic node type are serialization conventions, not biological predictions.
These are crop-level diagnostic exports, not a stitched whole-neuron result.

## Independent verification

A read-only CPU review reconstructs selected query IDs and target correspondence
from the balanced model's original saved assignments. It checks exact frozen
positions/radii/confidence/decoder features, independently rebuilds candidate
pairs and reference labels, and verifies their equality with the control cache.
This checks that target order is not substituted for query order or fed into
the scorer. Oracle selection itself remains deliberate reference knowledge.

All five saved checkpoints have the declared schema/specification, recipe,
provenance, optimizer cursor, next-crop cursor and cap. The initial model exactly
matches fresh seed-zero weights; its graph fails. All 20 crop/snapshot score
tensors are replayed from checkpoint weights on CPU. Maximum absolute CPU/GPU
score difference is **3.815e-5**, within `atol=1e-5, rtol=1e-5`; every binary
decision is identical. This cross-device numerical comparison is distinct from
the original GPU disk-reload delta of zero.

The review recomputes all graph errors, edge losses and geometric scores from
saved logits, then reads every existing SWC without overwriting it. Serialized
edges match decoded edges; parent order, node counts, roots, coordinates and
radii pass. Fifty current source hashes, 85 protected source hashes and frozen
cache fingerprints match. No optimizer update, GPU work or upstream model
forward is performed by the review. The 28 prior targeted tests are preserved;
model/training sources are unchanged, so those tests were not repeated.

Terminal checkpoint SHA256:
`25b052286165a52fbf20bf7ca34f5bd404571ad2da2f70c53f0852123db88c9a`.
See [full verified evidence](frozen_proposal_gat_results_seed0/summary.json).
Raw scores, checkpoints and SWCs remain under ignored
`tmp/pointneuron2/local_runs/frozen_proposal_gat_20261018/model`.

## Research interpretation and next discussion

This is positive evidence for the intended architecture: the existing graph
scorer can learn exact connectivity on useful but imperfect nodes generated by
the current PointNeXt/Transformer, **when reference knowledge selects the node
set and establishes training correspondence**. It removes an immediate reason
to replace GATv2 based on lack of learnability on these fixed examples.

It does not establish generalization, an advantage over the available pairwise
edge-head control, correct confidence selection, tolerance of extra/missing
nodes, boundary consolidation, or improvement over PN 1.0. Conditional graph
success does not repair or invalidate the earlier failed confidence fits.

Keep the balanced full proposal checkpoint and this successful conditional GAT
checkpoint separately. No model is promoted into production and no further fit
or extension is prepared.

The smallest proposed next comparison needs **no new training**: run this same
terminal GAT checkpoint on the same crops using the balanced model's existing
.5 automatic node selection, rebuilding candidates from those retained nodes.
Measure which reference connections survive, which wrong connections/bridges
appear, and the resulting geometry/components, alongside the known node extras
and omissions. Its purpose would be to quantify the oracle-to-automatic
integration gap, not demand an exact node graph when known nodes are absent or
declare a distribution-shift failure proof that GATv2 cannot work.

GATv2's current decoder retains accepted isolated nodes, so simply removing
edges is not a supervised node-cleanup mechanism. The comparison must not
silently add pruning, a new node head or a different cutoff. The user subsequently
agreed to that comparison; its outcome is linked above. It does not change the
architecture or deploy either model.
