# Query proposal decoder: target specification and first comparison

11 October 2026. This specification was reviewed and its implementation is now
complete. [Execution checks and the next local command](query_decoder_implementation.md)
report 26 passing tests and bounded CPU fits on two training crops. Learnability
is not yet established; full GPU training and new reconstruction remain pending.

## The hypothesis we will test

A decoder whose proposal slots attend to the whole crop, trained to represent
an explicit set of reference nodes, may preserve useful geometry and avoid
redundant proposals better than the current pointwise head. This is a
head-and-objective hypothesis. It does not assume the decoder will resolve
ambiguous neuron identity or recover evidence absent from the input.

## Training-only evidence

[The verified audit](query_target_counts_seed0/summary.json) reads only the 262
training NPZ records and source SWCs from their 33 training-neuron directories.
It checks cache hashes, identities, source hashes and exact cached SWC vertex
crops. It does not use the dataset scanner, which would parse SWCs outside the
training partition. No source volumes, validation records, reserved test
references or checkpoints are read. Final audit runtime was 8.6 seconds on CPU.

Counts include roots, leaves, junctions and crop intersections. Quantiles use
all 262 crops, including reference-empty crops.

| Target arc spacing | Median targets/crop | 95th percentile | Maximum | Crops exceeding 512 |
| --- | ---: | ---: | ---: | ---: |
| 4 voxels | 32.5 | 201.85 | 732 | 1 |
| 6 voxels | 23.5 | 143.00 | 523 | 1 |
| 8 voxels | 19.0 | 112.00 | 418 | 0 |

The largest crop is training neuron 86, patch 3: 2,253 stored SWC vertices become
418 curve targets at spacing 8. One previously vertex-empty crop (neuron 1,
patch 3) contains crossing reference geometry and yields 110 targets. Therefore
211 crops contain target geometry and 51 remain reference-empty. Roles overlap;
role counts must not be summed to obtain target counts.

**Proposed starting choice: spacing 8 voxels and 512 query slots per crop.** This
fits every audited training crop and gives both heads the same maximum output
count. It is a pilot choice based on training geometry, not reconstruction
scores. Eight-voxel spacing is not a product resolution guarantee. Denser target
sets can be tested later as a separate capacity comparison.

The 512 query slots are output capacity. The existing 512 selected points are
input capacity; these are distinct quantities. Both arms initially receive the
same 512 inputs from the same 1,024-candidate image-only pools.

## Target geometry contract

- Work from the verified full reference SWC forest. Do not join disconnected
  fragments or skeletonize the image to manufacture supervision.
- Split the forest into chains at roots, leaves and junctions. Sample at global
  chain arc distances 8, 16, 24, ... voxels, with phase anchored at the full
  chain start before cropping. Preserve structural nodes independently of phase.
- Intersect full SWC edges with the closed cube centered at the image-defined
  patch center with radius 48. Include intersections even when neither stored
  edge endpoint lies inside. Only actual clipping intersections are forced;
  extra vertices on a box face do not change target density.
- Deduplicate the same reference identity/sample at a junction or intersection.
  Keep distinct coincident reference nodes identifiable. Such coincidences and
  fragment ends are annotation properties, not certified biological landmarks.
- Interpolate SWC radius along edges. Preserve raw values and apply the existing
  one-voxel target floor in both training arms. Radius is an annotation-derived
  size in voxel units, separate from NMS/support radius.
- Store a target count, xyz, radius, role flags and provenance. Validate cycles,
  parents and finite values. Never truncate targets to fit query capacity.
  A future crop exceeding capacity must trigger an explicit overflow decision;
  passing these 262 crops does not guarantee all inference crops will fit.
- Use voxel-coordinate Euclidean distance in this pilot. The existing spacing
  configuration is (1,1,1); physical calibration and anisotropic data need a
  separate contract before broad deployment.

The audit implements this geometry independently of the production cache.
Nine focused tests cover crossings, radius interpolation, vertex subdivision,
junctions, sample phase, box-face curves, coincident nodes, zero-length edges,
cycles and empty crops. All nine passed. Old caches and label arrays are intact.

## Supervision policy and its limits

The immediate experiment measures agreement with the supplied reference.
For a crop containing reference geometry, supervise matched query slots as
used and unmatched slots as no-object. This teaches a compact representation
of the annotated target set, including rejection of redundant slots. It also
implicitly penalizes predictions of unannotated geometry. That is an explicit
weak-label assumption, not verified biological background.

The pointwise arm uses the same reference geometry and the existing proximity
classification rule; its positives and query-slot positives have different
definitions. Both arms skip the 51 reference-empty crops for supervised loss.
Normalize each crop's loss separately and average over supervised crops. Skip
optimizer updates for entirely unlabeled batches; no Adam momentum update or
weight decay should occur for such a batch.

Keep the existing foreground class weight 8 versus no-object/background weight
1 as the starting comparison. Objectness scores are not assumed calibrated or
interchangeable between heads. Report precision-recall curves and recall at
matched precision, rather than interpreting identical numeric thresholds as
identical confidence. Biological target discrimination remains unproven where
the reference is incomplete. The current data do not justify labeling all bright
off-reference structures as true negatives.

## Proposed decoder and loss

Use the inherited DGCNN architecture and its 1,216-dimensional per-input features.
Project them to dimension 128 and add learned positional features from local xyz.
Use 512 learnable query embeddings, three decoder layers, four attention heads,
a 512-dimensional feed-forward layer and dropout 0.1. These are starting settings,
not optimized values. Self-attention lets slots interact; cross-attention lets
them use the crop's input evidence. This is a DETR-inspired adaptation, not a
claim that DETR's object-detection results transfer to neurons.

Predict a local center vector, two objectness logits and positive radius for
each slot. Express local coordinates relative to the image-defined crop center,
scaled by radius 48; transform centers back to voxel coordinates for metrics
and export. Keep coordinate outputs unbounded initially because shared rotations
can move targets outside the original axis-aligned cube. Apply exactly the same
coordinate transforms to inputs, targets and crop center; check translation
equivariance and augmentation consistency.

Use one-to-one assignment on detached costs, followed by differentiable losses
on the selected predictions. Proposed matching cost is
`-p(used) + 3 * Euclidean(center,target)/6 + 0.2 * abs(radius,target_radius)/6`.
The six-voxel scale follows the declared existing spatial tolerance. Compute
assignment per supervised crop, fail on target overflow and handle empty-target
crops explicitly. Hard assignment itself is not differentiable; gradients flow
through matched predictions, not through the selected pairing.

Start with per-crop loss weights 2 for weighted objectness cross-entropy, 3 for
matched-center Smooth L1 on Euclidean voxel error (beta 4), and 0.2 for matched
radius Smooth L1 (beta 1). Average regression terms over matched targets. These
retain the repaired recipe's principal scales, but query matching and loss
terms differ from the pointwise arm. Do not compare raw training losses across
arms as if they were the same objective. Log each term and used-slot count.

Use a dedicated query output type and versioned checkpoint. Do not invent
pointwise offsets to make it resemble the old head. The legacy graph adapter
receives centers, scores and radii. For feature-based tracing, associate proposals
with the nearest input point's encoder feature in both arms and record that
association distance. This common adapter is an explicit integration change;
the historical checkpoint/output remains an archived reference, not the matched
new control.

## Controlled experiment and decisions

Retrain the local pointwise control and query arm on these common targets and
supervision policy. Initialize their encoders from identical seeded states;
match data order, point selection, augmentations, optimizer settings and update
budgets. Share AdamW lr 0.0001 and weight decay 0.0001 initially. Record model
size, update counts, wall time and peak GPU memory. The heads have different
parameter counts and initializations; equal epochs do not imply equal compute.

First perform a small learnability check on annotated training crops. Then
prepare, without launching, a proposed 30-epoch-per-arm pilot. Save epochs
5/15/30 and use the final epoch for the primary comparison. Five epochs alone
must not settle a new decoder's quality. Confirm the epoch budget and duration
after measuring the small check on the local RTX 3070 Ti Laptop GPU. The user
will launch any agreed pilot with live terminal output. Additional seeds and
whole-neuron tracing are separate agreed runs.

Evaluate raw proposals before suppression and every selection stage afterward.
Keep local/global consolidation and the downstream graph recipe fixed between
arms for the primary integration comparison. Query uniqueness within a crop
does not remove duplicates between overlapping crops. Legacy suppression can
still remove useful query outputs; raw and post-selection metrics distinguish
that limitation without automatically changing the graph rules.

Proposed advancement gates, to freeze before training:

- For raw proposals, improve mean reference-curve recall at matched precision
  0.80 across seven validation neurons, at the same output limits. If an arm
  cannot attain that precision, mark the gate unmet. Report the full curves.
- For whole-neuron SWCs, gain at least 0.02 mean spatial F1, lose at most 0.01
  mean precision, and have no individual F1 regression exceeding 0.05. Report
  branch/endpoint support, off-reference paths, runtime and memory alongside it.
- Repeat a successful comparison with a second seed before advancing the
  component. These are proposed research gates, not commercial release criteria.

The same seven validation neurons have already informed substantial development;
they support internal decisions only. Keep the eight reserved test neurons
untouched. Do not promote a component from training loss, one favorable neuron
or selected-node topology F1. If it cannot fit a few training examples, investigate
implementation/optimization before broad evaluation. If a corrected matched
comparison still fails, retain the control and redesign or drop the component.

## Files and next implementation scope

- Diagnostic: `scripts/pointneuron2/audit_query_targets.py`.
- Checks: `tests/test_query_target_audit.py`.
- Evidence: `docs/pointneuron2/query_target_counts_seed0/summary.json`.
- Next proposed source: curve target builder, query head, set loss, versioned
  checkpoint/inference adapter and matched runner in the real PN2 checkout.
  No PointNeXt, sampler or GATv2 change is bundled into that implementation.

Original set-prediction references: [DETR paper](https://arxiv.org/abs/2005.12872)
and [authors' matching/classification implementation](https://github.com/facebookresearch/detr/blob/main/models/detr.py).
They support the architectural mechanism, not these PN2 settings or an expected
neuron-reconstruction gain. All numeric PN2 settings above are our proposed
experiment choices.
