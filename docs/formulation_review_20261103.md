# Review of the proposed curve/voting reformulation

3 November 2026. The user supplied an external review proposing curve-based
diagnostics and dense per-point support/offset voting in place of exact node-set
prediction. This document evaluates that proposal against current code/evidence.
It does not authorize or implement an architecture replacement or launch a fit.

## Assessment

The locally identifiable supervision proposal is a serious research direction.
The evidence supports examining whether exact interior-node targets are an
unnecessary task, rather than continuing confidence/weight adjustments. It does
**not** yet establish that target phase caused most failures, that the encoder
is adequate, or that dense voting will work. The four-box fit establishes
memorization/learnability on those examples, not transferable geometry learning.
The small development-recall decline after five extra epochs is compatible
with overfitting, but one seed/two checkpoints is insufficient to diagnose it.

## Important metric clarification

The present scorer in `src/pn2/evaluation/node_cohort.py` already measures:

- Accepted-node distance to the actual reference line segments (and isolated
  target points), using the exact segment index in `geometry.py`.
- Reference-curve length within tolerance of accepted **node locations**, using
  arclength-weighted quadrature at 0.5 voxels.
- Support from the reference target grid itself, exposing its spacing limitation.
- Node agreement and geometry diagnostics in reference-present boxes, with
  unknown-box outputs separately disclosed.

The reverse direction is **curve-to-isolated-node support**, not distance to
predicted graph edges. Consequently it is node-spacing dependent. Earlier chat
language treating declining support as proof of missing whole neuronal branches
was too strong. It establishes lost proximity to accepted anchors under this
metric, not final skeleton fidelity. Node-only prediction has no predicted edges
yet, so a valid continuous reconstruction cannot be inferred either way.

Latest development results at two native voxels, macro-averaged over seven
neurons, directly aggregated from the immutable saved metrics:

| Measure | Original objective +5 epochs | Candidate +5 epochs |
|---|---:|---:|
| Exact-node precision in reference-present boxes | 9.01% | 20.31% |
| Accepted nodes near reference geometry, reference-present boxes | 25.85% | 48.18% |
| Reference curve within two voxels of accepted nodes | 27.20% | 17.94% |
| Same curve within two voxels of its reference target nodes | 55.11% | 55.11% |

There is appreciable disagreement between exact-node precision and on-curve
agreement. That supports investigating a representational mismatch. It does
not certify model quality: even the candidate has roughly half its accepted
nodes beyond two voxels of the annotated geometry in these boxes, and spatial
distribution/branch structure remain unresolved. These are reference agreement
fractions, not certified biological foreground precision.

The existing two-/six-voxel metrics and failed gates are preserved. A new
representation-invariant metric would be separately identified, not used to
retroactively change a failed experimental verdict.

## Corrections to the proposed diagnostics

1. **Input support is not a universal head-quality ceiling.** A learned model
   can infer a centerline between surrounding points or regress offsets larger
   than two voxels. Low two-voxel proximity is a warning, not proof that no head
   can succeed. For a bounded local voter with maximum displacement R, coverage
   within R can become a geometric reachability constraint. If offsets are not
   bounded, it must not be called a strict ceiling. Nearby dark points also
   contribute spatial proximity without necessarily retaining useful contrast.
2. **Resampled reference nodes need not score 100% under a point-set metric.**
   Eight-voxel samples can be four voxels from the midpoint between nodes on a
   straight branch. Reference curve-to-node support at a two-voxel tolerance is
   therefore inherently below 100%. A proper invariance fixture scores the
   same polyline with different subdivisions, retaining bends and topology,
   as a polyline. Sparse resampling that shortcuts bends changes geometry.
3. **On-curve precision alone is insufficient.** A model can concentrate nodes
   on a short correct fragment and get high point-to-curve agreement while
   missing the remainder. Report both directions, arclength weighting, spatial
   gaps, node density and structural incidence. Good geometry metrics still do
   not establish correct connectivity through crossings or nearby branches.
4. **Avoid duplicate infrastructure.** Exact segment distances, clipped curves,
   cached clouds, scores and existing curve diagnostics are already available.
   Assemble and validate these first; do not automatically spend 1–2 days
   implementing another approximate nearest-resampled-point metric.

The automatic Sato-threshold/skeleton/prune baseline is a reasonable separate
reference, not a PN1 rebuild or production adoption. Freeze its thresholds and
pruning convention using training material before development comparison.
It needs the same domain/coordinate/annotation treatment and a metric accepting
its edge representation. Cache availability does not establish its runtime.

## Dense support/offset voting: what is attractive

Supervising distance/support and displacement toward local reference curves
avoids arbitrary eight-voxel interior-node phases. It provides many local
supervision sites and makes geometric consolidation explicit. Point-based
voting has primary-method precedent, for example
[VoteNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Qi_Deep_Hough_Voting_for_3D_Object_Detection_in_Point_Clouds_ICCV_2019_paper.html).
That is motivation to test this neuronal task, not evidence that it will pass.

The proposed alternative would be:

```text
same automatic point clouds -> PointNeXt -> per-point support and curve offsets
    -> shifted-point consolidation -> candidate graph -> GATv2 -> decoding/SWC
```

This replaces the Transformer's current proposal responsibility; making it
optional or moving it after clustering is a substantive architecture departure.
The supplied review is not user approval to implement that replacement. Explain
and agree a bounded alternative-head experiment before coding it; do not retain
a decorative Transformer merely to claim architectural continuity.

## Dense voting still needs explicit contracts

- **Incomplete labels:** support approaching zero far from an annotated curve
  is not proof of biological background. Skip unknown crops and establish
  justified negative supervision/masking or explicitly restrict the reference
  task; dense labeling must not silently turn untraced branches into negatives.
- **Projection ambiguity:** nearest reference points may be nonunique at close
  branches, crossings/junctions and coincident fragmented paths. Crop clipping
  can also create artificial endpoint attraction. Specify ties, ambiguity masks,
  crop context and boundary behavior rather than assuming all offsets are unique.
- **Density choices remain:** sigma, offset radius R and clustering/NMS spacing
  r replace some learned-count behavior with geometric resolution parameters.
  They must be fixed and evaluated across branch spacing, not tuned per neuron.
  Ordinary Euclidean NMS/averaging can erase junctions or merge adjacent neurites.
- **Cross-crop consolidation remains:** clustering globally is a merging policy,
  not elimination of the problem. Prediction disagreement, missing boundary
  context, duplicate confidence, feature frames and nearby separate branches
  still require treatment. Current boxes are disjoint, not overlapping.
- **Graph labels remain conditional:** projection onto a reference tree requires
  reliable branch/component identity and an unambiguous path relation. Endpoint
  proximity or "consecutive" samples alone do not establish an edge at crossings.
  Preserve unknown relations where references are fragmented/ambiguous.

## Training strategy and literature

Slow convergence is documented for original DETR-family models, while
[Deformable DETR](https://arxiv.org/abs/2010.04159) and
[DN-DETR](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.pdf)
show approaches addressing it. Their detection schedules do not establish
how long this neuronal task needs, nor that Hungarian matching itself is the
sole problem. Our targets are already uniform in whole-chain arclength;
uniform resampling alone does not remove unknown phase outside a crop.
Deep supervision/denoising/one-to-many objectives remain separate changes,
not a list of fixes to add together.

A 4/16/64-box ladder is sensible if it expands across training neurons and
conditions, rather than many correlated boxes from one neuron. Predeclare
exposures/caps and judge curve fidelity/branch preservation, not just exact
interior-node matching. Do not select favorable boxes or tune on reserved data.
Repeatedly inspected development remains development.

## Recommended next decision

1. Assemble an existing-cache geometry/input-support report with reference
   subdivision/phase controls and precise node-versus-edge metric semantics.
   Preserve all unknown-output accounting and report reference-present views.
2. Determine how much error is tangential phase/spacing versus off-curve
   localization, absent branches or harmful rejection. Examine actual image
   evidence and matching domains; do not infer causality from one scalar.
3. If a dense local head is justified, propose it as an explicitly different
   proposal formulation with its own uncertainty/negative-label contract,
   simple baseline, small-data ladder and stopping criteria. Obtain agreement
   on the Transformer-role departure before implementing it.

No additional long fit, phase-one rebuild, target rewrite or dense-head adoption
is implemented by this review. The architecture and completed evidence remain
unchanged; only the interpretation of spacing-dependent diagnostics is clarified.
