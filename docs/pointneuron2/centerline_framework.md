# Current PN 2.0 experiment: learn a centerline field from the 3D image

12 October 2026. This explains the current prototype, not a completed commercial
architecture. [The query pilot](query_pilot_results.md) did not produce usable
filtered proposals on development validation. The user agreed to test an
image-native centerline field instead. Historical point-cloud implementations
and results remain available; they are not silently replaced by a proven winner.

## What changed

Earlier learned stages took a limited set of sampled image points, encoded
their features and predicted skeleton proposals with confidence scores. The
query variant could place points away from the sampled input, but its pilot
confidence filtering still rejected all annotated validation crops at 0.5.

The current model takes a full-resolution **3D image crop**, retaining its
spatial grid and intensity values. A compact 3D U-Net predicts one probability
per voxel: does this location belong to a narrow tube around the neurite's
centerline? It can use neighboring image structure without first reducing
the crop to 512 sampled points. No PointNext or point-cloud encoder is used
in this current learning stage.

```text
3D image crop
    -> learned centerline probability field       [implemented and being fitted]
    -> fixed threshold and skeleton extraction   [checked on two selected crops]
    -> points connected into center paths        [saved diagnostic geometry]
    -> merge crops and reconstruct full graph    [integration/evaluation pending]
    -> final neuron SWC                          [not yet validated for this model]
```

The center paths contain XYZ points and edges. They can supply downstream graph
reconstruction, so point geometry remains useful. We are currently testing a
voxel-based learned front end rather than an entirely point-cloud-based model.
A future learned point or graph refinement stage is an option, not an existing
component of this experiment.

## What the model learns

During training, an existing reference SWC provides the supervision. Continuous
reference segments are surrounded by a **two-voxel-radius tube**; the model is
trained to reproduce that tube. This radius is a target construction, not a
measurement of the neurite's biological radius or a full neurite segmentation.
Voxel distances currently use native image coordinates, not physical units.

The loss encourages reference-tube scores, penalizes the hardest known
background errors and rewards overlap. Padding and bright distant regions
marked uncertain are excluded from direct supervision. This weak-label rule
helps avoid labeling every untraced bright structure as a confirmed negative,
but it cannot guarantee that all remaining negative annotations are correct.

**At inference the input is the image alone.** No reference SWC or manual mask
would be supplied to the model. Threshold 0.5 and the fixed CPU skeletonizer
convert the field into center paths. The ground-truth-derived oracle is only
a diagnostic: its successful extraction does not count as learned performance.

## Why this is worth testing, and what it leaves unresolved

The hypothesis is that local image context and dense spatial supervision give
us more reliable centerline support than selecting and scoring isolated point
proposals. Two-crop extraction and oracle checks now separate field errors from
extractor feasibility. They do not prove the image-to-reference mapping,
generalization or correct graph connectivity.

Voxel processing costs more memory than sparse points; overlapping crop
inference and merging still need development. A binary tube can also merge
nearby branches and lose their identities. Good center placement alone does
not establish correct parentage, radius, topology or commercial readiness.

The immediate question is deliberately smaller: **can the present model fit
the same eight selected training crops well enough?** The 400-update
hard-negative fit improved reference agreement but failed the frozen gate.
The [approved duration-only check](centerline_duration_ready.md) tests 1,600
updates before any broader validation or new architecture decision.
