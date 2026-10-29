# Four-box image evidence and PointNeXt access check

29 October 2026. Fixed before scoring. Implements a bounded first pass of the
[branch evidence protocol](branch_retention_protocol.md), after the successful
64k execution replay. No new clouds, guides, trained features, losses or fits.

## Fixed scope and comparison

Two already-inspected training sources: fruitfly 32 and frog 22. Two boxes each,
selected by the median positive and maximum reference arclength under the existing
0.5-voxel quadrature. Selection did not examine selector scores. Both references
have one encoded component; all reference nodes and opaque type tags are retained.
No development or reserved source is read. This is not a representative-domain
sample, whole-neuron result or independent test.

| Source | Box | Native XYZ bounds, stop exclusive | Dense distinct points |
|---|---:|---|---:|
| Fruitfly 32, moderate reference length | 27 | (384,384,0) -> (512,429,27) | 2,430 |
| Fruitfly 32, largest reference length | 20 | (512,256,0) -> (640,384,27) | 6,912 |
| Frog 22, moderate reference length | 2 | (256,0,0) -> (384,128,81) | 20,736 |
| Frog 22, largest reference length | 5 | (128,128,0) -> (256,256,81) | 20,736 |

The frozen dense clouds of uniform, raw-intensity and Sato sampling are compared
at identical within-box counts, for all five original seeds. There is no padding
to 32k/64k. File hashes and exact source XYZ, intensities and quotas are checked.
References choose evaluation windows only; they never select or relocate inputs.
Configuration: `configs/real_branch_evidence.json`.

## Measurement definitions

Reference paths receive uniformly spaced arc midpoints, maximum step four native
voxels. Each point defines a tangent-aligned cylinder with axial half-length two.
Report windows at radial distances 1, 2, 4 and 8 voxels, using unique source voxel
centers. Background is the median available image intensity in the same slab at
radial distances 8..12 voxels. Record its median absolute deviation. No background
voxel is assigned a supervised class by this calculation.

Within each input box, compare available and selected positive contrast mass,
sum(max(I-background,0)). All selectors use the same background/window. A zero
denominator is undefined, not perfect retention. Source/box clipping is recorded;
voxels beyond a box are not charged as a sampling failure inside that box.
Background/width profiles may use available source voxels beyond the box.
Aggregate **arc-length-weighted local retention fractions**, not a sum falsely
claimed to be unique whole-neuron intensity mass: local windows can overlap.

Half-voxel radial bins give a background-subtracted mean image profile. Measure
the first half-height crossing after the inner (r<2) profile peak, with linear
interpolation. Profiles with nonpositive peak, absent crossing or a stronger
outer peak have unresolved widths. Preserve raw profiles and flags for inspection.
This is an apparent centered-profile width, not anatomical diameter or SWC radius.
Voxel spacing/PSF provenance is unverified; shell contamination, noise, reference
offset and neighboring structures can affect these measurements.

Exploratory width strata are <=3 native voxels, 3..5, >=5 and unresolved.
Contrast strata are positive-profile contrast terciles within each source across
its fixed boxes, computed **before cloud scoring**; retain tied/nonpositive cases.
These bins organize the audit, not model thresholds or universal definitions of
dimness/caliber. Synthetic Gaussian tubes with known widths/contrasts qualify
the estimator to within one voxel in the tested narrow/broad cases. Clipping,
constant images and retained bright evidence outside radius two are also tested.

Positive mass includes positive noise and potentially unrelated structures.
Untraced image structure is not certified background. Keep spatial support,
contrast evidence, apparent profile width and reconstruction accuracy distinct.

## Actual geometric access, seed zero only

Reorder each frozen cloud by a fixed source-ID hash, independent of sampling
rank and method. Use actual clipped-box center and scalar coordinate scale 64.
Run the pinned native CPU FPS and strict first-32 ball operators, without weights
or optimizer: four stages, quarter counts, radii 6.4/12.8/25.6/51.2 voxels.

Save actual source identities for centroids and ball members. Report distinct
neighbors, padding repeats and truncation at K for each stage. For the first
stage, measure image contrast mass accessed by the union of actual selected
ball members, both relative to full source-window mass and conditional on input
mass. This is access through the abstraction path. The fine-level feature skip
still exposes every input point; absence from this union is **not total loss
from encoder output**. Later stages operate on aggregated features, so their
centroid count is not itself an information-retention metric.

One ordering control uses frog box 5, Sato seed zero: hold the same first-stage
centroid positions fixed and compare actual first-32 members under original
sampling-race ordering against hash ordering. It isolates grouping order without
moving FPS queries. No new input points or neighborhoods outside the radius.

This first pass does not evaluate cross-path mixing as a biological wiring
error, train a feature decoder or claim to measure learned representation quality.
Those questions cannot be settled by untrained grouping statistics alone.

## Execution and stopping

Run the nine small measurement tests before scoring. Then perform only the four
declared boxes: 60 matched input arms, 12 seed-zero hierarchy traces and one
same-query ordering control. CPU only; no filtering, resampling or training.
Per-case JSON and trace arrays are saved in a new ignored `tmp` directory. Any
source/geometry mismatch stops the audit; do not repair coordinates or substitute
cases. No method-quality acceptance threshold is fit on these results.

Inspect thin/dim and unresolved strata, clipping and profiles before interpreting
differences. A geometry/access deficit can motivate the later matched trained
comparison; it does not by itself justify redesigning Phase 1 or choosing a
product point count. The next learned milestone remains one specified few-box
PointNeXt-to-Transformer learnability task, followed by a matched selection/budget
comparison if the task is learnable. Do not add another loss to this audit.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe `
  .\scripts\audit_real_branch_evidence.py --output .\tmp\real_branch_evidence_20261029
```
