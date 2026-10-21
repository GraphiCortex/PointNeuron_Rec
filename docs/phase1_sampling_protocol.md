# Phase 1 CPU probe: frozen comparison protocol

22 October 2026. This is an input representation diagnostic, not training,
reconstruction validation or a product default. The user agreed to include
uniform, raw-intensity-weighted and Sato-weighted controls. The settings below
are specified before reading reference contents or measuring the new clouds.

## Population and scope

Use exactly the two original training images in
[the image-only manifest](../configs/phase1_raw_review.json), aliases 32 and 46.
They were selected by smallest decoded uint8/uint16 volume, not outcomes. Cover
their complete native voxel domains; no trace-derived crops, masks, cached
clouds, old weights, development or reserved images. These two convenience
examples cannot establish broad applicability.

Image-only generation and reference evaluation are separate commands. Save
source voxel indices, native XYZ and unchanged intensities before reading SWCs.
The evaluator joins only these explicit case IDs to the training inventory.
It does not update the clouds. Hash input/config/output artifacts.

## Fixed construction

- Normalize each whole image affinely from its observed minimum/maximum to
  float32 [0,1], without clipping. Save the transform; original intensities stay
  in cloud packets. A constant image has a zero normalized field.
- Use the published scikit-image **0.25.2** Sato operator, bright-ridge polarity,
  `reflect` boundary mode, scales **1, 2, 4 native voxels**, max over scales.
  These logarithmically spaced scales probe small structures, not all possible
  radii. Unknown calibration remains unknown. No physical-isotropy claim.
- Filter each full image once, then partition into **disjoint 128-cube boxes**,
  clipped at image edges. There is no tile filtering seam, overlap duplication,
  or fabricated padding in the point cloud. Whole-image filtering is affordable
  for these two examples; it is not a scalable product implementation claim.
- Per tile with V voxels, request `ceil(V / d^3)` unique source points, for
  **d=4 and d=8**. These are denser/leaner representation probes, derived from
  voxel density rather than old model query counts. They are not spatial-grid
  sampling, a physical resolution specification, or final PointNeXt budgets.
- Uniform control: equal weights on every voxel. Intensity control: normalized
  raw intensity. Sato control: nonnegative guide response. In weighted arms use
  `p_i = 0.1/V + 0.9*w_i/sum(w)`. A zero-response tile explicitly uses uniform
  weights. The 10% component is an untuned context safeguard, **not a quota or
  guaranteed branch coverage**; it may be inadequate.
- Use exponential-race weighted sampling without replacement and seeds **0-4**.
  The same random numbers are paired across arms. Density outputs are nested
  prefixes of one ranking per tile/arm/seed. Every arm has identical counts.
  These weights describe the draw distribution, not marginal point inclusion
  probabilities after drawing without replacement.

The intensity arm is essential: beating uniform alone would not establish that
structural filtering adds value beyond brightness. Do not tune scales, context
mass, density or seeds after seeing outcomes in this probe.

## Execution checks before real-image generation

Check bright versus dark polarity, axis permutation, intensity-affine behavior,
scale response, a Y junction and boundary behavior on analytic 3D fixtures.
Check constant fields, exact unique counts, nested budgets, source XYZ/intensity
recovery and image-only manifest rejection of reference fields. A synthetic
fixture is an operator check, not neuronal accuracy evidence. If a check fails,
stop generation, explain the implementation issue and preserve diagnostics.

Use an isolated ignored CPU runtime with pinned requirements; do not repair the
old runtime in place. No GPU or model training is used.

## Measurements

Evaluate the full available SWC as continuous parent-child curves in its
declared native frame. Never fit an alignment or drop inconvenient trace
segments. Report reference samples outside the image extent separately. An
invalid/empty reference remains an explicit unscorable case. Degree-defined
branches are maximal paths between degree-not-two nodes, not certified
biological branches. Isolated/zero-length structures receive separate counts.

- Primary: arclength-weighted reference-curve support within **2 voxels** of a
  selected source point. Also show 6 voxels and nearest-point mean distance.
- Branch minimum and 10th-percentile coverage; longest unsupported arclength run
  within any branch; endpoint/junction nearest-point support. A high average
  must not hide local gaps. These are spatial-support proxies, **not graph
  correctness or a mathematical ceiling on learned reconstruction**.
- Midpoint quadrature step at most **0.5 voxel**. Re-evaluate primary support at
  0.25 voxel; report numerical differences. Longest gap is a sampled estimate,
  with sub-step boundary uncertainty, not exact gap geometry.
- Selected-point fraction within tolerance of continuous reference segments.
  Call this reference agreement, not biological foreground precision; untraced
  structures are not automatically background.
- Actual unique counts, per-tile zero-response fallbacks, per-seed scores,
  median/range, paired mean differences, decoding/filtering/selection/evaluation
  wall time and process peak working set. Five draws are not five independent
  neurons; do not derive population confidence from them.

## Decision and stopping rules

The only possible advancement is **replicate on a prospectively declared,
more diverse training subset**, not promote Phase 1 or launch a model fit.

Call Sato *promising for replication* only if:

1. The mean five-seed 2-voxel support gain at the lean density is at least
   **5 percentage points over each control in each image**. This is a practical
   complexity margin, not a universal anatomical threshold.
2. At the denser density it loses no more than **2 percentage points** of mean
   global support against either control in either image.
3. At both densities, each image's mean worst-branch and branch-tail coverage
   lose no more than **2 percentage points**, mean junction/endpoint support
   loses no more than **2 percentage points**, and mean maximum gap grows by
   no more than **2 voxels**, against either control. Undefined structural
   classes are explicit and cannot count as a passed test.
4. Numerical refinement changes primary support by at most **1 percentage
   point** for every saved output; all execution/source/count checks pass.

These margins make the decision reviewable; passing is not generalization.
If intensity already saturates lean support, Sato may have no useful margin:
that favors the simpler preparation here. If gains hide local losses, do not
declare success from the aggregate. Resource costs are reported for comparison,
not silently used to exclude an inconvenient case.

If the gate fails, stop at the report and interpret which hypothesis failed.
Do not add a loss, alter thresholds, swap seeds, expand the run or start training
automatically. A failure rejects this particular guide/scale/allocation setting,
not every structural guide or the full intended architecture. If the reference
frame or trace quality is questionable, report an inconclusive diagnostic and
resolve that evidence issue before using its scores to choose a method.

## Source

[Official Sato implementation](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/filters/ridges.py)
and [API documentation](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sato).
The method's documented properties justify testing this role; neuronal retention
benefits are an untested hypothesis.
