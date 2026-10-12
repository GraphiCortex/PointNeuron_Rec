# Draft for review: field-guided Phase 1 image-point sampling

12 October 2026. The user agreed to design this comparison after the
[eight-crop field learnability pass](centerline_duration_results.md).
**This is a proposed interface and experiment, not implemented sampling code
or authorization to launch a new training run.**

The intended system stays:

```text
3D image -> Phase 1 image point cloud -> PointNeXt
         -> Transformer skeleton proposals -> GATv2 connectivity -> SWC
```

The field guides which image voxels become input points. It does not replace
PointNeXt, output the final Transformer proposals, estimate biological radius
or decide final graph connectivity. Fixed skeleton extraction remains a
diagnostic; it is not needed by the first proposed sampling comparison.

## Questions and scope

Separate three questions:

1. Does field guidance make useful dim locations available beyond the old
   intensity threshold, or just add unsupported locations?
2. On the same available points and unique-point budget, does guided selection
   preserve reference branches better than random or spatial-only selection?
3. Do any benefits hold on development neurons not used to train the field?

The frozen eight-crop checkpoint can answer an interface/selection feasibility
question on its own training examples. It cannot answer the third question.
A separate broader field train/development-validation pilot must be agreed
before claiming generalization. No encoder, decoder or graph retraining is
bundled with this interface comparison.

## Proposed inputs and output contract

Inputs to the selector: native crop image, valid-voxel mask, crop origin,
image scaling/threshold metadata, full-precision predicted field, sampling
settings and deterministic seed. **No reference SWC, target/distance field,
known-label mask, uncertainty mask or reference node count enters selection.**
Those label-derived fields belong only to subsequent measurement. The existing
bright-uncertainty mask depends on the reference and must not filter inputs.

Proposed output is distinct image-voxel points plus provenance:

| Item | Meaning |
| --- | --- |
| `points[N,4]` | Global native XYZ voxel centers and observed raw intensity |
| `voxel_indices[N]` | Unique flat C-order ZYX indices into the native crop |
| `field_scores[N]` | Predicted scores retained as metadata, not a new encoder channel in this comparison |
| `selection_role[N]` | Guide or context, where applicable; all points remain ordinary input points |
| Metadata | Origin, scale, shapes, actual unique count, candidate counts, settings, seeds and hashes |

Coordinates use `local_xyz + origin_xyz`; intensity comes from the same image
voxel, not the field score. Source image values and acquisition dtype must be
preserved or reconstructed with a verified tolerance against the decoded image.
Field values are scores, not calibrated probabilities of biological identity.

The existing DGCNN consumes `[x,y,z,raw_intensity]` and applies its own intensity
normalization, whereas the field cache stores normalized intensities. Feeding
those 0-1 values directly into that encoder would normalize them again. The
interface must distinguish raw intensities from normalized field inputs.
PointNeXt's eventual feature normalization must be defined separately; this
design does not change the existing DGCNN or add score channels to any encoder.

No repeated coordinates will be used to masquerade as additional unique points.
Small pools return their available points and a count/shortfall. A future batch
adapter may use explicit padding masks; support for such masks is not currently
implemented and must not be assumed. This first diagnostic does not feed a
padded array to a neural encoder.

## Candidate construction: explicit field-assisted expansion

Use the same native crop and the saved original intensity cutoff in every arm.
Recover its exact raw-intensity rule before implementation; do not tune a new
threshold based on references.

```text
F = valid voxels above the original image-intensity cutoff
P = valid voxels with predicted field score >= 0.5
C = unique union of F and P
```

`P` can include dim or zero-intensity centers supported by surrounding image
context in the learned field. This is predicted support, not proof of a real
neurite; dim additions, zero-intensity additions and off-reference support must
be reported separately. Retain their actual intensities without painting
signals into the image. The selector never queries the reference to admit them.

This **changes candidate eligibility**, not only ranking. That change is
declared and tested separately. The old threshold-only pool remains a control.
Keep all eligible candidates until selection; do not reintroduce an upstream
1,024-point random cap that could discard a thin branch before the guide acts.

Primary selection must use freshly reproduced float32 checkpoint predictions
or a hashed full-precision field cache. Existing float16 display snapshots are
not primary selector inputs: rounding near 0.5 can change candidate membership.

## Four comparison arms

| Arm | Candidate pool | Selection | What its comparison tests |
| --- | --- | --- | --- |
| A: threshold reference | F | Uniform without replacement | Fresh threshold-only reference policy |
| B: expanded random | C | Uniform without replacement | Candidate expansion versus A at matched counts |
| C: expanded coverage | C | Spatial coverage with no score ranking | Coverage versus random on the same pool |
| D: guided plus context | C | Half field-guide coverage, half contextual coverage | Guided allocation versus C on the same pool |

Arm A reconstructs the threshold/uniform policy from the native crop. It is
not a byte-identical replay of the historical 1,024-to-512 cached input path,
which can sample with replacement. Historical saved clouds may be inspected
as a separate reference, with their duplicate counts reported.

Proposed constants for this first comparison: field threshold **0.5**, spatial
cell size **8 native voxels**, guide fraction **0.5**, budgets **512 and 2,048**,
sampling seeds **0-4**. These are initial hypotheses for review, not measured
optima. No fraction, cell-size or threshold sweep is bundled. The 512 budget
links to historical experiments; 2,048 is the larger diagnostic budget. Neither
defines the final software's input capacity.

### Explicit coverage primitive

Partition the eligible pool into occupied 8-voxel cells anchored at the crop
origin. Use native XYZ geometry and unit voxel spacing. Start from a seeded
cell, then order occupied cell centers greedily by distance from the already
ordered centers. Walk that order in rounds, taking one unselected point per
nonempty cell per round until the requested count is filled. A seeded uniform
order chooses points within each cell for the coverage-only control; the guide
uses descending predicted score with seeded tie-breaking.

If the point budget is smaller than the number of cells, some cells will be
omitted. Spatial coverage is therefore a safeguard to test, not a guarantee
that every biological branch or image component receives points. It does not
know reference branch identities and does not require connected-component
seeding or another Kimimaro run.

### Guided arm D

1. Allocate `g = min(floor(n/2), len(P))` points from P using guide coverage.
2. Allocate the remaining `n-g` points using score-independent coverage from
   `C minus selected guide points`, preserving image context and exploration.
3. Deduplicate by voxel identity and require exactly n distinct points when
   n candidates are available. Preserve both guide and context roles in metadata.
4. If P is empty, fall back to arm C with the same seed; do not invent centers.

A small P shifts unused guide quota to context. This is an explicit fallback,
not a different confidence threshold. Selecting a voxel does not convert it
into a final skeleton node or force an edge between selected points.

## Budget fairness and candidate shortages

For each nominal budget K, report two comparisons:

- **All-arm matched-count analysis:** `n = min(K, len(F))`. Every arm receives
  exactly n distinct points; C is a superset of F. This separates pool/selection
  effects without granting the guided arm more unique input points.
- **Expanded-pool allocation analysis:** B/C/D each receive
  `n = min(K, len(C))`. This is the primary comparison of D against C at the
  larger budget. A is also reported at its naturally available count, but an
  A-versus-D comparison with different counts is labeled an availability result,
  not an exactly matched-count selector improvement.

An empty F is a threshold-pool failure; it must be retained in availability
reporting, not filtered out using the reference. An empty matched comparison
is explicitly marked unavailable. Empty/short pools are never filled with
duplicates to conceal the failure. Report uncapped F/C coverage as the
candidate-availability ceiling, separately from sampled performance.

### Read-only availability check during design

The eight field-cache crops have the following distinct above-threshold voxel
counts. Guide/union counts below are **approximate**, read from existing float16
display snapshots only to assess availability; no points were selected and no
sampling performance was measured.

| Crop | Above-threshold F | Predicted P, approximate | Union C, approximate |
| --- | ---: | ---: | ---: |
| 0/1 | 385 | 861 | 1,013 |
| 0/2 | 613 | 9,815 | 10,092 |
| 32/0 | 1,267 | 1,395 | 2,158 |
| 32/3 | 3,936 | 4,119 | 6,428 |
| 96/0 | 487 | 5,375 | 5,560 |
| 96/3 | 5,507 | 8,535 | 11,900 |
| 162/1 | 2,931 | 6,423 | 7,936 |
| 162/2 | 603 | 979 | 1,256 |

Availability uses the stored threshold divided by the saved image scale on
normalized cache values. Exact raw cutoff equivalence will be checked in the
implementation. Some threshold-only pools cannot supply even 512 distinct
points. The saved field counts also show why 2,048 is a ceiling, not a promise
of 2,048 unique points in every crop.

## Measurements, performed only after all input clouds are frozen

Measure candidate availability and selected points using the same evaluator:

- Reference-curve coverage at 1/2/4/6 voxels using fixed <=1-voxel arc probes
  and length weighting; point distances, not distances to invented edges.
- **Primary allocation endpoint:** coverage within 2 voxels, D versus C on
  expanded C at nominal budget 2,048, with actual counts identical within each pair.
- Per-reference-chain coverage, worst-chain coverage and longest unsupported
  run at the same tolerances. Boundary-clipped chains and short chains are
  identified explicitly; point coverage does not prove continuity or topology.
- Selected-point distance to continuous reference segments; near-reference
  proximity, far support, selected intensity distribution and zero-intensity
  counts, including guide/context subsets.
- Unique candidate/selected counts, duplicate counts, occupied-cell counts,
  below-original-threshold additions, field-fallback cases, runtime and memory.
- The same diagnostic at 512 and uncapped candidate ceilings, with all five
  sampling seeds and per-crop deltas rather than only the best example.

Reference proximity is annotation agreement, not biological precision.
Legitimate image context around a thick neurite may be farther from a tracing,
and incomplete reference annotations can omit real structures. Report this
tradeoff, inspect image evidence and do not silently use labels to suppress it.
Reference-empty crops are not verified all-background negatives. Retain them
in runtime/availability reports with undefined reference-coverage metrics.

On the eight fitted crops, the report establishes interface behavior only.
For a later development pilot, average annotated crops within each neuron
and neurons equally, retain per-neuron and worst-chain results, and freeze
numeric advancement rules before launching that pilot. Better mean coverage
with substantial branch losses or more off-reference support is not an
automatic win. No point-cloud gate is retroactively borrowed from the field gate.

## Execution sequence and explicit decision boundaries

1. **Now:** review this design and its candidate-expansion and 50/50 allocation
   assumptions. No sampler code is implemented during drafting.
2. If agreed, implement an isolated selector/comparison entry script. Verify
   XYZ/ZYX origins, intensity restoration, reference-free inputs, uniqueness,
   shortage/fallback rules, reproducible seeds and unchanged saved checkpoints.
   Reproduce the frozen field predictions, then prepare a visible bounded
   eight-training-crop diagnostic. Estimate runtime before its launch.
3. Review all four arms and per-chain losses. An in-sample benefit is not a
   generalization or architecture claim. Do not automatically tune the fractions
   or convert a sampled cloud into a final skeleton.
4. Agree a separate field-training and cloud-evaluation pilot on the fixed
   **33 training / 7 development-validation neurons**, keeping **8 reserved test
   neurons untouched**. Use image-defined crop availability without selecting
   only reference-positive examples; exclude unknown crops from unsupported
   negative supervision. Dataset coverage, field training settings, stopping
   rules and actual duration require a separate protocol and timing check.
5. Only after input behavior is assessed, review PointNeXt implementation and
   its Transformer adapter. Then assess GATv2 connectivity. Each architectural
   change receives its own comparison; none is silently replaced by the field.

The proposed initial selector is discrete, and the field would be frozen in the
input comparison. This does not implement joint end-to-end training. A later
differentiable sampler or joint field/encoder strategy is a separate design
decision, rather than an implicit consequence of this interface prototype.

## Source contracts inspected

- [Field framework and actual module status](centerline_framework.md)
- [Completed field-fit evidence and reproducibility caveat](centerline_duration_results.md)
- `src/pointneuron/data/training_cache.py`: intensity-bearing points and replacement/spatial sampling rules.
- `src/pointneuron/data/query_cache.py`: versioned candidate cache and partition checks.
- `src/pointneuron/models/dgcnn.py`: four-channel input and existing normalization.
- `configs/pointneuron2/sampling_training.json`: historical 1,024 candidates / 512 inputs.
- `configs/pointneuron2/partition_seed0.json`: fixed neuron partitions.

No source sampler, model, active config, checkpoint, cache or environment was
changed, and no new training, model inference or point selection ran during
this design step. The availability check read the eight existing training
records and their saved display fields only; development/test files were not opened.
