# Common whole-neuron benchmark contract

21 October 2026. Initial milestone A contract for the
[revised proposal](revised_proposal.md). It fixes the intended input/output and
failure accounting for a shared evaluator. Structural correspondence and
quantitative advancement margins remain to be agreed before the decisive R/N
comparison. The new evaluator is a research tool, not a certified external
neuronal-reconstruction benchmark.

## Question and evaluation population

Does a fresh PN2 candidate N reconstruct new whole neurons more accurately
than a freshly trained, repaired PN1-style reference R under disclosed resource
budgets? [H, R and N](baseline_reference.md) have distinct roles. H supplies
historical context; its training exposure prevents calling it an unseen-data
comparator on the present development/reserved partition.

The first explicit [development inventory](../configs/development_benchmark.json)
contains exactly seven existing development identities: historical aliases
19, 21, 35, 44, 79, 94 and 135. Cases are identified by stable `sample_id`
strings, not by positions in a freshly scanned list. This is repeatedly used
development data. It supports diagnosis and model selection, not an independent
product claim. A selected crop, trace bounding box or capped crop cache does
not substitute for the declared full image volume.

The inventory was made by joining existing partition and source-manifest
metadata by sample ID. The source inventory has 163 records; its separately
filtered clean version has 155, so numerical row positions cannot be reused
interchangeably. Manifest source hashes and historical exposure are recorded.
Only existence checks on the seven listed image/reference paths were needed
for this inventory. No dataset-wide scanner or reserved annotation was opened.
File existence does not certify decoded image correctness, annotation quality
or calibration. Source image/SWC content hashes are still to be collected for
approved cases before a decisive run.

## Explicit manifest interfaces

Benchmark JSON uses `schema_version: 1`, a `dataset` description and a `cases`
list. Each case has:

- Unique string `case_id`, partition, explicit `reference_swc` and `image_path`.
- Explicit `coordinates`, `annotation_status` and optional source/provenance
  metadata. A source folder is a descriptive stratum, not proof of an
  independent acquisition group.
- Paths resolved relative to that manifest's folder. The evaluator opens only
  listed references/outputs. It must not discover or select replacement SWCs
  by walking the dataset.

Model output JSON uses `schema_version: 1`, string `model_id` and `outputs`:

```json
{
  "schema_version": 1,
  "model_id": "R_example",
  "outputs": [
    {
      "case_id": "exact-case-id-from-benchmark",
      "status": "completed",
      "swc_path": "relative/path/to/reconstruction.swc",
      "runtime_seconds": 12.5,
      "peak_memory_bytes": 100000000
    }
  ]
}
```

The numbers are schema examples, not measured runtime or resource targets.
Permitted reported statuses are `completed`, `missing`, `empty`, `invalid`,
`timeout`, `unsupported` and `failed`. Missing rows are counted as missing
outputs. Duplicate or extra unknown case IDs are manifest errors; they must
not silently replace declared cases. An observed invalid/empty file overrides
a caller's assertion that it completed successfully.

SWC validation checks identifiers, parent references, finite coordinates and
radii, valid radius signs and forest structure. Multiple roots are allowed.
Do not force a fragmented reference/prediction into one tree to satisfy
serialization. Output root/type conventions are recorded separately from
anatomical correctness.

## Coordinates, distances and radii

Both SWCs for one case must use the same declared XYZ coordinate convention
and frame, with crop origins already restored. No alignment fitted to the
reference, reference-dependent rescaling or silent translation is permitted.

- `{"mode":"native","units":"voxel"}` evaluates Euclidean distances in
  native voxel coordinates. The initial development inventory uses this mode.
- `{"mode":"spacing","units":"um","spacing_xyz":[sx,sy,sz]}` converts
  native voxel XYZ using a verified positive spacing vector before distances
  and arclength are measured. It is appropriate only when acquisition metadata
  and SWC coordinate conventions support that transformation.

The inventory lists pixel-size metadata for cases 35, 79 and 94, but its
contents/conventions have not been verified here. Their presence is not a
physical calibration claim. Missing calibration stays unknown rather than
being invented as one micrometer per voxel.

For the current native-voxel cohort, report tolerances 2 and 6 voxels, with
six-voxel F1 the initial primary geometric comparison for historical context.
These are not universal physical-resolution targets. A calibrated external
cohort requires its own prospectively declared physical tolerances; do not
pool mismatched voxel/physical units into one headline. An anisotropic XYZ
transform also does not specify a scalar SWC radius conversion; radius claims
require a separate convention and evaluator.

## Continuous geometry and meaning of scores

Treat SWC parent-child edges as continuous line segments. Let predicted and
reference total arclengths be P and G. At tolerance t:

- Precision is predicted arclength within t of the reference curve, divided
  by P; recall is reference arclength within t of the prediction, divided by G.
- F1 is their harmonic mean. Absolute extra length is the predicted length
  outside tolerance; absolute missed length is the reference length outside
tolerance. The initial implementation reports directed arclength-weighted mean
distance and a maximum over quadrature samples; full distance distributions
remain a scorecard extension.

Distance is to continuous segments, not only stored vertices. Arclength
weighting prevents densely sampled parts of an SWC from dominating because
of their vertex count. Numerical integration must report its sampling/error
policy, demonstrate convergence and remain stable when an identical curve is
subdivided into additional degree-two vertices. The initial midpoint
quadrature is an approximation, not exact analytical integration.

Isolated nodes have zero curve length. A nonempty valid SWC with no positive
curve length is a `segmentless` reconstruction and receives zero curve F1
against a positive-length reference. Isolated-node counts are retained as
separate diagnostics, rather than giving a point the support of an arbor.

The present annotation status is `reference_agreement`: these metrics measure
agreement with the available trace, not verified biological foreground
precision. Real unlabeled branches can count as extra reference-disagreeing
length. `incomplete` annotations retain an explicit limitation if scored;
`unknown` annotation status is unscorable. Do not fix such ambiguity by
declaring every unmatched image structure to be true background.

## Failures and denominators

Every declared case has a result row. Against a valid positive-length
reference, missing, empty, segmentless, invalid, timed-out, unsupported and
failed reconstructions receive zero reconstruction F1 and their own failure
status. Record precision/recall conventions explicitly, including the empty-
prediction case. Failures must affect the macro reconstruction result; a mean
over successful jobs alone is insufficient.

A malformed, unknown or zero-positive-length reference cannot supply ordinary
curve precision/recall. Retain that case with an unscorable-reference reason
and report declared/scorable/unscorable counts and reasons separately. Do not
silently remove it, treat it as certified negative background or blame the
model for reference invalidity. Eligibility changes before a comparison need
a new manifest version; discoveries during a run remain visible in that run.

## Structure is a separate assessment

Geometry can give a high score to the wrong connection between nearby
neurites. SWC validity or matching component counts cannot rule that out.
The initial evaluator may report endpoints, junctions, components and their
geometric support as diagnostics. These are not a verified topology score.

Before a structural advancement claim, specify and test correspondence for
nearby/ambiguous endpoints and junctions, reference branch continuity,
fragmentation, wrong merges and long shortcuts. Account for both missing and
extra structure, crop boundaries, close parallel branches and incomplete
annotations. A graph induced only on a model's surviving nodes cannot be the
sole target: omitted branches would disappear from its denominator. Avoid
arbitrary tie breaking that converts an ambiguous match into a claimed correct
connection. Do not label approximate local ESA/DSA/PDS as official plugin
equivalents.

## Comparison and advancement decision

Use the same explicit identities, image bytes, references, coordinate policy,
declared volume and evaluator for R and N. Final decisions are automatic:
references never determine accepted-node counts, roots, candidates or output
edges. A single development-selected operating policy may be frozen with its
selection history disclosed; no per-neuron cutoffs or oracle top-k.

Publish per-neuron paired results, macro mean, median, worst cases, absolute
error lengths, failure counts and descriptive source strata. Report entire
job runtime, peak memory and actual crop/point/proposal/graph counts. Seven
neurons are seven observations, not the number of crops or curve samples.
Show learning history and training/selection resources as well as inference
resources.

No quantitative advancement margin has been invented here. Freeze meaningful
geometric gains, precision tradeoffs, structural limits and failure/resource
limits after reviewing the evaluator and R's behavior, before N's decisive
results. A change after seeing those results is a new prospective comparison,
not a pass for the earlier experiment. Successful execution is the first
engineering check; training learnability, development improvement, reserved
internal confirmation and independent external generalization remain distinct.
