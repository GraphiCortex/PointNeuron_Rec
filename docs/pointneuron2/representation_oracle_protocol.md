# Training-only representation feasibility protocol

16 October 2026. The user agreed to start the complete-pipeline work after
reviewing the proposed schema. The first implementation is this bounded CPU
reference oracle. It does not reimplement Phase 1, alter the encoder/Transformer,
train GATv2, or promote a new production reconstruction path.

## Decision and scope

Can the existing spacing-eight target points, correct reference connectivity and
straight-line SWC export preserve known branching and reference geometry?

This is a necessary representation check. It does not measure automatic
reconstruction quality. Reference identities/order deliberately supply perfect
consolidation and connectivity; production matching, GATv2 edge prediction,
automatic graph initialization and uncertainty-aware decoding remain future work.

The intended automatic pipeline remains image point cloud -> PointNeXt ->
Transformer -> graph consolidation/candidates -> GATv2 -> graph decoding -> SWC.
The original PointNeuron paper motivates automatic graph initialization followed
by learned refinement. This diagnostic geometric candidate rule is not a choice
of final initializer or evidence that GATv2 can repair a tracer's errors.

## Fixed cases and provenance

Use only SWCs from three training-neuron directories. The saved training-target
audit, its SHA256, current neuron partition and each selected source SWC hash
must match. No dataset scanner, image reads, checkpoint loading, Torch, GPU,
optimizer updates, development/test reference reads or existing-cache writes.

| Case | Selection reason |
| --- | --- |
| Training neuron 92, crop 4 | Small branched case: 16 targets and two junctions |
| Training neuron 86, crop 3 | Maximum audited target count: 418, including 78 junctions |
| Training neuron 92, crops 1 and 2 | Neighbouring closed crop boxes with one shared reference crossing; ideal boundary joining |
| Training neuron 66, full reference | Most fragmented training reference: 200 roots |

These are structurally selected examples, not independent validation. Full
reference 66 bypasses image/crop admission. It is a forest-handling stress case,
not a claim that a biological single neuron has 200 disconnected components.

The initial run used crops 3 and 4 for the joining case. Although their boxes
touch, the reference did not cross their common face. That run's numerical gates
passed but did not exercise real joining. A training-only scan of existing crop
identities selected crops 1 and 2 (the first pair with a shared identity).
The corrected check explicitly requires shared identities and a reduction in
component count. Other cases, metrics and numerical thresholds are unchanged.
The initial run and exact source/config snapshots are preserved separately.

## Representation and candidate rules

- Reuse the existing continuous-curve target constructor: spacing eight native
  voxels, roots/leaves/junctions and crop intersections. Never truncate.
- Enforce the existing capacity 512 separately for each selected crop. Merging
  crops does not introduce a 512-node global cap. Do not apply the historical
  128-anchor budget; report it only as an illustration.
- Preserve distinct reference identities at identical XYZ. Consolidation uses
  known identities, not proximity. Rebuild adjacency along visible reference
  intervals so that curve excursions outside the crop cannot create shortcuts.
- Connect consecutive targets along each visible reference chain with straight
  chords. Preserve reference components. No image-geodesic tracing or synthetic
  bridge completion. Future inter-fragment labels may be unknown, not negative.
- Construct geometric candidate edges from XYZ alone: symmetrized 16-neighbour
  graph union all pairs within eight voxels. Report kNN-only recall separately.
  Spacing-eight consecutive targets should be within the radius by construction.
  The radius rule's recall is therefore a feasibility property, not evidence of
  learned topology or superiority. Report extra candidate counts as ambiguity.
- Export parent-before-child SWC, floor radius to one voxel as in the current
  query loss, reload and verify geometry/parents/components. Radius accuracy is
  not assessed here.

## Metrics, acceptance and stopping

Before interpreting results, synthetic checks must cover clipped crossing edges,
curve exit/reentry, junctions, coincident distinct components, crop joining,
missing kNN edges, curved-branch distortion, rewiring and empty/cyclic outputs.

For every case require:

1. All structural identities, branch incidence and component counts preserved.
2. All required oracle edges available in the union candidate graph.
3. Arclength precision and recall each at least 99% within two native voxels.
   Also report six-voxel values and maximum quadrature-sample distances.
4. Valid SWC reload with unchanged edges/components and coordinate error <=1e-6.
5. The real boundary-joining case actually contains a shared target identity and
   merges reference pieces that were separate in the per-crop graphs.

Geometry uses segment-length-weighted midpoint quadrature at <=0.5 voxel
intervals, with exact point-to-segment distances at sampled points. It is not
physical calibration or an official Vaa3D metric. Topological checks use graph
identity/incidence, separately from geometry. Empty geometry is not a zero-error
success. Invalid cases are failures, never excluded from the final gate.

Successful execution and a passing representation gate are separate. If any
representation criterion fails, keep the settings/results and stop progression
to learned connectivity. Discuss the first failing contract before changing
spacing, adding points, altering thresholds or extending model fits. A pass
would justify further integration design, not automatically authorize training.

Phase/reroot perturbations and imperfect/noisy proposals are not included in this
first bounded check. Existing synthetic subdivision checks and the oracle's new
geometry checks do not establish phase invariance on real references.

## Visible command and outputs

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\representation_oracle.py `
  --config .\configs\pointneuron2\representation_oracle.json `
  --data-root "C:\Users\Seywe\Documents\GitHub\PointNeuron\data\gold166" `
  --output-dir .\tmp\pointneuron2\local_runs\representation_oracle_verified_20261016
```

The runner refuses existing output directories. Generated SWCs, arrays, detailed
summaries and figures live under ignored `tmp/pointneuron2/local_runs`. Readable
results and selected figures are reviewed separately in this visible docs folder.
