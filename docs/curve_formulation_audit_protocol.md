# Curve formulation audit: fixed CPU protocol

3 November 2026. The user authorized the formulation audit after reviewing the
[curve/voting proposal](formulation_review_20261103.md). This protocol measures
existing artifacts and a classical reference. It does not replace the
Transformer, train a model, regenerate guides or change targets/partitions.

## Frozen scope

- All 40 cached images, 33 training / seven repeatedly exposed development.
- All 611 reference-present boxes; unknown-region model outputs remain disclosed.
- Both nested 2048/8192 input clouds; source indices/intensities are unchanged.
- Four saved model states: fresh epoch-20 2048 and 8192, plus the two epoch-25
  objective continuations. Training/development prediction packets are hash-checked.
- Native voxel units and the existing clipped 128-voxel domains, with no alignment.
- Exact point-to-segment distances and midpoint arclength quadrature at 0.5 voxels.
- Fixed training-defined apparent thin/dim cutoffs; unresolved profiles retained.

Configuration: `configs/curve_formulation_audit.json`. New metric and baseline
source hashes, original data/profile manifest hashes, saved packet hashes and
CPU package versions are recorded and checked. No reserved image is loaded.

## Diagnostics and interpretation

### Input evidence

Measure reference-curve length within 2, 6 and 8 voxels of selected input points,
per neuron, source context, budget and apparent thin/dim stratum. Macro-average
only after pooling within neurons. The 6/8 radii are prospective local-voting
reach diagnostics, not an adopted new inference parameter.

Two-voxel input proximity is **not** a mathematical ceiling on a learned head.
A model can infer medial positions or use offsets/context. Even R-radius input
support is a strict reachability bound only for a model actually bounded to R.
Nearby dark points can count toward proximity; per-box nearest-sample intensity
is a supplementary observation, not a certified contrast-retention metric.

### Saved prediction decomposition

For all 512 proposals and separately those accepted at the unchanged .5 cutoff:

- Distance to reference segments, including isolated reference points.
- Nearest-reference-grid distance and original one-to-one node matches.
- Fraction on reference geometry but beyond two voxels of any target node.
- Fraction near the grid but not uniquely matched, inferable from those counts.
- Reference-curve length near node centers and proposals outside the actual box.

On-curve positions beyond the grid are possible representation disagreement,
not proof of correct topology or useful branch distribution. The point support
metric remains node-spacing dependent and is not renamed curve reconstruction.
Unknown-box outputs are disclosed rather than treated as certified background.

### Reference controls

For one reference-present box per neuron, insert phase-shifted eight-voxel
samples while retaining every original bend and endpoint. Segment agreement
must remain perfect within 0.001 voxels; point/grid scores may differ.

For every reference-present box, evaluate the actual cached target-node graph
using its **reference-supplied edges** against the original clipped curve.
This is a reference-connectivity oracle, never a predicted reconstruction. It
tests the geometric adequacy of the target discretization and distinguishes
that from its target-node support at a two-voxel tolerance.

Synthetic controls additionally verify sparse-node versus edge scores, phase
changes, preserved bends, empty/isolated geometry, clipping at half-voxel box
boundaries, and crossing-versus-junction ambiguity under geometry-only metrics.

## Automatic classical baseline

Five previously examined training contexts (aliases 22/32/75/85/126) and all
seven development images are fixed before baseline results: 12 images, including
all 415 development boxes. Training coverage is a context panel, not the whole
33-neuron training cohort. A separate timing case is training Image5 / alias 41.

Image-only extraction reads the retained full-resolution Sato guide:

1. Global Otsu threshold on a 256-bin histogram including zero responses.
2. Three-dimensional Lee skeletonization on the full image mask.
3. An undirected 26-neighbor voxel graph, retaining cycles and coincidences.
4. One pass removing only leaf-to-junction spurs strictly shorter than two
   native voxels. Never delete a standalone component on that length rule.

Both unpruned and pruned graphs are retained and scored. No reference is read
until extraction is complete. There is no reference-driven component choice,
rooting, node quota, threshold tuning, welding, gap filling or learned inference.
All parameters are fixed; reference metrics do not select the better variant.

The automatic graph is clipped to the same box domains as the cached references
for scoring, with crossing edges clipped at the actual half-voxel planes.
Scores integrate both predicted and reference arclength against continuous
segments. Full-grid reference-agreement precision and reference-present-only
precision are both reported, along with unknown-region predicted length.
These are tiled geometric measurements, not unique whole-neuron topology scores.

Mechanical voxel adjacency can contain diagonal triangles, false joins and
incorrect biological connections; geometry cannot certify them. Classical
skeleton vertices have no matched 512-node quota. Its segment F1 must not be
compared directly with learned node-match F1 as if representations were equal.
The learned models currently supply no predicted edges; the audit does not
invent edges for them from the reference.

## Execution and outputs

Use the pinned Phase 1 CPU runtime, with no package installation:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\tmp\phase1_cpu_env\Scripts\python.exe -B .\scripts\audit_curve_formulation.py `
  --action audit --output .\tmp\curve_formulation_audit_20261103
```

The audit worker has a 30-minute cap and a flushed per-box journal. Classical
timing uses one image first; the declared baseline has 15-minute per-image and
60-minute total caps. Failure preserves partial evidence and does not retry or
silently omit a difficult case. Existing output directories are never reused.

Original-image orthogonal views are generated by
`scripts/plot_curve_formulation.py` for the four fixed historical fresh training
boxes (fruitfly32 27/20; frog22 2/5). It re-decodes the two original images with
source/decoded hashes checked, uses unchanged whole-image affine contrast,
and draws saved proposals only. Green lines are reference geometry for
evaluation, **not predicted edges**. Fixed crop views disclose outside-box counts.

Generated arrays, graph packets, figures, journals and reports belong under
ignored PN2/tmp. Readable results and selected reviewed figures will be linked
from a separate results report. Tests/metric validation are not a learning pass.

## Decision this audit can support

Decide whether to prioritize better input evidence, a locally identifiable
proposal task, encoder preservation, or confidence/optimization behavior. It
cannot establish that dense voting learns, that a particular offset radius is
universal, or that the intended architecture is wrong. Any alternate proposal
head, changed Transformer role, dense background-label policy or training ladder
will be explained and agreed separately after reviewing this evidence.

Execution is now complete. See the [results and reviewed figures](curve_formulation_audit_results.md).
