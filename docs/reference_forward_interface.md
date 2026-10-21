# Reference clouds and read-only proposal execution

**Historical scope detour, closed 22 October 2026.** The user clarified PN1's
conceptual role. No singleton correction or further PN1 run is scheduled.
Associated source/config/tests are preserved at pushed commit `48519ff` and
removed from active PN2. The unsupported result below is unchanged. Active work
starts at [original raw images and PN2 Phase 1](pn2_phase1_design.md).

22 October 2026. Continuation of the [baseline integration plan](baseline_integration_plan.md).
The repaired PN1 reference now has image-to-cloud-to-proposal interfaces. The
real-file whole-case forward check is **unsupported**, because singleton clouds
cannot pass the retained DGCNN neighborhood operator. This outcome is preserved;
no threshold, padding or retry was used to turn it into a pass.

## Scope and retained implementation

This is R, the PN1-style reference, not a replacement for N's planned
PointNeXt/Transformer/GATv2 stages. The encoder and pointwise proposal modules
were copied from the historical checkout with LF normalization only. Their
source/destination hashes are in [reused_components.json at the historical commit](https://github.com/GraphiCortex/PointNeuron_Rec/blob/48519ff/configs/reused_components.json).
The local implementation's neighborhood search uses XYZ at the first block and
the first three learned feature channels at later blocks; this port preserves
that behavior and does not claim an official DGCNN reproduction.

The adapter explicitly selects the repaired `local` head: local normalized XYZ
as head features, but predicted offsets, radii and centers in native voxel
units. The original BatchNorm layers and ordinary `eval()` policy are retained.
No per-crop normalization substitute, loss, targets, graph model or learned
checkpoint enters this execution control. PyTorch is an optional reference
dependency; the existing environment was reused without installation changes.

## Explicit input control, not a universal PN2 foreground rule

[reference_cloud_control.example.json at the historical commit](https://github.com/GraphiCortex/PointNeuron_Rec/blob/48519ff/configs/reference_cloud_control.example.json)
uses the inherited image-only reference cutoff:

```text
threshold = round(maximum of decoded whole image * 0.2)
eligible voxels in each tile = native intensity > threshold
input cloud = uniform draw of up to 512 distinct eligible voxels
```

The threshold is computed once per image, not from storage dtype or each tile's
maximum. Intensities are not transformed by cloud construction. The retained
encoder subsequently uses its existing intensity normalization; that operator
was not changed. The cutoff remains vulnerable to outliers and weak branches.
Success here does not establish segmentation quality or promote this rule as
the final PN2 Phase 1 policy.

Changes relative to the historical repaired sampling runner are explicit:

- Direct uniform selection from all eligible tile voxels replaces its bounded
  1,024-row intermediate pool and subsequent 512-row draw. This is not an exact
  random replay of that runner. For sufficiently large pools both procedures
  give a uniform fixed-size subset marginally, but their seeded realizations
  differ; the old padded small pools can also contain duplicate voxels.
- There is no padding, learned sampler, guide, 256-foreground minimum or tile
  cap. Empty and singleton masks remain in the inventory. Clouds with two or
  more points use their actual size, up to the explicit budget.
- The draw is keyed by configuration seed, stable case ID and tile bounds,
  rather than a numeric dataset row. It is reproducible independently of tile
  call order. This randomization does not create per-neuron threshold rules.

Each packet retains global native XYZ/intensity, exact global source-voxel
indices, actual/distinct point counts, mask count, threshold and byte digests.
Overlapping tiles may naturally share voxels; uniqueness is enforced within
each cloud. Aggregate counts are sums over overlapping tiles, not counts of
unique neurons, branches or image voxels.

## Read-only model control and artifacts

`audit_reference_forward.py` takes strict image-only cases, explicit tiling and
cloud configurations, and a fresh output directory. It creates seed-zero fresh
CPU model parameters with the retained conservative identity-head initialization.
It loads no old weights and creates no optimizer or trained checkpoint.

The conservative head initially returns centers at input XYZ, radius one and
constant class logits. That is a starting condition, not learned accuracy or
evidence of a working classifier. All pointwise proposals are saved without
confidence filtering; there is no accepted-node or reconstruction gate.

The command writes per-tile NPZ point/proposal packets, a JSONL tile manifest
and a case summary. Each tile is one of:

| Status | Meaning |
|---|---|
| `forward_completed` | At least two distinct input voxels; raw model output saved |
| `empty_input` | The declared mask supplied no points; no model call |
| `unsupported_singleton` | One real point retained; the unchanged model cannot consume it |

An empty input is not a verified background label. It is an inference input
outcome of the declared mask. Any unsupported singleton makes the whole-case
forward status unsupported. Missing/corrupt sources remain failed cases. A
completed forward is still not graph construction or SWC reconstruction.

Parameters and all normalization buffers are hashed before/after. The command
fails if forward execution changes them. Ordinary inference has no adaptation
to a crop. Unit checks also use a nonzero final head, nondefault BatchNorm
buffers, batch sizes one/two, coordinate translation and state reload, so an
identity-head output cannot be the sole evidence of correct feature flow.

## Observed checks

**150 tests passed**: 123 previous checks plus 27 new cloud/model checks. They
use synthetic inputs, no training updates or real annotations. New checks cover
whole-image thresholding, strict cutoff boundaries, distinct sampling, exact
source-coordinate/intensity correspondence, empty/short clouds, batching,
translation, point order, ordinary BatchNorm behavior, exact state reload and
case/failure accounting. Nonzero-head comparisons use stated floating tolerance;
same-input checkpoint replay is exact on the tested CPU environment.

The same small training image used in the [input-interface check](image_input_interface.md)
(historical alias 32, chosen previously by minimum decoded size) then traversed
every tile under the explicit control, on CPU with one thread:

| Observation | Result |
|---|---:|
| Accounted / planned tiles | 168 / 168 |
| Model forward completed | 139 tiles |
| Empty masks | 21 tiles |
| Unsupported singleton clouds | 8 tiles |
| Eligible voxels summed over overlapping tiles | 173,580 |
| Selected points summed over overlapping tiles | 37,197 |
| Raw pointwise proposals from completed clouds | 37,189 |
| Whole-image maximum / cutoff | 255 / 51 |
| Model-forward time | 13.45 seconds |
| Per-case decode/cloud/forward/output time | 17.84 seconds, excluding process startup |
| Model parameters and buffers unchanged | Yes |
| Whole-case forward status / CLI exit | Unsupported / 1 |

The eight singleton tiles contain **four unique source voxels**. One is selected
in a completed overlapping cloud; three are not. The packet review therefore
does not justify dismissing all singleton tiles as duplicated coverage. No SWC
was read, so this does not identify these voxels as true neurite or noise.

Raw report, packets and `short_cloud_review.json` are under ignored
`tmp/reference_forward_20261022/`. No reserved image, reference content, learned
checkpoint, optimization, quality evaluation or GPU run was used. No automatic
retry was launched.

## Next proposed boundary correction

The retained `knn` explicitly rejects N less than two. A genuine singleton has
no other point available to supply an edge message. Adding duplicate input rows
would misrepresent cardinality; silently dropping it would discard observed
image evidence. The next proposal is to define a **self-only neighborhood** for
N=1 while leaving every N>=2 operation unchanged. The central feature would be
the observed point and its neighbor difference zero; there would still be one
real input point, not fabricated voxel support.

This is a small but substantive operator extension, not an implemented fix in
this commit. Review it before changing the retained source. The comparison
should use the same frozen clouds and parameters, require unchanged regular-
cloud outputs, explicit singleton outputs, unchanged state and no new threshold,
learned head or training. Preserve this unsupported run as the before record.
Successful execution of that boundary rule would not establish correct
singleton classification or generalization; later training and complete-SWC
comparisons must test those claims.

Geodesic/path aggregation, memory behavior, export and the supervision decision
remain ahead. There is no proposal that another tiny-set fit should precede
those interfaces. The destination is still the matched whole-neuron R/N
comparison in the [revised proposal](revised_proposal.md).
