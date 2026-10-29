# Input and PointNeXt audit, 23 October 2026

Initial status on 23 October: source audit and CPU execution check complete; GPU
capacity sweep prepared, not launched. **29 October update:** the user-launched
sweep is [reviewed](pointnext_capacity_results.md), including a confirmed native
index-storage issue. Its [CPU-validated compact correction and user-launched 64k
GPU replay](pointnext_compact_replay64.md) now pass both forward and backward
execution. A [bounded real-cloud contrast/neighborhood pass](real_branch_evidence_results.md)
is also complete; trained comparisons have not been run. No product encoder,
replacement sampler or learned normalization
policy is adopted by this diagnostic. The intended architecture is unchanged.

## Correct the interpretation first

The repeated-value encoding round-tripped every one of 209 boxes exactly.
13/135 nonconstant boxes had an exception count fitting the old dense sampling
quota **before context**. That count describes neither encoding losslessness nor
PointNeXt capability. It cannot justify a Phase 1 redesign by itself.

The relevant questions are whether a finite cloud retains informative image
evidence, whether the hierarchy exposes it to useful features, and whether the
trained proposal/reconstruction stages use those features. These require different
measurements. A successful untrained forward pass answers only execution cost
and shape compatibility.

## Actual fresh Phase 1 interface

Audited `src/pn2/inputs/sampling.py`, `cases.py` and
`scripts/probe_phase1_sampling.py` in this worktree:

* Disjoint 128-cubed native-voxel boxes; source-edge boxes clipped, not padded.
  Current probe tiling is not yet the proposal's overlapping inference tiling.
* Dense quota is ceil(box voxel count / 64); lean quota is ceil(count / 512).
  A full box therefore has 32,768 or 4,096 **distinct source voxels**. These are
  sampling choices, not measured GPU limits, Transformer queries or graph nodes.
* Whole-source image normalization is `(I - source_min)/(source_max-source_min)`;
  constant images become zero. Uniform, intensity, or Sato guide weights enter
  sampling. Weighted arms mix 10% uniform probability mass, not a 10% point quota.
* Sato uses image-only bright-ridge responses at native-voxel sigmas 1, 2, 4.
  Neither Sato nor intensity requires a reference skeleton at inference.
* Saved clouds contain source flat indices, native XYZ and typed raw intensity.
  No guide feature channel is stored. Guide influence remains in the distribution
  of selected positions. Current points are ordered by sampling race rank.
* Image manifests reject reference-bearing keys. The generator has no SWC input;
  reference geometry is read later by the evaluator. This is an implementation
  audit of the present path, not a guarantee about every future data loader.

Historical field-derived clouds and historical checkpoints are not this path.

## Explicit candidate adapter for the execution diagnostic

There is no integrated encoder in fresh PN2 yet. The diagnostic imports the
retained native prototype read-only, checks both source hashes in
`configs/pointnext_capacity_probe.json`, and initializes fresh weights. It does
not load checkpoints. This reuse tests an available Windows-compatible backend;
it does not select the final production implementation.

For a full 128-cubed box:

| Quantity | Diagnostic definition |
|---|---|
| Coordinates | XYZ in native voxels, center (63.5,63.5,63.5), divide each axis by 64 |
| Clipped edge box, prospective | Actual box midpoint, same scalar 64; no axis stretching |
| Attribute tensor | `[local_x, local_y, local_z, normalized_intensity]`, float32 |
| Intensity | Whole-source affine range above; never selected-cloud min/max |
| Explicit Sato channel | None |
| Point order | Sampling-independent ordering required for real comparisons; see below |
| Encoder output | Aligned per-input-point descriptors `[1,N,32]` |

The script's synthetic grid has image I=x+2y+3z, full-source range 0..762,
unique voxel indices and a seed-zero random ordering independent of intensity.
The four budgets are nested prefixes of that ordering. No real-image retention
claim follows from this grid.

This adapter differs from the historical 97-wide crop adapter (scale 48 and
raw intensity divided by image scale, without subtracting its minimum). Radius
conversion changes accordingly. Explicitly freeze and review the adapter before
a real trained comparison; old weights do not transfer unchanged by assumption.
Native-voxel distances here assume isotropic voxel units. New calibrated,
anisotropic acquisitions require an explicit physical-coordinate contract.

## Audited native hierarchy

Stem 4 -> 32 features; four residual abstraction stages; four 3-nearest-neighbor
inverse-distance restoration stages with fine-level feature skips. FPS selects
one quarter of points at each abstraction. Relative offsets are divided by each
ball radius. Each abstraction uses max aggregation with at most 32 distinct
in-radius members, repeating the first member if fewer exist.

| Stage | Input -> output channels | Relative radius | Native-voxel radius at scale 64 | K | Remaining fraction |
|---|---|---:|---:|---:|---:|
| Stem | 4 -> 32 | — | — | — | 1 |
| 1 | 32 -> 64 | .1 | 6.4 | 32 | 1/4 |
| 2 | 64 -> 128 | .2 | 12.8 | 32 | 1/16 |
| 3 | 128 -> 256 | .4 | 25.6 | 32 | 1/64 |
| 4 | 256 -> 512 | .8 | 51.2 | 32 | 1/256 |

| Input points | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---:|---:|---:|---:|---:|
| 8,192 | 2,048 | 512 | 128 | 32 |
| 16,384 | 4,096 | 1,024 | 256 | 64 |
| 32,768 | 8,192 | 2,048 | 512 | 128 |
| 65,536 | 16,384 | 4,096 | 1,024 | 256 |

At 32,768/128^3 density, a volume-per-point cube has side 4 voxels. This is
not the actual spacing of a random/weighted cloud, and gives no thin-branch
coverage guarantee. A 6.4-voxel ball may connect selected evidence along a branch;
it may also mix nearby branches. Its usefulness cannot be established from
radius and nominal density alone. Restoration skips retain fine-point access,
but do not prove that useful thin-branch information survives feature learning.

Grouping uses strict distance < radius and the **first 32 support indices in the
ball**, not nearest 32. FPS starts at input index zero and breaks ties by index.
The retained OpenPoints CUDA ball-query source uses the same first-K convention.
Thus sampling-rank ordering can change local evidence, particularly in clustered
weighted clouds. This is not established as a native implementation bug.
For real comparisons, sort selected source IDs, then apply a fixed seed shuffle
independent of sampling weights; pair this convention across arms and retain an
ordering sensitivity control. Do not silently replace ball grouping with KNN.

The pinned official sources were inspected locally: OpenPoints
`bc778ec20c51fa9a72861c0ffe32718815367b6a`, PointNeXt S configuration
`b7ec10fd8b9acdb1c3b18e557f0d3465eb919164`. The native model remains a documented
prototype. Its Python FPS loops and dense chunked distance/sorting operations
are not the official accelerated kernels. A timeout is a finding about this
backend, not proof of a PointNeXt architecture limit.

## Execution protocol and completed CPU result

`scripts/run_pointnext_capacity.ps1 -Action sweep` prepares a fresh process per
budget on CUDA device zero, batch 1, float32, seed zero, without optimizer or
parameter updates. Per mode: one warmup and two timed iterations, synchronized
GPU clocks, peak allocated/reserved CUDA bytes, output shapes, observed stage
shapes and device/runtime metadata. Modes are eval forward and train forward +
backward of mean squared descriptors. Ordinary native BatchNorm is retained for
this cost check; its eval running statistics are untrained. No train/eval feature
equivalence or learned normalization policy is claimed.

Each budget has a 300-second process cap; timeout/OOM/error stops the sweep and
marks larger budgets unattempted. No retry, point reduction, mixed precision or
hidden chunk change. Earlier completed mode reports survive later failure.
Console output streams live and is saved. The full cap is 20 minutes plus
process overhead; it is a bound, not a runtime prediction. Encoder cost excludes
Transformer query attention, GAT, image preparation and full-volume processing.
An N-by-512 query interface can have additional substantial cost.

CPU check completed with Torch 2.12.1+cu130, 256 points, one CPU thread:
normalization endpoints and constant image, 6.3/6.5-voxel grouping boundary,
first-K ordering, finite `[1,256,32]` forward and all parameter gradients passed.
Raw verified launcher report: `tmp/pointnext_capacity_cpu_verified_20261023/report.json`.
The three CPU-only controller tests also pass: success-report preservation,
timeout cleanup/log preservation, and missing-report rejection without continuing
to larger budgets. The subsequent user-launched CUDA sweep and separate CPU
storage audit are recorded in the [results](pointnext_capacity_results.md).
No GPU sweep or training has been launched by the agent.

## Decision sequence

1. The [compact-index correction and 64k replay](pointnext_compact_replay64.md)
   pass. Close this bounded execution milestone; 32k is not the demonstrated
   ceiling. Interpret execution capacity separately from biological retention
   and from unmeasured whole-pipeline costs.
2. The [first bounded real-cloud measurement](real_branch_evidence_results.md)
   is complete on frozen clouds. Preserve the broader
   [stratified protocol](branch_retention_protocol.md); do not infer a universal
   selector, thin/dim adequacy or learned feature loss from the current proxy.
3. Freeze a few-box supervised task, normalization policy, head and training cap.
   Establish learnability, then run matched selectors and budgets. A capacity pass
   alone is insufficient permission to begin another long fit.
