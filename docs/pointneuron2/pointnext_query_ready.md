# Native PointNeXt → Transformer prototype: preparation and timing

13 October 2026. The user selected the native-PyTorch implementation route.
**Completion update:** preparation, GPU timing and the 2,000-update fit are now
complete. Reload passed; both learnability gates failed. See the
[results and saved-weight normalization diagnosis](pointnext_query_results.md).
The text below preserves the implemented protocol as it was prepared.

The prototype, immutable two-crop cache builder and separate timing/fit actions
are implemented in the real PN2 folder. **No real PointNeXt fit has run.**
Passing implementation checks is not measured reconstruction improvement.

## What was implemented

The [encoder](../../src/pointneuron/models/pointnext_native.py) follows the
small PointNeXt-S segmentation recipe: a linear four-channel stem, four
residual set-abstraction stages, widths 32/64/128/256/512, strides 1/4/4/4/4,
radii .1/.2/.4/.8 and 32 neighbors. Four feature-propagation stages restore
32-channel descriptors to each original input point, in its original order.
The encoder/restoration contains **790,720 parameters**. This S recipe has
no extra inverted-residual blocks; larger variants are separate experiments.

The [wrapper](../../src/pointneuron/models/pointnext_query.py) feeds those
descriptors to the existing Transformer query body: 512 queries, dimension 128,
three layers, four heads, feedforward width 512 and dropout zero. Its input
projection explicitly accepts 32 channels. Outputs retain native-voxel centers,
two-class confidence, positive radii and 128-channel query features. GATv2
connectivity remains a later stage; this component check does not trace an SWC.

The checkpoint schema is `pointneuron2-pointnext-query-v1`, with a strict
native-backend model identity. Old DGCNN loaders reject it. No historical
DGCNN weights are silently reused, and the old query factory remains unchanged.

## Native operators and short-cloud behavior

The [operators](../../src/pointneuron/models/pointnext_ops.py) use float32
PyTorch tensors without extensions or environment changes:

- FPS starts at input index zero, resolves ties by lowest index and selects
  distinct indices, including when coordinates coincide.
- Ball query uses strict distance `< radius`, support-index order and the
  first valid neighbor to fill short neighborhoods. This repeats neighborhood
  indices, not input points. Empty external balls raise an error.
- Feature propagation uses up to three actual nearest support points and
  weights `1 / (distance + 1e-8)`. One/two-point support uses only those points.
  Feature gradients pass through interpolation; membership/weights are fixed
  geometry operations. Distance processing is chunked at 256 query points.
- Inputs need at least 256 distinct selected voxel indices. Smaller clouds
  fail explicitly; stages, BatchNorm and input counts are not altered to fit.
  Train/eval checks pass at N=256 and N=358, including a final coarse count of
  one. The reference residual skip projection has no pooled BatchNorm.

Coordinates are `(native_xyz - patch_center) / 48`; intensity uses the cached
**full-image** scale. Field scores are saved for provenance, not fed as a fifth
channel. Radius .1 is 4.8 native voxels. The prototype uses the reference
radius grouping variant, while the proposal describes kNN; the difference is
explicit in the [agreed design](pointnext_query_design.md).

Reference URLs, revisions and hashes are retained in the
[manifest](pointnext_reference/reference_manifest.json), with the
[OpenPoints MIT license](pointnext_reference/LICENSE.openpoints).
Native tie handling and short-support interpolation are declared conventions;
official CUDA-kernel equivalence and published benchmark throughput are untested.

## Fixed inputs and future learnability check

Preparation uses exactly training neuron **0**, existing image-defined crops
**0 and 1**, and the frozen fresh **ungated baseline**, epoch 30, from
`tmp/pointneuron2/local_runs/field_attention_20261013/baseline/model`.
The source check passed against the actual local checkpoint/cache.

For each crop, form the union of original image foreground F and field scores
at least .5, then use the existing expanded-uniform B selector with seed
`0 + neuron_index*10000 + crop_index`. Select `min(2048, union_count)` unique
points without replacement. This is expanded mode, not the comparison cap
based on the original foreground count. Actual counts will be known after
preparation. No quota, padding or new selection rule is added.

Both point clouds are saved **before** reference targets are opened. Separate
target files retain existing 8-voxel curve targets (6 and 12 nodes), identity,
full clipped-reference geometry and per-chain probes. Preparation also produces
XY/XZ inspection images showing the raw image, selected image points, field-only
additions, reference curves and training targets. These projections show support,
not predicted edges. Only these two training crops are used; development and
reserved test neurons do not participate in fitting.

The [fixed recipe](../../configs/pointneuron2/pointnext_query.json) is batch one,
no augmentation, AdamW LR .0001, weight decay .0001, gradient clip 10, unchanged
query loss and score cutoff .5. A future fresh fit is bounded to 2,000 updates,
alternating the two crops, with saved checks at 0/500/1,000/2,000. The per-crop
gate requires distinct accepted matches within one voxel, zero unmatched
accepted queries, at least 95% accepted-node curve support within six voxels
and finite positive radii. Radius error and branch gaps are reported separately.
This establishes learnability only, not generalization or superiority to PN1.

## Your next visible commands

First freeze and inspect the inputs. This runs **two frozen field forward
passes**, with zero optimizer updates and no PointNeXt inference:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_pointnext_query.ps1 -Action prepare
```

Default output directory:
`C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\pointnext_query_20261013`.
The `cache` subdirectory contains `manifest.json`, two `_points.npz` files,
two `_targets.npz` files and two `_inspection.png` images. The source field
checkpoint and native cache are preserved and checked by hash.

Then measure the actual local runtime, separately:

```powershell
& .\scripts\pointneuron2\run_pointnext_query.ps1 -Action preflight
```

This uses the RTX 3070 Ti Laptop GPU through the existing local environment.
It performs **four throwaway updates** (two warm-up, two measured), verifies
finite gradients and reload, and saves `timing.json` with a 2,000-update estimate
and peak allocated CUDA memory. No trained checkpoint is retained. The estimate
is based on two measured alternating steps; it is a planning estimate, not a
guaranteed duration. CPU is available only through explicit `-Device cpu`.

**Review the actual point counts, inspection images and timing together before
launching training.** The implemented `-Action train` is a separate command;
neither preparation nor timing starts it. Timing/cache/backend/source hashes
must match before training can start. Existing and failed outputs are preserved;
there is no automatic retry, promotion or follow-on experiment.

Console logs are stored beside the experiment directory. The
[runner](../../scripts/pointneuron2/run_pointnext_query.ps1) and
[Python actions](../../scripts/pointneuron2/pointnext_query_experiment.py) are
visible local source files; generated artifacts remain in ignored `tmp`.

## Verification completed here

- **15 tests passed in 32.29s** across the
  [native model/operators](../../tests/test_pointnext_native.py),
  [synthetic prepare/timing workflow](../../tests/test_pointnext_runner.py)
  and existing query-normalization tests. All used synthetic CPU data.
- The read-only real-source action passed without inference or optimizer updates.
- A [topology audit](pointnext_reference/topology_audit.json) extracted the
  actual pinned reference network classes and supplied the **same native
  operators** on both sides. At synthetic `[1,358,4]`, identical copied weights
  produced `[1,358,32]` descriptors with maximum delta **0.0** and identical
  790,720 parameter counts. This validates layer wiring, not official kernels.
- The visible [audit script](../../scripts/pointneuron2/audit_pointnext_topology.py)
  verifies the reference source hash and runs CPU only. To reproduce it, save
  the manifest's pinned `pointnext.py` to
  `tmp/pointneuron2/pointnext_reference/pointnext.py`, then run it with the existing
  environment. It does not install or download code automatically.

Real preparation, GPU timing, the fit gate, generalization, GATv2 and final
matched reconstruction evaluation remain pending.
