# Candidate expansion helps; the 50/50 selector is not the strongest control

12 October 2026. The user completed the four-control Phase 1 comparison in
**1 minute 7 seconds** at the terminal; the recorded loop took **62.72 seconds**.
That measured runtime supersedes the conservative 3-6 minute estimate.
Execution checks pass. **These are eight crops used to train the field**,
with five sampling seeds; this is not development validation, a complete PN 2.0
reconstruction or independent evidence of biological branch recovery.

The intended architecture remains **Phase 1 point cloud -> PointNeXt ->
Transformer skeleton proposals -> GATv2 connectivity -> SWC**. This experiment
performed eight frozen field forward passes and zero optimizer updates. No
sampler has been promoted, no later module ran, and no new training or sampling
strategy was implemented while inspecting these results.

## Main result: eligibility is a major loss stage in these fitted examples

F is the original above-threshold image pool. C is its union with predicted
field support at score >=0.5, retaining actual intensities and voxel coordinates.
The field does not receive a reference at inference, but its weights have
already fitted the reference tubes for these particular crops.

Even before sampling, keeping **every** F point only covers **10.40%** of
reference length within two voxels in crop 0/2 and **15.65%** in crop 96/0.
No selection algorithm restricted to those F pools can exceed those ceilings
for this metric. C reaches **100% candidate coverage** within two voxels on all
eight crops. The old threshold therefore excludes useful reference-supported
locations in these examples; increasing the sample count alone cannot fix that.

### Same unique-point count in every arm

Means average the five seeds and two crops within each of four neurons, then
weight neurons equally. Each nominal budget K uses `min(K,len(F))` distinct
points in **all four arms**. This isolates benefits without granting the field
arms more unique points. The 2,048 column does not mean every crop supplied 2,048
points; some F pools have only 385, 487 or 613 locations.

| Arm | Reference coverage within 2 voxels, nominal 512 | Same metric, nominal 2,048 | Proximity within 6 voxels, nominal 2,048 |
| --- | ---: | ---: | ---: |
| A: threshold random | 57.64% | 65.26% | 94.47% |
| B: expanded random | **91.26%** | **97.61%** | 98.12% |
| C: expanded spatial coverage | 83.94% | 94.40% | 97.40% |
| D: 50/50 guide plus context | 89.08% | 95.70% | **98.27%** |

B gains **32.35 percentage points** over A at the larger matched budget. This
supports candidate expansion, not simply duplication or a larger input array.
Reference proximity remains annotation agreement, not biological precision.

### Expanded-pool budget: equal counts for B/C/D

Here B/C/D receive `min(K,len(C))` distinct points in each pair; A retains its
available F count. Different-count A comparisons are explicitly availability
comparisons rather than strict selector comparisons.

| Arm | Coverage within 2 voxels, nominal 512 | Coverage within 2 voxels, nominal 2,048 | Proximity within 6 voxels, nominal 2,048 |
| --- | ---: | ---: | ---: |
| A: threshold reference, sometimes fewer points | 57.64% | 65.26% | 94.47% |
| B: expanded random | **91.39%** | **99.973%** | 98.14% |
| C: expanded spatial coverage | 84.05% | 99.04% | 97.49% |
| D: 50/50 guide plus context | 89.24% | 99.37% | **98.37%** |

The larger budget helps the expanded methods, but neither 512 nor 2,048 is a
final product capacity. Two C pools have fewer than 2,048 unique points; those
cases use the complete pool and the three expanded methods return the same
locations, with different role metadata only.

## Primary allocation comparison and branch losses

The prespecified primary contrast was D minus C in the expanded group at
nominal 2,048, with 40 crop/seed pairs and identical actual unique counts:

- Mean coverage gain: **+0.328 percentage point**; 11 positive, 5 negative,
  24 ties. This is a small aggregate gain, not an overall sampler win.
- Mean proximity gain: **+0.876 percentage point**; 19 positive, none negative,
  21 ties.
- Worst-chain coverage: 7 pairs improve, **9 worsen**, 24 tie. A positive mean
  difference does not erase the local losses.

Against B, D has **0.600 percentage point less mean coverage**, with no positive
coverage pairs, 14 negative and 26 ties. It has a small +0.229 percentage-point
mean proximity tradeoff, which is not uniform across crops. B is therefore the
strongest coverage baseline in this in-sample comparison. Do not silently adopt
D as the default merely because its original primary comparator was C.

Concrete branch checks:

- Crop **162/1, seed 0**: reference chain 281->794 has length **12.41 voxels**
  and is not boundary-clipped or shorter than six voxels. Its two-voxel coverage
  is **100% with B and C**, versus **75.83% with D**. This is not only a tiny
  boundary-fragment effect.
- Crop **96/3, seed 4**: the same 5.58-voxel chain 776->787 falls from **71.99%**
  coverage under C to **47.90%** under D even as D improves that crop's total
  coverage. The chain is short but not boundary-clipped; it remains reported.

The selector knows image cells, not biological branch identities. These results
show that its spatial/score allocation does not guarantee branch preservation.
The exact cause of each loss was not diagnosed or repaired, and no fraction,
cell-size or threshold sweep was run.

## Per-crop expanded comparison at nominal 2,048

Coverage below averages five seeds. N is constant across seeds for an arm.
B/C/D share the same N; A may have fewer points. Worst-chain and gap details
remain in the saved summary and the raw per-cloud measurement files.

| Crop | N: A / B,C,D | Coverage A | Coverage B | Coverage C | Coverage D | Image |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0/1 | 385 / 1,013 | 73.08% | 100% | 100% | 100% | [Open](field_sampling_results_seed0/sample_0000_patch_001.png) |
| 0/2 | 613 / 2,048 | 10.40% | **99.94%** | 98.51% | 98.36% | [Open](field_sampling_results_seed0/sample_0000_patch_002.png) |
| 32/0 | 1,267 / 2,048 | 92.51% | 100% | 100% | 100% | [Open](field_sampling_results_seed0/sample_0032_patch_000.png) |
| 32/3 | 2,048 / 2,048 | 96.25% | 100% | 100% | 100% | [Open](field_sampling_results_seed0/sample_0032_patch_003.png) |
| 96/0 | 487 / 2,048 | 15.65% | 100% | 99.95% | 100% | [Open](field_sampling_results_seed0/sample_0096_patch_000.png) |
| 96/3 | 2,048 / 2,048 | 43.98% | **99.84%** | 94.32% | 97.09% | [Open](field_sampling_results_seed0/sample_0096_patch_003.png) |
| 162/1 | 2,048 / 2,048 | 90.20% | **100%** | 99.59% | 99.53% | [Open](field_sampling_results_seed0/sample_0162_patch_001.png) |
| 162/2 | 603 / 1,256 | 100% | 100% | 100% | 100% | [Open](field_sampling_results_seed0/sample_0162_patch_002.png) |

All eight XY/XZ figures were inspected. They show much fuller input support
when C is admitted, while some spatial/guided clouds retain small branch gaps.
Point proximity and projected overlap do not establish continuous paths, branch
parentage, radii or final SWC topology. Cyan is reference, orange ordinary/context
points and green guide points. Figures use seed 0, not the five-seed averages.

![Crowded training crop: all four controls, seed 0](field_sampling_results_seed0/sample_0096_patch_003.png)

## Zero-intensity additions and interpretation limits

The design explicitly allowed field-supported dim or zero-intensity locations.
The exported raw intensity remains exactly what the decoded image contains:
no bright signal was painted into any point.

For fly crop **162/1**, C includes **4,447 zero-intensity locations out of 7,935**;
B selects about **1,145 zero-intensity points out of 2,048** on average. For
162/2 the full C pool includes **632 out of 1,256** such locations. A two-voxel
target tube around a sparse trace can include surrounding dark voxels; the
reference-supported additions must not all be called recovered visible signal.
The figures alone cannot distinguish interpolation/context support from fitted
or hallucinated geometry. This matters for an intensity-bearing PointNeXt input
and must be evaluated on neurons not used to train the field.

Likewise, this compares the learned expansion against **one original intensity
cutoff**. It does not establish that a learned field is the only or cheapest
way to improve eligibility. A more permissive image-only candidate control is
a possible separately agreed pilot addition; it was not silently added here.

## Recommended next step, for discussion

Move past tiny-crop sampler tuning and **prepare a broader field
train/development-validation pilot** on the existing **33 training / 7
development neurons**, leaving **8 reserved test neurons untouched**. Carry B
as the current coverage baseline and retain D as a tradeoff control rather than
promoting it. The original threshold and spatial controls are cheap enough to
remain informative comparisons as well.

Before implementation or launch, agree native-cache coverage and annotation
policy, field model/loss/augmentation/optimization budget, checkpoint and
advancement rules, input-cloud controls, runtime and local command. Preset
image-defined crop boxes only assess their covered regions; a patch pilot must
not be presented as whole-neuron coverage. Field generalization and eventual
PointNeXt/Transformer/GATv2 behavior remain separate measurements.

No broader pilot or new sampling recipe was implemented in this result review.
The source field's earlier training-repeatability caveat also remains unresolved;
successful frozen inference does not establish reproducible training.

## Verification and visible evidence

The completed run reports 640 group/budget/crop/seed/arm rows, 40 primary pairs,
eight frozen forward passes and zero updates. A fresh independent file audit
verified all protected hashes, unique row keys, matching actual counts and all
**385 distinct saved clouds**. Every cloud has distinct native coordinates,
exact voxel-to-XYZ mapping, original raw intensities, matching full-precision
field scores and no selected padding. Reference traces did not select points.

CUDA peak allocated memory was **241.33 MiB**; retained input arrays were
**69.63 MiB**, not a measurement of total peak process RAM. No environment,
model, sampler code, cache or checkpoint was changed during inspection, and no
additional inference, selection or training was run.

- [Readable summaries, per-crop means, primary pairs, independent checks and source-report hash](field_sampling_results_seed0/summary.json)
- [All 640 compact metric rows](field_sampling_results_seed0/metrics.csv)
- [Completed protocol and source entry points](field_sampling_ready.md)
- [Approved design and candidate/selection contracts](phase1_field_guided_sampling_design.md)
- Raw clouds and per-chain measurement files remain in the user's ignored run:
  `tmp/pointneuron2/local_runs/field_sampling_compare_20261012_230621_14488/comparison`.
