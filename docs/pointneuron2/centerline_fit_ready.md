# Centerline field: first bounded learning check

12 October 2026. Historical protocol, prepared locally after the user approved
trying an image-native field. **The user has now completed the 400-update fit;
its learning gate fails.** See [the measured result and diagnosis](centerline_fit_results.md)
before running anything further. The commands below document the completed
experiment; another fit is not the current recommendation.

The conceptual change is to give every voxel a stable spatial target derived
from continuous reference segments, while retaining image context before point
compression. This is a new hypothesis, not evidence that the proposal is now
better than PN 1.0 or that query-based methods cannot work.

## What this version does

```text
native 3D image -> compact 3D U-Net -> probability of a reference-centerline tube
```

One field head, 88,673 parameters, base channels 8, two pooling stages and skip
connections. GroupNorm works separately within each crop; there is no BatchNorm,
dropout, slot matching or learned sampler. Images retain their full voxel grid
and all intensities, normalized by the observed full-volume maximum. No
foreground threshold removes input pixels. The fixed crops are 97 x 97 x 97;
out-of-volume padding is excluded from supervision and metrics. Original XYZ
coordinates map explicitly to ZYX arrays, without an invented offset or scale.

The target is a **two-voxel tube around complete SWC segments**, including
crop-crossing edges, roots and zero-length edges. It is not a biological neurite
segmentation mask or a radius estimate. Distances are acquisition voxels; these
volumes are not rescaled into physical micrometers. Exact segment distances
avoid changing the label when an SWC edge is subdivided into collinear nodes.

The first loss is class-balanced binary cross-entropy plus masked soft Dice.
The positive and negative BCE means contribute equally, avoiding domination by
the enormous background voxel count. A continuity objective such as clDice is
**deferred** until this supervision passes inspection and the field can learn.
It would be a separate substantive change, especially because unknown masks
can create artificial skeleton boundaries. No topology guarantee is claimed
for this prototype.

## Annotation uncertainty and its limits

The policy is deliberately weak supervision:

- Within two voxels of a trace: positive, including dim traced signal.
- More than six voxels from a trace and brighter than the saved image cutoff:
  unknown, excluded from direct loss rather than labeled background.
- Other valid voxels: weak negatives, including the annulus around a trace.
- Padding and any crop with no local trace: unknown.

This uses the existing image-derived cutoff only for uncertainty, **not for
input extraction**. It does not establish reference completeness: untraced dim
neurites could still be negative, and untraced bright distractors receive no
negative label. Predictions in excluded bright regions remain included in
reference-proximity reporting, and their count is printed separately. Thus
the model cannot pass the gate by hiding those predictions behind the mask.
Reference proximity is agreement with this SWC, not biological precision.

## Prepared images and supervision

Eight fixed crops from four **training neurons only**. Choosing annotated crops
here is allowed because this is a learnability diagnostic, not automatic crop
selection or a generalization evaluation. Development-validation and reserved
test images were not opened by this preparation.

| Training neuron | Crops | Dataset group | Reference-centerline voxels above old cutoff |
| --- | --- | --- | --- |
| 0 | 1 / 2 | Chick | 51.9% / 5.8% |
| 32 | 0 / 3 | Fruit-fly larva | 42.3% / 50.7% |
| 96 | 0 / 3 | Zebrafish horizontal cells | 9.3% / 33.9% |
| 162 | 1 / 2 | Fly circuit | 24.6% / 34.4% |

These percentages are measured on native voxel centers within 0.9 voxels of
the continuous trace. They show substantial dim reference-associated signal
below the old cutoff; they do not prove perfect local alignment. XY/XZ maximum
projections show broad correspondence between traces and image structures,
with reference offsets, crowded crossings and uncertain signal still visible.
Projection overlap is not evidence of 3D connectivity. Inspect the actual
predicted fields after fitting before accepting a numerical pass.

Images are contrast-stretched **for display only** using a per-crop white level
(99.5th percentile, falling back to maximum for extremely sparse crops).
Network intensities are unchanged. Red marks the reference; cyan marks excluded
bright regions. Exact source-image, SWC, partition and NPZ hashes are recorded
in [the preparation summary](centerline_fit_inspection/preparation_summary.json).

| Crop | Overlay |
| --- | --- |
| 0 / 1 | [Open](centerline_fit_inspection/sample_0000_patch_001.png) |
| 0 / 2 | [Open](centerline_fit_inspection/sample_0000_patch_002.png) |
| 32 / 0 | [Open](centerline_fit_inspection/sample_0032_patch_000.png) |
| 32 / 3 | [Open](centerline_fit_inspection/sample_0032_patch_003.png) |
| 96 / 0 | [Open](centerline_fit_inspection/sample_0096_patch_000.png) |
| 96 / 3 | [Open](centerline_fit_inspection/sample_0096_patch_003.png) |
| 162 / 1 | [Open](centerline_fit_inspection/sample_0162_patch_001.png) |
| 162 / 2 | [Open](centerline_fit_inspection/sample_0162_patch_002.png) |

## Frozen fit and stop rule

[Config](../../configs/pointneuron2/centerline_fit.json): seed 0, batch 1,
400 AdamW updates, learning rate 0.001, weight decay 0.0001, gradient clip 10.
Each block of eight updates visits all crops once in seeded shuffled order:
50 visits per crop. No augmentation in this fixed-crop learning check.
Evaluations, checkpoints, raw probability arrays and four-panel overlays are
saved at updates **0, 100, 200 and 400**. The final update is the primary result;
no best-snapshot or threshold selection.

At **probability 0.5**, the final gate requires:

1. At least 95% of each crop's dense reference samples within six voxels of a
   predicted voxel, and no empty predictions.
2. At least 90% of each crop's predicted voxels within six voxels of the actual
   continuous reference segments, counting unknown-region predictions too.
3. Mean hard voxel Dice at least 0.80 over the supervised regions.

This is a fit gate, not validation, topology accuracy or a commercial readiness
criterion. The report also includes a simple old image-cutoff baseline on the
same native crops; it is not PN 1.0 or the pointwise model.

**If it fails:** stop this version and review the loss trajectory and eight
overlays together. Do not automatically extend training, lower the threshold
or launch another architecture search. Separate supervision/placement issues
from optimization before deciding a next hypothesis.

**If it passes:** review the fields together, then agree one train/validation
comparison on the existing 33/7 split against the usable pointwise baseline.
That later method comparison must use the same source images, continuous
reference metric, neuron weights and fixed threshold; input representation and
architecture differ, so it tests the proposed method as a package. First assess
crop geometry/confidence, then agreed whole-volume output with the graph/SWC
settings held fixed. The eight reserved test neurons stay untouched. Training
success alone cannot justify promoting the new method.

## Run visibly on your GPU

Preparation is already done at:

```text
tmp/pointneuron2/local_runs/centerline_prepare_20261012/cache/manifest.json
```

The timing check on the local **RTX 3070 Ti Laptop GPU** measured 0.124 seconds
per update after two warm-up updates, with peak allocated memory **637 MiB**.
Its compute/evaluation estimate was 66 seconds; startup, image writing and
thermal/device variation are additional. Budget approximately **2–4 minutes**.
Four timing updates were discarded; no fitted model is retained from them.
[Saved timing report](centerline_fit_inspection/gpu_timing.json).

In your PowerShell terminal:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_centerline.ps1 -Action fit
```

Live losses print every 20 updates; all eight crop results print at each
evaluation. The runner prints a fresh local output folder under
`tmp/pointneuron2/local_runs/centerline_fit_<timestamp>_<PID>/model` containing
`fit_report.json`, four checkpoints, probabilities and overlays. The parent
folder contains `console.log`. Source/scripts/configs and this report are
ordinary visible files; generated training artifacts stay ignored.

Alternate cache paths can be supplied with `-Manifest`; alternate environment
paths with `-Python`. To reproduce CPU preparation into a fresh directory:

```powershell
& .\scripts\pointneuron2\run_centerline.ps1 -Action prepare
```

That prints the new cache path; pass it explicitly to the subsequent fit with
`-Manifest`. The default fit path is the already prepared cache above.

## Execution checks

Focused tests cover native axis/padding handling, fractional crop bounds,
continuous crossing edges and subdivision invariance, isolated/zero-length
geometry, unknown-label gradients, single-crop inference and checkpoint reload,
per-crop gate failures, reserved-neuron rejection and the bounded synthetic
runner. These are execution checks. **No measured field reconstruction
improvement exists yet.**

Validation: **20 tests passed in 28.00 seconds**, covering the new field tests
and the existing continuous-curve geometry tests. The synthetic runner used
one CPU update on generated tiny arrays, not the real eight-crop experiment.
