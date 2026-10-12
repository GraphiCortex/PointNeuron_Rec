# Ready: one-change centerline negative-loss comparison

12 October 2026. The user approved preparing the control after reviewing the
[failed field fit](centerline_fit_results.md). The new 400-update fit **has not
started**. Its command is ready for the user to run visibly on the local GPU.

## Exactly one training change

The original negative BCE term averages over every known-negative voxel. The
control instead averages the **highest-loss `min(N, 3P)` known negatives** in
each crop, where P is the known-positive count and N the known-negative count.
Selection follows the current model's negative BCE losses on each update.

```text
loss = 0.5 * (mean positive BCE + mean selected negative BCE)
       + the original masked soft Dice loss
```

The positive BCE term and Dice term are unchanged. Dice still uses **all known
voxels**, including negatives not selected for BCE. Unknown bright regions and
padding remain excluded from both terms; hard-example selection cannot include
them. If fewer than 3P negatives exist, all negatives are used.

This is intended to focus the existing background penalty on unwanted predicted
voxels rather than average them among numerous easy empty voxels. It might
sharpen broad predictions or remove off-reference support. It could also
suppress dim branches or reinforce incomplete negative annotations. Neither
improvement nor success of the field concept is assumed.

[Loss-only config](../../configs/pointneuron2/centerline_loss_control.json) stores
the aggregation and ratio separately. It does not replace or edit the frozen
image-cache/model/optimizer config. The original `fit` action and default
`field_loss` retain their original all-negative behavior.

## Fixed setup and measured pairing checks

Eight original training crops: neurons 0, 32, 96 and 162, two crops each. Reuse
the immutable prepared native-image cache, two-voxel tube targets and unknown
mask. Same 88,673-parameter U-Net, seed 0, batch 1, no augmentation, 400 updates,
AdamW LR 0.001/weight decay 0.0001 and gradient clip 10. Same CUDA/PyTorch choice.
No development-validation or reserved-test images are used.

The new model is **freshly initialized**, not continued from the failed final
checkpoint. Its initial tensors were checked against the saved original
`field_step_0000.pt`: exactly equal. All eight initial inference measurements
also match the original report (floating metrics within 1e-5; accepted voxel
counts exactly equal). The preflight passes these checks before any update.

Cache, partition/record hashes, full model/optimizer config, declared update
budget, evaluation steps, crop identities and saved baseline reload status
are checked. Baseline report and initial/final checkpoint hashes are recorded;
they remain unchanged through the preflight and are checked again at the end
of the real comparison. Version/device/config mismatches fail explicitly.

The unchanged seeded crop-order algorithm visits every crop once per eight
updates. The control now records its actual index sequence and checks it
against that algorithm. The old real run did not save its actual sequence:
its ordering is inferred from the declared seed and unchanged loop, not
retrospectively proven from an absent trace. This limitation is reported.

## Fixed assessment and stop rule

Save updates **0, 100, 200 and 400** as before; update 400 is primary. Probability
threshold stays **0.5**. The gate remains:

- Every crop: reference-sample coverage within six voxels >=95%, with nonempty predictions.
- Every crop: predicted-voxel proximity to continuous segments within six voxels >=90%.
- Mean supervised-region hard Dice against the two-voxel tube >=0.80.

`loss_comparison.json` reports original versus control final metrics, every
crop's deltas, both gate decisions and pairing checks. The live terminal prints
both mean results. Original baseline: mean Dice **0.4007**, proximity **0.8086**,
coverage **1.0000**; those averages alone do not satisfy the per-crop gate.

The raw loss values are not directly comparable between the two objectives;
use the unchanged geometry measurements to assess improvement.

Review the final images and per-crop coverage as well as the aggregate. A
smaller prediction is not useful if it erases a branch. If the control fails,
stop and inspect it together; do not automatically extend updates, adjust the
threshold, change labels or launch validation. A pass establishes training-crop
learnability only. It does not establish generalization, SWC topology, a PN 1.0
improvement or commercial readiness.

## Local GPU command

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_centerline.ps1 -Action loss-control
```

The defaults reuse the completed baseline at
`tmp/pointneuron2/local_runs/centerline_fit_20261012_144121_14488/model` and the
prepared cache at `tmp/pointneuron2/local_runs/centerline_prepare_20261012/cache`.
Other local paths can be supplied with `-BaselineModels` and `-Manifest`.

Outputs go to a fresh ignored directory:

```text
tmp/pointneuron2/local_runs/centerline_loss_control_<timestamp>_<PID>/
  console.log
  model/
    fit_report.json
    loss_comparison.json
    field_step_0000.pt / 0100.pt / 0200.pt / 0400.pt
    step_0000/ / step_0100/ / step_0200/ / step_0400/
```

Each step folder contains probability arrays and eight prediction overlays.
The original run is preserved. Live losses print every 20 updates; all eight
crop measurements print at every evaluation, then the paired final summary.

Budget about **1.5–3 minutes** on the RTX 3070 Ti Laptop GPU. The matched
preflight measured 0.0935 seconds per update and peak allocated memory **636
MiB**. Its compute/evaluation estimate is 45.5 seconds, with startup and image
writing additional; the original visible fit took 1 minute 31 seconds.
[Saved preflight and pairing evidence](centerline_loss_control_preflight.json).

Only short throwaway timing checks were run during preparation. Each uses
four updates and saves no trained checkpoint. The final preflight additionally
checks the eight initial inference results. No environment/library changes
were made, and this comparison does not depend on the incompatible optional
scikit-image thinning diagnostic.

## Verification

**23 focused tests passed in 30.78 seconds**: original loss numerical equivalence,
hard-negative selection/count, unchanged positive/Dice terms, unknown-region
gradient exclusion, all-negative fallback, geometry and checkpoint checks.
A synthetic one-update-per-arm CPU control verifies shared initialization,
saved comparisons, crop order and rejection of mismatched settings/weights.
These are execution checks, not measured reconstruction improvements.

The real-cache GPU preflight then independently confirms pairing and identical
initial inference. Read-only audits recognize the saved loss definition, so a
future control audit will report the correct selected-negative BCE term.
