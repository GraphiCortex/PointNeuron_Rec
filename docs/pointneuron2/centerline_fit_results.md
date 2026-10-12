# Field fit learns structure but fails centerline localization

12 October 2026. The user completed the eight-crop, 400-update fit in **1 minute
31 seconds** at the terminal (82.41 seconds inside the training/evaluation
loop). The final checkpoint reload is exact. **The frozen learning gate fails;
this model is not promoted to validation or SWC reconstruction.**

## What failed, and what improved

These are training-crop measurements, not unseen-neuron results. All evaluations
use the original probability threshold 0.5. Means weight the eight fixed crops
equally; each of four neurons contributes two crops.

| Measurement | Before training | Update 100 | Update 200 | Update 400 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Mean reference coverage within 6 voxels | 100% | 100% | 100% | 100% | Each crop >=95% |
| Mean predicted-voxel proximity within 6 voxels | 5.6% | 75.0% | 73.7% | 80.9% | Each crop >=90% |
| Mean hard Dice against the two-voxel tube | 1.0% | 30.0% | 31.0% | **40.1%** | Mean >=80% |

Coverage was already perfect in the untrained model because hundreds of
thousands of voxels exceeded the threshold. Therefore, perfect coverage alone
is not evidence of learning. The large decrease in extra support, increased
reference proximity and increased Dice are the meaningful improvements. The
trajectory is still improving at update 400; this bounded failure does not
prove that longer optimization could never fit the crops or that field learning
is conceptually impossible. It does not justify automatically extending the run.

Unlike the unsuccessful query pilot, this model does not reject every useful
prediction. Its mean scores on the reference tube range from **0.889 to 0.999**,
and **93.6–100%** of the labeled tube voxels exceed 0.5. It mainly fails by
predicting too much, not by missing or suppressing all traced signal.

## Per-crop final result

"Near-reference" here means predicted voxels within six acquisition voxels of
the continuous SWC segments; this is reference agreement, not biological
precision. "Volume ratio" is predicted voxel count divided by labeled tube
voxel count. All eight crops have 100% reference-sample coverage within six
voxels; that does not mean exact centerline recovery or correct connectivity.

| Neuron / crop | Near-reference predictions | Hard tube Dice | Predicted / target volume | Final overlay |
| --- | ---: | ---: | ---: | --- |
| 0 / 1 | **22.6%** | 11.4% | 16.6x | [Open](centerline_fit_results_seed0/sample_0000_patch_001.png) |
| 0 / 2 | 89.0% | 45.2% | 3.2x | [Open](centerline_fit_results_seed0/sample_0000_patch_002.png) |
| 32 / 0 | 99.8% | 58.3% | 2.4x | [Open](centerline_fit_results_seed0/sample_0032_patch_000.png) |
| 32 / 3 | 100.0% | 64.8% | 2.0x | [Open](centerline_fit_results_seed0/sample_0032_patch_003.png) |
| 96 / 0 | 87.5% | 42.4% | 3.6x | [Open](centerline_fit_results_seed0/sample_0096_patch_000.png) |
| 96 / 3 | 61.1% | 30.4% | 5.3x | [Open](centerline_fit_results_seed0/sample_0096_patch_003.png) |
| 162 / 1 | 88.1% | 36.5% | 4.6x | [Open](centerline_fit_results_seed0/sample_0162_patch_001.png) |
| 162 / 2 | 98.6% | 31.7% | 5.3x | [Open](centerline_fit_results_seed0/sample_0162_patch_002.png) |

Five crops fail the proximity criterion. No crop reaches tube Dice 0.80, though
the gate requires a mean rather than an individual Dice minimum.

All eight saved final overlays were inspected. Thin, isolated branches are
visually followed but expanded into a thick tube. On crowded images, the model
also highlights image structures without a nearby reference trace. In crop
0/1, most predicted support is elsewhere in the image. Crop 96/3 includes broad
predicted regions rather than separated reference center paths. Maximum
projections can merge structures at different depths; these are not measured
topology errors yet.

## Read-only label and loss audit

The final checkpoint was reloaded for eight inference passes, with **no
optimizer or weight changes**. Coverage, proximity and hard Dice reproduce the
saved full-precision report within 1e-5. The checkpoint SHA256 is unchanged.
The original NPZs and partition hashes were checked. The audit does not inspect
development-validation or reserved-test images.

There are two kinds of excess:

- **Width error even on simple branches.** In crop 162/2, almost every labeled
  tube voxel is accepted, and 98.6% of predictions lie within six voxels of the
  trace. But the predicted tube has 5.3 times the target volume. The known-label
  precision is only 18.8%. Extra width can explain a large Dice failure without
  establishing that the eventual extracted center path is wrong.
- **Off-reference selection on difficult crops.** Crop 0/1 has 10,335 accepted
  known-negative voxels farther than six voxels from the trace; crop 96/3 has
  13,315. These counts exclude the bright unknown regions. Consequently,
  excluded annotations are not the sole numerical explanation. Some of those
  structures may still be real but untraced neurites; the audit cannot determine
  their biological identity or reference completeness from projections.

The current negative BCE averages across all known-negative voxels, most of
which are easy empty background. A single positive voxel has **67–953 times**
the BCE coefficient of a negative voxel, depending on crop density. This is a
direct consequence of giving the two class means equal weight. It favors
recovering scarce positives; it may contribute to broad support and poor
precision. The Dice term should counter excess support, so these arithmetic
facts alone do **not** prove that BCE weighting caused the failure.

There is also substantial below-threshold background probability. For example,
crop 162/2 has about 956 units of positive probability mass and 41,563 units of
known-negative probability mass, despite a mean negative score near 0.046.
Accumulation over a large background explains why soft Dice remains poor even
when the visible 0.5 mask largely follows the branch. A hard-mask image does
not show all of the current optimization error.

An optional 3D thinning diagnostic could not run: the installed scikit-image
binary is incompatible with the active NumPy ABI. The audit records the error,
continues with voxel/loss checks, and makes **no dependency or environment
changes**. There is no thinned-path result or SWC/topology improvement to report.
This optional diagnostic issue is separate from the successfully completed fit.

## Recommended next decision — not implemented

Keep this field architecture, targets, uncertainty mask, eight crops, seed,
400-update budget, optimizer, threshold and learning gate fixed. Compare one
fresh fit that changes **only the negative BCE aggregation**: focus its mean
on the highest-loss known-negative voxels rather than dilute them among all
empty background. A concrete fixed proposal is the top `min(N, 3P)` negative
losses per crop, where P/N count known positive/negative voxels. Keep the
positive BCE and masked Dice terms unchanged.

This tests whether concentrating the existing penalty on unwanted predicted
voxels sharpens the field and removes off-reference support. It risks losing
dim branches and amplifying incomplete negative annotations, so per-crop
coverage and the unchanged unknown mask remain essential. Its intent is not
to redefine the gate or select a more flattering threshold.

Selecting difficult examples is an established optimization idea;
[the original OHEM paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.html)
studied region-based object detection. That supports the general rationale,
not this voxel-level adaptation or guaranteed neuron improvements. Whether
the proposed aggregation helps this field must be measured in the fixed control.

No loss change or follow-up training has been implemented or launched. Discuss
this control together before preparing it. Retain the existing stop rule:
validation stays paused until the fixed learning gate and inspection pass.

## Visible evidence and reproduction

- [Original fit report](centerline_fit_results_seed0/fit_report.json)
- [Verified frozen-checkpoint audit](centerline_fit_results_seed0/audit.json)
- [Original protocol and prepared image evidence](centerline_fit_ready.md)
- Audit entry script: `scripts/pointneuron2/audit_centerline_fit.py`

The user's original checkpoints, logs and arrays remain in the ignored run
directory `tmp/pointneuron2/local_runs/centerline_fit_20261012_144121_14488/model`.
Only readable reports and the final inspection images are copied into docs.

To reproduce the read-only audit into a **fresh** directory:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\audit_centerline_fit.py `
  --models .\tmp\pointneuron2\local_runs\centerline_fit_20261012_144121_14488\model `
  --manifest .\tmp\pointneuron2\local_runs\centerline_prepare_20261012\cache\manifest.json `
  --output-dir .\tmp\pointneuron2\local_runs\centerline_fit_audit_repeat
```
