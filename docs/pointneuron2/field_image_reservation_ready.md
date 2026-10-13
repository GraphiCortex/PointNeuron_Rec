# Approved image-reservation control: cached fields, unchanged weights

13 October 2026. The user approved preparing one bounded allocation comparison
after the [attention sampling review](field_attention_sampling_results.md).
The comparison reuses the **fresh ungated baseline** from the attention study,
fixed at epoch 30. No additional training, model inference, field threshold
change or architecture replacement is included. The real comparison is now
[complete and reviewed](field_image_reservation_results.md): 1m28s, zero inference
and zero updates. The fixed quota is not promoted. The commands below retain
the original protocol; no rerun is needed.

## Hypothesis and tradeoff

The union of raw foreground F and predicted field P recovers dim support, but
large off-reference additions can dilute useful image evidence. In crop 35/2,
for example, image-only coverage is 83% at N=325 while the fresh baseline's
expanded uniform control covers 47.93% at the same count. The guide/spatial
selector is weaker still there. Attention cleans one such case but loses
dim support elsewhere.

The approved hypothesis is that a fixed original-image quota can protect useful
image evidence without retraining the field. This is an allocation test, not a
promise of better inputs everywhere. A bright soma can dominate F, and reserving
more foreground points can reduce slots for field-only dim branches. We must
measure that cost, especially in neuron 94. It cannot recover structures missing
from both candidate pools or impose connectivity.

## Exact comparison

- **Control B:** the existing frozen expanded-uniform clouds and fresh-baseline
  fields from `field_attention_20261013/baseline/comparison/`. No control
  reselection or checkpoint substitution.
- **New control:** draw `floor(N/2)` points uniformly without replacement from F;
  then draw the remaining points uniformly from `(F union P) minus reserved`.
  Fill may choose more foreground points, so the quota is a **minimum**, not an
  exact final 50% foreground fraction. It differs from the existing 50/50 guided
  selector, whose context is drawn from the union without a raw-image quota.
- **Same actual count:** matched mode only, `N = min(budget, |F|)` in both arms.
  Nominal budgets 512 and 2,048; sampling seeds 0–4 and the existing
  `seed + neuron_index * 10000 + crop_index` rule. Uniform selection in both
  stages; no field-score ranking, spatial ordering or reference-based selection.
- Same native raw image, validity mask, original intensity threshold, cached
  float32 final field and .5 field-admission threshold. Source inputs are checked
  against the native cache and saved final inference fields.
- All 56 image-defined development crops from the existing seven development
  neurons; 560 paired rows, 1,120 rows including both arms. Training data and
  reserved-test structures are not scored. All development crops are included,
  including those without an in-box reference; their reference coverage and
  proximity remain unknown rather than being treated as biological negatives.
- **CPU only**, cached fields, zero model inference passes, zero optimizer
  updates. The runner does not instantiate a field model or load weights for
  execution. It verifies the trained-checkpoint hash and source reports.

The new selector is isolated in a separate module; existing A/B/C/D and
production samplers are unchanged. New clouds store native XYZ/raw intensity,
voxel indices, full-precision field scores and `reserved_foreground_mask`.
That mask marks quota provenance, not learned-guide confidence. Selection
roles remain ordinary/context in the shared metrics interface.

## Visible commands

From the real PN 2.0 working directory:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

# Optional read-only input/hash checks; does not create comparison clouds.
& .\scripts\pointneuron2\run_field_image_reservation.ps1 -Action check

# Separate visible CPU comparison, with live freeze/scoring progress.
& .\scripts\pointneuron2\run_field_image_reservation.ps1 -Action compare
```

No timing/throwaway optimizer preflight is needed because no model runs. Expect
a few minutes of CPU scoring and plotting; the completed local command took
1m28s. Progress prints each of the 56 freeze steps, then each
of the 56 scoring steps. There is no automatic next experiment or retry.

Default outputs are in:

`C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\field_image_reservation_20261013`

- `clouds/`: only newly selected image-reservation clouds; control clouds are
  reused from their existing location and hashes are protected.
- `measurements/`: recomputed control and reservation coverage/proximity,
  critical-point support, per-chain support and longest unsupported runs.
- `inspection/`: paired XY/XZ projections at nominal 2,048, seed 0. Green marks
  the reserved image quota, orange union fill, cyan reference. Projections are
  not connections.
- `comparison.json` and `metrics.csv`: per-crop/seed measurements, macro
  summaries, paired deltas and source hashes. Positive coverage/proximity deltas
  improve those measures; negative longest-gap deltas improve gap length.

The console log is saved beside that directory as
`field_image_reservation_20261013.console.log`. Existing output directories are
refused; use a fresh `-ExperimentDirectory` for any deliberate later rerun.
`-SourceExperiment`, `-NativeExperiment`, `-ArtifactRoot` and `-Python` allow
explicit source/runtime paths without modifying the recipe.

## What we will assess together

First verify execution and source preservation. Then assess mean and
per-neuron/crop coverage and proximity, with actual counts equal. Specifically:

- Does reservation rescue 35/0 and 35/2 dilution relative to the saved baseline B?
- Does it sacrifice neuron-94 dim support or other neurons' coverage?
- Does it improve the same neuron-79 reference-chain gaps, rather than merely
  raising an average? Per-chain identities and lengths are retained in the
  measurement files; longest-gap summaries alone do not identify the same chain.
- Are critical points supported? None of these geometric measurements establishes
  final SWC topology or biological precision with incomplete annotations.

This is one fixed control, not a quota sweep. A gain in selected cases alone
does not authorize promotion. After reviewing this comparison we decide which
input protocol, if any, is suitable for the next intended PointNeXt stage.
The PN 2.0 architecture remains point-cloud preparation → PointNeXt →
Transformer skeleton proposals → GATv2 connectivity → SWC.

## Source and verification

- [Isolated reservation selector](../../src/pointneuron/data/field_image_reservation.py)
- [Fixed recipe](../../configs/pointneuron2/field_image_reservation.json)
- [Cached-field comparison runner](../../scripts/pointneuron2/compare_field_image_reservation.py)
- [PowerShell entry point](../../scripts/pointneuron2/run_field_image_reservation.ps1)
- [Quota and cached-only contract tests](../../tests/test_field_image_reservation.py)

The runner verifies source fields, native inputs, checkpoint/report hashes,
development identities and saved counts before outputs. It freezes every new
cloud before loading reference geometry, recomputes the existing control's
geometric measurements, validates quota provenance and checks protected hashes
again before declaring completion. Failed outputs/logs are retained.

Verification: **27 targeted tests passed in 51.30 seconds**, covering the quota
selector, the existing field sampling/geometry contracts and a complete tiny
cached-only paired run. Tests reject changed fields/control clouds, changed
quota settings and reserved-test identities before creating outputs; a patched
model constructor confirms the paired runner does not instantiate a model.
The read-only check on the actual local sources also passed: seven development
neurons, 56 crops, 560 matched pairs, zero inference and zero updates. No real
reservation comparison clouds or quality measurements were produced by that check.
