# Approved attention comparison: fresh baseline versus two skip gates

13 October 2026. The user approved testing two additive attention gates inside
the Phase 1 field U-Net and chose **a fresh baseline replay alongside the gated
model**. The runner is prepared; no real attention preflight or training has run.
Point-cloud preparation -> PointNeXt -> Transformer proposals -> GATv2 -> SWC
remains the intended architecture. The image-backbone selector change is deferred:
it is not mixed into this architecture comparison.

The hypothesis is that decoder context can filter fine skip features to reduce
off-reference field support without losing dim branches. This is inspired by
[Attention U-Net](https://arxiv.org/abs/1804.03999); the published medical-imaging
results are not evidence it will improve these neuron data. Our experiment uses
the existing small field backbone, not a reproduction of the full published model.
These are feature gates, not pairwise Transformer self-attention, global path
reasoning or an enforced connectivity constraint.

## Agreed comparison

- Same native cache: 262 training crops from 33 neurons, **211 supervised**;
  56 development crops from seven neurons, **47 supervised**. Unknown/padding
  masks and entirely unsupervised crops retain the existing rules. Reserved
  test neurons are excluded.
- **Baseline:** unchanged U-Net forward computation, 88,673 parameters.
- **Attention:** same backbone plus two skip gates, **89,167 parameters**:
  494 additional parameters. Both skip paths are gated before concatenation
  with the corresponding upsampled decoder features.
- Each gate projects skip and decoder features with 1x1x1 convolutions, adds
  them, applies ReLU and a learned scalar sigmoid per voxel, then multiplies
  the skip features. There is no new BatchNorm, dropout or auxiliary loss.
- The final gate projection starts with zero weights and bias logit(0.99).
  The initial coefficients are therefore 0.99 everywhere: close to passing
  features through, with a nonzero derivative. They are learned afterwards.
  Initial predictions need not match the ungated model because attenuation is
  not exactly the identity. The gate is not a calibrated neuron probability.
- Both arms load the **same exact shared tensors** from the saved pilot's
  `field_epoch_000.pt`. The attention arm additionally initializes its gates;
  neither arm starts from the already-trained epoch-30 weights.
- Same 30 epochs, batch 1, float32, seed 0, AdamW LR .001, weight decay .0001,
  gradient clip 10, no augmentation, and existing balanced BCE/top min(N,3P)
  negative BCE plus masked Dice. **6,330 updates per model, 12,660 total**.
- Same crop order in both fresh fits, checked against the recorded original
  pilot order. Save train/development checks and checkpoints at 0/5/15/30.
  Train the two arms sequentially on the local RTX 3070 Ti Laptop GPU.
- A separate frozen comparison uses fixed epoch 30 and the unchanged four
  sampling controls A/B/C/D, budgets 512/2,048, seeds 0-4, field threshold .5,
  cell size 8 and guide fraction .5. No selector changes or threshold sweep.

The generic field runners now accept explicit model factories and record model
identity in new reports/checkpoints. Default baseline computation is retained.
An ordinary baseline comparison refuses to silently load a gated checkpoint.
Previous caches, original reports, checkpoints and clouds are preserved; current
source files have evolved after the completed pilot, so their historical source
hashes describe that earlier snapshot. Fresh timing checks bind the new source.

## What would support the hypothesis

Evaluate the field and final sampled inputs together. Report per-neuron and
per-crop changes, especially the known neuron-35 dilution, neuron-94 dim support,
neuron-44 omissions and neuron-79 branch gaps. Higher mean Dice alone is
insufficient. Attention should reduce off-reference support without sacrificing
the dim structures or long reference chains we want to retain.

The fresh baseline is the primary architecture comparator. Its difference from
the saved baseline is reported separately, because repeated training has not
been established as numerically identical. One paired seed is preliminary:
matched initial weights and crop order do not prove attention-specific causality
or commercial readiness. Extra parameters are part of this architecture change;
an equal-capacity convolutional control is not included. No automatic adoption
rule, checkpoint winner selection or later-stage bypass is implemented.

Matched sampling groups have equal actual N between the two models. Expanded
groups may differ in actual available N if their predicted candidate pools
differ; those rows are flagged instead of treated as pure equal-count gains.
The image-only reference cloud must contain identical points in both comparisons.

## Run now: timing only, with no cache preparation

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_field_attention.ps1 -Action preflight
```

This reuses the existing verified native cache and saved initial checkpoint.
It does **four throwaway updates per model**, eight total, retaining no trained
checkpoint. It prints parameter counts and a total duration estimate for both
fits. Share the timing output before launching training visibly.

The previous baseline fit took 17m14s. Two fits would take roughly twice that
before gate overhead and runtime variation, but the new preflight will measure
the current setup; its estimate is not a guaranteed duration.

All outputs use a separate ignored directory:

`C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\field_attention_20261013`

Pass the same `-ExperimentDirectory` to all actions if using another fresh
directory. Both arms' existing output folders and completed/failed status files
are preserved. There is no automatic restart, resume or environment installation.

## Subsequent visible commands

After reviewing the timing estimate, launch the two agreed fits:

```powershell
& .\scripts\pointneuron2\run_field_attention.ps1 -Action train
```

Both timing contracts are checked before the first fit starts. The baseline
fit runs first, then attention, with live progress. Outputs include:

- `baseline/model/` and `attention/model/`: reports, checkpoints, final float32
  development fields and overlays.
- `training_comparison.json`: paired execution checks and final field metrics,
  including fresh-baseline versus saved-baseline differences.
- `train.console.log`: visible output from both models.

Review training/development results before the separate sampling action:

```powershell
& .\scripts\pointneuron2\run_field_attention.ps1 -Action compare
```

It runs the unchanged four controls independently on both final models, with
no optimizer updates. `baseline/comparison/` and `attention/comparison/` retain
all clouds/chain measurements/overlays; `sampling_comparison.json` records the
between-model metric deltas and flags availability differences. No SWC or
PointNeXt/Transformer/GATv2 training starts.

## Visible implementation and verification

- [Attention module](../../src/pointneuron/models/centerline_attention.py)
- [Fixed attention recipe](../../configs/pointneuron2/field_attention.json)
- [Two-arm orchestrator](../../scripts/pointneuron2/field_attention_experiment.py)
- [PowerShell entry point](../../scripts/pointneuron2/run_field_attention.ps1)
- [Gate and paired-run tests](../../tests/test_field_attention.py)
- [Prior development sampling findings](field_pilot_sampling_results.md)

Tests cover gate initialization/context gradients, exact shared tensors, odd
spatial dimensions, strict reload, explicit architecture loading, a tiny CPU
two-arm train/sampling path, baseline preservation and unequal availability
flags. These are execution checks, not measured real-neuron improvements.

Verification: **43 targeted tests passed in 76.13 seconds**, covering the new
attention experiment and existing field cache/loss/sampling contracts. The real
318-record cache and original checkpoint contracts also pass; shared backbone
tensors match exactly. That read-only check performed no model inference,
optimizer update or GPU training. A final reporting-label check is included in
the paired end-to-end test.
