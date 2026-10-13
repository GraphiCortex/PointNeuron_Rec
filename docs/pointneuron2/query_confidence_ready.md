# Fixed-placement confidence diagnostic: ready for visible GPU timing and fit

13 October 2026. The user agreed to the confidence-only diagnostic following
the [normalization and query-score audit](pointnext_cropnorm_results.md).
**Completion update:** GPU timing and the bounded fit are complete. Crop 1
passes and crop 0 fails; a subsequent linear oracle verifies capacity for both.
See the [results and proposed optimization control](query_confidence_results.md).
The remaining text preserves the prepared protocol.

The implementation and immutable CPU feature cache remain available alongside
the completed run. Preparation itself made no optimizer updates.

## What question this tests

Can the existing frozen features distinguish which queries are positives in
each of the two fitted training crops, when placement and assignments stop
changing? The earlier model placed matched targets within one voxel, but missed
four crop-0 targets and accepted queries belonging to the other crop. Fixing
the features/labels isolates the last confidence projection from that moving
geometry and assignment process.

This retains Phase 1, PointNeXt, Transformer proposals and intended later GATv2.
It does not add a new architecture or change the sampler, targets or threshold.
It is a two-training-crop capacity/optimization diagnostic, not generalization
or final reconstruction evaluation.

## Precisely what trains

The source is the explicit per-crop-normalization candidate at original update
2,000. Each source crop is evaluated separately at batch one. A hook captures
the **128-channel activation immediately before the final six-output linear
layer**, after the existing hidden Linear/ReLU. These are the exact features
used by the existing confidence output, not raw encoder or decoder features.

All PointNeXt/Transformer weights, the hidden prediction layer, coordinates,
radii and reference assignments are frozen. Only a copy of the last linear
layer's **first two output rows and biases** trains: 2 × 128 + 2 = **258
parameters**. Its weights start from the saved confidence projection. Separate
confidence-only checkpoints cannot be mistaken for full PN2 models.

The immutable cache stores both crops' detached features, original logits,
geometry/radii, initial confidence weights and fixed labels. The original
one-to-one query assignments become the positive labels (6 and 12); other
queries retain the existing unmatched-negative training rule. Those labels
never enter the source model forward passes and matching is not recomputed
during the confidence fit.

## Fixed recipe and limits

| Setting | Value |
| --- | --- |
| Training data | Neuron 0, existing crops 0/1; no development/test fitting |
| Source input counts | 848 / 2,048 image points, unchanged |
| Cached classifier inputs | Two independent crops × 512 queries × 128 features |
| Optimizer | Fresh AdamW; LR .0001, weight decay .0001, gradient clip 10 |
| Objective | Confidence weight 2 × mean of the two per-crop weighted cross-entropies |
| Positive-class weight | 8, unchanged |
| Updates | 2,000, fixed; no scheduler or augmentation |
| Saved checks | 0 / 100 / 500 / 1,000 / 2,000 |
| Acceptance cutoff | .5, unchanged |

Both cached crops contribute to every update, removing alternating-crop order
from this diagnostic. Their weighted losses are averaged equally, matching the
existing per-crop confidence-loss definition; this is not one pooled weighted
cross-entropy with a different denominator. The encoder is not run at batch
two: its features were computed independently at batch one before caching.
The optimizer is newly initialized; this is not a continuation of the original
full-model optimizer trajectory.

The frozen-label gate requires all 6/12 positives accepted, zero unmatched
accepted queries, fixed distinct-match errors no greater than one voxel, and
at least 95% accepted-node curve coverage within six voxels. Radii stay finite
and positive in the protected cache. Reports include each crop's loss, lowest
matched score, highest unmatched score and curve support. Original geometry,
features and labels must remain bitwise unchanged and strict head reload must
agree within 1e-5 logits. Execution checks are separate from quality gates.

Stop at the declared budget. A pass shows that this fixed feature/head pair
can fit these labels; it supports investigating joint-training assignment or
optimization behavior, but does not uniquely prove its cause. A failure at
this recipe/budget does not prove the features inherently cannot separate the
labels: inspect loss progress and score margins before proposing a new head,
loss or extension. Neither result promotes a full model automatically.

## Preparation completed here

The real cache is prepared at:
`C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\query_confidence_20261013\cache`.

Preparation ran two frozen **CPU** model forwards and zero optimizer updates.
Replayed confidence logits differ from the source by at most **4.77e-7**.
The fixed assignments reproduce the saved control's 6/12 matches exactly.
Protected source checkpoints, point cache and code hashes remain unchanged.
No field inference, new point selection or GPU work occurred.

Three synthetic tests passed: exact feature/logit capture, confidence-head
fitting/reload while other parameters stay frozen, and the complete synthetic
timing/2,000-update runner with preserved cache and outputs. The synthetic fit
is test data, not a research result or the prepared real-data fit.

The [preparation/timing/fit code](../../scripts/pointneuron2/query_confidence_experiment.py),
[head](../../src/pointneuron/models/frozen_query_confidence.py),
[recipe](../../configs/pointneuron2/query_confidence.json) and
[PowerShell entry point](../../scripts/pointneuron2/run_query_confidence.ps1)
are visible source files. The generated cache and future outputs are ignored.

## Visible commands for this completed protocol

First measure the real GPU runtime:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_query_confidence.ps1 -Action preflight
```

This uses the existing environment and RTX 3070 Ti Laptop GPU for four throwaway
confidence updates, two warm-up and two measured. It checks reload and frozen
cache integrity, times one evaluation and prints a 2,000-update estimate.
No fitted checkpoint is retained. The completed GPU preflight estimated 12.2s
of fit/evaluation compute, with a 12.2–28.2s planning range plus Python startup.
The actual fit recorded 8.25s; user-observed wall time was 14s.

The separate visible fit command was:

```powershell
& .\scripts\pointneuron2\run_query_confidence.ps1 -Action train
```

It prints paired loss every 100 updates and per-crop quality at each saved
check. Outputs go to `query_confidence_20261013/model/fit_report.json`, five
confidence-head checkpoints and ten score CSVs. Original full-model checkpoints
remain untouched. The runner refuses changed source/cache/backend hashes and
preserves existing/failed outputs. It does not retry or start another stage.

Preparation is already complete; there is no need to rerun `-Action prepare`.
The confidence fit uses cached features and performs **zero PointNeXt or
Transformer forwards**. There is no automatic full-model weight replacement.
