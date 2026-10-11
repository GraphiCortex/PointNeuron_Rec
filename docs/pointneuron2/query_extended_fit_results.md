# Completed 2,000-update fit: confidence learns, pooled normalization limits inference

12 October 2026. The user completed the approved local GPU run in **2 minutes
25 seconds**. Both models received 2,000 fresh updates on the same two annotated
training crops from neuron 0. Dropout remained zero. Pairing and exact final
checkpoint reloads passed. The saved 500-update geometry and every one of the
first 500 training losses reproduce the previous dropout-zero run exactly.

**The query model now learns useful confidence on these crops. Its remaining
large placement gap is localized to the pooled encoder BatchNorm mode.**
This is fitted-crop evidence, not validation or a whole-neuron reconstruction
comparison. No model, graph rule or training setting changed during inspection.

## Saved inference trajectory

Coverage is the equal-weight average across the two crops, using tolerance six
voxels against their sparse curve targets. Each query model still outputs 512
slots per crop; accepted counts refer to the unchanged score threshold of 0.5.

| Query updates | Mean target distance, voxels | All-slot coverage | Score-filtered coverage | Accepted slots, crops 0 / 1 |
| --- | ---: | ---: | ---: | ---: |
| 500 | 2.68 | 95.8% | 29.2% | 1 / 15 |
| 1,000 | 3.46 | 95.8% | 75.0% | 9 / 12 |
| 2,000 | 4.56 | 91.7% | 91.7% | 8 / 12 |

The query evaluation objectness loss falls from 0.1933 to 0.0816 to 0.0184.
Training center loss also falls: means over the last 50 updates at these checks
are 0.07134, 0.02793 and 0.01220. In contrast, evaluation center loss grows from
1.2162 to 2.0557 to 2.7675. More updates therefore improve classification but
do not close the difference between training and inference placement.

At 2,000 updates, ordinary inference covers five of six targets in crop 0 and
all 12 in crop 1. Two of crop 0's eight accepted slots lie farther than six
voxels from every sparse target. These are off-target according to this
reference proxy; their existence alone does not establish false neurites in a
full image. In crop 1, all 12 accepted slots have distinct nearby targets.

The matched pointwise arm stays at 79.2% target coverage after filtering at all
three checks and retains 484 / 512 slots. Its final mean target distance is
2.88 voxels. This is the local comparator, not a PN1 whole-neuron benchmark;
raw coverage and accepted counts are not a complete model ranking.

## Fixed-weight normalization audit

The existing [audit script](../../scripts/pointneuron2/audit_query_fit.py) ran
bounded forward passes on fresh copies of the saved models, on the same CUDA
device. It performed no optimizer updates. Original checkpoint hashes are
unchanged. Ordinary two-crop inference reproduces the recorded final metrics.

| Query mode, same saved weights | Mean target distance, voxels | Score-filtered coverage | Accepted slots, crops 0 / 1 |
| --- | ---: | ---: | ---: |
| Normal inference, two crops | 4.56 | 91.7% | 8 / 12 |
| Normal inference, one crop at a time | 4.54 | 91.7% | 8 / 12 |
| Only pooled encoder BatchNorm uses batch statistics | 0.26 | 100% | 6 / 12 |
| All encoder BatchNorm layers use batch statistics | about 0.25 | 100% | 6 / 12 |

The pooled-only control isolates one layer: `encoder.global_mlp[1]` in
[dgcnn.py](../../src/pointneuron/models/dgcnn.py). All other layers keep ordinary
inference behavior. It recovers a distinct accepted proposal for each of the
18 targets. Direct float64 Euclidean matching confirms every assigned error is
below **0.36 voxels**. There are no extra accepted proposals in this control.
The recorded small-distance `torch.cdist` metrics have minor floating-point
cancellation; the direct-distance verification reaches the same conclusion.

This intervention demonstrates that pooled normalization mode accounts for
most of the saved model's placement gap on these fitted crops. It does not
establish performance on unseen crops. The weights have learned a close fit
under the training normalization conditions, including meaningful scores;
ordinary inference changes the features that those weights receive.

After max pooling, this BatchNorm sees only **two crop vectors**, rather than
hundreds of points per crop. PyTorch normalizes training batches using current
batch statistics and evaluates using saved running statistics. It also uses
different variance estimators for the training forward and the saved variance.
Those definitions help explain why a very small pooled batch is vulnerable to
a scale mismatch; this audit does not separately quantify the contribution of
each estimator or running-statistic lag.
[PyTorch 2.12 BatchNorm documentation](https://docs.pytorch.org/docs/2.12/generated/torch.nn.BatchNorm1d.html).

Using training batch statistics during production inference is not a proposed
fix. It couples predictions to companion crops, and the pooled layer cannot
use training BatchNorm with a single crop. The ordinary single-crop control
above also shows that changing inference batching alone does not restore the fit.

## Approved next comparison

Compare one fresh **2,000-update query fit with only the pooled BatchNorm
replaced by LayerNorm** against this saved query baseline. Keep dropout zero,
the same two crops and selected inputs, all unchanged-layer initial weights,
loss, learning rate, remaining encoder/decoder layers and score threshold.
Retain checks at 500/1,000/2,000 and ordinary single-crop inference verification.
Estimated duration is roughly **1–2 minutes** on the user's GPU.

LayerNorm over the pooled feature vector uses each crop's feature statistics
in both training and evaluation, removing this layer's cross-crop dependence
and running-statistic switch. It changes the feature distribution, so a fresh
fit is required; the fixed-weight diagnostic does not prove that replacement
will succeed. Other BatchNorm layers remain a possible residual source of error.
[PyTorch 2.12 LayerNorm documentation](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LayerNorm.html).

Judge the control by ordinary inference placement, score-filtered target support,
distinct matched proposals and behavior when evaluated singly versus together.
If it preserves the close training fit during inference, the next decision is
the broader train/validation pilot. That pilot and whole-neuron reconstruction
are not launched here. The user approved the normalization control; its source,
completed execution checks and visible command are in
[the LayerNorm preparation protocol](query_layernorm_ready.md). The GPU fit has
not been launched by the assistant.

## Evidence

[The verified machine-readable summary](query_extended_fit_seed0/summary.json)
records trajectory metrics, per-crop distances, confidence, matching checks,
source hashes and file-hash verification. Original local artifacts remain at:

```text
User run: tmp/pointneuron2/local_runs/query_smoke_20261012_005258_4464/models/
Saved-array inspection: tmp/pointneuron2/local_runs/query_extended_fit_inspection_20261012/
Forward-only audit: tmp/pointneuron2/local_runs/query_extended_fit_normalization_audit_20261012/
```

Model source, cache, targets, thresholds and reserved test neurons remain
unchanged. No new tests were needed for this results-only report; the preceding
runner preparation passed 31 targeted tests.
