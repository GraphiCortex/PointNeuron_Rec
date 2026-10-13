# PointNeXt/query fit: normalization discrepancy and unresolved confidence

13 October 2026. The user completed the agreed fresh 2,000-update fit on the
RTX 3070 Ti Laptop GPU in 9m10s. **Subsequent update:** the user agreed to the
[explicit per-crop normalization control and saved-score audit](pointnext_cropnorm_results.md).
It is now complete with zero updates; confidence remains unresolved. The
sections below retain the original fit diagnosis and its proposed next step.

The original execution and checkpoint reload checks
passed (reload maximum delta 0.0), but **the learnability gate failed on both
fixed training crops**. No development/test fit, graph reconstruction or
production promotion occurred.

## Ordinary inference over the fit

These values come from the saved GPU `fit_report.json`. Distances are native
voxels, using distinct Hungarian matches. Accepted curve coverage uses the
unchanged score cutoff .5 and six-voxel tolerance.

| Update | Crop | Mean matched distance | Maximum matched distance | Accepted queries | Matched accepted within 6 | Unmatched accepted | Accepted curve coverage within 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 62.63 | 74.90 | 0 | 0/6 | 0 | 0% |
| 0 | 1 | 42.69 | 57.42 | 0 | 0/12 | 0 | 0% |
| 500 | 0 | 5.00 | 7.36 | 0 | 0/6 | 0 | 0% |
| 500 | 1 | 12.57 | 17.68 | 4 | 0/12 | 0 | 0% |
| 1,000 | 0 | 3.12 | 6.41 | 2 | 0/6 | 2 | 0% |
| 1,000 | 1 | 13.88 | 30.11 | 6 | 0/12 | 0 | 4.86% |
| 2,000 | 0 | 1.93 | 3.56 | 4 | 2/6 | 2 | 42.58% |
| 2,000 | 1 | 12.82 | 16.31 | 10 | 0/12 | 1 | 9.05% |

Nearest-node coverage can differ from distinct assignment support: an accepted
query can be near a target without being its assigned query. Neither metric
alone establishes correct geometry/confidence for all reference nodes.

The last logged training loss was .09488, but the 50-step logging cadence
always prints an even update: **crop 1 only** in this alternating fit. It is
not an aggregate training loss or evidence that crop 0 has converged.

## Saved-weight diagnostic, with no retraining

The visible [CPU audit](../../scripts/pointneuron2/audit_pointnext_inference.py)
loaded the final checkpoint and the same immutable cache. On separate disposable
model copies, it compared ordinary `eval()` with only the 16 encoder BatchNorm
layers using current-crop statistics. All learned parameters remained identical;
only the disposable copies' normalization buffers could change. Source, cache,
fit-report and checkpoint hashes were checked before and after. No optimizer,
GPU run, checkpoint write or deployment change occurred.

| CPU inference rule | Crop | Loss | Mean / maximum matched distance | Accepted queries | Matched accepted within 6 | Unmatched accepted | Accepted curve coverage within 6 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Ordinary eval | 0 | 2.2477 | 1.923 / 3.562 | 4 | 2/6 | 2 | 42.58% |
| Ordinary eval | 1 | 32.6649 | 12.839 / 16.315 | 10 | 0/12 | 1 | 9.05% |
| Current-crop BN diagnostic | 0 | .4632 | .432 / .568 | 5 | 2/6 | 3 | 33.15% |
| Current-crop BN diagnostic | 1 | .0493 | .216 / .369 | 14 | 12/12 | 2 | 100% |

Small CPU/GPU numerical differences exist; the table compares both normalization
rules on CPU. **Both diagnostic crops still fail the complete gate.**

## Conceptual conclusion

The same PointNeXt/Transformer weights can place distinct proposals within
one voxel of every target in these two fitted crops under training-style
normalization. That is evidence of placement learnability; it does not show
generalization or a working final reconstruction system.

The large deterioration when switching to running-statistics inference strongly
identifies normalization behavior as a major placement bottleneck. Batch-one
updates alternate two clouds with different counts and feature distributions;
running statistics aggregate those populations while training uses each crop's
current statistics. This is the likely mechanism, not proof that a particular
single layer or point operator is defective. The audit changed all BN modes
together and has not established the best production normalization rule.

Confidence is a separate remaining problem. With placement restored, crop 0
still rejects four matched targets and accepts three unmatched queries; crop 1
accepts all targets but adds two unmatched queries. Lowering the score threshold
would not resolve the geometry shift or reliably remove extras. No threshold,
target set, loss, sampler or gate was altered for this diagnosis.

## Proposed next step for discussion

Prepare a **normalization-only consistency control** before another training
extension. At batch one, using per-crop statistics in both train and eval would
remove this particular running-statistics switch while retaining the layers,
learned affine parameters, point operators, inputs, decoder and loss. This is
an explicit deviation from reference running-statistics inference, so it needs
to be agreed and versioned rather than silently adopted. Later multi-crop
batching would require a separate normalization contract.

First assess geometry/confidence with the saved weights under that explicit
policy; the current diagnostic already warns that it is insufficient for the
full gate. Inspect matched versus unmatched query scores/identities before
selecting a bounded confidence control or deciding whether more updates are
warranted. Any new fit needs per-crop loss reporting. No larger model, new
sampler, architecture replacement or additional fit is queued automatically.

Local raw results:

- [GPU fit report](../../tmp/pointneuron2/local_runs/pointnext_query_20261013/model/fit_report.json)
- [CPU normalization audit](../../tmp/pointneuron2/local_runs/pointnext_query_20261013/diagnostics/inference_normalization.json)
- [Immutable input manifest](../../tmp/pointneuron2/local_runs/pointnext_query_20261013/cache/manifest.json)

The generated artifacts remain ignored. The code and this readable report are
kept in the PN2 source folder and development history.
