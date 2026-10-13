# LR-only replay: 17/18 matches accepted, no extras, one confidence miss

14 October 2026. The user completed the agreed fresh LR .001 confidence replay
in 14s observed wall time (8.36s recorded fit time). Initial paired metrics,
source/cache checks and strict reload passed. The **only recipe change was
LR .0001 → .001**. Geometry, radii, features, assignments and baseline outputs
remain unchanged. The final combined gate still fails because crop 0 rejects
one of its six assigned positive queries.

## Paired result at 2,000 updates

| Crop | LR .0001 accepted matches / extras | LR .001 accepted matches / extras | Accepted curve coverage within 6, baseline → replay |
| ---: | --- | --- | --- |
| 0 | 3/6 / 2 | **5/6 / 0** | 54.69% → **96.10%** |
| 1 | 12/12 / 0 | **12/12 / 0** | 100% → **100%** |

The paired weighted confidence loss falls .15875 → .05047, approximately
68% lower. Crop 0's classification loss falls .15552 → .04994 and crop 1's
.00324 → .00054. Distinct-match placement remains .432/.216 mean and
.568/.369 maximum error in native voxels; only confidence changed.

At update 2,000 both crops meet the curve-coverage criterion and have no
unmatched accepted queries. Crop 1 passes all criteria. Crop 0 fails only
the requirement to accept **every** assigned positive at cutoff .5. The gate
was not relaxed because coverage alone looks good.

## The remaining miss and score separation

The rejected positive is **crop 0, query 408**. It is an existing fixed match,
not a newly missing position. Its trajectory is:

| Update | Query 408 positive score | Highest crop-0 unmatched score | Crop-0 accepted matches / extras |
| ---: | ---: | ---: | --- |
| 0 | .01808 | .71557 | 2/6 / 3 |
| 100 | .01900 | .69073 | 3/6 / 3 |
| 500 | .06128 | .58027 | 4/6 / 1 |
| 1,000 | .15236 | .50732 | 4/6 / 1 |
| 2,000 | **.41525** | **.44683** | **5/6 / 0** |

The positive score is still rising and negative confidence falling. The paired
loss decreases .09240 → .05050 between logged updates 1,000 and 2,000; there
is no demonstrated plateau at the declared stop. A threshold change cannot
cleanly select all positives and reject all negatives in crop 0 at this
snapshot: its weakest positive is below its strongest negative.

## Conceptual conclusion

The higher learning rate improves acceptance and suppresses extras without
changing representation or geometry. This supports an optimization bottleneck
on the fixed two-crop task. The earlier
[linear-capacity oracle](query_confidence_results.md) already verified that a
shared linear boundary can separate all cached labels, so this near miss is
not evidence that the current feature/head family cannot represent them.

This remains a frozen-feature diagnostic. It does not establish that joint
PointNeXt/Transformer fitting will maintain stable assignments, that per-crop
normalization generalizes, or that the intended GATv2/full-neuron reconstruction
is ready. A later all-target pass would still be a learnability milestone,
not commercial or state-of-the-art performance evidence.

## Proposed duration-only next step

Prepare one fresh **4,000-update replay at LR .001**, keeping the same cached
data, source initialization, weighted objective, AdamW settings, cutoff and
fixed geometry/labels. Save the original overlapping checks and additional
2,500/3,000/4,000 checks. Require its first 2,000 updates to agree with the
completed replay before interpreting the longer budget.

The saved classifier checkpoints contain weights and provenance but **no
optimizer state**. Reloading update 2,000 with a new AdamW would reset moments
and would not be an unchanged continuation. A fresh replay with verified
overlapping predictions/parameters makes the duration comparison explicit.
The observed 8.36s fit suggests roughly twice the compute for 4,000 updates;
measure GPU timing before a separate visible launch. Stop at the new budget,
keep both crops and the original gate, and inspect the whole score separation.

This proposal is not implemented or launched during this results review.
No classifier architecture, assignment policy, threshold or full-model training
strategy has changed. If the longer fit passes, discuss the next joint-training
consistency check before broadening the data or fitting the graph stage.

## Evidence and preservation checks

- [Paired comparison](../../tmp/pointneuron2/local_runs/query_confidence_lr_20261014/comparison.json)
- [Replay fit report](../../tmp/pointneuron2/local_runs/query_confidence_lr_20261014/model/fit_report.json)
- [Crop 0 final scores](../../tmp/pointneuron2/local_runs/query_confidence_lr_20261014/model/step_2000_crop_0_scores.csv)
- [Preserved baseline](../../tmp/pointneuron2/local_runs/query_confidence_20261013/model/fit_report.json)

The review rechecked source-file, final-checkpoint, baseline-report and
feature-manifest hashes against the saved fit: all unchanged. The paired report
records successful checks, the fit reports frozen cache/baseline preservation,
and reload maximum logit delta is 0.0. The review used file reads only: no
additional model forwards, optimizer updates or GPU runs. Raw artifacts remain
ignored; this readable review remains in the real PN2 source folder.
