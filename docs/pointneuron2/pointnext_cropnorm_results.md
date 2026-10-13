# Per-crop normalization control and saved-query confidence audit

13 October 2026. The user agreed to prepare the normalization consistency
control and inspect saved scores before another fit. **Subsequent update:** the
user agreed to the [fixed-placement confidence diagnostic](query_confidence_ready.md).
Its source/CPU feature cache are now prepared; the visible GPU timing and fit
remain pending. The text below preserves the normalization-control findings.

The normalization control and saved-score audit are both complete.
The comparison used **CPU inference only, zero optimizer updates and zero field
inference**. Original checkpoints, cache, training scripts and source hashes
remain unchanged. The complete learnability gate still fails.

## Explicit model control

The [new model](../../src/pointneuron/models/pointnext_query_cropnorm.py) uses
the same sixteen BatchNorm layers and learned affine parameters, with
`track_running_stats=False` and no running-statistic buffers. At batch one,
train and eval consequently use current-crop statistics. This deliberately
departs from reference running-statistics inference; it is a research control,
not a production policy or benchmark reproduction. Batch sizes other than one
are rejected to prevent mixing crops under the declared contract.

Conversion strictly preserves every learned parameter and discards only the
48 named BN buffers. The candidate checkpoint has its own schema,
`pointneuron2-pointnext-query-cropnorm-v1`, strict loader and original-checkpoint
hash provenance. The original model and loader remain unchanged and reject
this new identity. No sampler, target, loss, score cutoff, decoder structure or
point operator changed.

Two new synthetic tests pass at N=256/358, covering parameter preservation,
equivalence to the earlier current-crop BN diagnostic, train/eval consistency,
reload, crop-order independence, batch-size rejection and schema separation.
The real saved-weight comparison additionally measured:

| Check | Maximum absolute difference |
| --- | ---: |
| Train/eval predictions, fixed inputs/parameters | 0.00006104 |
| Disk checkpoint reload | 0.0 |
| Reverse crop order / repeated inference | 0.0 |

The comparison checks native centers, confidence logits and radii against
declared float32 tolerances (train/eval 1e-4; reload/order 1e-5).

## Placement and confidence across saved snapshots

All rows use the new explicit policy, the original immutable two training
crops and the existing score cutoff .5. Distances refer to distinct matches.

| Source update | Crop | Mean / maximum matched distance | Matched accepted within 6 | Unmatched accepted | Matched-slot changes from previous snapshot | Accepted curve coverage within 6 |
| ---: | ---: | --- | --- | ---: | --- | ---: |
| 500 | 0 | .866 / 1.215 | 0/6 | 0 | — | 0% |
| 500 | 1 | .646 / 1.028 | 7/12 | 4 | — | 60.08% |
| 1,000 | 0 | .472 / .695 | 1/6 | 3 | 5/6 | 15.20% |
| 1,000 | 1 | .201 / .302 | 9/12 | 6 | 4/12 | 76.04% |
| 2,000 | 0 | .432 / .568 | 2/6 | 3 | 5/6 | 33.15% |
| 2,000 | 1 | .216 / .369 | 12/12 | 2 | 0/12 | 100% |

This preserves the earlier diagnostic's subvoxel placement at update 2,000 in
ordinary `eval()`. It does not solve confidence. Snapshot changes do not measure
assignment churn at every training update; that finer trajectory was not logged.

## What the score audit reveals

**All five unmatched accepted queries at update 2,000 are matched positives
in the other crop.** The association is exact by query index:

| Crop where unmatched | Query | Score here | Score as a matched positive in other crop | Distance to reference curve here |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 124 | .595 | .997 | 3.98 |
| 0 | 217 | .716 | .995 | 2.43 |
| 0 | 404 | .657 | .995 | .83 |
| 1 | 308 | .709 | .664 | 19.68 |
| 1 | 436 | .750 | .836 | 42.93 |

Unmatched does not always mean off-curve: query 404 on crop 0 is within one
voxel of the reference curve but is not an assigned target. Conversely, the two
crop-1 extras are far from the reference, so this is not exclusively duplicate
suppression. Scores for crop 0's four rejected matches are .064/.240/.462/.018,
despite placement errors of only .36/.50/.57/.36 voxels. A threshold reduction
would retain high-scoring unwanted queries as well as some missed targets.

These observations support the hypothesis that confidence is partly carrying
active query identities between crops rather than fully conditioning acceptance
on the current crop. Crop 0's changing assignments may also give its classifier
unstable labels. This is evidence of those failure patterns, not a causal proof
that a particular head or loss must be replaced, or that PointNeXt cannot learn
the task.

## Next comparison proposed for discussion

The next focused diagnostic should test **confidence with placement fixed**:
freeze the encoder, Transformer and geometry/radius predictions; fix each crop's
existing one-to-one assignments; assess whether the confidence output can learn
the resulting crop-specific labels. This separates feature/classifier capacity
from changing assignments and moving geometry. It would be a new training
comparison and needs an agreed budget, optimization recipe, visible command
and stop criteria before running. No such fit or head replacement is implemented
or launched by this step.

If that control succeeds, investigate joint-training assignment stability. If
it fails, inspect the classifier's available crop information before proposing
an architecture change. Do not extend the full fit or adjust thresholds merely
to make the existing gate pass. These remain two fitted training crops, not
development, test or final SWC performance.

## Visible files and artifacts

- [Comparison script](../../scripts/pointneuron2/compare_pointnext_cropnorm.py)
- [Model and checkpoint conversion](../../src/pointneuron/models/pointnext_query_cropnorm.py)
- [Model contract tests](../../tests/test_pointnext_cropnorm.py)
- [Full comparison and matched/excess query scores](../../tmp/pointneuron2/local_runs/pointnext_cropnorm_20261013/comparison.json)
- [Candidate converted checkpoint](../../tmp/pointneuron2/local_runs/pointnext_cropnorm_20261013/cropnorm_step_2000.pt)

The ignored `tmp/pointneuron2/local_runs/pointnext_cropnorm_20261013` folder also
contains all 512 query scores/assignments in six CSV files and six prediction
arrays for the three snapshots and two crops. Source files and this readable
report remain in the real PN2 directory.

For a separate reproduction, choose a fresh output directory; the script refuses
to overwrite completed/failed controls:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe `
  .\scripts\pointneuron2\compare_pointnext_cropnorm.py `
  --output-directory .\tmp\pointneuron2\local_runs\pointnext_cropnorm_repeat
```

The comparison is already complete; this is a reproduction command, not a
request for another run.
