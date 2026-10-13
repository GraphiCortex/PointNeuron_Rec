# Fixed-placement confidence: one crop passes; linear capacity verified

14 October 2026 review. The user completed the agreed 2,000-update confidence-only fit
on the RTX 3070 Ti Laptop GPU in 14s wall time (8.25s recorded fit time).
Only 258 parameters trained; geometry, radii, features and labels remained
bitwise unchanged, and checkpoint reload matched exactly. **Crop 1 passes;
crop 0 and the combined gate fail.** No full-model fit, development/test
evaluation or SWC reconstruction occurred.

## Measured improvement

| Update | Crop 0: accepted matches / extras | Crop 0 curve coverage within 6 | Crop 1: accepted matches / extras | Crop 1 curve coverage within 6 |
| ---: | --- | ---: | --- | ---: |
| 0 | 2/6 / 3 | 33.15% | 12/12 / 2 | 100% |
| 100 | 3/6 / 3 | 54.69% | 12/12 / 2 | 100% |
| 500 | 3/6 / 3 | 54.69% | 12/12 / 0 | 100% |
| 1,000 | 3/6 / 2 | 54.69% | 12/12 / 0 | 100% |
| 2,000 | 3/6 / 2 | 54.69% | 12/12 / 0 | 100% |

Cutoff .5 and fixed one-to-one labels are unchanged. Placement remains at
.432/.216 mean matched distance and .568/.369 maximum distance in native voxels.
Crop 1 has passed since update 500. Its lowest positive score is .969 and highest
negative .263 at update 2,000. This is fixed-feature training performance, not
ordinary inference of an automatically updated full model.

Crop 0's weighted classification loss decreases .1954 → .1734 → .1555 at
updates 0/1,000/2,000. Its unresolved positive queries 90/102/408 have final
scores .459/.088/.033. Unmatched queries 217/404 remain accepted at .621/.544.
The lowest positive rises .018 → .033 and the highest negative falls .716 →
.621. The loss and margins continue improving; the fixed budget has not
established convergence or a representational impossibility.

## Linear-separability oracle: what the features can represent

A CPU-only [numerical audit](../../scripts/pointneuron2/audit_confidence_separability.py)
asked whether **any affine linear classifier** on these exact cached features
could correctly separate all fixed labels. A binary two-logit linear head has
the same decision boundaries as one affine score, so this tests the capacity
of the existing 128-to-2 final output without adding features or a hidden layer.

The linear program requests signed margins of at least one for every sample,
using an invertible feature centering/scaling on nonconstant columns for numerical
conditioning. It minimizes L1 witness size, then transforms the witness back
to the original feature coordinates and checks the original float32 values.
Witness parameters are temporary and are **not saved as a classifier**. This
uses known training labels; it is an oracle capacity diagnostic, not an SGD
fit, new inference policy or generalization result.

| Feature scope | Labels checked | Result | Minimum float32 signed margin | Label errors |
| --- | ---: | --- | ---: | ---: |
| Both crops, one shared boundary | 1,024 | Verified strict separator | .999996 | 0 |
| Crop 0 alone | 512 | Verified strict separator | .999994 | 0 |
| Crop 1 alone | 512 | Verified strict separator | 1.000000 | 0 |

All 18 positive and 1,006 negative labels can therefore be separated jointly
by the existing linear function class on these cached inputs. The current
training recipe has not found such a decision boundary. This narrows the next
question to optimization, budget and regularization under the fixed-feature
control; it does not identify learning rate as the unique cause. The oracle
does not compare GPU kernels or prove that joint full-model training will
learn stable assignments. It also does not establish useful biological or
out-of-sample confidence calibration.

The audit made three linear-solver calls, zero neural optimizer updates and
zero classifier checkpoint writes. Protected source/cache/fit/checkpoint
hashes were unchanged. Three known-case tests passed: separable points,
identical features with opposing labels, and XOR. In infeasible cases the
script requires a numerically checked dual witness; timeouts/errors remain
unresolved rather than being labeled capacity failures.

## Proposed next comparison for discussion

Prepare **one fresh learning-rate-only replay at .001**, versus the saved
.0001 baseline: same cached features, fixed labels/geometry, original confidence
initialization, 2,000 updates, AdamW weight decay .0001, clipping, objective,
cutoff and snapshots. Both cached crops still contribute each update. No larger
head or encoder change is part of this proposal.

This tests whether faster optimization reaches the available separator within
the same bounded budget. The baseline fit took 8.25s of compute; the same
operation count suggests comparable runtime, to be measured before launch.
A pass would support an optimization bottleneck on the fixed task; a failure
would require reviewing loss/margins and optimizer behavior, not declaring
the verified function class incapable. No learning-rate change, replay runner
or new fit is implemented or launched in this results-review step.

## Visible evidence

- [GPU fit report](../../tmp/pointneuron2/local_runs/query_confidence_20261013/model/fit_report.json)
- [Crop 0 final query scores](../../tmp/pointneuron2/local_runs/query_confidence_20261013/model/step_2000_crop_0_scores.csv)
- [Crop 1 final query scores](../../tmp/pointneuron2/local_runs/query_confidence_20261013/model/step_2000_crop_1_scores.csv)
- [Linear-feasibility audit](../../tmp/pointneuron2/local_runs/query_confidence_20261013/diagnostics/linear_separability.json)
- [Oracle audit source](../../scripts/pointneuron2/audit_confidence_separability.py)
- [Known-case oracle tests](../../tests/test_confidence_separability.py)

Generated files remain ignored; the diagnostic source and this readable report
are kept in the real PN2 directory and Git history.
