# Dropout-zero control: geometry improves, confidence remains incomplete

12 October 2026. The user completed the approved fresh 500-update-per-arm fit
in 43 seconds on the local RTX 3070 Ti Laptop GPU. Dropout was the only changed
training setting: query decoder 0.1 to 0.0. This experiment fits the same two
training crops, with 6 and 12 reference targets, and is not validation or a new
whole-neuron reconstruction.

## Verified matched comparison

The saved runs agree on cache/partition hashes, initial encoder hashes, selected
input hashes, crop identities, update counts and the pointwise recipe. Both
pass their paired checks and reproduce their final predictions after checkpoint
reload. The effective configuration differs only in decoder dropout.
The pointwise control's final geometry reproduces exactly across the two runs.

| Arm / setting | Mean target distance | Raw target coverage within 6 voxels | Coverage after score 0.5 | Accepted slots in crops 0 / 1 |
| --- | ---: | ---: | ---: | --- |
| Pointwise control, both runs | 2.84 | 79.2% | 79.2% | 484 / 512 |
| Query, dropout 0.1 | 7.02 | 66.7% | 0% | 0 / 0 |
| Query, dropout 0.0 | 2.68 | 95.8% | 29.2% | 1 / 15 |

Coverage is nearest-proposal support for the sparse target set, averaged equally
across the two crops. It is not connected-curve recall or SWC F1. Output slots
and class positives mean different things in the two heads; slot counts alone
do not indicate quality. The score-filtered results precede suppression.

Disabling training dropout substantially improves deterministic query geometry
at the same update budget. It also begins to learn useful confidence separation:
matched versus unmatched mean scores are 0.199 versus 0.056 in crop 0 and
0.621 versus 0.043 in crop 1, compared with nearly flat scores in the prior run.
However, the selected set still fails to represent the fitted reference reliably.

- Crop 0: all six targets have nearby raw proposals, but the one slot passing
  score 0.5 covers none of them within six voxels. Its six matched scores range
  from 0.050 to 0.372, while an unmatched slot reaches 0.503.
- Crop 1: raw proposals cover 11 of 12 targets; the 15 accepted slots cover seven
  targets. Eight accepted slots are within six voxels of a sparse target.
  Some matched slots have scores as low as 0.058; an unmatched slot reaches 0.947.

Matched/unused refers to one-to-one supervision, not biological true/false
structures. Sparse-target proximity also is not a test of neuron identity or
reference completeness. High-confidence unused slots can include duplicates.

## Remaining normalization gap

A forward-only replay reproduces the saved GPU metrics exactly. Fresh in-memory
model copies isolate normalization, without optimization or checkpoint writes.

| Query mode, dropout stays zero | Mean target distance | Raw coverage within 6 voxels | Coverage after score 0.5 |
| --- | ---: | ---: | ---: |
| Normal inference | 2.68 | 95.8% | 29.2% |
| Encoder BatchNorm uses the two-crop batch statistics | 0.48 | 100% | 37.5% |

Switching only the pooled global encoder BatchNorm to batch statistics gives
mean target distance 0.47 and full raw coverage, accounting for most of this
geometry gap. Evaluating the saved inference model one crop at a time gives
2.69 voxels and the same coverage, so the ordinary inference result is not
rescued by changing inference batch size. With dropout probability zero,
head training mode and three diagnostic random seeds give identical geometry.

This shows that the head can place proposals close to every target under the
training normalization conditions. It also shows a remaining mismatch when
using saved inference statistics. That gap was obscured by dropout dependence
in the earlier 0.1 run. It does not mean batch statistics should be used for
production inference: that would make results depend on which examples share
a batch and fails for the pooled BatchNorm at batch size one.

Confidence remains incomplete even when training statistics restore geometry.
Crop 0 still accepts no slots, and crop 1 covers nine of 12 targets after score
filtering. Normalization alone therefore does not resolve the selected-set fit.

## Cutoff diagnostics are not a new default

The fixed saved weights were also inspected at a declared grid of thresholds,
using direct float64 Euclidean distances on the saved arrays. At threshold 0.1,
mean target support reaches 83.3%, but only 61.1% / 37.5% of accepted proposals
lie near a sparse target in crops 0 / 1. At 0.05, raw coverage is recovered,
but those proximity fractions are 80.2% / 26.1% and 167 / 69 slots are retained.
This illustrates the ranking/duplicate tradeoff; it does not justify lowering
the default cutoff or establish full-volume precision.

## Proposed next decision

The suggested next comparison is **2,000 fresh optimizer updates per arm with
dropout still zero**, keeping all other training settings and the same fixed
inputs/targets unchanged. Save and inspect deterministic geometry, score-filtered
coverage and confidence at 500, 1,000 and 2,000 updates. A longer fit will show
whether classification is still converging, rather than changing the decoder
before its learning trajectory is measured. The query objectness loss is still
falling: means over updates 101–150, 301–350 and 451–500 are 0.338, 0.286 and
0.219. These are comparable training windows within this one arm, not quality
scores or a cross-head ranking. Estimated local duration is roughly
2–3 minutes from the observed 26.9 ms pointwise and 40.9 ms query update medians.

This comparison has been proposed to the user; it has not been launched or
implemented as a new run in this report. A separate controlled normalization
comparison is another possible next step. No architecture, loss, threshold or
full-training default is promoted from these results. A 30-epoch pilot remains
pending because the selected-set fit is incomplete.

## Evidence and scripts

[The verified summary](query_dropout_zero_seed0/summary.json) records matched-run
checks, checkpoint hashes, per-crop matched errors/scores, normalization controls
and the declared threshold grid. Historical evidence remains in
[the prior dropout diagnosis](query_gpu_fit_results.md).

```text
User run: tmp/pointneuron2/local_runs/query_smoke_20261012_001222_4464/models/
Forward-only audit: tmp/pointneuron2/local_runs/query_dropout_zero_audit_20261012/
```

[audit_query_fit.py](../../scripts/pointneuron2/audit_query_fit.py) performs the
saved-weight diagnostic. [summarize_query_dropout_control.py](../../scripts/pointneuron2/summarize_query_dropout_control.py)
verifies the two saved comparisons and summarizes the arrays without training.
The diagnostic check confirms original checkpoint files are unchanged.
