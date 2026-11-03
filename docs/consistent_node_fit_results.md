# Paired objective continuation: higher precision with lost recall

3 November 2026. Both methods completed the agreed 2300 new updates and all
terminal evaluations. User-observed combined wall time: **3h15m43s**. Execution
passed; the candidate **failed every prospective relative screening condition**.
It is not adopted as the default objective or a verified reconstruction fix.

[Protocol](consistent_node_fit_ready.md),
[visible evidence](evidence/consistent_node_fit_20261103.json).
Raw report: `tmp/consistent_node_fit_train_20261103_155059/report.json`.

## Matched evidence

Both arms independently restored the same epoch-20 2048 model, saved Adam
moments/update clocks and RNG state. Each received the identical five-epoch
schedule over all 460 supervised training boxes. Only the objective differed.
Final checkpoints have stage 2300 / historical Adam clock 11500. Source/config
bindings, completion-marker hashes and all terminal metric/prediction hashes
were verified. No unknown/development box entered gradient training.

All numbers below are macro means over neurons, within **two native voxels**.
Training precision uses reference-present boxes; development uses the entire
image grid. These precision domains must not be compared directly.

| Measure | Epoch-20 origin | Original objective +5 epochs | Candidate +5 epochs |
|---|---:|---:|---:|
| Training precision | 7.65% | 9.91% | 19.71% |
| Training recall | 29.89% | 32.51% | 18.17% |
| Training F1 | 11.87% | 14.82% | 18.47% |
| Training curve-length support | 32.63% | 34.03% | 20.86% |
| Development precision | 3.19% | 4.66% | 15.88% |
| Development recall | 25.68% | 24.49% | 15.34% |
| Development F1 | 5.40% | 7.39% | 14.80% |
| Development curve-length support | 29.65% | 27.20% | 17.94% |

The candidate raises precision/F1, but loses actual reference nodes and curve
coverage. Against the matched continuation, development recall decreases by
9.15 percentage points and curve support by 9.26 points. Training recall falls
by 14.34 points. Both arms also fail the original absolute engineering gates.

Across all development boxes, baseline accepted 39542 proposals and matched
1606/6930 targets; candidate accepted 6209 and matched 1086/6930. These pooled
counts differ from neuron-macro percentages. Unknown-box outputs remain in the
full-grid disclosure; unmatched outputs are not certified biological negatives.

## Geometry versus confidence filtering

Before cutoff, all-query development recall is **27.70% baseline versus 26.76%
candidate**. Training all-query recall is 35.00% versus 32.21%. Thus there is no
evidence here of improved raw two-voxel proposal recovery. The candidate also
discards more of its geometrically useful proposals at the fixed .5 cutoff:
development recall drops from 26.76% before filtering to 15.34% afterward.
This is an operating-behavior observation, not a separate calibration proof.

Lowering the candidate threshold could recover some proposals, but its all-query
recall remains poor. No threshold is retuned and no failed gate is relabeled.
Reference-curve proximity to accepted node centers also declines. This is a
separate spatial-support measurement, subject to the spacing limitation below.

Interpretation clarification from the subsequent [formulation review](formulation_review_20261103.md):
this is reference-curve proximity to **isolated accepted node centers**, not to
predicted graph edges. It is itself node-spacing dependent. A decline proves
less spatial support by accepted anchors, not necessarily loss of a whole
biological branch in a connected reconstruction. No predicted connectivity is
available here to settle that question. The reported numbers and prospective
failed gates remain unchanged.

The sparse/dense training strata show recall losses in every count group:

| Reference targets per box | Baseline pooled recall | Candidate pooled recall |
|---|---:|---:|
| 1–16 | 42.97% | 21.22% |
| 17–64 | 38.67% | 22.32% |
| 65–128 | 25.02% | 16.06% |
| 129–512 | 16.10% | 11.85% |

Count-stratum precision improves in each group; its denominators and full
tables are retained in each worker report. This is not a sparse-box-only loss.
Apparent caliber/contrast and structural diagnostics remain in the raw metrics.

## Optimization and interpretation

Baseline clipped 2300/2300 updates; candidate clipped 165/2300 (7.17%). Candidate
loss decreased from mean .467 in new epoch 1 to .388 in epoch 5. Baseline loss
decreased from 5.629 to 5.211. Different objective values are not comparable.
Reduced clipping did not establish better geometric learning or reconstruction.

The unchanged-objective continuation modestly improves training precision and
recall, so extra training has a measured benefit under this schedule. Its
development F1 improves through precision while recall declines slightly.
Neither convergence nor an eventual benefit of hundreds of epochs is established.

The candidate clearly changes rejection behavior and reduces excess proposals.
It does not solve accurate placement and coverage. This is one warm start, one
seed, inherited Adam moments and a bundled objective change (class weighting,
geometry normalization and assignment). It cannot identify the individual
mechanism or rule out different training of that objective.

## Decision

Preserve this candidate and failed screening as evidence; do not promote it or
automatically extend it. No further GPU run is started. The strongest unresolved
problem remains learning accurate curve geometry and locally identifiable
targets, with candidate confidence adding further losses at the fixed cutoff.
The target-phase counterexample is a confirmed possibility, not a cohort-level
causal attribution. Phase 1 is not rebuilt and GATv2 readiness is not inferred.
