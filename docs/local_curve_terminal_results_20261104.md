# Completed four-box curve-head fit — 4 November 2026

**Decision: stop this recipe; do not advance to sixteen boxes.** The user-launched
recovery completed the original 2000-update trajectory. The fixed four-box gate
failed point agreement; all three coverage checks passed. No cutoff, loss,
architecture, input preparation or exposure cap was changed to obtain a pass.

This supersedes the pending terminal verdict in the
[interruption report](local_curve_head_results_20261104.md). The earlier report,
artifacts and original interrupted run remain preserved. The
[protocol](local_curve_head_protocol.md) and [recovery command](local_curve_recovery_ready.md)
retain their frozen settings.

## Execution and provenance

The user ran `scripts/resume_local_curve_four.py` locally on the RTX 3070 Ti
Laptop GPU. The recovery guard measured 409.82 seconds (6m50s), below its
1800-second limit. The worker reports 398.70 seconds after prediction replay.
The terminal model has 2000 historical updates, 500 exposures per box. The
interrupted run's 700 unsaved, logged updates were replayed; this is extra
compute, not an extra 700 updates in the terminal model's history.

All four restored prediction arrays matched exactly, and all **700 replayed
optimizer updates** matched the original persisted loss/gradient trace exactly.
Complete model/Adam/RNG state was retained, and checkpoints at 1250/1500/1750/2000
were saved. These are strong execution/continuity checks, not quality gates.
The model, data order, reference labels and scientific recipe are unchanged.

Generated outputs:
`tmp/local_curve_head_20261104/recovery4_dense_20261104T155743_952998Z/`.
Its sibling `*_wall_guard/report.json` records successful bounded execution.
The terminal metrics and raw predictions are in `eval_step002000_training/`.
Portable aggregates/bindings: [terminal evidence](evidence/local_curve_terminal_20261104.json).

## Terminal result

Metrics use native voxels and the same frozen scoring policy: point agreement
against full supplied reference geometry, arclength coverage on clipped box
curves, counts pooled within neuron and then macro averaged over the two
training neurons. The four boxes are not four independent held-out neurons.

| Two-voxel training metric | Initial | Update 1000 | Update 2000 | Required |
|---|---:|---:|---:|---:|
| Accepted-point reference agreement | 7.68% | 57.00% | **74.96%** | >=90% |
| Reference-curve coverage | 63.06% | 79.14% | **83.32%** | >=80.23% here |

The numerical coverage requirement combines the >=80% floor and >=90% of the
89.14% reference-only diagnostic. The terminal ratio is about 93.47%. Per-box
diagnostic ratios are 100.00%, 98.96%, 100.00%, 84.73%, all above the 80% floor.
Thus precision fails; absolute coverage, macro diagnostic ratio and minimum
box ratio pass. The verdict is not changed by the meaningful intermediate gains.

At six voxels the terminal agreement is 98.72% and coverage 97.50%. Many
predictions are broadly near the traced geometry while lacking two-voxel
localization. Spatial proximity does not establish branch connections.

| Training context | Accepted proposals | Agreement at 2 | Curve coverage at 2 |
|---|---:|---:|---:|
| Fruitfly 32 / 27 | 202 | 92.57% | 100.00% |
| Fruitfly 32 / 20 | 984 | 75.30% | 98.73% |
| Frog 22 / 2 | 76 | 77.63% | 100.00% |
| Frog 22 / 5 | 316 | 70.25% | 64.91% |

Fruitfly 32/20 covers most length but retains many displaced proposals. Frog
22/5 also has substantial missing point support: its longest unsupported
cropped run is 19.49 voxels. Its apparent-thin/dim arclength coverage is
8.84/29.86 = 29.60%. Four of six junction locations and two of six crop-boundary
locations have nearby accepted points. These are descriptive structural flags,
not added post-hoc gates, calibrated anatomical strata or topology scores.

## What improved, and where localization still fails

For eligible, unambiguous source points, the fraction of raw votes within two
voxels of reference geometry improves by source-distance band:

| Source distance | Update 1000 | Update 2000 |
|---|---:|---:|
| [0,2) | 589/618 = 95.31% | 598/618 = 96.76% |
| [2,4) | 721/1062 = 67.89% | 896/1062 = 84.37% |
| [4,6] | 310/628 = 49.36% | 487/628 = 77.55% |

The head learns useful displacements, including from farther seeds. The result
is not explained solely by rejecting more predictions. These are pooled raw
source-point diagnostics, not the macro precision of selected proposals.

A small **post-hoc descriptive** split of the saved terminal proposals keeps
the fixed .25 cutoff and original consolidation. It performs no model execution,
changes no predictions/targets and supplies no new gate:

| Source domain | Accepted | Within 2 | Beyond 2 |
|---|---:|---:|---:|
| Eligible, unambiguous offset supervision | 1426 | 1148 | 278 |
| Ambiguous projection, source within 6 | 91 | 52 | 39 |
| Source beyond 6 but below 8 | 48 | 9 | 39 |
| Source at least 8 away | 13 | 0 | 13 |

The categories exhaust all 1578 accepted proposals. Of the 369 off-reference
positions, 278 (75.34%) originate in the domain with explicit unambiguous offset
supervision. Ambiguity contributes 39 (10.57%); sources beyond the offset-training
domain contribute 52 (14.09%). The residual error therefore is not predominantly
an ambiguity-mask or outside-domain effect. This does not isolate encoder
preservation, head capacity, optimization or annotation validity as the cause.

All four crops are reference-present. Distance from the supplied trace remains
reference disagreement, not certified biological false positivity. The declared
trace-agreement task and incomplete-annotation caveat still apply.

## Confidence and consolidation

| Predeclared cutoff | Point agreement | Curve coverage |
|---|---:|---:|
| .10 | 64.53% | 84.00% |
| .25, fixed operating policy | 74.96% | 83.32% |
| .50 | 81.17% | 80.26% |
| .75 | 91.44% | 63.17% |

None of these predeclared diagnostics meets the joint gate. No new threshold
search or operating-policy change was made.

At .25, raw accepted votes have 81.57% agreement and 83.50% coverage. After
consolidation those values are 74.96% / 83.32%. Suppression keeps real positions;
it does not introduce coordinate displacement. It changes the population by
removing redundant votes, and the off-reference fraction increases. But raw
agreement also remains below 90%, so changing aggregation alone is not a
demonstrated solution.

The accepted-proposal 512-cap diagnostic gives 80.03% agreement / 81.48% coverage.
It is not adopted as a gate or operating policy. The many dense proposals and
different policies prevent direct ranking against the earlier sparse exact-node
Transformer fit or the classical graph's segment F1.

## Research interpretation and next decision

- **Execution:** the recovery is complete and exact over the available replayed
  trajectory. The ordinary run finished within the repaired guard's limit;
  actual host suspension still has not been retested.
- **Training learnability:** useful curve attraction and coverage were learned,
  but the agreed fine-localization prerequisite failed at 2000 updates. This was
  a bounded schedule, not demonstrated convergence or a capacity limit.
- **Development accuracy:** the fitted model has no development evaluation. The
  earlier eight-update throwaway timing output is not a development result for
  this terminal checkpoint. Sixteen boxes were not started.
- **Final reconstruction:** no predicted edges, learned connectivity, rooted
  graph, radius or final SWC claim follows. Dense proposal coverage is not
  continuous reconstruction or topology quality.

Stop this recipe under its agreed rule. Preserve the local-curve supervision
idea as a research option, along with this failed control and useful caches.
Do not promote the dense MLP/NMS arm, bypass the product Transformer permanently,
restart Phase 1, tune cutoffs or extend training automatically. A next model or
aggregation change requires a new discussed hypothesis and comparison. The
remaining uncertainty is how to obtain tightly localized, useful representative
proposals from the available image evidence; this run does not assign the cause
to a single stage.
