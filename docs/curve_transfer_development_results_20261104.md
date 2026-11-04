# Frozen four-box curve-head transfer — completed 4 November 2026

**Result:** the learned dense head fits its four training boxes but transfers
poorly to the seven development neurons. The Sato-top-512 forest beats it on
all seven neurons at two voxels. No original gate is changed or retroactively
passed. Transformer work remains parked; no new training launcher or run is prepared.

The user completed scoring-only continuation
`tmp/curve_transfer_scoring_20261104T211240_017254Z`. The worker took 563.20s;
guard 568.10s, exit 0 under 1800s (terminal about 9m31s). The first 12 fit
graphs passed exact original-scorer parity for decoded and all three candidate
metrics (48 dictionaries). All 453 development graph records are complete.
This invocation reused all 151 predictions and 453 graph/SWC exports; it ran no
model, graph construction, filtering, GPU or training. See
[portable measurement/evidence](evidence/curve_transfer_development_20261104.json).

## Matched geometry, transfer versus fitting

Same original radius-1 consolidation and fixed k=6 Sato-cost minimum spanning
forest family, native voxel coordinates, .5-voxel arclength quadrature. Counts
differ. Directed length is pooled within neuron, then P/R/F1 macro averaged
across neurons. Macro F1 is not the harmonic mean of macro P/R.

| Development policy | P2 | R2 | F1 at 2 | P6 | R6 | F1 at 6 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen dense head, original .25 cutoff | 18.59% | 33.28% | 23.20% | 54.41% | 62.69% | 56.99% |
| Raw 2048 input points, unit-score consolidation | 6.64% | 77.94% | 12.00% | 20.01% | 98.76% | 31.80% |
| Cached Sato-ranked top 512 | 29.02% | 70.25% | 38.49% | 53.96% | 86.13% | 63.13% |

| Policy | Four-box TRAINING fit F1 at 2 | Development F1 at 2 |
|---|---:|---:|
| Dense | 81.49% | 23.20% |
| Raw | 13.33% | 12.00% |
| Sato-top-512 | 31.36% | 38.49% |

Dense falls 58.28 percentage points from fit to development at two voxels.
It beats raw inputs on development F1, but loses to the stronger Sato control
by 15.28 points at two voxels and 6.14 points at six. Six-voxel precision is
similar to Sato; much lower recall remains. The active proposal's six-voxel
geometric comparison is reported, not replaced with only the stricter result.

This is inference on all 151 reference-present boxes of seven neurons absent
from this four-box checkpoint's training. The other 264 development-grid boxes
are annotation-unknown and excluded, not negatives. The cohort has been
repeatedly inspected and is not an untouched final test. The learned training
set had only two neurons and limited contexts, so the drop does not establish
that broader training or the head's asymptotic capacity must fail.

## Every development neuron

| Alias / context | Boxes | Dense P2 | Dense R2 | Dense F1 at 2 | Raw F1 at 2 | Sato F1 at 2 | Dense candidate R2, k6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 19, utokyo fly | 9 | 12.74% | 36.74% | 18.92% | 6.12% | 25.61% | 43.39% |
| 21, frog | 8 | 31.75% | 39.86% | 35.35% | 23.55% | 56.42% | 44.02% |
| 35, human Allen | 78 | 10.81% | 22.68% | 14.64% | 6.92% | 25.86% | 27.27% |
| 44, cultured cell | 6 | 16.82% | 50.14% | 25.19% | 8.56% | 29.83% | 55.68% |
| 79, mouse RGC | 31 | 26.25% | 40.39% | 31.82% | 18.46% | 50.65% | 46.07% |
| 94, adult zebrafish | 9 | 6.57% | 7.28% | 6.91% | 4.24% | 12.30% | 9.87% |
| 135, Janelia fly | 10 | 25.19% | 35.88% | 29.60% | 16.14% | 68.74% | 40.17% |

Sato wins all seven F1 comparisons at two voxels, and six of seven at six
(cultured-cell alias 44 is the exception). This is not one bad image hiding
an otherwise strong two-voxel mean. No significance/population claim is made
from seven convenience development neurons.

## Where the fixed graph loses coverage

| Candidate-union reference recall at 2, diagnostic only | k4 | k6 | k10 |
|---|---:|---:|---:|
| Dense | 35.65% | 38.07% | 41.57% |
| Raw | 84.69% | 89.00% | 93.74% |
| Sato-top-512 | 73.66% | 76.80% | 80.06% |

The learned k=6 candidate union supplies only 38.07% geometric opportunity;
the forest reaches 33.28%, a 4.79-point gap. Selecting a different subset of
those fixed edges cannot repair the missing candidate geometry. This does
not bound a future system that changes nodes/candidates or inserts new paths.
Union recall measures proximity, not known correct reference connections.

Raw and Sato inputs come from the same 2048 sources and offer much higher
candidate coverage, so input density alone cannot explain the additional
collapse after the learned proposal/selection stage. Density still limits
some strata (see the retained 2048/8192 input audit), and different retained
counts contribute to this comparison. Current confidence selection, offset
behavior, learned features, limited training diversity and optimization are
not isolated causal explanations by these observations.

| Policy | Nodes | Forest edges | Components | Empty boxes |
|---|---:|---:|---:|---:|
| Dense | 48,108 | 47,738 | 370 | 1 |
| Raw | 286,130 | 285,978 | 152 | 0 |
| Sato-top-512 | 77,312 | 76,960 | 352 | 0 |

All policies have zero isolated nodes; this does not mean zero spurious nodes.
The empty dense box is alias 135, tile 45. Across box-local domains the dense
predicted curve length is 122,494.98 versus reference 43,082.02 voxel units,
alongside low recall. Extra predicted structure and missing reference support
therefore coexist. Off-trace structure may include untraced neurites; these
metrics express reference agreement, not exhaustive biological foreground.

## Conservative reference-path shortcut diagnostic

| Policy | Shortcut flags / scorable edges | Off-reference-endpoint edges | Ambiguous edges | Different-component unknown | Path-leaves-box unknown |
|---|---:|---:|---:|---:|---:|
| Dense | 6 / 6,956 | 39,860 | 757 | 134 | 31 |
| Raw | 19 / 17,067 | 266,836 | 1,688 | 238 | 149 |
| Sato | 16 / 15,954 | 59,190 | 1,386 | 288 | 142 |

The unchanged flag is a same-component, unambiguous traced path longer than
twice its predicted edge plus four voxels. Distinct fragments remain unknown;
no scored edge was removed. The dense conditional flag rate is .086%, but
83.50% of its forest edges have an off-reference endpoint and are excluded from
that denominator. The analogous raw/Sato shares are 93.31%/76.91%. These are
not comparable biological wrong-join rates, nor evidence that topology is safe.
Short wrong joins, untraced relations and much off-reference structure escape
this conservative test. Full per-edge categories and paths are retained.

## Verification and interpretation

After the user run, all 453 durable record hashes/source bindings, unique
policy/case/tile identities, edge-category totals, flag counts and decoded
candidate-recall ceilings were checked, as were frozen source/input hashes.
The guard completed successfully. No extra geometry audit or model run was
launched by the assistant. The scoring repair recovered the measurement with
exact fit parity; no scientific threshold, prediction or original result changed.

- Execution: both CPU forward/export prefix and scoring now complete.
- Training learnability: useful fitting on four boxes is established; the
  original joint point gate and sole consolidation variant still failed.
- Development accuracy: the frozen tiny-fit model underperforms Sato-top-512
  under this matched graph scaffold on untrained neurons.
- Final reconstruction: whole-image merging, learned GATv2, topology, roots
  and radii remain unqualified/unbuilt. Per-box geometric results do not qualify
  a full reconstruction product or a replacement pipeline.

## Decision to discuss

Do not change the head, cutoff, point budget, consolidation or graph rule based
on these results. Keep Transformer work parked. A bounded sixteen-box test of
the SAME head with the already selected fourteen training neurons would test
whether training diversity improves this transfer baseline. It is not evidence
that such training will work or that sixteen boxes suffice.

The [concrete prospective protocol for discussion](curve_head_sixteen_downstream_discussion_20261104.md)
fixes proposed comparisons, floors, budget and stopping rules. It is expressly
post-hoc relative to the original failure, pending user agreement, and does not
authorize a GPU run. No new trainer/manifest/PowerShell training command was
implemented. Do not use the old supported-r2/min3 sixteen launcher: its variant
failed and that is not the proposed consolidation.
