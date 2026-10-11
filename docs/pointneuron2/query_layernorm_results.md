# Pooled LayerNorm passes the two-crop learnability check

12 October 2026. The user completed the approved 2,000-update query control on
the local RTX 3070 Ti Laptop GPU in **1 minute 30 seconds**. Saved-run matching,
unchanged-layer initialization and final native checkpoint reload checks pass.
Only the pooled encoder normalization differs from the saved BatchNorm baseline.

**Ordinary inference now retains both accurate placement and meaningful
confidence on the two fitted training crops.** This passes the local
learnability check and supports moving to a matched train/validation pilot.
It does not measure validation or full-volume reconstruction performance.

## Measured trajectory

Mean distances and coverage are averaged equally across the two crops. Coverage
uses six-voxel tolerance; accepted slots use the unchanged score cutoff of 0.5.

| Updates | BatchNorm mean distance | LayerNorm mean distance | BatchNorm filtered coverage | LayerNorm filtered coverage | LayerNorm accepted slots, crops 0 / 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 2.68 | 0.48 | 29.2% | 45.8% | 4 / 16 |
| 1,000 | 3.46 | 0.47 | 75.0% | 87.5% | 8 / 14 |
| 2,000 | 4.56 | **0.35** | 91.7% | **100%** | **6 / 12** |

LayerNorm raw target coverage is 100% at all three checks. Confidence takes
longer to converge. At the fixed final update, all 18 targets have distinct
accepted proposals within one voxel, with maximum assigned error **0.55 voxels**.
No unmatched slot passes the score cutoff. Counts alone would not establish
uniqueness; direct float64 Euclidean assignment independently verifies it.

The final evaluation center loss is 0.01630, close to the mean training center
loss of 0.01712 over the last 50 updates. The preceding BatchNorm model's
corresponding values were 2.76750 and 0.01220. The large inference placement
gap identified in the earlier audit is therefore substantially closed in this
controlled fresh fit. No training objective or threshold was changed to obtain
the improvement.

## Single-crop inference and companion control

Final single-crop inference gives mean target distance **0.34 voxels**, 100%
filtered coverage, 6 / 12 accepted slots and all 18 distinct nearby matches.
The maximum center difference from paired inference is 0.036133 voxels across
all slots; the maximum score difference is 0.000785. Outputs are close, not
bit-identical when batch shape changes. Acceptance and target coverage agree.

A further bounded inference check keeps batch size two but replaces the
companion crop with a copy of the first crop. The original first crop's centers,
logits and radii remain **bit-identical**. This separates companion identity
from the small numerical differences observed when evaluating with a different
batch shape. It checks this final fitted model, not every possible image.

## Verification and scope

The source hashes, saved baseline integrity, cache/partition, selected inputs,
budget, milestones and all checkpoint/prediction-array hashes were verified.
The final `proposal.pt` weights exactly match the 2,000-update snapshot. Native
CUDA inference exactly replays its saved centers, logits and radii. The
checkpoint file hash is unchanged after the bounded inference audit.

These results establish learnability on **two crops from training neuron 0**.
They do not establish generalization, improvement over the full PN1 pipeline,
survival through suppression/graph construction, or commercial readiness.
Other encoder BatchNorm layers remain in place. The successful candidate is
pooled LayerNorm with zero decoder dropout; production defaults stay unchanged.

## Approved next step

The user approved preparation of the matched **30-epoch pilot**, using the same
33 training / 7 development-validation neurons, a shared pooled-LayerNorm
encoder for both pointwise and query heads, and zero query dropout. Save epochs
5/15/30, retain the final epoch as primary, add validation geometry/confidence
reporting, and estimate duration with a short representative timing diagnostic.
The user will launch the full run visibly. Reserved test neurons and downstream
reconstruction settings stay unchanged. Whole-neuron evaluation is a subsequent
agreed stage; the seven validation neurons already inform development.

## Evidence

[Verified machine-readable results](query_layernorm_seed0/summary.json) include
the paired trajectory, independent per-crop assignment checks, saved hashes,
single-crop diagnostics and companion-control outcome. The preceding
[preparation protocol](query_layernorm_ready.md) records source and test checks.
No production model source changed during this results audit.

```text
User run: tmp/pointneuron2/local_runs/query_normalization-smoke_20261012_012749_4464/models/
Read-only inspection: tmp/pointneuron2/local_runs/query_layernorm_results_audit_20261012/
```
