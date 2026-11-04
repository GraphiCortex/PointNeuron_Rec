# Local curve-head results and interruption — 4 November 2026

**The four-box fit is interrupted, not a completed accuracy verdict.** No
sixteen-box or GATv2 fit started. The last logged update was 1700; the last
complete model/Adam/RNG checkpoint and geometry evaluation are at update 1000.
No restart or extension is currently running.

The user approved a bounded experimental Transformer bypass, restricted
trace-agreement supervision and conditional CPU -> four -> sixteen-box work.
The [frozen protocol](local_curve_head_protocol.md) and
[configuration](../configs/local_curve_head.json) specify the experiment.
The product architecture has not been replaced. Original source, caches,
historical checkpoints and failed gates remain unchanged.

## What executed

- Existing 2048-point image clouds and PointNeXt/GN8; no new Sato filtering,
  classical extraction, manual masks or reference-guided input selection.
- Dense 32 -> 64 -> 64 -> 4 head, 797252 total model parameters. Its initial
  encoder tensors exactly match the geometric-query Transformer control.
- Twelve new geometry/head tests, six stop-rule tests and four subsequent
  wall-guard tests passed (22 total). CPU model checks passed finite gradients
  on all four boxes, exact train/eval behavior, image-only inference and exact
  synthetic model/Adam/RNG restoration through the next update.
- CPU preparation of the selected training contexts and the existing development
  grid took 88.21 seconds. This is new label/control preparation, not a rerun
  of the completed 611-box curve audit.
- Eight throwaway GPU timing updates averaged about .471 s/update after startup.
  Full 415-box development evaluation took 244.97 seconds. That development
  output is timing evidence, not accuracy of the fitted model. PyTorch peak
  allocation was 105465856 bytes (100.58 MiB), excluding other GPU/system memory.
- A fresh four-box fit started with a 2000-update terminal cap, 500 exposures
  per box, lr=1e-4, cutoff=.25, no augmentation, RTX 3070 Ti Laptop GPU.
  Its measured planning estimate with 50% margin was 25.34 minutes, under the
  two-hour wall cap.

All generated outputs are retained in `tmp/local_curve_head_20261104/`.
The [portable evidence](evidence/local_curve_head_20261104.json) preserves
numeric results, source bindings and artifact/figure hashes.

## Feasibility and controls without learning

These are isolated-point agreement and arclength coverage at two native voxels,
macro averaged over neurons after pooling their scored boxes. Point agreement
uses full supplied reference geometry; coverage uses clipped box curves.
Outside-box counts are disclosed. This boundary convention differs from the
old crop-only point diagnostic, whose historical scores remain unchanged.

| Reference-only projected-source diagnostic | Point agreement | Curve coverage |
|---|---:|---:|
| Four training boxes | 100% | 89.14% |
| Sixteen selected training boxes / fourteen neurons | 100% | 95.52% |
| Seven development neurons | 100% | 91.74% |

The diagnostic uses eligible, unambiguous source points and the same
consolidation. It is not image-only prediction, an achievable-learning guarantee
or a universal ceiling. Ambiguous points still receive support supervision;
their offset targets are masked. All reference curve length stays in denominators.

| Development control | Point agreement | Curve coverage |
|---|---:|---:|
| Sato-ranked sampled points, full consolidated output | 9.79% | 69.78% |
| Sato-ranked sampled points, cap 512 | 28.14% | 59.90% |
| Classical graph nodes, full consolidated output | 64.09% | 53.47% |
| Classical graph nodes, geometric reduction to cap 512 | 64.96% | 49.76% |

Sato ranking uses the same selected source points; classical nodes use the
fuller retained guide. These counts/policies differ, so this table describes
tradeoffs rather than establishing a matched-count winner. The sixteen-box
protocol includes actual equal-count comparisons. No such learned development
comparison has yet executed. All seven development classical controls exist;
classical training controls exist only for the previously extracted contexts.

Selection was frozen before new oracle outcomes: retain 32/27,32/20,22/2,22/5;
include fragmented identities 23/66/75; add distinct training neurons across
source groups with alternating low/median/high target-count boxes. The manifest
and printed condition counts expose sparse/dense, thin/dim, junction and
boundary contexts. No development outcomes or reserved identities selected them.

## Actual learned evidence: update 1000, training only

This is 250 exposures per box, halfway through the declared recipe.

| Measurement at two voxels | Initialization | Update 1000 | Terminal requirement |
|---|---:|---:|---:|
| Accepted-point agreement | 7.68% | 57.00% | >=90% |
| Reference arclength coverage | 63.06% | 79.14% | >=80.23% here |

The effective coverage requirement is max(80%,90% of the 89.14% diagnostic).
Each box must also retain >=80% of its diagnostic coverage. That terminal gate
was **not evaluated**, because the run never produced the declared update-2000
checkpoint/evaluation. The update-1000 checkpoint is below the gate, not proof
that the completed recipe would fail or that the architecture cannot fit.

| Training context | Accepted points | Point agreement | Curve coverage |
|---|---:|---:|---:|
| Fruitfly 32 / 27 | 244 | 80.33% | 100.00% |
| Fruitfly 32 / 20 | 1109 | 59.60% | 97.88% |
| Frog 22 / 2 | 84 | 65.48% | 100.00% |
| Frog 22 / 5 | 374 | 47.33% | 56.63% |

Loss continued decreasing through the last logged updates. That alone does
not establish convergence or better reconstruction. Qualitatively the saved
orthogonal views show votes concentrating toward curves, with considerable
residual spread, especially visible in depth and in frog box 5.

![Observed learning](../tmp/local_curve_head_20261104/figures/four_box_learning.png)

[Orthogonal vote views](../tmp/local_curve_head_20261104/figures/four_box_votes.png)
show sampled image points, not newly rendered raw-image MIPs. Green lines are
references; accepted orange/red points are within/beyond two voxels of supplied
geometry. There are no predicted edges.

The source-distance diagnostics show a real offset-accuracy problem beyond
the closest source band. For eligible unambiguous sources at update 1000,
589/618 votes from [0,2) voxels land within two voxels of a curve (95.31%),
721/1062 from [2,4) do (67.89%), and 310/628 from [4,6] do (49.36%). These are
pooled source-point diagnostics, not macro accepted-point precision. They do
not isolate encoder preservation, head capacity, optimization or ambiguity as
the cause. Masked sources and other domains remain separately identifiable.

Predeclared cutoff diagnostics at update 1000:

| Cutoff | Point agreement | Curve coverage |
|---|---:|---:|
| .10 | 47.13% | 79.61% |
| .25 (fixed operating policy) | 57.00% | 79.14% |
| .50 | 66.54% | 76.81% |
| .75 | 85.85% | 47.28% |

None of these diagnostic operating points meets both terminal numerical
requirements at this intermediate checkpoint. No cutoff was selected or changed.

## Saved earlier misplaced-position domain

The earlier candidate's 51.82% off-reference figure is a neuron macro over
**reference-present development boxes only**. It does not include unknown boxes.
Direct aggregation of immutable saved counts gives:

| Epoch-25 state | Accepted in 151 reference-present boxes | Within two | Beyond two | Accepted in 264 unknown boxes |
|---|---:|---:|---:|---:|
| Original objective | 19397 | 4573 | 14824 | 20145 |
| Consistent objective | 5070 | 2643 | 2427 | 1139 |

The candidate's pooled off-reference fraction is 2427/5070=47.87%; it differs
from 51.82% because pooling points and macro averaging neurons are different
aggregations. Neither certifies biological error where annotation is incomplete,
including inside a reference-present box. Unknown biological precision remains
unassigned. No old prediction or gate was recomputed or relabeled.

## Interruption and resource guard

The training log changes from elapsed 683.6 seconds at update 1550 to 54473.8
seconds at 1600. The cause of that inactive/tool/host gap is not established.
It must not be described as fifteen hours of measured GPU computation.
The next observed log reached 1700 at elapsed 54654.6 seconds. The controller's
native subprocess timeout failed to enforce the agreed wall limit across this
gap. This is an execution-control defect, not evidence about model quality.

The verified owned process tree (24588,32444,3240,38764,42364, including its
console helper) was stopped. Separate sandboxed diagnostics initially could
not see that tree; the stop receipt records the verified external identification
and actual termination. The first interruption note is an initial observation,
not proof that the processes had already stopped.

The original scientific driver/config/protocol remain byte-identical to the
run. A separate [wall guard](../src/pn2/training/local_curve_control.py) and
[future launcher](../scripts/run_local_curve_guarded.py) now check wall and
monotonic elapsed time every half second, including after resumption. A
completion discovered beyond the cap is also budget-limited. Four simulated
clock-jump/completion tests passed. Actual host suspension has not been
retested. No new model run was launched to test the guard.

## Recovery decision

`train4_dense/checkpoint_step001000.pt` is the latest complete recovery point.
Its sidecar hash, all 64 Adam clocks/moments and full RNG state are verified.
The last logged 1700 state is not checkpointed; resuming at 1700 or silently
resetting Adam would be invalid. Logs/predictions/checkpoints are preserved.

To finish this question, the next bounded option is to restore update 1000
with unchanged model, labels, optimizer, RNG and box order, execute updates
1001–2000, then apply the original terminal gate. This replays already executed
but unsaved work. Report that extra compute separately from the final model's
2000-update history. Measured active update costs suggest about 10–15 minutes
with evaluation/margin; a prospective 30-minute wall cap is a reasonable
recovery allowance. It is a new resource decision after the old cap was
exceeded, not an automatic continuation. No continuation is pending/running.

Sixteen boxes remain conditional on a genuine four-box terminal pass. The
paired Transformer comparison, fitted-model development accuracy, 64 boxes,
GATv2, global graph connectivity, radii and final SWC quality are unmeasured
by this interrupted run. Do not restart Phase 1 or promote this head as PN2.
