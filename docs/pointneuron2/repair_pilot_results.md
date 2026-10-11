# Corrected-input coordinate pilot: training and crop diagnostics

**Update:** the user chose the score/suppression audit; its
[stage-by-stage results are now available](selection_audit.md). The proposed
comparison below records the discussion before that completed audit.

The user-launched training completed on 11 October 2026 in 1 minute 31 seconds.
Both five-epoch arms completed. Matched initialization, cache, partition, RNG
streams, batch order and augmented candidate hashes passed. Eight test neurons
remain untouched. No full reconstruction has been launched for this experiment.

**Local coordinates give a modest loss and location improvement. This pilot
does not establish a reconstruction improvement. The current 0.85 graph score
cutoff is a candidate downstream bottleneck that needs a controlled check.**

## Training results

| Arm | Best validation proposal loss | Saved epoch | Training seconds |
|---|---:|---:|---:|
| Repaired raw-coordinate control | 5.1943 | 4 | 38.60 |
| Repaired local-coordinate features | 5.1040 | 5 | 38.40 |

Local best validation loss is 1.74% lower. At epoch five, train losses are
5.3140 raw versus 5.1878 local; validation losses are 5.2169 versus 5.1040.
These losses are normalized over annotated records (210 train / 47 validation),
not all 262 / 56 cached crops. Both arms use random 512-input selection from
1,024 candidates with auxiliary guide loss disabled. The displayed alpha stays
0.25 because neither arm trains the learnable sampler.

This is one seed and five epochs. Different best epochs are compared as specified
by the validation checkpoint rule; no statistical significance is established.
Historical wrong-reader losses use different decoded images/crop boxes and are
not a clean before/after comparator.

## Confidence is present, but does not prove correctness

The existing `scripts/pointneuron2/inspect_ablation_confidence.py` checked both
saved checkpoints on all 56 validation crops, using identical deterministic
selected inputs and no optimizer updates. Its [report](repair_pilot_confidence_seed0/summary.json)
contains checkpoint hashes, training curves and per-crop values.

| Arm | Crops with rows at score >=0.5 | Crops with rows at score >=0.85 | Raw rows >=0.85 |
|---|---:|---:|---:|
| Raw | 56 / 56 | 46 / 56 | 22,702 |
| Local | 56 / 56 | 46 / 56 | 17,589 |

These are raw rows before suppression, not independent reconstructed nodes.
Nine validation crops have no local reference nodes and are not confirmed
background. Confidence in them cannot establish accuracy or false positives.
An identical count of crops with any confident row can hide large coverage losses.

## Locations and score filtering

A small read-only CUDA diagnostic inspected all 47 annotated validation crops
using the same selected points as the confidence check. The [geometry evidence](repair_pilot_confidence_seed0/geometry.json)
records checkpoint hashes, per-crop values and loss components. It took about
3.7 seconds after environment/cache setup, with zero optimizer steps.

For each crop, nearest-node distances are computed between input/predicted XYZ
and cached SWC node XYZ. Coverage is the fraction of local reference nodes with
at least one retained row within three voxels. The following means weight all
47 annotated crops equally, including zero coverage when a threshold retains
no rows. They measure neither reference segments nor whole neurons; sparse SWC
vertices, padding duplicates and crop boundaries limit interpretation.

| Mean across 47 annotated crops | Raw | Local |
|---|---:|---:|
| Input reference-node coverage, all 512 points | 73.29% | 73.29% |
| Proposal reference-node coverage, all rows | 72.91% | 73.54% |
| Proposal reference-node coverage, score >=0.5 | 72.91% | 73.53% |
| Proposal reference-node coverage, score >=0.85 | 61.20% | 51.22% |
| Mean input distance to nearest reference node, all rows | 4.280 voxels | 4.280 voxels |
| Mean proposal distance to nearest reference node, all rows | 4.285 voxels | 4.209 voxels |

Local centers move modestly toward the references on average; all-row coverage
increases only 0.25 percentage points relative to the input points. At six voxels,
all-row input coverage is 83.55%, versus 83.50% for local proposals. This is not
a broad recovery of missing branches.

The graph preset's 0.85 cutoff would remove substantial support before node
suppression in these cached crops. Local coverage falls 22.30 percentage points
from the 0.5 retained set. This demonstrates a filtering effect on this crop
diagnostic, not that lowering the threshold will improve the final SWC. Lower
scores can retain spurious proposals and change graph connectivity.

In float32 evaluation, local mean objectness loss is higher (0.2263 versus
0.1897), while center loss is lower (1.4343 versus 1.4803) and radius loss lower
(1.6999 versus 1.8341). These are unweighted components; recipe weights determine
the total. Geometry and classification therefore need separate inspection.
Small differences from saved validation totals reflect float32 diagnostic versus
mixed-precision training evaluation.

## Recommended next comparison, awaiting user direction

Audit score filtering and distance-based node suppression on the same saved
proposals with fixed weights and budgets. Compare score 0.5 versus 0.85 and
preserve pre/post-filter counts and reference coverage. This would test whether
selection is discarding useful support before a lengthy full reconstruction.
No model, threshold, suppression setting or experimental strategy has been
changed. A direction question has been sent to the user.

The alternative is the existing two-arm full reconstruction command, which
reconstructs seven validation neurons per arm, then scores all 14 outputs for
spatial and topology quality. Historical 21-run evaluation took 81 minutes;
rough scaling suggests about an hour for 14, but corrected-input runtime is
unknown. Whole-volume behavior can differ from the limited cached crops.

## Files and reproducibility

Experiment and original logs/checkpoints remain in:

```text
C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\repair_20261011_130158
```

To repeat the existing confidence check, choose a fresh output directory:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\inspect_ablation_confidence.py `
  --experiment .\tmp\pointneuron2\local_runs\repair_20261011_130158 `
  --output-dir .\tmp\pointneuron2\repair_confidence_repeat `
  --device cuda
```

The geometry diagnostic used the same checkpoint-loading and deterministic
random-selection procedure, `torch.cdist` to local reference nodes, and
`conservative_skeleton_proposal_loss` with each checkpoint's saved recipe. Its
per-threshold `macro_means` in the JSON describe only crops with retained rows;
the table above recomputes coverage over all 47 crops with empty sets scored zero.
Do not compare the conditional threshold means across different crop subsets.

The [preparation audit](repair_pilot_ready.md) remains a record of the prepared
state; its experiment-plan hash predates the user's training update. The frozen
manifest and source/configuration hashes still match. Scripts/configurations
are unchanged, so the staged reconstruction remains compatible with this plan.
