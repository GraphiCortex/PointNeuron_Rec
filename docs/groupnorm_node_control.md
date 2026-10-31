# Normalization-only PointNeXt/Transformer control

31 October 2026; preparation begun 30 October. The user agreed to work through the normalization correction
after the [failed paired node fits](sampler_node_fit_results.md). **GN8 is now
implemented, CPU-verified and passes the user-launched GPU timing.** The
[timing review and exact fit command](groupnorm_node_timing_results.md) record
the next manual action. Full training has not been launched by the agent.
This is one declared model revision; no sampler, query, target or loss revision
is included. It does not relabel the BatchNorm checkpoints as successful.

## Reason and architecture impact

The saved Sato fit has 5/68 all-query geometric node matches in frog tile 5 under
ordinary BatchNorm inference, versus 56/68 with current-crop statistics at the
same learned parameters. Both the inference discrepancy and remaining confidence
failure are established. A normalization change addresses the first contract;
it is not assumed to solve the second.

The architecture remains image -> point preparation -> PointNeXt -> Transformer
-> consolidation -> GATv2 -> SWC. Within PointNeXt, the sixteen BatchNorm layers
are replaced by GroupNorm. The Transformer, 512 queries, input/feature interfaces,
relative-coordinate/offset processing, restoration skips and point operators are
unchanged. This is a GN variant of the native prototype, not an official PointNeXt
benchmark reproduction or an implemented complete reconstruction system.

GroupNorm uses input-derived statistics in both training and evaluation, instead
of switching to accumulated running estimates. This supports a train/inference
consistency design; it does not establish useful features or generalization.
See [PyTorch GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.GroupNorm.html)
and [Wu and He, Group Normalization](https://arxiv.org/abs/1803.08494).
The installed Torch 2.12.1 implementation's `GroupNorm.forward` calls
`F.group_norm` without a training/evaluation flag. Its behavior was also tested
directly rather than inferred solely from documentation.

## Exactly what changes

One fixed grouping: **eight groups**, epsilon **1e-5**, learnable per-channel
scale/bias. The audited feature widths are 32/64/128/256/512; all divide by eight
and have at least four channels per group. Eight groups is an explicit starting
design, not a tuned optimum. There is no group-count or epsilon search.

Fresh initialization starts with the same encoder seed 0 and head seed 101 as
the failed Sato baseline. Every named parameter retains its exact initial value
and shape. BN's untrained running buffers are absent from GN. The model retains
**1,688,518 trainable parameters**. No learned parameters or optimizer moments
from the failed fit are transferred. Conversion explicitly rejects BN layers
whose counters/running statistics show training; the factory only replaces
fresh normalizers before any forward pass.

The original runner/common files, original recipe and historical source remain
unchanged. The new variant uses the existing data, objective, optimizer,
checkpoint, evaluation and worker-log helpers. GN-specific source hashes are
bound into the derived recipe and therefore its checkpoint fingerprint.
Baseline implementation identity remains independently verifiable.

The saved baseline and current configuration are checked before execution. A
compact config, `configs/groupnorm_node_fit.json`, derives its learning settings
from the frozen original recipe. It cannot quietly introduce another LR, loss,
budget, query count, point selection or target policy.

## Fixed next comparison

Only **one** fresh model will train on the existing Sato clouds. Its comparator
is the completed Sato/ordinary-BN fit, not a repeat of both sampler arms.

| Item | Unchanged specification |
|---|---|
| Training boxes | Fruitfly 32/27, 32/20; frog 22/2, 22/5 |
| Source points | Existing frozen 2048-point Sato clouds, unchanged order/features |
| Targets | Audited rectangular 18/85/7/68 node sets; eight-voxel grid unchanged |
| Query head | 512 learned queries, three layers, 128 channels; no geometric anchors |
| Loss | Same Hungarian matching, balanced used-slot CE and center SmoothL1 |
| Radius | Still excluded from matching/loss |
| Optimizer | Fresh AdamW LR 0.0001, weight decay zero, same betas/epsilon |
| Updates | 2000 total, 500 per box; clip norm 10 |
| Numerics | Batch one, float32, deterministic math attention, TF32 off |
| Evaluation | 0/8/500/1000/2000; ordinary `.eval()` supplies the node verdict |
| Node criterion | Precision AND recall >=95% at two voxels in each box, cutoff 0.5 |

The derived recipe differs only in normalization, model-variant label and the
metadata needed for a one-arm fresh control. Its fresh state hash differs because
BN running buffers are removed; the underlying parameter values are identical.
No reference or neuron identity is added to forward inputs. No guide or cloud
is regenerated, and no development/reserved data are opened.

## CPU verification completed

`tmp/groupnorm_node_cpu_20261030/report.json` contains the executed checks;
the [visible evidence summary](evidence/groupnorm_node_cpu_20261030.json) also
records the exact derived recipe, source identities and normalizer manifest.

- All four real Sato boxes produce **bitwise-identical** train/eval centers,
  local centers, confidence logits, radius outputs and decoder features under
  the same no-gradient execution context. The state remains unchanged.
- Whole-model synthetic 256- and 4096-point grids pass the same exact mode and
  state checks. These are compatibility checks, not capacity/quality claims.
- Actual node-loss backward is finite for every parameter on all four real
  inputs. The radius output row has zero direct loss gradient. There is no
  optimizer update on real data and no fitted model from these CPU checks.
- Fourteen GN-specific tests and twelve shared node tests pass: **26 tests**.
  They cover fresh-state replacement, rejection of trained/incompatible BN,
  same-sample independence from other batch samples, exact mode behavior,
  checkpoint recipe binding, separation of consistency/learnability verdicts,
  changed target denominators and failed/mismatched worker/timing evidence.
- PowerShell launcher parsing passes. No new dependencies/environment changes
  were needed. Generated tests/reports are under ignored `tmp` directories.

This is a functional consistency result. No learned-node improvement has been
measured yet. CPU equality does not replace a check of the actual CUDA operators.

## Manual GPU timing command

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_groupnorm_node_fit.ps1 -Action timing
```

One fresh CUDA process on the local RTX 3070 Ti Laptop GPU performs eight
throwaway updates (two cycles through the four boxes), checks train/eval
consistency at updates 0 and 8, and verifies full model/Adam/RNG serialization
reload in memory. No trained checkpoint is retained. The worker has a 300-second
cap; there is no fallback/retry or full training action triggered by timing.

Logs, source/recipe identity, predictions and reports go to a fresh timestamped
`PN2/tmp/groupnorm_node_timing_*` directory. The printed time estimate for one
2000-update fit uses the second four-box cycle and a 0.8-1.2 heuristic; evaluation,
checkpoint and system overhead remain additional. The old BN fit took about
31 minutes for its Sato arm, but GN changes the kernels, so that is context,
not a new runtime promise. Review this GN timing before launching training.

The old BN timing cannot qualify the GN train action. If source/config changes,
the GN CPU check becomes stale. A fresh CPU check can be written with
`-Action check -Output <new folder>` and supplied via `-CpuPreflight <report.json>`.
The current completed CPU check does not need repeating.

## Later fit and stopping rules

The separate train action requires this control's passed GN timing. Its first
eight updates must replay that GN model/Adam/RNG prefix exactly. Checkpoints
are saved before evaluations with model, optimizer, RNG, recipe/source identity
and immutable completion markers. There is no resume action or automatic cap
extension in this launcher.

Every GPU evaluation compares `.eval()` outputs with the same GN model in
training mode under no gradients. Centers/logits/scores must be bitwise equal.
Consistency files are saved separately; a discrepancy stops the control rather
than allowing substitution of whichever mode scores better. Dropout remains
zero and attention fast paths remain fixed as in the baseline.

At update 2000, the report pairs the new ordinary-inference result with the
saved ordinary-BN result by box identity and verifies equal target denominators.
The original 95% gate remains unchanged. Functional mode consistency and node
learnability remain separate fields; completed execution with a false node gate
is a failed learnability outcome.

If the node gate passes, close this small readiness task and discuss the broader
development comparison. If consistency passes but node learning still fails,
**stop normalization variants** and review the query/assignment formulation.
Do not automatically vary group counts, LR, cutoff, targets or the update cap.
Geometric queries and revised losses are not part of this control. Even a pass
would establish only training-set learnability, not reconstruction quality or
robustness on future microscopy inputs.
