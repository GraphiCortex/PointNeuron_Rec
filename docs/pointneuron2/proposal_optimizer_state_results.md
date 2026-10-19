# Saved proposal optimizer state prevents the early crop-zero collapse

**Follow-up:** the agreed [eight-update CPU joint check](joint_saved_adam_stability_results.md)
also passes its declared stability criteria with saved proposal Adam and fresh
GAT Adam state. This does not complete GPU readiness or the retained joint fit.

19 October 2026. The agreed four-update CPU comparison is complete. **Restored
AdamW state passes the declared crop-0 stability criterion; fresh state fails.**
This isolates a consequential optimizer-restart effect in this bounded control.
The current joint fit remains stopped; no joint optimizer policy has been changed
or promoted and no new GPU run has been launched.

## Controlled comparison

Both arms start with exactly the same balanced-fit proposal weights, frozen
GAT weights, four cached TRAINING crops and initial predictions. Only the
proposal objective trains: the existing balanced confidence/center/radius loss.
The frozen GAT is used to probe the existing graph interface, not optimized.
All original runtime source/cache hashes verify before and after the comparison.

The [visible recipe](../../configs/pointneuron2/proposal_optimizer_state_control.json)
declares CPU, two threads, seed zero, four cyclic updates, LR 0.0001, weight decay
0.0001, global gradient clipping at 10, unchanged normalization and thresholds.
The arms have identical optimizer parameter groups and all options, copied from
the original checkpoint. Fresh state has no moments/counters; restored state
copies the saved first/second moments and step counters at update 4,000.
Parameter ordering, moment shapes/dtypes/finiteness and source cursors are
validated. Neither arm inherits another arm's updated weights.

This is an explicitly bounded **new diagnostic stage**, not an exact continuation
of the terminal GPU run or an extension of its 4,000-update fit. The restored
optimizer cursor advances 4,000 to 4,004; the fresh cursor advances zero to four.
No source checkpoint is edited or new trained checkpoint retained. The original
CUDA RNG state is not restored on CPU. Both arms use the same fresh CPU seed;
the model has zero dropout. CPU results are not exact GPU-trajectory replays.

The criterion was declared before execution: retain all six crop-0/0 assigned
matches within two voxels after one complete four-crop cycle. Every crop is
probed at initialization and after every update; unknown graph labels remain
unknown. Because the objective is proposal-only, missing graph-label classes
are recorded, without redefining the existing joint trainer's stop rule.

## Measured outcomes

| Training crop | Targets | Initially trusted | Fresh state after four | Restored state after four |
|---|---:|---:|---:|---:|
| 0/0 | 6 | 6 | 0 | 6 |
| 0/1 | 12 | 12 | 4 | 12 |
| 92/4 | 16 | 16 | 16 | 16 |
| 86/3 | 418 | 403 | 343 | 413 |

The fresh crop-0 distance probes reproduce the prior proposal-only CPU control
**exactly**. Initial weights, initial predictions and optimizer options match
between arms. Even the first preclip gradient norm is identical, 467.1667;
the resulting updates differ because the optimizer states differ.

| Updates completed | Fresh crop-0 trusted / 6 | Restored crop-0 trusted / 6 | Fresh maximum error, voxels | Restored maximum error, voxels |
|---:|---:|---:|---:|---:|
| 0 | 6 | 6 | 1.162 | 1.162 |
| 1 | 0 | 6 | 3.251 | 1.014 |
| 2 | 0 | 6 | 5.579 | 0.908 |
| 3 | 0 | 6 | 6.720 | 0.777 |
| 4 | 0 | 6 | 7.237 | 0.576 |

At the end, restored crop-0 graph supervision retains five known positive and
eight known negative edges; the fresh arm has zero of both. On crop 0/1,
fresh state leaves two positive and zero negative edges, while restored state
has ten positive and twelve negative edges. Both arms complete exactly four
updates; both leave GAT weights unchanged. The full comparison takes 53.40 seconds
after setup. Three targeted optimizer-transfer tests pass, including source-state
immutability, invalid mapping rejection and identical next-update behavior
against the ordinary AdamW loader.

## Limits and decision

The intervention changes saved moments **and** bias-correction step counters
together; this does not isolate their individual effects. It establishes that
discarding the saved optimizer state matters for this four-update CPU task, not
that every fresh restart fails or that preserved state always solves joint
optimization. The earlier GPU timing failure lacks the final state needed for
an exact replay of its failed fifth attempt.

Restored state is not uniformly stable across all intermediate crop probes:
dense trusted counts are **403 -> 413 -> 399 -> 364 -> 413**; crop 0/1 temporarily
has 11/12 trusted nodes after the first update. Geometry is not uniformly better:
restored final mean matched error increases from 0.688 to 0.809 voxels on crop
0/1, 0.395 to 0.611 on crop 92/4, and 1.006 to 1.045 on the dense crop. This is
why the final trusted-node count alone cannot be called reconstruction improvement.

Confidence remains unsolved. After four restored-state updates, accepted assigned
+ unmatched counts are **6+11, 12+32, 16+17, 412+3**. No automatic reconstruction
gate, generalization result or improvement over PN 1.0 follows from this control.

The supported next step for discussion is a **short joint stability check that
preserves proposal optimizer state**, with graph optimizer initialization stated
explicitly, the same losses/masks/cutoffs, and failure-time label/position/update
logging. This isolates the proposed change against the failed fresh-state joint
setup. GAT initialization, mixed optimizer cursors and checkpoint semantics must
be made concrete before implementation. Passing a short CPU check would still
require a separate visible GPU timing/reload/memory check before a retained fit.
Do not change the architecture, relax the supervision guard, extend the fit or
launch the full 4,000 updates on the strength of this proposal-only result.

## Reproduction and evidence

Visible implementation:
[compare_proposal_optimizer_state.py](../../scripts/pointneuron2/compare_proposal_optimizer_state.py).
The command is CPU-only, refuses to overwrite existing evidence, and retains no
trained checkpoint:

```powershell
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u .\scripts\pointneuron2\compare_proposal_optimizer_state.py
```

Raw report: `tmp/pointneuron2/local_runs/proposal_optimizer_state_20261019/report.json`.
Targeted tests: `tmp/pointneuron2/optimizer_state_checks_20261019/tests.xml`.
See [compact measurements and provenance](proposal_optimizer_state_results_seed0/summary.json).
