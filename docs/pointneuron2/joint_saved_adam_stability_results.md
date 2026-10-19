# Eight-update joint CPU check passes with saved proposal Adam state

19 October 2026. The agreed check completes exactly eight throwaway joint CPU
updates and **passes the declared short stability gate**. Proposal AdamW moments
and counters are restored; GAT AdamW state is fresh. Both models receive joint
gradients and update. No GPU action or retained fit is launched; no trained
checkpoint is retained. The current full-training launcher is unchanged.

## Declared comparison and criteria

The preceding [matched proposal-only comparison](proposal_optimizer_state_results.md)
isolated the effect of preserving Adam state on early geometry drift. This check
reintroduces the original joint loss while retaining that proposal optimizer
history. It uses the same four TRAINING crops, frozen Phase 1 clouds, warm-start
proposal/GAT weights, cyclic order, batch one, seed zero, current-crop normalization,
512-query GAT context, unknown-edge mask and final 0.5 cutoffs as the stopped
joint setup. There is no new node head, pruning, topology loss or broader data.

The [visible recipe](../../configs/pointneuron2/joint_saved_adam_stability.json)
declares CPU/two threads, eight updates, LR 0.0001, weight decay 0.0001, one global
gradient clip at 10, and the unchanged balanced proposal loss plus one balanced
edge-loss term. The original supervision guard remains: stop before updating
when either known edge class is absent on the active training crop. All four
crops are probed after every update, without applying their labels to inference.

Before execution, success required:

1. Complete eight updates without the original supervision stop.
2. Retain all six crop-0/0 matches within two voxels after updates four and eight.
3. Have both known edge-label classes available on every crop after update eight.

All three pass. These are short stability criteria, distinct from the existing
strict automatic reconstruction gate for a retained joint fit.

## Optimizer implementation and evidence capture

One AdamW optimizer keeps the original joint parameter group and identical
options. Saved proposal slots are mapped to the corresponding proposal parameters;
GAT slots start empty. Source parameter order, moment shapes/dtypes, settings and
initial model weights are verified. Proposal counters advance **4,000 -> 4,008**;
GAT counters advance **0 -> 8**, while the new diagnostic budget is exactly eight.
This is a new bounded CPU stage, not an exact GPU continuation or an extension
of the terminal source fit. The source checkpoint's CUDA RNG state is not loaded.

The isolated runner calls the existing joint loss and trust-mask guard. It records
the attempted update, active crop, incoming loss terms, label counts, matched
positions/distances and query/target assignments **before** that loss can stop.
Before each step it records proposal/GAT gradient norms and the global clipping
norm; after the step it records completion and both optimizer cursors. Every
post-update probe is saved. A failure saves the incoming prediction packet and
traceback, without saving a resumable trained model. Source/cache hashes are
verified again at completion.

## Results

| Training crop | Targets | Initially trusted | After four joint updates | After eight joint updates |
|---|---:|---:|---:|---:|
| 0/0 | 6 | 6 | 6 | 6 |
| 0/1 | 12 | 12 | 12 | 12 |
| 92/4 | 16 | 16 | 16 | 16 |
| 86/3 | 418 | 403 | 415 | 414 |

At the end, positive/negative edge-label counts are **5/8, 10/12, 15/34,
399/2,961** respectively. The existing guard is not hit. Both model parameter
sets change; every gradient is present and finite on each update. All eight
updates are clipped: preclip norms range from 211.35 to 505.35. Clipping was
already part of the recipe; it is not evidence of unchanged output geometry.
The run takes 55.81 seconds after setup, with no extension past eight.

Placement is not uniformly preserved at every intermediate point. Crop 0/1
temporarily has 11/12 trusted matches after update one; dense trusted counts are
**403 -> 414 -> 395 -> 365 -> 415 -> 416 -> 411 -> 403 -> 414**.
Crop-0 maximum error is 1.162 voxels initially, 0.751 at update four and 1.521
at update eight. Its mean error increases from 0.956 to 1.263 by update eight.
Other final mean/max errors are 0.760/1.351 for crop 0/1, 0.533/1.453 for crop
92/4 and 0.700/2.635 for crop 86/3. Four dense matches remain outside two voxels.

Confidence is still imperfect. Final accepted assigned + unmatched counts are
**6+10, 12+32, 16+16, 409+3**. Nine dense assigned queries remain rejected, and
the sparse crops still accept many unmatched queries. No full automatic
reconstruction or generalization gate has been passed by these eight updates.

## Interpretation and next decision

Preserving proposal optimizer history permits this short CPU joint sequence to
cross the point where the fresh-state GPU timing stopped and the fresh-state CPU
joint replay lost supervision. The prior proposal-only control provides the
matched evidence for the optimizer-state effect. This run adds evidence that
joint updates can execute for two cycles with that initialization. It does not
prove long-run stability, that graph loss helps reconstruction, or superiority
to PN 1.0. CPU execution is not a GPU timing/reload/memory result.

The next step to discuss is a separate visible GPU timing check using this
optimizer initialization and the same instrumentation. A retained fit would need
explicit checkpoint/resume handling for the two parameter-counter histories and
the unchanged 4,000-new-update cap. The old launcher still starts both optimizers
fresh and must not be used as though this policy were already integrated.
No full trainer, GPU command or retained run is prepared or launched here.
The existing automatic node/graph/SWC and 99% arclength precision/recall gate
remains the destination of the bounded joint experiment.

## Verification and artifacts

Fifteen targeted tests pass, covering mixed-state equivalence to separate AdamW
steps under one global clip, counter mapping, unchanged options, gate failures,
and preservation of incoming counts/predictions on an actual supervision stop.
The first test invocation had a missing temporary-directory parent; a later
new test had a misplaced assertion, corrected in the test file. Both earlier
outputs are preserved; final verification uses `tests_complete.xml`. Neither
issue changed the real-model diagnostic or original runtime sources.

- Visible runner: [check_joint_saved_adam_stability.py](../../scripts/pointneuron2/check_joint_saved_adam_stability.py).
- Raw report: `tmp/pointneuron2/local_runs/joint_saved_adam_stability_20261019/report.json`.
- Tests: `tmp/pointneuron2/joint_saved_adam_checks_20261019/tests_complete.xml`.
- [Compact measurements, gates and provenance](joint_saved_adam_stability_results_seed0/summary.json).

The CPU command below refuses existing output directories and cannot start a
GPU or full fit:

```powershell
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u .\scripts\pointneuron2\check_joint_saved_adam_stability.py
```
