# Hard-negative confidence comparison: implementation and CPU readiness

18 October 2026. The user agreed to the [specified 50/25/25 comparison](hard_negative_objective_proposal.md).
It is implemented as a separate loss, recipe, runner and launcher. **User-launched
GPU timing and the 4,000-update fit are complete.** The
[paired results](hard_negative_query_results.md) reject this variant for the
current protocol; no further learning experiment is prepared. The protocol below
records the agreed run. All completed balanced/weighted-loss sources
and outputs remain preserved. This is a four-training-crop learnability
experiment, not a reconstruction benchmark or a change to the architecture.

## Implemented policy

`0.5 mean_positive_CE + 0.25 mean_all_negative_CE + 0.25 mean_hard_negative_CE`

K=min(P,N), giving 6/12/16/94 hard negatives in the four crops. Recompute their
membership after the unchanged assignment each update, sorting detached used-score
margins, with lower query index resolving ties. Positive queries cannot enter
the hard set. Every unmatched query retains negative supervision. Empty local
annotation remains unknown and is skipped; an all-positive supervised crop uses
its positive mean. For K=N the implementation uses the balanced reduction
directly, ensuring exact dense-crop value and gradient equality at fixed outputs.

The inherited positive weight eight is used only by the unchanged assignment
validator/reference weighted-loss evaluator; it does not weight this objective.
Outer classification/center/radius weights remain 2/3/.2. Confidence still
selects unique skeleton-set representatives; target count is used only in the
training loss and never in automatic proposal selection at inference.

Files:

- [Loss](../../src/pointneuron/models/hard_negative_query_loss.py)
- [Comparison recipe](../../configs/pointneuron2/hard_negative_query_comparison.json)
- [Runner](../../scripts/pointneuron2/hard_negative_query_experiment.py)
- [PowerShell launcher](../../scripts/pointneuron2/run_hard_negative_query.ps1)
- [Targeted tests](../../tests/test_hard_negative_query_comparison.py)

The runner reuses the preserved balanced evaluator for primary gates and R8/R12
candidate diagnostics, then adds the new loss and hard-subset membership. Weighted
reference loss, balanced reference loss and actual optimization loss are named
separately. Their raw magnitudes are not interchangeable quality metrics.
It also records the number, sum and maximum pre-clip gradient norms and counts
above clip ten. These statistics describe the current timing/training/resume
segment; they are not falsely carried over as a complete history after resume.

## Pairing and checks

The primary comparator is the completed balanced fit, pinned to report SHA256
`06781fee1d01b8e8b19d80d82b2791b1cfc46ee56fd921c86c0f54d35e223775`.
Its exact cache is copied, including manifest bytes. Inputs remain 848/2,048/
1,219/2,048 distinct points; targets remain 6/12/16/418. Phase 1 stays frozen.
Fresh seed-zero model, PointNeXt/Transformer, explicit per-crop normalization,
512 queries, batch one, LR .0001, AdamW weight decay .0001, clip ten, float32,
crop order and 4,000 updates match the comparator. No augmentation or development/
test access is introduced. Shared weights mean dense predictions can change
even though its direct classification formula is unchanged.

Twenty-eight targeted tests pass, covering analytic logit-gradient coefficients
at all four target counts, all-negative supervision, positive exclusion, exact
ties, dense loss/gradient equality, unchanged assignments and geometry-output
gradients, unknown/all-positive cases, and all 123 full-model parameter gradients.
A discarded synthetic CPU update has an exact next-update match after disk
checkpoint/optimizer/RNG restore using one CPU thread. A balanced-policy recipe
cannot resume the new state. The checkpoint-before-evaluation-failure path is
also verified. Existing cropnorm/fit-state tests pass in the same invocation.

The real four-crop CPU action passes in 65.0 seconds, verifies all 123 initial
state tensor hashes and reloads with maximum output difference 0.0. Its five
cache files match the comparator byte-for-byte and all 54 captured source hashes
match. It makes zero optimizer updates. The implemented loss also reproduces the
proposal's four saved-score calculations, with maximum absolute difference
4.3e-8 (float32 versus float64 arithmetic). See the
[readiness evidence](hard_negative_query_ready_seed0/summary.json)
for its status, source/cache/initialization checks, reload difference and test XML
digest. Neither these tests nor fresh prediction failures establish learnability.

Checkpoints preserve loss policy, optimizer, RNG, crop cursor and source/cache
identity. They are written atomically before evaluation. Explicit resume uses
a separate folder and cannot extend the original 4,000-update cap. Completed
balanced checkpoints cannot be substituted as warm starts.

## Decision at 4,000

Use the agreed [success/failure rules](hard_negative_objective_proposal.md#decisions-at-the-fixed-cap).
All four selection gates must pass together: accepted assigned counts at least
6/12/16/398, extras at most 0/0/0/20, and curve support within six voxels at
least 95%. Trusted targets and junction counts must not regress from 6/12/16/404
and 0/0/2/76. Report center/radius errors and accepted-and-trusted/edge retention.

Preserving 76/77 dense junctions would still fail the full proposal milestone.
The original combined gates remain primary, with R8 candidates. R12 remains a
separate diagnostic. A selection result alone does not authorize GATv2 training
or establish PN 1.0 improvement. Stop at 4,000; no extension, cutoff search,
K/coefficient sweep or automatic production promotion. Stop on nonfinite values
or provenance/checkpoint failure and preserve evidence for review.

## User-launched GPU commands

The prepared cache, CPU check and user-launched eight-update GPU timing are
complete. Timing verifies matching backend, cache, initialization and source
hashes, zero reload difference, and no retained trained checkpoint. Torch peak
allocated memory is 93.1 MiB and reserved memory 118 MiB, excluding driver/context
and other-process memory. See [timing evidence](hard_negative_query_ready_seed0/timing.json).

The completed timing command was:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_hard_negative_query.ps1 -Action timing
```

The device is the NVIDIA GeForce RTX 3070 Ti Laptop GPU. This timing estimates
**85.5–128.2 minutes**, plus checkpoint IO/system variability. Its four measured
updates take .729/1.431/1.099/1.638 seconds, about 3.4–4.5 times the previous
balanced timing, including 3.7 times on the dense crop with an identical direct
loss formula. There is no contemporaneous baseline replay, so this does not
isolate objective overhead from runtime conditions. The cause is unestablished;
the earlier 19.7-minute fit is context, not a duration promise. No additional
timing or GPU diagnostic was launched during review.

All eight throwaway updates exceed gradient clip ten, with maximum pre-clip norm
616.7. These early updates do not establish late-training behavior or divergence;
the retained fit will record its own segment-local clipping statistics.
Fresh evaluation gates fail as expected after eight updates; this is an execution
check, not a failed or successful learnability comparison.

The agreed settings and cap are unchanged. The separate bounded command is:

```powershell
& .\scripts\pointneuron2\run_hard_negative_query.ps1 -Action train
```

Outputs go to ignored `tmp/pointneuron2/local_runs/hard_negative_query_20261018`
under separate cache/cpu_check/timing/model folders. Existing action folders
cannot be overwritten. Resume, only after interruption, uses explicit
`-Action resume -Checkpoint <full path to this comparison's step checkpoint>`;
weights alone with a fresh optimizer are not an exact continuation.
