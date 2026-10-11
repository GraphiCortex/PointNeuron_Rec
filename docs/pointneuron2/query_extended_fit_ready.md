# Approved 2,000-update fixed-crop comparison

Completed follow-up: [GPU results and the fixed-weight pooled-normalization audit](query_extended_fit_results.md).
The preparation protocol below is retained for reproducibility.

12 October 2026. The unchanged longer fit is approved and prepared for the
user's visible local terminal. The 2,000-update GPU experiment has not been
launched by the assistant.

The [500-update dropout-zero result](query_dropout_zero_results.md) gives 95.8%
raw target coverage but only 29.2% coverage after the fixed score threshold of
0.5. The query classification loss is still falling. This comparison asks
whether more updates improve confidence and the selected set before deciding
on a separate normalization or decoder change.

## Fixed experimental settings

Both pointwise and query models start fresh and receive **2,000 optimizer
updates each**, with inference checks at **500, 1,000 and 2,000**. This is an
extension of the update budget, rather than a resume of the saved 500-update
checkpoint. It uses the same first two annotated training crops from neuron 0,
the same selected image inputs and curve targets, seed, initialization, encoder,
losses and learning rate. Query decoder dropout remains zero. No augmentation
is added. The immutable cache and base configuration remain unchanged.

Every saved check uses ordinary inference with the learned BatchNorm running
statistics. It records all-slot target distance/coverage, coverage after score
0.5, accepted slot counts and evaluation loss components. Raw centers, logits
and radii are retained for later confidence/duplicate inspection. We do not use
training batch statistics as an inference workaround. The fixed final update
remains the primary result; intermediate checks describe the trajectory.

This measures learnability on fitted training crops, not validation or
whole-neuron reconstruction quality. The pointwise arm is a matched local
comparator, not a full PN1 reconstruction benchmark. Neither arm is promoted
from this experiment alone.

## Run visibly on your GPU

Expected duration: **roughly 2–3 minutes** on the NVIDIA RTX 3070 Ti Laptop GPU,
from the prior measured update times, plus variable startup/checkpoint overhead.
No cache preparation is needed.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_query.ps1 `
  -Action smoke `
  -Steps 2000 `
  -EvalSteps 500,1000,2000 `
  -LogEvery 50 `
  -DecoderDropout 0 `
  -Manifest ".\tmp\pointneuron2\local_runs\query_implementation_20261011\cache\manifest.json" `
  -Device cuda
```

The runner prints the device and a fresh local directory under
`tmp/pointneuron2/local_runs/query_smoke_<timestamp>_<PID>`. Live output appears
every 50 updates and at every saved inference check. That directory holds
`console.log`, `launch.json` and `models/comparison.json`. Each arm's directory
contains:

- `run.json`: settings, input/source hashes, individual training losses and saved
  evaluation records; milestone progress is written while the run continues.
- `step_000500.pt`, `step_001000.pt`, `step_002000.pt`: inference checkpoints with
  their actual optimizer update count in provenance. These are not resume files
  with optimizer state.
- Corresponding `step_<update>_proposals.npz`: predictions, selected inputs,
  crop centers, target nodes and target masks; no pickled objects.
- `proposal.pt`: the fixed final model, reloaded for exact prediction verification.

The paired checks require identical initial encoder weights, cache/partition
hashes, inputs, update counts and saved milestones. No next stage starts
automatically.

## Preparation checks

**31 targeted tests passed in 25.26 seconds.** These cover the snapshot
regressions below as well as target geometry, matching, unknown-crop handling,
cache/partition guards, paired training and native checkpoint reloads.

The observation helper preserves CPU/device random state and restores every
module's training mode. Inference runs without gradients and never advances
BatchNorm or the optimizer. Regression checks compare a four-update run with
and without an intermediate observation for both arms, including stochastic
query training: final weights, BatchNorm buffers, optimizer state, random state
and per-update losses are exactly equal. Intermediate checkpoints also reproduce
their saved centers, logits and radii exactly after reload.

The actual PowerShell wrapper completed a four-update **CPU** check on the real
two-crop cache with milestones 2/4 and dropout zero. Both arms completed, final
checkpoint reloads were exact, and pairing passed. Artifacts are at:

```text
tmp/pointneuron2/local_runs/query_smoke_20261012_004608_2580/models/comparison.json
```

The short preparation run verifies execution; it is not the requested fit or
evidence of reconstruction improvement. After the GPU run, inspect the saved
500/1,000/2,000 trajectory together, particularly score-filtered support and
whether extra accepted slots correspond to targets or duplicate/off-target
proposals. A stalled selected-set fit would justify a separate controlled
normalization/decoder comparison. The full 30-epoch pilot remains pending.
