# Five-epoch paired objective continuation

3 November 2026. The user agreed to prepare this comparison after the
[candidate objective](consistent_node_objective.md) passed manual GPU timing.
The launcher and CPU proof are complete. **The agent has not launched GPU
training.** This is a separate experiment; the original failed 20-epoch study
and its stopping rules remain unchanged.

## Question and fixed comparison

Does the candidate objective improve actual node accuracy more than simply
giving the original objective additional training, under an identical warm start?

| Setting | Both arms |
|---|---|
| Origin | Completed 2048-point epoch-20 checkpoint, historical update 9200 |
| Restored state | All model weights, Adam moments/update clocks, RNG streams |
| Difference | Original objective versus consistent per-query/normalized objective |
| Input/model | Same original-image Sato clouds, GN8 native PointNeXt, geometric-query Transformer, 512 slots |
| Training data | 33 neurons / all 460 supervised boxes; unknown boxes abstain |
| New training | Five epochs / 2300 updates **per arm**; maximum history 11500 |
| Order | Exact same frozen first-five-epoch schedule, replayed as new epochs 1–5 |
| Optimizer | Same saved AdamW, LR 0.0001, zero weight decay, clip 10 |
| Inference | Ordinary GN inference, unchanged cutoff 0.5 |
| Terminal evaluation | All 415 development boxes and all 1964 training boxes |
| Other stages | No radius/GAT/topology training, source filtering or SWC claim |

No test/reserved neuron enters the loader or gradient training. Development is
the repeatedly exposed seven-neuron development cohort. No augmentation,
scheduler, new target convention or foreground relabeling is introduced.

The old optimizer moments were learned under the original loss. Keeping them
identical controls the starting state, but does not establish that they are
optimal for the candidate. This tests a specific warm-start protocol, not the
best possible training of either objective. Five additional epochs also do not
settle eventual convergence or prove a root cause if little changes.

## Completed execution evidence

The user's GPU timing at
`tmp/consistent_objective_timing_20261103_143451/report.json` passed for both
methods. Initial model/Adam/RNG signatures agree exactly and next-update
serialization/reload is exact. Excluding startup, mean update times were
1.495 seconds for baseline and 1.533 seconds for candidate. Allocated peaks
were about 238.7 MiB each. Clipping occurred on 8/8 versus 2/8 updates.
These are four-box execution measurements, not quality measurements.

The [visible GPU evidence](evidence/consistent_objective_gpu_timing_20261103.json)
and [continuation CPU evidence](evidence/consistent_node_fit_cpu_20261103.json)
bind the configuration, source and origin checkpoint. CPU verification checks:

- Full five-epoch schedule, 33 training identities, no skipped/duplicate/unknown exposures.
- Full-model model/Adam/RNG next-update replay for **both** objectives on one real training box.
- Rejection of an epoch checkpoint whose Adam clock disagrees with its cursor.
- No CUDA initialization and no retained trained CPU checkpoint.

The real-box CPU proof executes three throwaway optimizer calls per method:
one first update and two executions of the next update for replay verification.
These calls do not change the origin checkpoint or represent a quality fit.
67 CPU tests pass across the candidate, continuation, model, state, evaluator
and process-control contracts. A completed-stage recovery is tested to perform
zero additional optimizer calls. CPU mode cannot be redirected to a GPU worker.

## Manual GPU command

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_consistent_node_fit.ps1 -Action train
```

Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU. Baseline runs first, then a
separate candidate worker. **Plan approximately 3–4 hours for both arms**, based
on measured update costs plus prior full-grid evaluation. This remains a
provisional allowance: laptop power, thermal behavior and scoring costs vary.
Each worker has a separate **four-hour wall limit including evaluation**; the
scientific cap remains 2300 new updates regardless of wall time.

Output defaults to `tmp/consistent_node_fit_train_<timestamp>`. Each method has
its own console log, update journal, launch record, immutable checkpoints and
terminal predictions/metrics. Full checkpoint plus completion marker/hash is
saved at stage zero and every 460 new updates, **before** terminal evaluation.
Adam clocks at stage k must equal 9200+k. No file is overwritten and no worker
automatically retries or extends its cap. Execution failure stops the parent;
scientific screening failure still allows the prespecified other arm to run.

Initial accuracy is taken from the immutable original epoch-20 evaluations:
the full GPU state signature must equal the state verified in passed timing.
Those old scores are not selected or retuned. Terminal metrics use the same
existing full-grid evaluator and training-defined thin/dim profile cutoffs.

## Verdict and stopping rules

The paired report provides epoch-20 origin scores, each terminal score, each
arm's change from origin, and candidate-minus-baseline differences. Training
precision/recall use reference-present boxes; development uses the entire image
grid. These two precision domains are explicitly different. Unknown outputs
remain disclosed in the full-grid results, not certified biological negatives.

Prospective **relative pilot screening** requires:

1. Candidate training precision and recall both exceed the unchanged-objective continuation.
2. Development F1 exceeds that control without reduced development recall.
3. Development continuous-curve support at two voxels does not decline.

Sparse/dense count strata (1–16, 17–64, 65–128, 129–512 reference nodes), all-query
recall, structural roles, six-voxel diagnostics, duplicate counts and apparent
thin/dim profiles remain available. Count-stratum precision/recall are pooled
within the labeled stratum; primary metrics pool within neurons before taking
macro means. Unmatched predictions have no certified reference caliber label.

The original absolute training/development engineering gates are also reported
per arm. Passing relative screening while both models remain weak **does not**
qualify the proposal stage for production or establish final graph/SWC quality.
Smaller gradients, lower candidate loss, or fewer accepted nodes are insufficient.

No best-development checkpoint selection and no automatic extension follow a
failed screen. If the candidate loses sparse positives, fails to improve actual
geometry, or produces invalid execution/state, preserve the outcome and discuss
the next decision. Do not change targets, thresholds or optimizer state mid-run.

## Explicit recovery only

Use a completed checkpoint with its `.complete.json` marker, from the same
method/configuration, and a **new output directory**. Recovery consumes only
the remaining updates within the original 2300 cap. Example with a placeholder
to replace by an actual checkpoint path:

```powershell
& .\scripts\run_consistent_node_fit.ps1 -Action resume -Method consistent `
  -Checkpoint "C:\path\to\consistent\checkpoint_stage000920.pt"
```

Recovering stage 2300 performs terminal evaluation only, with zero SGD. A
single-arm recovery report deliberately has no paired verdict; combine it with
the other verified arm only after checking common origin, bindings and terminal
cursors. Recovery never resets Adam or changes the objective/method identity.

Implementation: [driver](../scripts/fit_consistent_node_objective.py),
[launcher](../scripts/run_consistent_node_fit.ps1),
[configuration](../configs/consistent_node_fit.json).
