# Geometric-query CUDA timing: passed, bounded fit ready

31 October 2026. The user completed
`tmp/geometric_query_timing_20261031_155707`. **Eight updates, exact mode
consistency and model/Adam/RNG reload pass.** Current source/recipe and CPU
preflight validators accept the saved run. No GPU work or fit was launched
by the agent. The [candidate implementation](geometric_query_implementation.md)
and prior [GN baseline failure](groupnorm_node_fit_results.md) remain unchanged.

## Verified measurements

RTX 3070 Ti Laptop GPU, Torch 2.12.1+cu130, FP32 deterministic math attention.
Four fixed Sato training boxes, 2048 input points, 512 geometric queries,
1,688,518 trainable parameters. Each box receives two updates in this timing.

| Check | Result |
|---|---|
| Updates | 8/8, all journal losses/gradients/times finite |
| Ordinary eval versus train-mode diagnostic | Bitwise-equal centers, logits and scores in all four boxes at 0 and 8 |
| Saved prediction packets | Both hashes match; all arrays finite; 24 exact mode-array pairs checked |
| Model/Adam/RNG serialization reload | Exact match reported by the unchanged worker |
| Update-eight model hash | Matches the post-reload terminal evaluation hash |
| Second-cycle mean update time | 1.5608 seconds |
| Peak allocated / reserved CUDA memory | 0.234 / 0.264 GiB |
| Estimated 2000-update time | 41.6-62.4 minutes, plus evaluation/checkpoint overhead |
| User-observed timing command time | 43 seconds |
| Retained trained checkpoints | Zero |

The estimate is the fixed 0.8-1.2 times heuristic from four measured updates,
not a confidence interval. Memory applies only to this small component setup
and includes reload. Relative to GN timing, execution is slower in this
observation; no per-operator profiling establishes which operations or runtime
conditions account for the difference.

## Interpretation

No nodes exceed 0.5 at either evaluation. This does not fail the timing gate:
the purpose is execution/reload/cost, and two updates per box do not test the
declared training-set learnability requirement. `learnability=None` is deliberate.

Unfiltered two-voxel matches change from 4/18, 6/85, 1/7, 6/68 at initialization
to 2/18, 9/85, 1/7, 7/68 after eight updates. Early geometry is mixed, rather
than an established improvement. Low initial localization loss relative to the
old learned-token model includes the geometric starting prior and is not a
learned-quality claim. No cutoff, target, loss or model setting is selected from
these eight updates.

## External wall allowance corrected before fitting

The original controller allows 3600 seconds for a full fit. This timing's upper
update estimate already exceeds 60 minutes, before evaluation/checkpoint I/O.
To avoid a preventable controller timeout, the visible PowerShell train action
now uses `scripts/geometric_query_train_controller.py`, with an external
**5400-second (90-minute) wall allowance**.

The qualified worker file, model, recipe, CPU evidence and CUDA prefix remain
byte-identical. The new parent invokes the same `geometric_query_experiment.py
--action _worker --stage train`, validates the existing CPU/timing evidence,
and cannot change the **2000-update** cap. There is no LR/schedule change,
resume, retry or additional learning time beyond the agreed update count.
Timing still uses its original 300-second cap.

The worker recipe retains the legacy parent's 3600-second timeout field so its
identity remains unchanged. The effective external policy, controller/launcher
hashes, worker command and timing-report hash are recorded explicitly in each
fit's `launch.json` and final report. They are separate from model/checkpoint
recipe identity. This is why another GPU timing is unnecessary.

Twenty focused tests pass after the launcher change: geometric-query contracts
plus four mocked controller tests. They verify immutable worker settings,
current-evidence gating, correct 5400-second external timeout, failure on a
missing worker report, and no retry after timeout. PowerShell parsing passes.
The [visible evidence](evidence/geometric_query_timing_20261031.json) includes
raw report/journal/prediction hashes and the controller validation record.

## Next manual command

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_geometric_query_fit.ps1 -Action train `
  -TimingRun ".\tmp\geometric_query_timing_20261031_155707" `
  -Output ".\tmp\geometric_query_train_20261031"
```

One fresh model, same four training boxes and 18/85/7/68 targets, 500 updates per
box. Same GN8, initial parameter values, 2048 input points, 512 queries, loss,
AdamW LR 0.0001, clipping and cutoff 0.5. Comparator: the completed GN model with
image-independent initial tokens and crop-center position parameterization.

The first eight updates are included in the cap and must exactly replay the
passed timing's model/Adam/RNG prefix. Immutable state files are saved before
evaluations at 0/8/500/1000/2000. A mode discrepancy, prefix mismatch or execution
failure stops the run. A false intermediate node gate does not select settings
or extend training.

Terminal success requires >=95% two-voxel one-to-one node precision **and** recall
in every box under ordinary inference, at cutoff 0.5, with mode consistency.
Geometry and accepted-node quality are reported separately, paired against the
saved GN baseline. A terminal failure closes this candidate and leads to the
explicit supervision/formulation review in the [agreed design](geometric_query_proposal.md),
rather than another automatic tuning loop or GAT handoff. A pass establishes
only four-box training learnability; held-out and final reconstruction quality
remain separate questions.
