# Completed fit controls and scoring-only development continuation

## Measured fit result

The user completed the new four-box fit diagnostic in 90.34 seconds (terminal
about 93 seconds), output `tmp/curve_transfer_fit_20261104T193753_641182Z`.
These are four training boxes from two neurons; no development claim follows.
The [portable evidence](evidence/curve_transfer_fit_and_interruption_20261104.json)
preserves the full fit report, completed development inference/graph receipts,
CPU parity and the user's interruption trace.

| Same consolidation/forest family | P2 | R2 | F1 at 2 | P6 | R6 | F1 at 6 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen learned dense nodes | 76.16% | 87.64% | 81.49% | 96.71% | 97.54% | 97.12% |
| Raw 2048 inputs, radius-1 consolidation | 7.46% | 72.15% | 13.33% | 24.80% | 97.28% | 37.74% |
| Sato-ranked top 512, radius-1 consolidation | 23.81% | 46.07% | 31.36% | 40.77% | 53.95% | 45.73% |

Raw nodes retained per box: 1723/1868/1747/1865; Sato retains 512 each; dense
retains 202/984/76/316. The graph rule/cost is shared; node count, learned offsets
and confidence selection differ. Learning therefore adds substantial downstream
fitting beyond these two image-only controls, jointly through its representation
and selection. It does not establish which learned component accounts for the
gain or how much survives on untrained neurons. The unfiltered raw cloud is not
already an approximately 80%-F1 skeleton through this fixed forest.

Shortcut flags: dense 6/1080 scorable edges; raw 3/355; Sato 2/283. These
denominators exclude 423/6803/1729 off-reference-endpoint edges respectively,
plus ambiguity and other unknown categories. The small shortcut percentages
are not comparable total error rates and cannot certify topology. In particular,
raw inputs do not become safer because fewer of their edges are scorable.

## What happened to development

The user interrupted after about 70 minutes. Preserved source:
`tmp/curve_transfer_development_20261104T195033_977395Z`.

- All 151 CPU forwards completed, around .4 seconds per box. A complete
  prediction receipt exists. Model state was unchanged, no optimizer was
  created, no training update or CUDA context occurred.
- CPU/GPU parity passed on all four fitted boxes. Maximum coordinate difference
  0.00006104 voxel; maximum score difference 0.000002086; retained sets identical.
- All 453 graph/SWC pairs (151 each dense/raw/Sato) and the aggregate
  `all_inference_complete.json` receipt were written by 19:56:42 UTC, about
  six minutes after launch. Prediction receipt: 19:51:51 UTC.
- The remaining roughly hour was inside reference scoring. The old scorer
  processes all decoded and candidate geometry before writing any score records
  or shortcut files. No development report/shortcut files exist; unknown how
  many in-memory scores were completed. Do not infer development accuracy.
- The original guard's KeyboardInterrupt path killed its owned worker but did
  not persist a cancellation receipt. The user's pasted terminal is evidence
  of cancellation, not a wall-limit failure or model-learning failure.

The original driver, model/scientific config, scorer and saved outputs are
unchanged. The useful inference/graph work is retained. Do not run the original
development command again to recover scoring; that would redo completed work.

## Execution repair, no scientific change

`scripts/score_saved_curve_transfer.py` accepts only the saved, hash-bound source.
It verifies all prediction packets and graph/SWC receipts, then scores only.
It never imports Torch, runs a model, reranks points, builds/re-exports graphs,
filters images, changes a threshold or prepares training.

The original scorer needs exact distances even for far-away samples, although
only <=2/<=6 decisions enter these metrics. The new `bounded_agreement.py` uses
the SAME positive segments, fragment index, .5-voxel arclength samples/weights,
projection residual arithmetic and scoring tolerances. It batches candidate
projection arithmetic and censors distances >6 to infinity. A fragment capable
of supplying distance <=6 must have midpoint distance <=6+maximum half-length;
roundoff expansion applies only to candidate search, never to the tolerance.
This is not approximate nearest-neighbor scoring or relaxed localization.

Before development scoring, require EXACT equality to the completed original
fit report on all 12 graphs, each decoded score and all k=4/6/10 candidate scores
(48 metric dictionaries). Failure stops before development. No metric is chosen
from this parity check. Synthetic parity covers random geometries, exact
boundaries, empty/zero-length curves, subdivisions and large coordinates.

Five short CPU checks pass, including the actual saved-packet scoring boundary
and an interrupted/complete/mismatched scoring-cache fixture. The cache tests
use the authorized PN2/tmp workspace (atomic rename in the sandbox's separate
default temporary directory was denied). Read-only preflight verified all 151
prediction packets and all 453 graph/SWC pairs. These checks do not establish
real-data fit parity or a measured scoring speedup; those happen in the user run.

For every development graph, save both geometry and unchanged reference-path
shortcut diagnostics in an atomically published complete JSON record, then
print progress and time. Interruption leaves completed records reusable. Source/
config bindings must match before any record can be reused. Summary still
requires all 453 graphs. Cancellation now persists a guard receipt.

## User-local command

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\phase1_cpu_env\Scripts\python.exe' -B '.\scripts\score_saved_curve_transfer.py'
```

CPU scoring only, fresh `tmp/curve_transfer_scoring_<UTC timestamp>` output,
hard cap 30 minutes including fit parity. Speedup/completion time is unmeasured;
the command prints every graph, elapsed time and saved status. The prior
70-minute invocation plus this 30-minute cap remains below the original two-hour
allowance in cumulative invoked wall time; no training budget is extended.
There is no automatic retry, cap extension, GPU fallback or next-rung launch.

If a later interruption requires another user-approved continuation, the same
launcher supports explicit `--resume '<prior scoring output>'` to reuse complete
records. No old source artifact is overwritten. That is a capability, not an
instruction to extend this cap or automatically run again.

Transformer work remains parked. Original fit and consolidation gates remain
failed. A sixteen-box decision still awaits complete development/control and
shortcut evidence and a separately agreed prospective protocol.
