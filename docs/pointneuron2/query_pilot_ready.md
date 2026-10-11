# Approved matched train/validation pilot

12 October 2026. The [pooled LayerNorm fit](query_layernorm_results.md) passes
the two-crop learnability check. The user approved preparation of this broader
pilot. Its config, validation diagnostics, pairing checks and local timing
diagnostic are ready. The assistant has not launched the 30-epoch run.

## Frozen comparison

Both heads train fresh on the same corrected Gold166 internal subset, with the
same image candidates, complete curve-target cache and supervision policy.

| Setting | Both arms |
| --- | --- |
| Training | 33 neurons; 262 image-defined crops; 211 annotated crops |
| Development validation | 7 neurons; 56 crops; 47 annotated crops |
| Epochs | 30 per arm; pointwise first, then query |
| Common encoder | DGCNN with only pooled BatchNorm replaced by LayerNorm |
| Selected image inputs | 512 from the same 1,024-candidate pools |
| Query outputs | 512 slots; zero decoder dropout |
| Curve targets | Eight-voxel arc spacing, preserve special/boundary nodes, capacity 512 without truncation |
| Optimization | AdamW, learning rate 0.0001, weight decay 0.0001, batch size 2, gradient clip 10 |
| Data order and augmentation | Same seeds, shuffled batches, selected inputs, rotations/flips and transformed targets/crop centers |
| Checkpoints | Epochs 5, 15 and 30; fixed final epoch is primary |

The query decoder and its set objective differ from the local pointwise head
and objective. This compares those complete proposal methods under a shared
candidate encoder; it does not isolate attention alone or benchmark the final
PN1 reconstruction pipeline. Parameter counts and compute differ between heads.
The inherited losses are unchanged from the preceding learnability checks.

The explicit [pilot config](../../configs/pointneuron2/query_pilot_layernorm.json)
may differ from the immutable cache's base config only by pooled LayerNorm and
zero decoder dropout. The runner rejects unrelated changes. The original
config/cache remain unchanged. Default BatchNorm models and old checkpoints
retain their previous interpretation.

Unknown-reference crops supply no supervised loss. All-unknown batches skip
forward training and optimizer updates; mixed batches train from their annotated
crops. Matching gates verify identical encoder initialization, batches, selected
inputs and update counts between arms. Reserved test neurons remain excluded.

## Visible GPU command

Plan for **approximately 15–25 minutes total** on the NVIDIA RTX 3070 Ti Laptop
GPU. The actual target mix, thermal conditions and background workload can
change runtime. No new cache preparation is needed.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_query.ps1 `
  -Action pilot `
  -LogEvery 25 `
  -Manifest ".\tmp\pointneuron2\local_runs\query_implementation_20261011\cache\manifest.json" `
  -Device cuda
```

The runner prints the device and a fresh directory under
`tmp/pointneuron2/local_runs/query_pilot_<timestamp>_<PID>`. It trains pointwise
for 30 epochs, then query for 30 epochs, with live batch output and validation
summaries. No reconstruction or additional seed starts automatically.

Use `-Action pilot` for this approved configuration. The older `train` action
retains the cache's historical base model settings.

## What to inspect

Each validation epoch records target distance, raw coverage within six voxels,
coverage after score 0.5, accepted-slot counts and accepted sparse-target
proximity. Distances use explicit float64 coordinate differences. Diagnostics
are retained for each annotated crop and summarized per neuron, with both crop
and neuron averages. Console summaries give each neuron equal weight. All seven
validation neurons have annotated crops; nine unknown crops are excluded.

Sparse-target proximity is a diagnostic: duplicates count separately, and empty
accepted sets have undefined proximity. Their zero recall and count remain
explicit. It is not reference-curve precision or final SWC precision. The two
arms' training objectives differ, so comparing their scalar losses alone would
not establish which reconstructs better.

Inspect whether placement, confidence and coverage hold across the seven
development neurons and whether any individual neuron fails. The final epoch
remains primary; earlier checkpoints describe learning rather than selecting a
cross-objective best-loss winner. The [reviewed design](query_decoder_design.md)
defines subsequent raw-proposal, selection-stage and whole-neuron comparison
gates. A successful local pilot still needs those evaluations and replication.

Outputs in the printed directory:

- `console.log` and `launch.json`: live output and exact invocation/status.
- `models/comparison.json`: frozen config, dataset counts, config hash, both arm
  reports and final pairing status.
- `models/pointwise/run.json` and `models/query/run.json`: epoch losses, validation
  diagnostics, data-order/input hashes, actual update counts, runtime and memory.
- Each model folder's `epoch_005.pt`, `epoch_015.pt`, `epoch_030.pt` and final
  `proposal.pt`: inference checkpoints with model config and input provenance.

## Completed preparation and timing evidence

**66 targeted tests passed in 18.05 seconds.** Checks include the frozen pilot
config, rejection of unrelated overrides, unknown-crop exclusion, empty accepted
sets, equal-neuron aggregation, and an actual one-epoch matched LayerNorm loop
for both heads with validation metrics and checkpoint reloads. Existing model,
target, cache, loss, observation and normalization tests also pass.

The real PowerShell `pilot -PreflightOnly` action passes the full cache/config
and partition checks without neural forwards or training. That optional action
can be used to inspect the setup; it is not required before the command above.

The bounded [timing script](../../scripts/pointneuron2/time_query_pilot.py) used
fresh throwaway models: one warmup plus four annotated training updates and
three validation forwards per arm. Selected target counts covered small,
typical and large sets, including training sets of 296/418 and validation sets
of 158/161. It saved no model checkpoints and was not a quality experiment.

Measured pointwise training batches took 0.085–0.108 seconds; query batches
0.062–0.175 seconds. Validation with diagnostics took 0.025–0.039 seconds for
pointwise and 0.023–0.049 seconds for query. Projecting those samples to 131
training and 28 validation batches per epoch for each arm gives approximately
10.3 / 13.4 / 19.8 minutes using sample minima / means / maxima. These are rough
projections, not bounds; all batches are assumed supervised. Startup/checkpoint
overhead and workload variability motivate the 15–25 minute planning estimate.

[Verified preparation summary](query_pilot_checks_seed0/summary.json) records
the config/source hashes, full preflight counts, measured batches and estimates.
Local artifacts remain at:

```text
Preflight: tmp/pointneuron2/local_runs/query_pilot_20261012_014234_6496/models/preflight.json
Timing: tmp/pointneuron2/local_runs/query_pilot_timing_20261012/timing.json
```

Shared algorithms remain in `src/pointneuron`, with visible entry scripts in
`scripts/pointneuron2`. Full-run outputs remain in the printed ignored local
directory. No change to downstream suppression, global budgets or graph rules
is bundled into this pilot.
