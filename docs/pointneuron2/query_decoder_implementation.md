# Query decoder implementation and execution checks

11 October 2026. The reviewed target builder, query decoder, matching loss,
checkpoint format, inference adapter and paired local runner are implemented.
The implementation checks pass. A short CPU fit shows learning, but has **not
fitted the two training crops well enough to proceed to the full pilot**.
No GPU training or new whole-neuron/SWC evaluation was launched in these checks.

## Code to read

| Responsibility | Local source |
| --- | --- |
| Reference-curve geometry and crop targets | [curve_targets.py](../../src/pointneuron/data/curve_targets.py) |
| Cache validation and batching | [query_cache.py](../../src/pointneuron/data/query_cache.py) |
| 512-slot attention decoder | [query_proposal.py](../../src/pointneuron/models/query_proposal.py) |
| Set matching and per-crop losses | [query_loss.py](../../src/pointneuron/models/query_loss.py) |
| Shared encoder, checkpoints and graph features | [query_pipeline.py](../../src/pointneuron/models/query_pipeline.py) |
| Derive the common target cache | [build_query_cache.py](../../scripts/pointneuron2/build_query_cache.py) |
| Train/check both heads with matched inputs | [train_query_comparison.py](../../scripts/pointneuron2/train_query_comparison.py) |
| Image-only proposal inference | [aggregate_query_proposals.py](../../scripts/pointneuron2/aggregate_query_proposals.py) |
| Visible local stage runner | [run_query.ps1](../../scripts/pointneuron2/run_query.ps1) |
| Comparison settings | [query_comparison.json](../../configs/pointneuron2/query_comparison.json) |

The existing DGCNN encoder remains the first comparison's encoder. Both arms
start from identical encoder weights and receive identical selected inputs.
The query head uses native centers, objectness and radii; its proposals are not
represented as pointwise offsets. The shared graph adapter attaches the nearest
input point's encoder features and records that association distance.
The decoder and set objective change together, as specified in
[the reviewed design](query_decoder_design.md). This experiment will not isolate
attention alone. PointNeXt, learned guide sampling and GATv2 are separate work.

Targets follow the full reference forest before clipping to crops. Roots,
leaves, junctions and boundary crossings survive arc resampling. Target rows
have synthetic IDs and parent `-1` because this head learns a set, not edges;
these parent values must not be interpreted as connectivity supervision.
Radius supervision uses annotated voxel-coordinate radii, distinct from the
downstream suppression radius. Empty-reference crops remain unknown and do
not supply supervised losses. Entirely unknown batches skip both forward
training and optimizer updates. Unmatched slots in reference-containing crops
are trained as unused: incomplete references can therefore penalize real
unannotated neurites. This remains a reference-agreement assumption.

## Prepared data and execution evidence

The derived cache is local at:

```text
tmp/pointneuron2/local_runs/query_implementation_20261011/cache/manifest.json
```

| Partition | Neurons | Crops | Reference-containing | Unknown | Largest target set |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 33 | 262 | 211 | 51 | 418 |
| Validation | 7 | 56 | 47 | 9 | 161 |

All 318 original image-candidate arrays are exactly equal in the derived cache.
Original and derived file hashes pass. Target spacing is 8 voxel arc units,
with capacity 512 and a hard overflow error; targets are never silently capped.
Reserved test neurons were excluded. This preparation derives labels from
train/validation SWCs; those labels never enter image-only prediction.

**26 targeted tests passed in 26.72 seconds.** Checks cover target geometry
against the independently frozen audit, overflow, masked targets, unique
matching, translation and input permutation, augmentation of crop centers,
checkpoint reloads, common graph-feature association, cache/partition guards,
original image-sampling parity and an actual one-epoch paired loop on synthetic
records. The CPU checks also reload their actual saved checkpoints and reproduce
centers, scores and radii exactly. Pairing checks confirm identical initial
encoder weights, inputs and update counts.

The inference adapter and sampling helpers were checked. A new whole-volume
inference run and downstream reconstruction have **not** yet been evaluated.
The machine-readable evidence, hashes and source snapshots are in
[query_decoder_checks_seed0/summary.json](query_decoder_checks_seed0/summary.json).

## What the short fit actually shows

Both checks use the same first two reference-containing training crops from
neuron 0, fixed inputs, no augmentation and CPU execution. They are fresh runs
at 20 and 100 optimizer updates per arm, not continued checkpoints.

| Arm / updates | Initial mean target distance | Final mean target distance | Initial coverage within 6 voxels | Final coverage within 6 voxels |
| --- | ---: | ---: | ---: | ---: |
| Pointwise / 20 | 2.89 | 2.90 | 87.5% | 87.5% |
| Query / 20 | 52.20 | 52.01 | 0% | 0% |
| Pointwise / 100 | 2.89 | 2.86 | 87.5% | 79.2% |
| Query / 100 | 52.20 | 9.06 | 0% | 33.3% |

These are evaluation-mode distances to **any output slot**, averaged equally
across the two fitted crops. They do not filter confidence, measure specificity,
or establish validation/reconstruction improvement. The pointwise head starts
near its input points, while the query head starts near the crop center, so the
starting geometry differs substantially. Training losses between arms are not
directly comparable because their assignment/objective rules differ.

The query head's 100-update training loss drops from 151.53 to 17.03, and its
center loss from 50.22 to 5.43. This supports a working gradient path and a clear
learning trend. Its 33.3% final training coverage is still poor. We should inspect
a longer fixed-crop fit before spending time on the 30-epoch comparison.

Artifacts remain in ignored local storage:

```text
tmp/pointneuron2/local_runs/query_cpu_smoke_20261011/models/
tmp/pointneuron2/local_runs/query_cpu_fit100_20261011/models/
```

Each arm has `run.json` and `proposal.pt`; each run has `comparison.json`.

## Next visible GPU command

Run this in your PowerShell terminal. It performs **500 optimizer updates on
the same two fixed training crops for each arm**, with live output. It does not
start the full dataset training or reconstruction afterward.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_query.ps1 `
  -Action smoke `
  -Steps 500 `
  -Manifest ".\tmp\pointneuron2\local_runs\query_implementation_20261011\cache\manifest.json" `
  -Device cuda
```

The runner prints the actual CUDA device; the previously used local device was
the NVIDIA RTX 3070 Ti Laptop GPU. GPU duration for this decoder is not yet
measured. CPU medians were 0.21 seconds/update for pointwise and 0.57 for query,
equivalent to roughly 6–7 minutes for 500 updates each, excluding startup.
The GPU run will provide its own observed timings.

A fresh `tmp/pointneuron2/local_runs/query_smoke_<timestamp>_<PID>` directory
contains `console.log`, `launch.json` and `models/comparison.json`, plus each
arm's report/checkpoint. Send the final output when it finishes; we can inspect
geometry and confidence on your saved outputs before deciding whether the full
pilot is justified. A falling training loss alone is insufficient.

The `train` stage is implemented for the agreed 33/7 split, 30 epochs per arm,
snapshots at epochs 5/15/30 and the final epoch as the primary checkpoint.
It is prepared, **not launched or recommended yet**. Whole-neuron reconstruction
and the reviewed quality gates come after learnability is established. None of
the current checks establishes an improvement over PN1 or commercial readiness.
