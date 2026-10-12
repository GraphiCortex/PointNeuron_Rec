# Completed field train/development pilot and four-control sampling protocol

12 October 2026. The user approved a bounded generalization pilot after the
[four-control training-crop comparison](field_sampling_results.md). Commands are
prepared; **native preparation, GPU timing and training are complete**. The
development four-control sampling comparison is also complete; see the
[sampling results, gains and dilution](field_pilot_sampling_results.md) and
[training results, curves and image limits](field_pilot_results.md).
Commands below record the completed protocol, not a request to rerun it.
Execution tests below use tiny synthetic CPU fixtures, not research results.

**13 October update:** the user completed CPU preparation in seven minutes
(manifest elapsed 416.29 seconds). All 318 native record hashes and raw/normalized
image and supervision-mask contracts were subsequently verified without model
inference. Training: 33 neurons, 211 supervised / 51 unsupervised crops.
Development: seven neurons, 47 supervised / nine unsupervised crops. Every neuron
has usable supervision; zero reserved test records were included. The agreed
30 epochs will perform **6,330 optimizer updates**. This verifies preparation,
not learnability or reconstruction quality. The subsequent timing and training
have now completed; do not repeat actions into their existing output folders.

The intended architecture remains Phase 1 point cloud -> PointNeXt -> Transformer
skeleton proposals -> GATv2 connectivity -> SWC. This pilot tests only a candidate
Phase 1 image guide. Neither the 50/50 selector nor another full pipeline has
been promoted.

## Agreed settings and what this comparison can answer

- Existing partition: **33 training / 7 development / 8 reserved test neurons**.
  Source: `query_implementation_20261011/cache/manifest.json`, 262 train and 56
  development crop boxes. All 318 source crop hashes and the partition hash were
  independently checked before implementation. Each native box is 97 x 97 x 97.
- Fresh existing 88,673-parameter 3D U-Net, base channels 8, GroupNorm. Full
  native images normalized by the observed full-image maximum; raw integer
  intensities are also retained exactly.
- Full continuous SWC segments define the two-voxel tube; distant bright evidence
  beyond six voxels and padding remain unknown. A crop without local trace support
  provides **no** training supervision. Full segments crossing a box count even
  when their endpoints lie outside it. No crop is removed from development reporting.
- Existing loss: balanced positive BCE and top `min(N,3P)` known-negative BCE,
  plus Dice over all known voxels. No model/loss change or augmentation.
- AdamW, learning rate **0.001**, weight decay 0.0001, gradient clip 10,
  **batch size 1**, float32, seed 0, **30 epochs**. Each epoch visits each usable
  training crop once. The usable count is established by native preparation;
  zero query targets alone are not used to classify a crop as background.
- Training/development evaluations and checkpoints at **0/5/15/30**. One crop
  enters CUDA at a time; an eight-record LRU bounds retained CPU crop arrays.
- Sampling uses the **fixed epoch-30 checkpoint**, never a retrospective winner.
  Four controls A/B/C/D, budgets 512/2,048, seeds 0-4, field threshold 0.5,
  eight-voxel cells and 50/50 guide/context are unchanged. All development clouds
  are sealed before reference scoring.

The question is whether candidate expansion retains its reference-coverage gain
on development neurons. Evaluate B minus A at **matched actual unique counts**,
then D minus B and D minus C at equal expanded counts. Report per-neuron means,
chain losses, longest unsupported gaps, proximity and zero-intensity additions.
The old D-minus-C contrast remains recorded; it is not enough to choose a sampler
if expanded random is stronger. References never admit inference points.

Report training and development fit separately. If training fits poorly or is
still improving at epoch 30, discuss a duration extension before concluding that
the concept fails. If training fits but development does not, investigate
generalization and annotation/domain differences. A coverage gain accompanied
by much off-reference support is a tradeoff, not an automatic success. There is
**no automatic advancement gate or parameter sweep**.

This is a patch pilot, not coverage of complete arbors: the image-defined boxes
were originally chosen with the old threshold and capped at eight per neuron.
Field supervision remains weak; annotation proximity is not biological precision.
Dark predicted support may reflect interpolation or hallucination. The seven
development neurons have informed earlier experiments, so they are not an
independent final test. The eight reserved test neurons remain unopened.
Seeds, crop order and backend flags are recorded; exact repeated training has
not been established, and the earlier repeatability caveat remains relevant.

## Completed: CPU native-image preparation command

In the PowerShell terminal you want to watch:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_field_pilot.ps1 `
  -Action prepare `
  -ArtifactRoot "C:\Users\Seywe\Documents\GitHub\PointNeuron"
```

This decodes the original train/development images, verifies provenance and
caches native arrays/labels. It uses CPU and **does not start training**. Each
crop prints progress; the completed run took seven minutes. The native check
confirmed the 51 train and nine development crops without query targets remain
unsupervised under the agreed rule. They are not labeled as verified background.

All actions default to the same experiment directory:

`C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\field_pilot_20261012`

For a different fresh pilot, pass the same `-ExperimentDirectory` to every
action. Existing cache/model/timing/comparison folders are preserved; rerunning
an action into existing output is refused. Incomplete outputs are never consumed
as successful runs. No packages or environments are installed.

## Completed: timing and GPU training commands

After reviewing the preparation counts:

```powershell
& .\scripts\pointneuron2\run_field_pilot.ps1 -Action preflight
```

On the local **RTX 3070 Ti Laptop GPU**, this does four throwaway updates and one
development inference/metric pass. It retains no trained checkpoint. The estimate
includes loading and transfers; checkpoint/overlay writing and sustained GPU
conditions add variability. The earlier timing and complete fit had different
per-update costs, so the new preflight is required rather than promising an old
estimate. Sampling comparison time is separate.

Training is a separate visible command, after the estimate is reviewed:

```powershell
& .\scripts\pointneuron2\run_field_pilot.ps1 -Action train
```

The trainer verifies the preflight matches the cache/config/source/backend.
It prints live batch progress and saves:

- `model/pilot_report.json`: separate training/development metrics at each check,
  per-neuron summaries, crop order, update counts, backend flags and reload checks.
- `model/checkpoints/field_epoch_000.pt`, `005.pt`, `015.pt`, `030.pt`: weights,
  optimizer state and provenance. They are snapshots; resume is not implemented.
- `model/development_fields/`: all final development probabilities as **float32**
  with native origins, plus one readable overlay per development crop.
- `prepare.console.log`, `preflight.console.log`, `train.console.log`.

Finally, after training results are reviewed, the separate comparison command is:

```powershell
& .\scripts\pointneuron2\run_field_pilot.ps1 -Action compare
```

It reloads the fixed final field and requires reproduced scores and **exact
threshold membership**. It saves `comparison/comparison.json`, `metrics.csv`,
all input clouds, per-chain measurements and XY/XZ overlays. No optimizer or
later architecture stage runs. Crops without in-box reference length remain
visible but do not receive invented coverage or precision scores.

## Visible source and execution verification

- [Approved pilot config](../../configs/pointneuron2/field_pilot.json)
- [PowerShell runner](../../scripts/pointneuron2/run_field_pilot.ps1)
- [Native preparation](../../scripts/pointneuron2/prepare_field_pilot.py)
- [Training and preflight](../../scripts/pointneuron2/train_field_pilot.py)
- [Development sampling comparison](../../scripts/pointneuron2/compare_field_pilot.py)
- [CPU cache and metrics](../../src/pointneuron/data/field_pilot.py)
- [Synthetic execution tests](../../tests/test_field_pilot.py)

Tests cover crossing-segment labels, uint16 raw intensity, unknown/no-trace
handling, reserved-data rejection, bounded loading, stale preflight rejection,
the tiny complete 30-epoch CPU path, checkpoint reload and clouds sealed before
development reference scoring. These validate execution contracts; they do not
measure real-neuron generalization.

Verification: **38 targeted tests passed in 32.71 seconds** across the pilot,
existing sampler runner, point selector and field-label/loss contracts. The
PowerShell entry script parses successfully. A final small timing-accounting
fix was subsequently rechecked with the pilot tests.
