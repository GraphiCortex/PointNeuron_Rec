# Corrected-input coordinate pilot: ready to train

**Update:** the user has completed training. See
[the measured results and proposed next comparison](repair_pilot_results.md).
The preparation description below records the state before that run; do not
repeat its training command on this now-trained experiment.

Prepared on 11 October 2026. CPU preparation and cache verification are complete;
no training or reconstruction has been launched for this experiment. The next
step is the visible GPU command below. This is a repaired baseline comparison,
before PointNeXt, the query decoder and GATv2.

## What passed

The corrected Vaa3D reader (`vaa3d-pbd-v2`) prepared 262 training crops from
33 neurons and 56 validation crops from seven neurons, without failures. Neuron
108 supplied six eligible crops; the others supplied eight. Preparation took
approximately 5 minutes 36 seconds. Eight reserved test neurons were not used.

The [saved audit](repair_pilot_ready_seed0/summary.json) verifies frozen code and
configuration hashes, all cache file hashes and split identities, and all 40
source image/reference hashes. Arrays are finite; candidate voxel indices map
exactly to their XYZ coordinates and intensities exceed the recorded cutoff.
All extracted guide vertices are retained. Training guides range from zero to
16,730 vertices; empty guides retain the existing uniform fallback.

| Crops with local reference annotations | Earlier reader/cache | Corrected cache |
|---|---:|---:|
| Training | 143 / 264 (54.2%) | 210 / 262 (80.2%) |
| Validation | 25 / 56 (44.6%) | 47 / 56 (83.9%) |

Both caches use the same neuron partition and identical hashes for all 40 source
image/reference pairs. Crop boxes change because their selection depends on the
decoded images. Reference annotations do not select or filter the crop boxes.
The improved annotation availability is encouraging input evidence, not measured
reconstruction improvement or proof of perfect anatomical alignment.

The 52 training and nine validation crops without local annotations remain in
the cache. They do not receive fabricated background targets. With auxiliary
guide loss disabled, wholly unannotated batches skip optimizer updates; forward
passes can still affect batch-normalization statistics. Validation proposal loss
is normalized over annotated records.

## The controlled comparison

Both arms start from matched random initialization, use matched RNG streams and
augmentation, and train on this same frozen cache for five epochs with batch size
two. They use random selection of 512 inputs from 1,024 candidates, full extracted
guides, and zero auxiliary guide loss. Foreground uses 20% of the decoded image
maximum. Mixed candidate pools and farthest-point selection remain diagnostics;
neither was promoted into this cache.

- `repaired_raw_control`: absolute XYZ coordinate features.
- `repaired_local`: centered, normalized local coordinate features.

Voxel offsets and radii retain their physical coordinate units in both arms.
Only proposal-coordinate features differ. The 512-input budget stays fixed to
isolate this comparison; it does not decide the final PN 2.0 input budget.

## Run in your terminal

The checked device is the local NVIDIA GeForce RTX 3070 Ti Laptop GPU, with
8 GiB total memory and about 6.95 GiB free at preflight. The environment reports
PyTorch `2.12.1+cu130` and CUDA available. Preflight took zero optimizer steps.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\sampling_ablation.py --stage train `
  --experiment "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\repair_20261011_130158" `
  --device cuda
```

This runs the two arms sequentially and prints each batch live. Earlier five-epoch
arms took roughly 0.5–8 minutes each; corrected-cache runtime is not yet measured.
This command does not launch whole-neuron reconstruction.

Experiment directory:

```text
C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\repair_20261011_130158
```

It already contains `experiment.json`, `cache/manifest.json` and `cache.log`.
Training adds `repaired_raw_control.log`, `repaired_local.log`, and each arm's
`proposal.pt` and `run.json` inside its named folder. These generated files remain
in ignored `tmp`; scripts, configurations and this report are tracked and visible.
The runner refuses changes to frozen source/configuration or cache hashes.

## How we will judge the result

First confirm that pairing checks pass, both arms finish, and losses remain
finite. Compare annotated-normalized validation proposal objectives on the same
cache, then inspect proposal confidence at scores 0.5 and 0.85 with
`scripts/pointneuron2/inspect_ablation_confidence.py`. Better loss alone does not
establish better reconstructed neurites. We will inspect those results together
before agreeing on the longer whole-neuron reconstruction comparison, where
spatial coverage, topology, failures and runtime matter.

The previous reader's training losses cannot serve as a clean before/after
comparison because decoded images, boxes and annotation availability changed.
The raw/local arms provide the matched comparison for coordinate handling.

## GitHub

Verified groundwork and diagnostics have been pushed to the
[pointneuron2 branch](https://github.com/GraphiCortex/PointNeuron_Rec/tree/pointneuron2).
The user has authorized ongoing commits and pushes. The original parent checkout
remains on `main`; generated data, environments and checkpoints stay ignored.
