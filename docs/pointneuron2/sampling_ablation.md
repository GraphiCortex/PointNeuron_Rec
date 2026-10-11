# Matched sampler pilot

**Historical recipe:** do not repeat this recipe for new training. The input
decoder and intensity-range assumption were incorrect. See
[the repairs and next comparison](root_cause_findings.md). Old caches and
checkpoints remain available for historical inspection.

The user has completed all three five-epoch CUDA training runs and all 21
whole-neuron reconstructions. See [the reconstruction results](sampling_ablation_reconstruction_results.md)
for measured spatial/topology scores and the proposed next diagnostic. See
[the pilot results](sampling_ablation_pilot_results.md) for validation losses,
confidence checks and the next reconstruction command. The staged commands below
remain the experiment specification and reproduction instructions.
This experiment follows the failed noise-aware preprocessing comparison.
It tests whether guide-based sampling helps the existing proposal model before
we spend more time refining threshold masks or move to PointNeXt.

## The comparison

| Arm | Input selection | Auxiliary guide Chamfer weight |
|---|---|---:|
| `random` | Uniform random, without replacement | 0 |
| `learned_no_aux` | Guide-distance prior with trainable global alpha | 0 |
| `learned_aux` | Same guide-distance sampler | 0.05 |

All arms start from identical fresh encoder/proposal weights, use one immutable
cache, and run **five epochs each, sequentially**, with seed 0. They use the
existing DGCNN and proposal head. This is a short internal pilot; it is not a
matched-duration comparison against the completed 30-epoch PN 1.0 baseline.
Five epochs may still leave all models below useful proposal confidence.

The shared cache covers the frozen **33 training and 7 validation neurons**,
with up to eight image-defined patches per neuron. The eight test neurons stay
reserved. A missing source neuron or cache failure stops the experiment.

Inputs keep the existing fixed threshold fraction **0.2**, uniform **1,024**
candidate rows, **512** selected encoder inputs, radius/stride **48**, batch
size **2**, and **every extracted guide vertex**. The five-epoch override is
explicit in each training report; the reusable cache config still defaults to
30 epochs for other runs. SWC references remain separate from foreground,
guide extraction and candidate selection. Experimental noise/Otsu masks are
not promoted into this cache or inference.

All image-defined patches are retained. No local SWC nodes does not establish
background: such patches contribute no proposal supervision. The auxiliary
arm can learn from their guides. Without that loss, a wholly unannotated batch
does not advance the optimizer, although the shared training forward pass
still runs. This difference is part of what the auxiliary-loss arm tests.

The trainer isolates random streams for loader order, augmentation and point
selection. Reports save starting-weight, cache, order and augmented-candidate
hashes; the runner checks that they match across all arms. Checkpoints are
selected by the same annotated validation proposal objective, not each arm's
different total loss. Chamfer is skipped, and recorded as `null`/`off`, in the
two arms where it is disabled. Its zero display is not a measured perfect fit.

Random versus `learned_no_aux` tests the guide prior plus learned alpha together.
It does not isolate learned alpha from an otherwise frozen guide prior. If this
pilot supports proceeding, a frozen-alpha control and repeated seeds would be
needed before claiming that learning itself improves reconstruction. Alpha is
one scalar; this sampler cannot recognize individual neurites semantically.

## Commands in your terminal

The configured Python environment is local, and the training/evaluation commands
request CUDA. Every training batch prints progress. No reliable full-run duration
has been measured yet; the cache stage is CPU work, and full-guide sizes vary.

First prepare the common cache. This command **stops before training**:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

$ablationRun = Join-Path $PWD.Path ("tmp\pointneuron2\local_runs\sampling_ablation_" + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& .\scripts\pointneuron2\run_local.ps1 `
  -Action ablation-prepare `
  -ArtifactRoot "C:\Users\Seywe\Documents\GitHub\PointNeuron" `
  -OutputDirectory $ablationRun
```

After `Status: prepared`, inspect the printed record counts and
`$ablationRun\experiment\experiment.json`. We can review that output together
before training. In the same terminal, the following command explicitly launches
all three five-epoch CUDA runs; it stops before reconstruction:

```powershell
& .\scripts\pointneuron2\run_local.ps1 `
  -Action ablation-train `
  -ArtifactRoot "C:\Users\Seywe\Documents\GitHub\PointNeuron" `
  -Experiment (Join-Path $ablationRun 'experiment')
```

If you reopen the terminal, set `$ablationRun` to the actual preparation run
directory printed earlier. No outputs are overwritten; failed runs and logs
are preserved. The runner currently has no automatic resume. A failed preparation
or partially trained experiment needs inspection before choosing a fresh run.
Changes to source/settings after preparation require a fresh experiment.

After discussing the training output, this command runs full-neuron inference,
connectivity and SWC export for **all seven validation neurons in each arm**, then
scores spatial and topology metrics:

```powershell
& .\scripts\pointneuron2\run_local.ps1 `
  -Action ablation-evaluate `
  -ArtifactRoot "C:\Users\Seywe\Documents\GitHub\PointNeuron" `
  -Experiment (Join-Path $ablationRun 'experiment')
```

Evaluation restores each checkpoint's sampler automatically. All downstream
settings are shared and saved in `configs/pointneuron2/sampling_ablation.json`:
the existing geodesic initializer, adaptive connected coverage selection,
graph score cutoff 0.85, distance NMS 18 and maximum 128 graph nodes. Proposal
aggregation uses the existing 0.5 cutoff and up to 256 coverage patches per
whole neuron, as implemented in `run_end_to_end.py`. No old proposals are reused.
Validation image/reference hashes are checked against the cache.

The evaluation stage stops if reconstruction fails or either metric report lacks
any validation neuron. It preserves the failure report rather than silently
averaging the surviving cases. Spatial F1 uses voxel threshold 6 and SWC sampling
step 1; topology F1 is the existing GT-induced graph metric on predicted nodes.
Both have limitations, and partial references do not establish that every
unmatched reconstructed branch is false. Inspect SWCs/overlays, bridge dependence
and failures alongside the scores. This validation set has already informed
development, so it is not untouched external evidence.

## Visible files and outputs

- Runner: `scripts/pointneuron2/sampling_ablation.py`.
- Experiment arms/settings: `configs/pointneuron2/sampling_ablation.json`.
- Shared training implementation: `scripts/pointneuron2/train_sampling.py`.
- Per-experiment `experiment.json`: stages, settings, hashes and final metrics.
- `cache/manifest.json`: per-patch hashes, source identities and annotation counts.
- `random/`, `learned_no_aux/`, `learned_aux/`: `run.json` and `proposal.pt`.
- Per-arm `.log` files: the same live output shown in your terminal.
- `*_reconstruction/`: proposals, graphs, SWCs, comparison HTML, spatial/topology
  JSON and CSV reports. These are created only by the evaluation command.

Scripts, settings and this note are visible outside ignored `tmp`. Large generated
artifacts remain under the explicitly printed local run directory.

## Separate noise regression

The new `guarded_noise_hysteresis` diagnostic marks a MAD estimate at its floor
and falls back to background Otsu. For above-floor estimates, strong and weak
thresholds are bounded below by Otsu. Being above the floor does **not** validate
the noise assumptions. This rule is always a subset of background Otsu and cannot
recover branches below that threshold. Historical `noise_hysteresis` is retained
for reproducing the completed experiment.

The small local check reused exact saved scores and verified their artifact hashes:

| Training case | Background Otsu voxels | Historical noise voxels | Guarded voxels |
|---|---:|---:|---:|
| Fish 10:0 | 1,976 | 186,658 | 1,976 |
| Fish 10:6 | 7,415 | 180,907 | 7,415 |
| Fish 9:5 | 430 | 60,399 | 430 |
| Chick 1:7 | 239,127 | 1,364 | 1,364 |

All four subset checks passed. These are deliberately chosen mask regressions;
there was no new guide extraction or reconstruction evaluation. Source hashes
and thresholds are in [the saved report](noise_guard_check_seed0/summary.json).
The training cache remains unchanged so this repair does not confound the ablation.

The CPU synthetic three-arm training check confirms matched starting weights,
batch order and augmentation; proposal-only learning reaches alpha; checkpoints
retain their strategies; and disabled Chamfer is not computed. This is execution
evidence, not a real-neuron quality result.

Verification after these changes: **149 repository tests passed** with unrelated
pytest plugins disabled, and the PowerShell launcher parsed without errors.
