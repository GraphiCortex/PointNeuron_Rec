# Five-epoch matched sampler pilot results

**Historical input pipeline:** decoder defects were confirmed after these runs.
See [the root-cause findings](root_cause_findings.md) before interpreting them as
evidence about PN 2.0 on correctly decoded images.

**Update:** whole-neuron evaluation has now completed for all three arms. See
[the reconstruction results](sampling_ablation_reconstruction_results.md).
The remainder of this note records the earlier training/confidence inspection.

All three user-launched CUDA runs completed on the RTX 3070 Ti Laptop GPU.
The experiment reports `status: trained` and `matching_checks: passed`.
The subsequent local confidence check performed inference only, with no optimizer
steps, on all 56 cached validation patches. Whole-neuron reconstruction has not
yet been launched for this comparison.

## Training outcome

| Arm | Best validation proposal loss | Selected epoch | Final alpha | Training/validation time |
|---|---:|---:|---:|---:|
| Random | 5.7832 | 5 | 0.25, unused | 34.1 s |
| Learned, no guide loss | 5.3784 | 5 | 0.252366 | 405.9 s |
| Learned, guide loss 0.05 | 5.4401 | 5 | 0.251406 | 459.8 s |

Lower is better for this common proposal objective. The no-auxiliary arm is
approximately 7.0% below random; the auxiliary arm is approximately 5.9% below
random and 1.1% above the no-auxiliary arm. These are single-seed, five-epoch
validation-loss comparisons, not reconstruction improvements or a conclusive
ranking of the methods. All arms started from identical fresh weights.

| Epoch | Random validation loss | Learned without auxiliary | Learned with auxiliary |
|---|---:|---:|---:|
| 1 | 6.4912 | 6.0555 | 6.0173 |
| 2 | 6.0320 | 5.5827 | 5.5779 |
| 3 | 6.0302 | 5.6027 | 5.6867 |
| 4 | 5.8922 | 5.4533 | 5.5100 |
| 5 | 5.7832 | 5.3784 | 5.4401 |

The no-auxiliary sampler received nonzero alpha gradients in 100–108 training
batches per epoch; the auxiliary sampler did so in 131–132. Each training epoch
has 132 batches. Entirely unannotated batches have no proposal-learning signal,
while the auxiliary arm can still learn from their image guides.

The timings cover each trainer's loop, including validation, and exclude the
six-minute cache preparation and process startup. Learned training was about
12–13.5 times slower than random in this pilot. The current straight-through
sampler draws 512 points sequentially. These timings are a reason to measure
the sampler's quality benefit before committing to longer runs.

## Cached validation confidence and selection

| Arm | Patches with raw scores >= 0.5 | Patches with raw scores >= 0.85 | Maximum raw score |
|---|---:|---:|---:|
| Random | 52/56 | 18/56 | 0.9999 |
| Learned without auxiliary | 53/56 | 25/56 | 0.9999 |
| Learned with auxiliary | 54/56 | 31/56 | 0.9996 |

These are raw network proposal rows before NMS, aggregation and graph selection.
They are neither distinct recovered branches nor precision measurements. Unlike
the two-epoch smoke checkpoint, these models produce proposals above 0.5.
Some patches assign high confidence to most or all input rows; this does not
establish anatomical specificity. Compare images, SWCs and reference metrics.

Number of cached patches containing any raw score >= 0.85:

| Validation neuron | Random | Learned without auxiliary | Learned with auxiliary |
|---|---:|---:|---:|
| 19 | 4/8 | 5/8 | 7/8 |
| 21 | 8/8 | 8/8 | 8/8 |
| 35 | 4/8 | 5/8 | 6/8 |
| 44 | 2/8 | 4/8 | 4/8 |
| 79 | 0/8 | 2/8 | 4/8 |
| 94 | 0/8 | 0/8 | 1/8 |
| 135 | 0/8 | 1/8 | 1/8 |

Whole-neuron inference processes more coverage patches than these eight cached
patches, so these counts do not determine whether full reconstruction succeeds.
Keep the agreed thresholds fixed across arms and retain failures in the report.

At the same image pools and evaluation seeds, replacing trained alpha with its
initial value changes **at most two candidate rows out of 512** in any cached
validation patch. No-auxiliary selections are unchanged in 30/56 patches and
replace a mean 0.536 rows; auxiliary selections are unchanged in 38/56 and replace
a mean 0.393 rows. Counts compare unordered candidate-row sets, not draw order
or unique voxel coordinates. The current learned adjustment changes selection
very little. The guide prior could explain much of the loss difference, but this
pilot cannot isolate that explanation; a frozen-alpha training control would be
needed if reconstruction supports pursuing the sampler.

## Next command: reconstruction evaluation

Run visibly in the same terminal:

```powershell
& .\scripts\pointneuron2\run_local.ps1 `
  -Action ablation-evaluate `
  -ArtifactRoot "C:\Users\Seywe\Documents\GitHub\PointNeuron" `
  -Experiment "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\ablation-prepare_20261010_205859_1596\experiment"
```

This runs all three arms on the seven full validation neurons, using the same
geodesic reconstruction and saved cutoffs, then scores spatial and topology
metrics. It does not train, change thresholds or use the eight reserved test
neurons. Reconstruction may take longer than this short training pilot; no
reliable complete-run duration has been measured. The runner preserves neuron
failures and stops rather than averaging an incomplete denominator. Send the
final output or failure so we can decide the next step from the actual evidence.

## Evidence and reproducibility

- [Saved training trajectories, confidence counts and per-patch selections](sampling_ablation_pilot_seed0/summary.json).
- [Experiment design, limits and stage commands](sampling_ablation.md).
- Training artifacts:
  `tmp/pointneuron2/local_runs/ablation-prepare_20261010_205859_1596/experiment`.
- Diagnostic script: `scripts/pointneuron2/inspect_ablation_confidence.py`.

The diagnostic verified the shared manifest and cache-record hashes, reran the
initial-weight/order/augmentation matching checks, and verified all checkpoint
hashes before inference. It checks finite scores and preserves existing reports.
To reproduce it, choose a fresh output folder:

```powershell
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\inspect_ablation_confidence.py `
  --experiment .\tmp\pointneuron2\local_runs\ablation-prepare_20261010_205859_1596\experiment `
  --output-dir .\tmp\pointneuron2\ablation_confidence_recheck `
  --device cuda
```

The earlier 149-test run remains the training implementation's verification;
this new diagnostic was execution-checked on all real cached validation patches.
