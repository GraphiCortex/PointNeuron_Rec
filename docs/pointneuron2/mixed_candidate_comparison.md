# Bright-core plus weak-signal candidate control — 11 October 2026

The 50/50 mixed pool improves average reference coverage, but loses some coverage
of branches already represented by the bright mask. It is an experimental control,
not an adopted training policy. A spatial-selection control is the next check.

## Experiment

Reused all 12 corrected training crops from the six-neuron foreground comparison.
Nine contain reference curves; three keep null scores. The fly 15/1 crop contains
only six reference samples. No mask re-extraction, full-volume processing, model
training, validation selection or reserved-test evaluation occurred.

Image-only pools, each 1024 rows, with eight matched seeds:

- **Bright:** uniform samples of the fixed mask.
- **Permissive:** uniform samples of the noise-hysteresis mask.
- **Mixed:** 512 bright samples plus 512 samples from permissive-minus-bright.
  Shortages transfer to the other group. Every eligible voxel is retained before
  padding if the union is smaller than the budget. Bright voxels outside the
  permissive mask remain eligible. Raw intensities and XYZ are preserved.

Each pool is scored directly and after selecting 512 rows using random sampling
or the existing fixed medial prior. **All methods use the same full fixed-mask
guide** for this prior, isolating pool changes. No guides are capped; references
never enter point selection. Baseline pools at seed zero exactly reproduce the
saved foreground-comparison pools.

CPU execution took **17.4 seconds**; 12 crops completed without errors.

## Results

Mean reference proximity within three voxels, first averaged over seeds inside
each crop and then over the nine annotated crops:

| Representation | Bright | Permissive | Mixed |
|---|---:|---:|---:|
| 1024-row pool | 61.8% | 66.2% | **71.6%** |
| 512 random inputs | 53.2% | 45.9% | **55.8%** |
| 512 fixed-medial inputs, same guide | 55.1% | 55.8% | **57.9%** |
| Along-edge supported length, 512 random | 54.1% | 46.8% | **56.1%** |

The mixed pool gains 9.8 percentage points before selection, but only 2.6 points
after random selection. These are **representation diagnostics, not reconstruction
F1 or semantic precision**. Along-edge support measures spatial proximity gaps,
not connectedness of a predicted graph.

![Candidate pools and random 512-input coverage](mixed_candidate_seed0/coverage.png)

To expose losses hidden by the mean, reference samples are separated only for
scoring: those within three voxels of the bright foreground, and those farther
from bright foreground but within three voxels of the permissive foreground.
The latter are called fixed-missed/noise-available, not verified weak neurites.

| 512 random input coverage within 3 voxels | Bright | Permissive | Mixed |
|---|---:|---:|---:|
| Bright-available samples, mean across nine crops | 71.6% | 45.7% | 64.1% |
| Fixed-missed/noise-available samples, mean across seven crops | 0.0% | 36.1% | 35.6% |

Conditional means have different denominators; some recovery groups contain only
one or eight samples. They must not be interpreted as dataset-wide recall.
Per-crop counts and all seed measurements remain in the JSON report.

Examples after random selection:

- Chick 0/0: bright 11.9%, permissive 73.4%, mixed 54.8%; the mixed pool restores
  useful missing coverage while preserving its small bright core well.
- Fly 15/0: bright 87.3%, permissive 33.0%, mixed 61.7%; even the mixed rule
  dilutes a crop where bright foreground already represents the trace well.
- Cultured cells 46/1: bright 86.4%, permissive 34.9%, mixed 80.4%.
- Adult fish 92/1: bright 59.5%, permissive 36.1%, mixed 66.3%.

This is evidence for separating bright and additional candidates, but not for
using a universal 50/50 ratio. Dense regions, incomplete traces, finite candidate
budgets and input selection still matter. Increasing input capacity is plausible;
its training benefit and GPU cost have not been measured in this experiment.

## Next control

Compare the existing farthest-point selector on these same pools at the same
512-input budget. This tests whether spatial spreading retains useful coverage
through selection without changing masks or tuning allocations. Its coverage
gain would still need model/reconstruction evaluation; spreading can retain noise
and does not establish which points belong to the neuron.

No real training recipe has been switched to the mixed pool. Integration into
cache preparation and inference must preserve matching image-only rules before
a user-visible GPU pilot. The repaired raw/local proposal comparison remains a
subsequent controlled experiment.

## Files and reproduction

- `src/pointneuron/data/mixed_candidates.py`: reference-free disjoint allocation,
  shortage transfer and padding rules.
- `scripts/pointneuron2/compare_mixed_candidates.py`: verifies corrected-reader,
  training partition and saved artifact hashes, prepares pools, measures separate
  reference groups and applies fixed CPU selectors.
- `configs/pointneuron2/mixed_candidate_comparison.json`: preset 50/50 experiment.
- `mixed_candidate_seed0/comparison.json`: all metrics, counts, source hashes and
  runtime. Arrays and log live in ignored `tmp/pointneuron2/mixed_candidate_seed0`.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\compare_mixed_candidates.py `
  --comparison .\docs\pointneuron2\foreground_repaired_seed0\comparison.json `
  --output-dir .\docs\pointneuron2\mixed_candidate_repeat `
  --artifact-dir .\tmp\pointneuron2\mixed_candidate_repeat\artifacts
```

Use fresh directories. This command performs CPU diagnostics, not training.

Verification: 195 pre-existing tests passed before checkpoint `5d18ed8`; the
19-test targeted candidate suite then passed, including eight new mixed-allocation
cases. Checks cover a bright minority in a large permissive population, overlapping
strata, disjoint bright eligibility, shortage transfer and retaining every unique
voxel before padding. Execution checks do not establish model improvement.

Git: the previous PN 2.0 groundwork is now committed locally as `5d18ed8` on
`pointneuron2`. No GitHub push has occurred. The original checkout remains on
`main`; generated arrays, environments, logs and checkpoints remain ignored.
