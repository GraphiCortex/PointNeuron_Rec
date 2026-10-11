# Next local experiment: masks and candidate coverage

**Completed by the user locally on 2026-10-10:** six training neurons, 48 patches,
zero failures, 8 minutes 24 seconds terminal wall time. See the
[expanded results and contrasting overlays](preprocessing_comparison_expanded_results.md).
The command below reproduces the experiment in a fresh directory; it is not a
required rerun.
The latest checkpoint inspection found weak full-guide support on the chick
example and substantial candidate/selection loss on the fly validation example.
This training-only comparison separates those effects without changing the main
preprocessing recipe or training a model.

## Fixed experiment

Six neurons from the frozen training partition are included: chick indices
0 and 1, zebrafish 9 and 10, fly 15 and 18. Up to eight image-defined patches
per neuron are chosen using the existing fixed-mask spatial coverage rule:
at most 48 patches. Exact patch boxes are identical across mask methods.
References do not choose patches or input points. This retains a selection bias
toward the original mask; it does not measure weak branches outside those boxes.

The first stage compares the existing fixed threshold, raw Otsu and Gaussian
background correction plus Otsu. It holds TEASAR settings, sigma 8 and the saved
isotropic voxel ruler fixed. Actual source spacing metadata is recorded, but
not applied in this controlled comparison. No new semantic segmentation model
has been introduced. Every extracted guide vertex is retained.

The second stage reads the saved masks/full guides, without repeating extraction:

| Variable | Compared settings |
|---|---|
| Image candidate pool | 1,024 versus 4,096 rows |
| Candidate preparation | Uniform versus brightest voxel per spatial cell plus farthest-point coverage |
| Selected encoder inputs | Always 512 rows, unless the mask is empty |
| Fixed input selector | Random, exponential full-guide prior, farthest-point coverage |
| Seed repeats | Three pool/selection seeds per setting |

Spatial cells have side length eight voxels and are anchored at the patch origin.
Representatives use raw intensity. If there are too few cell representatives,
unused foreground voxels fill the pool uniformly. If the mask has fewer unique
voxels than the candidate budget, all voxels are retained before padding. Both
pool-row counts and unique-voxel counts are recorded. Empty masks remain empty
failures of input availability, with zero support where a reference exists.

Selectors act on the exact same pool for a given mask/budget/method/seed.
Random and fixed-medial selection share seeded Gumbel noise. The fixed prior
uses alpha 0.25 and 25% uniform exploration. It uses the existing NumPy distance
and probability helpers, implementing the same distribution rather than
PyTorch's identical random stream. There is no learned checkpoint, optimizer or
model inference in this experiment. Farthest-point selection is deterministic
for a given pool. A repeated deterministic pool is not a new independent sample.

The 1,024-point uniform pool at the original seed is checked against the mask
runner's saved candidate array, including XYZ and raw intensities. This catches
coordinate or seed mismatches before treating alternatives as a comparison.

## Outputs to inspect together

Mask figures show XY/XZ raw images, masks, full guides and clipped references.
Candidate figures compare selected inputs from several pool/selector settings
at the first seed. Up to one informative patch per neuron is visualized for each
mask; every selected patch remains in the machine report, including no-reference
patches and any explicit failures.

Measurements include foreground fraction and components, full-guide/pool/input
reference proximity within 1/3/6 voxel units, terminal/junction support, and
arc-length-weighted reference-segment support. The latter samples segment
midpoints at intervals no larger than one saved distance unit and reports the
largest unsupported run **within each clipped SWC edge**. It does not concatenate
gaps across adjacent edges or prove predicted graph connectivity. Missing
reference stretches remain null; incomplete traces do not establish background
or permit a trustworthy foreground false-positive rate.

Reference terminals and junctions are classified from the full SWC tree before
cropping; patch boundary cuts are not biological terminals. Expanded mask
artifacts save their coordinates for the candidate comparison. Historical mask
artifacts lacking these arrays remain readable with null role metrics.

Candidate summary statistics average seed repeats within each patch before
averaging patches and reporting source groups separately. Zero-reference and
singleton-reference patches are excluded from these summaries. Reported neuron
counts make the small sample size visible; patch averages are descriptive,
not significance tests or dataset-wide anatomy estimates.

Keep failures, source differences, branch gaps and raw-image specificity visible
when choosing the next recipe. A higher proximity score from a denser noisy guide
does not by itself justify promotion. The experiment saves no training cache,
changes no checkpoint, and evaluates no validation or reserved-test neuron.

## Run in your PowerShell terminal

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"

& .\scripts\pointneuron2\run_local.ps1 `
  -Action preprocessing-compare `
  -ArtifactRoot "C:\Users\Seywe\Documents\GitHub\PointNeuron"
```

This is **CPU preprocessing**, not GPU training. Runtime depends on source-volume
loading, mask extraction, spatial selection and plotting; the completed local run
took 8 minutes 24 seconds in total. Progress prints each mask/patch and candidate step.
The launcher creates a fresh folder automatically and prints its absolute path.

- `console.log`: mask-stage live output.
- `masks/comparison.json`: full-guide mask metrics, source/config hashes and status.
- `masks/*.png`: mask comparison figures.
- `candidates.log`: candidate-stage live output.
- `candidates/comparison.json`: every pool/selector/seed result, gaps, summaries and failures.
- `candidates/*.png`: candidate comparison figures.
- `*/artifacts/*.npz`: raw/saved inputs and selection indices for inspection.
- `launch.json`: both commands, Python environment and overall status.

If mask preparation fails, the launcher stops before candidate comparison and
retains the failure report. Bring back the printed run path and final output;
we can inspect the figures and choose the next change together.

## Preparation checks

All **131 repository tests passed** with unrelated pytest plugins disabled.
The compact checks are saved in
[preprocessing_comparison_preparation.json](preprocessing_comparison_preparation.json).

New candidate/gap tests check agreement with the current cache, raw-value
preservation, separated-region coverage, sparse/empty masks, deterministic
selection, uniform fallback, full-guide retention, arc-length/spacing semantics,
invalid settings and correct averaging of repeated seeds.

A small runner check used the first two saved training-neuron-0 patches from the
historical foreground comparison: six mask artifacts, 72 pool variants and
216 selections. It completed without failures and checked the original uniform
arrays exactly. It is a software preparation check, not the six-neuron experiment
or evidence for a preferred method. Its report remains in
`tmp/pointneuron2/candidate_comparison_prepare_smoke`.

Source files:
`scripts/pointneuron2/compare_foreground.py`,
`scripts/pointneuron2/compare_candidates.py`,
`scripts/pointneuron2/reference_gaps.py`, and
`src/pointneuron/data/candidate_diagnostics.py`.
Recipes:
`configs/pointneuron2/foreground_comparison_expanded.json` and
`configs/pointneuron2/candidate_comparison.json`.
The historical 12-patch recipe and artifacts remain unchanged.
