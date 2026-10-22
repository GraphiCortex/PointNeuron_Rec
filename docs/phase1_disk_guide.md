# Disk-backed Phase 1 guide

22 October 2026. This is an implementation/scalability change to the candidate
image guide, not a new sampling policy or a change to the intended learned
architecture. The original whole-volume backend remains the default and the
reference for parity checks. No model training is involved.

## What changes

`src/pn2/inputs/disk_guide.py` stores float32 intermediate arrays in `.npy`
files. For each separable Gaussian axis pass, it reads one core block plus the
full finite kernel radius along that axis, computes with the pinned SciPy
operator, and writes only the core. Reflect boundaries apply at the real image
extent. Saving every intermediate pass avoids treating an intermediate
derivative as if its boundary extension were the original image's extension.
Every mapping is closed after a block read/write; the module does not retain
full-volume mappings in the process working set. Operating-system file caching
is separate and can still use RAM.

The pinned upstream calculation uses two Gaussian first-derivative operations
per Hessian element, in float32 and the original axis order. It chooses
`truncate=100` for scales at or below one, otherwise eight, with each pass's
sigma equal to `sigma / sqrt(2)`. The recipe's 1/2/4 scales therefore have
**71/11/23-voxel radii per axis pass**. This implementation retains those
kernels; it does not substitute a customary smaller halo. Hessian eigenvalues,
bright-ridge sign, clipping, scale normalization and maximum across scales
use the same calculation and public upstream eigenvalue routine.

The block side (default 64) controls execution memory and overhead, not the
number or location of sampling points. The largest axis-pass input block is
bounded by `B * B * (B + 2r)` voxels, clipped at image edges. Eigenvalue work is
limited to `B^3` voxels. These are algorithmic array-size limits, not a promised
process peak or an end-to-end hardware minimum.

The generation command still decodes the original raw image and calculates its
whole-image affine normalization. It saves normalization to disk, releases the
in-memory normalized array, filters, and reads intensity/guide tiles separately
for sampling. Source decoding and initial normalization still scale with the
volume. Point packets, sorting buffers, the operating system, and eventual
learned stages add their own resource requirements. This is not yet a fully
streaming reconstruction product.

The cost is additional disk space, I/O and repeated halo computation. The
filter creates ten full float32 scratch arrays plus its output; normalization
adds one more file in the generator. Scratch is deleted only after success,
and failure evidence is retained. Cloud hashes are now calculated in 1 MiB
chunks instead of loading the entire file. No source images, old clouds,
scales, draw probabilities, seeds or point-count rules are changed.

## Verification and limits

Eight operator tests compare every guide voxel and ordered sampled indices,
with exact equality rather than a relaxed acceptance tolerance. Fixtures cover
random fields, multiple scales, different block sizes, a constant field,
boundary impulses, a tube crossing seams, non-divisible extents and a
one-voxel-thick dimension. Invalid inputs and existing output paths are rejected.
The source-to-cloud-to-reference CLI test runs both backends; its reference is
created only after cloud generation. The disk CLI writes its output directly
to the final path, avoiding the Windows rename failure caught during testing.

`scripts/verify_disk_guide.py` replays one of the two previously inspected
training images and compares against its frozen guide and all ten saved Sato
clouds (five seeds, two budgets). It hashes sources and outputs and records
runtime and process memory. It reads no SWC and calculates no new biological
score. Each case runs in a separate process; historical memory peaks were
cumulative within a two-case process, so they are not matched filter-only
memory benchmarks.

## Completed replay results

Both full original images replayed exactly. Every guide voxel is equal, both
saved `.npy` files have the same SHA-256 hashes as their frozen originals, and
all **20 ordered Sato point selections** match (two cases x two budgets x five
seeds). The source image bytes, original saved guides/clouds and recipe remain
unchanged. No reference was read and no new biological score was calculated.

| Observation | Fruitfly, uint8 | Series021, uint16 |
|---|---:|---:|
| Original volume, million voxels | 11.78 | 15.73 |
| Unequal guide voxels | 0 | 0 |
| Exact ordered cloud selections | 10/10 | 10/10 |
| Disk filter time, including filter I/O | 9.41 min | 7.70 min |
| Historical whole-volume filter time, excluding output save | 3.88 min | 5.56 min |
| Replay process peak working set | 137.3 MiB | 186.5 MiB |
| Replay peak commit/pagefile counter | 1.31 GiB | 1.33 GiB |
| Historical cumulative peak working set | 2.09 GiB | 2.65 GiB |

Working set is resident process memory; the commit/pagefile counter is a
different Windows memory measure, not the number of bytes actually read from
or written to a pagefile. The much smaller resident peaks support the intended
execution strategy, but **do not imply that PN2 needs only 137--187 MiB RAM**.
Libraries and other allocations still commit memory. Historical runs lack the
same commit measurement, the historical peaks are cumulative, and host load
was not matched. Do not report a matched total-memory reduction factor or
hardware minimum from this table. The second replay's memory snapshots show
its resident peak occurred during source preparation, not filtering.

Disk execution was slower here. A crude per-voxel projection from these two
replays suggests roughly 1.6--2.6 hours of filtering for the four new cases,
excluding decoding, selection, compression and reference scoring. The deeper
volumes can have different halo overhead and I/O behavior; this is not a bound
or reliable completion estimate. Include resources in the qualification verdict.

**158 distinct regression/operator tests pass across the regression run and
targeted reruns.** The initial expanded regression run had 149 passes and one
failure: Windows denied renaming a completed guide. Writing directly to the
final fresh destination fixed the CLI path; nine targeted tests then passed.
Both CLI variants passed again after adding the memory metadata fields.
The scientific settings and parity criteria were not relaxed.

The first replay collected code hashes at report-writing time while the
optional output-destination API and an unused import were being completed;
the filtering arithmetic was unchanged. It has no launch-time source snapshot.
The second replay saves source snapshots and hashes before decoding/filtering;
those snapshots match the current backend and verification script. This
provenance difference is retained in the summary rather than silently revising
the first raw report.

[Visible machine summary, raw-report hashes and checks](phase1/disk_guide_parity_20261022_summary.json).
Raw reports, arrays and logs are under `tmp/disk_guide_parity_20261022/`.

The backend is ready to use for the **already fixed four-case qualification**.
Numerical parity on these two images does not establish neuronal
generalization, reconstruction improvement, or a minimum RAM specification.
The subsequent [four-context qualification](phase1_qualification_results.md)
records actual large-image execution and support results. No four-case
generation, reference scoring, GPU run or training was launched as part of the
implementation checks described here; the separate fixed CPU comparison
followed user authorization.

## Commands

Run from the PN2 root, with fresh output directories. The replay is an
implementation check on already inspected training inputs:

```powershell
& .\tmp\phase1_cpu_env\Scripts\python.exe .\scripts\verify_disk_guide.py `
  --clouds .\tmp\phase1_sampling_20261022\clouds.json --case-index 0 `
  --output-dir .\tmp\disk_guide_parity_20261022\fruitfly
```

Use case index 1 and a fresh `series021` directory for the other saved case.
Do not overwrite the completed evidence.

After parity review, the existing launcher supports the fixed qualification
with an explicit disk backend. Generation and scoring remain separate:

```powershell
& .\scripts\run_phase1_sampling.ps1 -Action generate `
  -CasesManifest configs\phase1_qualification_cases.json -GuideBackend disk `
  -OutputDirectory tmp\phase1_qualification_disk_20261022

& .\scripts\run_phase1_sampling.ps1 -Action evaluate-geometry `
  -OutputDirectory tmp\phase1_qualification_disk_20261022
```

These are CPU commands; no GPU or training is launched. Generation requires a
fresh output directory, retains all four declared cases and uses the unchanged
comparison recipe. Inspect generation completion before scoring. Do not
automatically retry, change the recipe, exclude a case or extend the comparison
after an execution or scientific failure.

## Upstream implementation references

The local pinned sources were inspected directly. The implementation uses
public SciPy/skimage functions; no third-party implementation is vendored.

- [scikit-image 0.25.2 Sato implementation](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/filters/ridges.py)
- [scikit-image 0.25.2 Gaussian Hessian implementation](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/feature/corner.py)
- [SciPy 1.15.3 separable Gaussian filtering and radius calculation](https://github.com/scipy/scipy/blob/v1.15.3/scipy/ndimage/_filters.py)
