# Phase 1: four additional training contexts

22 October 2026. The user agreed to prepare this fixed comparison and check
resources. **Preparation is complete; generation and scoring have not started.**
The resource preflight advises against launching the existing whole-volume
implementation on this laptop. No sampling setting or model has changed.

## Fixed selection and comparison

The image-only [case manifest](../configs/phase1_qualification_cases.json)
contains the following original training files, selected before inspecting
their image or reference contents in fresh PN2:

| Historical alias | Context | XYZ | Storage | Selection purpose |
|---|---|---|---|---|
| 22 | Frog, done_Recon112012no3-2 | 512 x 512 x 81 | uint16 | Another uint16 context |
| 75 | Mouse RGC, sv_140918c8 | 800 x 800 x 31 | uint8 | Relatively shallow volume |
| 85 | Silkmoth, Confirmed_hard_0984_seg | 512 x 512 x 164 | uint8 | Deeper volume and different source context |
| 126 | Janelia fly, extract_0 | 511 x 511 x 420 | uint8 | Large, deep-volume scaling stress |

Stable case IDs and explicit paths in the manifest are authoritative; aliases
are historical inventory labels. The source-group labels motivate diversity,
but do not establish biological or acquisition independence. These remain
training data, including historically exposed identities, not untouched tests.
Physical calibration is unverified. No development or reserved files are read.

Reuse [the existing recipe](../configs/phase1_sampling_probe.json) byte for byte:
uniform/intensity/Sato, scales 1/2/4 native voxels, whole-image affine min/max
normalization, 10% uniform draw mass, disjoint clipped 128-cube selection boxes,
dense cell side 4, lean cell side 8, and seeds 0--4. This is **120 new clouds**.
The original two-case evidence is retained without rerunning it.

Dense clouds are the candidate input for later integration. Lean clouds remain
a budget stress test; the original lean gaps disqualify treating that budget as
an already adequate representation. Input point counts are separate from
PointNeXt internal subsampling, proposal counts and graph-node counts.

Image-only generation must save clouds before separate reference evaluation.
Use the already reviewed explicit `opaque_integer` reference policy: retain
all valid geometry and unknown integer type metadata, without anatomical
interpretation. Preserve malformed, unscorable or frame-questionable cases in
the report; do not replace them with easier images or fit an alignment.

## Measurements and decisions

Apply the same [comparative checks](phase1_sampling_protocol.md) to **each** new
case: lean mean support gain of at least 5 percentage points versus both
controls; dense global support loss at most 2 points; local coverage loss at
most 2 points and maximum-gap increase at most 2 voxels at both budgets.
Report the existing 2/6-voxel metrics, every seed, exact unique counts,
source-coordinate/intensity verification, 0.5/0.25-voxel numerical refinement,
reference extent checks, overlays, wall time and memory. Undefined structural
classes cannot be counted as successful checks.

These are unchanged relative checks, not new absolute sufficiency thresholds.
A passing average cannot hide a failed case, poor absolute branch retention,
long gaps, or unacceptable resource costs. Inspect dense local support before
an integration decision; passing relative checks alone does not prove that the
encoder receives adequate information.

- If complete evidence supports the guide, discuss freezing it as **Phase 1 v0
  for integration** and proceed to the intended PointNeXt/Transformer/GATv2
  pipeline. No automatic training follows this report.
- If a comparison fails, distinguish lack of benefit beyond brightness, local
  information loss, and reference uncertainty. Stop at one complete report;
  do not tune scales, densities, seeds or per-image settings to rescue it.
- An unscorable case or unresolved frame issue makes that part inconclusive.
  A resource hold is an execution/scalability finding, not a biological failure
  of Sato or a model-learnability result. Keep the selected case visible.

This is one fixed qualification, not a parameter grid. Its destination is a
runnable raw-image-to-SWC candidate and measured reconstruction, not indefinitely
perfecting the sampler.

## Resource preflight: hold the current implementation

The visible [header-only resource script](../scripts/preflight_phase1_resources.py)
ran successfully. It read four 43-byte image headers, the fixed case/recipe
files, previous generation metadata and current host memory/disk counters.
It read **no new image payload or reference**, and performed no filtering,
sampling, inference or training.

| Case | Million voxels | Linear peak proxy, GiB | Planning allowance, GiB |
|---|---:|---:|---:|
| Frog 22 | 21.23 | 3.76 | 5.70 |
| Mouse 75 | 19.84 | 3.51 | 5.39 |
| Silkmoth 85 | 42.99 | 7.61 | 10.51 |
| Janelia 126 | 109.67 | 19.42 | 25.27 |

The laptop has **15.71 GiB installed RAM**. Available RAM was **1.21 GiB** at
the recorded preflight, after an earlier snapshot of about 2 GiB. Available
memory varies. The largest image's proxy already exceeds installed RAM before
the planning reserve. Closing other programs alone does not resolve that
projection.

The proxy scales the maximum observed peak working set per voxel from the two
completed smaller cases. Their per-case Windows peaks are cumulative, so these
are deliberately rough planning values, not measured requirements, statistical
bounds or proof of allocation failure. Working set also excludes paged-out
allocations. The allowance adds 25% and 1 GiB as an explicit planning heuristic;
it is unrelated to scientific success criteria. No process was terminated and
no pagefile or system setting was changed.

The same crude scaling projects **64--69 minutes for filtering alone**, excluding
selection, compression, decoding and all reference scoring. Shape and memory
pressure may change this substantially. Guides plus uncompressed point-packet
payloads total approximately **2.30 GiB**; this excludes temporary/evaluation
artifacts and compression overhead. Free disk was about 79 GiB. Memory is the
immediate concern.

Raw report: `tmp/phase1_qualification_20261022/resource_preflight.json`, SHA-256
`ecefeafd97cfaecc2e24c281d4b03a8789694f37e83095a35ac0e5a4bdd53605`.
It records case/config/baseline/script hashes and all counts/projections.

## Next engineering decision

Prepare a memory-bounded implementation of the same guide before launching this
cohort. Explain and review its numerical equivalence checks first. Blocking
must preserve normalization, scale combination, global boundary behavior and
the filter's actual spatial dependencies; the current disjoint selection boxes
are not automatically valid filtering blocks. Do not silently substitute a
shorter halo, downsample the image or replace the large case.

Validate against the existing whole-volume operator on manageable fixtures and
saved inputs, including boundaries and seams, before collecting new biological
scores. If numerical differences can alter sampling, disclose them and agree
how the updated implementation is compared. This changes how Phase 1 is
computed, not the intended learned architecture. No new implementation is
adopted or large-volume run launched by this planning document.

## Reproduce the small preflight

From the PN2 root, with a fresh output filename:

```powershell
& .\tmp\phase1_cpu_env\Scripts\python.exe .\scripts\preflight_phase1_resources.py `
  --cases .\configs\phase1_qualification_cases.json `
  --recipe .\configs\phase1_sampling_probe.json `
  --baseline .\tmp\phase1_sampling_20261022\clouds.json `
  --output .\tmp\phase1_qualification_20261022\resource_preflight.json
```

The resource verdict is advisory and does not launch another command. Existing
generation/evaluation entry points remain unchanged; no expensive run command
is scheduled while the resource issue remains unresolved.
