# Image-only node cache: implemented and parity-verified

Latest status, 2 November: the [full 40-image cache and independent review](node_cloud_cache_completion.md)
are complete. The larger point-budget comparison, targets/loader and broader
training remain subsequent work. The initial
implementation result below is retained as its original evidence snapshot.

31 October 2026. The raw-image point-cache builder is implemented. **Synthetic
checks pass; two original training images are cached; all four successful
learnability inputs replay exactly.** No fresh real-image filter timing, bulk
preparation, target creation, model forward or training has been launched.

This implements the image-preparation part of the
[broader node protocol](node_generalization_protocol.md). PointNeXt, geometric
queries, loss, targets, confidence cutoff and training budget are unchanged.

## Data path and separation

```text
Explicit image-only manifest
    -> decode original raw image and hash the exact source/decoded bytes
    -> same whole-source normalization and Sato guide
       (verified guide copy when available; otherwise existing disk operator)
    -> every disjoint clipped 128-voxel box
    -> 2048 distinct source locations, same PPS and source-ID hash order
    -> native XYZ/intensity + crop-local XYZ/intensity features + provenance
```

The builder never consumes the reference manifest or reference-derived target
counts. Every box is prepared, including constant and reference-empty regions.
The image-only input loader rejects reference keys and reserved partitions.
References/targets remain a separate later builder; no label or model weight
enters point selection. Point packets alone do not make training ready.

Source: `src/pn2/inputs/node_cache.py`; controls:
`scripts/node_cloud_cache.py`, `scripts/run_node_cloud_cache.ps1`, and
`configs/node_cloud_cache.json`. The scientific identity binds the explicit
40-image manifest, settings, package versions and core source hashes. Run
receipts additionally bind the driver/config identity.

Input behavior matches the successful task: Sato scales 1/2/4 voxels, 0.1
uniform draw-distribution mass, seed zero, 2048 actual source voxels per box,
source-ID hash ordering seed zero, crop center `(start+stop-1)/2`, coordinate
scale 64 and whole-source affine intensity normalization. The intensity feature
uses the original float64 calculation followed by float32 rounding; the guide
input retains its original float32 calculation. No added/jittered locations,
duplicate padding, foreground threshold, new image channel or guide retuning
is included. A too-short box fails explicitly.

## Guide reuse and cache ownership

`configs/node_guide_reuse.json` lists six already computed PN2 image guides.
Byte-identical copies of their original image-only receipts and recipes are
preserved under `docs/evidence/node_guide_reuse_*`. Original image and decoded
hashes, normalization, guide/operator scales, package versions, mathematical
source hashes and guide hashes are verified before reuse. Original generation
wrapper hashes are preserved as provenance; the early wrapper predates disk
support, while its mathematical sampling/guide module still matches exactly.

Reused guides are copied into their new case folders and checked again. Each
new case is self-contained with respect to its guide; it is not a link into
the old run. Original guides/packets are not changed. The remaining registered
guides must remain available until copied or explicitly regenerated under a
reviewed plan. No historical learned image field or pointneuron2 model cache
is adopted as an image guide.

Case data live under `tmp/node_cloud_cache_20261031`, in stable case-ID-derived
folders. Case records and a completion hash are written once after every image
box is saved and checked. Existing/partial cases are preserved, never overwritten
or automatically retried. No frequently replaced Windows status file is used.
Only owned normalization/filter scratch is removed after success. Space is
checked before decoding, with the guide/work-array estimate plus a 2-GiB free
reserve; this is an execution allowance, not a scientific parameter.

Cache validation checks all image-defined boxes, exact point counts/bounds,
packet/guide/receipt hashes and policy identity. Cross-split byte-identical raw
or decoded image payloads cause an explicit failure among the cases available
so far; same-partition duplicates are reported. This does not prove independence
for not-yet-prepared images, transformed duplicates or acquisition families.

## Completed checks and real replay

The synthetic uint16 case verifies exact whole-volume versus disk-backed guide
values, source-coordinate/feature conversion and fresh versus reused guide
point arrays. It executes no real image or model. Package versions remain the
existing Phase 1 CPU lock; no environment was changed.

**Thirty focused tests pass** (102.13 seconds): new cache integrity/controller
tests plus the existing disk-guide and sampling fixtures. Checks include
clipped boxes, feature rounding, constant-image retention, stale source/guide
rejection without fallback, immutable output, low-space failure, tamper and
incomplete-case detection, forbidden reference/reserved manifest fields,
cross-split duplicate detection and timeout without retry. PowerShell parses.

The bounded real replay then decoded fruitfly 32 and frog 22 from their
original raw files, verified/copied the two existing guides, and prepared all
48 boxes. No real-image filter was recomputed and no reference was read.

| Already inspected training source | Image boxes cached | Source point observations | Case preparation time | Process peak working set |
|---|---:|---:|---:|---:|
| fruitfly 32 | 32 | 65,536 | 4.36 seconds | 78.9 MiB |
| frog 22 | 16 | 32,768 | 20.81 seconds | 136.9 MiB |

These times include decoding, guide-copy verification and packet creation,
not fresh filtering or Python/controller startup. Memory describes these replay
workers, not a fresh large-image filter or minimum hardware requirement.

For the four original Sato inputs (32/27, 32/20, 22/2, 22/5), **all arrays and
dtypes match exactly**: selected source IDs, native XYZ, raw intensities, local
XYZ and normalized features. The comparison reads old input packets only after
the new reference-free generation finishes; it never uses them to choose points.
This preserves the exact inputs that passed the node-learnability experiment.

Current status: **2/40 images cached**, 48/2379 boxes. No target cache or broader
training loader is claimed complete. Portable [evidence](evidence/node_cloud_cache_builder_20261031.json)
binds the synthetic/real reports, code/policy identities and exact comparisons.
Raw reports: `tmp/node_cloud_cache_cpu_20261031/report.json` and
`tmp/node_cloud_cache_replay_20261031/report.json`.

## Next manual action: one fresh CPU preparation timing

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_cloud_cache.ps1 -Action timing
```

This prepares only the smallest uncached training image by the inspected
image dimensions: cultured-human neuron 41 / `Image5`, XYZ **640 x 640 x 29**
(11,878,400 voxels), 25 boxes. It uses the existing disk-backed Sato operator,
with a **3600-second wall cap** and no automatic retry. Historical comparable
disk filters took roughly 8-10 minutes; this case has not been timed, so that
is context rather than a promised duration.

The command prints scale and box progress. A scale can take several minutes.
It keeps a successful guide and all 25 point packets in the same cache for
later use; this is useful data preparation, not a throwaway training fit.
Logs and timing go to a fresh `tmp/node_cloud_cache_timing_<timestamp>` folder.
It does not launch the other images, create reference targets or use the GPU.

Review that result before choosing the bulk-preparation wall allowance and
providing a full-cohort command. A bulk launcher is deliberately not offered
yet. The current implementation only exposes synthetic checking, the completed
two-image replay and one-image timing. After input preparation, target-cache,
unknown-supervision loader, metrics and actual GPU timing are still required
before the 9200-update fit can launch.
