# CPU cache timing and exact-output block comparison

31 October 2026. The fresh Image5 preparation completed. An approved
block64-to-block128 execution comparison then produced a byte-identical Sato
guide and identical arrays/dtypes for all 25 point packets. Block128 is ready
for the remaining image preparation; the bulk run has not been launched.
This is an execution improvement, not a measured reconstruction improvement.

## Completed single-image preparation

The case is training alias 41, cultured human Image5: XYZ 640×640×29,
11,878,400 voxels, 25 disjoint clipped model boxes. The command read the original
raw image, calculated the existing Sato guide and cached every image box.
No reference, target, model or GPU was involved.

| Block64 preparation part | Seconds |
| --- | ---: |
| Decode and provenance | 2.09 |
| Normalization, guide save and filtering | 685.74 |
| Point generation and packet verification | 4.56 |
| Case total | 692.41 |

The user-observed command duration was 11m34s. The verified scientific cache
now contains **3/40 images and 73/2379 boxes**. There are 37 images left:
33 fresh filters and four copies of previously computed, provenance-verified
PN2 guides. The previous block64 rate implied about 43 remaining filter hours.

## What the larger blocks change

The disk filter can process more voxels in each CPU work block, reducing
repeated halo work and disk access. This **does not change the model's crop
size**. Both processing blocks and model boxes happen to be 128 voxels now,
but they serve different purposes.

The comparison keeps the mathematical kernels, scales 1/2/4, float32
calculations, whole-image normalization and global reflect boundary behavior.
The input still contains 2048 distinct source voxels per model box, with the
same sampling probabilities, seed, order, coordinates and intensity features.
PointNeXt, the 512 geometric queries, losses and planned training are unchanged.

Block128 was chosen as one conservative execution comparison on the 15.7 GiB
laptop, rather than a search across many new settings. Two synthetic volumes
crossing block seams in all three axes first matched whole-volume results
exactly. The real comparison then decoded the same original Image5 into a
separate ignored output folder; it did not overwrite the successful cache.

| Real-image measurement | Block64 baseline | Block128 candidate |
| --- | ---: | ---: |
| Guide preparation | 685.74 s / 11.43 min | 484.73 s / 8.08 min |
| Process peak working set | 137.1 MiB | 182.6 MiB |
| Windows peak process commit indicator | 1.32 GiB | 1.38 GiB |
| Guide file SHA256 | Identical | Identical |
| All 25 point packets | Baseline | All arrays and dtypes identical |

Guide SHA256: `042ab2be59f9916d417d0faad9bce33cabf5f59d2463dbc8127432e45e89d55a`.
The commit indicator is not a measurement of actual disk paging or a minimum
hardware requirement.

The observed filter speed ratio is 1.415×, or **29.3% less filter time**.
The baseline was measured earlier, so this is not a contemporaneous controlled
timing pair. Host load, file caching and thermals were not controlled. Larger
or deeper images have not been timed with the candidate. Exact-output parity
is the stronger conclusion; the timing is useful planning evidence.

## Remaining preparation and explicit launch

Scaling the measured block128 rate by remaining fresh-image voxel count gives
**30.4 hours of filtering**, excluding decoding, sampling, copying, checking
and system variability. This is a rough estimate from one thin image, not a
duration bound or a product performance claim. CPU preparation remains a real
software cost even though the cache can be reused for later model experiments.

The new sequential controller prepares one image per worker, checks completed
case hashes before skipping them, and stops on partial cases or failures.
It does not delete, overwrite or automatically retry partial work. It checks
source and decoded-image content for train/development duplicates. The eight
reserved identities are excluded.

Fresh-image wall allowances scale by volume, with a 1–6 hour range and margin
over the measured rate; the largest current allowance is 13,982 seconds.
Verified guide copies allow ten minutes. The outer 72-hour wall cap is a stop
limit, not an expected duration. Fourteen focused comparison/controller tests
pass; PowerShell launcher parsing passes. No new real bulk filters were run
by those checks.

The completed read-only plan is
`tmp/node_cloud_cache_bulk_plan_20261031/plan.json`. Launch once from PN2:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_cloud_cache_bulk.ps1 -Action prepare `
  -PlanRun ".\tmp\node_cloud_cache_bulk_plan_20261031" `
  -Output ".\tmp\node_cloud_cache_bulk_prepare_20261031"
```

The shared image cache stays at `tmp/node_cloud_cache_20261031`. The run folder
holds the launch plan, per-case console logs and receipts, completion journal
and final report. If it stops, inspect the evidence before any new launch.
Completed images remain usable; partial images need an explicit decision.

This command performs **data preparation only**. Reference targets, the broader
training loader and all-image metrics still need implementation and checking.
The proposed 9200-update GPU fit is a subsequent separately launched step;
complete point caches alone do not make training ready.

## Evidence

- [Portable baseline timing and remaining-work arithmetic](evidence/node_cloud_cache_timing_20261031.json)
- [Portable exact-output comparison, controller checks and bulk-plan identity](evidence/node_guide_block128_20261031.json)
- Raw baseline: `tmp/node_cloud_cache_timing_20261031_201525/report.json`
- Synthetic block parity: `tmp/node_guide_block_cpu_20261031/report.json`
- Real comparison: `tmp/node_guide_block128_20261031/report.json`
- [Bulk execution profile](../configs/node_cloud_cache_bulk.json)
- [Original cache builder and input contract](node_cloud_cache_builder.md)
