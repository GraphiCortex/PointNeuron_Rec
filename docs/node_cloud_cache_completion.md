# Completed image-only cache and independent review

2 November 2026. The user-launched CPU preparation completed successfully:
**40/40 images, 2379/2379 image-defined boxes**. An independent read-only review
then rechecked every guide and point-packet hash, loaded every packet and
checked geometry/features, and inspected every guide box for finite nonnegative
response and correct fallback flags. No filter, model or training was rerun.

## What is ready

| Partition | Images | All image boxes | Stored source-point observations |
| --- | ---: | ---: | ---: |
| Training | 33 | 1964 | 4,022,272 |
| Development | 7 | 415 | 849,920 |
| Total | 40 | 2379 | 4,872,192 |

Every box contains 2048 distinct source voxel locations, with no duplicate
padding or reference-dependent exclusion. All full-resolution Sato guides are
retained alongside point packets and immutable provenance receipts. Existing
source, guide, sampler and model settings remain unchanged. Six guides were
verified copies of earlier PN2 computations; the other 34 were freshly computed
for this cache, including the initial Image5 timing case.

The source-byte and decoded-image signatures recorded at generation show no
exact duplicate images within or across the two partitions. This is a content
identity check, not proof of independent acquisitions or an untouched test.
The seven development neurons remain repeatedly exposed development material;
the reserved eight identities were not included.

The user-observed command duration was **1 day 5 hours 31 minutes 47 seconds**.
This includes the reported sleep/hibernation interruption and is not an
uninterrupted throughput benchmark. The earlier remaining-time estimate was
optimistic: later per-image elapsed times varied more than the first completed
images suggested. No pause duration or compute-only total is inferred here.

## Meaning of `fallback=True`

The existing sampler uses uniform draws without replacement when the entire
Sato response in a box is zero. It still retains real locations and their
intensities. It does not skip the box, invent points, use a reference mask,
or certify that the box is biological background.

The review confirmed **669 fallback boxes**: 591 training and 78 development.
Every fallback box is recorded as constant intensity by the generation-time
full-raw-box check; its cached samples are also constant. There are no
nonconstant-source fallback boxes in these receipts. The saved guide values
in all fallback boxes were independently checked as exactly zero.

Joining the existing reference-preflight metadata gives:

| Partition | Guided, reference present | Guided, reference empty/unknown | Uniform fallback, reference empty/unknown |
| --- | ---: | ---: | ---: |
| Training | 460 | 913 | 591 |
| Development | 151 | 186 | 78 |

No fallback box has local targets in that audit. None of the 774 source-constant
boxes has targets either. Some constant boxes still have nonzero Sato response
because filtering has spatial context beyond each model box. These observations
do not turn reference-empty boxes into verified negative training labels.
The existing unknown-supervision rule remains unchanged, and every image box
remains part of inference/output accounting.

## What was verified, and its limits

- Current recipe/source/runtime identities match the completed bulk report.
- All 40 completion receipts, 40 guide files and 2379 point-packet hashes pass.
- Every packet has 2048 unique in-range source IDs; native XYZ agrees with those
  IDs and lies inside its box. Crop centers, coordinate scaling, intensity
  normalization, feature arrays and data types agree with the input contract.
- All guide values are finite/nonnegative; zero-response decisions agree with
  every fallback flag. All 37 newly completed workers agree with the journal.
- Existing source/decoded content signatures have no exact duplicate images.

The review did not re-decode original raw images, reopen SWCs, retrain a model
or compare reconstruction quality. Full-raw intensity/coordinate fidelity and
constant-box flags were checked during generation; the review verifies their
immutable receipts and packet consistency. It additionally joins saved
reference metadata without creating targets or changing supervision.

## Point-budget decision remains open

2048 is the successful sanity-check baseline, not an established universal
input budget. The user requested a bounded matched comparison with a larger
budget, such as 8192, before treating the input budget as settled. Retaining
the full guides allows new cloud packets without rerunning Sato filtering.
New budgets need separate identified packets; do not edit completed cache
receipts or silently repurpose the old 2048 cache configuration.

Image-box metadata already reveals one required edge case: development
`p_checked6_human_allen_confocal/in_house1`, tile 134, contains only **7650 source
voxels**. Every other box contains at least 8192. A proposed `min(budget,
box_voxels)` distinct-point interface would preserve that clipped box without
padding; this needs explicit implementation/variable-count model verification
as part of the comparison, rather than a silent new rule or reference-driven
box adjustment. No new packet or model interface has been implemented here.

Next work should prepare that comparison's input/target/metric contract and
measure the complete PointNeXt/geometric-query model at the larger budget.
The earlier 64k result covers an untrained encoder diagnostic, not the complete
current model. The original four boxes already fit at 2048, so fitting them
again alone cannot establish a larger budget's generalization benefit.
Dataset, exposure, metrics and success/failure criteria must be specified before
a paired trained study or a broader single-budget fit is launched.

The reference-target cache, broader loader/metrics and training runner are
still subsequent work. **The image cache is complete; training is not ready.**
No GPU command or automatic GATv2 handoff is authorized by successful preparation.

## Evidence and reproduction

- [Portable completion-review evidence](evidence/node_cloud_cache_completion_20261102.json)
- User run: `tmp/node_cloud_cache_bulk_prepare_20261031/report.json`
- Review: `tmp/node_cloud_cache_review_20261102/report.json`
- Image cache: `tmp/node_cloud_cache_20261031`
- [Earlier block comparison and original launch command](node_cloud_cache_timing_results.md)

The original CPU preparation command is complete and should not be repeated.
An existing read-only verification entry point is:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_cloud_cache_bulk.ps1 -Action plan `
  -Output ".\tmp\a_fresh_completed_cache_verification"
```

This verifies completed cache hashes/identities and should report zero pending
images. It does not filter images or repeat the packet-array/guide-value review.
