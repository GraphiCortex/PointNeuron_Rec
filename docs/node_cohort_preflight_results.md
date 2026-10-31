# Broader node cohort: capacity and supervision preflight

31 October 2026. **The bounded CPU audit completed in 107.15 seconds.** It
read the explicitly listed 33 training and seven development image headers and
SWCs. It decoded no image payload, generated no guide/cloud, executed no model
and opened no reserved image/reference files. This is data/capacity evidence,
not learned performance or a ready training cache.

## Counts under the unchanged representation

Disjoint clipped 128-voxel boxes, eight-voxel full-chain targets with required
structural/boundary nodes, capacity 512, candidate input count 2048. The actual
frozen target constructor was called with a large counting-only capacity; all
target identities were retained and counted against the model's unchanged limit.

| Population | Neurons | Image boxes | Reference-present boxes | Reference-empty / unknown boxes | Maximum targets per box | Overflows |
|---|---:|---:|---:|---:|---:|---:|
| Training | 33 | 1964 | 460 | 1504 | 419 | 0 |
| Development | 7 | 415 | 151 | 264 | 233 | 0 |

Every box contains at least 2048 source voxel locations. No short-input padding,
target truncation, smaller crop or increased query count is needed for this
cohort. This is a count/capacity finding, not proof that 2048 selected points
retain adequate image evidence. Across the fixed grid there are 20,643 training
and 6,930 development target identities; crop-boundary identities and annotation
multiplicity prevent interpreting those as unique whole-neuron anatomical nodes.

All 40 references parse as valid forests under `opaque_integer` type metadata.
No reference vertices lie outside the image voxel-cell bounds in this audit.
Neither finding certifies image alignment, biological completeness or topology.
The existing fragmented/coincident encoding findings remain unresolved; target
coincidence also occurs in development cases 21, 35 and 79. Nothing was welded,
merged, pruned or assigned an anatomical radius/type interpretation.

## Isolation and annotation meaning

Stable sample IDs match the retained 33/7/8 partition with no identity overlap.
Exact image/reference paths do not cross training/development, and none of the
40 selected reference files have identical content hashes. Acquisition-family
independence and nonidentical duplicate images remain unverified. Full source
image hashes and duplicate-image checks belong in the raw-image cache build;
a 43-byte header hash cannot identify image content.

The seven development aliases remain 19, 21, 35, 44, 79, 94 and 135. They have
been inspected repeatedly and were exposed to historical PN1 training. They
can be gradient-disjoint from this fresh PN2 fit, but are development data,
not independent product confirmation. The eight reserved identities were read
only as partition metadata to guard against accidental inclusion; their source
files were not opened.

Reference-empty crops are **unknown supervision**, not certified background.
They remain part of the image domain and must be inferred/scored/accounted for
without a reference-dependent mask. The 460 reference-present training boxes
are the proposed gradient-bearing examples; development never supplies gradients.
Unmatched predictions measure disagreement with the available trace, not proven
biological false positives.

## Resource arithmetic before bulk preparation

The 40 images contain **2,915,485,754 voxels**. Keeping one float32 guide per
image would use about **10.86 GiB**, before file headers. Twelve float32 work
arrays for the largest image would use about **7.33 GiB of disk space**, not a
measured process RAM requirement. All 2379 clouds at 2048 points contain
4,872,192 point slots, with their provenance/features stored separately.
At inspection, drive C had about 59.29 GiB free; this must be rechecked before
bulk generation rather than treated as reserved capacity.

The [existing exact disk guide](phase1_disk_guide.md) avoids the prior large
resident intermediate arrays, but decoding/normalization still have their own
memory cost. Prior small-image guide measurements show that full-cohort CPU
preparation can be a many-hour task. No trustworthy whole-cohort completion
estimate or hardware minimum follows from this arithmetic. Reuse only exact
PN2 guides whose image, normalization, operator/recipe and guide hashes match;
do not substitute historical learned-field or reference-selected caches.

## Reproduction and checks

Visible sources: `scripts/audit_node_cohort.py` and
`configs/node_cohort_preflight.json`. Raw report:
`tmp/node_cohort_preflight_20261031/report.json`. Portable
[case-level evidence](evidence/node_cohort_preflight_20261031.json) includes all
40 summaries and source/header hashes.

Six audit fixtures verify cohort isolation before source access, reserved-ID
rejection, exact source-path overlap rejection, unknown-crop accounting,
header-only behavior, literal overflow/coincident-identity counting, short-input
reporting and the wall cap. Together with five exposure-planning tests,
**11 tests pass**. No real reference/source bytes were changed.

The [broader protocol](node_generalization_protocol.md) uses these counts to
make a reviewable 9200-update plan. It does not launch training or qualify the
cohort's input preparation as complete.
