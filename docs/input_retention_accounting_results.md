# Input-retention accounting: six existing training sources

23 October 2026. **Complete.** Exact repeated-value accounting is feasible,
but it does not resolve finite-cloud compression across the six inputs.
Only 13 of 135 nonconstant boxes fit all pixel-value exceptions within the old
dense quota, before context. Do not interpret the many constant boxes as
evidence that the difficult input regions now fit.

This is the bounded check specified in the
[Phase 1 discussion draft](phase1_input_retention_draft.md). It reads the six
already inspected source images, with source and decoded hashes checked against
the frozen sampling provenance. No SWCs, guides, point arrays, new clouds,
model, development/reserved images or training are involved.

## What was counted

Use the unchanged disjoint clipped 128-cube boxes. In each box, `v0` is its
most frequent pixel value; the smaller value wins a tie. An exception is any
pixel differing from `v0`, brighter or darker. These descriptions include
noise and do not identify neuronal foreground or biological background.

Compare exception count with the two existing quotas, `ceil(V/4^3)` and
`ceil(V/8^3)`. Context points and encoder features are excluded. A count fitting
the quota is therefore only a capacity lower bound, not a verified model input.

| Source / historical alias | All boxes | Constant boxes | Nonconstant boxes whose exceptions fit dense quota | Nonconstant boxes whose exceptions fit lean quota |
|---|---:|---:|---:|---:|
| Fruitfly 32 | 32 | 0 | 0/32 | 0/32 |
| Cultured human 46 | 16 | 0 | 0/16 | 0/16 |
| Frog 22 | 16 | 0 | 0/16 | 0/16 |
| Mouse 75 | 49 | 0 | 0/49 | 0/49 |
| Silkmoth 85 | 32 | 27 | 5/5 | 4/5 |
| Janelia 126 | 64 | 47 | 8/17 | 4/17 |
| Total | 209 | 74 | 13/135 | 8/135 |

Counting constant boxes too would give 87/209 dense fits and 82/209 lean fits.
That obscures how much detail still lies in over-capacity boxes. Only **5.76%**
of Janelia's exceptions lie in boxes that fit the dense quota; **94.24%** lie in
the nine boxes that do not. At the lean quota only 0.34% lie in fitting boxes.
For silkmoth, dense fitting boxes contain all 10,543 exceptions, but lean
fitting boxes contain only 20.54%; the remaining 79.46% lie in one crowded box.
The global point total alone therefore does not determine local feasibility.

## Storage arithmetic is a separate finding

The hypothetical payload is one typed default value plus uint32 local C-order
indices and original typed exception values. Compare that with the decoded
dense voxel-value payload, choosing the smaller per box. Common domain/type
metadata, Python/allocator overhead and compression are excluded. This is not
a serialized format, source-file size, measured RAM saving or model budget.

| Source | Exceptions across boxes | Dense decoded payload, bytes | Smaller mixed payload, bytes | Hypothetical payload reduction |
|---|---:|---:|---:|---:|
| Fruitfly 32 | 9,899,927 | 11,779,911 | 11,779,911 | 0% |
| Cultured human 46 | 12,039,361 | 31,457,280 | 31,457,280 | 0% |
| Frog 22 | 13,307,661 | 42,467,328 | 42,467,328 | 0% |
| Mouse 75 | 3,350,159 | 19,840,000 | 14,966,629 | 24.56% |
| Silkmoth 85 | 10,543 | 42,991,616 | 52,747 | 99.88% |
| Janelia 126 | 1,355,922 | 109,670,820 | 6,779,674 | 93.82% |

Modal values were 1 or 3 in fruitfly, 36 in cultured human, 258 or 259 in frog,
1 in mouse, and 0 in silkmoth/Janelia. Repetition is not synonymous with zero.
In mouse, storage arithmetic improves even though no box's exceptions fit a
model quota. This directly illustrates why a smaller source representation
does not prove a manageable PointNeXt cloud.
The old quotas are experimental comparators, not measured GPU capacity limits.
These counts do not establish that PointNeXt cannot handle the data, nor choose
a future crop size or point cap.

## Decision and stopping point

Keep exact default/exception accounting as a possible optimization for sparse
regions. Do not introduce a mandatory new source cache or promote this as a
general solution to point selection. The proposed retain-all branch has a
bounded useful domain; lossy selection remains necessary at the recorded caps
throughout the first four sources and in most of Janelia's exception data.

The next decision is the PointNeXt input contract: operator implementation,
coordinates, local neighborhoods, supported short inputs, point cap and explicit
context. Inspect the reference and retained prototype before preparing a visible
bounded device check. No cap or normalization policy is selected by this result,
and no GPU run is launched. After that agreement, one actual cloud-to-encoder
path can be assessed. Another counting exercise, source codec or sampling-weight
grid is not the next deliverable.

The prior [guide comparison](phase1_qualification_results.md) remains the
retention evidence for the existing clouds. This accounting neither fixes nor
relabels its failed checks, and cannot choose an overflow selector or establish
neural learnability. The graph-reference interpretation remains a separate
supervision issue.

## Execution, checks and reproduction

The visible [audit script](../scripts/audit_input_retention.py) completed in
**40.86 seconds** on CPU. Process peak working set was 251.36 MiB, with a
peak pagefile/commit counter of 848.86 MiB. These describe this accounting job,
not hardware requirements for model inference or training.

Five synthetic round trips passed: constant nonzero values, brighter and darker
details, a modal tie, all-unique values, and noncontiguous big-endian uint16.
Three invalid dtype/shape/empty inputs were rejected. Every one of the 209 real
boxes also reconstructed exactly in a temporary in-memory representation.
Input arrays and source files remained unchanged; all six source/decoded hashes
agreed with the frozen provenance. The script imports no filter, reference
evaluator or neural library. These are numerical/engineering checks.

The [complete visible evidence](evidence/input_retention_accounting_20261023.json)
is a byte-preserved copy of the raw report, retaining all per-box counts and
headroom, source/code hashes, fixture results and times. Raw case reports and
launch metadata are in ignored `tmp/input_retention_accounting_20261023/`.
Report SHA-256:
`8929ed44fc4948ea044df8f712b264e4a0e88fadb3e8b50eae9f1deac40b4103`.

Executed from the PN2 root:

```powershell
& .\tmp\phase1_cpu_env\Scripts\python.exe .\scripts\audit_input_retention.py `
  --cases .\configs\phase1_raw_review.json .\configs\phase1_qualification_cases.json `
  --frozen-clouds .\tmp\phase1_sampling_20261022\clouds.json .\tmp\phase1_qualification_disk_20261022\clouds.json `
  --recipe .\configs\phase1_sampling_probe.json `
  --output-dir .\tmp\input_retention_accounting_20261023
```

The completed output directory is preserved. A fresh directory is required
for any justified replay; no replay or downstream experiment follows automatically.
