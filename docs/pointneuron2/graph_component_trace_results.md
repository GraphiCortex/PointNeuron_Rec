# Neuron 79: actual component allocation confirmed

Follow-up: the [repaired SWC comparison and image/supervision audit](remaining_budget_results.md)
are now complete. The arithmetic repair removes starvation but lowers spatial F1;
the report below records the earlier unchanged trace evidence.

The user completed the unchanged CPU graph replay in **231.7 seconds**
(terminal total 3m57s). Every saved graph array and its metadata match the earlier
hybrid graph exactly. Recording did not change selection or reconstruction.
A second CPU replay of the unchanged selector using the cached matrices matches
the actual component records, selected indices and adaptive metadata exactly.

This establishes an allocation defect on the real neuron, beyond the earlier
synthetic counterexamples. It does not measure a repaired reconstruction.

## Where the 128 anchors went

The 625-candidate pool has 80 connectivity components. The existing size rule
admits 13 components with at least three candidates, totaling 545 candidates.
It excludes 67 components of size one or two, totaling 80 candidates.

| Component rank | Candidates | Mean score | Anchors | Candidates within 6 voxels of reference |
| --- | ---: | ---: | ---: | ---: |
| 0 | 429 | 0.830 | 101 | 404 |
| 1 | 44 | 0.854 | 27 | 0 |
| 2 | 31 | 0.902 | 0 | 0 |
| 3 | 9 | 0.986 | 0 | 9 |
| 4 | 6 | 0.892 | 0 | 6 |
| 5 | 4 | 0.981 | 0 | 4 |
| 6 | 4 | 0.978 | 0 | 4 |
| 7 | 3 | 0.991 | 0 | 3 |
| 8 | 3 | 0.983 | 0 | 3 |
| 9 | 3 | 0.977 | 0 | 3 |
| 10 | 3 | 0.976 | 0 | 3 |
| 11 | 3 | 0.832 | 0 | 3 |
| 12 | 3 | 0.548 | 0 | 3 |

Only two component coverage calls occurred: 101 anchors from component 0,
then 27 from component 1. Components 2–12 were never visited. They passed
eligibility; their zero allocations were not score rejection.

The existing loop uses the original `max_nodes` numerator as its source
denominator shrinks:

```text
component 0: round(128 * 429 / 545) = 101; 27 anchors remain
component 1: round(128 * 44 / 116) = 49; clipped to the remaining 27
remaining budget = 0; the other 11 eligible components are not visited
```

The shrinking denominator therefore lets an early component spend the budget
intended for later components. The defect is in graph-anchor allocation;
additional neural training cannot directly correct this arithmetic.

## Relation to missing reference geometry

The same fixed reference has 16,793 resampled curve points. The saved hybrid
SWC misses 7,456 at the established 6-voxel evaluation tolerance.
Components below are candidate connectivity groups, not biological branches.

| Group | Components | Candidates | Anchors | Missed SWC samples within 18 voxels of group | Missed samples exclusively near this group at 18 voxels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Allocated components | 2 | 473 | 128 | 5,176 | 3,452 |
| Eligible but starved | 11 | 72 | 0 | 985 | 452 |
| Excluded by size rule | 67 | 80 | 0 | 3,164 | 1,472 |

The proximity columns can overlap across groups; the exclusive column requires
every candidate outside that group to be farther than 18 voxels. Nearest-pool
assignment partitions supported missed samples into 4,088 / 792 / 2,424 across
these groups, respectively. The 18-voxel diagnostic matches the established
graph spacing context; final SWC recall remains evaluated at 6 voxels.

**41 of the 72 starved candidates are within 6 voxels of the reference.** They
belong to all ten eligible components ranked 3–12, including several with
mean scores above 0.97. Allocation starvation therefore discards substantial
reference-supported candidate geometry. However, component 2 is high scoring
and has no candidate within 6 voxels of the reference. Component 1 also has
none at that tolerance despite consuming 27 anchors. Scores alone do not
identify reference proximity; annotation incompleteness or other image
structures remain possible. Spreading anchors can also introduce unwanted paths.

The bug explains a relevant subset of losses, not the whole reconstruction.
Many missed samples are near allocated components, while the independent
size eligibility rule excludes other supported geometry. Paths can pass near
a component without having an anchor in it; zero allocation does not prove
zero traced coverage. Proximity does not prove continuous image connectivity,
neurite identity or that a replacement anchor will improve the final SWC.

## Agreed next comparison

The user approved a narrow, separately versioned repair: use the remaining
anchor budget in proportional allocation and reserve one anchor for every
eligible component when the budget permits. Keep the 625-candidate pool,
128-anchor total, component ordering, size eligibility, within-component
coverage selection, model weights and graph settings fixed.

The [cached repair comparison and visible reconstruction command](remaining_budget_comparison.md)
were prepared from this trace. All 13 eligible components get anchors, but bridge
edges increase from 4 to 21. The completed SWC regresses overall; see the follow-up
above. Actual SWC
precision, recall, spatial F1, paths and bridging must determine improvement;
candidate support is not a substitute. Changing the size eligibility policy,
training again or promoting a new default would be separate decisions.

## Reproducible saved-output inspection

[Machine-readable summary](graph_component_trace_results_seed0/summary.json)
contains all 80 component rows, support counts, provenance hashes and limits.
The [read-only summarizer](../../scripts/pointneuron2/summarize_component_trace.py)
rechecks frozen sources, inputs, repeated graph arrays and unchanged selector
replay before measuring support. Its successful execution took approximately
13 seconds on CPU; it starts no image decoding, geodesics or inference.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\summarize_component_trace.py `
  --trace-dir .\tmp\pointneuron2\local_runs\graph_component_trace_neuron79_20261011 `
  --output .\tmp\pointneuron2\component_trace_summary_repeat.json
```

Use a fresh output filename for repeats. The actual trace arrays and logs remain
in the ignored run directory documented in the [recording protocol](graph_component_recording.md).
This is one selected validation case; no reserved test neurons were used.
