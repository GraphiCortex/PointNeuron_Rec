# Neuron 79: remaining-budget allocation comparison

**Prepared; repaired SWC has not run.** The user approved repairing allocation
after the [real trace confirmed starvation](graph_component_trace_results.md).
The repaired selector has passed cached-matrix inspection, but its graph
requires substantially more bridging. The visible SWC comparison below measures
whether covering the skipped regions actually improves reconstruction.

## Fixed comparison and implemented repair

The [versioned allocator](../../scripts/pointneuron2/remaining_budget_selection.py)
uses the **remaining** anchor budget in proportional component shares. If the
total budget can cover every eligible component, it also reserves one anchor
for each later eligible component. The component ordering, size-at-least-three
eligibility rule, highest-score-seeded Euclidean coverage within components,
fallback for no eligible components and existing global fill remain unchanged.
Budgets smaller than the eligible component count cannot guarantee every group;
the existing order determines which groups get the available anchors.

The comparison keeps the same saved 4,096 hybrid proposals, 625 filtered pool
candidates, **128 anchors**, checkpoint/features, score cutoff 0.5, spacing 18,
foreground threshold adapting from 5 to 15, foreground cap 350,000, candidate
k=6, geodesic/bridge settings and exporter. Neither thresholds nor component
eligibility are tuned. The reference is used only to measure diagnostic support.
Production defaults and earlier frozen sources remain unchanged.

The [comparison driver](../../scripts/pointneuron2/compare_component_allocation.py)
first verifies the original unchanged trace and graph. It then applies the
versioned allocator in a temporary scope, restoring the original function on
exit. It freezes the repaired selection and all experiment sources before any
new graph computation. The run stage must reproduce the cached pool geometry,
geodesic/Euclidean matrices, allocation, selected identities and adaptive
metadata exactly. A changed pool or setting fails the comparison.

## Cached result, not SWC improvement

| Diagnostic | Original allocation | Repaired allocation |
| --- | ---: | ---: |
| Eligible components with at least one anchor | 2 / 13 | 13 / 13 |
| Allocations by rank | 101, 27, then 11 zeros | 101, 10, 7, then 10 ones |
| Total anchors | 128 | 128 |
| Anchors within 6 voxels of reference | 99 | 109 |
| Reference samples within 6 voxels of anchors | 1,629 / 16,793 | 1,800 / 16,793 |
| Reference samples within 18 voxels of anchors | 7,088 / 16,793 | 7,533 / 16,793 |
| Cached dense-graph bridge edges | 4 | 21 |
| Cached reachable edge fraction | 0.9921 | 0.9055 |

The arithmetic repair eliminates eligible-component starvation and improves
sparse anchor proximity. **Bridging increases sharply.** The changed node set
spans disconnected image regions; the existing bridge policy may add incorrect
connections. Some eligible components also have high scores but little reference
proximity. A graph that reaches more components is not automatically a better
neuronal reconstruction. These are graph/geometry diagnostics; no new SWC
precision, recall, F1 or topology result exists yet.

Fourteen targeted checks passed: seven allocator cases (including both synthetic
starvation examples, the actual size pattern, minimal feasible/infeasible
budgets, input preservation and scope restoration), four existing recorder
checks, and three worker integration checks for successful cached selection,
matrix drift rejection and fixed-setting drift rejection. The worker checks use
synthetic arrays without image/path calculation. The real cached replay and
source/input verification passed independently.

[Visible cached summary](remaining_budget_selection_seed0/summary.json) records
the measurements, allocations, provenance and limits. Generated selection
arrays, the full diagnostic and prepared plan are in:
`tmp/pointneuron2/local_runs/remaining_budget_neuron79_20261011`.

## Run in your visible terminal

This is **CPU/SciPy**, using saved proposals. It runs no training or neural
inference and needs no CUDA. It recomputes one repaired foreground graph and its
paths, checks them against the cached matrices/selection, then reuses that graph
in the unchanged SWC exporter/viewer and spatial/topology evaluators.

The original saved hybrid SWC and metrics are the paired control. Their hashes
and metric reports are checked; the preceding unchanged replay already reproduced
every original graph array exactly. No second control inference or graph is needed.
Control spatial precision/recall/F1 are **0.8961 / 0.5560 / 0.6862**.

Expect roughly **4–7 minutes**: the unchanged 625-candidate graph took 231.7s,
and export, viewer and evaluation add work. Machine load can change this estimate.
Progress shows CPU decoding, foreground graph construction and geodesics. The
geodesic stage has no percentage counter. Export and evaluation then print their
own commands and outputs.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\compare_component_allocation.py `
  --stage run `
  --output-dir .\tmp\pointneuron2\local_runs\remaining_budget_neuron79_20261011
```

Outputs in that directory:

- `comparison.json`: prepared settings, frozen sources and final paired results.
- `cached_selection.npz`, `cached_diagnostic.json`: precomputed expected selection.
- `graph.log`, `export.log`, `spatial.log`, `topology.log`: live terminal output saved verbatim.
- `actual_repaired_component_trace.json`: actual allocation and equality checks.
- `remaining_budget/sample_0079/`: repaired graph, SWC and interactive viewer HTML.
- `remaining_budget/spatial.json`, `topology.json`: final metric reports.

Completed/partial runs cannot be overwritten or silently resumed. A repeat needs
a fresh preparation directory. `--stage check` verifies this plan without running
the graph. `--stage prepare --trace-dir` can prepare a fresh comparison from the
completed unchanged trace.

## What decides the next move

Inspect the final precision/recall/spatial F1 together, with gained/lost geometry
and the new bridge paths. A recall gain accompanied by unwanted connections must
be weighed against precision. The topology targets change with selected nodes
and the reference is a 191-root forest; its edge F1 is not a fixed-target
biological-connectivity benchmark. Keep that limitation visible.

This run assesses the one agreed allocator repair on a selected validation case.
It does not promote the hybrid or repair globally. Size-rule exclusion,
within-component coverage, bridge acceptance, neuron 19's specificity guard and
neuron 94's earlier proposal deficit remain separate questions. No reserved
test neurons are used.
