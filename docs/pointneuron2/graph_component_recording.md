# Neuron 79: record actual component allocation

**Completed; recording checks passed.** The user ran the unchanged graph replay
in 231.7 seconds. Eligible components 2–12 receive zero anchors; components 0/1
consume all 128. Every graph array and metadata match the original exactly.
See the [actual trace results and fixed-reference support audit](graph_component_trace_results.md).
The separate [repaired reconstruction and follow-up audit](remaining_budget_results.md)
have since completed. The protocol below describes the unchanged recording run.

The user approved recording the actual
allocation after the [saved graph/path and image audit](hybrid_graph_path_audit.md).
This is one unchanged hybrid graph replay to resolve whether the synthetic
component-allocation starvation also occurs on the real 625-candidate pool.
It is not a repaired selector comparison or a new reconstruction-quality result.

## What is fixed, and what is recorded

Use the same saved hybrid proposals from
`tmp/pointneuron2/local_runs/hybrid_swc_neuron79_20261011/hybrid_50_50_proposals`.
The CPU preparation replays the input filter and freezes the exact 625 candidate
centers, radii, scores, features and original proposal indices. The graph uses
cutoff 0.5, distance NMS 18, 128 anchors, the requested threshold 5 that adapts
to 15, foreground cap 350,000, candidate k=6 and the same geodesic/MST/bridge
settings as the saved hybrid graph. No settings or model weights change.

The [recorder](../../scripts/pointneuron2/record_graph_components.py) calls the
existing initializer and selector functions. Temporary in-process wrappers
observe their arguments and original returned values; original functions are
restored even if the recording fails. The production source file is not edited.
Full candidate geodesic/Euclidean matrices are saved after their computation.
The recorder stores actual per-component eligibility, requested allocations,
visited/skipped status and selected indices, including eligible components with
zero anchors. It distinguishes small components excluded by the existing
eligibility rule from eligible ones that receive zero final anchors.

Every numeric array in the repeated graph, including node identities, features,
edges and voxel paths, must match the saved graph exactly. Graph metadata and
the selected-pool-to-original-proposal mapping must also match. Those checks
assess recording transparency, not reconstruction improvement.

Four targeted tests passed: original versus recorded selection equality,
actual call/zero-component logging, input preservation and original-function
restoration after interruption. CPU preparation verified the exact saved pool,
proposal/graph hashes and frozen experiment sources. The completed actual trace
and a cached-matrix replay now confirm eligible-component starvation on neuron 79.

One graph calculation is necessary because the earlier saved graph lacks the
full 625-candidate geodesic matrix. The large foreground distance/predecessor
tables are not exported; the compact pool matrices are enough to replay and
inspect anchor selection later. Exporting new paths after a changed selector
would be a separate experiment.

## Run visibly on your machine

This step uses **CPU/SciPy**, not CUDA: foreground decoding, graph construction
and geodesic paths are CPU operations in the current implementation. It does
not run the proposal model, train, export a new SWC or rerun evaluation scores.
Your earlier hybrid graph stage took about 2m26s including graph/export/viewer
work. Expect roughly **2–4 minutes**, with variation by machine load; this is
an estimate, not a guaranteed duration.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\record_graph_components.py `
  --stage run `
  --output-dir .\tmp\pointneuron2\local_runs\graph_component_trace_neuron79_20261011
```

Progress messages show image decoding, foreground graph construction and the
geodesic calculation. The latter is the main wait and does not print a percent
complete. At the end each component's candidate count, eligibility, visited
status, requested budget and selected count are printed, followed by the
exact repeated-graph check and elapsed time.

Ignored output directory:
`tmp/pointneuron2/local_runs/graph_component_trace_neuron79_20261011`.

- `trace_plan.json`: frozen settings/input/source hashes and status.
- `expected_pool.npz`: exact prepared pool and original proposal indices.
- `console.log`: user-visible worker output, saved verbatim.
- `pool_geodesics.npz`: pool geometry, pairwise matrices and selected indices.
- `component_trace.json`: observed components and allocation records.
- `repeated_graph.npz`: the unchanged graph reproduced for verification.

Completed and interrupted runs cannot be overwritten or silently resumed.
Use a fresh preparation directory if repeating the run. For CPU preparation,
use `--stage prepare --comparison` with the completed hybrid `comparison.json`
and a fresh `--output-dir`. `--stage check` verifies a prepared plan without
starting geodesics.

## Decision after reading the actual trace together

If real eligible components receive zero nodes, a separately versioned repair
can compare remaining-budget allocation at the **same 128 nodes and same pool**.
If they do not, the synthetic problem remains valid but does not explain this
neuron; geometric versus image-path anchor coverage becomes the better next
comparison. No repair or alternative selector is implemented by this recorder,
and no metric is changed. The outcome must come from the observed trace.
