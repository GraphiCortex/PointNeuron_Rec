# Seven-neuron graph-cutoff validation: prepared

The user approved extending the neuron-21 comparison to all seven frozen
validation neurons. Preparation completed on 11 October 2026. All source
image/reference and checkpoint hashes match. No new reconstructions have run.

## Fixed comparison

Validation neurons: **19, 21, 35, 44, 79, 94, 135**. Neuron 21's completed pair
is verified and reused from its original files. New work therefore covers
**six proposal generations and twelve graph reconstructions**, with no training.
Eight test neurons remain untouched.

Every neuron uses the same epoch-five repaired-local checkpoint. Cutoffs 0.85
and 0.5 share a 128-node maximum, distance anchors at 18 voxels, geodesic adaptive
selection and connectivity defaults. Within each neuron, both arms reuse the
exact aggregated proposals, verified by path and hash. The full image-defined
grid keeps the existing maximum 256-patch coverage preset, foreground recipe,
random 512-input selection and sphere suppression. Only the graph score gate
varies. No production cutoff has been promoted.

The [neuron-21 results](graph_cutoff_results.md) show a development-case spatial
benefit with a topology tradeoff. This internal comparison tests consistency
across sources; it is not an independent test set.

## Run in your terminal

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\compare_validation_cutoffs.py --stage run `
  --output-dir .\tmp\pointneuron2\local_runs\validation_graph_cutoffs_20261011 `
  --device cuda
```

Proposal inference uses the local RTX 3070 Ti Laptop GPU. Image processing,
tracing, selection and evaluation also involve CPU work, so GPU load varies.
Neuron 21 took 2m14s; the other sources differ in volume/foreground size, so
multiplying that time by six is not a reliable ETA. The earlier heterogeneous
wrong-reader evaluation took 81 minutes for 21 reconstructions; corrected broader
runtime remains unknown. Expect a longer run than neuron 21, with live output.
Send the final seven-neuron mean lines, status and any failures.

## Outputs and failure handling

Root directory:

```text
C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\validation_graph_cutoffs_20261011
```

- `validation_comparison.json`: live status, expected denominator, case paths,
  failures, elapsed time for this new run and mean metrics when complete.
- `neuron_0019`, `neuron_0035`, `neuron_0044`, `neuron_0079`, `neuron_0094`,
  `neuron_0135`: each contains its paired `comparison.json`, shared proposals,
  two cutoff folders, graph/SWC files, metric JSON/CSV, HTML views and logs.
- Neuron 21 references `graph_cutoffs_neuron21_20261011`; no copies or reruns.

Failures remain recorded and remaining neurons are attempted. Means are withheld
if a pair is missing or failed; there is no mean over survivors. Foreground-cap
failures remain in the seven-neuron scores and are flagged. Keyboard interruption
stops execution and preserves partial outputs. Failed/partial folders are not
overwritten or silently resumed; inspect failures and prepare a fresh run if needed.

## Assessment and verification

Compare per-neuron spatial precision/recall/F1 and equal-neuron means. Look for
gains across sources rather than one case dominating the average. Read topology
separately because its induced reference graph changes with selected nodes.
Inspect bridges, missing branches, foreground behavior, graph counts and geometry.
Neuron 94 may remain poor if useful proposals are already absent before selection.

The [validation driver](../../scripts/pointneuron2/compare_validation_cutoffs.py)
uses the unchanged [single-neuron runner](../../scripts/pointneuron2/compare_graph_cutoffs.py).
The [prepared snapshot and per-case hashes](validation_graph_cutoffs_seed0/prepared.json)
are tracked; the ignored run JSON is the live status.

Eight orchestration tests passed: shared proposals, tampering detection, only
cutoff/output differences, avoiding neuron-21 reruns, continuing after failure,
withholding survivor-only means and including cap failures in equal-neuron F1.
External reconstruction commands are simulated; quality requires actual runs.
Preparation verified the corrected cache/partition, paired training, all seven
validation source pairs and neuron 21's saved outputs. The new plan freezes 119
source/configuration files. Child runs recheck provenance, and all completed
outputs and metric reports are reverified before aggregation.

For a fresh preparation if needed (CPU/provenance only):

```powershell
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\compare_validation_cutoffs.py --stage prepare `
  --experiment .\tmp\pointneuron2\local_runs\repair_20261011_130158 `
  --reuse-comparison .\tmp\pointneuron2\local_runs\graph_cutoffs_neuron21_20261011\comparison.json `
  --output-dir .\tmp\pointneuron2\local_runs\validation_graph_cutoffs_repeat
```
