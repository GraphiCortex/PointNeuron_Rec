# Neuron 79: paired global-selection reconstruction

**Prepared; reconstruction has not run.** The user approved this focused
comparison after the fixed hybrid diagnostic. It tests whether retaining more
available branch evidence survives graph-node selection and tracing into a
better SWC. Neuron 79 is a selected validation development case; this is not a
generalization benchmark or a completed first pipeline phase.

## Fixed comparison

Both arms share the corrected epoch-five local-coordinate checkpoint, the same
156 image-defined patches, random 512 inputs from 1,024 candidates with seed 0,
local score threshold 0.5, local sphere suppression IoU 0.1, and a 4,096 global
proposal budget. The score arm uses the existing confidence priority. The
hybrid retains its first 2,048 accepted confidence anchors and uses the already
measured spatial fill for the remaining slots, with the same sphere suppression
and 16,384 candidate limit. No ratio is tuned and labels never enter selection.

Both arms use graph cutoff **0.5**, adaptive connected coverage, distance NMS
18 voxels, at most 128 graph nodes, the same geodesic connectivity settings and
foreground cap 350,000. Cutoff 0.85 would remove much of the recovered hybrid
evidence. Both arms use 0.5 so the cutoff is not a confound.

The fixed checkpoint is
`tmp/pointneuron2/local_runs/repair_20261011_130158/repaired_local/proposal.pt`.
There are no optimizer steps, training neurons or reserved test neurons in this
run. CUDA inference runs on the local RTX 3070 Ti Laptop GPU; selection, image
processing, tracing and evaluation also contain CPU work. It will not keep the
GPU fully occupied throughout.

## Preserve real features and historical provenance

The saved diagnostic pool did not include all encoder features. A separate
[aggregation snapshot](../../scripts/pointneuron2/aggregate_with_features.py)
saves the complete features and patch indices during one repeat inference.
Before every run, an AST equivalence check proves it differs from the original
aggregator only in those two output fields, its location and documentation.
The earlier frozen production files are left intact, keeping their existing
provenance checks usable.

The [paired driver](../../scripts/pointneuron2/compare_hybrid_swc.py) requires the
regenerated pool and previously accepted score-arm rows/features to match the
saved arrays **exactly**. It checks the saved hybrid source indices, fixed counts
and protected anchors, then gathers each hybrid proposal's actual encoder
feature from the same inference. It never invents features or replaces actual
objectness scores with selection priorities. Stale score-arm proposal metrics
are removed from hybrid metadata.

If the repeat pool differs, the driver stops before reconstruction and preserves
the artifacts for investigation. Both SWCs are reconstructed afresh from this
paired inference; the old score-arm SWC is retained as a repeatability reference.
Input, checkpoint, source and output hashes are recorded. Failed or interrupted
runs cannot silently resume or overwrite results; prepare a fresh directory.

## User-visible run

The CPU preparation has checked the saved experiment and hybrid identities.
No GPU inference or SWC reconstruction was launched during preparation.
Seven targeted tests passed, covering aggregation equivalence, source-row
feature retention and rejection of changed pools, misaligned features and
invalid protected anchors. These are execution checks, not reconstruction gains.
Run this from your own PowerShell terminal:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\compare_hybrid_swc.py `
  --stage run `
  --output-dir .\tmp\pointneuron2\local_runs\hybrid_swc_neuron79_20261011 `
  --device cuda
```

The earlier neuron-79 inference/two-reconstruction pair took about seven minutes.
Expect several minutes; this is a reference duration, not a guaranteed ETA.
The terminal prints patch progress, graph/tracing stages and each arm's spatial
precision, recall, F1 and topology F1. Logs, proposal files, graph arrays, SWCs,
comparison HTML files and JSON/CSV metrics stay in the named ignored run folder.
The final comparison is `comparison.json` there.

## Interpretation together

- Higher recall with preserved precision supports the allocation hypothesis.
- Higher recall with lower precision identifies a rejection/connectivity tradeoff.
- Little SWC improvement despite the retained proposal support points to a
  downstream graph/tracing limitation.

Inspect the SWCs alongside those measurements. Topology evaluation uses each
arm's selected nodes and is not a fixed identical edge-classification target.
Nothing is automatically promoted. Neuron 19's regression risk and neuron 94's
upstream missing-proposal deficit remain unresolved even if neuron 79 improves.

To prepare a fresh repeat after source changes, use `--stage prepare` with
`--baseline-comparison tmp/pointneuron2/local_runs/validation_graph_cutoffs_20261011/neuron_0079/comparison.json`,
`--hybrid-report docs/pointneuron2/hybrid_global_comparison_seed0/summary.json`
and a fresh `--output-dir`. CPU-only `--stage check` verifies an existing plan
without starting inference or tracing.
