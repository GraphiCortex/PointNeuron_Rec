# Neuron 21 graph-cutoff comparison: prepared

**Update:** this run completed in 133.7 seconds. See
[the measured reconstruction results and projections](graph_cutoff_results.md).
The prepared-state instructions below are historical; do not run the completed
comparison again in the same directory.

The user agreed to the paired full-volume development check after the
[score/suppression audit](selection_audit.md). CPU preparation and provenance
checks have completed. No proposal inference, reconstruction or training has
been launched for this comparison.

## Fixed inputs and controlled change

- Dataset: Gold166, validation neuron 21,
  `p_checked6_frog_scrippts/done_err_Recon112012no2-2`.
- Image dimensions: 512 x 512 x 41, one channel. Image and full-reference SWC
  hashes match the corrected cache. No train or reserved test neurons are used.
- Checkpoint: epoch-five `repaired_local/proposal.pt` from
  `repair_20261011_130158`, with local coordinate features and random sampling.
- One full-volume image-defined inference grid, limited to the existing
  256-patch coverage preset; 48-voxel patch radius/stride, 1,024 candidates and
  512 selected inputs, foreground at 20% of decoded image maximum.
- Aggregation: score 0.5, sphere IoU 0.1, local maximum 512 and global maximum
  4,096 proposals. Both graphs reuse the exact same aggregated NPZ, verified
  by path and SHA-256 before/after execution.
- Graph: existing geodesic/adaptive-connected-coverage selection, distance
  anchors at 18 voxels, maximum 128 selected graph nodes. Exported SWC nodes
  can exceed 128 because graph edges are resampled.
- **Only the graph minimum proposal score varies: 0.85 versus 0.5.**

The checkpoint is not retrained. Production settings and the original coordinate
experiment remain unchanged. The existing foreground cap, connectivity settings
and fallback bridges remain part of the controlled reconstruction. Their actual
behavior will be reported, rather than assumed to be correct.

## Run in your terminal

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\compare_graph_cutoffs.py --stage run `
  --output-dir .\tmp\pointneuron2\local_runs\graph_cutoffs_neuron21_20261011 `
  --device cuda
```

The script prints live output for each stage:

1. Generate the shared proposals once, with your RTX 3070 Ti Laptop GPU for
   neural inference. Image processing and selection also involve CPU work.
2. Reconstruct cutoff 0.85 and score spatial precision/recall/F1 and topology.
3. Reconstruct cutoff 0.5 from the same proposal file and score the same metrics.

There is no training. Full-volume graph tracing and evaluation use CPU work too,
so GPU utilization will vary. Corrected-input runtime is not measured yet. The
historical heterogeneous evaluation averaged about four minutes per reconstruction;
that is context, not a neuron-21 timing estimate. Shared inference avoids
generating the proposals twice.

## Where to find outputs

Everything generated belongs under:

```text
C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2\tmp\pointneuron2\local_runs\graph_cutoffs_neuron21_20261011
```

| Path under that directory | Contents |
|---|---|
| `comparison.json` | Prepared provenance, status, results, hashes and total runtime |
| `shared_proposals/sample_0021_proposals.npz` | Shared full-volume proposals |
| `shared_proposals/sample_0021_proposals.html` | Proposal/reference viewer |
| `cutoff_0_85/sample_0021/` | First graph, SWC and comparison HTML |
| `cutoff_0_5/sample_0021/` | Second graph, SWC and comparison HTML |
| Each cutoff folder's `spatial.json` and `topology.json` | Metric results, also saved as CSV |
| Top-level `.log` files | Live inference, reconstruction and evaluation logs |

The runnable [comparison script](../../scripts/pointneuron2/compare_graph_cutoffs.py)
and [preparation snapshot](graph_cutoff_comparison_seed0/prepared.json) are visible
and Git-tracked. The ignored run's `comparison.json` is the live status; the
documentation snapshot records only the prepared state. A completed run should
print `Status: completed; paired checks: passed` and metrics for both cutoffs.
Send those final metric lines so we can inspect the result together.

## What determines success

We need a useful gain in spatial recall/F1 with acceptable precision and visible
branch geometry, not simply more nodes or a valid tree. Inspect both SWCs, bridge
dependence, reachable edges, foreground-threshold adaptation and whether its cap
was satisfied. Topology F1 uses each output's own node set and can change when
nodes change; it must be read alongside spatial coverage. Spatial metrics use
the full reference, six-voxel matching and one-voxel SWC resampling.

Neuron 21 was chosen using validation failure evidence. This is a development
case, not confirmation of generalization. A benefit here would justify the same
comparison across all seven validation neurons, including failures. Neuron 94's
poor proposal support requires a separate investigation. No cutoff is promoted
on the basis of the crop audit alone.

## Verification and preservation

Three orchestration tests passed: commands vary only by graph cutoff and output
directory, both arms reuse one proposal file, and proposal-file tampering aborts
the comparison while preserving the first arm's result. These tests simulate
external commands; they are not a reconstruction smoke test or quality result.

Actual preparation verified the original paired-training checks, all corrected
cache hashes, validation identity, source image/reference/checkpoint hashes and
reader compatibility. The new plan freezes 117 source/configuration files and
rechecks them before and after the run. The script verifies graph settings,
shared proposal identity/hashes and a one-neuron metric denominator in both arms.
Errors and partial results remain saved; a failed/interrupted run is not silently
resumed or overwritten. Use a fresh comparison directory if a rerun is needed.

To prepare a separate repeat (CPU/provenance only):

```powershell
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\compare_graph_cutoffs.py --stage prepare `
  --experiment .\tmp\pointneuron2\local_runs\repair_20261011_130158 `
  --output-dir .\tmp\pointneuron2\local_runs\graph_cutoffs_neuron21_repeat
```

Then run with `--stage run --output-dir` pointing to that fresh directory.
