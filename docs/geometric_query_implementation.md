# Geometric queries: implemented candidate and CPU qualification

31 October 2026. The user approved the [geometric-query proposal](geometric_query_proposal.md).
The separate candidate is implemented and CPU-verified. **No GPU timing or
training has been launched by the agent.** The completed failed GN fit remains
the comparator; its sources, configuration, checkpoints and verdict are unchanged.

## Implemented mechanism

`src/pn2/models/geometric_queries.py` reuses the fresh native head's named modules.
It introduces no parameters and performs no random initialization. The model
factory checks identical parameter/state hashes, names and RNG before/after
wrapping. All **1,688,518** trainable parameters retain the exact initial GN
baseline values. No learned checkpoint is loaded.

```text
2048 fixed image points -> same GN8 PointNeXt -> 2048 restored feature vectors
                                                       |
             +-----------------------------------------+-------------------+
             |                                                             |
  All 2048 vectors + positions as memory           FPS512 on input XYZ only
             |                                      gather features/position
             |                                      add learned query token
             |                                                             |
             +------------------> same Transformer decoder <---------------+
                                             |
                          center = input starting point + 64 * learned offset
                          existing confidence, radius and decoder features
```

Coordinates and intensity retain the fixed input adapter. FPS uses native
hash-ordered source points in crop-local XYZ, starting at index zero with the
pinned operator's first-index tie behavior. Queries receive the existing
projected feature plus positional representation at their starting point and
the existing learned token. All input points remain in attention memory.
FPS indexing is discrete and is not differentiated; feature gathering and the
Transformer remain differentiable.

The output displacement is unbounded. The public `local_centers` output remains
`(predicted_center - crop_center)/64`, rather than being redefined as an offset
from the starting point. This preserves the loss/downstream coordinate contract.
Radius is still unsupervised; no radius-quality claim is made.

There are 512 queries on the real 2048-point packets. With fewer input points,
the head uses `min(512,N)` queries, selecting distinct indices even when positions
coincide. This avoids invented locations/padding but does not make duplicate
coordinates informative. Whole-encoder minimum input size remains 256; empty
clouds and broader product input handling are not qualified here.

## Fixed comparison

Same four training boxes: fruitfly 32/27,32/20 and frog 22/2,22/5. Target counts
18/85/7/68. Same frozen Sato clouds, fresh encoder/head parameter values, GN8,
2048 input slots, 512 queries, node matching/loss, AdamW LR 0.0001, clipping,
cutoff 0.5, FP32 deterministic math attention and 2000-update cap. No augmentation,
schedule, extra loss or fitted confidence stage is added.

The two deliberate changes are query content initialization and position
parameterization. Their effects are tested as one candidate; a positive result
would not isolate their separate causal contributions. PointNeXt/Transformer
are retained. Consolidation/GATv2/final SWC are still later stages and are not
implemented or trained by this comparison.

`configs/geometric_query_fit.json` derives from the frozen GN configuration,
pins the completed GN report and the two new candidate source files. Baseline
code and target/cloud hashes are checked through the existing helpers. The
candidate-specific hashes are bound into saved checkpoint recipes/fingerprints;
the shared checkpoint payload's base `code_identity` remains unchanged.

## Completed CPU evidence

Raw report: `tmp/geometric_query_cpu_20261031/report.json`.
Visible [evidence](evidence/geometric_query_cpu_20261031.json).

| Check | Result |
|---|---|
| Fresh parameter names/values/count versus GN baseline | Exact match; 1,688,518 parameters |
| Real-cloud ordinary eval versus train-mode output | Exact centers, local centers, logits, radii and decoder features in all four boxes |
| Actual node loss backward on all four real clouds | All parameter gradients present and finite; confidence/position gradients nonzero; direct radius-row gradients zero |
| Real-data model state | Unchanged; zero real-data optimizer updates |
| Synthetic budgets | N=256 gives 256 queries; N=4096 gives 512; exact mode consistency |
| Serialization | One synthetic throwaway Adam update; exact model/Adam/RNG reload and output replay |
| GPU execution during preflight | None |
| Focused tests | 47 pass in 19.42 seconds |
| PowerShell launcher parsing | Pass |

Tests cover zero offsets reproducing raw starts, unbounded movement, translation
and coordinate-scale semantics, distinct index behavior on tied coordinates,
selected feature/position query content and retained full attention memory.
Unselected feature vectors receive finite nonzero gradients in the fixture;
perturbing them changes the predictions. This establishes a computation path,
not useful branch information or a guarantee of learned feature use.

The 47-test command covers `test_geometric_queries.py`, `test_groupnorm_node.py`,
`test_node_learning.py` and `test_sampler_node_controller.py`, with external
pytest plugin autoload disabled in the retained runtime.

### Starting prior versus actual untrained model

One-to-one target matches within two voxels, before any learned update:

| Training box | Targets | Raw FPS starts | Actual candidate, all queries | Accepted at 0.5 | Mean assigned candidate error |
|---|---:|---:|---:|---:|---:|
| fruitfly 32/27 | 18 | 5 | 4 | 0 | 2.71 voxels |
| fruitfly 32/20 | 85 | 7 | 6 | 0 | 3.74 voxels |
| frog 22/2 | 7 | 1 | 1 | 0 | 3.97 voxels |
| frog 22/5 | 68 | 4 | 6 | 0 | 4.59 voxels |

The actual untrained head's small offsets can help or hurt raw start coverage.
Its closer initial geometry comes from the spatial prior, not training. Raw
starts alone do not cover the skeleton accurately, and low initial confidence
does not constitute a failure of this execution check. Input retention,
encoder feature sufficiency and node learnability remain unproven.

## Next manual action: throwaway GPU timing only

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_geometric_query_fit.ps1 -Action timing
```

One worker on the local RTX 3070 Ti Laptop GPU, eight throwaway updates on the
same four training boxes, a 300-second hard cap. It checks actual node-loss
execution, exact mode consistency and model/Adam/RNG reload, saving generated
reports/predictions/logs under `tmp/geometric_query_timing_<timestamp>`.
No trained checkpoint is retained and timing does not launch training.

The CPU check is already complete and current; it does not need repeating.
If a future source change makes it stale, `-Action check -Output <fresh path>`
and a corresponding `-CpuPreflight <report>` are explicit options, not retries
for a failed model-quality gate.

The separate train action requires this candidate's own passed `-TimingRun`;
the old GN timing is rejected. Review candidate timing before launching the
fit. Its first eight updates must reproduce timing's model/Adam/RNG prefix and
are included in the 2000-update cap. States are saved before evaluations at
0/8/500/1000/2000; no automatic resume or extension is offered.

Success remains >=95% two-voxel one-to-one precision **and** recall in every
box under ordinary inference, at cutoff 0.5, with exact mode consistency. A
failed intermediate node gate is reported but does not select settings or
change the cap. A terminal failure closes this candidate and triggers the
explicit query/supervision formulation review described in the proposal.
Even a pass would establish only four-box training learnability, not broader
reconstruction improvement, PN1 superiority or a universal software capability.
