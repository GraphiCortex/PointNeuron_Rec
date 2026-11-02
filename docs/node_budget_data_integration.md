# Paired data, shared targets and core model integration

2 November 2026. The full 2048/8192 comparison data are built and independently
verified. The real-data CPU model checks, core cohort metrics and exact state
restore checks pass. **No integrated GPU timing or broad fit has been launched.**
The unchanged architecture remains automatic image points -> PointNeXt ->
geometric-query Transformer -> later GATv2/connectivity/reconstruction.

## Completed data integration

The new image-only cloud builder reads original raw intensities and the retained
full-resolution guide. It accepts no reference/target input and runs no filter.
Every higher-budget cloud retains all baseline locations and identical shared
attributes. The old 2048 cache is reused intact.

| Partition | Images | All boxes | Reference-present boxes | Reference-empty/unknown |
| --- | ---: | ---: | ---: | ---: |
| Training | 33 | 1964 | 460 | 1504 |
| Development | 7 | 415 | 151 | 264 |

The 2048 view contains **4,872,192** source-point observations; the nominal 8192
view contains **19,488,226**. One clipped development box, human Allen
`in_house1` tile 134, uses all 7650 real source locations. There is no duplicate
padding, dropped box, added channel, relocated point or reference-guided sampler.
Preparation took **875.3 CPU-process elapsed seconds (about 14.6 minutes)**,
without repeating the Sato filtering. No original image/guide/cache was edited.

The separate reference-only builder creates **2379 shared target packets**.
Both budgets use exactly the same XYZ, structural roles, identities and known
reference edges/runs. Full-chain step eight, closed voxel-cell crop boundaries,
coincident identities and fragmented encodings are preserved. Radii remain
metadata only. Every box's target count/grid matches the prior cohort audit;
overflow raises rather than truncates or merges.

One schedule is shared by both arms: **20 epochs / 9200 updates**, covering each
of the 460 supervised training boxes once per epoch. Its arrays exactly match
the prior exposure plan. Development and unknown boxes cannot enter the
training-loader path. Inference opens only an image packet; the evaluator
opens the shared reference packet after the model forward. All image boxes
remain in evaluation/output accounting.

Generated data: `tmp/node_budget_data_20261102`. The original 2048 packets and
guides remain at `tmp/node_cloud_cache_20261031`; new packets refer to them
without overwriting/copying the guide volumes.

## Verification and core metrics

Independent verification loaded both cloud views and shared targets for all
2379 boxes, checked packet hashes/counts/coordinates/features/reference-run
schemas and baseline inclusion, and verified schedule coverage. It did not
train a model or evaluate learned reconstruction quality.

The Torch CPU integration check then confirmed:

- All original four training clouds/targets exactly reproduce the qualified
  timing inputs at both budgets.
- The **real 419-target crop** (training alias 86, tile 20) passes full-model
  mode consistency, node-loss backward and finite-parameter-gradient checks
  at 2048 and 8192 points, without an optimizer update.
- The actual 7650-point development box passes inference only, with 512 finite
  proposals and no development gradients. An empty-reference loss abstains;
  the loader rejects unknown boxes from training.
- A synthetic-only state round trip reproduces the next update exactly,
  including model, Adam and RNG, with budget/data/schedule/cursor bindings.

The new core scorer reports one-to-one node precision/recall/F1 at two/six
native voxels, pools all boxes per neuron before macro averaging, and discloses
reference-present-only scores and outputs in unknown boxes. Curve support is
reported alongside the same target grid's own support baseline; a perfect
eight-voxel node grid is not assumed to cover an entire curve within two voxels.
Continuous reference geometry can explain strict node-grid disagreements.
Structural-role matches and duplicate proposal pairs are diagnostics, not
anatomical correspondence or graph connectivity.

**22 focused tests pass** across data/target separation, changed-source/packet
rejection, coincident identities, unknown/development exclusion, schedule
integrity, full-grid denominators, per-neuron macro weighting, duplicates,
curve baselines, state restore and failed-worker handling. These include exact
reproduction of an interrupted stochastic Adam update. No runtime packages were
installed or changed. Torch pytest checks disable unrelated installed plugin
autoload for that invocation to avoid its existing incompatible napari/skimage
plugin; model execution does not import that plugin.

## Next manual GPU command: integrated resource timing

The previous four-box resource result is preserved. This next check goes
through the real loader/exposure order and the entire development grid. It
measures the integration costs that the earlier 8.068-hour projection omitted.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_budget_integration_timing.ps1
```

Device: RTX 3070 Ti Laptop GPU. Each budget starts fresh, makes eight throwaway
updates from the same real training schedule, and runs inference/core scoring
on **all 415 boxes of seven development neurons**, without gradients. Inputs
are the two matched cloud views; targets/loss/initialization/query count and
cutoff stay fixed. The model/Adam/RNG state and next data-order cursor must
reload exactly; evaluation must not alter their values.

The previous forward timings suggest roughly **15 minutes for the pair's
development passes**, before newly measured loading/scoring/startup costs.
The explicit worker limits are 15 minutes at 2048 and 30 minutes at 8192.
These are stop limits, not duration promises. Fresh output goes to
`tmp/node_budget_integration_timing_<timestamp>`; logs, prediction arrays and
core metric diagnostics are retained. No trained checkpoint, full training,
automatic retry or source filtering is involved.

This is the **last planned resource timing before the long fits**, unless a
failure or material implementation change requires another check. It is not
another learnability experiment. Scores after eight updates are execution
diagnostics and cannot settle development quality or the input budget.

## Remaining work before the fit command

The thin/dim image-profile covariate cache is still separate work. Its
training-defined width/contrast cutoffs and unresolved strata must be fixed
before learned development outcomes; the current core timing does not include
those diagnostics. The final bounded fit/resume launcher also remains to be
completed around the tested state/data-order helpers. Neither is claimed ready
by successful data preparation.

After these steps, provide explicit manual commands for two fresh 9200-update
arms. Keep the fixed terminal verdict, no development gradients, no posthoc
threshold/loss changes, no automatic cap extension and no GATv2 handoff. This
study measures node proposals, not complete skeleton/SWC reconstruction or
unseen-domain/product robustness.

## Sources and evidence

- [Portable build, verification, CPU model integration and test evidence](evidence/node_budget_data_integration_20261102.json)
- Build: `tmp/node_budget_data_build_20261102/report.json`
- Verification: `tmp/node_budget_data_verify_20261102/report.json`
- CPU model integration: `tmp/node_budget_data_integration_cpu_20261102/report.json`
- Builders: `scripts/prepare_node_budget_data.py`, `src/pn2/inputs/budget_views.py`,
  `src/pn2/supervision/node_targets.py`
- Loader/metrics/state: `src/pn2/inputs/node_dataset.py`,
  `src/pn2/evaluation/node_cohort.py`, `src/pn2/training/node_fit_state.py`
- [Paired study specification](node_budget_comparison.md)
- [Prior resource-only GPU timing](node_budget_timing_results.md)
