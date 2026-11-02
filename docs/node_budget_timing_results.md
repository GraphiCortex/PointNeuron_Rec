# Passed complete-model timing at 2048 and 8192 input points

2 November 2026. The user completed
`tmp/node_budget_timing_20261102_092848` on the RTX 3070 Ti Laptop GPU.
**Both eight-update arms passed.** The source/input/CPU gate is current, both
fresh parameter hashes match, and saved reports/journals are consistent.
No additional GPU work, learned comparison or full training was launched by
the result review. The user-observed command duration was 61 seconds.

## Measured resources

Same GN8 native PointNeXt/geometric-query Transformer, 512 proposals, fresh
initial parameters, AdamW LR 0.0001, FP32 and fixed four TRAINING boxes. Targets
remain 18/85/7/68. The larger cloud is a source-location superset; targets and
initial parameters are unchanged between arms.

| Measurement | 2048 points | 8192 points |
| --- | ---: | ---: |
| Mean update, second four-box cycle | 0.6716 s | 1.7041 s |
| Mean inference, four boxes | 0.5681 s | 1.4156 s |
| Peak CUDA allocated | 239.2 MiB | 630.1 MiB |
| Peak CUDA reserved | 274.0 MiB | 716.0 MiB |
| Projected 9200 updates | 1.716 h | 4.355 h |
| Projected 3624 forwards | 0.572 h | 1.425 h |
| Updates plus forwards | **2.288 h** | **5.780 h** |

The projections sum to **8.068 model-computation hours** for the pair. The
forward count is explicit planning arithmetic: four development passes of 415
boxes plus one terminal training pass of 1964 boxes. It does not assert that
the broader loader/evaluation implementation is already complete.

Four times as many input points cost about **2.54 times as much per update**
in this short measurement. Memory figures are PyTorch allocator counters,
including the timing/reload procedure, not total GPU memory usage or a measured
production hardware requirement. They show that 8192 is feasible in this
specific complete-model setup; they do not settle the biological input budget.

## Saved-evidence verification

The CPU-only review verified current control/model source, recipe and input
manifest identities; both per-arm worker reports equal their parent records.
All 16 journal rows have the expected update indices, finite metrics and exact
agreement with the reported durations. The update/inference means and both
projection formulas were independently recomputed.

Both arms started at parameter-state SHA256
`35d1503b7374d87bbe18f8459cc40e9cc7a2b5a9c439ee69f422602a1ec89e89`.
The worker's train/eval consistency and exact model/Adam/RNG reload checks passed
in each arm. No trained checkpoint was retained. This review checked the saved
reports and verified worker logic/identities; it did not replay CUDA or inspect
nonexistent retained trained weights.

## Revised planning allowance

Replace the earlier unmeasured 12–24 GPU-hour allowance with **10–14 hours as a
provisional combined-run planning allowance**. This adds a heuristic margin to
the measured 8.068-hour update/forward projection; it is not a confidence
interval, measured complete-run duration or guaranteed upper bound.

Remaining unmeasured costs include broader data loading and host/device
transfer, checkpoint writes, CPU matching/curve/width/contrast metrics,
different target counts up to 419, uniform/constant boxes and sustained laptop
power/thermal/load behavior. CPU input/target preparation and implementation
time are additional to this compute allowance. Full-model timing through the
actual broader loader/metrics remains necessary before a fit launcher.

The new 2048 timing is faster than the earlier 1.5608-second update measurement.
We did not isolate why; do not attribute that difference to an algorithmic
optimization or assume it will persist through an extended fit. The current
matched measurement is useful planning evidence, with the limits above.

## Interpretation and next work

This closes the immediate complete-model resource check for the two declared
budgets. It does not show which budget learns better. In particular, higher
early loss in the 8192 arm is not a reconstruction-quality verdict after two
updates per box. Query starting locations/features change with the input; both
arms must receive the specified matched training before learned comparison.

Proceed with the [paired-study specification](node_budget_comparison.md):
finish the broader separately identified cloud views, shared reference-target
cache, variable-count loader and reference-agreement metrics. Check unknown
supervision, the 7650-point clipped box, 419-target case, deterministic exposure
and model/optimizer/RNG/data-order restoration. Then measure the integrated
pipeline and provide manual bounded fit commands for the two 9200-update arms.
No loss, threshold, model stage or update-cap change follows from this timing.

Do not repeat this resource-only timing command. There is no fit command in the
current timing launcher, and no full fit or GATv2 handoff has been started.

## Evidence

- [Portable timing result and independent arithmetic/identity review](evidence/node_budget_timing_results_20261102.json)
- Raw report: `tmp/node_budget_timing_20261102_092848/report.json`
- Per-arm logs/journals: `points2048/` and `points8192/` under that run folder
- [Input preparation, CPU verification and prospective study](node_budget_comparison.md)
