# Two-budget proposal study and complete-model timing

Latest status: the user subsequently completed the [two-budget GPU timing](node_budget_timing_results.md).
Both arms pass, with an 8.068-hour paired update/forward projection and a
provisional 10–14-hour combined-run allowance. The original preparation record
below is retained; its 12–24-hour estimate is superseded. Broader input/target/
loader/metric implementation and integrated timing remain subsequent work.

2 November 2026. The user approved preparing a matched input-budget comparison
after the full image cache completed. This document specifies the comparison
and records the implemented, CPU-verified resource-timing step. **The broader
paired training runner is not yet implemented; no GPU timing or fit has run.**

## Question and controlled change

Does a larger image-derived cloud improve reference agreement of the same
PointNeXt/geometric-query Transformer, particularly in weakly supported regions,
enough to justify its resource cost? Compare nominal budgets **2048 and 8192**.
This does not replace any intended PN2 stage or establish final SWC quality.

Retain the 33 training/seven repeatedly exposed development identities, complete
image-defined clipped 128-voxel grid, source normalization, Sato scales 1/2/4,
0.1 uniform probability mass, seed-zero PPS selection and source-ID hash order.
Larger clouds must retain every smaller-cloud source location at the same seed.
All full guides remain immutable; new cloud views get separate identified
packets. There is no Sato rerun, reference-guided selection, extra channel,
duplicate padding, jitter, new confidence cutoff or radius supervision.

The distinct-point count is proposed as `min(budget, actual_box_voxels)`, an
image-only rule. One development box has 7650 voxels and must retain those real
locations rather than manufacture 8192 observations. The CPU check below tests
that count with synthetic input only; it neither trains on development data nor
implements the broader variable-count cache/loader yet.

Both models retain GN8, identical fresh parameter initialization, 512 geometric
queries and full input-feature memory. A larger input changes the number of
encoder hierarchy locations and attention memory entries, not only observed
voxel coverage. This is a comparison of practical input-budget settings in
this model, not an isolated test of image information with identical internal
activation counts.

## Paired training/evaluation specification

Use the existing [broader protocol](node_generalization_protocol.md) as the
starting specification, with **two independent fresh arms**. Each gets the same
460 supervised training boxes, deterministic exposure order, 20 epochs/9200
updates, node objective, AdamW LR 0.0001, clipping and fixed final checkpoint.
The proposed total is 18,400 updates. Do not transfer the fitted four-box model
or either arm's optimizer/weights into the other. Equal exposures are not equal
GPU time. Reference-empty training boxes remain unknown and receive no
negative-only loss; every image box remains in inference/output accounting.

Targets are identical between arms. Retain eight-voxel chain targets,
endpoints/junctions/crop-boundary identities, coincident/fragmented reference
encodings and explicit query overflow failure. No target thinning or merging
is introduced to make a budget pass.

Primary statistics are paired per-neuron full-grid reference-agreement node
precision, recall and F1 at two native voxels, with macro summaries across the
seven development neurons. Keep cutoff 0.5 and six-voxel diagnostics separate.
Show training agreement, per-neuron/source-group failures, curve support,
endpoint/junction support, duplicate proposals and outputs in unknown boxes.
Width/contrast strata follow the training-defined image-profile conventions
in the broader protocol; unresolved profiles and insufficient strata remain
explicit. Do not interpret SWC radii as calibrated branch caliber.

The earlier proposed pilot milestones (90% training precision/recall, 80%
development precision/recall, minimum development F1 60%) remain screening
targets, not biological certification. Budget selection should consider paired
metric differences alongside measured time and memory. One seed per arm is a
bounded pilot; small differences do not establish repeatability. No numerical
budget-win threshold or universal winner is claimed by this preparation step.
Finalize the fit recipe, scoring implementation and resource allowances before
training; do not tune them using the learned development outcomes.

Stop at the declared cap and review either outcome. Nonfinite values,
coordinate/identity errors, overflow, timeout or missing state stop explicitly.
There is no automatic extension, threshold/loss search or GATv2 handoff.

## Compute estimate and what is still unknown

The historical complete-model timing at 2048 observed 1.5608 seconds/update on
the RTX 3070 Ti Laptop GPU. That implies 3.99 update hours for one 9200-update
arm. The earlier 4–6 hour planning range includes an allowance for evaluation
and output, but has not been measured on the broader loader.

A **12–24 GPU-hour provisional allowance for the two arms together** is a
planning guess, not a measured duration or hard bound. The 8192 complete-model
timing is unknown; multiplying by four is not a reliable model of all operators.
Broader loading, checkpoint writes and CPU matching/curve/profile metrics are
also unmeasured. CPU input/target preparation and interpretation are additional
work; the expensive full-image Sato preparation is already complete.

## Implemented resource diagnostic

`scripts/node_budget_timing.py` and `configs/node_budget_timing.json` reuse the
unchanged geometric model factory. There is no training/resume action.

Preparation read original fruitfly 32/frog 22 images and retained guides, with
four existing TRAINING boxes in fixed order: 32/27, 32/20, 22/2, 22/5. The same
frozen target packets contain 18/85/7/68 nodes. It generated eight small packets
under `tmp/node_budget_timing_inputs_20261102`: exact 2048 baseline replay and
8192 supersets with the same target coordinates. No SWC was opened, no guide
was filtered and no model ran in this preparation path.

The subsequent Torch CPU check passed complete-model train/eval consistency,
512-output shapes and finite node-loss gradients at both budgets. A separate
synthetic 7650-point case with 419 synthetic targets passed forward/backward,
without an optimizer update. One synthetic 256-point throwaway update verified
exact model/Adam/RNG in-memory serialization and output reload. No real-data
optimizer update or GPU work was used for CPU verification. Five focused
controller/count/gate tests pass.

Manual GPU timing uses one fresh process per arm, eight throwaway updates per
budget (two cycles over the four boxes), inference timing, train/eval consistency
and exact model/Adam/RNG reload. The two arms must start from the same parameter
state. No trained checkpoint is retained. Each process has a 600-second cap;
failure stops the controller without retry or launching the next arm.

This is a short resource test, not another learnability fit. Its estimates use
the second four-update cycle for 9200-update arithmetic, and measured forwards
for an explicit 3624-box forward projection (four 415-box development passes
plus one 1964-box terminal training pass). They exclude data loading, checkpoint
IO and CPU metrics. Four positive boxes cannot establish a cohort runtime bound
or cover every broad-run failure mode. The old 64k encoder-only diagnostic is
not used as a whole-model capacity claim.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_node_budget_timing.ps1 -Action timing
```

Output: `tmp/node_budget_timing_<timestamp>`. The launcher uses the retained
Torch runtime, unchanged. A passed result informs the GPU allowance; it does
not automatically construct the remaining denser clouds, train either arm or
qualify prediction quality. Target/cache-loader/evaluation implementation and
broader data/state checks remain necessary before a fit launcher.

Evidence:

- [Portable timing preparation and CPU verification](evidence/node_budget_timing_preparation_20261102.json)
- Inputs: `tmp/node_budget_timing_inputs_20261102/manifest.json`
- CPU check: `tmp/node_budget_timing_cpu_20261102/report.json`
- [Complete image-cache review](node_cloud_cache_completion.md)
- [Original successful four-box fit](geometric_query_fit_results.md)
