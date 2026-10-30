# Paired sampler timing: completed and ready for the bounded fit

30 October 2026. The user launched
`tmp/sampler_node_timing_20261030_215037/`. **Both timing workers completed all
eight updates and passed exact model/Adam/RNG serialization reload.** The saved
reports pass the current recipe/code/source validator. Full training remains
unlaunched by the agent; no model, loss, normalization or budget change is needed
to proceed to the [agreed capped fit](sampler_node_fit.md).

## Measured execution

Device: NVIDIA GeForce RTX 3070 Ti Laptop GPU. Torch 2.12.1+cu130. Fresh
1,688,518-parameter PointNeXt/Transformer, float32, batch one, 2048 input slots,
512 queries, deterministic math attention. Four fixed training boxes per method;
two updates per box. No historical trained checkpoint was loaded.

| Preparation | Completed updates | Exact model/Adam/RNG reload | Mean update time, second four-box cycle | Peak allocated / reserved GPU memory | Estimated 2000-update time |
|---|---:|---|---:|---:|---:|
| Range-aware threshold/random control | 8 | Passed | 1.025 s | 0.233 / 0.262 GiB | 27.3-41.0 min |
| PN2 Sato | 8 | Passed | 1.057 s | 0.233 / 0.262 GiB | 28.2-42.3 min |

The paired update-time estimate is **55.5-83.3 minutes**, plus evaluation and
checkpoint overhead. It uses a 0.8-1.2 times heuristic around the measured second
cycle, not a statistical confidence interval or a guaranteed upper bound.
The central extrapolation is about 69 minutes of updates. Laptop/system
variability can change sustained throughput. The configured process cap is
3600 seconds per sampler; reaching it preserves saved states and stops the run.

The roughly 80-second timing command includes fixed startup, initial/final
evaluations and reload overhead. Those fixed costs should not be multiplied by
2000/8 as though they occurred on every update. Peak timing memory includes the
reload probe and applies only to this 2048-point encoder/proposal experiment,
not dense inputs, GAT or complete reconstruction. No trained checkpoint was
retained by timing.

## What the early predictions show

At update eight both preparations have zero accepted nodes in every box. Assigned
positive-query scores under ordinary inference lie approximately between 0.273
and 0.288, below the fixed 0.5 cutoff. All-query node recall within two voxels is
also zero, so position learning is not established even without filtering.
Current-crop normalization diagnostics also accept zero nodes.

This is not a pass for confidence, geometry or normalization. Equally, two
exposures per box do not establish that the bounded fit cannot learn. Timing's
gate is completed finite computation and exact state reload; its
`learnability_gate=None` is deliberate. The per-box `gate=False` logs are not
timing execution failures.

Updates cycle across different boxes, so adjacent total losses need not decrease.
Compare the same box's first and second exposure instead. Fruitfly tile 27, for
example, changes from about 116.2 to 113.6 in each method; frog tile 2 changes from
about 192 to 187. These early changes show responses to optimization, not
recovered branches or generalization. The very similar early losses provide
no defensible basis for ranking the two samplers.

## Saved-evidence verification

- The two arms start from the exact same model state.
- Both eight-row update journals have steps 1-8, with finite recorded losses,
  components, gradient norms and durations.
- All four raw prediction packet hashes match their reports and all saved arrays
  are finite.
- The post-update-eight model hash equals the evaluation model hash, consistent
  with the evaluation/normalization diagnostic leaving the scored state intact.
- Both workers passed complete model/optimizer/RNG serialization reload on CUDA.
- There are zero retained `.pt` files in the timing directory.

The visible [evidence summary](evidence/sampler_node_timing_20261030.json)
contains source/recipe identities, exact prefix hashes, memory/time measurements,
paired same-box losses, score ranges and raw file hashes. Original run files are
unchanged. No new tests or training runs were needed for this read-only review.

## Next manual command

The reviewed timing qualifies the existing training action. Use the same recipe
and this explicit fresh output directory:

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_sampler_node_fit.ps1 -Action train `
  -TimingRun ".\tmp\sampler_node_timing_20261030_215037" `
  -Output ".\tmp\sampler_node_train_20261030"
```

This starts two independent fresh fits sequentially, each capped at 2000 total
updates (500 per box). The first eight updates are part of the cap and must
exactly replay the corresponding passed timing's model/Adam/RNG prefix. This is
not a continuation from retained timing weights; timing retained none.

The threshold control runs first, then Sato. A completed first fit with a failed
learnability gate still permits the second planned fit. An execution failure or
timeout stops the control and marks the remaining method unattempted. There is
no automatic retry, extension, threshold change or normalization substitution.

Checkpoint states are saved before evaluation at 0/8/500/1000/2000; they include
model, optimizer, RNG, recipe and source identity. Each evaluation reports ordinary
inference and a separately labeled current-crop diagnostic. Only ordinary
inference supplies the terminal gate: node precision and recall >=95% at two
voxels in every box, cutoff 0.5. Keep the final paired report even if that gate
fails. Intermediate results explain learning; they do not select the most
favorable checkpoint or establish a generally better sampler.

The result will answer a bounded training-set learnability question. Broader
training and neuron-disjoint development evaluation remain necessary before a
Phase 1 policy decision; complete reconstruction remains untested here.
