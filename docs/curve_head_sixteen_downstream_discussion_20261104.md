# Sixteen-box downstream transfer screen — discussion only

**Pending agreement; not an active protocol or run authorization.** This is a
proposed new experiment after the original dense-head point gate and supported
consolidation failed. Those verdicts remain failed. Its downstream criteria are
post-hoc relative to that evidence and would be frozen prospectively before
any new training. No architecture change, new trainer or launch command is made.
The user's no-GPU instruction remains in effect until a specific run is agreed.

## Question and unchanged mechanism

Does broader training of the existing dense PointNeXt/GN8 support/offset head
improve per-box reconstruction on untrained development neurons beyond fixed
image-only controls? This tests budgeted training diversity, not attention,
asymptotic capacity, individual loss attribution or product readiness.

Use the already frozen training selection: 16 boxes, 14 neurons, 11 source
groups, aliases/tiles 32/27,32/20,22/2,22/5,23/71,66/22,75/24,0/37,9/32,15/10,
25/54,39/17,53/41,85/7,92/25,96/24. This selection predates the new transfer
scores and uses training metadata only. It is not selected from development
outcomes. Keep 2048 input points, original support/offset labels and ambiguity
masks, encoder/head, optimizer/loss constants, .25 score cutoff, radius-1
consolidation and fixed k=6 Sato-cost forest unchanged. No radius-2/min3 filter.
Transformer remains an intended product stage, bypassed only in this bounded
experimental arm; GATv2 remains intended and is not replaced by the scaffold.

Fresh initialization from the original seed-0 initialization procedure; do not
fine-tune the two-neuron update-2000 checkpoint. Maximum 4000 updates, uniform
250 exposures per box. Terminal checkpoint only for the verdict. No checkpoint,
cutoff, k or point-budget selection from development, no extension on near miss.
This differs in training cohort and update budget from the four-box fit; it is
not an equal-exposure Transformer comparison or proof of a single causal factor.

## Evaluation and proposed numeric screens

Evaluate all 16 training boxes and the SAME 151 reference-present development
boxes/seven neurons, with their identities frozen before launch. Report the 264
unknown boxes as excluded; no relabeling. These neurons are untrained by this
model but repeatedly inspected development data, not an independent final test.
No reserved data access. Reuse the exact raw-input and Sato-top-512 graph
artifacts/scores and current four-box checkpoint result as fixed comparators.
Counts differ and are disclosed. No claim of equal node-count attribution.

Use the same segment scorer, .5 quadrature, 2/6 voxels, within-neuron length
pooling then macro P/R/F1. Proposed necessary screens for continuing this pilot:

1. Terminal training macro F1 at 2 >=75% and recall at 2 >=80%. This checks
   whether the broader set was fitted within the budget; no exact-node gate is
   substituted for reconstruction. Still report original point metrics.
2. Development macro F1 at 2 >=45%, recall at 2 >=65%, F1 at 6 >=70%, recall
   at 6 >=80%. At EACH tolerance require F1 gain >=5 percentage points over
   the strongest no-learning comparator on the identical domain. Current Sato
   baselines are 38.49%/63.13%; the absolute floors remain necessary.
3. Every development neuron: F1 at 2 >=15% AND no more than five percentage
   points below that neuron's Sato F1 at 2. This prevents an aggregate gain from
   hiding a severe case regression. Approximate per-alias lower bounds from
   current evidence: 19:20.61%,21:51.42%,35:20.86%,44:24.83%,79:45.65%,94:15%,
   135:63.74%; implementation must use the full-precision baseline, not rounding.

These are proposed engineering screens for a small pilot, not biological
acceptance standards or calibrated product targets. The four-box model does
not meet them. Numeric passing would justify further discussion only, not
adoption, automatic 64-box training or promoting the bypass/forest pipeline.

Always report per-box/per-neuron candidate-union R2 for k=4/6/10, endpoints,
junction/incident probes, thin/dim support, uncovered gaps, counts, extra length
and full shortcut categories. k diagnostics select nothing. The earlier
95%/90% candidate-union suggestion remains an unadopted discussion target,
not a gate silently relaxed here. Missing candidate geometry remains a ceiling
on fixed-edge decoding and must be interpreted even if summary F1 improves.

No exhaustive wrong-join ground truth has been established. Report the same
reference-path shortcut flags, flagged lengths, off-reference endpoints,
ambiguity and component/boundary unknowns for all policies. Do not gate safety
on the tiny conditional flag fraction or reward an unscorable denominator.
Review these records before any subsequent experiment; a geometric screen
cannot qualify biological topology.

## Resources, stopping and decision

Proposed device if agreed: user's RTX 3070 Ti Laptop GPU for training, retained
Torch runtime. CPU-only terminal graph/geometry/shortcut scoring with durable
records; no repeated control inference/extraction. A new fixed manifest/source
binding and visible PowerShell launcher would be prepared only after agreement.
Keep artifacts under a new ignored PN2/tmp root, preserving all prior evidence.

Maximum four hours including training, terminal inference/scoring and I/O,
sleep-aware owned-process guard, checkpoint/Adam/RNG/cursor receipts, progress
and durable score records. Rough planning range 40–70 minutes: earlier four-box
updates were ~.4s and current seven-neuron scoring ~9.5min, but a sixteen-box
run and combined end-to-end overhead have not been timed. This is an estimate,
not an execution promise. A source/timing preflight must stop if the resource
projection plus margin cannot fit the cap. No launch occurs in this document.

Stop on binding/label/parity failure, NaNs, invalid outputs, resource failure,
wall cap, or any proposed terminal numerical screen failure. No automatic
extension, threshold change, second consolidation or further architecture run.
If training fit fails, distinguish budgeted learnability from generalization.
If fitting succeeds but transfer loses to controls, stop this dense recipe
rather than launch the Transformer hypothesis automatically. Even a numerical
pass requires user review before any 64-box or GATv2 work.
