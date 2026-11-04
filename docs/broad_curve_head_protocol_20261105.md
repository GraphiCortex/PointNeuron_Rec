# Authorized broad same-head experiment — 5 November 2026

The user rejected the sixteen-box draft and explicitly authorized the assistant
to run CPU diagnostics first and then launch ONE broad same-head training run.
That authorization supersedes the previous no-GPU/user-launch restriction for
this specific experiment. No repeated approval is needed. Earlier failed gates
remain failed; this is a new, disclosed post-hoc protocol, fixed before training.
The Transformer remains parked and experimentally bypassed only in this arm.

## CPU checks first; select nothing

1. Match the preserved classical raw/pruned graph to exactly the same 151
   reference-present development boxes. Reuse its already-computed per-box
   segment metrics (same clipping, scorer and quadrature) and reaggregate by
   neuron, rather than repeating extraction/filtering/audits. The historical
   53.87%/68.52% pruned F1 used the full grid, including unknown-box prediction
   length. The matched result is 54.3311257940%/69.1368664294% at 2/6 voxels.
2. On the hash-bound saved update-2000 CPU predictions, with the same image-only
   k=6 Sato-cost forest and 2/6-voxel segment scorer, measure:
   - all 2048 predicted votes, BEFORE cutoff and suppression;
   - learned-support-ranked source points, radius-1 source consolidation, first
     512, NO learned offsets and NO score cutoff;
   - the EXACT original Sato-top-512 source identities, moved to their learned
     vote coordinates, NO second suppression and NO learned-score cutoff.

The last contrast changes only coordinates of the Sato-selected set. It does
not change which sources were selected. Duplicate moved coordinates are not
averaged. Counts are disclosed. These three learned ablations are NOT no-learning
comparators and cannot select a revised recipe. Preserve all graph/SWC packets
and per-graph durable score records. No new model forward or GPU is needed.

## Broad training cohort, initialization and schedule

All 460 reference-present/gradient-allowed TRAINING boxes from 33 neurons,
unchanged 2048-point cached inputs. Exclude development and unknown training
boxes. Reuse the frozen dataset's first 20 training-only epoch schedules, every
box exactly once per epoch: 9200 terminal updates. No sixteen-box ladder.

Fresh original seed-0 model initialization, SAME PointNeXt/GN8 encoder and dense
support/offset head, labels, ambiguity masks, optimizer, learning rate, losses,
offset bound, gradient clip, .25 cutoff, radius-1 consolidation and k=6 forest.
Do not fine-tune the four-box checkpoint. Reuse the sixteen existing training
label packets and generate only the missing 444 with the identical full-curve
projection code. No image filtering, sampler change, radius task or topology
loss. CPU input caching is allowed; stage one batch to GPU at a time.

Use the RTX 3070 Ti Laptop GPU and retained Torch runtime. Measure throughput on
throwaway updates with the actual streamed batches; then reset RNG/model/Adam
to the original initialization before the real run. Record peak VRAM and device.
Keep the 9200-update science budget fixed. Set an execution wall budget from
measured training throughput plus a conservative terminal-scoring allowance and
50% margin, bounded by four hours. Stop before training if that projection does
not fit. Report measured estimate and exact output paths before launch.

Intermediate model/Adam/RNG/cursor checkpoints are recovery evidence only.
Terminal checkpoint ONLY for evaluation/selection; no intermediate development
scoring, best-checkpoint selection, early stopping on accuracy or extension.

## Terminal evaluation and frozen decision

Use the original .25/radius-1 proposals and fixed scaffold. Report final training
metrics/loss but NEVER gate them. Evaluate all 151 reference-present boxes/seven
development neurons; disclose the other 264 development boxes as unknown and
excluded. This is repeatedly inspected development, not an untouched final test.

Fixed no-learning pool: raw-input forest, Sato-top-512 forest, preserved classical
raw and pruned graphs. For EACH tolerance, freeze the pool member with strongest
macro F1 as its comparator before training (lexicographic policy tie break).
Compare the learned model against that SAME comparator per neuron, not a
different post-hoc choice per neuron. Retain all paired results.

PASS iff, at BOTH 2 and 6 voxels:

- development macro F1 >= comparator macro F1 + .05;
- strictly better F1 than that comparator on at least 5 of the same 7 neurons.

The matched pruned classical graph currently wins both tolerances, giving
full-precision absolute F1 floors .5933112579402621 and .7413686642936104.
No 45% floor, training-fit floor, oracle ratio, new point-precision gate or
additional numeric accuracy gate. Report geometry, candidate opportunity,
counts/structural strata and shortcut/unknown categories candidly; these do not
certify topology. A valid per-box SWC is not whole-image reconstruction.

If it passes, proceed to whole-image assembly and the intended graph stage,
with concrete implementation details discussed with the user. It does not
promote the dense bypass or forest as the product pipeline. If it fails, STOP
tuning this head. Learned refinement of strong classical candidates is a future
framing requiring the professor's agreement because it changes the Transformer's
proposal role. Do not implement that alternative or contact the professor here.

## Execution and preservation

Fresh ignored PN2/tmp folders; hash-bound readable source/config/manifests.
CPU diagnostics/label prep precede GPU timing/training. Sleep-aware owned-tree
guards, durable progress/score records, cancellation receipts. No automatic
retry, budget extension, parameter change or second training run. Stop on
bindings, invalid data/outputs, nonfinite gradients, resource or wall failure.
Budget failure is incomplete execution, not a scientific accuracy failure.
