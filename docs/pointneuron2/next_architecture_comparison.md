# Next PN 2.0 comparison: supervised query proposals

Draft for discussion, 11 October 2026. This is a proposed experiment, not an
implemented decoder, an authorized training run or a promoted configuration.

The training-only sizing check and [concrete target/decoder design](query_decoder_design.md)
are now complete for review: proposed spacing 8 voxels, 512 output slots,
maximum 418 training targets/crop, and an explicit reference-agreement label
policy. Model implementation and GPU runs are subsequent stages.

## Question

Can a query-based proposal head learn to cover the annotated arbor while
rejecting off-reference foreground better than the current pointwise head?
It will still receive a limited point representation. This experiment does
not establish target-neuron identity where annotations are incomplete, and
cannot establish commercial readiness.

## Why this step

The completed diagnostics show upstream proposal deficits, downstream coverage
losses and high-confidence proposals along bright off-reference structures.
Spending the graph budget more evenly increased recall on neuron 79 but lowered
precision and F1. More heuristic allocation changes are therefore not the main
next experiment. We will test a proposed learned component without waiting for
the inherited sampling and graph stages to become perfect.

## Comparison scope

Keep the corrected reader, image threshold policy, cached crops, neuron split,
random point selection, 1,024 candidates / 512 encoder inputs, DGCNN encoder
architecture and downstream graph configuration fixed across arms. This input
budget is a controlled pilot setting, not the final product specification.

Compare the existing local-coordinate pointwise head with a query decoder
predicting local centers, objectness and physical radius. The query decoder
requires a matching objective; this tests a head-and-objective combination,
not attention in isolation. PointNeXt, learned guide sampling and GATv2 are
separate later comparisons.

Archive the current best verified outputs. Train both comparison arms with
matched data order, sampling, augmentation, encoder initialization and declared
training budgets. A new decoder trained from scratch must not be judged solely
against an old checkpoint trained under different conditions. Record wall time,
parameter counts and memory alongside quality.

## Decisions to specify before implementation

1. Define targets from reference SWC geometry rather than image-extracted guides.
   Use an explicit arc-length sampling rule, preserve true endpoints and
   junctions, and handle edges crossing crop boundaries. Merely copying all SWC
   vertices would make target density depend on tracing/export conventions.
2. Choose node spacing and query capacity using training data only. Check target
   count distributions first; do not silently truncate targets to fit a chosen
   query count. State the units and the handling of unavailable voxel spacing.
3. Keep physical radius distinct from a suppression/support radius.
4. Specify the negative-label assumption. Reference-empty crops remain unknown
   under the existing policy. Reference-containing crops are not automatically
   fully annotated either. If unmatched queries receive no-object supervision,
   document its reference-agreement assumption explicitly and its limitations;
   alternatively, specify verified negative regions. Apply a consistent label
   policy across arms. The decoder does not solve incomplete supervision itself.
5. Specify matching costs, loss normalization, crop coordinate transforms and
   checkpoint/inference compatibility before training. Use the same target
   geometry where applicable in both arms and record necessary loss differences.
6. Specify proposal consolidation across overlapping crops. Query uniqueness
   inside a crop does not eliminate duplicates across crops. Keep downstream
   settings fixed for the primary comparison and report losses at consolidation.

## Bounded execution sequence

1. Produce the target specification and a small training-only target-count check.
   Review these choices together before substantive model changes.
2. Implement the decoder, matching loss and inference adapter in the real PN2
   checkout. Verify coordinate translation, variable target counts, empty-label
   handling, finite gradients and checkpoint reload. Fit a few annotated training
   examples to check learnability; this is an execution check, not validation.
3. Prepare one matched GPU pilot command for the user to run visibly on the
   RTX 3070 Ti Laptop GPU. State epochs, settings, output paths and a measured
   runtime estimate after the execution check. Agree on the run before launch.
4. Evaluate proposal coverage and precision across all seven validation neurons.
   Agreement on a whole-neuron reconstruction run is separate. Evaluate both
   arms using the same reference resampling, spatial tolerance and graph settings.
   Keep the eight reserved test neurons untouched.

Sources remain under scripts/pointneuron2, src/pointneuron and configs/pointneuron2;
reports remain under docs/pointneuron2. Generated runs belong under ignored
tmp/pointneuron2/local_runs/query_proposal_<timestamp>.

## Decision criteria

Before the pilot, record a numerical minimum useful gain and allowable regression
limits. Select them from scientific/product requirements, not the decoder's
results. Existing seven-neuron validation data have been examined repeatedly;
they are development evidence, not independent commercial confirmation.

Primary evidence is paired whole-neuron spatial F1 and recall at comparable
precision. Report every neuron, endpoint/branch coverage, off-reference geometry,
runtime and memory. Training loss alone does not qualify a change. Existing
selected-node topology F1 is secondary because its target changes with selection.

If execution checks fail, repair the implementation before interpreting quality.
If the matched pilot fails, permit a follow-up only for a named, testable cause
(such as failure to fit training examples). Define that follow-up's stopping
condition before running it. If the corrected comparison still fails, retain the
control and drop or redesign this component. Do not respond automatically with
more threshold sweeps or simultaneous backbone/connectivity changes.

Passing this comparison supports a research milestone. Broader independent data,
external method comparisons and domain-specific release requirements remain
necessary before a commercial claim.
