# Branch evidence and learned retention: prospective protocol

23 October 2026. Discussion specification, not a completed experiment or frozen
training recipe. Complements the [audited input/encoder contract](pointnext_input_contract.md).
No reference-guided inference sampler is proposed.

29 October progress: the bounded [four-box real-cloud pass](real_branch_evidence_results.md)
is complete, with contrast profiles, input evidence, actual hierarchy access and
an ordering control. It does not complete the broader trained comparison or
establish thin/dim adequacy; only six sampled windows occupy that joint stratum.
The [implementation plan](real_branch_evidence_plan.md) records the fixed scope.

The subsequent user-requested [PN1 component comparison](pn1_input_comparison_results.md)
is also complete. It includes decoder/range and spatial controls. The next
learned comparison should retain a strong thresholded reference arm: current
PN2 Sato has weak-region benefits and matched-budget counterexamples, rather
than an established universal selection advantage.

## Questions and attribution

At a fixed distinct-point budget, does the cloud retain useful evidence for
thin/dim processes? Does the encoder's sampled hierarchy include that evidence,
and do its trained descriptors support downstream node predictions?

Distinguish three observations: selected raw evidence, geometric access through
the hierarchy, and learned task performance. Untrained descriptors cannot answer
the third question. A good node fit also does not establish learned connectivity
or generalization. Reference SWCs enter measurement and supervision only.

## Replace the single support headline with an evidence profile

Keep historical distance-to-reference results unchanged and available as geometry
diagnostics. They count selected zero-valued voxels as support and can miss useful
bright evidence outside a narrow skeleton tolerance. They are insufficient as
the primary input-quality measure.

First qualify the proposed measurement on synthetic image tubes with known
centerline, width, contrast, background/noise and close-branch spacing. Include
thin/dim, thin/bright, thick/dim and thick/bright cases, subvoxel offsets and gaps.
Fix generation parameters and metric definitions before comparing selectors.
This validates the measurement, not biological generalization.

Then use the six already-inspected training sources and existing frozen clouds;
do not inspect untouched test images. Record actual box IDs, boundary truncation,
reference availability and unscorable intervals. Start with image evidence:

* At regularly spaced reference arc-length locations, inspect native image
  cross-sections and multi-scale tubes (candidate radii 1, 2, 4, 8 voxels).
  Avoid making an arbitrary one-radius foreground mask the biological truth.
* Estimate a local background from an outer shell, flagging shell contamination,
  nearby known branches and crop truncation. Keep signed contrast and background
  variability; a shell is a measurement assumption, not certified background.
* For each tube, report selected distinct-voxel counts, raw intensity/contrast
  distributions and retained positive contrast mass relative to all source
  voxels in that same tube. Define positive mass as sum(max(I-B,0)), with the same
  background estimate B for full and selected voxels. Zero full mass is undefined,
  not perfect retention. Account for shared voxels once within each tube union.
* Report both absolute retained evidence and its fraction of available evidence,
  plus support along arc length and the worst supported continuous intervals.
  A bright isolated point cannot substitute for continuous branch support.
  Compare budgets explicitly: retaining a larger fraction by using more points
  is not greater efficiency at a matched budget.

These are measurements of image evidence near the reference, not biological
foreground precision. Positive noise and unrelated structures can contribute
contrast mass. Synthetic controls, visual cross-section checks and flags make
those limitations visible.

True caliber is known in synthetic examples. Real SWC radius provenance is not
verified, and four audited references report radius 1 everywhere. Keep reported
SWC radius as metadata. For real images, estimate apparent width from background-
subtracted cross-sectional profiles, report uncertainty and unresolved widths,
and call it apparent image width. PSF and unknown calibration prevent equating
it with anatomical diameter. Do not label a one-voxel SWC radius as a true thin
branch. Freeze width/contrast strata on the training calibration subset before
selector scoring; report unresolved cases rather than silently dropping them.
Exact shell, width estimator and stratum boundaries remain to be reviewed.

## Audit geometric access through the encoder

Instrument selected real clouds at each abstraction stage, using the same fixed
normalization and sampling-independent point order across methods. This requires
no optimizer. Record actual FPS indices and actual selected ball members, not
only the theoretical radius:

* Distinct members per neighborhood, fractions filled by repeats, fractions
  truncated at K=32, and raw intensity/contrast evidence among the members.
* How much evidence near each width/contrast stratum remains among FPS centroids,
  is reached by local grouping, and is available through fine-level skip paths.
  FPS survival alone is not the whole encoder: grouped points and restoration
  skips can also influence descriptors.
* Along-reference gaps between informative selected points and between informative
  centroids, and whether actual neighborhoods span those gaps. A gap smaller than
  a ball radius is necessary in some configurations but not sufficient for the
  first-K operator to include both sides.
* Same-branch access versus mixing from nearby distinct reference paths, with
  unknown graph relationships explicitly masked. Larger radii can improve access
  and worsen ambiguity simultaneously. Do not automatically tune the radius up.

Repeat ordering on a small diagnostic subset to expose first-K/FPS sensitivity.
Report graph-label uncertainty for the three disconnected/shared-coordinate
reference encodings already audited. Do not weld them to make this measurement.

## Matched trained comparison, after the execution and measurement checks

"Dense quota" is a point count, not a selection method. Use a factorial comparison:
uniform, existing image-intensity weighting and existing Sato weighting at the
**same count**, followed by the feasible 8k/16k/32k/64k budgets. Existing dense
32,768-count clouds can be reused where their box and ordering contracts match.
The intensity arm is the current simple brightness-weighted control; it should
not be relabeled as local contrast weighting. A new background-subtracted contrast
selector would be a separate specified arm, tested for noise/background sensitivity.

Choose a few explicit training boxes covering the four width/contrast strata and
at least one close-branch situation, using source images and usable reference
geometry. Freeze their IDs and annotation policy before results. Box choice can
use training references; input selection within every box remains image-only.
Use the same boxes across arms, fixed point order convention, paired selection
seeds, model initialization, head, normalization, optimizer, supervised exposures
and update cap. Log time and memory separately; an equal-update comparison is
not an equal-wall-time comparison. Keep other features and radii fixed.

First fit one declared reference-backed node task on these boxes with the intended
PointNeXt-to-Transformer interface. Agree target discretization, treatment of
unknown image structure, loss, matching, confidence policy, success tolerance,
numerical gate and step cap **before implementing/launching the fit**. Do not
inherit the historical 512-query/threshold recipe automatically or use oracle
node selection at evaluation. No new topology loss or GAT training is needed to
answer this initial node-representation question.

Report geometrically matched node recall and reference-agreement precision by
width/contrast stratum, continuous curve coverage and duplicate predictions.
Exact node assignment alone is sensitive to reference discretization. With
incomplete annotation, unmatched predictions are not certified biological false
positives. Keep an explicit unknown category and inspect examples. Calibrate any
global operating threshold on development data later, not on each training box.

After the sanity fit succeeds, compare selectors at one matched feasible budget
before paying for the full budget sweep. Preserve neuron-level separation for
development; boxes from a training neuron are not independent generalization
examples. Expand only after the comparison answers its predeclared question.

## Success, failure and stopping rules

* Capacity success: finite forward/backward, expected shapes and recorded cost
  at a declared budget. Timeout/OOM stops the native sweep; review the backend
  before concluding the architecture is infeasible. No automatic cap extension.
* Measurement success: synthetic controls distinguish lost contrast evidence,
  narrow-tolerance misses and genuine longitudinal gaps; ambiguous real widths
  and background estimates are flagged. Otherwise repair the measurement before
  ranking selectors. Do not select thresholds using the desired method ranking.
* Sanity-fit failure: stop the multi-arm trained sweep. Audit implementation,
  targets, normalization and optimization using saved outputs; do not blame Phase
  1 simply because a joint task fails. A fixed cap is not a proof of impossibility.
* Selection/budget effect: paired improvements in thin/dim node recall accompanied
  by improved retained evidence, without unacceptable duplicate/precision loss,
  support an input bottleneck under that tested recipe. Freeze numerical minimum
  effect and allowed regression with the fit protocol, before training starts.
* No measured effect: leave the sampler unchanged provisionally. This does not
  exonerate Phase 1 if all arms are saturated, unlearnable, underpowered or blocked
  by encoder/head/annotation errors. Inspect those alternatives rather than
  repeatedly extending the same run.
* A training-box win authorizes neither a whole-neuron quality claim nor a
  universal-product claim. Require held-out neuron/context evidence and final
  graph/SWC evaluation in later milestones.

29 October update: the user-launched untrained capacity sweep is
[reviewed](pointnext_capacity_results.md). Before interpreting the 64k training
failure as a capacity ceiling, address the confirmed native index-storage artifact
through an explicitly validated implementation correction. The
[compact correction now passes CPU validation and the user-launched 64k GPU replay](pointnext_compact_replay64.md).
No additional capacity replay is needed for this milestone. The metric and trained fit remain specifications with
unresolved measurement choices, not an automatically launched sequence of fits.
