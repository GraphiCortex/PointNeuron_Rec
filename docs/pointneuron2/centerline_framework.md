# Intended PN 2.0 architecture and the Phase 1 field experiment

12 October 2026. Architecture clarification agreed with the user: keep
**PointNeXt -> Transformer skeleton proposals -> GATv2 connectivity** in the
intended system. Investigate the image-native field as a possible improvement
to Phase 1 point-cloud preparation. The earlier version of this explanation
described a field-to-path-to-graph alternative too broadly as the new framework;
that alternative has not been adopted as the PN 2.0 architecture.

13 October update: field, attention and fixed-quota development comparisons
are complete. No new sampler is promoted. The selected
[native PointNeXt/Transformer prototype](pointnext_query_ready.md) is implemented
and checked on synthetic CPU data. The real bounded fit subsequently completed
but failed ordinary-inference learnability. The
[saved-weight audit](pointnext_query_results.md) locates a large normalization
discrepancy and unresolved confidence; no new normalization policy is adopted.

## Intended pipeline and actual implementation

```text
3D image
    -> Phase 1: informative image point cloud
         possible addition: learned centerline-field guidance
    -> PointNeXt point-cloud encoder
    -> Transformer decoder: skeleton positions, confidence and radius
    -> GATv2: predicted connectivity between proposed nodes
    -> final SWC
```

| Stage | Intended role | Current implementation status |
| --- | --- | --- |
| Phase 1 | Preserve neurite evidence and useful coverage in the input cloud | Isolated field-guided selector/comparison implemented and checked; production/PointNeXt integration pending |
| PointNeXt | Encode local and broader point geometry | Native S prototype implemented; real bounded fit exposes train/eval normalization discrepancy |
| Transformer decoder | Predict skeleton-node positions, confidence and radius | PointNeXt pairing learns subvoxel placement with current-crop statistics on two training crops; confidence and ordinary inference fail |
| GATv2 | Learn candidate-edge probabilities using node features and geometry | Not implemented; previous reconstructions used inherited graph machinery |

The [completed query pilot](query_pilot_results.md) failed confidence selection
with **DGCNN**, not PointNeXt. It does not establish that the complete proposed
PointNeXt/Transformer/GATv2 system would fail. None of the current field runs
trains or evaluates those three modules together.

## What the isolated field test measures

The compact 3D U-Net sees a full-resolution 97-cubed image crop and predicts
one score per voxel for membership in a narrow centerline tube.

During training, an existing reference SWC provides the supervision. Continuous
reference segments are surrounded by a **two-voxel-radius tube**; the model is
trained to reproduce that tube. This radius is a target construction, not a
measurement of the neurite's biological radius or a full neurite segmentation.
Voxel distances currently use native image coordinates, not physical units.

The loss encourages reference-tube scores, penalizes the hardest known
background errors and rewards overlap. Padding and bright distant regions
marked uncertain are excluded from direct supervision. This weak-label rule
helps avoid labeling every untraced bright structure as a confirmed negative,
but it cannot guarantee that all remaining negative annotations are correct.

**At inference the input is the image alone.** No reference SWC or manual mask
would be supplied to the model. The current input experiment uses score 0.5 to
admit native image locations to the candidate pool. The earlier CPU center-path
extraction and reference-derived oracle were diagnostics; neither replaces the
Transformer proposal task or counts as final learned reconstruction performance.

The immediate question was whether this model could fit the same eight selected
training crops. The 400-update hard-negative fit failed, but the user-completed
[1,600-update replay passes the field gate](centerline_duration_results.md)
with mean tube Dice 0.9173. Its 400-update checkpoint differs from the earlier
run, so the cleanest duration comparison is within this replay. This is a field
learnability result, not acceptance of Phase 1 or a final PN 2.0 model. Better
sampled inputs have since been demonstrated on those fitted crops; independent
generalization and downstream reconstructions have not been demonstrated.

## How the field could serve Phase 1

The user approved the [Phase 1 interface and comparison design](phase1_field_guided_sampling_design.md).
Its isolated selector/comparison has now [completed the eight-crop check](field_sampling_results.md).
Candidate expansion helps reference coverage in these fitted examples; expanded
random is the strongest coverage baseline, and the 50/50 allocation has some
branch losses. This is not production/PointNeXt integration or a promoted recipe;
the [broader field train/development-validation pilot](field_pilot_results.md) is
complete: useful development candidate expansion, with missing branches and
off-reference support. The [fixed-budget development comparison](field_pilot_sampling_results.md)
is complete: expansion improves mean coverage but dilutes some useful image
pools and introduces branch losses. No production sampler is promoted.

The controlled comparison is implemented. The field could guide the selection of informative **image points**,
while retaining intensities, local image evidence and spatial coverage. An
extracted medial path could help as a guide; it would not automatically become
the final skeleton or replace the Transformer proposal task.

Important interface decisions remain open:

- How to include dim but image-supported candidates without assuming every
  thresholded foreground voxel or every high field score belongs to the neuron.
- How to balance field scores with coverage so that bright trunks do not consume
  the entire input budget and erase thin branches.
- Which point attributes to supply to PointNeXt, and how to retain context around
  the centerline rather than reducing the input to bare path vertices.
- How to compare field-guided and existing input clouds with matched budgets
  and downstream settings, including per-branch coverage and false support.

These are design questions, not implemented behavior or additional training
changes hidden inside the duration command. We will review them together before
changing the sampling interface, input budget or downstream strategy.

## What the skeleton extraction checks do not establish

The existing fixed-threshold Kimimaro checks turn fields into center paths to
diagnose geometric errors. The ideal-reference-tube oracle uses ground truth
and is not learned performance. Neither check supplies a final PN 2.0 graph or
substitutes for GATv2 training.

Hard thresholding and this CPU skeleton extraction are outside the current
training graph. The isolated experiment does not implement the proposal's joint
end-to-end training. How to train or freeze a Phase 1 field guide alongside later
modules must be specified before claiming such integration. Nearby branches may
merge in a binary tube, so accurate center placement alone cannot establish
correct neuronal connectivity.

Architecture replacements or stage bypasses must be discussed and explicitly
agreed before implementation. A successful component experiment does not by
itself authorize adopting a different complete system.
