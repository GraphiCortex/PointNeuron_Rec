# PointNeuron 2.0 development

**Prepared retained fit, 20 October:** [capped saved-proposal-Adam joint fit](joint_saved_adam_fit_ready.md).
The new check/train/resume launcher caps at 4,000 new updates, retains the tested
optimizer initialization and requires automatic terminal reconstruction gates.
Mixed-state checkpoints precede evaluation; the first eight GPU updates must
replay the passed timing before continuing. Fifteen tests and a zero-update
real-model CPU reload check pass. No GPU fit has been launched.

**Completed GPU timing, 20 October:** [readiness and automatic-output review](joint_saved_adam_timing_results.md).
Eight saved-Adam joint updates complete, exact mixed-state reload passes and
PyTorch peak allocation is 0.279 GiB. Eight saved outputs/SWCs independently
verify. All automatic gates fail: sparse quality regresses while two other crops
improve arclength metrics. Recommend the declared bounded joint learnability fit;
mixed-state full-trainer integration remains unimplemented and no fit is launched.

**Prepared GPU timing, 19 October:** [saved-proposal-Adam timing command](joint_saved_adam_timing_ready.md).
New check/timing-only launcher preserves proposal Adam history, starts GAT Adam
fresh, records eight-update diagnostics and peak memory, and verifies exact
mixed-state disk reload. Twelve targeted tests pass. CPU zero-update preparation
is separate from the still-unlaunched GPU timing; no full trainer or resume action.

**Completed joint stability check, 19 October:** [eight CPU updates with saved proposal Adam state](joint_saved_adam_stability_results.md).
All eight joint updates complete; crop 0/0 retains 6/6 trusted matches at updates
four/eight, and every crop retains both edge classes at the end. Proposal Adam
history is restored and GAT Adam is fresh. Confidence extras and intermediate
geometry drift remain; no reconstruction-quality or GPU-readiness claim. The
old joint trainer remains unchanged and cannot be treated as this initialization.

**Completed restart control, 19 October:** [fresh versus saved proposal AdamW state](proposal_optimizer_state_results.md).
Identical weights/loss/LR/crops and four CPU updates per arm: saved optimizer
state preserves 6/6 trusted crop-0 proposals; fresh state leaves 0/6. Final dense
trusted counts are 413/418 versus 343/418, with intermediate drift still present.
Confidence extras remain. Three transfer tests pass; no joint policy change,
GPU run, retained training or reconstruction-improvement claim.

**Stopped timing, 19 October:** [joint timing failure and matched CPU control](joint_query_graph_timing_results.md).
Four GPU throwaway updates complete; missing graph-label classes stop update five.
Historical GPU proposal replay is exact. Four-update CPU joint and proposal-only
arms both move crop-0 geometry outside the trust mask, so graph loss is not
necessary for that drift. Preserve the guard and current artifacts; no retained
joint fit. Recommend isolating fresh versus saved proposal optimizer state next.

**Prepared joint prototype, 19 October:** [joint proposal/connectivity contract](joint_query_graph_ready.md).
Existing proposal and edge heads share a loss, warm-started from the completed
balanced proposal and conditional GAT fits with a fresh optimizer. All 512 queries
reach GAT before final automatic selection. The strict gate includes automatic
nodes, decoded topology and 99% arclength precision/recall at two voxels on every
training crop. Separate CPU checks, GPU timing and capped training actions;
no new node head, topology loss or broader-data claim.

**Design recommendation, 19 October:** [node-selection/connectivity responsibility](node_graph_design_decision.md).
The proposal calls for joint proposal/connectivity training; the saved fits have
not tested that integration. Its GAT role is edge scoring, so a node head would
be an explicit extension. Recommend specifying a minimal joint variant with the
existing heads and realistic proposal context before changing stage roles.
Hard selection, gradient/masking rules, forest-aware topology and automatic
reconstruction gates remain decisions to make concrete. No model/configuration
change or new fit is implemented by this review.

**Completed inference integration check, 19 October:** [oracle versus automatic node selection](gat_automatic_selection_results.md).
The same successful GAT checkpoint is evaluated without retraining. Automatic
selection reduces small-crop arclength precision at two voxels to 57.11%, 45.12%
and 67.21%; dense recall becomes 95.79%. All surviving required pairs are present
in candidates; dense losses originate in seven confidence-rejected nodes.
Direct-pair mismatches are distinguished from paths through unmatched nodes.
Eighteen tests, eight exact CPU score replays and eight SWCs are verified.
No pruning, threshold change, node head, new training or production promotion.

**Completed connectivity control, 18 October:** [frozen-proposal GATv2 results](frozen_proposal_gat_results.md).
All four oracle-selected training graphs pass at terminal update 4,000 and every
evaluated snapshot from 500 onward: 434/434 required edges, zero wrong edges,
correct degree/junction incidence and components. All 3,735 negative candidates
are rejected. Five checkpoints, 20 CPU score replays and 20 SWCs are verified.
This establishes conditional training learnability; automatic node selection,
extra/missing-node robustness and generalization remain open. No new fit or
production promotion follows automatically.

**Prepared connectivity control, 18 October:** [frozen-proposal GATv2 protocol](frozen_proposal_gat_ready.md).
The agreed training-only control freezes balanced proposals and uses saved
assignments to select nodes. Only the existing 119,041-parameter GATv2 scorer
trains; this diagnostic bypasses confidence selection and the placement mask,
without changing production behavior. All required pairs are candidates.
Twenty-eight tests and the zero-update CPU check pass, with exact reload and
unchanged frozen tensors. The user subsequently launched GPU timing and the
capped 4,000-update fit; their completed result is reviewed above.

**Completed CPU control, 18 October:** [frozen four-crop confidence results](four_crop_frozen_confidence_results.md).
The 4,000-update confidence-only stage completes with exact six-snapshot replay
and unchanged frozen tensors/sources. All strict gates fail: sparse extras
11/32/17 → 4/28/8; dense assigned acceptance 411/418 → 409/418 with four extras.
Loss decreases 39.59%, without establishing reconstruction improvement. Stop at
the cap; retain the balanced full-model reference and discuss the selection
contract before another fit. No new run or model promotion is prepared.

**Preparation, 18 October:** [frozen four-crop confidence stage](four_crop_frozen_confidence_ready.md).
Only 258 confidence parameters train from the balanced source head, with a fresh
AdamW optimizer, balanced loss, all four crops per update and a 4,000-stage-update
cap. Thirteen tests and an eight-throwaway-update CPU readiness check pass.
Features/labels/geometry remain fixed; the user-launched CPU fit has now completed
and is reviewed above. This is a new training stage, not
continuation of the completed joint run or reconstruction improvement.

**Capacity result, 18 October:** [four-crop confidence oracle](four_crop_confidence_capacity_results.md)
verifies one shared linear separator for all 2,048 fixed balanced-fit labels:
452/452 matched positives, zero unmatched acceptances, zero float32 label errors.
Actual post-hidden-ReLU inputs reproduce saved logits first. Eight known-case
tests pass. This establishes frozen training-feature/head capacity, not learned
selection, generalization or reconstruction. No classifier is saved/deployed,
and the subsequently agreed frozen-head control is reviewed separately above.

**Design audit, 18 October:** [assignment and confidence contract](query_assignment_contract_review.md).
Saved target/query changes and binary confidence-label flips are counted
separately. In the failed dense fit, 267 changed target matches include only
62 positive-query replacements (124 binary flips); 205 queries stay positive
and switch targets. This does not diagnose harmful instability. The next proposed
question was fixed four-crop feature/head separability, using the actual hidden
classifier inputs; the subsequent result is reviewed above. No new loss, matcher
change or fit is prepared.

**Completed failed comparison, 18 October:** [hard-negative results](hard_negative_query_results.md).
The matched 4,000-update fit regresses: sparse extras 11/32/17 → 260/151/116;
dense trusted targets 404 → 349, accepted assigned 411 → 158 and junctions
76 → 60. Execution/source/checkpoint checks pass, but all selection/combined
gates fail. Reject this variant for the current protocol; retain balanced as the
stronger incomplete reference. Stop loss sweeps and return to the assignment/
selection contract discussion. No further fit, cutoff change or GATv2 run is queued.

**Prepared, 18 October:** [hard-negative comparison and CPU checks](hard_negative_query_ready.md).
The agreed 50/25/25 objective is isolated from the completed fits. Twenty-eight
targeted tests pass; the exact-cache/fresh-model CPU check is reviewed in the
linked evidence. User-launched GPU timing passes with zero reload difference;
its estimate is 85–128 minutes, substantially slower than the prior fit for an
unestablished reason. The retained fit subsequently completed in 18.94 minutes
and is reviewed above. Same four crops, seed, architecture, optimizer, original gates
and 4,000-update cap; no development/test evaluation or automatic GATv2 fit.

**Design discussion, 18 October:** [hard-negative objective proposal](hard_negative_objective_proposal.md)
specifies one candidate loss: retain half the negative mass on all unmatched
queries and put half on the highest-scoring subset, K=min(P,N). Saved-score
algebra confirms the intended sparse-crop emphasis and exact dense-crop equality.
The subsequently agreed implementation and preparation are linked above.
The completed GPU fit is reviewed above; no additional fit is prepared.

**Completed comparison, 18 October:** [balanced confidence results](balanced_query_results.md).
The paired 4,000-update fit improves ranking on all four training crops. Smaller
crops accept all assigned targets but retain 11/32/17 extras; the dense crop
improves 384 → 404 trusted targets and 94 → 4 extras, with one junction still
outside tolerance. All combined gates fail. Keep the useful objective result,
stop at the cap and discuss proposal-selection supervision before another fit.
No GATv2 training, cutoff change or production promotion is started.

**Prepared, 18 October:** [matched class-normalization comparison](balanced_query_ready.md).
The exact four-crop cache and fresh initialization match the completed baseline;
18 tests and a zero-update CPU reload check pass. User-launched GPU timing also
passes, with zero reload difference and a 22.6–33.8 minute fit estimate; training
subsequently completed and is reviewed above. A saved-prediction radius-eight/twelve check
recovers smaller-crop candidate omissions without fixing confidence or dense-crop
placement. Radius eight remains the primary gate; radius twelve is diagnostic.

**Completed fit, 17 October:** the [four-crop results](branched_query_results.md)
verify successful 4,000-update execution and zero reload delta, but every
learnability gate fails. Sparse crops place 6/6, 12/12 and 16/16 targets;
confidence accepts 0/6, 8/12 and 6/16 with 0/25/17 extras. The dense crop has
384/418 trusted matches, 74/77 junctions and accepts all 512 queries. Candidate
availability is also below the gate. Stop at the cap; no GATv2 fit or extension.
The subsequently agreed class-normalized confidence comparison and candidate
margin check subsequently completed and are reviewed above.

**Evaluation repair, 17 October:** the first four-crop fit failed at update 500
on two empty JSON probe arrays in dense-crop scoring. The
[repair report](branched_query_evaluation_repair.md) documents the cause and
ten passing tests. Checkpoints now precede evaluation and native stderr is fully
logged. The original attempt is preserved; fresh retry cache/CPU checks pass.
The launcher defaults to `branched_query_restart_20261017`; no retry GPU work
has started. Model/loss/settings/criteria are unchanged.

**GPU timing reviewed, 17 October:** the [four-crop fit](branched_query_ready.md)
has matching source/cache contracts and zero reload delta. The eight throwaway
GPU updates estimate 76–114 minutes for the capped fit; no trained timing
checkpoint remains. The full fit is ready for the user to launch separately.

**Ready, 17 October:** the agreed [four-crop branched proposal fit](branched_query_ready.md)
is prepared with fresh seed-zero weights, fixed clouds and a 4,000-update cap.
CPU data/inference/checkpoint checks pass; the visible GPU timing and training
commands remain user-launched. Phase 1 is fixed, no GATv2 fit starts, and no
real-neuron optimizer update or GPU workload has run in this preparation.

**Latest, 17 October:** the [real upstream connectivity audit](upstream_connectivity_results.md)
rules out freezing the existing two-crop query checkpoint for the declared
branched fit. Trusted matches are 0/16 and 0/418 despite 94.74%/98.72% selected
cloud support at six voxels; both crops supply zero known edge labels. The two
fitted controls reproduce saved predictions exactly. No training was launched.
A capped branched proposal learnability experiment is proposed for discussion;
GATv2-versus-pairwise fitting and the 4,000-update confidence replay stay deferred.

**Latest, 17 October:** [per-crop connectivity implementation](gatv2_connectivity_implementation.md)
is complete: native GATv2, matched pairwise control, masked edge labels and
score-to-forest/SWC decoding. 42 targeted tests and fixed CPU execution checks
pass with zero optimizer updates. No GATv2 fit, learned reconstruction or
automatic whole-neuron crop matching is established. Existing Phase 1 and
proposal-model policies remain unchanged.

**Reviewed design:** [GATv2 connectivity contract](gatv2_connectivity_design.md)
specifies actual Transformer features, reference-independent candidates, masked
edge labels, a separate edge classifier and deterministic forest decoding.
The per-crop components are now implemented as described above; full-volume
matching remains open. No training is prepared or launched by this document.

**Current direction, 16 October:** the [training-only representation oracle](representation_oracle_results.md)
passes four branched/dense/crop-joining/fragmented cases with perfect nodes and
known connections. This is CPU reference feasibility, not learned reconstruction
or GATv2 training. The next discussion is the learned connectivity/interface
contract; Phase 1 and existing model recipes are unchanged. The proposed
4,000-update frozen-confidence replay below remains deferred.

**Latest:** [LR .001 replay result and remaining confidence miss](confidence_lr_results.md).
At 2,000 updates crop 0 improves 3/6 + two extras → 5/6 + no extras, with
96.10% accepted curve coverage; crop 1 stays 12/12 + no extras. Query 408
remains below .5 but is still improving. A duration-only 4,000-update replay
with overlapping-prefix checks is proposed; no additional fit has started.

**Ready:** [learning-rate-only confidence replay](confidence_lr_ready.md).
The user agreed to LR .001 versus the saved .0001 baseline. Source checks and
four tests passed; starting weights, fixed features, labels,
geometry, loss, 2,000 updates and backend are paired. Separate visible GPU
timing and training commands have now completed and are reviewed above.

**Latest:** [fixed-placement confidence result and linear-capacity audit](query_confidence_results.md).
The user completed the bounded fit: crop 1 accepts all 12 matches with no extras;
crop 0 remains at 3/6 matches and two extras. A CPU oracle verifies a shared
linear separator for all 1,024 frozen labels, with zero float32 label errors.
The subsequently agreed optimization-only learning-rate replay is prepared
above; no new fit or architecture change has started.

**Ready:** [fixed-feature, fixed-placement confidence diagnostic](query_confidence_ready.md).
The user agreed; the head/runner and real CPU feature cache are prepared.
Only 258 confidence parameters train against fixed 6/12 assignments, with
both crops contributing each update. GPU timing and the bounded 2,000-update
fit subsequently completed and are reviewed above. PointNeXt, Transformer
features, geometry, radii and source checkpoints remain frozen.

**Latest:** [explicit per-crop normalization control and query-score audit](pointnext_cropnorm_results.md).
Saved-weight CPU comparison passes train/eval/reload/crop-order checks, with no
optimizer updates. Subvoxel placement survives ordinary eval; confidence still
fails. All five unmatched accepted queries are matched positives in the other
crop. Crop-0 assignments change across snapshots. The subsequently agreed
confidence-only diagnostic with frozen placement is complete and reviewed above.

**Latest:** [PointNeXt/query fit and saved-weight normalization diagnosis](pointnext_query_results.md).
The user completed the bounded fit; reload passed but both crop gates failed.
With identical weights on CPU, current-crop BN reduces matched placement error
to .43/.22 voxels, while confidence misses/extras remain. No policy change or
additional fit has started; normalization consistency is the proposed next discussion.

**Ready:** [native PointNeXt → Transformer prototype and visible commands](pointnext_query_ready.md).
The selected native backend restores 32-channel features to original points and
reuses the query body/loss under its own checkpoint schema. Fifteen synthetic
CPU tests, a pinned-reference topology audit and real read-only source checks
passed. The user subsequently completed preparation, GPU timing and the bounded
fit; its quality failures are reviewed above. No environment change occurred. The
[agreed design](pointnext_query_design.md) records the bounded two-crop protocol.

**Latest:** [image-reservation control is complete and reviewed](field_image_reservation_results.md).
At matched nominal 2,048, mean coverage falls 85.69% → 84.49%, while proximity
rises 74.74% → 76.55%. Neuron-35 dilution improves, but neuron-94 dim support
and average longest gaps worsen. This fixed quota is not promoted. The proposed
PointNeXt/Transformer prototype is now implemented as described above;
real input preparation and fitting remain separate, pending actions.

**Latest:** [attention sampling is complete and reviewed](field_attention_sampling_results.md).
At matched counts, expanded uniform retains 85.68% coverage versus baseline
85.69%, while proximity improves 74.74% → 76.52%. Crop 35/0 improves strongly;
19/79/94 retention and guided path gaps remain concerns. Neither attention nor
the 50/50 guide is promoted. The user approved preparing one image-reservation control.

**Latest:** [fresh baseline versus skip attention is complete and reviewed](field_attention_results.md).
Attention cleans the targeted neuron-35 field, but mean development coverage
falls from 92.21% to 89.67%; Dice rises .5653 to .5723 amid baseline replay
variation. All 112 saved fields were independently audited. The agreed separate
sampling comparison has now completed; attention is not promoted. The image-backbone
reservation control has subsequently completed and is reviewed above.

**Latest:** [development sampling is complete and reviewed](field_pilot_sampling_results.md).
At matched counts, expanded random raises mean coverage from 71.47% to 85.47%,
with severe dilution in some neuron-35 crops. The 50/50 selector is not a universal
winner. A subsequent bounded image-reservation comparison is complete and
reviewed above; neither that quota nor attention was promoted.

**13 October results:** the [field pilot is complete and reviewed](field_pilot_results.md).
Final development field coverage is 90.19% within two voxels, tube Dice 0.5772.
Full expanded candidates raise eligible coverage from 75.27% to 95.33%, with a
proximity cost and difficult per-crop failures. The fixed four-control development
comparison has now completed; no retraining or sampler promotion has occurred.

**Ready:** the user approved the [33-training / 7-development field pilot](field_pilot_ready.md):
existing field model/loss, 30 epochs, batch 1, LR .001, no augmentation, checks
0/5/15/30, fixed final-checkpoint four-control comparison. Native preparation and
GPU timing are separate commands before training. Native preparation is complete:
211 supervised training and 47 supervised development crops; all 318 records
were verified. GPU timing and 30-epoch training are complete. Reserved test neurons
and later architecture stages stay separate.

**Latest:** the user completed the [Phase 1 four-control comparison](field_sampling_results.md).
Candidate expansion improves reference coverage on fitted crops; expanded random
is the strongest coverage control and the 50/50 selector has some branch losses.
The [completed protocol](field_sampling_ready.md) records execution details. The
[design](phase1_field_guided_sampling_design.md) separates candidate expansion,
coverage and guidance at matched unique-point counts and 512/2,048 budgets.
Production/PointNeXt integration remains pending; broader field training and
development sampling are complete;
no sampler has been promoted automatically.

**Latest field result:** the user completed the 1,600-update eight-crop fit;
the frozen gate passes, with mean tube Dice 0.9173. The replay's 400-update
result differs from the earlier run despite matching settings and initialization.
See [the independent audit, images and limits](centerline_duration_results.md).
The isolated selector is implemented and the broader pilot is prepared; actual
development training and production integration remain pending.

**Current architecture clarification, 12 October 2026:** retain Phase 1
point-cloud preparation -> PointNeXt -> Transformer skeleton proposals -> GATv2
connectivity. The image-field candidate is an isolated experiment for potential
Phase 1 guidance, not an adopted replacement of the encoder or decoder. PointNeXt
is now implemented as a native prototype; GATv2 remains unimplemented and the
historical query pilot used DGCNN. See
[the agreed framework and status](centerline_framework.md) and
[the approved field-duration command](centerline_duration_ready.md).
Updates below describe historical milestones and are not a current run queue.

**12 October update:** the dropout-zero 500-update control is complete. Raw
query target coverage improves from 66.7% to 95.8%, but score 0.5 retains only
29.2%. Training normalization restores 100% raw support, while confidence still
rejects targets. See [the verified comparison and proposed next step](query_dropout_zero_results.md).
The longer fit is proposed; full training and SWC evaluation remain pending.

**Latest:** the user completed the 500-update GPU fit. Query inference reaches
66.7% target coverage but accepts zero slots at score 0.5. Forward-only controls
show dropout dependence: restoring decoder dropout yields 95.8–100% coverage,
while changing BatchNorm statistics does not restore it. The user approved a
fresh fit changing only training dropout to zero. See [the measured evidence and
prepared local command](query_gpu_fit_results.md). Full training remains pending.

**Current architecture milestone:** the reviewed query target builder, decoder,
set loss and paired local runner are implemented. See [the execution evidence
and next GPU command](query_decoder_implementation.md). All 26 targeted tests
pass. A 100-update CPU check improves query target distance from 52.2 to 9.1
voxels, but only reaches 33.3% coverage within six voxels on two fitted crops.
This is insufficient to proceed to the full pilot. A longer visible fixed-crop
GPU check is next; full training and reconstruction evaluation remain pending.
[The design](query_decoder_design.md) retains 512 output slots, spacing 8 voxels
and the explicit reference-agreement label policy. These checks do not establish
PN2 superiority.

**Current update (11 October 2026):** the investigation found confirmed Vaa3D
decoder errors, an invalid uint16 threshold assumption and absolute-coordinate
dependence. See [the evidence and implemented repairs](root_cause_findings.md).
The baseline/pilot results below are historical and need reassessment on
correctly decoded inputs. The agreed direction is input and coordinate repair first.

Latest: the user completed the [actual component recorder](graph_component_recording.md)
in 231.7 seconds with every graph array and metadata unchanged. The
[verified real trace](graph_component_trace_results.md) confirms that the first
two components consume all 128 anchors and 11 eligible components get none.
41 of their 72 candidates lie within 6 voxels of the reference, but the audit
also identifies separate size-rule exclusions and losses near allocated components.
The user approved a fixed-budget allocation repair and cached-matrix inspection.
The [versioned repair comparison](remaining_budget_comparison.md) is now complete:
all 13 eligible components receive anchors at the same 128 budget, but spatial
F1 declines 0.6862→0.6734 as precision falls despite a recall gain. The
[verified path, image and supervision audit](remaining_budget_results.md) locates
much added off-reference geometry in high-scoring component 2. Both off-reference
groups follow visible bright image structures. 52/262 training crops lack local
reference nodes and have zero classification supervision, but most off-reference
candidates arise in reference-containing crops; that policy alone is not the
whole explanation. No new training, bridge rule or default is promoted.
The earlier [saved graph/path audit](hybrid_graph_path_audit.md) provides image context.
The user completed the focused [neuron-79 SWC comparison](hybrid_swc_results.md).
Spatial F1 improves from 0.6560 to 0.6862, with higher precision and recall.
The control reproduces its saved scores exactly. Graph selection/tracing still
fails to retain much recovered proposal support; a fragmented-reference topology
metric limitation is documented. No selector is promoted. The prior
[fixed hybrid control](hybrid_global_comparison.md) still flags neuron 19's
proximity regression; neuron 94's upstream deficit remains unresolved.

The [full-volume audit](full_volume_selection_audit.md) locates major
global-selection losses on 35/79 and an earlier missing-proposal deficit on 94.
The user-approved [matched spatial control](global_coverage_comparison.md)
recovers support at the same budget, but fails the specificity proxy on neuron 19.
These are CPU representation diagnostics, not new SWC improvements or promoted
settings. Discuss the next coverage/confidence tradeoff before changing strategy.

The [corrected training-only foreground comparison](repaired_foreground_comparison.md)
is now complete: 12 crops, no failures. Permissive masks retain faint signal but
greatly increase guide density; exactly matched point counts expose substantial
losses. Neither mask is promoted. The next proposed control preserves bright-core
points while reserving space for weak signals, before the corrected GPU pilot.

The [bright/weak candidate control](mixed_candidate_comparison.md) has now run on
the saved corrected crops. Mixed-pool coverage improves, but gains shrink after
512-input selection and some well-covered branches lose support. Spatial selection
also shows mixed results. Next: the prepared corrected raw/local-coordinate model
control. Its full cache is now prepared and verified: 262 train / 56 validation
crops, with all 40 source image/reference hashes checked. See
[the preparation audit](repair_pilot_ready.md). The user has now completed both
five-epoch training arms in 1m31s. Local coordinates reduce best validation loss
by 1.74%, but proposal locations improve only modestly and the 0.85 score cutoff
removes substantial crop reference coverage. See
[the training and crop-diagnostic results](repair_pilot_results.md). The user-approved
[score/suppression audit](selection_audit.md) is complete: sphere suppression loses
little six-voxel support, while the 0.85 graph gate reduces local proposal coverage
from 81.04% to 60.19% across seven cached neuron unions. No full reconstruction has
run at that point. The user then completed the neuron-21 full-volume pair in
2m14s: spatial F1 improves from 0.7880 to 0.8326 at cutoff 0.5, with a small
precision loss and lower topology F1. See [the results and projections](graph_cutoff_results.md).
The approved seven-neuron extension completed in 27m52s, reusing neuron 21.
Mean spatial F1 changes from 0.6598 to 0.6671: two gains, one loss, four unchanged.
See [the measured results, progress against PN1 and conceptual bottlenecks](validation_graph_cutoff_results.md).
Global proposal selection and sparse score-anchor behavior need investigation;
the lower cutoff is not promoted. No matched final PN1-versus-PN2 benchmark exists.

Development starts from PN 1.0 commit `1719a56e63328782dda7e160a880f16ec1f49937`
on the `pointneuron2` branch in a separate Git worktree. Existing architecture,
loaders, evaluators and SWC export are retained. Clean baseline retraining and
validation are complete. The fixed-parameter image-guided sampling path is
implemented; learnable sampling now trains jointly with the existing proposal
model and reloads from checkpoints. Stage 1 is partially complete; see
`sampling_training.md` for the corrected local smoke results. PointNeXt, DETR
and GATv2 are later stages.

## Local folders and current status

PN 2.0 entry scripts live in `scripts/pointneuron2`; its README maps each runner
to its implementation and result files. From the original checkout, the visible
`pointneuron2` folder is now a real local Git checkout on the `pointneuron2`
branch. The earlier Windows directory junction has been removed. All completed runs
used this machine's Python; training and inference used its CUDA GPU. The local
launcher in `scripts/pointneuron2/run_local.ps1` saves live output and run status.

| Pipeline stage | Status | Evidence / remaining work |
|---|---|---|
| Data partition and baseline | Complete for this internal comparison | 33 train / 7 validation / 8 reserved test neurons; 30-epoch baseline; all seven validation runs completed |
| Image-medial preprocessing | Implemented; expanded automatic mask/candidate comparison completed | 48 patches from six training neurons; automatic masks improve chick/fish guide proximity but worsen fly; no alternative promoted |
| Optional inspection tool | Implemented; outside the required pipeline | Slice/3D viewer and parameter comparisons; manual mask painting is not implemented and is deferred |
| Learnable adaptive sampling (current stage) | Three five-epoch arms and all 21 reconstructions completed; stage 1 incomplete | No-aux spatial F1 0.2823 versus random 0.2306; mean topology F1 lower; guide-loss arm bridge dependence worsens; see `sampling_ablation_reconstruction_results.md` |
| PointNeXt encoder | Not started | Follows completion/evaluation of sampling |
| Query decoder and GATv2 connectivity | Not started | Later architecture stages |

The subsequent source-image audit found substantial guide-extraction loss on
one train patch and guide/candidate compression loss on the informative
validation patch. After the completed noise comparison, we are testing whether
the current guide sampler helps proposal learning at all. See
[the matched pilot](sampling_ablation.md) for the next commands, noise guard
check and decision limits. Earlier evidence remains in `input_audit.md` and
`input_audit_seed0/inspection.png`.

**The agreed direction is a fully automatic main pipeline.** Manual verification
is no longer a prerequisite for continuing. The user's intended semi-automated
feature was correction of a neurite mask, such as adding missed branches;
the parameter viewer does not implement that feature. Such editing is deferred
until automatic reconstruction has been evaluated.

The first automatic foreground comparison is complete:
[foreground_comparison.md](foreground_comparison.md). Chick full-guide proximity
improves, but clean segmentation is not established; other sources show different
tradeoffs. Bounded-guide coverage remains weak on the chick example. See
[automatic_preprocessing.md](automatic_preprocessing.md) for the research sequence.
Following that comparison, we agreed to preserve every extracted guide vertex.
New caches now use full guides independently of the 512 encoder inputs, with
blocked distance reductions. See [full_guides.md](full_guides.md) for the local
cache check, measured GPU memory/runtime and remaining preprocessing questions.
The user's two-epoch full-guide run also completed. Its
[checkpoint visual inspection](checkpoint_inspection_full_guides.md) found
unchanged learned/fixed input sets and no proposals above score 0.5 across eight
patches. Train guide quality and validation candidate coverage remain limitations.
The user's [expanded mask/candidate comparison](preprocessing_comparison_expanded_results.md)
completed on six training neurons and 48 patches without failures. Background
correction improves chick/fish mean guide proximity but worsens all three
annotated fly patches; larger candidate pools give small, mixed selected-input
gains. Foreground reliability remains the priority. The current preprocessing
recipe remains unchanged.
The subsequent [actual-slice and noise-aware pilot](noise_aware_preprocessing.md)
found clipping and broad fluorescent regions in difficult chick images. The
experimental rule suppresses a noisy lattice but gives mixed guide support.
Its [full 48-box comparison](noise_comparison_results.md) completed without
failures, but permissive masks on repeated-intensity fish images reduced mean
selected-input support from 72.0% to 60.1% relative to background Otsu. The
noise-scale floor/growth rule needs revision; no alternative is promoted.
The latest preparation test run passed all 142 tests.
Pool alternatives have mixed results, and using recorded train-image spacing
does not resolve its weak guide; none has been promoted. The existing
[viewer](manual_review.md) remains useful for optional development diagnostics.

Execution checks are healthy. Reconstruction quality is still uneven: the clean
baseline averages topology F1 0.412 and spatial F1 0.374 on seven mixed-domain
neurons, with severe failures in some cases. The sampling probe improved coverage
of its own extracted image axis on three training patches; it has not established
a reconstruction improvement. Compact configs/reports and code are outside
ignored `tmp`. Verified development changes are committed and pushed to the
`pointneuron2` branch on GitHub. The scores in this historical section precede
the reader repairs and are not measurements of the corrected model.

## Frozen baseline and fair comparison

`configs/pointneuron2/frozen_pointneuron1.json` records the exact historical code
revision, proposal checkpoint SHA-256, saved training arguments and promoted
reconstruction settings. Checkpoint and cache paths are relative to an explicitly
provided artifact root (the original checkout); no large dataset is copied into
the worktree. Do not overwrite the historical checkpoint or experiment outputs.

The historical split shares 44 neurons between train/validation and 41 between
train/test. Historical scores are development and diagnostic evidence. Even a new
partition of already inspected Gold166 data is an internal comparison, not an
untouched external test. Use a compatible independent dataset for confirmation.

The new baseline starts from freshly initialized weights, with the historical
training recipe and neuron-disjoint splits. It does not fine-tune the historical
checkpoint, which has already seen some newly reserved neurons. Reusing the
existing GT-filtered training cache controls the first comparison; evaluation of
full held-out volumes must retain failures and must not use GT-based filtering.

## Prepare and run

Run from this worktree with a CUDA-enabled Python environment containing the
project dependencies. Replace `python` with its full executable path when needed.

```powershell
python scripts/pointneuron2/baseline.py --artifact-root "C:/Users/Seywe/Documents/GitHub/PointNeuron"
python scripts/pointneuron2/baseline.py --artifact-root "C:/Users/Seywe/Documents/GitHub/PointNeuron" --smoke
python scripts/pointneuron2/baseline.py --artifact-root "C:/Users/Seywe/Documents/GitHub/PointNeuron" --train
```

The first command validates cache files and writes the split. The smoke run uses
one epoch, two training batches and the full validation split, in
`tmp/pointneuron2/baseline_smoke`; it only
checks execution, not reconstruction quality. The full run uses 30 epochs in
`tmp/pointneuron2/baseline_seed0`. Both save logs, configuration, the command,
manifest/split hashes and status. Existing runs cannot be overwritten; use a new
`--output-root` for another experiment. Existing splits are reused only when their
provenance agrees. Seeded RNGs improve reproducibility; identical GPU numerical
results across environments are not guaranteed.

`build_split.py` now groups by `sample_id` by default. Supply `--record-root` to
resolve cache paths against a different checkout, `--dataset-id` for provenance,
or `--group-by donor_id` when explicit donor metadata is present in every record.
Missing identities, inconsistent groups, duplicate records and fewer than three
groups fail before writing a split. Ratios apply to groups, not patch counts.
Legacy manifest IDs ending in `#patch-N` are normalized to the parent neuron ID;
the original ID is retained in metadata. Sample indices are checked against neuron
identity to catch inconsistent manifests. Schema v3 records this identity rule.
This first version does not stratify by imaging source. `--group-by record` is
only for reproducing historical patch-level experiments.

New training commands use `--require-neuron-split`, which checks every recorded
neuron/group assignment, trains only on train, validates only on val and reserves
test. The new runner always enables this guard and saves a seed.

The prepared seed-0 split contains 33 training neurons (560 patches), seven
validation neurons (153 patches), and eight test neurons (170 patches), with zero
neuron/sample-index overlap. Source distributions differ: the validation and test
sets include Allen source groups absent from training. Report per-source results
and avoid interpreting this small random split as a balanced domain benchmark.

Baseline training completed all 30 epochs. All seven full validation neurons
completed; the results and limitations are in `baseline_validation.md`.
After integrating trainable sampling, the then-current 93 repository tests passed with unrelated pytest
plugins disabled (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`). The sampling component also
passed a training-image probe, cache-generation smoke, and CUDA proposal-inference
smoke. The corrected trainable smoke changed alpha from 0.25 to 0.249359 and
reloaded its saved model, but produced no proposals above score 0.5 on two image
patches. These execution checks do not establish reconstruction improvement.

## Next experiments

1. Finish stage 1: assess guide fidelity and candidate coverage, then evaluate
   sampling against random/spatial inputs using matched budgets, new caches and
   the same encoder. Training/checkpoint integration is complete. Patches with
   no local SWC nodes contribute guide loss only, since absent annotation does
   not establish background. Then implement PointNeXt with the existing
   per-point feature interface.
2. Compare selectors on identical proposals and node budgets. Track terminal,
   junction and segment coverage before and after selection. Historical hard
   samples are development diagnostics, not an untouched test set.
3. Construct candidate edges at several spatial scales and measure reference
   edge recall before fitting a simple edge scorer and a GATv2 alternative.
4. Implement DETR-style proposals and GATv2; compare each architecture change and
   image-guided sampling separately.
   Integrate changes only when their final reconstruction benefit is measured.
5. Keep topology F1 separate from spatial F1. Report failures, fallback bridge
   dependence, runtime and memory. Confirm the final choice externally.

Joint optimization requires explicit gradient paths through discrete operations;
tree validity alone does not establish anatomical correctness. Reference-derived
nodes or sampling are oracle diagnostics, never deployed held-out inputs.
