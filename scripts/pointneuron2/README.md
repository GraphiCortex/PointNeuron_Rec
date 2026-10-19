# PointNeuron 2.0 scripts

These scripts and their dependencies are local files on your computer. This is
the permanent folder for PN 2.0 entry scripts; new stage-specific runners belong
here. Generated caches, checkpoints and logs stay in ignored `tmp/pointneuron2`.
Development snapshots are kept on the `pointneuron2` branch. Generated caches,
environments, logs and checkpoints remain outside those snapshots.

Completed: [proposal optimizer-state control](../../docs/pointneuron2/proposal_optimizer_state_results.md).
`compare_proposal_optimizer_state.py` pairs exactly four CPU updates with fresh
versus saved AdamW moments/counters, copying identical optimizer group options.
Saved state passes the declared 6/6 crop-0 trust criterion; fresh state fails.
All four crops are probed at every update. Three transfer tests pass. No source
fit is extended, no new trained checkpoint retained and no joint/GPU run launched.

Stopped: [joint timing failure and CPU comparison](../../docs/pointneuron2/joint_query_graph_timing_results.md).
`audit_joint_timing_stop.py` makes at most four throwaway CPU updates; its
`--proposal-only` arm omits graph loss at the same restart settings. Both show
early geometry drift; the current GPU timing stop remains in force. Original
runtime sources, trust masks, losses and output thresholds are unchanged.

Prepared: [bounded joint proposal/GATv2 stage](../../docs/pointneuron2/joint_query_graph_ready.md).
`run_joint_query_graph.ps1` separates CPU prepare/check, throwaway GPU timing,
capped train and exact intermediate resume. Existing heads, frozen Phase 1,
all 512 queries before final .5 node selection; combined proposal/edge loss and
fresh joint optimizer from two completed warm starts. The automatic node, graph,
geometry and SWC gate replaces conditional success as the outcome of this stage.
GPU actions remain user-launched.

Completed: [automatic-selection GAT integration comparison](../../docs/pointneuron2/gat_automatic_selection_results.md).
`evaluate_gat_automatic_selection.py` uses CPU inference only, with the unchanged
successful GAT checkpoint and balanced cached proposals. It pairs oracle versus
the existing .5 automatic selection, rebuilds candidates, exports SWCs and
separates node/candidate/edge-score/forest losses. No reference data enter the
automatic selector or scorer. Eighteen targeted tests pass; all eight saved
scores replay exactly and SWCs verify. No pruning, fit or policy change.

Completed: [frozen-proposal GATv2 results](../../docs/pointneuron2/frozen_proposal_gat_results.md).
The user-launched fit passes all four conditional training graph gates from
evaluated update 500 onward, including terminal 4,000. Five checkpoints and 20
SWCs are verified; source/cache hashes match. The subsequent automatic-selection
evaluation is reviewed above; no new fit or production promotion is prepared.

Preparation: [frozen-proposal GATv2 control](../../docs/pointneuron2/frozen_proposal_gat_ready.md).
`run_frozen_proposal_gat.ps1` exposes prepare/check/timing/train/resume actions.
The existing scorer learns connections on four frozen, oracle-selected TRAINING
graphs. CPU preparation and zero-update checks pass; 28 targeted tests pass.
GPU timing and the fresh 4,000-update fit were separate user-launched commands. Production
selection, proposal supervision and existing checkpoints remain unchanged.

Completed: [four-crop frozen confidence results](../../docs/pointneuron2/four_crop_frozen_confidence_results.md).
All strict gates fail after the user-launched 4,000-update CPU stage. Six saved
checkpoints replay exactly; frozen tensors/sources match. No extension or new
fit is prepared. The head is not promoted into reconstruction.

Preparation: [four-crop frozen confidence stage](../../docs/pointneuron2/four_crop_frozen_confidence_ready.md).
`run_four_crop_frozen_confidence.ps1` exposes separate CPU prepare/check/train/
resume actions through `four_crop_frozen_confidence.py`. Only the saved 258
confidence parameters train; geometry and fixed features/labels stay unchanged.
Thirteen tests and eight throwaway readiness updates pass. The retained fit was
user-launched and capped at 4,000 new-stage updates; no GPU workload is involved.
Fresh optimizer and all-four-crop updates are explicit training-stage changes.

Latest capacity result: [four-crop fixed-feature oracle](../../docs/pointneuron2/four_crop_confidence_capacity_results.md).
`audit_four_crop_confidence_capacity.py` reconstructs actual hidden classifier
inputs, verifies saved logits, then solves one CPU affine feasibility problem
for all four crops together. Float32 verification gives zero errors on 2,048
cached labels. Eight known-case tests pass. No neural training, GPU work,
classifier checkpoint, inference-policy change or reconstruction evaluation.

Design audit: [assignment/selection contract](../../docs/pointneuron2/query_assignment_contract_review.md).
`audit_query_assignment_contract.py` checks saved target-to-query map changes
separately from binary positive/negative label changes across both fits' five
snapshots. It performs no matching, forwards, updates, GPU calls or dev/test
access. The subsequently agreed feature-capacity oracle is reviewed above;
no new learning experiment is prepared.

Latest result: [failed hard-negative comparison](../../docs/pointneuron2/hard_negative_query_results.md).
`review_hard_negative_query_fit.py` audits both saved fits on CPU, reusing the
preserved paired reviewer with an explicit read-only weighted-CE name adapter.
It checks hard-set membership/objective, all original gates, sources, cache,
initialization, predictions and checkpoints. No forwards, updates, GPU work,
cutoff search or development/test access. The variant fails its predeclared
tradeoff; no new learning experiment is prepared.

Previous preparation: [hard-negative confidence comparison](../../docs/pointneuron2/hard_negative_query_ready.md).
`run_hard_negative_query.ps1` separates prepare/check/timing/train/resume through
`hard_negative_query_experiment.py`. The new loss changes only confidence
redistribution, keeping all geometry, matching and primary candidate gates.
Twenty-eight targeted tests pass. Exact-cache CPU checks and fresh initialization
are recorded in the linked report. User-launched GPU timing passes with zero
reload delta and unchanged memory, estimating 85–128 minutes under those measured
conditions. The user subsequently completed the capped fit in 18.94 minutes;
results are reviewed above. Timing retains no trained checkpoint. No additional
GPU work was launched during review.

Design: [hard-negative objective specification](../../docs/pointneuron2/hard_negative_objective_proposal.md).
`audit_hard_negative_objective.py` computes proposed CE and analytic score-margin
derivatives on saved final predictions, without forwards, backpropagation or
updates. It is an algebraic check, not an implemented training loss or evidence
of better predictions. The subsequently agreed comparison is prepared above.

Latest result: [completed matched confidence comparison](../../docs/pointneuron2/balanced_query_results.md).
`review_balanced_query_fit.py` checks both fits' saved predictions/checkpoints on
CPU, including initialization/cache/backend identity, independent distances,
class losses, candidate labels, curve support and original gates. It makes no
model forwards, updates, GPU calls or cutoff search. Ranking improves, but all
combined gates fail; no further fit or graph training is prepared.

Previous preparation: [matched confidence normalization and candidate-margin comparison](../../docs/pointneuron2/balanced_query_ready.md).
`run_balanced_query.ps1` provides separate prepare/check/timing/train/resume actions;
`balanced_query_experiment.py` changes only classification normalization in the
matched four-crop fit. Exact cache/initialization, 18 tests and CPU reload pass.
`compare_candidate_margin.py` compares radii eight/twelve on saved predictions
without model forwards. User-launched GPU timing passes with zero reload delta
and a 22.6–33.8 minute fit estimate; the user completed the capped fit, reviewed
above. Original gates and baseline sources are preserved.

Previous: [completed four-crop fit and research decision](../../docs/pointneuron2/branched_query_results.md).
`review_branched_query_fit.py` checks cached predictions/checkpoints on CPU and
reports geometry, confidence ranking/class-prior baselines, assignment changes
and candidate-edge losses. No model forwards, optimizer updates, GPU work or
threshold search. All four final learnability gates fail; no training extension
or GATv2 fit is queued. The subsequently agreed bounded comparison is prepared above.

Previous: [500-update evaluation repair and restart](../../docs/pointneuron2/branched_query_evaluation_repair.md).
The four-crop runner restores empty probe array dimensions, saves complete
checkpoints before evaluation and writes full Python tracebacks. The launcher
preserves full native stderr and now defaults to `branched_query_restart_20261017`.
Ten tests and repaired CPU checks pass; the original failed run is preserved.
The model, loss and training protocol remain unchanged. GPU timing/retry remain
separate user-launched actions.

Ready: [four-crop branched proposal fit](../../docs/pointneuron2/branched_query_ready.md).
`run_branched_query.ps1` exposes separate prepare/check/timing/train/resume
actions through `branched_query_experiment.py`. Exact clouds and CPU checks
are complete; GPU timing and the capped 4,000-update fit remain user-launched.
Phase 1 stays fixed. Full optimizer/RNG state is saved; explicit resume cannot
extend the original budget. This is proposal learnability, not reconstruction.

Previous: [real upstream connectivity audit](../../docs/pointneuron2/upstream_connectivity_results.md).
`audit_upstream_connectivity.py` performs four training-crop CPU forwards with
pinned source weights and image-only clouds, then measures trusted node/edge
availability. No optimizer/GATv2/GPU workload is created. Both branched cases
fail frozen-upstream feasibility; the proposed next learning experiment is
documented for discussion, with no training command prepared or launched.

Previous: [connectivity implementation and CPU execution checks](../../docs/pointneuron2/gatv2_connectivity_implementation.md).
`check_connectivity.py` verifies native GATv2/control gradients and strict reload
on identical synthetic proposals, then tests forest decoding with reference
edge scores. No optimizer updates or upstream/image forwards occur. The runner
is not a real training command; checkpoints it writes are explicitly untrained.

Current: [reference-only representation results](../../docs/pointneuron2/representation_oracle_results.md).
`representation_oracle.py` checks four fixed training-reference cases on CPU,
without images, models or training. All representation criteria pass; learned
connectivity and automatic crop matching remain unimplemented. See the
[protocol and command](../../docs/pointneuron2/representation_oracle_protocol.md).
The older frozen-confidence duration proposal below is deferred.

Latest: [LR replay improvement and one remaining miss](../../docs/pointneuron2/confidence_lr_results.md).
The user completed the .001 replay: 17/18 matches accepted and no extras,
with a failed combined gate because crop-0 query 408 scores .415. Hash/reload
and paired checks passed. No new fit is prepared; a duration-only fresh replay
is proposed, since the existing head checkpoints do not save AdamW moments.

Ready: [matched learning-rate-only confidence replay](../../docs/pointneuron2/confidence_lr_ready.md).
`run_confidence_lr.ps1` separates read-only `check`, four throwaway timing
updates in `preflight`, and fresh `train`. LR changes .0001 -> .001; all other
recipe settings, initial weights, fixed cache and labels match. Real source
checks and four tests passed. Replay GPU timing/fit are complete and reviewed
above; the completed baseline runner and outputs are unchanged.

Latest: [confidence fit and linear-capacity findings](../../docs/pointneuron2/query_confidence_results.md).
The real confidence fit is complete: crop 1 passes, crop 0 does not.
`audit_confidence_separability.py` is a CPU training-label oracle that checks
whether the cached labels admit a linear separator. It verified a shared
separator without neural updates or classifier checkpoint writes. An LR-only
replay is now agreed and prepared above; no new fit or architecture change has started.

Ready: [frozen-placement confidence diagnostic](../../docs/pointneuron2/query_confidence_ready.md).
`run_query_confidence.ps1` separates CPU `prepare`, four throwaway timing updates
in `preflight`, and a visible bounded `train`. The real feature cache is already
prepared; GPU timing/fit are complete and reviewed above. Only the existing final two-class output
trains (258 parameters), with fixed features/assignments and no model forwards
during the cached-feature fit. No full-model checkpoint is overwritten.

Latest: [per-crop normalization consistency and detailed query-score audit](../../docs/pointneuron2/pointnext_cropnorm_results.md).
`compare_pointnext_cropnorm.py` is CPU only and makes zero optimizer updates.
It explicitly converts saved parameters into a separate checkpoint identity,
checks train/eval/reload/crop order and writes score CSVs at 500/1000/2000.
The completed control fixes the inference-statistics switch; confidence still
fails. The subsequent frozen-placement confidence diagnostic is prepared above.

Latest: [PointNeXt fit and inference-normalization audit](../../docs/pointneuron2/pointnext_query_results.md).
`audit_pointnext_inference.py` compares ordinary eval and current-crop BN on
temporary CPU copies of the saved weights, with no optimizer/checkpoint writes.
The real audit completed; placement improves substantially but confidence
still fails. No new training or production-normalization policy is queued.

Ready: [native PointNeXt/Transformer prototype and visible commands](../../docs/pointneuron2/pointnext_query_ready.md).
`run_pointnext_query.ps1` separates read-only `check`, two frozen field forwards
in `prepare`, four throwaway timing updates in `preflight`, and a future explicit
2,000-update `train`. Synthetic CPU tests and the pinned-reference topology
audit passed. Real preparation/timing and the bounded fit are complete;
the failed learnability gate is reviewed above. Do not repeat the fit automatically.

Latest: [reviewed image-reservation gains and dim-support losses](../../docs/pointneuron2/field_image_reservation_results.md).
`run_field_image_reservation.ps1 -Action check` verifies existing sources read-only;
`-Action compare` separately freezes the quota clouds and scores paired coverage,
proximity and branch gaps against saved expanded random. CPU only, no model
inference or training. Fixed half-budget uniform foreground reservation, uniform
union fill, matched 512/2048 nominal budgets and seeds 0–4. Real comparison
completed in 1m28s; the fixed quota is not promoted. No further run is queued.

Latest: [completed attention sampling and branch-support findings](../../docs/pointneuron2/field_attention_sampling_results.md).
`run_field_attention.ps1` has separate `preflight`, `train` and `compare` actions.
It reuses the native cache; two fresh 30-epoch fits share initial backbone tensors
and crop order. Attention adds two skip gates; sampling rules are unchanged.
Timing, both 30-epoch fits and the fixed-final-checkpoint sampling comparison
are complete. Comparison took 8m20s and made zero optimizer updates. No new
run is queued; attention is not promoted. The image-reservation control is
implemented separately and its completed comparison is reviewed above.

Latest: [completed development sampling and input dilution findings](../../docs/pointneuron2/field_pilot_sampling_results.md).
The user completed the fixed final comparison in 4m24s, with zero updates.
The subsequent bounded image-reservation comparison is complete and reviewed
above. No selector was promoted and no new run is queued.

Latest: [field train/development results and saved-field audit](../../docs/pointneuron2/field_pilot_results.md).
The user completed 6,330 updates in 17m14s. Training/development and full-pool
eligibility were reviewed. `run_field_pilot.ps1 -Action compare` has subsequently
completed with fixed epoch 30, four development controls and zero updates.

Ready: [approved field train/development pilot](../../docs/pointneuron2/field_pilot_ready.md).
`run_field_pilot.ps1` has separate `prepare`, `preflight`, `train` and `compare`
actions. First prepare full native crops/weak labels from the existing image-defined
33/7 cache, then measure local GPU runtime. Training is fresh, 30 epochs, batch 1,
LR .001, no augmentation, with checks 0/5/15/30. The sampling comparison uses fixed
epoch 30 and unchanged four controls; no automatic production integration.

Latest: [completed Phase 1 input results and branch losses](../../docs/pointneuron2/field_sampling_results.md).
The user completed the four-control comparison; candidate expansion helps fitted
crops, while the 50/50 selector is not promoted. The broader field pilot is
approved and prepared; native caching, GPU timing and training are complete.
The development sampling comparison is complete; no sampler is promoted.
Completed protocol: [isolated input comparison](../../docs/pointneuron2/field_sampling_ready.md).
`run_field_sampling.ps1 -Action compare` reuses the frozen passing field checkpoint
for eight training-crop forward passes, then compares four selectors on CPU with
reference scoring only after inputs are frozen. `-Action preflight` checks/times
one crop. Neither action trains or modifies the production sampler; the broader
field pilot and PointNeXt/Transformer/GATv2 integration remain separate steps.

Latest: [eight-crop field learnability passes at 1,600 updates](../../docs/pointneuron2/centerline_duration_results.md).
The user completed the fit; review its within-run improvements and common-budget
reproducibility caveat before another experiment. `plot_centerline_duration.py`
renders the saved 400/1600 fields with equal scales and no inference.
Completed protocol: [duration-only field check](../../docs/pointneuron2/centerline_duration_ready.md).
`run_centerline.ps1 -Action duration` starts one fresh uninterrupted hard-negative
fit with saved checks 0/400/800/1600 and compares against the completed 400-update
run. `-Action duration-preflight` checks initial matching and times only four
throwaway updates. The [framework explanation](../../docs/pointneuron2/centerline_framework.md)
retains Phase 1 point-cloud preparation, PointNeXt, Transformer proposals and
GATv2 as the intended architecture. This command fits an isolated field candidate
for possible Phase 1 guidance; its point-cloud interface is not implemented.
Latest: [reference-tube oracle localizes crowded-case errors upstream](../../docs/pointneuron2/centerpath_oracle_results.md).
`audit_centerpath_oracle.py` runs the unchanged extractor on explicitly
ground-truth-derived tube input for two training crops, then compares with
the saved learned paths. This is a diagnostic oracle, not model inference.
It leaves checkpoints, labels, the primary gate and validation unchanged.
Earlier result: [fixed center-path extraction improves the simple case; crowding remains problematic](../../docs/pointneuron2/field_centerpaths_results.md).
`audit_field_centerpaths.py` compares two pre-agreed training crops using frozen
checkpoint inference and the existing CPU image-medial extractor, with no
optimizer or reference-guided extraction. It measures arc-length-weighted
distances to continuous segments and saves native path arrays and overlays.
Earlier result: [negative-loss control improves all eight crops; remaining width and placement errors](../../docs/pointneuron2/centerline_loss_control_results.md).
`plot_centerline_loss_comparison.py` renders two illustrative original/control
fields at identical display scales. The read-only audit also reports reference
distance to accepted support at several tolerances without changing inference
or the primary gate.
Earlier result: [field learns structure but fails centerline localization](../../docs/pointneuron2/centerline_fit_results.md).
Completed protocol: [loss-only paired control](../../docs/pointneuron2/centerline_loss_control_ready.md).
`run_centerline.ps1 -Action loss-control` fits one fresh matched model using
the top min(N,3P) known-negative BCE, then compares against the preserved
original run. `-Action loss-control-preflight` checks pairing and eight initial
inferences, then times only four throwaway updates. Neither action changes the
threshold, labels, model or gate; the ordinary `fit` action remains all-negative.
`audit_centerline_fit.py` verifies the frozen final checkpoint and diagnoses
known-label false positives, tube width and loss contributions without an
optimizer. Its optional thinning step reports incompatible/missing libraries
explicitly and leaves the environment unchanged.
Completed protocol: [centerline-field prototype and eight-crop fit](../../docs/pointneuron2/centerline_fit_ready.md).
`run_centerline.ps1 -Action fit` launches the 400-update full-resolution 3D image
learning check visibly; preparation is already cached. `-Action prepare` writes
native crops/weak labels/inspection overlays to a fresh cache, and
`-Action preflight` performs only four throwaway updates for GPU timing. This
prototype has not yet been evaluated on validation neurons or reconstructed SWCs.

Latest results: [completed query pilot fails selection; proposed centerline-field direction](../../docs/pointneuron2/query_pilot_results.md).
Earlier result: [pooled LayerNorm passes the two-crop learnability check](../../docs/pointneuron2/query_layernorm_results.md).
Completed protocol: [matched 30-epoch train/validation pilot](../../docs/pointneuron2/query_pilot_ready.md).
`run_query.ps1 -Action pilot` selects the frozen pooled-LayerNorm/dropout-zero
config and reports per-neuron validation geometry/confidence every epoch.
`-PreflightOnly` checks that full setup without training. `time_query_pilot.py`
provides the bounded representative timing used for the duration estimate.
Earlier results: [completed 2,000-update comparison and pooled-normalization audit](../../docs/pointneuron2/query_extended_fit_results.md).
Completed protocol: [query-only pooled LayerNorm comparison](../../docs/pointneuron2/query_layernorm_ready.md).
`run_query.ps1 -Action normalization-smoke` calls `compare_query_normalization.py`
to reuse the saved BatchNorm run and fit only the LayerNorm query model. It
checks initialization/input/budget matching and saves single-crop diagnostics.
Earlier results: [completed dropout-zero comparison](../../docs/pointneuron2/query_dropout_zero_results.md).
Completed protocol: [2,000 updates per arm with saved inference checks](../../docs/pointneuron2/query_extended_fit_ready.md).
Use `-Steps 2000 -EvalSteps 500,1000,2000 -LogEvery 50 -DecoderDropout 0` with
the prepared manifest. Checkpoints and raw predictions remain in the printed
local run directory; intermediate observations preserve the training trajectory.
`summarize_query_dropout_control.py` verifies matched saved runs and reports
per-crop geometry/confidence diagnostics, without optimization or source-image reads.

Current architecture work: [query decoder implementation, checks and next GPU
command](../../docs/pointneuron2/query_decoder_implementation.md).
Latest: [the completed GPU fit and approved dropout-only control](../../docs/pointneuron2/query_gpu_fit_results.md).
`audit_query_fit.py` diagnoses saved confidence, normalization and dropout with
fixed checkpoints. `run_query.ps1 -Action smoke -DecoderDropout 0` prepares the
fresh control without changing the base settings or cache.
`run_query.ps1` separates `prepare`, fixed-crop `smoke` and full `train` actions;
no stage automatically starts the next. `build_query_cache.py` derives common
curve labels without changing the corrected image candidates.
`train_query_comparison.py` compares pointwise/query heads with matched initial
encoder weights, inputs and updates. `aggregate_query_proposals.py` provides
image-only inference with raw/local/global outputs and a common graph adapter.
The 2,000-update fit completed and learned useful confidence. A fixed-weight
audit isolates the pooled encoder BatchNorm mode as the dominant remaining
placement gap on these two crops. The pooled LayerNorm control completed and
passes the two-crop learnability check. The 30-epoch pilot completed but the
query model accepts no validation proposals at score 0.5. It is not promoted.
The proposed image-native centerline-field direction awaits agreement.
`audit_query_targets.py` preserves the independent training-only sizing audit.
See [the reviewed target specification](../../docs/pointneuron2/query_decoder_design.md).

Latest saved-output diagnostics:
`compare_hybrid_global.py` adds one fixed 50/50 confidence-anchor/coverage-fill
control, retaining 2,048 currently accepted anchors at the same candidate/output
budgets and sphere rule. It verifies and reuses the earlier controls, then measures
gate/pool effects without training or tracing. See
[the completed hybrid comparison](../../docs/pointneuron2/hybrid_global_comparison.md).

`audit_full_volume_selection.py` follows the full-volume proposals through
score ranking, suppression/capping, graph gating, actual graph selection and
saved SWC geometry. `compare_global_coverage.py` tests score versus spatial
priority with matched candidate/output limits and sphere suppression on three
capped neurons. Both run on CPU, with no model inference/training/tracing.
See [the audit](../../docs/pointneuron2/full_volume_selection_audit.md) and
[the spatial comparison](../../docs/pointneuron2/global_coverage_comparison.md).

Current work: [reader/range/coordinate repairs and measured evidence](../../docs/pointneuron2/root_cause_findings.md).
`audit_root_causes.py` runs bounded diagnostics. Old sampling caches are
historical data and cannot be reused for new training. The staged runner accepts
`--config configs/pointneuron2/repair_comparison.json` at preparation for the
new corrected-data comparison; no stage automatically starts the next.
`probe_repaired_foreground.py` follows a single corrected image crop through
automatic masks, full guides, candidates and 512-point inputs; it does not train
a model or score a connected reconstruction.

Latest: [corrected training-mask results and next proposed input control](../../docs/pointneuron2/repaired_foreground_comparison.md).
`compare_foreground.py` accepts `--artifact-dir` to keep raw arrays in ignored
storage while retaining reports and figures in the visible documentation folder.
`summarize_repaired_foreground.py` compares saved guides at exactly matched counts;
it performs no extraction or training.

`compare_mixed_candidates.py` compares bright, permissive and preset mixed pools
on corrected saved training crops with matched seeds and fixed budgets.
See [the measured tradeoffs](../../docs/pointneuron2/mixed_candidate_comparison.md).

`audit_proposal_selection.py` checks the two frozen corrected-input checkpoints
through local/global sphere suppression, graph score gating and the existing
adaptive candidate pool. It compares graph cutoffs 0.5/0.85 and adds fixed-budget
controls, without training or geodesic reconstruction. See
[the completed audit and repeat command](../../docs/pointneuron2/selection_audit.md).

`compare_graph_cutoffs.py` prepares provenance separately from execution, then
generates one shared proposal file for validation neuron 21 and reconstructs
graph cutoffs 0.85/0.5 with fixed weights/settings. It checks shared-file hashes
and scores both SWCs. See [the prepared local command](../../docs/pointneuron2/graph_cutoff_comparison.md).

`summarize_graph_cutoffs.py` verifies saved paired outputs/metrics and renders
shared-axis SWC projections on CPU. See [neuron 21's measured results](../../docs/pointneuron2/graph_cutoff_results.md).
`compare_validation_cutoffs.py` extends the same comparison to all seven frozen
validation neurons, reusing neuron 21 and retaining failures.
See [the completed protocol](../../docs/pointneuron2/validation_graph_cutoffs.md).
`summarize_validation_cutoffs.py` verifies every saved pair, recomputes means,
checks numerical graph/SWC equality and renders per-neuron F1 and SWC plots on
CPU. See [the seven-neuron results and conceptual interpretation](../../docs/pointneuron2/validation_graph_cutoff_results.md).

| File | Purpose | Current status |
|---|---|---|
| `baseline.py` | Prepare the neuron-disjoint split and run fresh PN 1.0 baseline training | 30-epoch baseline and seven-neuron validation completed |
| `sampling_probe.py` | Compare random, spatial and image-medial sampling on three patches of a training neuron | Completed implementation probe; not a reconstruction benchmark |
| `run_local.ps1` | Launch either runner with local Python, a fresh output directory and saved console logs/status | Local launcher |
| `build_sampling_cache.py` | Preserve bounded image candidates, full medial guides, and separate SWC labels | Corrected reader/range checked on six new patches; historical caches are for read-only analysis |
| `train_sampling.py` | Jointly train sampler and proposal model; random-input control uses the same pools | Two-epoch CUDA integration smoke completed |
| `sampling_ablation.py` | Prepare one cache, train three matched arms, then reconstruct/score in separate visible stages | Three training arms and all 21 user-launched reconstructions completed |
| `summarize_ablation.py` | Validate completed metrics, audit saved proposal selection and render matched SWC projections | CPU audit and figures completed; no additional inference or training |
| `inspect_ablation_confidence.py` | Check every cached validation patch's proposal scores and changes versus initial alpha | Completed inference-only check on all three checkpoints; no reconstruction claim |
| `check_noise_guard.py` | Check bounded noise thresholds on exact saved training scores | Four mask-only regression cases passed; no new extraction or training |
| `evaluate_sampling.py` | Compare random/fixed/learned inputs against local validation-reference curves | Representation diagnostic; not whole-neuron reconstruction scoring |
| `visualize_sampling.py` | Inspect image candidates, guides and SWC references in XY/XZ | Two annotated patches inspected; broader guide assessment remains |
| `inspect_sampling_checkpoint.py` | Overlay raw images, full guides, learned/random inputs and checkpoint proposal scores | Cached-patch inference diagnostic; no training or connected reconstruction |
| `audit_sampling.py` | Check original image support, full-guide/pool coverage and boundary-crossing references | Eight existing patches audited on CPU; see input audit report |
| `compare_foreground.py` | Compare fixed, automatic raw and background-corrected masks at identical locations and budgets | CPU comparison completed on 12 patches from three training neurons; no method promoted |
| `compare_candidates.py` | Compare candidate budgets/preparation and fixed selectors on saved masks/full guides | User-run six-neuron, 48-patch experiment completed without failures; no recipe promoted |
| `inspect_foreground_slices.py` | Read saved masks and show actual slices with depth-clipped references/guides | Four training cases inspected; contrast/clipping diagnostic available |
| `prepare_noise_pilot.py` | Test noise hysteresis on exact saved background scores | Four-case pilot completed; helps noisy lattice, mixed guide support |
| `profile_full_guides.py` | Measure local GPU memory and forward/backward time for capped/dense-full/blocked-full guides | Completed on four training patches up to 13,056 guide vertices, plus DGCNN/proposal backward; zero optimizer steps |
| `verify_full_guides.py` | Compare full-guide probabilities, selections, loss and alpha gradient against direct dense distances | Diagnostic reference for voxel-distance ties; no optimizer steps |
| `review_sampling.py` + `review_sampling.html` | Inspect image slices/3D skeletons, adjust extraction and save review decisions | Optional development diagnostic; no mandatory manual step; does not paint neurite masks |

Reusable algorithm implementations belong to the Python package:

- [Image-medial extraction and fixed sampling](../../src/pointneuron/data/medial_sampling.py)
- [Learnable sampler and Chamfer loss](../../src/pointneuron/models/adaptive_sampling.py)
- [Full-guide implementation and GPU results](../../docs/pointneuron2/full_guides.md)
- [User-run checkpoint visual inspection](../../docs/pointneuron2/checkpoint_inspection_full_guides.md)
- [Expanded mask/candidate results and contrasting overlays](../../docs/pointneuron2/preprocessing_comparison_expanded_results.md)
- [Noise-aware method, slice results and reproduction command](../../docs/pointneuron2/noise_aware_preprocessing.md)
- [Completed noise-aware comparison and identified failure](../../docs/pointneuron2/noise_comparison_results.md)
- [Next experiment: three matched training arms and commands](../../docs/pointneuron2/sampling_ablation.md)
- [Completed sampler pilot results and next reconstruction command](../../docs/pointneuron2/sampling_ablation_pilot_results.md)
- [Completed reconstruction results and proposed next diagnostic](../../docs/pointneuron2/sampling_ablation_reconstruction_results.md)
- [CPU mask/candidate experiment specification and reproduction command](../../docs/pointneuron2/preprocessing_comparison_expanded.md)
- [Training integration results](../../docs/pointneuron2/sampling_training.md)
- [Source-image audit and next steps](../../docs/pointneuron2/input_audit.md)
- [Automatic foreground results and reproduction command](../../docs/pointneuron2/foreground_comparison.md)
- [Automatic preprocessing: next proposed comparison](../../docs/pointneuron2/automatic_preprocessing.md)
- [Optional viewer and restart command](../../docs/pointneuron2/manual_review.md)
- [Sampling parameters](../../configs/pointneuron2/medial_sampling.json)
- [Sampling results and limitations](../../docs/pointneuron2/adaptive_sampling.md)
- [Clean baseline results](../../docs/pointneuron2/baseline_validation.md)
- [Current pipeline status](../../docs/pointneuron2/README.md)

Stage 1 remains partially complete. Learnable-sampler training and checkpoint
inference are connected and execution-tested. The longer matched reconstruction
comparison and broader image-skeleton assessment come before PointNeXt, the query decoder
and GATv2. The two-epoch model produced no accepted proposals at score 0.5; it is
an integration check, not a usable reconstruction model.

## Run on your machine

**Current discussion:** [completed reconstruction results and proposed next diagnostic](../../docs/pointneuron2/sampling_ablation_reconstruction_results.md).
The completed noise comparison exposed a collapsed-MAD failure. A separate
bounded mask check now passes, while the three training arms keep the same
original fixed-threshold inputs. Use `ablation-prepare`, then `ablation-train`,
then `ablation-evaluate` as separate terminal commands; none starts the next stage.
The viewer and historical reproduction commands below remain optional.

New caches retain the **full extracted guide** (`guide_points: null`). Encoder
input remains 512 points selected from 1,024 image candidates; these are separate
settings. Historical caches/checkpoints retain their saved caps. The four-patch
full-guide cache is an execution diagnostic with training patches only, so it is
not the train/validation manifest for the matched pilot. The pilot's cache
preparation and three five-epoch CUDA runs have completed through user-launched
commands, as has reconstruction evaluation of all three arms. Those commands
below now serve as reproduction instructions; another full run is not needed
to inspect the saved results.

From the original `Documents/GitHub/PointNeuron` folder, the local `pointneuron2`
folder is a real local checkout. The earlier Windows directory junction has
been removed, and all files, including ignored run artifacts, were transferred
and verified. The original checkout stays on `main`; the development checkout
is on `pointneuron2`. The parent checkout excludes this nested checkout locally;
source changes are tracked by Git inside the PN 2.0 folder.

```powershell
& .\pointneuron2\scripts\pointneuron2\run_local.ps1 -Action sampling-probe -ArtifactRoot $PWD.Path
& .\pointneuron2\scripts\pointneuron2\run_local.ps1 -Action sampling-smoke -ArtifactRoot $PWD.Path
```

That command uses the already configured local CUDA Python environment. It prints
the fresh run directory and saves `console.log`, `launch.json` and the probe's
JSON report there. Use `-Python` to supply another compatible executable. Heavy
runs use the same launcher, with `-Action baseline-smoke` or `baseline-train`;
these actions rerun the baseline and are not needed to continue sampler development.
Nothing is scheduled or sent to a remote compute service.

For a visible full-guide training check, run the `sampling-smoke` command above.
It prepares four train-neuron-0 and four validation-neuron-19 patches on CPU,
then runs two epochs on CUDA (at most two training batches per epoch). It prints
each batch's losses, alpha, elapsed time and current CUDA allocated memory. Logs,
run settings and the best checkpoint are saved under the printed local run folder.
This is a small training/checkpoint check, not a reconstruction-quality benchmark.
Direct trainer runs accept `--log-every 1` for the same batch visibility.

`-Action preprocessing-compare` reproduces the completed training-only CPU mask and
candidate experiment, with saved figures and live logs. It uses full guides,
six neurons and up to eight patches each. It does not launch GPU training or
modify the main recipe. See the linked experiment note for exact settings.

`-Action noise-compare -Comparison <absolute-prior-masks-comparison.json>` runs
the noise-aware mask comparison on exactly the prior patch boxes, then
checks 512 inputs from 1,024 uniform candidates. It is CPU-only and does not
change training settings. The linked noise-aware note records the mixed small
pilot results and reproduction command. The user-run six-neuron comparison is
complete: repeated-intensity fish images expose a degenerate noise-estimate
failure and selected-input coverage worsens overall. No alternative is promoted.

`-Action sampling-cache` prepares pools for all train/validation neurons, preserving
test. `-Action sampling-train -Manifest <absolute-manifest-path>` trains the sampler
with its saved cache configuration. The trainer also accepts `--strategy random`
for the matched control. Full runs need fresh output directories and local runtime.

You can also invoke the Python entry scripts directly from any working directory.
The old root-level PN 2.0 script names remain thin compatibility entry points.
Historical result files retain the original script paths/hashes recorded when
those experiments ran.

## Completed: neuron 79 reconstruction control

`compare_hybrid_swc.py` prepares, checks and runs the approved score-versus-fixed
hybrid SWC comparison. `aggregate_with_features.py` preserves actual encoder
features for every source proposal using an equivalence-checked aggregation
snapshot. Preparation is CPU-only; the GPU run is launched by the user.
`summarize_hybrid_swc.py` verifies the saved pair and measures support changes
on CPU, then exports SWC projections. Spatial F1 improved from 0.6560 to 0.6862
on this selected validation case; no selector is promoted.
See [the results](../../docs/pointneuron2/hybrid_swc_results.md) and
[the protocol and command](../../docs/pointneuron2/hybrid_swc_comparison.md).

`audit_hybrid_graph_paths.py` inspects fixed graph anchors, saved paths and the
source image on CPU. `probe_component_allocation.py` reproduces eligible-component
starvation in synthetic cases using the unchanged production selector. Neither
changes production settings or reruns geodesics. See the
[image-backed audit and limits](../../docs/pointneuron2/hybrid_graph_path_audit.md).

`record_graph_components.py` is prepared to record actual allocation during one
unchanged CPU graph replay. It keeps the 625-candidate pool and 128-anchor budget,
records actual selector calls and requires identical graph arrays/settings.
See [the protocol and user command](../../docs/pointneuron2/graph_component_recording.md).
