# PointNeuron 2.0 development

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
