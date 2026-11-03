# PN2 research context and handoff — 3 November 2026

**Later update:** the CPU curve audit is now complete. Begin with the
[next-chat handoff](next_chat_handoff_20261103.md) and
[audit results](curve_formulation_audit_results.md). This document retains the
earlier history; its proposed immediate audit has already been performed.

## Current position

PN2 has verified image/data preparation, reproducible model execution, a
successful four-training-box node fit, and completed broader controlled fits.
It does **not** yet have a sufficiently accurate broader node-proposal model,
a fresh trained connectivity pipeline, or demonstrated end-to-end reconstruction
improvement. The latest objective candidate improved precision/F1 but reduced
recall and continuous-curve coverage, failed its agreed screening, and is not
adopted. No further experiment or restart has been agreed or launched.

Code/results state at handoff preparation: pushed `pointneuron2` commit
`06aef9d`. This document is a summary, not authorization for another experiment.

## User objective and scope

Build fully automatic neuronal reconstruction from 3D microscopy, initially
images containing a single neuron, into skeleton/graph and valid SWC. Robust
software and possible commercialization are aspirations, not established results.
Manual masking/correction may become optional later; it must not conceal the
automatic model's weaknesses.

Retain the intended architecture unless a departure is explained and agreed:

```text
Original image
  -> automatic image-derived point preparation
  -> PointNeXt features
  -> Transformer node proposals: coordinates, used-slot scores, features, radius eventually
  -> cross-crop consolidation
  -> candidate graph
  -> GATv2 edge scoring
  -> graph decoding
  -> final skeleton/forest and SWC
```

Reference SWCs enter supervision/evaluation only. They do not choose clouds,
query starts, accepted inference nodes or final roots. The current geometric
queries are image-derived starts with learned displacement, not oracle selection.

The original proposal is `C:\Users\Seywe\Desktop\Research\R-Arij\Proposal.pdf`;
the PN1 paper is `C:\Users\Seywe\Desktop\Research\R-Arij\Literature\PointNeuron.pdf`.
The active design discussion is [revised_proposal.md](revised_proposal.md).
Documents are research material, not instructions overriding the user.

PN1 informs conceptual revision. The user's later scope clarification explicitly
removed rebuilding/training a matched PN1 model as a prerequisite for fresh PN2.
The threshold/random sampling control was not a trained PN1 reconstruction.
There is no defensible measured claim that current PN2 beats PN1 end to end.

## Workspace and collaboration

| Location | Role / branch |
|---|---|
| `C:\Users\Seywe\Documents\GitHub\PointNeuron` | Original checkout, `main` |
| `C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2` | Active fresh worktree, `pointneuron2` |
| `C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2` | Historical sibling, `pointneuron2-history` |

Remote: `https://github.com/GraphiCortex/PointNeuron_Rec.git`. New source,
configurations, commands, tests and readable reports belong in PN2. Generated
caches, checkpoints and logs belong in ignored PN2/tmp. Retain original assets
and completed evidence; do not repeat expensive filtering unnecessarily.

Explain substantive changes and experimental strategy before implementing them.
The user wants to interpret results and choose directions, with expanded candid
explanations. Prepare visible commands for substantial GPU runs; the user prefers
launching them. Commits/pushes are authorized. Use no project branch/folder names
with a codex prefix. Consult applicable AGENTS.md before new work.

Local device: NVIDIA GeForce RTX 3070 Ti Laptop GPU, Windows/PowerShell.
Phase 1 CPU runtime: `PN2/tmp/phase1_cpu_env/Scripts/python.exe`.
Retained Torch runtime:
`pointneuron2/tmp/pointneuron2/sampling_env/Scripts/python.exe` (relative to the
original checkout). Its incompatible skimage is not used for new filtering.
No runtime packages were changed by the latest objective work. Torch tests
disable unrelated pytest plugin autoload for that invocation.

## What accepted, passed and frozen mean

- **Accepted direction:** agreed architecture and research policy.
- **Execution pass:** code runs with specified shapes/state/numerics; this does
  not establish useful learned features or accurate reconstruction.
- **Learnability pass:** fits a named training task; not new-neuron performance.
- **Frozen comparison:** settings/artifacts are kept immutable to interpret an
  experiment; freezing is not product acceptance.
- **Reconstruction improvement:** requires evaluated final geometry/connectivity,
  not successful execution, loss decrease or node-only F1.

| Component / decision | Accepted or verified | Still unestablished |
|---|---|---|
| Automatic, raw-first architecture | Accepted research direction | Overall effectiveness |
| Image decoding, XYZ/crop/intensity interface | Verified for declared sources/fixtures | All future formats, calibration and acquisitions |
| Image-only Sato sampling/cache | Working implementation and reusable verified artifacts; held fixed in recent comparisons | Universal sampler superiority or complete branch retention |
| Disk-backed guide and larger processing blocks | Output parity on tested cases; used for cache generation | Universal production throughput |
| Compact interpolation-index correction | Exact tested output/gradient parity; retained in prototype | Broader feature/reconstruction quality |
| GN8 encoder normalization | Working policy with tested exact train/eval consistency | Optimal normalization for all datasets |
| Geometric-query initialization | Accepted working formulation; passes four-box fit | Broad node accuracy |
| Native PointNeXt/Transformer | Verified execution, gradients and state restoration | Official benchmark reproduction or validated generalization |
| 2048 points | Working baseline used for controlled continuation | Universal optimal/sufficient budget |
| 8192 points | Executed and compared fairly for 20 epochs | Benefit under convergence or another acquisition |
| 512 queries | Frozen current capacity; covers current target counts | Universal node capacity |
| Eight-voxel whole-chain targets | Reproducible cached candidate | Local identifiability and best skeleton representation |
| Original objective | Preserved baseline, no broad accuracy pass | Suitable final objective |
| Consistent objective | Math/CPU/GPU execution verified | Failed learned comparison; not adopted |
| Fresh GATv2 / merging / decoding | Intended stages | Not qualified as a fresh end-to-end system |

No entire learned stage is certified as finished for broad software use. In
particular, "PointNeXt executes correctly" does not mean all remaining problems
are in the Transformer, and "Phase 1 held fixed" does not mean lossless sampling.

## Historical work before the fresh PN2 start

The historical sibling contains sampling/coordinate/anchor repairs, learned
image-field experiments, native PointNeXt/Transformer learnability controls,
BatchNorm/confidence investigations, oracle-node GAT and joint-training controls,
and target-canonicalization work. Those gave useful local findings but no matched,
convincing end-to-end improvement. Oracle-selected graph nodes used reference
information to isolate connectivity and were never automatic inference evidence.

The fresh PN2 effort restarted from original image files, not historical field
clouds or historical trained checkpoints. It explicitly reuses hash-checked
native encoder/head code as a documented Windows-compatible prototype. The
latest continuations use a checkpoint trained in fresh PN2, not the abandoned
historical model.

## Fresh image preparation and data

Current preparation uses whole-image affine intensity normalization and an
image-only Sato bright-ridge guide at sigmas 1, 2 and 4 native voxels. Weighted
sampling mixes 10% uniform **probability mass**, not a guaranteed 10% point quota.
Selections are distinct original voxel locations, without jitter or duplicate
padding. Guide information affects location selection; no Sato feature channel
is fed to the current encoder.

Current boxes are disjoint clipped 128-voxel boxes. They are not yet the planned
overlapping whole-neuron inference/merging solution. XYZ is centered on the
actual box midpoint and divided by scalar 64; edge boxes are not stretched.
Input attributes are float32 `[local_x, local_y, local_z, normalized_intensity]`.
Native XYZ/raw intensity/source indices remain preserved in packets.

Uniform, intensity and Sato were compared at matched counts and fixed seeds.
Sato helped strongly in some contexts, especially frog, but the four-context
qualification did **not** pass all prespecified checks. It is a working candidate,
not an established universally superior replacement for intensity sampling.
Spatial proximity to a reference curve is a retention proxy, not proof of
retained contrast, foreground correctness or a model's achievable prediction ceiling.

An early repeated-value accounting found only 13/135 nonconstant boxes fitting
all pixel-value exceptions into a dense point quota. That was a capacity result
about exact information storage, not evidence against PointNeXt or a reason by
itself to redesign Phase 1. The encoding itself round-tripped the tested boxes.

Disk-backed filtering was implemented to control memory. Larger processing
blocks produced byte-identical guide/packet results in the tested comparison.
Processing-block size is distinct from model-box size and point budget.

The user completed cache preparation: **40 images / 2379 boxes**. Its observed
bulk command took 1d5h31m47s, including an interruption; this is not pure compute
time. Guides are retained, so 8192 packets were generated without rerunning Sato.

| Partition | Neurons | All boxes | Reference present | Reference empty/unknown |
|---|---:|---:|---:|---:|
| Training | 33 | 1964 | 460 | 1504 |
| Development | 7 | 415 | 151 | 264 |

All 2048 packets and the nested 8192 view have been checked for source identity,
coordinates, intensities and distinct counts. One clipped development box has
7650 source voxels and retains all of them without padding. The 8192 clouds
include the original 2048 locations with unchanged shared attributes.

Zero-guide boxes use uniform fallback. There are 669 such boxes; their recorded
raw boxes are constant intensity, and none contains local reference targets.
That does not certify them as biological background or justify new negatives.

The seven development neurons have been repeatedly inspected. Eight reserved
fresh-PN2 identities remain outside this work, but historical PN1 exposure means
they are not a common untouched PN1/PN2 test. Exact duplicate-image checks passed;
independent acquisitions and external generalization remain unestablished.

## Current model, targets and metrics

The native PointNeXt-S prototype produces aligned 32-channel per-input features.
Four abstraction stages downsample by four, with up to 32 neighbors and native
voxel radii 6.4/12.8/25.6/51.2 under the current coordinate scale. Features are
restored to all input positions through interpolation and fine-level skips.
Deepest location counts are 8 for 2048 inputs and 32 for 8192; these are **not**
the Transformer memory counts, which remain N after restoration.

A retained interpolation view previously kept excessive index backing storage.
Compact index storage passed exact CPU neighbor/output/gradient parity, including
ties/short supports and a whole-encoder comparison. A user-launched untrained
65536-point encoder forward/backward replay then passed. This is not a trained
full-model 64k result or proof of deployment on smaller hardware.

Sixteen encoder BatchNorm layers were replaced by GroupNorm with eight groups
in a separately tested fresh variant. The preceding BatchNorm fits had a large
training-versus-running-stat inference discrepancy. GN resolved the tested mode
gap but did not alone solve node learning.

The three-layer Transformer uses 512 queries. FPS chooses image-cloud starting
positions; each query combines a learned token with the encoded feature and
position at its start. Every input feature remains in attention memory. Predicted
centers are starts plus learned **unbounded** offsets scaled by 64. Starts are
not final selected skeleton nodes. The current model has 1,688,518 parameters.

Targets preserve roots/leaves/junctions, crop-boundary intersections, distinct
coincident identities and interior samples every eight voxels along whole SWC
chains. Current maximum training count is 419, below 512; overflow errors rather
than truncating. Current node loss trains centers and used/unused slots only.
Radius output exists but is unsupervised; no anatomical radius claim is valid.

Unknown reference-empty boxes receive no negative-only gradient training. They
remain in full-grid evaluation. Unused-slot labels describe the reference node
representation, not guaranteed absence of real visible neurites.

Metrics use one-to-one node agreement within two voxels (primary) and six
(diagnostic), plus continuous-curve support, target-grid support baseline,
structural roles, duplicates, outputs in unknown boxes and apparent thin/dim
strata. Counts pool within each neuron before macro averaging. They are not
graph connectivity or final SWC metrics. Native voxels are not calibrated
physical distances across acquisitions. Current radii/type metadata require care.

## Completed training sequence

1. **Threshold/random versus Sato, learned queries and BatchNorm:** both failed
   four-box node learnability. A train/inference normalization gap was established;
   this was not a PN1-model-versus-PN2 reconstruction comparison.
2. **GN8 normalization-only control:** train/eval consistency passed, but the
   same four-box node gate still failed. Normalization was a real issue, not the
   complete explanation.
3. **Geometric-query control:** passed the original gate on four training boxes:
   fruitfly 32/27 and 32/20, frog 22/2 and 22/5, targets 18/85/7/68. At 2000 updates
   (500 exposures per box), all **178/178** targets matched within two voxels with
   **zero accepted extras**, cutoff .5, ordinary inference. This is tiny-set fitting,
   not new-image performance or trained connectivity.
4. **Broader point-budget comparison:** fresh models at 2048/8192, same seed,
   targets/order/objective/optimizer/cutoff; each had 20 epochs / 9200 updates over
   460 supervised boxes. Full-grid development at epochs 0/5/10/20; all training
   boxes scored terminally. Both completed; both accuracy gates failed.
5. **Objective candidate:** implemented separately after read-only audit and
   explanation. CPU mathematical/gradient checks passed; manual GPU timing and
   exact model/Adam/RNG reload passed. Smaller gradients were execution evidence.
6. **Matched five-epoch continuation:** both objectives forked identical saved
   2048 epoch-20 weights, Adam moments/clocks and RNG; same 2300 new updates per
   arm, history 11500, same full terminal grids. Completed in 3h15m43s. Candidate
   improved precision/F1 but lost recall/coverage and failed all relative criteria.

20 epochs was a bounded compute choice, not demonstrated convergence. The
four-box success used 500 exposures per box, not 512 epochs; 512 is query count.
Matching that exposure broadly would require 230000 updates per arm, 25 times
the 20-epoch budget. Shared learning means 500 is not established as necessary.

### Broader 20-epoch result

| Mean within two voxels | 2048 points | 8192 points |
|---|---:|---:|
| Training reference-present precision | 7.65% | 7.44% |
| Training recall | 29.89% | 21.97% |
| Development full-grid precision | 3.19% | 3.00% |
| Development recall | 25.68% | 18.42% |
| Development F1 | 5.40% | 4.90% |

The completed pair took 20h22m50s observed, above the earlier provisional
10–14-hour estimate. Increasing input count was insufficient under this
schedule; neither optimal budget nor converged performance is established.

### Latest paired continuation

| Mean within two voxels | Original objective +5 epochs | Candidate +5 epochs |
|---|---:|---:|
| Training precision | 9.91% | 19.71% |
| Training recall | 32.51% | 18.17% |
| Development precision | 4.66% | 15.88% |
| Development recall | 24.49% | 15.34% |
| Development F1 | 7.39% | 14.80% |
| Development continuous-curve support | 27.20% | 17.94% |

Training precision is reference-present-only; development precision uses the
whole grid. Do not compare those two precision domains as if identical.

Original loss averages positive and negative CE groups separately, uses
voxel-unit matched Huber geometry and approximate matching cost. Candidate uses
equal per-query CE, dimensionless geometry in the existing six-voxel unit and
Hungarian assignment minimizing that same objective. It adds no head, count
penalty, threshold tuning or changed target. The three related objective changes
are bundled, so their individual effects are not isolated.

The candidate reduced clipped updates from 2300/2300 to 165/2300 but did not
improve raw proposal recovery. Across development it accepted 6209 rather than
39542 proposals, matching 1086 rather than 1606 reference nodes. Reduced excess
predictions and higher F1 do not meet the agreed requirement to preserve recall
and continuous-curve support. Both arms also fail original absolute gates.

67 CPU tests covered the latest implementation/state/evaluator/controller
contracts. Tests, exact reload and verified artifacts establish execution,
not root-cause identification or accurate learned reconstruction.

## The unresolved problem and evidence boundaries

**Observed:** the model does not yet learn sufficiently accurate/selective
node sets across the broader training cohort. This is not solely generalization
failure. The candidate additionally rejects many useful proposals at .5.

With confidence filtering removed, latest development node recall is only
27.70% baseline and 26.76% candidate. Training all-query recall is 35.00% and
32.21%. These are limits of threshold-only changes to these saved centers,
not limits on what the architecture or source images could achieve.

The cause has **not** been established. Leading hypotheses and their limits:

- **Insufficient training / difficult optimization:** baseline extra training
  modestly improved training precision/recall, and loss still decreases. This
  does not prove that hundreds of epochs will succeed or explain every error.
- **Target identifiability:** a constructed example proves identical local curves
  can inherit different target nodes/counts from full-chain sampling outside
  the crop. Its prevalence/contribution in this cohort is unmeasured. Current
  curve-to-node support also depends on spacing, so its low value alone does not
  distinguish target ambiguity from genuinely missing branch geometry.
- **Loss/assignment/cardinality behavior:** class weighting changes with target
  count in the baseline; dense-box recall is worse, and sparse boxes have excess
  outputs. The candidate improves rejection but loses coverage. Neither loss is
  validated as the final formulation.
- **Encoder preservation / query use:** execution and tiny fitting do not prove
  that thin/dim evidence survives the hierarchy or is used effectively. More
  input points did not by itself fix this under 20 epochs.
- **Annotation and acquisition variability:** references may be incomplete,
  native scales/contrast differ, and some graph encodings are fragmented.
  These can complicate learning; they are not established global causes.

## What remains to solve

### Immediate research decision

Use existing saved predictions, references and image views to distinguish
off-curve placement from along-curve sampling disagreement, missing branches
from redundant nodes, and score rejection from absent raw proposals. Interpret
this by training/development case, structural role and apparent caliber/contrast.
Then agree one justified intervention and its comparison before implementation.
No automatic threshold change, target rewrite, larger-budget sweep, long
extension, architecture bypass or Phase 1 restart follows from the current data.

The goal is useful geometry on broader inputs, not just lower loss, fewer
clipped gradients or a tiny-set gate. A relative gain between weak models is
not completion of the proposal stage.

Subsequent [formulation-review clarification](formulation_review_20261103.md):
the current curve-to-node support metric is spacing dependent; it does not
evaluate predicted edges. The reference target grid itself scores 55.11% mean
development support at two voxels. This limits interpretation of low support
as evidence of missing branches and strengthens the need to distinguish exact
node matching, on-curve positions, spatial distribution and final polyline quality.

### Remaining system responsibilities

1. Establish defensible geometry supervision and sufficient broader fitting;
   quantify accuracy/coverage and the chosen operating policy.
2. Decide deployment tiling, overlap/context, boundary handling and automatic
   consolidation without reference counts or reference-based node selection.
3. Construct a candidate graph with measured true-edge coverage and manageable
   cost on actual imperfect automatic proposals.
4. Resolve connectivity-label semantics. Three training SWCs have many encoded
   components and shared coordinates; coincidence does not license welding,
   and disconnected IDs are not automatically known negative edges.
5. Train/evaluate fresh GATv2 on realistic proposals, distinguishing missing
   nodes, missing candidate edges and wrong edge decisions. Historical oracle
   GAT fitting does not satisfy this.
6. Implement/qualify decoding, fragments, wrong joins, gaps/cycles/root handling,
   radius supervision/estimation and SWC serialization in consistent coordinates.
7. Evaluate complete automatic reconstructions at whole-neuron level with
   geometry, branch retention and connectivity errors, then obtain genuinely
   new external/acquisition evidence before broad-use claims.
8. Qualify inference runtime, memory, formats, calibration and failure behavior.
   Cached training preparation is reusable, but image filtering is also an
   inference cost for new images; it is not exclusively a training concern.

## Most useful files and artifacts

- [Active proposal](revised_proposal.md)
- [Four-context sampling evidence](phase1_qualification_results.md)
- [Input/encoder contract](pointnext_input_contract.md)
- [Compact 64k execution replay](pointnext_compact_replay64.md)
- [Completed image cache](node_cloud_cache_completion.md)
- [Passed four-box fit](geometric_query_fit_results.md)
- [Broader 20-epoch comparison](node_budget_fit_results.md)
- [Candidate objective design and limitations](consistent_node_objective.md)
- [Latest continuation protocol](consistent_node_fit_ready.md)
- [Latest continuation results](consistent_node_fit_results.md)
- [Reference graph/radius ambiguity audit](training_reference_contract_audit.md)

Frozen artifacts under PN2/tmp:

```text
node_cloud_cache_20261031/                 image-only guides and 2048 clouds
node_budget_data_20261102/                 shared targets and paired cloud views
node_budget_profiles_20261102/             fixed training-defined profile metadata
geometric_query_train_20261031/            successful four-box fit
node_budget_fit_train_20261102_141941/      completed fresh 20-epoch 2048/8192 arms
consistent_objective_timing_20261103_143451/ passed throwaway objective GPU timing
consistent_node_fit_train_20261103_155059/  latest completed paired continuation
```

The original-objective epoch-20 checkpoint is the shared origin. The two
stage-2300 checkpoints are controlled results, not production models. Keep their
full model/Adam/RNG state and immutable reports; do not silently resume weights
with a new optimizer or overwrite historical evidence.
