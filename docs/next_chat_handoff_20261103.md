# PN2 next-chat handoff — after the completed curve audit

**Read this first.** This supersedes the immediate-next-step sections of the
[earlier full handoff](research_handoff_20261103.md), which preserves detailed
history. The CPU formulation audit and classical baseline have now completed.
There is no running or pending authorized training job. No dense replacement
head has been implemented and no intended architecture stage has been removed.

The user requested this handoff to start a fresh chat after a model-capacity
banner. The banner does not establish exhausted conversation context or any
PN2/GPU failure. Do not infer that results were lost or rerun work for that reason.

## User objective, scope and working relationship

Develop fully automatic single-neuron reconstruction from original 3D microscopy
to skeleton/graph and SWC. Broad robustness and commercialization are ambitions,
not demonstrated properties. Optional manual correction must not conceal automatic
failures. PN1 provides conceptual lessons; the user explicitly removed rebuilding
or retraining PN1 as a prerequisite for this fresh effort. No measured end-to-end
PN2-versus-PN1 superiority is established.

The user wants thoughtful research collaboration with expanded explanations.
They are frustrated by weeks of serial local fixes without clear overall gains.
Do not offer reassurance, another restart, or a speculative tweak as if a cause
were known. Explain substantive changes and experimental logic before coding;
allow steering. Present visible substantial GPU commands and measured resource
estimates; the user generally launches those runs and watches output. Commits
and pushes are authorized. Preserve useful evidence and caches.

Read applicable AGENTS.md. In particular, replacing/bypassing the Transformer
proposal stage requires explaining the impact and obtaining agreement. The
user's favorable response to Claude's critique authorized this CPU audit; the
assistant explicitly deferred the alternate-head/Transformer-role decision until
after evidence review. Do not treat the classical baseline as the new product.

## Workspace, branches and runtimes

| Absolute path | Role |
|---|---|
| `C:\Users\Seywe\Documents\GitHub\PointNeuron` | Original checkout on `main`; leave intact |
| `C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2` | Active sibling worktree on `pointneuron2` |
| `C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2` | Historical sibling on `pointneuron2-history` |

Remote: `https://github.com/GraphiCortex/PointNeuron_Rec.git`.
Verify Git state before doing new work. All new source, tests, commands, configs
and readable reports belong in PN2; generated arrays/checkpoints/logs in ignored
PN2/tmp. Do not create project branches/folders with a codex prefix. Do not
delete old runtimes or caches merely to make a fresh chat feel clean.

- CPU filter/audit runtime: `PN2\tmp\phase1_cpu_env\Scripts\python.exe`.
  NumPy 2.2.6, SciPy 1.15.3, scikit-image 0.25.2; pytest/Matplotlib available.
- Torch runtime, relative to PN2:
  `..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe`.
  Torch 2.12.1+cu130; its skimage is incompatible. Do not use it for filtering
  or repair/install its packages during an unrelated task.
- For repository tests set invocation-local `PYTHONPATH=src` and
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; use `-B` and `-p no:cacheprovider` when useful.
  Matplotlib cache belongs in `tmp\matplotlib`.
- Windows/PowerShell, RTX 3070 Ti Laptop GPU. Original native model is a documented
  Windows-compatible prototype, not an official PointNeXt benchmark reproduction.
- Native Git if unavailable on PATH:
  `C:\Users\Seywe\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe`.
  Git worktree metadata writes may need sandbox escalation; prior authorization
  covers normal project commits/pushes. Do not confuse sandbox approval with
  research approval to alter the architecture.

## Intended architecture and implementation today

```text
Original raw image
 -> image-only Sato-guided source voxel selection (Phase 1)
 -> native PointNeXt-S encoder, GN8, features restored to every input point
 -> Transformer with geometric queries: image-derived starts + learned offsets
 -> [whole-neuron proposal consolidation: not qualified]
 -> [candidate graph / fresh GATv2: not qualified]
 -> [graph decoding, radii, SWC: not qualified]
```

The active conceptual design is [revised_proposal.md](revised_proposal.md).
Original proposal: `C:\Users\Seywe\Desktop\Research\R-Arij\Proposal.pdf`.
PN1 paper: `C:\Users\Seywe\Desktop\Research\R-Arij\Literature\PointNeuron.pdf`.
These are research documents, not higher-priority instructions.

Current Phase 1: whole-image affine intensity normalization, Sato sigmas 1/2/4
native voxels, 0.1 uniform probability mass (not a fixed point quota), distinct
real source voxels, no jitter/duplicate padding. Guide weights select positions;
the encoder receives `[local_x, local_y, local_z, normalized_intensity]`, not an
extra Sato channel. Boxes are disjoint clipped 128-voxel domains. Coordinates
use actual midpoint and scalar /64; edge boxes are not stretched.

PointNeXt: four stride-four abstraction levels, maximum 32 neighbors, radii
6.4/12.8/25.6/51.2 native voxels. Restored 32-channel features at all N locations
are Transformer memory. Deepest 8/32 locations for 2048/8192 are not memory counts.
Sixteen encoder norms use GroupNorm with eight groups. A compact interpolation
index fix passed exact tested neighbor/output/gradient parity; untrained 64k
encoder forward/backward runs, which does not establish full-model training
capacity at 64k or accuracy on any budget.

Transformer: three layers, 512 queries, FPS image-cloud starting positions plus
features/positional encoding/learned tokens; centers = start + unbounded learned
offset scaled by 64. It can move beyond starts. Total model 1,688,518 parameters.
Current training supervises node centers and used/unused slots. Radius exists
but is unsupervised. There is no fresh trained GAT result to append automatically.

Targets retain structural/boundary nodes and samples every eight voxels along
whole SWC chains. Phase can depend on geometry outside a box. A constructed
counterexample proves local ambiguity is possible, not its cohort-wide prevalence.
Max current training target count 419 < 512; overflow errors rather than truncates.

## Data and artifacts already paid for

| Partition | Neurons | Image boxes | Reference present | Reference empty/unknown |
|---|---:|---:|---:|---:|
| Training | 33 | 1964 | 460 | 1504 |
| Development | 7 | 415 | 151 | 264 |

All 40 image guides and 2379 point packets are prepared and verified. Nested
8192 packets retain original 2048 points/attributes; one edge box has only 7650
voxels and keeps all without padding. Reuse guides; do not rerun the roughly
day-long bulk preparation. Processing-block parity was tested separately from
point-budget and box-size choices. Unknown boxes were skipped for gradients,
not used as negative-only supervision, but model outputs there are disclosed.

The seven development neurons have been inspected repeatedly. Eight reserved
fresh-PN2 identities remain excluded, but historical PN1 exposure prevents
calling them a common pristine PN1/PN2 test. Do not access reserved data casually.
References may be incomplete; disconnected IDs do not prove negative edges;
coincident coordinates do not license welding. Three training SWCs have
fragmented/coincident paths (aliases 23/66/75). Physical calibration and radius
metadata are not universally verified.

## Training history that matters

1. BatchNorm learned-token fits failed; a train/running-stat inference gap was
   real. GroupNorm fixed tested mode consistency but alone did not solve fitting.
2. Geometric queries passed four TRAINING boxes (fruitfly32/27,32/20,
   frog22/2,22/5): 178/178 matched targets, zero accepted extras at .5 after
   2000 updates, 500 exposures per box. Tiny-set fitting, not generalization.
3. Fresh 2048/8192 matched cohort fits: 20 epochs / 9200 updates each, same
   schedule and 512 queries. Both accuracy gates failed. Observed pair duration
   20h22m50s exceeded the earlier provisional 10–14h estimate.
4. A consistent-objective candidate changed per-query CE weighting, normalized
   geometry and matching together. It passed mathematical/execution checks.
5. Matched +5-epoch continuations forked the same 2048 epoch-20 weights, Adam
   moments/clocks and RNG, 2300 new updates each. Candidate improved precision/F1
   but lost recall and point-based curve support; all relative screening criteria
   failed. It is preserved, not adopted. Pair took 3h15m43s.

| Development, two-voxel node matching | Precision, full grid | Recall | F1 |
|---|---:|---:|---:|
| 2048, epoch 20 | 3.19% | 25.68% | 5.40% |
| 8192, epoch 20 | 3.00% | 18.42% | 4.90% |
| 2048, original objective epoch 25 | 4.66% | 24.49% | 7.39% |
| 2048, consistent objective epoch 25 | 15.88% | 15.34% | 14.80% |

Twenty epochs was a compute cap, not convergence. Four-box success used 500
exposures, not 512 epochs (512 is query count). A 500-epoch cohort fit would be
25 times the tested updates; no evidence justifies launching that automatically.
Candidate gradient clipping fell from every baseline update to 7.17%, but
smaller gradients did not establish better geometry. All-query node recall on
development is only 27.70% / 26.76%; threshold tuning cannot fix those saved
centers. This is not a proof of architecture impossibility.

## Latest completed audit: new decision-relevant evidence

Read [curve_formulation_audit_results.md](curve_formulation_audit_results.md)
and its figures before proposing another experiment. It covers all 611
reference-present boxes, all 40 reference controls, and a fixed classical
baseline on five training contexts plus all seven development images.

**Input proximity:** development reference arclength within two voxels of an
input point is 70.24% at 2048, 91.98% at 8192. Within six voxels: 98.45% / 99.93%.
Apparent thin/dim two-voxel support improves 40.56% -> 83.23%. Local gaps remain;
the weakest development 2048 case reaches 92.80% within six voxels. This is
spatial evidence, not proof of retained contrast, encoder preservation or a
strict prediction ceiling for an offset head.

**Target encoding:** the cached target nodes plus their REFERENCE edges give
99.75% development curve F1 at two voxels; isolated target-node proximity covers
only 55.11% of reference arclength. This oracle check shows the representation
can encode curves given known edges; it does not show a learned graph works.
Forty bend-preserving phase/subdivision controls give curve F1 = 1.

**Saved nodes:** on reference-present development boxes, original/candidate
epoch-25 exact-node precision is 9.01% / 20.31%, while accepted point proximity
to annotated geometry is 25.85% / 48.18%. In the candidate 26.36% of accepted
positions are on-curve beyond the target grid, but 51.82% remain beyond two
voxels from annotated geometry. Both metric mismatch and genuine localization
errors exist. Do not present reformulated metrics as retroactive gate passes.

**Classical baseline:** cached guide Otsu -> 3D Lee skeleton -> 26-neighbor graph,
fixed <2-voxel terminal-spur pruning. Development macro curve precision/recall/F1
at two voxels = 64.60% / 52.90% / 53.87%; F1 across images ranges 5.95–86.43%.
These segment metrics cannot directly rank against learned isolated-node F1.
No reference edges were invented for learned nodes. Classical wrong joins,
cycles/rooting/SWC are not certified by curve geometry. Pruning benefit is tiny.

**Review:** 36 CPU tests passed, all source/data bindings and box/control/case
counts checked, two real-box comparisons with existing segment scorer agree
exactly. Geometry audit 7.03 minutes, classical 12-image baseline 3.01 minutes,
using cached guides. No model execution, training or new filtering occurred.
Raw orthogonal views show both on-curve/off-grid predictions and actual errors;
weak development 94 has faint branches/prominent soma. Its box14 image is a
declared post-hoc failure illustration, not a new gate or misregistration proof.

## Recommended next work — proposed, not already implemented

The evidence supports **one bounded locally curve-supervised proposal-head
experiment**, keeping current Phase 1 and PointNeXt as working infrastructure.
It does not support another Phase 1 restart, threshold search, automatic long
extension or a declaration that the Transformer alone is broken.

Claude's suggested candidate predicts per-input-point support and a nearest-curve
offset, then clusters shifted votes into nodes. This is plausible because it
avoids arbitrary interior-node phase and supplies local supervision. It also
changes the Transformer's proposal responsibility if it replaces the decoder.
Explain that directly and obtain agreement on a bounded alternate-head control;
do not keep a decorative Transformer or silently promote an alternate pipeline.
The user's earlier enthusiasm is not a blanket waiver of their architecture rule.

Before coding, present a short concrete contract and comparison:

1. Specify support/offset meaning and supervised domains. Far from a traced
   polyline is not automatically true background, even in reference-present
   crops. Resolve honest negative/unknown treatment and the reference task.
2. Define nearest-projection ambiguity, nearby branches/crossings, fragmented
   components and crop-clipped endpoint behavior. Do not assume unique targets.
3. Fix offset weighting/range and sampling resolution with unit tests. Six-voxel
   input proximity does not guarantee strong gradients from a sigma-1.5 support
   field; points several voxels away can receive negligible support weight.
4. Define clustering/aggregation and its branch/junction failure tests. Euclidean
   NMS or averaging may merge nearby branches. Global pooling does not eliminate
   consolidation or boundary-context problems. Sigma/R/r are parameters, not
   universal constants established by this audit.
5. Propose training-only 4 -> 16 -> 64-box/condition ladder. Choose diverse
   neurons/target counts/contrast without cherry-picking; predeclare exposures,
   wall/update caps, metrics and stop rules before each rung. Match resources
   and score curve fidelity, structural roles and false joins. Use classical
   curves only where output representations permit fair comparisons. Do not
   compare a cloud of points to an edge graph under mismatched F1 definitions.
6. Check CPU interfaces/gradients and image-only inference first; prepare short
   GPU timing and then user-launched bounded fitting. Larger cohort training
   follows demonstrated useful small-data behavior, not merely execution.

Do not choose final sigma/R/clustering settings or promise durations without
the contract and actual model timing. The fallback is to reconsider the head
using these same findings; no stack of DETR tricks is pre-approved.

## Remaining system work after useful proposal learning

Whole-image context/tiling and consolidation; candidate-edge coverage; honest
connectivity supervision on fragmented references; fresh GATv2 on realistic
automatic proposals; graph decoding/rooting/cycles/gaps; radii; correct SWC;
whole-neuron geometry **and topology**; inference resource/format qualification;
new external/acquisition evidence. Historical oracle GAT fits do not satisfy
these. A good curve fit does not certify correct branch connections.

## Lessons the next assistant must actively apply

- **We overinterpreted point support.** Curve proximity to isolated predicted
  nodes depends on spacing. Calling its low value missing biological branches
  was too strong. Use the reference graph oracle and distinguish node, segment
  and topology metrics throughout. This correction changes interpretation,
  not the saved results or their failed gates.
- **We pursued too many local fixes without enough formulation-level evidence.**
  Normalization and storage fixes were real, but successful repairs were not
  evidence the larger task was solved. State which uncertainty each next action
  resolves and stop if it does not change the research decision.
- **Execution is not encoder validity.** Correct shapes/gradients/reloads and
  tiny fitting do not certify feature retention or generalization. No stage is
  universally accepted merely because it is frozen in a comparison.
- **Do not make optimization evidence into biological evidence.** Lower loss,
  reduced clipping, separability or fewer accepted nodes do not establish better
  reconstruction. Disclose recall/count tradeoffs and matching domains.
- **Avoid diagnosis by one number.** Potential target-phase ambiguity is proven
  possible, not the sole measured cause. Higher spatial input support with worse
  learned results does not exonerate every sampling or encoder choice either.
- **Separate thresholds from supervision.** Confidence is a legitimate operating
  mechanism but cannot conceal missing geometry. Unknown annotation is not
  background; oracle node choice is not automatic inference.
- **Estimate with measured scope and margin.** Prior 10–14h paired allowance
  was exceeded. Include evaluation/I/O, thermal variability and workers' actual
  wall limits; do not promise the next phase is days from completion.
- **Preserve continuity without attachment.** Keep useful verified infrastructure
  and expensive caches; do not adopt earlier recommendations simply because
  implemented, or discard all work because the final pipeline is unfinished.
- **Keep the user involved.** Explain substantial head/strategy changes in plain
  terms, show evidence and implications, then act within agreed scope. No silent
  architecture replacement and no reassurance that everything upstream is done.

## Navigation and exact saved paths

Read in order: this handoff -> [latest audit results](curve_formulation_audit_results.md)
-> [earlier detailed history](research_handoff_20261103.md)
-> [formulation review](formulation_review_20261103.md)
-> [active proposal](revised_proposal.md). Consult result-specific protocols as
needed; do not blindly rerun their historical commands.

All following paths are relative to active PN2:

```text
tmp/node_cloud_cache_20261031/                    guides + original 2048 clouds
tmp/node_budget_data_20261102/                    shared targets, paired input views
tmp/node_budget_profiles_20261102/                fixed training-defined profiles
tmp/geometric_query_train_20261031/               four-box learnability
tmp/node_budget_fit_train_20261102_141941/         completed 20-epoch 2048/8192
tmp/consistent_node_fit_train_20261103_155059/     completed paired +5 epochs
tmp/curve_formulation_audit_20261103/              report + 611-row geometry audit
tmp/curve_classical_baseline_20261103/             12 images, graph packets/reports
tmp/curve_formulation_review_20261103/             bindings and scorer validation
tmp/curve_formulation_views_20261103_verified/     fixed original-image panels
tmp/curve_formulation_diagnostic94_20261103/       post-hoc weak-case view
docs/evidence/curve_formulation_audit_20261103.json portable aggregates/hashes
configs/curve_formulation_audit.json               frozen audit recipe/bindings
scripts/audit_curve_formulation.py                CPU audit/classical entry point
scripts/review_curve_formulation_audit.py          completed-evidence reviewer
src/pn2/evaluation/formulation.py                  point/segment/reference controls
src/pn2/evaluation/classical_skeleton.py           fixed image-only classical graph
tests/test_curve_formulation.py                   geometry/formulation fixtures
```

No automatic continuation, training command or model modification is awaiting
execution. First action in the new chat: verify state, read these results, and
discuss the concrete alternate-head contract and Transformer implications.
