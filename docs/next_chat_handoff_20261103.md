# PN2 next-chat handoff — after failed dense-head fit and CPU graph measurement

**Latest direction, 5 November:** User approved a NEW experimental arm: no
Transformer node proposals; learn to select/connect strong image-derived
candidates using PointNeXt features and graph models. Explicit departure from
PDF Phase3, professor discussion pending. Main/product architecture not silently
replaced. Earlier dense-head failures remain on record and no further head
tuning is allowed. Read [active selection protocol](candidate_selection_protocol_20261105.md).

Stage0 prepared, NOT run: four development oracle arms classical-pruned,
saved BROAD-run nodes, original Sato-top512, and union; same151 boxes. Greedy
nontransitive radius1 real representatives keep source flags. Primary edges
k10+inherited (k6 is a subset; diagnostics select nothing). Probability-only
forest decoder, .5 node/edge thresholds, -log p plus endpoint-node costs,
one-pass leaf-to-junction spurs<2, image-only roots, placeholder radii. Explicit
geometric oracle uses reference geometry only after candidate packets are
saved; it is not real inference or a topology/learnability ceiling.

User's fixed Stage0 gate: union oracle macroF1_2>=.80 proceed;.65–.80 proceed
ONLY alongside candidate-recall work;<.65 stop. No timed audit or GPU job has
been launched by the assistant. User-local PowerShell command will use
`tmp/phase1_cpu_env/Scripts/python.exe -B scripts/measure_candidate_headroom.py`.
Source/input preflight and short fixtures only; actual headroom is pending.
Later-stage candidate labels/models wait for this gate. Raw inputs are
diagnostic only and not added to the pool.

User explicitly answered BOTH CV clarifications: use SELECTOR-ONLY leave-neuron-
out CV conditional on frozen upstream broad-trained encoder AND learned nodes.
Both used all33 training-neuron labels, so disclose leakage/conditioning; do not
call this strict full-pipeline CV. Final development models score once each,
no tuning. No assistant GPU launch; give inference/training commands as needed.
Trees first; strict stop-on-failed-criterion means a failed final tree gate stops
before GAT unless user later steers otherwise. GAT must gain>=2pp F1_2 over trees
AND clear matched classical+5pp at2/6,>=5/7 wins each; training metrics ungated.
Fragments/coincident projections masked unknown, trace-agreement not biological
foreground. Requested wrong joins must distinguish valid junction transitions,
unknown fragment relations and identifiable errors rather than relabeling the
old conservative shortcut fraction as exhaustive topology accuracy.

Proposal-to-implementation table updated. Earlier “nothing authorized” statements
below apply to the failed head; the new gated selector arm is now authorized.

**FINAL broad verdict, 5 November: FAILED; STOP tuning this head.** Read
[completed broad results](broad_curve_head_results_20261105.md) and
[full terminal evidence](evidence/broad_curve_terminal_20261105.json).
All9200 updates/20 epochs on460 boxes/33 neurons completed successfully,
terminal checkpoint only. Training F1_2/F1_6=37.84/73.28% (reported NOT gated).
Development F1_2/F1_6=36.04/67.54%, versus matched strongest classical
54.33/69.14%. Both+5pp mean gates fail; strict neuron wins2/7 at2 and3/7 at6
also fail the>=5 rule. No accuracy gate/threshold/cohort changed after training.

Broader training recovers recall/candidate geometry: development R2=74.40%,
k6 candidate R2=81.69% versus the four-box38.07% (Sato76.80/raw89.00%). P2
remains24.80%, with too much extra geometry. This is not merely another tiny
training/transfer gap: broad training F1_2 itself37.84%. Causal/asymptotic limits
remain open, but the precommitted practical decision is to stop, not extend.

Root `tmp/broad_curve_head_20261105`: train/checkpoint_step009200.pt and complete
marker (SHA256bc13efbaef2ab012b344e81b87dd4a445d9f85b4fddfb421ccf4f6b76c30cab9),
611 terminal predictions, evaluation/report.json/md,611 graph/SWC packets and
per-edge diagnostic records. All hashes/schedules/aggregate/gate checks passed;
64 Adam clocks9200, all460 boxes exactly20 exposures, input caches unchanged.
Train optimization1h46m; GPU stage incl inference1h54m; terminalCPU16min.
Both guards complete exit0. Unified session41968 and monitorcellg2:203 ended
successfully. NOTHING is running or pending launch. Earlier active-job notes
below are history; do not restart any command.

The PASS condition for assembly/graph-stage work was NOT triggered. No further
head tuning, Transformer hypothesis, radius/loss/cutoff change or GPU run is
authorized by this failed experiment. Useful source/labels/checkpoints/scorers
are retained as evidence, not adopted as a fresh model.

Prepared [a professor-facing framing note](professor_candidate_refinement_discussion_20261105.md)
for the user to share. Learned refinement/selection of strong image-derived
candidates would change the Transformer's proposal role and needs professor
agreement. It is NOT implemented or approved; no message was sent externally.

**Active execution, 5 November:** CPU diagnostics COMPLETE and ONE authorized
broad training pipeline LAUNCHED by the assistant. Read
[current broad results/status](broad_curve_head_results_20261105.md).
Root `tmp/broad_curve_head_20261105`. GPU train console/owner receipt in
`resident_train_guard`; terminal predictions under `train/terminal_predictions`.
The already-launched PowerShell pipeline runs CPU `--action score` automatically
after successful GPU completion. Evaluation final report will be
`evaluation/report.json`/report.md; guard `resident_score_guard`. Do not launch
another train/timing/score process while this one is active. Unified-exec
session41968 owns the train->score command; owned worker identity is logged in
`resident_train_guard/owner.json`. No final broad verdict yet.

CPU ablations: all raw votes F1_2=10.25%; learned-support top512 without offsets
24.82%; EXACT Sato source set +learned offsets24.47%; original Sato38.49%.
Both learned outputs lose to Sato on transfer; removing cutoff restores recall
but lowers F1. These select nothing. Matched classical remains strongest.
Preparation COMPLETE:460 labels (16 reused/444 generated),122.33s guard; frozen
20-epoch schedule9200 updates. Training metrics NEVER gated.

First streamed timing failed only the assistant's4h resource projection, with
NO real training updates. Preserve it. Separate resident execution wrapper
checks EXACT16-update loss/gradient parity and immutable460-batch inputs;
passed, p90=.929151s/update, VRAM170.52MB. Training estimate~2h22m;40min scoring
reserve;50% margin =>16440s overall4h34m. Disclosed execution addendum allows
<=6h under user's instruction to set budget from throughput. Labels/scientific
recipe/updates/gates did not change. Old sources/configs/timing preserved.
See [execution addendum](broad_curve_execution_addendum_20261105.md) and new
`configs/broad_curve_resident_execution_20261105.json`; base preparation binds
`configs/broad_curve_head_20261105.json`. Do not edit frozen runtime sources.

If interrupted, distinguish execution/resource failure from accuracy failure;
no automatic retry or extension. If terminal gate fails, STOP tuning this head.
If passes, whole-image assembly/intended graph stage follows with concrete
implementation discussion. No professor message or alternative node-refinement
architecture has been implemented. Earlier statuses below are history.

**Latest explicit authorization, 5 November:** User REJECTED the sixteen-box
draft. Run saved-packet CPU diagnostics first, then the assistant launches ONE
broad SAME-head run on all 460 reference-present training boxes/33 neurons.
This explicitly supersedes the no-GPU/user-launch preference for this specific
run; do not ask again for approval already given. Read
[the active broad-run protocol](broad_curve_head_protocol_20261105.md).
Twenty frozen dataset epochs =9200 updates, fresh original initialization,
unchanged 2048 inputs/labels/loss/optimizer/.25 cutoff/r1 consolidation/k6 forest.
Measure throughput before fixing the wall budget (<=4h); terminal checkpoint
only. Training metrics reported, NEVER gated. No sixteen-box ladder.

Strongest matched no-learning control is classical PRUNED, 54.3311257940% F1
at2 /69.1368664294% at6 on the SAME151 reference-present boxes. Historical
53.87/68.52% included unknown-box prediction length. Reaggregate existing
per-box metrics; do not repeat extraction/filtering. Broad PASS requires >=5pp
macro F1 gain at BOTH tolerances AND strict wins on>=5/7 neurons at EACH,
against the same strongest-macro fixed comparator for that tolerance. Floors
currently .5933112579402621/.7413686642936104. No training-fit floor.

CPU diagnostic `scripts/diagnose_broad_curve_controls.py` is running under
`tmp/broad_curve_diagnostics_20261105`, guard sibling. It compares all raw votes
(no cutoff/suppression), learned-support top512 UNMOVED source positions, and
the EXACT Sato source set moved by learned offsets (no second suppression).
It selects nothing and does not alter broad-run settings. Three short tests
pass; separate training-manifest test rejects altered/development schedules.
Only16 training label packets exist; reuse those and prepare444 with the
unchanged projection code before GPU timing. Dataset already has all460 images
and the complete20-epoch training-only schedule. No image filtering restart.

If broad passes: whole-image assembly and intended graph stage next. If broad
fails: STOP tuning this head. A future strong-classical-candidate refinement
framing changes the Transformer's role and needs professor agreement; do not
implement it or contact the professor as an implied action. Transformer remains
parked for this run. Original failures remain on record; no private gate changes.

**Latest completed measurement, 4 November:** User-local scoring continuation
finished all 453 development graphs in ~9.5min, after exact parity on 12 fit
graphs/48 metric dictionaries. Read
[development transfer results](curve_transfer_development_results_20261104.md)
and [portable evidence](evidence/curve_transfer_development_20261104.json).
Output: `tmp/curve_transfer_scoring_20261104T211240_017254Z` (report.json/md,
records, fit parity, binding; sibling guard receipt completed exit0).

Development segment P2/R2/F1: dense 18.59/33.28/23.20%; raw 6.64/77.94/12.00%;
Sato512 29.02/70.25/38.49%. Six-voxel F1: dense56.99/raw31.80/Sato63.13%.
Sato beats dense on all seven neurons at two voxels and six of seven at six.
Dense k=6 candidate-union R2=38.07%, versus raw89.00/Sato76.80%; fixed edge
selection cannot repair this upstream geometric opportunity loss. Input
density alone does not explain the additional learned-stage collapse, though
counts differ and causal mechanisms are unresolved. Tiny fitting succeeded;
transfer is poor under this budget, not a proof of asymptotic head incapacity.

Dense shortcut flags6/6956 scorable edges, with39860 off-reference-endpoint
edges (83.50% of its forest) excluded. Fragment bridges134 remain unknown;
do not interpret the tiny conditional flag fraction as topology certification.
All 453 score hashes/bindings/category counts and original frozen sources were
verified. No new model/graph/filter/GPU/training executed in the continuation.

Transformer stays parked. The [sixteen-box downstream discussion draft](curve_head_sixteen_downstream_discussion_20261104.md)
proposes a bounded fresh SAME-head diversity test, old training-only selection
(16 boxes/14 neurons), with concrete post-hoc/prospective downstream floors and
stopping rules. It is PENDING USER AGREEMENT, not an adopted protocol, trainer,
manifest or run command. Original failed gates stay failed. No GPU run is
authorized. Do not use `run_supported_vote_sixteen.py`: its r2/min3 variant
failed and is not the proposed consolidation. No old training command should
be repurposed without discussing and binding the new experiment.

Do not repeat inference, graph creation, completed curve/input audits or this
completed scoring job. Earlier pending/interruption notes below are history.

**Latest user outputs, 4 November:** Matched fit forest controls completed,
but development was user-interrupted after ~70min during unjournaled scoring.
Read [fit results and scoring-only continuation](curve_transfer_scoring_resume_20261104.md).
Fit segment P2/R2/F1: dense 76.16/87.64/81.49%; raw input 7.46/72.15/13.33%;
Sato-top512 23.81/46.07/31.36%. This demonstrates a learned fitting contribution,
not transfer. Shortcut flags are dense 6/1080, raw 3/355, Sato 2/283 scorable
edges; off-reference endpoints exclude 423/6803/1729 edges, so do not compare
these fractions as total biological wrong-join rates.

Preserve `tmp/curve_transfer_fit_20261104T193753_641182Z` and interrupted
`tmp/curve_transfer_development_20261104T195033_977395Z`. All 151 CPU predictions
and 453 graph/SWC exports are complete and hash-verified. All graphs were saved
by ~6min; the remaining ~hour was reference scoring with no durable per-graph
results. CPU/GPU fit replay passed (max coordinate error .00006104 voxel,
retained sets equal). There is NO development accuracy report yet.

Do NOT rerun `measure_curve_transfer.py --stage development`. A separate
`scripts/score_saved_curve_transfer.py` reuses every saved prediction/graph,
does scoring only, uses the same .5-voxel quadrature/2&6 decisions in a batched
bounded search, and first requires exact original fit metric parity on all
12 graphs/48 metric dictionaries. Five CPU tests pass. Real parity, speedup and
development scores remain pending; the assistant did not launch scoring.
User-local command: `tmp/phase1_cpu_env/Scripts/python.exe -B
scripts/score_saved_curve_transfer.py`. Fresh output under `tmp/curve_transfer_scoring_*`,
30min cap, progress and atomic score+shortcut JSON per graph, cancellation
receipt. No model, Torch import, inference, graph building, training or changed
gate. Explicit future --resume can reuse completed records but is never launched
automatically or used as an implicit cap extension. Transformer remains parked;
no sixteen-box manifest or new training is authorized.

**Latest user steering, 4 November:** Park the Transformer hypothesis. Before
any new training decision, compare no-learning points through the SAME forest,
measure frozen update-2000 development transfer with controls, surface the
already-completed two-voxel input/stratum results, and add reference-only
wrong-join diagnostics. See [ready commands and evidence](curve_transfer_diagnostics_ready_20261104.md)
and [new diagnostic protocol](curve_transfer_diagnostics_protocol_20261104.md).
Two user-local CPU stages are prepared, NOT run: `scripts/measure_curve_transfer.py
--stage fit` and `--stage development`, using `tmp/phase1_cpu_env/Scripts/python.exe`.
Fit cap 30min; development CPU-forward + scoring cap 2h, runtime unmeasured.
Eight short CPU fixtures and source/artifact preflight pass. No GPU or training.

The input audit already measured 2 voxels: development raw 2048/8192 support
70.24/91.98%, thin/dim 40.56/83.23%; fitted frog 22 box5 all-curve 36.05/74.46%,
thin/dim 16.68/69.53%. Reuse saved evidence; do not rerun this audit/filtering.
The new no-learning graph scores and frozen-checkpoint transfer are PENDING.
All 151 development reference-present boxes/7 neurons are included; 264 unknown
boxes excluded and disclosed. Development is repeatedly inspected, not a new
untouched test. CPU/GPU parity is checked before development inference; failures
stop without GPU fallback. No sixteen-box manifest or revised training gate.
Any future downstream criterion is post-hoc relative to the old failure and
must be declared prospectively before a new experiment, assessed on untrained
neurons. Conservative shortcut flags do not establish an exhaustive biological
wrong-join rate. No later hypothesis text below authorizes a Transformer change.

**Latest completed result, 4 November:** The user-local A/B/C CPU measurement
completed in 13.20 seconds (guard 14.52 seconds, exit 0), using saved update-2000
artifacts only. Read [the measured results](saved_vote_scaffold_results_20261104.md)
and [portable evidence](evidence/saved_vote_scaffold_20261104.json) first.
Output: `tmp/saved_vote_scaffold_20261104T182353_539123Z`.

- A: 212/369 off-reference retained points (57.45%) have fewer than three
  accepted-raw supporters; precision 47.26% versus 86.65% for supported points.
  Only 52/369 errors (14.09%) come from source distance >6. Same-scorer Sato
  and classical-node controls are recorded, including raw classical nodes.
- B: the predeclared trigger was met; exactly one radius-2/minimum-3-support
  variant ran. It **failed all four original checks**, P2=81.89%, C2=61.35%.
  Frog 5 C2 fell 64.91% ->25.27%, incident probes 16/22 ->6/22. This is
  post-failure evidence, not a rescue or a new adopted consolidation policy.
- C: fixed image-only k=6 Sato-cost minimum-spanning-forest SWCs completed for
  original and variant nodes, with no reference edges/roots in inference.
  Original segment P2/R2/F1=76.16/87.64/81.49%; variant=86.90/70.95/76.43%;
  preserved classical graph on the same boxes=56.19/53.14/52.46%.
  Original candidate-union R2 for k=4/6/10=90.86/92.60/94.53%; selects nothing.
  These are four TRAINING boxes/two neurons, per-box fit geometry, not development
  or topology results. Whole-image merging, qualified GATv2 and decoding remain
  absent. The scaffold is not a product stage or a GATv2 replacement.
- All 12 frozen source/input hashes and eight graph/SWC pairs were verified.
  No sixteen-box manifest was produced. No model/GPU/training/filtering ran.

**Stop this recipe.** No extension, second variant, threshold/gate change or GPU
run. The failed original gate is unchanged. The one formulation hypothesis to
discuss is curve-supervised Transformer selection/refinement of representative
anchors, because dense source-affinity/offset fitting does not directly learn
a compact coverage-preserving representation. It is not implemented or approved.
Discuss supervision, comparison and a bounded budget before substantive changes.
Candidate-edge union coverage, not isolated-node coverage, is the geometric
recall ceiling for selecting fixed edges. Strong spurious connected structure
remains a separate risk. No development inference was run in this follow-up.

D/E are complete in [the metric/proposal/code audit](proposal_implementation_audit_20261104.md).
Its prospective node-stage numbers are discussion only, not adopted gates.
Earlier notes below are historical and superseded by this completed result.

**CPU launch fix:** The user's first saved-vote measurement stopped after about
three seconds in A because a JSON-list box key reached a tuple-keyed guide
lookup. A did not finish; B/C did not run. The failed folder
`tmp/saved_vote_scaffold_20261104T175407_894949Z` is preserved. Both new launcher
evaluation boundaries now normalize identities to tuples. Ten CPU tests pass,
including the actual A-worker path with a JSON-list identity. Only execution
source/test bindings changed; criteria, variant constants, graph recipe and
all input artifact bindings are unchanged. The same user-local CPU command
was subsequently retried successfully by the user as recorded above. No rerun
was launched by the assistant. See `docs/evidence/saved_vote_launch_fix_20261104.json`.

**New CPU-only follow-up prepared, 4 November:** The user requested A: saved-vote
support/error-band/control diagnostics; B: exactly one conditional radius-2,
minimum-3-support consolidation; C: image-only kNN/MSF/SWC measurement scaffold;
D/E: metric/proposal/implementation audit. The criterion and all constants were
written before the new support-count results. See
[the frozen follow-up recipe](saved_vote_scaffold_protocol_20261104.md) and
[metric/architecture audit plus proposed criterion](proposal_implementation_audit_20261104.md).
The timed CPU measurement must be launched by the user with
`tmp/phase1_cpu_env/Scripts/python.exe -B scripts/measure_saved_vote_scaffold.py`.
At preparation time A/B/C results were pending; ten unit/integration fixtures
eventually passed after the JSON-key fix. Actual completed results are now
linked above; C ran regardless of B. A sixteen-box
manifest/PowerShell command is produced only on the post-hoc variant's numerical
pass, never launched automatically. This scaffold does not replace GATv2; no
GPU work or frozen-gate change is authorized by this follow-up. After the user
runs it, inspect report.json/report.md, preserve portable results and update
this handoff before any next formulation discussion. Those steps are now complete.

**Latest terminal update, 4 November:** The user-launched recovery completed
the original 2000-update four-box curve-head fit with exact prediction replay
and all 700 available optimizer-prefix updates matched. The accuracy gate
**failed**: two-voxel agreement 74.96% <90%; coverage 83.32% passed all coverage
conditions. Stop this recipe; no sixteen-box run, cutoff change or extension.
Read [the completed terminal results](local_curve_terminal_results_20261104.md)
first. The interruption/recovery notes below preserve earlier history. No
further timed job is running or automatically authorized after this failure.

**4 November update:** The user subsequently approved the bounded local
curve-head control. CPU checks/controls and GPU timing passed, but the four-box
fit was interrupted after a large wall-clock gap. Its latest complete checkpoint
and geometry evaluation are at update 1000; the last logged update is 1700.
There is no terminal accuracy verdict, sixteen-box run or pending continuation.
Read [the current results and recovery decision](local_curve_head_results_20261104.md)
and [the frozen experiment protocol](local_curve_head_protocol.md) before using
the historical immediate-next-step text below. Original evidence is retained.

**Local execution preference and recovery command:** The user requires visible
commands for timed experiments and launches them locally. Do not launch timed
CPU/GPU jobs from the agent unless explicitly requested. The
[checkpoint-resume command](local_curve_recovery_ready.md) is now prepared and
CPU-checked: restore update 1000, finish at 2000 under a fresh 30-minute wall
cap, preserve Adam/RNG, verify original predictions and replayed updates. The
assistant has not launched this recovery. It cannot automatically start sixteen
boxes. Do not use the older fresh-training command as a resume command.

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
