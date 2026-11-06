# PN2 next-chat handoff — after failed dense-head fit and CPU graph measurement

**LATEST, 6 November: BigNeuron recovery and first scoring COMPLETE. No further
job is queued. Stage 2 remains STOPPED with its gate, STOP.json and artifacts
unchanged.** Read [benchmark results](bigneuron_benchmark_results_20261106.md),
[portable evidence](evidence/bigneuron_benchmark_completed_20261106.json), and the
[editing contract DRAFT](classical_anchored_editing_contract_DRAFT_20261106.md).
The user-local CPU job completed in 2,227.08 seconds, guard exit 0, reusing the
archive. All seven identities resolved: 237 SWCs, 44 variants, 308 method/case
slots (208 accepted, 71 missing, 29 rejected). All 237 SWC hashes match; stored
primary and whole-image aggregates replay exactly. No new training labels,
model fitting, GPU work, scoring-rule change or reserved-neuron inspection.

Fixed archive leader `neutube_updated`: same 151-box neuron-macro segment F1
64.00/73.12%, resampled F1 58.72/68.14% at 2/6 voxels, pooled length 1.25017x.
Median across all 44 including missing/rejected zeros: segment 30.61/39.73%,
resampled 27.29/35.51%. This is parameter-optimized external context with unknown
seed provenance, not a certified independent reference-free baseline.

IMPORTANT: leader Series017 scores a protocol zero because 1/16 nodes lies
0.2044 voxel beyond the native cell bound. The frozen 99%-of-vertices check is
coarse and sampling-density dependent; passing also does not establish origin.
Do not silently relax, clip, shift, exclude or rescore. Native comparison remains
provisional; discuss a separate coordinate audit before treating it as decisive.

Editing(b) still passes ONLY the fixed drafting gate: resampled 61.74/68.54%,
length .55105x; (a) 58.58/64.74%. Its apparent margin over the archive leader is
3.02/.40 percentage points and depends on the Series017 protocol zero. Also,
(b) hit its 128-edit cap in 106/151 boxes; 97 used budget component drops (767
in total). It is a feasible lower bound, not a solved ceiling. Do not extend the
search. No learned advantage is established. Hold substantive implementation
and training for discussion of coordinate/source interpretation, an actual
image-only editing-unit pool and a prospective comparison contract. No new
floor is adopted from these repeatedly inspected development results.

All five post-stop computational diagnostics are now complete under their
declared rules; interpretation has the caveats above. Professor discussion is
pending. Earlier readiness/pending statements below are chronological history,
not launch instructions. Do not restart Phase 1 or rerun completed jobs.

**LATEST,6 November: post-stop diagnostics completed;benchmark scoring still
pending;editing DRAFT now authorized by the measured gate.** Read
[full results](post_stage2_diagnostics_results_20261106.md),
[portable evidence](evidence/post_stage2_diagnostics_completed_20261106.json),
[editing contract DRAFT](classical_anchored_editing_contract_DRAFT_20261106.md),
and [benchmark identity-repair command](bigneuron_benchmark_recovery_ready_20261106.md).
User ran the frozen4212b95 CPU jobs locally,not the agent. All guards exit0.
Ridge/review2342.56s,archive345.86s,decompose326.28s,editing6828.77s.
Benchmark_score2s wrote incomplete-mapping status and computed NO metrics.
Old Stage2 STOP/gate/artifacts remain unchanged;no exclusion,retuning,relabeling,
neuron-model training,GPU,GATv2 or editing architecture adoption.

Editing(a)resampledF1_2/6=58.58/64.74%,segment64.77/77.01%,length.49075x.
(b)=61.74/68.54%,segment68.56/80.43%,length.55105x. Fixedb>=60 draft gate
PASSES;aaloneFAILS. All151 boxes included,maxfinalboxLR1.24205<=1.25.
Frog/mouse use declared neuron(a)fallback. Native2 segmentrecall55.49%,resampled
recall51.53%;this is still incomplete geometry,NOT a learned or final result.
Alias94remainsweak27.29%resampled2. Draft only proposes scaffold-chain keep/prune
and anchored branches. Oracle-filtered/ordered forests are NOT image-only
training candidates;validate an actual image-only unit pool before claiming
transfer of this opportunity. Discuss before substantive implementation.

Ridge40-neuron packets complete. Pooledref length>2vox offproxy:train16.60%,
dev15.06%;WEDonly2.37%. Lowadultalias95=73.10%,median8.61vox;proxy threshold/
weak/gapped structure can cause large offsets,not proof goldwrong.40-neuron
native/conditional-physical CSV saved;all15XY/XZ/YZ sheets visuallyinspected.
WEDoutputs flood diffuse texture;larval soma/thick-process oversampling and
lowadultdim-branch omission are distinct failures. Automaticflags often soma
volume/puncta,not validated extra targetneurites. No relabeling.

DecompremovingisolatesworsensC0 andactualpolicyat2;not auniversalfix.
C1minusallresampled43.87%,policyminusall47.29%(classicaldev47.43%). Spurs
remove substantial length but also coverage;longduplicateproxynegligibleexcept
conditionaloracle. Itsall3cleanup61.08% is reference-assisted,not adoption.

Benchmark archive alreadydownloaded726,174,044bytes. Original extractor only
matchedSLP36,Series01720,Janelia40,notall7. Actualnames encode ordinal+dataset
IDs,so oldalgorithm_variant labels also incorrectly contained per-neuron IDs.
Prepared `scripts/recover_bigneuron_benchmark.py`,config
`configs/bigneuron_benchmark_recovery_20261106.json`,NEWoutputroot
`tmp/bigneuron_benchmark_recovery_20261106`,cap2h,NOredownload. No repair scan/
extraction/scoring launched yet. Maps six publishedIDs251/245/103/122/276/52;
all96 priornamedfiles corroborate onmetadata. Humanin_house1sharedCell1name
cannot be assignedID256 just frombasename(thepublishedSWCtimestampdiffers);
requires exactnormalizedsourcegroup/caseleaf,otherwise reportincomplete7mapping.
Method labels removeordinal/datasetIDs,retainparams andseed-encodedmode;all
replicas contribute meanprimitivecounts,no bestreplica selection. FIRST metric
scoring only,unchanged nativebounds99%,no registration,originprovisional.
User launches the one repaircommand;no completed diagnostic needs rerunning.
Benchmark best/median/per-algorithm numbers do NOT yet exist.

Verification:302final-row graphhashchecks/length bounds pass,357output files
inventoried(40offsetNPZ+15PNGs+302SWCs),stored-row aggregates replayexact.
OriginalSTOPsha0edf832a39d52bd5cd48296dbb5d613b1af51c0bdd5ea762f1af797425de6eae.
Originalgeneratedreports unchanged;newdocs recordvisualreview. Professor
discussionpending;repeatedlyinspecteddev/reserveduntouchedlimitations persist.
Earlier readiness notesbelow are history.

**LATEST,6 November: five post-Stage2 CPU diagnostics prepared for user-local
launch;NONE RUN. Stage2 remains STOPPED unchanged.** User explicitly requests
gold-to-ridge offsets40 neurons,a five-case visual review,BigNeuron algorithm
outputs on7 development neurons,resampled-F1 decomposition,and classical-anchored
editing oracle headroom. Read [fixed definitions](post_stage2_diagnostics_protocol_20261106.md),
[five commands](post_stage2_diagnostics_ready_20261106.md),
[metadata inventory](evidence/post_stage2_diagnostic_inventory_20261106.json) and
[51 passing checks/readiness](evidence/post_stage2_diagnostic_readiness_20261106.json).
Runner `scripts/post_stage2_diagnostics.py`,config
`configs/post_stage2_diagnostics_20261106.json`;new output root ONLY
`tmp/post_stage2_diagnostics_20261106`. No output action folder exists yet.

Caps:ridge_review2h,benchmark_download30min,benchmark_score2h,decompose2h,
editing4h. User launches each separately;no automatic next action/retry/extension.
No real diagnostic metrics,review of real images,archive download,training,GPU,
GATv2 or new development training labels occurred during implementation.
OriginalStage2 source/runtime/input preflight still passes;STOP unchanged.

Fixed ridge proxy:sigma1,Hessian two transverse negative directions,corrected
derivative-kernel DC,normalized smoothed intensity>=.05,NMS+/-1,search12 native
voxels,full polyline .5-voxel weighted quadrature including outside-image queries.
Report>1/>2/>3 shares,censoring. Authors' scaling_gold.csv joins40/40 identities
and agrees with15 local sidecars. Microns are CONDITIONAL published metadata,
not verified acquisition calibration;physical search12*minspacing/physical weights.

Panel fixed from saved C1 ratios:WED_SLP,two high larval zebrafish1_1 and1_5dpf,
two low adult zebrafishhard_hard1_10 and1_9. Original max projections andref/
retained/C0/C1 overlays;flags are possible untraced image structures ONLY.
After user launch inspect generatedPNGsinsharedworkspace before marking anything
visually obvious. No relabeling or training follows visual review.

User did not know where benchmark outputs were because the provided gold166
download contains manual traces. Official automatic outputs are a SEPARATE
gold166_bt_v1.0 release. Local command fetches fixed optimized5571 archive,
726,174,044bytes;extracts/parses only7 approved development identities. Source
filenames need boundary matching(SLP !=WED_SLP);Cell1 basename duplicatesrequire
exactsourcegroup or publishedID. If archive identities are incomplete,reportthe
mapping gap,no fake seven-case score. No gold references substituted as algorithms.
Coordinate checks use nativeXYZ identity,>=99% withinbounds,no fittedalignment;
origin/subvoxel alignment remains disclosed/provisional until image review.

Decomposition:original saved scores replay first;isolates,spurs<6 voxels,
duplicate-path proxy>=80% within2 with abs tangentcos>=.9. Independentandordered
ablations;F1 deltas are nonadditive. KeepC0/C1 trainingCV separatefromdevpolicy.
Editing(a)exactclassicaledgeprune forest;(b)add only existing connectedSato/
learned candidate geometry,no classical-classical shortcuts. Bothoracleonly,
length<=1.25 PER BOX,bounded128-step/16-path search,not globaloptimalceiling.
Keep(a)feasible fallbackperbox/perneuron. User's NEW gate:b developmentmacro
resampledF1_2>=60% allows draftingeditingcontract;statewhetheraalonepasses.
No draft has been adopted or trained;after measuredresultsdraftonlyifgatepasses.
This does not reopen Stage2,alter its failed gate or promote an inference stage.

**LATEST,6 November: Stage2 C0/C1 CV finished and STOPPED. No extension.**
Read [results and interpretation](stage2_trees_results_20261106.md),
[portable evidence](evidence/stage2_trees_stopped_20261106.json) and
[paired33-neuron table](evidence/stage2_trees_paired_neurons_20261106.csv).
Run implementation c53cd11,root `tmp/stage2_trees_20261105`;all33 folds completed
in8,454.36s guard exit0. STOP reason is the predeclared training-CV pooled
length<=1.25 rule: C0=1.254998,C1=1.740456. Gate unchanged;do not omit neurons,
relax thresholds,rerun,extend or continue to PointNeXt/GATv2 automatically.

C0 CV segmentF1_2/6=51.32/82.47%,resampled39.41/55.88%;C1=58.49/83.72%,
resampled40.85/55.06%. C1 wins31/33 at2 overC0 but increases length38.68%.
Same460-box classical training:segment55.97/77.13,resampled48.96/66.13%,
length3.43198x. C1's2.51pp gain overclassical at2 is only16/33 wins and
resampled8.10pp worse;this is not a qualifying reconstruction recipe.

WED heldout fold07 dominates excess length:C0=20.19x,C1=36.59x. Node policy
keeps53,394 nodes(830 trace-positive,52,493 trace-negative,71 unknown,nineboxes).
Untraced real structure may contribute;trace-negative is not biologicalbackground.
Excluding it from saved sums gives.91280/1.11063x ONLY as concentration diagnostic;
it is NOT excluded from any gate.7/10 neurons individually exceed1.25x.

Earlier policy development diagnostic selectingnothing:image-only scaffold
segment58.22/79.62%,resampled43.83/55.73%,length1.56156x. Same selectednodes
with reference-assisted EDGE oracle:74.74/88.27%,46.78/58.26%,1.05603x.
Compare earlier compactunion88.52/95.10%,64.80/72.54%,1.18475x. Newimagepolicy
has substantial loss before learned edges;oracle diagnostic is not realinference
or a mathematical universalceiling. Development already repeatedlyinspected.

No PointNeXt-edge CV,finaledgefits,finaledge development scoring,same-contractAP,
finaltrainfitparity,GATv2,GPU orwholeimageassembly occurred. Policy diagnostics
DID scoredevelopment;don't say development was untouched. CV remainsconditional
on upstream encoder AND positions trainedall33,not strictfullpipelineCV.
Hashverified99models,920foldSWCs,460originaltrainlabels;storedaggregates replay
exact;source/runtime/input preflightpasses. Preparation526s,policy76.56s. No
RUNNING lock;guardcomplete. All useful caches/models/evidence preserved.

Discussion-only hypothesis:image-local trace-support selection plus greedy
representatives does not reliablyidentify a compact annotated-neuron instance.
Review saved image/reference context and node/decoder responsibility before
proposing a revision. No new architecture,labels,thresholds,sourceorder orlaunch
implemented. Professor discussionpending. Earlier readiness notesbelow arehistory.

**LATEST continuation,5 November: Stage2 CPU tree implementation APPROVED and
ready for user-local launches, NOT RUN.** Read
[fixed protocol](stage2_trees_protocol_20261105.md),
[commands](stage2_trees_ready_20261105.md), and
`configs/stage2_trees_20261105.json`. User explicitly answered: keep handcrafted
node selection/representatives fixed; add PointNeXt to EDGES ONLY. Require its
>=2pp development segment-F1 gain at2 voxels and every6-voxel floor. No GPU,
GATv2,automatic extension or new development training labels.

New runner `scripts/run_stage2_trees.py` has separate prepare/policy/compare
actions (caps2h/1h/6h), isolated `tmp/stage2_cpu_env`/requirements lock, source/input
bindings, write-once outputs under `tmp/stage2_trees_20261105`, STOP guard and
first-fold resource projection. Older environments/packets/protocols untouched.
Only synthetic CPU tests and metadata checks have run during implementation;
no real preparation/model/CV/development action. No Stage2 quality number exists.
Readiness evidence: [48 passing synthetic checks and verified56 source/163
artifact bindings](evidence/stage2_trees_readiness_20261105.json). Tests include
real HistGBM on toy rows and mocked33-fold orchestration; no retained neuron
packet entered a fit. No Stage2 run directory exists yet and nothing is running.

Policy: handcrafted p>=.5, NMS inclusive radius2, source classical>Sato>learned,
then Sato descending,selector p descending,XYZ/index. Fixed611 union pools and
induced existing edges; no new kNN/rewiring. Policy diagnostic on151 dev selects
nothing: actual image-cost forest plus reference-assisted EDGE oracle conditional
on frozen image-selected nodes, compared with88.52/64.80% compact oracle,length
1.18475. Earlier oracle ordering was probability-first; not the new inference rule.

C0/C1 conditional selector LONO33: each node/edge model fits32 training neurons,
same representatives per feature arm; C1 retained-set adjacency preserves original
unknowns and changes no old labels. Choose contract on training-fold decoded
geometry with pooled length<=1.25,then F1_2/F1_6/resampled F1_2. No AP comparison
across contracts. Compare HC and +PN edges under chosen contract; same-contract
AP only. Upstream encoder AND learned positions saw all33: user accepted this
conditional CV limitation. Global33 node model never used for CV.

Both final edge models frozen before any final dev score; training fit ungated.
HC must pass dev segment59.33/74.14,strictwins>=5/7 both,resampled52.43/59.72
and actualclassical+5pp,length<=1.25. On HC failure STOP before PN dev scoring.
PN also needs>=2pp gain at2; failure stops,no tuning. Repeatedly inspected dev,
per-box measurement only. Reference-only join/shortcut/fragment and break
diagnostics are explicitly conditional,not biological topology certification.
DraftB stays rejected,DraftA held; older status notes below are preserved history.

**LATEST continuation,5 November: compact/resampled audit COMPLETE. DraftB
REJECTED,DraftA HELD by user.** Read [full new metrics](resampled_compact_results_20261105.md),
[new Stage2 draft](stage2_candidate_geometry_contract_DRAFT_20261105.md),
[portable evidence](evidence/resampled_compact_completed_20261105.json) and
[fixed protocol](resampled_compact_audit_protocol_20261105.md).
User explicitly authorized CPU audits;no Stage2 features/model/contract code
or training. Root `tmp/resampled_compact_audit_20261105`,372.34s guard exit0.
5,361 historical+611 compact=5,972 graph entries,26 methods;5,450 distinct
metric cache files. All source/graph hashes and aggregates verified;all611
old-oracle replays exact before NMS;old460 label packets unchanged. Awake
helper released;finished.json verified. Sessions68122/57087 done;nothing running.

Original Stage0 gate PASSED at90.42% segment F1_2 with2.39785x pooled length.
New COMPACT UNIONdev:segment F1_2/F1_6=88.52/95.10%,resampled F1=64.80/72.54%,
pooled length1.18475x,macro1.16129x. ORIGINAL headroom gate PROCEED>=80%,
65–80/<65 branches not triggered;NMS not adopted. Weakdev alias94=64.20%F1_2,
length.5736x,disclosed not a new per-neuron gate. Matched classicaldevsame151:
segment54.33/69.14%,resampled47.43/54.72%,pooled length.91754x/macro1.27092x.

Original LABEL oracletrain:segment90.82/96.68%,resampled54.40/61.17%,length
1.56241x. COMPACT LABELtrain:segment72.28/93.38%,resampled68.96/81.88%,length
.51322x;18,336isolates/54,143 retained nodes,23,860edges. Smaller ratio is
missing structure,not automatic success. No relabeling or repair performed.

NMS FIXED:eligible oracle p>=.5,greedy inclusive radius2,probability descending
then classical>Sato>learned source priority,XYZ/index ties. Keep real coordinates,
remove suppressed nodes' incident edges,NO rewiring/new kNN/relabeling. Original
spanning/pruning unchanged. Both node oracles binary,so eligible ties use sources.
Newmetric:clipfirst,maximal degree-two chains every2vox,structural endpoints/
junctions once by identity,short remainder endpoint included,isolates once,
cycles deterministic;do NOT weld disconnected coincidences. Bipartite max
cardinality at2/6vox,pool counts withinneuron thenmacro. New reference samples
ephemeral scoring only,no development label packets. Native units(1,1,1).

Fullperbox `per_box_metrics.csv`,perneuron `per_neuron_metrics.csv`,and paired
33-neuron `paired_training_neuron_oracles_NOT_CV.csv` under outputroot. Scope:
actual saved curve-stage decodes/oracles,raw/pruned classicalknown-box baselines,
cached reference-edge representationoracle;candidate unions are opportunity
diagnostics,unknown regions not relabeled. Historical rejectedB gets metrics
only,never promoted. Whole-image assembly/topology/radius remain unqualified.

New Stage2 contract DRAFT ONLY: C0 original full-pool labels mapped to common
image-only representatives versus C1 adjacency conditional on retained set.
Candidate features proposed:ALL-candidate intervening count<=1,local spacing,
not reference-positive witnesses. Proposed guide-ordered real-inferenceNMS is
UNTESTED/unadopted;oracle ordering unavailable at inference. No C1 label packet,
feature/model implementation or trained CV comparison. Draft specifies paired
33-neuron conditional-selector LONO F1/length comparisons;existing oracle
tables are explicitly NOT CV. Any trained comparison requires later authorized
implementation;do not fabricate it. Keep professor discussion/upstream leakage,
handcrafted-only controls,original final floors/wins and GAT>=2pp disclosed.
No automatic training or pipeline promotion follows the headroom pass.
Earlier draft statuses and “next work” notes below are preserved history.

**LATEST continuation,5 November: edge-label geometry audit COMPLETE; Stage2
HELD for contract/decoding discussion.** Read [full audit](edge_label_geometry_results_20261105.md),
[unadopted PRE-TRAINING revision](edge_supervision_revision_DRAFT_20261105.md)
and [portable evidence](evidence/edge_label_geometry_completed_20261105.json).
User explicitly requested CPU audits before ANY Stage2 feature/model work.
Root `tmp/edge_label_geometry_audit_20261105`,167.63s guard exit0. All460 native
overlap packets/920 draft SWCs verified,aggregate replay exact,old label hashes
unchanged. No development labels,new model/feature/training/GPU work. Nothing
running; awake helper released,finished.json verified. Sessions40488/56923 done.

Of180,269 nonconsecutive negatives,179,171 (99.39%) have in-box overlap>=.5
at2vox;weighted overlap98.70%. All33 neurons96.42–99.90% on-trace negatives.
96,363 same-chain skips have an intervening positive within1 of the segment;
80,197 same-chain other;3,709 cross-reference-chain pairs. Of163,536 positives,
444 have overlap<.5;24 have no in-box length. Existing near-oracle F1 did NOT
validate alignment of consecutiveness labels with geometric agreement. Earlier
statement supporting the whole contract was too strong; earlier gate pass
stays on record,not retroactively changed.

User's>=30% training-only trigger fired. TWO predeclared draft diagnostics,
NEITHER ADOPTED: A mask179,171 conflicting negatives,all460 fresh SWCs exact
old replay,F1_2/F1_6=90.82/96.68%,pooled length ratio1.5624. B binary geometric
labels on otherwise KNOWN two-positive-endpoint edges,preserve ALL unknowns
and node labels,F1=91.05/96.78%,length ratio2.1363 (+36.73% length for+.23pp F1).
No new training-label manifest; outputs only `draft_only/`. Contract discussion
must address supported-node versus representative-node semantics and decoder
multiplicity/topology. Do not proceed with Stage2 or silently apply a draft.

Full length ledger:11,340 per-box/aggregate metric entries in
`tmp/edge_label_geometry_audit_20261105/all_prior_length_ratios.csv`.
Stage0dev union2.398x reference length; training label oracle1.562x/geometric
2.235x; four-fit dense2.402x;broad train6.503x/dev6.236x;matched classicaldev
.9175x. High tolerance-overlap F1 cannot certify unique correct tracing.
Candidate scripts score native XYZ,effective spacing(1,1,1),NOT microns.
Conditional micron view uses12 training sidecars/224 boxes,XYZ/calibration
UNVERIFIED;missing21 not converted. Tolerances.5/1/2um,quadrature.1um.
Label-oracle macroF1=.6482/.8074/.9189 onthat12-neuron subset(native2vox84.90%).
No development physical scoring/labels. Source sidecar text/hashes retained.
The earlier “next Stage2” statuses below are historical and superseded here.

**Latest continuation,5 November: saved-label CPU audit COMPLETE,gate PASSED.**
Read [label-oracle/fragment/AP results](candidate_label_audit_results_20261105.md)
and [portable evidence](evidence/candidate_label_audit_completed_20261105.json).
User explicitly requested agent CPU execution, overriding local-launch preference
for these diagnostics only. Root `tmp/candidate_label_audit_20261105`,941.63s
guard completed exit0. All460 records/920 SWCs/460 score packets hash-verified;
aggregate replay exact,original Stage0/Stage1 recipe/source bindings intact.
Label oracle F1_2/F1_6=90.82/96.68%; same-training geometric oracle92.38/97.21%.
Label gate90.8218>=87.4232 (Stage0dev90.4232 minus3pp),PASSED. Matched training
gap1.56pp at2. Label-positive edge union R2=85.07%,decodedR2=84.75%.
No labels/geometry/thresholds changed; no selector trained or dev rescored.

All2,940 cross-component edges with two unambiguous positive endpoints masked;
zero silently positive/negative. Mask among both-positive endpoint edges16.54%
versus all-edge2.62%. Fragment aliases23/66/75 have51/200/80 reference components;
mask2.21/5.52/4.14%, so23 is not above overall. Three node-empty/seven edge-empty
boxes named in results,span5neurons. Fixed-feature macroAP:Sato straight-path
16.92%,nearest-input-distance5.44%,macroprevalence5.27%. Pooledprevalence4.03%
has different weighting. Sato control is NOT geodesic; that feature remains
unbuilt. Rankings fixed/no fitting,per-held-out-neuron scoring; no accuracy.
Read [execution contract](candidate_label_audit_protocol_20261105.md).

Existing user-started CPU feature export COMPLETED,guard390.42s exit0.
All611 feature packet hashes/shapes/finiteness verified,32channels,model/RNG
unchanged,updates0,CUDA not initialized. Do NOT run feature export again.
[Feature results](candidate_pointnext_results_20261105.md) include query-distance
summaries; no distance policy selected. AC connected; scoped awake helper was
released and finished.json confirms it. Sessions29435/36212 completed; NOTHING
is running. No automatic model training follows the diagnostic pass. Next work
is handcrafted image features and GBT controls before GAT,including handcrafted-
only versus +PointNeXt,conditional selector CV/upstream leakage disclosure,
CV/dev gap and original final floors/wins/stop rules. Professor discussion pending.

**Latest continuation,5 November: Stage1 COMPLETE and verified.** Read
[Stage1 results/next command](candidate_stage1_results_20261105.md) and
[portable completion evidence](evidence/candidate_stage1_completed_20261105.json).
`tmp/candidate_stage1_20261105/report.json` completed,worker442.43s exit0
(console7m34s); all611 candidate/460 label/33 classical packet hashes verified,
no development labels/model/GPU/training. Six classical extractions reused,
27 missing training extractions from retained guides only. Training totals:
673,235 nodes (118,681+/549,054-/5,500unknown),4,172,031 edges
(163,536+/3,899,191-/109,304unknown). Known positive fractions17.77%/4.03%;
mask0.82%/2.62%. All33 neurons have positives;3 boxes no positive nodes,
7 no positive edge labels; details in results. Alias20 dominates~24%nodes/
23%edges. No recipe or gate changed; do not remove hard cases or rerun Stage1.

NEXT user-local CPU inference-only command:
`../pointneuron2/tmp/pointneuron2/sampling_env/Scripts/python.exe -B scripts/export_candidate_pointnext.py --device cpu`.
Metadata/checkpoint preflight passed. No real feature export has run. Default
output `tmp/candidate_pointnext_20261105`,611 source32D/interpolated feature
packets,3600s cap, no automatic retry/training. Inspect completeness/distances
before later handcrafted feature/GBT preparation. Conditional selector CV,
final floors/wins, GAT>=2pp and stopping rules unchanged. No job is running.
Earlier readiness/pending statuses below are preserved history.

**Current continuation,5 November: Stage0 COMPLETE, gate PASSED.** Read
[Stage1 ready/commands](candidate_stage1_ready_20261105.md),
[Stage1 contract](candidate_stage1_contract_20261105.md) and
[portable Stage0 evidence](evidence/candidate_headroom_completed_20261105.json).
User-local Stage0 output `tmp/candidate_headroom_20261105T084249_809302Z`:
604 oracle records/SWCs over151 boxes/seven repeatedly inspected dev neurons,
575.09 seconds,guard completed exit0. Union oracle F1_2=.9042320394860465,
F1_6=.9567199197684702; original>=.80 gate passed. Classical/Sato/broad oracle
F1_2=74.00/83.45/85.33%; union k6/k10 candidate R2=88.49/90.44%.
Alias94 oracleF1_2=67.97%, disclosed not a new gate. These are reference-assisted
geometric results, not learned selection accuracy, topology or final quality.

Stage1 CPU preparation and frozen PointNeXt feature export are IMPLEMENTED,
fixture-tested and source-bound, NOT LAUNCHED. Give user command
`tmp/phase1_cpu_env/Scripts/python.exe -B scripts/prepare_candidate_selection.py`.
Output `tmp/candidate_stage1_20261105`,3600s cap,no automatic retry/training.
Reuse six training classical graphs; extract missing27 from cached guides only.
460 training pools/labels plus151 reused Stage0 development pools. Save all
geometry before training refs; never load oracle probability files for training.
Label API rejects dev;2-voxel trace labels,32-voxel consecutive-path bound,
.25 distance/.5 path ambiguity mask, distinct nearby positive junction anchors
change LABEL memberships only, never inference geometry. Per-neuron counts.

After successful preparation/count inspection, user-local inference-only
`../pointneuron2/tmp/pointneuron2/sampling_env/Scripts/python.exe -B scripts/export_candidate_pointnext.py --device cpu`
exports611 source32D features and3NN candidate queries/distances to
`tmp/candidate_pointnext_20261105`,3600s cap. Optional user --device cuda:0 is
inference only; assistant launches no GPU. Handcrafted edge intensity/geodesic
features, GBT/GAT training and wrong-join/break final reporting remain later work.
CV remains CONDITIONAL selector-only on upstream trained on all33, user accepted.
No timed preparation/export/model job is running. Earlier pending Stage0 states
below are historical, superseded by this current continuation.

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
