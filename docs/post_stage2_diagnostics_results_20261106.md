# Post-Stage2 diagnostics — results and editing draft authorization

**Completion addendum:** the user subsequently ran archive identity recovery and
first scoring successfully (37m07s). All seven cases are resolved. Read the
[completed provisional comparison](bigneuron_benchmark_results_20261106.md) and
[portable evidence](evidence/bigneuron_benchmark_completed_20261106.json).
The incomplete-mapping statements below describe the original attempt. No
original output or frozen rule was changed. Editing(b) reached its search cap
in 106/151 boxes; its score is achieved feasibility, not an upper bound.
Stage 2 remains stopped and no further run is queued.

Stage2 remains STOPPED. Its failed gate,STOP.json,33-neuron CV artifacts,
labels and checkpoints are unchanged. The user ran the five CPU actions locally.
Ridge/review,decomposition and editing completed. The archive downloaded,but
benchmark scoring evaluated NO algorithms because case mapping was incomplete.
No training,GPU,GATv2,exclusion,retuning or relabeling occurred.

| Action | Measured wall time | What completed |
|---|---:|---|
| Ridge/review | 2,342.56s (39m03s) | 40 neuron reports/offset packets;five three-view review sheets |
| Benchmark download | 345.86s (5m46s) | Correct726,174,044-byte archive;96 outputs matched to3/7 cases |
| Benchmark scoring | 2.00s | Incomplete-mapping report;NO metric computed |
| Decomposition | 326.28s (5m26s) | Saved C0/C1 and both policy diagnostics,all fixed ablations |
| Editing | 6,828.77s (1h53m49s) | Both oracle arms on151 development boxes |

All guards exited0 within their caps. Exit0 for the mapping report means the
check completed,not that a seven-case benchmark score exists. This is why the
user's guarded PowerShell sequence continued to the independent diagnostics.

## 1. Gold-to-ridge offsets

These are arclength-weighted offsets to the FIXED operational intensity-ridge
proxy,not certified anatomical correspondence or a correction to gold labels.
The full supplied polyline remains the denominator,including outside-image
geometry. Native search is censored beyond12 voxels. Reported-metadata physical
distances use an independent anisotropic search and physical length weights;
they are not simply voxel distances multiplied by one scalar.

| Cohort | Reference length >1 voxel off-ridge | >2 | >3 | No ridge within12 |
|---|---:|---:|---:|---:|
| Training,pooled length | 45.06% | 16.60% | 9.82% | 3.79% |
| Development,pooled length | 30.65% | 15.06% | 10.48% | 2.55% |

Neuron-macro shares>2 are9.29% training and16.06% development. Pooled and macro
weight different things;neither should be silently substituted for the other.

WED_SLP is largely aligned with this ridge proxy:only2.37% of reference length
is>2 voxels away,median.723 voxel. The two high-length larval cases are6.20%
and2.35%>2;their medians are.875 and.674 voxel. A broad gold-to-ridge displacement
therefore does not explain their severe excess reconstruction length.

The low-length cases differ. Adult alias93 has only3.66%>2 and median.593 voxel,
yet reconstruction misses many visible dim branches. Alias95 has73.10%>2,
median8.61 voxels and42.76% censored;its median physical offset is1.823um under
reported spacing. Development alias94 has53.52%>2 and median2.372 voxels/.510um.
Weak punctate branches,interpolation through image gaps,the fixed intensity
threshold/smoothing and annotation/image mismatch can all contribute. These
numbers alone do NOT establish that gold is wrong. Thresholds/tolerances and
labels remain unchanged.

All40 neurons have conditional published spacing metadata,consistent with15
local sidecars. Acquisition calibration and original coordinate origin remain
unverified. Per-neuron native shares,physical quantiles/shares,censoring and
spacing are preserved in the [40-neuron table](evidence/post_stage2_ridge_neurons_20261106.csv).

## 2. Visual review of five neurons

The assistant inspected all15 generated XY/XZ/YZ sheets. Original generated
reports remain unchanged;this document records the later visual interpretation.
Overlays contain only saved reference-present boxes in the original global frame;
they are not whole-image reconstruction or crop assembly.

| Case | What is visible | Interpretation and limit |
|---|---|---|
| WED_SLP,alias20 | Recognizable bright reference arbor amid broad diffuse/punctate image texture;retained/C0/C1 outputs densely fill rectangular sampled regions | Much of the enormous output is distributed through texture rather than isolated extra arbor segments. Some unmatched bright structures are visible,but the flood is not credibly explained entirely as untraced target neurites. MIPs cannot prove neuronal identity. |
| Larval zebrafish1_1,alias9 | Central complex arbor plus detached bright clusters/process-like image structures;outputs oversample thick regions and extend into unmatched structures | Flag detached unmatched structures as POSSIBLE untraced image processes or other-cell signal. Their target-neuron membership and connectivity are unresolved;no label changes. |
| Larval zebrafish1_5dpf,alias11 | Bright soma and axonal/dendritic processes;many retained nodes and several red callouts occupy soma interior | The automatic far-from-trace flag can highlight true bright soma volume rather than an omitted neurite. This exposes centerline-versus-foreground semantics;the callouts do not validate untraced branches. |
| Adult1_9,alias93 | Bright soma/varicosities and many dim reference-supported branches;outputs remain fragmented on bright portions | Clear undercoverage despite small gold-to-ridge offsets. The sheets do not separate input sampling from feature/selector failure. No strong additional untraced target-neurite claim. |
| Adult hard_hard1_10,alias95 | Strong soma/main processes with a broad field of dim punctate branches represented in the reference;outputs keep mainly strong portions | Low coverage and a large proxy-ridge offset occur together. The ridge proxy may miss weak/gapped branches;do not conclude reference error or repair gold from this alone. |

No trustworthy biological untraced-neurite rate can be read from these MIPs.
Automatic callouts are suggestions,not annotations. The images support different
failure modes across neurons and do not justify one global label-mask change.
The unchanged review packet is `tmp/post_stage2_diagnostics_20261106/ridge_review/index.html`.

## 3. Benchmark comparison remains incomplete;repair prepared

The archive is present. Original extraction matched SLP36 outputs,Series01720,
andJanelia40;frog,humanAllen,mouse andadult zebrafish had zero identified outputs.
My first extractor was too literal about image/folder names. Actual archive
filenames such as `01_103.v3dpbd_app2new3.swc` encode a file ordinal and dataset
ID. Those are also not valid parts of a cross-neuron algorithm label.

Existing extracted metadata independently confirms published IDs251/103/52 for
the three named cases. A separate repair command maps the remaining unique
published IDs,handles the shared Cell1 basename conservatively,and normalizes
method suffixes BEFORE first scoring. It does not redownload,overwrite earlier
files,change coordinate rules or look at benchmark accuracy to choose a mapping.
If all seven cannot be resolved,it reports the remaining gap rather than silently
excluding a case. No best,median or per-algorithm accuracy number exists yet.
See [repair command and fixed aggregation](bigneuron_benchmark_recovery_ready_20261106.md).

## 4. Resampled-F1 decomposition

These fixed geometry ablations select nothing. C0/C1 are33 training-neuron
conditional CV;policy rows are seven-neuron development diagnostics. Keep those
cohorts separate. The table uses2-voxel resampled F1.

| Arm | Original | Minus isolates | Minus spurs | Minus duplicate proxy | Minus all3 |
|---|---:|---:|---:|---:|---:|
| C0 training CV | 39.41% | 37.55% | 39.67% | 39.41% | 37.34% |
| C1 training CV | 40.85% | 43.22% | 41.35% | 40.86% | 43.87% |
| Policy image scaffold,development | 43.83% | 43.49% | 47.64% | 43.83% | 47.29% |
| Policy conditional edge oracle,development | 46.78% | 53.48% | 49.60% | 47.09% | 61.08% |

Isolated nodes are NOT uniformly the cause:removing them worsens C0 and the
actual image scaffold at2. Some contribute valid matching coverage. The oracle
diagnostic has many unsupported isolates and benefits more. F1 effects are
nonadditive because matching,sampling and junction identities change.

Spurs account for substantial length:43,558.63vox C0,65,254.22 C1,19,117.14 policy
image and19,598.58 policy oracle. Removing spurs reduces C1 pooled length from
1.74046x to1.23581x but also lowers its segment F1_2 from58.49% to52.89%. This is
not a retroactive pass or an adopted postprocessing fix. For the actual policy,
all3 removal gives segment55.57/77.08%,resampled47.29/62.67%,length1.11757x;
its resampled F1_2 still does not beat matched classical47.43%.

The declared long near-parallel duplicate proxy flags0 C0 paths,4 C1 paths,1
image-policy path and192 conditional-oracle paths. Negligible changes in the
learned/actual-policy rows mean THIS proxy is not their main measured explanation.
It does not prove absence of short redundant twigs,soma overrepresentation or
other multiplicity outside that definition. All per-neuron ratios/deltas and
category counts remain in the complete report and portable CSV.

## 5. Classical-anchored editing headroom

| Reference-assisted oracle | Segment F1_2/F1_6 | Resampled F1_2/F1_6 | Pooled length |
|---|---|---|---:|
| (a) Prune-only | 64.77/77.01% | 58.58/64.74% | .49075x |
| (b) Prune+connected additions | 68.56/80.43% | **61.74/68.54%** | .55105x |

The fixed(b)>=60% drafting gate PASSES by1.74pp. Prune-only does NOT reach it.
All151 boxes remain included;maximum final per-box length ratio is1.24205x,
within the stronger declared per-box1.25 bound. Final row graph hashes and
length checks pass. There are302 saved SWCs;reference-assisted geometry stays
explicitly named and is not promoted to inference.

The extra branches add3.16pp resampled F1_2. Largest gain is weak adult alias94:
8.51% to27.29%;it remains weak. Human Allen rises65.01% to68.03%. Frog andmouse
use the predeclared whole-neuron prune-only fallback. This is not exclusion.
Only55.49% neuron-macro reference length is covered within2 and resampled recall
is51.53% for(b);passing the draft gate does not mean complete reconstruction.

This is a feasible bounded oracle search,not a global optimum. References filter,
order and prune edges. The actual image-only candidate/branch policy needed by
a learner has not been built or validated by this result. No training learnability,
held-out learned development accuracy,whole-image assembly,radius or topology
quality is established. Development has been repeatedly inspected.

The user-authorized [editing contract draft](classical_anchored_editing_contract_DRAFT_20261106.md)
is now written for discussion only. It proposes scaffold-chain keep/prune plus
anchored branch additions,with an image-only unit pool and explicit uncertainty.
It does not adopt a stage,revive failed dense-head training or alter either gate.

## Evidence and next action

Portable [completed-diagnostic evidence](evidence/post_stage2_diagnostics_completed_20261106.json),
[native/physical40-neuron table](evidence/post_stage2_ridge_neurons_20261106.csv),
[decomposition per-neuron table](evidence/post_stage2_decomposition_neurons_20261106.csv)
and [editing per-neuron table](evidence/post_stage2_editing_neurons_20261106.csv).
No completed diagnostic needs rerunning. The only prepared timed command is
benchmark identity recovery/FIRST scoring on the downloaded archive. No new
model,architecture,labels,training or GPU run is authorized by this readout.
