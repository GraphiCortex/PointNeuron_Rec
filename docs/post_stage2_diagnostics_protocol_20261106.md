# Post-Stage2 CPU diagnostics — fixed before results

Stage2 stays STOPPED. Its gate,STOP.json,models,packets,labels,config and protocol
are read-only. No exclusion,training,GPU,retuning or architecture adoption.
These new outputs live ONLY under `tmp/post_stage2_diagnostics_20261106`.
The user launches every timed action locally using the existing isolated
`tmp/stage2_cpu_env`. Results select no model or parameter. The only decision is
the user's new editing-headroom gate,which permits drafting a contract,not training.

## 1. Gold-to-intensity-ridge offsets

All33 training and7 development neurons. Original images are decoded again for
this distinct image diagnostic;no Phase1 filtering/cache construction is repeated.
Verify original source and decoded hashes. Use full supplied reference polylines,
not only the611 reference-present boxes. Report reference arclength outside the
image separately,but do not remove it from the offset denominator. Map outside
queries to boundary search blocks without moving the actual query coordinates;
missing image ridges are censored at the same declared radius. Sample edges at midpoint step<=.5 native
voxel,weighted by arclength. Neither reference direction nor distance builds ridges.

Fixed operational ridge:normalize by the retained whole-image intensity range;
Gaussian sigma1 native voxel,truncate4,reflect at actual image boundaries. Compute
the3D intensity Hessian,correct the truncated diagonal derivative's nonzero DC
response,require its second-smallest eigenvalue<-1e-6 and smoothed normalized
intensity>=.05. Require transverse nonmaximum suppression along each of the two
most-negative eigenvectors at+/-1 voxel using trilinear interpolation. Ridge
locations are integer voxel centers;quantization and sigma are limitations.
This is an intensity-ridge proxy,not certified anatomy or a revised gold trace.

For each reference sample find the nearest ridge within12 native voxels. Scan
only fixed64x64x32 query blocks containing reference samples,with12 search halo
plus6 derivative/NMS halo;the ridge operator itself is image-only. The search
uses neighborhood availability,not a reference-selected local maximum. Missing
ridges are censored as>12. Report per neuron shares of reference length strictly
>1,>2,>3 voxels off-ridge,weighted median/p90/p95,mean-distance lower bound,and
censored share. Null quantiles mean censored,not an exact infinite distance.

Physical view:join the authors' [scaling_gold.csv](https://github.com/lmanubens/BigNeuron/blob/main/scaling_gold.csv)
by image/reference name,record all matching table IDs and cross-check available
retained sidecars. Metadata joins cover40/40 approved neurons and agree with the
previous12 training sidecars. Reported XYZ=(xy,xy,z);this is published metadata,
not independently verified acquisition calibration. Missing/conflicting metadata
must remain unknown. Do not assume1um/voxel. Use independent anisotropic nearest
ridge distances with physical search radius12*min(spacing),so the finite search
is complete within that physical radius. Weight physical results by physical
reference arclength;report .5/1/2um exceedance only when below the search radius,
plus physical quantiles/censoring. Native results remain primary and unchanged.

## 2. Fixed visual panel

Choose WED_SLP plus the next two highest C1 CV per-neuron length ratios,and the
two lowest,from the already completed run. Frozen identities:

- WED_SLP_PN_neuron.
- Zebrafish larva hard_1_1_Live_2-2-2010_9-52-24_AM_med_Red.tif_uint8.
- Zebrafish larva hard_1_5dpf_Live_1-30-2010_12-39-26_PM_med_Red.tif_uint8.
- Adult zebrafish hard_hard_1_10_Ch2.
- Adult zebrafish1_9_Ch2.

For each,write XY/XZ/YZ max projections of the original full image,gamma.5 for
display only,with separate reference,retained held-out nodes,C0 andC1 overlays.
All saved boxes are shown in their original global frame;they are not merged
into a new model output. No rendering point cap or geometry modification.
Flag up to5 longest C1 chains of length>=10 with>=80% samples farther than6
from the reference AND>=80% raw-intensity samples>=.1 normalized. These are
**possible untraced image structures**,not automatic neurite labels. Orthogonal
projections can hide depth overlaps. Generated packets explicitly remain
`visually_reviewed:false`;after the local launch the agent/user must inspect
the PNGs before calling anything obvious. No relabeling follows visual review.

## 3. BigNeuron comparison and coordinate check

The supplied local gold166 download contains gold/manual traces and alternate
annotations,not identified per-algorithm benchmark outputs. BigNeuron publishes
the latter in a [separate release](https://github.com/BigNeuron/Data/releases/tag/gold166_bt_v1.0).
Freeze the published `optimization_yes__20161101_reconstruction_for_gold166_rhea_5571.tar.gz`
collection,726,174,044 bytes. It is published parameter-optimized external context,
not an independently untuned baseline. Download is a separate timed user action.
Save archive SHA256 (release has no published digest),extract only the seven
approved development identities,never reserved SWC contents. No algorithm is run.

Match archive path-component prefixes to original image names,never substring
SLP inside WED_SLP. A duplicated Cell1 image basename requires an exact source
group or published gold identity;ambiguous members are not parsed/extracted.
If any case lacks unambiguous algorithm outputs,stop this scoring action with an
identity-mapping report;do not substitute gold traces or fabricate seven-case scores.
Keep exact algorithm/parameter suffixes as variants. Duplicate case/variant files
require inventory review,not best-replicate selection.

Before each score,validate SWC topology and finite coordinates,original image
identity,standard SWC XYZ order and native bounds. Fixed transform is identity:
no fitted translation,axis permutation,rescaling or gold-dependent alignment.
Require>=99% vertices inside native image cell bounds;failures count as zero
with explicit coordinate-failure status. Bounds alone cannot certify origin or
subvoxel alignment;record that limitation. Interpret scores as provisional
native-convention comparisons pending visual alignment confirmation,not a
calibrated independent benchmark claim.

Primary comparison:the same151 development boxes and original segment scorer
(.5 midpoint quadrature,2/6vox) plus the same2-voxel-arclength resampling/maximal
bipartite matching at2/6. Aggregate within neuron then macro over seven. Also
report separate whole-image-clipped scores,never mix them with per-box scores.
Every algorithm variant retains all seven cases;missing/empty/invalid outputs
count as zero. Report per-variant/per-neuron results,best fixed variant and median
variant macro at each tolerance/metric. No best-per-neuron composite is adopted.

## 4. Resampled-F1 decomposition

Use saved C0/C1 training-CV graphs and BOTH saved development policy diagnostics
(image scaffold and conditional edge oracle). Keep the cohorts separate.
Clip first and replay original resampled counts/lengths exactly before ablation.

- Isolated nodes:degree zero in the clipped graph.
- Spurs:leaf-to-junction maximal degree-two paths strictly shorter than6 voxels;
  a complete short linear component is not called a spur.
- Duplicate-path proxy:process maximal paths by decreasing length/index. Flag a
  shorter path if>=80% of its .5-voxel samples are within2 of an earlier nonflagged
  path with absolute tangent cosine>=.9. Exclude points within2 Euclidean voxels
  of its endpoints to avoid classifying a shared junction as a duplicate. Genuine
  close parallel neurites can still be flagged;this is geometric redundancy only.

Report original,minus-isolates,minus-spurs,minus-duplicates,minus-isolates-and-spurs,
and minus-all-three. Keep untouched original isolated identities in independent
edge ablations;drop orphan vertices created by removed edges. Report counts,
flagged lengths,category overlap,per-neuron F1 and length ratios. Independent F1
deltas are not additive causal percentages:matching,resampling and junction
identities change. The fixed sequence original->minus-isolates->minus-isolates-
and-spurs->minus-all supplies an ordered diagnostic as well. Nothing is adopted.

## 5. Classical-anchored compact editing oracle

Only151 development boxes;reference-assisted measurement,never inference labels.
(a) Start from the exact preserved pruned classical graph. Score existing edges
by the original .5-voxel quadrature agreement at2;retain agreement>=.5 and positive
in-box length. Minimum spanning forest uses length*(1+4*(1-agreement));all edges
remain a subset of the original classical inventory. No classical vertex moves.

(b) Augment with existing Stage1 Sato/learned vertices and candidate edges touching
nonclassical vertices. Map existing pooled classical representatives back to exact
original classical vertices within1,not to new averaged positions. No new
classical-classical shortcut or new neighbor search. Additional edges require
the same>=.5 overlap and positive in-box length. Starting from the classical
forest,add them in increasing fixed oracle cost,never joining two already
classical-rooted components. Drop unattached components;new branches must attach
to the classical graph. This is deliberately a constrained editing opportunity,
not another unconstrained union set predictor.

Clip geometry first. Both arms may prune nodes/edges;no moved nodes or invented
paths. Optimize actual2-voxel resampled F1 with a predeclared bounded search:
up to128 accepted terminal-path deletions,consider16 candidate paths per step,
ranked by unsupported length then total length/index. Accept the highest-F1
deletion if it improves F1 or is needed to meet budget. If still over budget,
drop complete components in increasing reference-support fraction/decreasing
length/index order. Enforce<=1.25x PER BOX,which also guarantees pooled<=1.25x;
report the stronger per-box constraint. Empty graphs are feasible and score zero.
Remove any added component that loses all original classical anchors. Preserve
the prune-only solution as a feasible fallback for(b),so(b) cannot score worse
than(a) on a box's objective. This is a feasible lower bound on optimal oracle
headroom. Also retain the entire(a) solution as an oracle fallback per neuron,
because per-box F1 improvements can change pooled matching denominators;record
any such fallback and preserve the unused candidate(b) files. This ensures(b)
contains(a)'s feasible solution on the gate's aggregation scale. This is not a
learned-model choice. The bounded search is not a globally optimal solution.
A failed search does not prove impossibility.

Export explicitly reference-assisted SWCs and score with the unchanged segment
and resampled scorers;report per-box/per-neuron ratios and sizes/provenance.
Fixed gate:**draft an editing contract only if(b) development neuron-macro
resampled F1_2>=60% with the length constraint met**. State whether(a) alone
reaches60. No automatic drafting,training or continuation is implemented;the
user reviews results first. Stage2's failed gate and STOP remain unchanged.

## Resources and execution

No job is launched by implementation/preflight. Caps:ridge_review7200s,
benchmark_download1800s,benchmark_score7200s,decompose7200s,editing14400s.
Actual runtime unmeasured:ridge and decomposition may take tens of minutes to
hours;editing may take hours;download depends on the network. The own-process
watchdog enforces caps including sleep. Four compute threads,CUDA disabled,no
Torch imports. One action at a time,write-once folders,no automatic retries or
extensions. A partial result is not a passed gate. Review failures rather than
deleting evidence. Frozen config binds code,metadata,input manifests and original
Stage2 STOP. No new development training-label packet is created.
