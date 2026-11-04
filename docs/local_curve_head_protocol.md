# Bounded local curve-head experiment — 4 November 2026

The user approved the experimental Transformer bypass and restricted
trace-agreement supervision, including conditional CPU -> four-box -> sixteen-box
progression. Check in before 64 boxes or any GATv2 work. This is not approval to
replace the product architecture or launch full-cohort training. Existing inputs,
guides, checkpoints and historical verdicts remain immutable.

## Question and interpretation

Does a dense support/offset head on the existing PointNeXt/GN8 implementation
learn useful curve-local proposals within a declared budget? Head and objective
change together. A win measures this formulation's budgeted learning behavior;
it does not identify a sole cause, establish asymptotic superiority over a
Transformer, validate the encoder universally, or demonstrate final SWC quality.

The arm replaces the Transformer only inside this experiment. It uses the same
2048-point image packets, restored 32-channel PointNeXt features, and a shared
32 -> 64 -> 64 -> 4 ReLU MLP. One output is a support logit; three parameterize
an offset with norm strictly below eight native voxels. The encoder is trainable.
No radius or connectivity task is introduced. No reference is opened by model
inference, confidence selection or consolidation.

## Supervision contract

Support is max(0, 1 - distance_to_supplied_trace / 8). It is trace affinity,
not calibrated foreground probability. Far points in reference-present boxes
receive zero under this restricted reference-agreement task. This can penalize
real untraced branches. Reference-empty boxes remain unknown, never negative-only
training. Reference-present is not a claim of exhaustive annotation.

Nearest projections use original full-reference segments, with indexed search
covering the eight-voxel support tube plus projection-ambiguity margin. Segment
subdivision is acceleration only. Clipped-box endpoints are not projection
targets. Isolated reference vertices are geometric point targets. References
are source-hash checked and restricted to the existing 40-case manifest.

Offsets are supervised only within six voxels. Mask them if a projection within
0.25 voxel of the best distance lies more than 0.5 voxel from the chosen
projection. Do not average alternatives. Coincident projections can share a
geometric vector; they do not establish shared branch/component identity.
Ambiguity counts and unsupervised domains remain disclosed.

Support BCE is averaged equally over nonempty fixed source-distance bands
[0,2), [2,4), [4,6], (6,8), [8,infinity). Offset Smooth L1 uses residuals divided
by six, beta=1/6, averaged over eligible points/components. Loss coefficients
are one each. Offset loss is independent of predicted support and distance
support weight. No target count, class-adaptive loss, radius loss or new
augmentation. AdamW lr=1e-4, weight_decay=0, betas=(.9,.999), eps=1e-8;
gradient clipping=10; float32, deterministic native operations and no TF32.

## Consolidation and controls

Accept scores >= .25. Sort by descending score, tie by stable source identity;
suppress within one voxel of a retained vote. Keep a real vote and its feature,
never a coordinate average. This can collapse branches closer than the radius;
synthetic close-branch, Y-junction, crossing, boundary and permutation fixtures
report that limitation. No predicted edges or global welding are introduced.

Reuse cached Sato values at the same source voxels: rank, consolidate, and
report full and 512-capped output. This control does not predict offsets.
Reuse existing pruned classical graph packets, score their nodes as isolated
points; use deterministic geometric farthest-point reduction for a 512 cap
because those nodes have no confidence. It uses the fuller guide, not just
the 2048 selected voxels. Do not rerun filtering or classical extraction.
Classical coverage exists only on the already extracted 12 images; missing
training controls are explicitly unavailable, never zero or silently filled.

A reference-only diagnostic projects eligible unambiguous source points and
uses the same cutoff/consolidation. It checks label/policy attainability, not
a mathematical prediction ceiling. Ambiguous source regions are disclosed and
all reference curve length stays in the scored denominator. The diagnostic
never selects model predictions. Also score a 512 cap separately.

## Selection, comparison and metrics

Four-box order: training aliases 32/27,32/20,22/2,22/5 (the existing diagnostic).
Sixteen boxes include these and deterministic training-only diversity selection
from cached metadata. Cover new neurons first, source groups, sparse/dense
target counts, apparent contrast/width, junctions, boundaries and fragmented
references (23/66/75). Save the exact manifest before model fitting. No selection
uses development outcomes, new model outputs or oracle success. No reserved
identity is opened.

The four-box saved Transformer fit is historical context, not a newly paired
run. At sixteen boxes, train a fresh original-objective geometric Transformer
and fresh dense model from bit-identical encoder tensors. Same boxes/order,
exposures, optimizer recipe and evaluation policy. Heads differ. Both arms
restart at sixteen rather than warm-starting the dense arm's four-box fit.
Record actual compute; the smaller decoder may cost less per exposure.

Score accepted-point agreement and arclength coverage at two/six voxels with
existing segment-distance and .5-voxel quadrature utilities. Pool within neuron,
then macro-average. Preserve per-box/per-neuron results, counts, 512-cap views,
endpoint/junction/boundary incidence, gap lengths and apparent thin/dim coverage.
Raw and consolidated outputs are separate. Cutoffs .1/.25/.5/.75 are diagnostic
only. Source-distance bands report vector and vote-to-curve error and counts.

Point agreement uses the full supplied reference geometry, so a valid vote just
outside a crop is not falsely rejected because the training label used a halo.
Coverage uses clipped reference runs in the scored box; outside-box counts are
disclosed. Apply this same rule to all new arms/controls. It differs from the
old crop-only point diagnostic; do not rewrite or directly equate old scores.

At sixteen boxes evaluate all 415 development boxes at initialization,
midpoint and terminal checkpoints. Score agreement on the 151 reference-present
boxes; disclose outputs on the 264 unknown boxes separately. Unknown biological
precision is not assigned zero. All seven development neurons remain in macro
denominators. Development is repeatedly inspected, not an independent test.
No best-development checkpoint selection. For the paired proposal-count
diagnostic, rank/consolidate all hypotheses and select the same K per box,
K=min(512,available_dense,available_transformer); disclose K. Apply the same K
to available no-learning controls, reducing K for that comparison if necessary.
This is reference-free comparison accounting, not production node selection.

No isolated-point metric is named graph reconstruction F1. False joins cannot
be scored before predicted edges exist. Incident-branch spatial coverage is
not connectivity. Classical segment scores retain their historical meaning.

## Frozen gates and budgets

Machine-readable settings: `configs/local_curve_head.json`, source-bound before
CPU preflight and before GPU timing/training. CPU feasibility checks the chosen
four/sixteen boxes and the existing development grid, not the completed audit.

Four boxes: maximum 2000 updates (500 exposures/box), 2h wall including evaluation.
Terminal macro accepted-point agreement >=90%, curve coverage >=80%, coverage
>=90% of the reference-only diagnostic; no box below 80% of its diagnostic
coverage. Zero diagnostic coverage is a preflight failure, not a skipped case.
The four-box gate establishes fitting only; controls remain in the report.

Sixteen boxes: maximum 4000 updates/arm (250 exposures/box), 4h wall/arm including
evaluation/I/O. Terminal training agreement >=85%, coverage >=80%; development
agreement >=60%, coverage >=60%, worst-neuron coverage >=25%. At equal proposal
count require development coverage gain >=10 percentage points over the paired
Transformer, with agreement loss <=2 points; and >=5 points over each available
no-learning control, with agreement loss <=2 points. Compare controls only on
their common domains; all seven development classical controls are available.
These are engineering screening margins, not biological acceptance standards.
Structural-role support must not fall more than five points versus Transformer
on scorable roles; report unavailable/ambiguous roles. Thin/dim coverage must
not fall more than five points. A mean cannot conceal a development floor failure.

The proposed 64-box absolute floors are the same 85/80 training and 60/60
development percentages, but its identities, comparison and resource limits
require the user's next decision; the runner must reject 64.

CPU tests first (labels, units, ambiguity, consolidation, gradients, train/eval,
full optimizer/RNG reload, image-only inference). GPU timing: eight throwaway
updates/arm on the laptop RTX 3070 Ti, maximum ten minutes/arm. Timing changes
no final starting weights. Report measured update/evaluation cost and peak VRAM;
do not launch fitting if projected complete work plus 50% margin exceeds its
wall cap. Give the visible command and exact output paths before launching.

Stop on failed CPU/feasibility/timing/accuracy gate, NaNs or resource failure.
No automatic extension, new threshold, loss, point budget or strategy. Wall-cap
failure is budget-limited, not convergence. Correct independently demonstrated
implementation bugs with visible evidence; changed scientific settings require
a new prospective decision. Save model/Adam/RNG/cursor and immutable logs under
`tmp/local_curve_head_20261104/`; no existing results are overwritten.
