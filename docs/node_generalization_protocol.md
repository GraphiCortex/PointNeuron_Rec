# Geometric-query generalization pilot: proposed protocol

31 October 2026. The user asked to proceed from the successful
[four-box learnability result](geometric_query_fit_results.md) to a broader
neuron-disjoint training/development study. **This is a concrete protocol for
review, not an implemented broader training runner or a launch command.**
The model is frozen as a design; the data/supervision/resource policy is made
explicit before substantial cache/runner implementation.

Current implementation status, 2 November: the [40-image cache and independent
review](node_cloud_cache_completion.md) are complete, covering all 2379 boxes.
The broader loader, metrics, reference-target cache and training runner are not
yet implemented. The user requested a matched larger-point-budget comparison;
2048 is a provisional baseline. The original single-budget protocol below is
retained as a proposal, not an approved launch recipe or selected final budget.

## Research question and scope

Can the current geometric-query PointNeXt/Transformer learn across the retained
33 training neurons and produce accurate, selective proposals on seven
gradient-disjoint development neurons under the same automatic image-only input
rule? The study covers node positions/confidence. It does not train GATv2,
consolidate overlapping crops or produce a qualified final SWC.

The architecture remains raw image -> automatic point cloud -> PointNeXt ->
geometric-query Transformer -> future consolidation/GATv2/graph decoding.
PN1 remains conceptual/historical context; no PN1 model arm is required.
This first broader fit characterizes the frozen candidate. It is not a matched
larger-data comparison proving that Sato, PointNeXt or geometric queries are
individually superior to alternatives.

## Cohort and input manifests

The [completed preflight](node_cohort_preflight_results.md) counts:

| Role | Neurons | All image boxes | Local reference targets present | Reference-empty / unknown |
|---|---:|---:|---:|---:|
| Training | 33 | 1964 | 460 | 1504 |
| Development | 7 | 415 | 151 | 264 |

The existing 33/7/8 identity partition is retained. Development aliases are
19/21/35/44/79/94/135; they are repeatedly exposed development material, not a
fresh test. Reserved source files remain unopened. Exact-image-content
duplication checks and acquisition-family review are still needed before a
strong independence claim; sample-ID separation alone is insufficient.

Separate visible inputs:

- `configs/node_generalization_images.json`: 40 explicit image-only entries;
  no reference paths, target counts or annotation-based crop selection.
- `configs/node_generalization_references.json`: explicit target/evaluation
  inputs and source hashes; only training rows permit gradients.
- `configs/node_generalization_protocol.json`: the proposed scientific and
  resource settings. It is explicitly not an executable fit recipe.

No filesystem scan chooses alternative images/SWCs. No reference selects an
inference crop, its cloud or its proposals. Every full image is covered by the
same disjoint clipped 128-voxel grid used in the sanity check. This is a
whole-image **tiled node-task** evaluation, not cross-crop graph reconstruction.

## Frozen model and preparation rule

Same native GN8 PointNeXt prototype, 512 geometric queries, three-layer
128-channel Transformer and 1,688,518 parameters. Fresh encoder seed 0 and head
seed 101 retain the successful design's initial parameter values. The fitted
four-box checkpoint stays as a regression reference; its learned weights/Adam
state are not transferred into the first broader fit.

Each box receives 2048 distinct source voxel locations, full-cloud attention
memory, and XYZ/intensity features under the existing crop-center/scale-64 and
whole-source affine intensity normalization. Sato scales remain 1/2/4 voxels,
with 0.1 uniform probability mass in the same exponential-race sampling rule;
cloud seed and source-hash ordering seed remain zero. No hard mask, guide input
channel, point jitter, foreground label, new budget or per-source selector is
introduced. This fixes Sato for this study, not as a universal product winner.

Input-cloud, query and future graph budgets remain distinct. The maximum target
count is 419 in training and 233 in development, so all audited boxes fit the
current 512 queries. Future overflow is an explicit failure, never permission
to truncate targets or choose reference-dependent smaller boxes.

## Supervision and exposure

Use the unchanged eight-voxel full-chain target grid, required endpoints,
junctions and true crop-boundary nodes. Preserve coincident identities and
fragmented encodings; do not silently weld them. Radius remains unsupervised.
Type/radius metadata do not become verified biological labels.

Train on the 460 reference-present training boxes. Reference-empty crops
receive no negative-only loss; they remain unknown. This is a supervision
eligibility rule, not an inference mask. All 1964 training boxes and all 415
development boxes remain available for terminal output accounting.

**Twenty complete epochs = 9200 updates**, batch one. Each supervised box is
used exactly 20 times, once per epoch, under a deterministic epoch permutation
of a stable case-ID/tile-ID list. No box is selected by model error, branch
brightness or development performance. No development case enters gradients.
This is equal exposure per box, not equal exposure per neuron: individual
neurons have 3-80 supervised boxes, or 60-1600 updates over the fit. The exact
per-neuron exposure table is saved in the [plan evidence](evidence/node_generalization_plan_20261031.json).
Macro per-neuron evaluation prevents large images alone from determining the
reported score, but does not erase this training-distribution imbalance.

Keep AdamW LR 0.0001, zero weight decay, betas 0.9/0.999, epsilon 1e-8,
global gradient clipping 10, balanced used-slot loss and center loss unchanged.
No augmentation, dropout, scheduler, extra confidence stage or new loss is
added. These are controlled starting settings, not established broad-data
optima. More diverse samples share the update budget; a tiny memorization
task's 500 exposures per box are not promised here.

Save full model/Adam/RNG/data-order state before evaluations at epochs
0/5/10/20 (updates 0/2300/4600/9200). The fixed terminal checkpoint is the
primary verdict. Intermediate development scores diagnose trajectories;
they do not select a better checkpoint, cutoff or run extension.

## Evaluation and proposed screening milestones

Ordinary inference, cutoff 0.5. Run every development image box regardless of
reference presence, signal appearance or expected target count. Pool each
neuron's per-box one-to-one counts, then report macro precision/recall/F1 across
the seven neurons. Keep every neuron's counts and source-group stratum visible.
Crop-boundary labels are part of this specified grid task; do not call their
pooled counts unique whole-neuron nodes.

Primary node tolerance stays **two native voxels**; also report six-voxel
results. Include unmatched proposals from reference-empty boxes in full-grid
reference-agreement precision. Separately show reference-present-only scores
and outputs in unknown boxes. Unknown annotations do not justify declaring
those predictions verified biological false positives, or hiding them.

These are proposed engineering screening targets, to freeze before learned
development results, not universal biological quality standards:

| Purpose | Proposed terminal criterion |
|---|---|
| Interpret whether broader training itself learned | Reference-present training macro node precision and recall each >=90% at two voxels |
| Pilot development signal | Full-grid development macro node precision and recall each >=80% at two voxels |
| Avoid hiding a failed neuron in a mean | Minimum per-neuron development F1 >=60% at two voxels |

The 95% per-box memorization gate remains unchanged in its original report.
These new-cohort criteria do not relabel an old failure or alter inference.
They are not implemented or represented as accepted software qualification yet.

Required explanatory diagnostics:

- All-query versus accepted-query geometry and score distributions, plus
  duplicate/near-duplicate counts and accepted/reference cardinality.
- Accepted-point distance to the continuous reference curve, at two voxels;
  arclength-weighted reference-curve support by accepted nodes at two/six
  voxels, alongside support from the target nodes themselves.
- Endpoint/junction support, worst neurons and the known annotation-encoding
  cases; radius is excluded from anatomical scoring.
- Apparent-width and contrast strata measured from image profiles under the
  existing evaluation-only `branch_evidence.py` conventions. Use training-only
  arclength-weighted 25th-percentile boundaries before predictions, then apply
  them unchanged to development. Sample reference windows every four voxels;
  retain the existing profile radii 1/2/4/8, half-length 2, background shell
  8-12 and bin width 0.5 voxels. Normalize inner contrast by the observed
  whole-source intensity range. Width uses resolved image half-height profiles;
  report unresolved profiles rather than assigning an invented width. SWC
  radii are not assumed to be true caliber. These covariates never enter sampling.

Interior eight-voxel grid phase can depend on chain structure outside a crop.
Strict node mismatch can therefore differ from missing branch geometry. A good
curve result with a poor strict-node score is a formulation/evaluation finding,
not an automatic two-voxel pass. Six-voxel/curve diagnostics must not silently
replace the primary criterion after results are known. Calibration remains
unverified; macro scores are native-voxel engineering summaries, not a common
physical-resolution or universal reconstruction claim.

## Resource plan and implementation sequence

At the observed 1.5608 seconds/update, 9200 updates alone take about **3.99 hours**.
Planning target: roughly 4-6 GPU hours including evaluation, to be checked on
the actual broader loader/metrics before launch. Propose an external eight-hour
wall allowance, separate from the scientific update cap. If measured cost
requires a different allowance, explain it before launch without silently
changing the 9200 updates. Device: RTX 3070 Ti Laptop GPU, deterministic FP32.

Raw-image preparation is a **separate substantial CPU task**. All guides would
occupy about 10.86 GiB and a largest-image twelve-float-array working payload
about 7.33 GiB of disk, before other cache/output costs. Many CPU hours are
possible; the GPU estimate does not include them. No bulk-preparation duration
has been measured. Do not start all 40 filters on that assumption.

1. Implement the image-only cache builder using the existing decoder/guide/
   sampler, separate from the reference-target builder. Hash full source images,
   check cross-split exact duplicates, and reuse existing PN2 guides only when
   all source/normalization/recipe/content identities match. Preserve original
   files. Process one image at a time under ignored `PN2/tmp`.
2. Verify sampled source IDs/intensities, actual bounds and frozen target
   counts; reproduce the original four Sato input packets where applicable.
   Test missing-supervision handling and the 419-target training stress case.
   Measure an uncached preparation case before a visible full-cohort CPU command.
3. Implement the broader loader and metrics, including reference-empty
   evaluation without empty-positive reductions, deterministic exposure order,
   exact state/data-order reload and terminal per-neuron reporting. CPU tests
   and a visible throwaway GPU timing must precede a fit command.
4. Provide the user a concrete manual 9200-update fit command with measured
   duration, external wall allowance and fresh output folder. No full fit or
   cache generation is launched by this protocol.

Generated guides, point/target caches, schedules, checkpoints and logs belong
in ignored `PN2/tmp`; source/configuration/readable reports remain visible.
The current [exposure-plan utility](../scripts/plan_node_generalization.py)
already generates and validates the 9200-row schedule without a model or images.
It is not a replacement for the missing cache builder or training runner.

## Interpretation and stopping

- Strong training agreement with weak development agreement indicates a
  transfer/generalization problem to examine, with input-retention and
  reference-definition diagnostics retained. It does not warrant more fitting
  on the seven development neurons.
- Weak training and development agreement leaves broader learnability,
  exposure/optimization or representation unresolved; it is not an isolated
  generalization verdict. Stop at the declared cap and review evidence.
- Passing the proposed development milestones provides development evidence
  for this node predictor under this input rule. It does not qualify radius,
  GATv2, crop consolidation, final SWC topology or unseen-domain robustness.

Nonfinite values, incompatible state/data order, coordinate mistakes, overflow,
missing cases or reserved-data access stop explicitly. No automatic retries,
target pruning, mask corrections, loss/threshold searches, update extensions or
GAT handoff follow a failure. No genuinely untouched test data is consumed.
