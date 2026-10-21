# Research reset: the evidence we should build on

Active document for the [fresh PN2 workspace](../README.md).
New source, scripts, configs, tests and reports belong under `PN2/`.

21 October 2026. Read-only research and saved-artifact audit supporting the
[revised proposal](revised_proposal.md). No model, loss, input policy, threshold,
training script or historical verdict changes in this reset. No training or
model inference is run. The original proposal PDF remains unchanged.

## 1. The direct answer to "are we better than PN 1.0?"

**We have not measured a fair end-to-end PN 1.0 versus native PN 2.0 comparison.**
This is an evidence gap, not a measured tie or proof that PN 2.0 is worse.
There is real engineering progress, useful negative evidence and substantial
local fitting progress. None supplies the missing matched comparison.

| Record | Actual scope | Supported conclusion |
|---|---|---|
| Historical promoted PN 1.0 | 63 stable-domain whole-neuron outputs, original code and training history | Functional automatic reconstruction; spatial F1 0.8150, precision 0.9699, recall 0.7195 at six voxels |
| Those historical outputs excluding training neurons | 55 of the same repeatedly evaluated neurons | Spatial F1 0.8051; the useful result is not explained away by the eight training-neuron overlaps |
| Initial mixed-domain PN 1.0-style control | Seven development whole neurons; neuron-grouped split, then defective decoding | Spatial F1 about 0.3743; not a corrected-input control |
| Repaired PN 1.0-style local control | Seven development whole neurons, corrected input/range/local-coordinate path | Spatial F1 0.6598 or 0.6671 at graph cutoffs 0.85 or 0.5; not an isolated native-PN2 architecture gain |
| Learned field and sampling experiments | 33 training and seven development neurons; component outputs | Some substantial point-support gains, heterogeneous failures; no learned-pipeline final-SWC result |
| Latest native PointNeXt/Transformer/GAT fit | Four crops from three training neurons: 0, 92, 86 | 440 assigned canonical nodes and 422 required edges retained, one extra spur, three complete crop passes; no held-out or whole-neuron native-PN2 result |

The rows use different datasets, protocols and in some cases metrics. They
cannot be ranked by their headline numbers. In particular 0.8150 versus
0.6671 is not a valid PN 1.0 versus PN 2.0 comparison. Neither whole-neuron
figure can be compared with a four-crop training gate.

## 2. What PN 1.0 actually succeeded with

The promoted local system is DGCNN/EdgeConv pointwise skeleton proposals,
aggregation and adaptive foreground-geodesic graph construction, followed by
SWC export. Its promoted checkpoint is
`tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt` in the
original checkout. The graph recipe uses `adaptive_connected_coverage_nms`,
confidence 0.85, distance NMS 18 and at most 128 graph nodes.

The learned GAE was tested but **was not the promoted connectivity path**.
Its 46-record evaluation is on records used in fitting, not an independent
validation cohort: threshold edge F1 0.4983; top-k diagnostic F1 0.5200 versus
initialized top-k F1 0.8261. The top-k count uses reference information and is
not an automatic inference policy. This rejects that fitted implementation;
it does not establish that learned connectivity cannot help.

The stable 63-output induced-edge F1 is 0.8295, with mean bridges 1.4444 and
foreground reachability 0.9504. **That edge F1 is not the spatial F1 0.8150.**
The target is induced on selected predicted nodes and can overlook absent
branches. Local ESA/DSA/PDS implementations are approximations, not verified
official Vaa3D plugin equivalents.

Useful PN 1.0 lessons are to preserve local image-supported geometry, evaluate
the exported SWC, compare against the working initialized graph, and reject
longer training or learned modules when reconstruction worsens. Its high
precision with incomplete recall is a concrete improvement target.

## 3. Historical independence and input provenance

The actual guarded30 split contains 618/132/133 training/validation/test
**patches**, from 48/44/41 neurons. Every validation and every test neuron
also occurs in training. The original splitter shuffled patch paths rather
than grouping neuron identities. Stable-domain evaluation 100-162 includes
training neurons 102, 108, 126, 135, 143, 149, 158 and 162.

The current PN 2.0 partition correctly separates 33/7/8 neuron identities.
However, **all 48 of those identities had patches in historical PN 1.0
training**, including the seven development neurons and eight reserved neurons.
The eight PN2-reserved identities are 2, 26, 37, 48, 50, 52, 80 and 84.
Existing historical output summaries additionally document prior runs on
2, 48, 50, 52 and 84. This reset reads split/run metadata only, not reserved
images, references, cached arrays or predictions.

Consequences:

- Historical PN 1.0 remains a useful frozen practical reference with disclosed
  training exposure, not an unseen-data comparator on the current seven/eight.
- The eight remain PN2-reserved internal cases for a fresh model trained only
  on the current training partition. They are not globally untouched project
  data and should not be described as such.
- A controlled learning comparison needs a freshly trained PN 1.0-style
  reference under the same neuron-level restrictions. New external data are
  needed for convincing independent product confirmation.

The original `main` also retains the decoder/range/coordinate patterns already
fixed in PN2: reversed PBD8 packed-difference order, incorrect PBD16 difference
handling, uint16 storage range used as signal range, and absolute XYZ in the
raw proposal head. Historical outputs and scores remain historical evidence;
they are not silently reinterpreted as corrected-input experiments.

The early hard-case oracle controls expose failures under that historical
preprocessing and forced-tree recipe. Four of five cases still fail with
reference-derived nodes; sample 22 improves strongly. These controls do not
isolate graph architecture as the sole cause, nor prove that the same failures
persist after corrected decoding. Repeating them is a prospective decision,
not a justification for erasing the earlier work.

## 4. What the PN 2.0 experiments establish

### Keep the demonstrated repairs and negative findings

- Correct decoder fixtures, explicit image scaling and crop-local coordinates
  address confirmed defects. They are foundational repairs, not optional
  architectural performance claims.
- Train/eval normalization caused a large inference discrepancy on fitted
  PointNeXt crops. Explicit per-crop normalization is a documented experimental
  policy; it is not validated universally by that diagnosis.
- The native PointNeXt-S topology/descriptor audit matches its reference wiring
  on the audited synthetic CPU case, including 790,720 encoder/restoration
  parameters and zero descriptor delta. Official CUDA-kernel equivalence,
  task suitability and published throughput remain unproved.
- Native GATv2 attention already agrees with an independent float64 loop in
  the recorded forward/gradient checks. Symmetry, permutation, translation and
  reload checks also pass. This is useful implementation evidence, not evidence
  that its edge supervision or full reconstruction task is solved.
- The DGCNN/query 30-epoch pilot accepted no queries at 0.5 on all 47 annotated
  development crops and six inspected training crops. That particular
  implementation/recipe failed; it did not test the full native PN2 pipeline.
- Attention helped a targeted field dilution case but reduced mean two-voxel
  development field coverage from 92.21% to 89.67%. Dice improved slightly.
  The tested attention and quota rules are not general winners.
- At matched actual point counts and nominal 2,048, field expansion increased
  development two-voxel input support from 71.47% to 85.47%, while proximity
  within six voxels decreased from 78.13% to 74.70%. This is component evidence,
  not a final reconstruction ceiling or proof of biological false positives.
- The hard-negative 50/25/25 variant regressed. It remains rejected under its
  protocol. Do not rediscover it under a new name because extras persist.
- Oracle-selected GAT fits establish conditional edge fitting only. Subsequent
  automatic integration exposed the effect of imperfect proposal selection.
- Canonicalization fixed real near-endpoint representation problems while
  preserving structural incidence. Its latest matched fit has small measured
  training-case curve gains, but both original terminal verdicts remain failed.

### Do not inflate explanations into established causes

Set prediction of a deterministic skeleton graph is a legitimate hypothesis.
Interior-node placement conventions create an additional learning obligation;
they do not prove the Transformer formulation is intrinsically invalid.
Matching changes have not been shown to cause harmful optimization instability.

The current query head is DETR-inspired, not an exact DETR or 3DETR reproduction.
It uses learned embeddings as initial decoder content, adds learned XYZ features
once to memory and supervises the final output. Official DETR uses zero initial
content with positional/query embeddings supplied to successive attention
layers; 3DETR uses sampled query locations and Fourier positional embeddings.
These are testable implementation/optimization differences, not established
bugs or causes of the saved failures. A faithful reference control could be
useful if query optimization remains the measured bottleneck; adopting every
reference detail at once would create another unisolated change.

The balanced confidence objective gives classes equal total weight, not each
query equal weight. For 11 versus 408 targets among 512 queries, the coefficient
on one negative is 0.5/501 versus 0.5/104. The latest extra at score 0.67917
contributes about 0.00227 to the weighted proposal objective. This helps explain
why small aggregate loss need not imply exact selection; it does not establish
that changing class weights is the cure.

Known-edge supervision masks unmatched/untrusted endpoints. The latest spur's
attachment is therefore unsupervised as an edge, while its query is a negative
for proposal confidence. Graph messages can still propagate gradients through
its features. The gap is a conditional supervision/interface issue, not absence
of every training signal or a proven defect in GATv2 itself.

The current objective does not directly optimize the hard final selection and
decoded SWC. That is common in structured prediction and not automatically a
bug. It makes final-output evaluation and realistic training graph context
necessary. Adding cycle penalties cannot eliminate a wrong but acyclic spur.

## 5. Dataset and product constraints

Gold166 source diversity, weak or fragmented references, uncertain calibration,
dim branches and background structures must enter the design explicitly.
The selected 48-neuron partition does not represent all Gold166 or all future
microscopy. Current fixed crop caches are not exhaustive whole-volume tests.

Reference-empty crops are unknown, not certified background. In reference-
containing crops, query no-object labels still assume reference completeness.
The field loss uses a different weak-label policy: distant bright voxels are
unknown; a trace tube is positive and some annulus/dark voxels are weak negatives.
Neither policy establishes biological background everywhere.

Distances currently use native voxel coordinates and spacing (1,1,1). Tube
radius 2, target spacing 8 and graph radius 8 are research settings, not shared
physical scales across acquisitions. Native input points, query slots,
cross-crop proposal limits and graph nodes are different budgets. Current
PointNeXt N>=256 support is a prototype restriction, not a universal input
contract. All automatic tiling, short-cloud, overflow and stitching behavior
must be specified before broad deployment.

## 6. Implementation and evaluation checks that should precede another claim

An implementation can pass local tests yet solve the wrong task; synthetic
fixtures can also repeat the same misunderstanding as production code. The
decoder investigation is a concrete example. Use independent references for
format/operator behavior, and invariance tests for coordinates, units,
subdivision, crop boundaries, selection and serialization.

The inherited paper-style evaluator skips rows marked `swc_valid=False` and
scans the dataset before selecting requested rows. Do not use that path for a
new reserved-data audit. A common evaluator should consume explicit image/SWC
manifests, retain all declared failures, and compare continuous geometry with
arclength weighting. `representation_oracle.py` supplies reusable geometry
primitives, not a ready-made arbitrary-reconstruction topology benchmark.

## Sources and reproducibility

The [metadata and saved-score evidence](../../pointneuron2/docs/pointneuron2/research_reset_20261021/evidence.json)
records source hashes, exact split overlaps and independently reaggregated PN1
scores. Source images and SWC annotations are not opened by that metadata check.

- Original checkout: [PN1 report](../../docs/pointneuron1_report.tex),
  [promoted-system README](../../README.md),
  [patch splitter](../../src/pointneuron/data/splits.py).
- PN2: [input repairs](../../pointneuron2/docs/pointneuron2/root_cause_findings.md),
  [seven-neuron whole-volume control](../../pointneuron2/docs/pointneuron2/validation_graph_cutoff_results.md),
  [DGCNN/query pilot](../../pointneuron2/docs/pointneuron2/query_pilot_results.md),
  [field sampling](../../pointneuron2/docs/pointneuron2/field_pilot_sampling_results.md),
  [attention comparison](../../pointneuron2/docs/pointneuron2/field_attention_results.md),
  [latest canonical joint result](../../pointneuron2/docs/pointneuron2/canonical_joint_fit_results.md).
- Implementation: [native PointNeXt audit](../../pointneuron2/docs/pointneuron2/pointnext_reference/topology_audit.json),
  [GAT implementation checks](../../pointneuron2/docs/pointneuron2/gatv2_connectivity_implementation.md),
  [balanced query loss](../../pointneuron2/src/pointneuron/models/balanced_query_loss.py),
  [partial joint loss](../../pointneuron2/src/pointneuron/models/partial_joint_loss.py),
  [edge labels](../../pointneuron2/src/pointneuron/data/connectivity_targets.py),
  [forest decoder](../../pointneuron2/src/pointneuron/graph/connectivity_decoder.py),
  [continuous geometry](../../pointneuron2/src/pointneuron/graph/representation_oracle.py).

The objective of the reset is to replace the local-repair loop with a common
benchmark, a coherent candidate and bounded decisions. Earlier failed gates,
checkpoints and useful code remain intact.
