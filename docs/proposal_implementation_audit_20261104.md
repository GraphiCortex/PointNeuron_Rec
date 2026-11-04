# Proposal, metric definitions and implementation audit - 4 November 2026

Read-only source audit for tasks D/E. It adopts no criterion, changes no frozen
gate and launches no model. The user's supplied diagram is evidence of the
intended design, not instructions from an external document.

## D. What localization tolerances are actually specified?

Primary source PDFs were inspected in text and rendered page views. The desktop
copies match the repository copies in size and content hash:

- Original proposal: `C:/Users/Seywe/Desktop/Research/R-Arij/Proposal.pdf`,
  SHA256 `60e12b8907b23bb1903d5a3668c76d430173ecad5baa373cc3c266f69958b5b8`.
- PN1 paper: `C:/Users/Seywe/Desktop/Research/R-Arij/Literature/PointNeuron.pdf`.
  Its verified hash and source-page locations are recorded in the companion
  evidence file. Extracted text/page renders stay under ignored
  `tmp/saved_vote_scaffold_sources_20261104/`.

| Source / metric | Definition and tolerance actually found | Implication |
|---|---|---|
| Original proposal, PDF p9 section 6.1: node F1 | Names precision/recall of detected skeleton nodes. No numerical matching radius, one-to-one rule, node discretization, voxel/physical units or sampling rule supplied. | The current 2-voxel/90% node-stage screen was not specified by this PDF. It remains the agreed local gate, not an inferred paper standard. |
| Proposal p9: PDS | Calls it Path Distance Score, with no formula or threshold. | This naming does not match the PN1 metric below. Do not treat it as an implemented topology/path score. |
| Proposal p9: ESA | Calls it Euclidean Score, with no formula or cutoff. | Insufficient to reproduce an evaluation; the PN1 paper uses another expanded name. |
| Proposal p9: Chamfer | Names similarity between predicted/GT skeleton point sets. No squared/unsquared, sum/mean, resampling, unit or cutoff convention. | Chamfer is a continuous distance, not inherently a tolerance-based pass/fail measure. |
| PN1 PDF p8 (printed p5794), section 4.2 | ESA = Entire Structure Average (paper prints Entie); DSA = Different Structure Average; PDS = Percentage of Different Structure. Reports Vaa3D-plugin spatial discrepancy. Describes nearest-GT-point distances for P/R/F1, but gives no numerical localization threshold or plugin version here. | Numeric threshold cannot be recovered from this text. It would require the exact plugin/configuration. Do not assert the repo's defaults reproduce PN1 results. |
| PN1 PDF p6, equation (3) | Bidirectional nearest-point squared Euclidean Chamfer sums for offset learning. No distance cutoff in that loss. | A training loss is not the final spatial matching tolerance. Radius-based objectness in the paper is also not a universal 2-voxel evaluation rule. |
| Original checkout and historical sibling `scripts/evaluate_paper_style_swc.py`, CLI/evaluate_row | `--distance-threshold` defaults to **6.0** for nearest sampled-point P/R/F1 and approximate PDS; `--sample-step` defaults **1.0**; cap **250000** points. Coordinates are used directly, with no spacing conversion here. P/R use distance <=threshold; PDS uses distance >threshold. | Native coordinate units, usually voxels in this data. This is point-sampled, not exact-segment or one-to-one node matching. Defaults are overridable; run records must identify overrides. |
| Same evaluator: approximate ESA | Half the sum of directed mean nearest-point distances. No threshold applied to ESA. Approximate DSA is the maximum directed mean. | Explicit local approximations, not verified official Vaa3D metrics. PDS shares the chosen threshold; ESA/DSA are continuous distances. These formulas are not a topology certificate. |
| Original/historical `scripts/evaluate_proposals.py` | `--match-distance` defaults **6.0**, `--coverage-distance` **8.0**. Counts selected proposals near SWC nodes and SWC nodes near proposals independently. | Asymmetric tolerances and many-to-one nearest support; not PN2 maximum-cardinality node F1. NMS radius 8 and score cutoff .5 are selection parameters, not evaluation radii. |
| Original `models/proposal_loss.py`, paper_skeleton_proposal_loss | Symmetric squared nearest distances, weighted means, divided by maximum cloud extent squared. No Chamfer tolerance. Objectness/radius targets use separate radius semantics. | This is a normalized training objective; its scalar is not voxel error or a node F1 threshold. |
| Historical `models/adaptive_sampling.py`, symmetric_chamfer | Sum of two mean squared nearest distances, unthresholded. Caller controls units/normalization. | Does not establish final localization tolerance or justify replacing the active sampler. |
| PN2 `evaluation/node_proposals.py`, one_to_one | Maximum-cardinality one-to-one matching at supplied radius. Proposal metrics report **2** and **6** native voxels. | Exact-node agreement is representation-dependent and distinct from point-to-polyline precision. |
| PN2 local curve-head gate | Accepted-point distance to continuous full reference geometry and arclength-to-isolated-points support at **2** voxels; **6** diagnostic. | Not one-to-one node F1 and not segment reconstruction F1. Its original thresholds remain frozen. |
| PN2 `evaluation/formulation.py`, segment_agreement; `geometry.py`, geometry_metrics | Directed arclength agreement at **2/6** by default, **.5**-voxel maximum quadrature step. Exact point-to-segment projections. geometry_metrics applies declared XYZ spacing first. | Tolerance uses the resulting units. Current native uncalibrated comparisons are voxels; physical accuracy across acquisitions is unverified. |
| Active revised proposal, section 6 | Requires both **2** and **6** native-voxel results; **6-voxel F1** is the initial primary whole-reconstruction geometric comparison, with 2-voxel errors/structure still disclosed. | Whole-reconstruction reporting and the stricter local node-stage screen have different roles. This is no license to change an already failed gate retrospectively. |

The original/historical edge-classification evaluator's .5 cutoff is an edge
probability threshold, not a localization distance. No official Vaa3D plugin
implementation/configuration was established by this local audit. The paper's
statement that its evaluation accounts for connectivity must not be transferred
to the repo's nearest-point approximation or PN2's geometry-only segment scorer.

## D. Proposed node-stage criterion - discussion only

The diagram gives Phase 3 representative positions/features and gives later
stages candidate-edge scoring/selection. A missing node near every short curve
interval is not necessarily a lost interval: a segment between anchors can cover
it. The correct geometric upper bound for an edge-only decoder is **coverage
of the union of its permitted fixed candidate edges**, under the same scoring
domain/tolerance. A decoder selecting subsets cannot exceed that union's recall.
It may do much worse because of precision, forest or topology constraints.
Cross-crop merging is currently absent, so this is only a per-box bound.

Propose the following NEW prospective screen for discussion, not adoption:

1. A necessary candidate-geometry screen: reference-arclength support by the
   fixed candidate union at two voxels >=95% neuron macro and >=90% in every
   declared box/neuron, with six-voxel results also shown. These are provisional
   engineering margins, not calibrated biological standards. Freeze a single
   candidate-building policy first; k sweeps here select nothing.
2. Preserve structural opportunities: >=90% spatial support of scorable true
   endpoint/junction locations and incident reference branches at two voxels,
   with boundary crossings and weak/thin strata separately reported. Expose
   entirely unsupported runs and ambiguity. Nearby anchors do not prove correct
   degree or parentage. The exact interior-node grid F1 is supplementary.
3. Treat spurious connected structure explicitly. Report off-reference proposals,
   their incident image/GAT edge-score distributions, which ones acquire decoded
   edges, extra decoded arclength and known wrong joins. Report isolated proposals
   separately; segment precision must not hide them. A low point precision alone
   need not imply failure if false candidates can be rejected, but strongly
   supported wrong edges are serious risks. Annotation-unknown relations stay
   unknown, never automatically negative.
4. Downstream confirmation must jointly preserve curve recall and control extra
   length/wrong joins on genuinely automatic proposals. A provisional discussion
   target could be P2/R2 >=90% plus no known false-join regression. The non-learned
   MSF is a diagnostic scaffold, not a substitute for that learned-connectivity
   qualification; its failure alone does not prove the nodes unusable by GATv2.

Items 1/2 are necessary opportunity tests, not sufficient reconstruction gates.
Item 3 needs calibrated score/risk limits from an agreed connectivity protocol;
inventing such a cutoff from these four training boxes would be unjustified.
No number above changes the original node gate or activates training. C's
candidate-union measurement is the evidence needed to discuss this distinction.

## E. Proposal-to-implementation handoff table

| Responsibility | Original PDF | Supplied current diagram | Active implementation / evidence |
|---|---|---|---|
| Input preparation | pp1-2: semi-automated skeleton extraction/manual adjustment and differentiable adaptive sampling with Chamfer | Automatic image-derived cloud; hold preparation fixed initially | Fixed image-only Sato-guided distinct source voxels, cached 2048/8192. No manual mask requirement; no learned differentiable sampler adopted. Preparation executes; universal feature retention is not established. |
| Encoder | pp3-4: PointNeXt, optional lightweight Transformer fusion, advanced augmentation/training | PointNeXt features from image points | Native Windows PointNeXt-S prototype, GN8, restored 32-channel point features. Optional encoder Transformer fusion and proposed augmentation stack are absent. Execution/tiny fitting are not official benchmark reproduction or broad encoder qualification. |
| Proposal responsibility | pp4-5: DETR-style fixed-query Transformer detection | Phase 3 Transformer: position/radius/confidence/features | Geometric-query Transformer remains intended; the approved bounded dense MLP support/offset control bypasses it experimentally. The failed control is not a product replacement. |
| NMS / consolidation | pp4-5 and p9: detection removes separate NMS/clustering | Orange, separate overlapping-crop consolidation | Cross-crop consolidation handles a distinct tiling problem, so it is not automatically contradictory to single-crop set detection. Nevertheless this dense control uses within-crop greedy NMS, an explicit departure from the PDF's claim. The authorized r2/min3 variant is post-failure. Global crop merging is unbuilt. |
| Radius | pp4-5: radius prediction and MAE supervision; p6 edge features include radius differences | Radius output in Phase 3 | Transformer radius output exists but fresh node losses do not supervise it. Dense control has no learned radius. Scaffold SWC radius=1 is a declared placeholder, never an anatomical estimate. |
| Multi-scale queries | p5 explicitly proposes coarse/fine multi-scale queries | Diagram does not spell out query scales | Current three-layer, 512-query head attends to one restored per-point memory. Hierarchical encoder radii are not implementation of explicit multi-scale query streams. Missing. |
| Supervision wording | p5: match predicted objects to GT skeleton nodes | Derive target nodes and known branch connections | Original control uses canonical nodes plus Hungarian matching. Dense control projects sources onto original reference polylines for support/offset losses. Discrete structural nodes remain evaluation labels; neither wording licenses using reference edges at inference. |
| Candidate graph | p6: spatial kNN/distance candidates | Orange candidate-graph construction | Fresh realistic learned pipeline not qualified. New fixed kNN graph is a CPU measurement scaffold only. Candidate-union arclength recall is geometric opportunity, not known true-edge recall. |
| Connectivity | pp6-7: GATv2 with geometric/node features and edge classification | Orange Phase 4 GATv2 | No fresh qualified GATv2 result on actual PN2 automatic proposals. Historical oracle-node experiments remain historical. New MSF stand-in is explicitly non-learned and does not replace GATv2. |
| Topology losses | pp6-8: cycle/multiple-parent/disconnection regularizers | Diagram names graph decoding, not the loss terms | No such fresh topology loss is implemented/trained. The revised proposal treats them as hypotheses. MSF acyclicity is a decoding constraint, not learned topology or anatomical correctness. |
| Graph decoding / gaps / roots | pp7-8: output SWC; aspirational joint optimization | Orange selection and resolution of cycles/gaps | Fresh full decoding/rooting/radius/export path unqualified. Scaffold produces per-box MSFs with image-guide roots, no added gap bridges, placeholder type/radius. It does not establish biologically correct roots or joins. |
| Whole-image output | Unified end-to-end trainable aspiration | Orange overlapping-crop merge and downstream stages | Current boxes are disjoint; whole-image merging/context and reliable cross-boundary candidates remain unbuilt. This measurement is per-box and cannot certify whole-neuron quality. |

Orange means unbuilt or unqualified in the active fresh system, not permanently
removed. Existing historical utilities or a working scaffold do not make those
stages qualified. The user's reference-SWC box says training only; in practice
references are also opened for evaluation, strictly after image-only inference.
They do not select inference nodes, edges, roots or model query counts.

## Reproducible source locations

- Original/main PN1 approximation: `../scripts/evaluate_paper_style_swc.py`,
  arguments near lines 31-34 and `evaluate_row` near lines 82-105.
- PN1 proposal diagnostics: `../scripts/evaluate_proposals.py`, defaults near
  lines 26-34 and nearest-distance counts near lines 125-140.
- PN1-style loss: `../src/pointneuron/models/proposal_loss.py`,
  `paper_skeleton_proposal_loss` / squared distances / `coordinate_scale`.
- Active evaluators: `src/pn2/evaluation/node_proposals.py`, `local_curve.py`,
  `formulation.py`, `geometry.py`; active design: `docs/revised_proposal.md`.

The companion evidence file records exact source hashes so line drift does not
silently change this audit. No official paper metric was reproduced or rescored.
