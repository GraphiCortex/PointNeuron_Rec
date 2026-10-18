# Node selection and connectivity: a design recommendation before another fit

19 October 2026. Research/design review after the user questioned whether we were
advancing through stages while automatic proposal selection remained unsolved.
**No model code, loss, configuration or training run is changed by this review.**
Recommendation: retain PointNeXt, Transformer proposals and edge-scoring GATv2
initially. Specify and test a minimal **joint proposal/connectivity prototype on
real proposal inputs**, with proposal confidence still owning final node selection,
before adding a second node classifier or replacing an architecture stage.
This is a recommendation for discussion, not approval or implementation of a fit.

## What the proposal says, and what has actually been tested

The source [research proposal](C:/Users/Seywe/Desktop/Research/R-Arij/Proposal.pdf)
has ten pages, SHA256
`60e12b8907b23bb1903d5a3668c76d430173ecad5baa373cc3c266f69958b5b8`.
It was reread in full; pages 4-8 were also rendered and visually inspected.
Its statements of improved quality and efficiency are hypotheses, not results.

| Responsibility | Proposal | Current evaluated prototype |
|---|---|---|
| Proposal generation | Fixed Transformer queries predict position, objectness and radius; Hungarian set matching (pp. 4-5, sections 3.2-3.3) | Implemented native prototype with one-to-one assignment and classification/center/radius losses; automatic selection still fails |
| Connectivity | GATv2 predicts edges from detected nodes and geometric/features; edge loss plus topology regularizers (pp. 6-7, section 4) | Existing GATv2 learns edges on frozen reference-selected proposals using balanced edge BCE; no learned node deletion/creation/motion head |
| Training integration | Joint backpropagation with combined sampling, detection and connectivity objectives (pp. 7-8, sections 5.1-5.4) | Saved proposal fits and GAT fits are separate; connectivity loss has not jointly trained the current PointNeXt/Transformer |
| Node rejection | Proposal objectness is intended to distinguish unique skeleton proposals; no graph node-classification head is specified | Hard .5 filtering determines graph inputs in the automatic comparison; the decoder retains all accepted nodes, including isolates |
| Topology | Penalize cycles, multiple parents and disconnection (pp. 6-8) | Evaluation checks topology; a reference-free forest decoder forbids cycles. Decoder constraints are not a topology training loss |

The proposal does not specify a .5 cutoff, when filtering happens relative to graph
reasoning, differentiable handling of membership changes, or a complete treatment
of uncertain/unmatched endpoints. Those details require design decisions; the
word "end-to-end" does not resolve them. Adding a graph node head would extend
section 4's stated edge-scoring responsibility, not merely finish an explicit head
already specified in the document.

Phase 1 in the document permits classical/manual verification. The user's current
fully automatic objective overrides that preference. This review does not restore
manual masks or modify the frozen image-derived point-cloud preparation. Native
PointNeXt/GATv2 remain documented prototypes, not benchmark reproductions.

## What the saved evidence does and does not establish

- [Balanced proposal fit](balanced_query_results.md): useful geometry on four
  training crops, with residual extras and rejected assigned nodes. The 14 dense
  placement misses are 2.011-2.596 voxels, not evidence of 14 absent branches.
- [Linear capacity witness](four_crop_confidence_capacity_results.md): one shared
  affine classifier can separate all four fixed feature/assignment sets. A larger
  head is not required by this particular representability test.
- [Frozen-confidence fit](four_crop_frozen_confidence_results.md): the declared
  finite AdamW fit fails even with fixed labels/features. Assignment movement is
  therefore not necessary for this failure. Missing joint training is not a
  diagnosed explanation for every confidence optimization failure.
- [Conditional GAT fit](frozen_proposal_gat_results.md): exact edges, degrees and
  components on reference-selected learned proposals from these four training
  crops. This proves conditional learnability, not automatic node quality,
  generalization or superiority to a pairwise edge head.
- [Automatic integration comparison](gat_automatic_selection_results.md): same
  checkpoint, existing .5 selection, no retraining. Small-crop precision at two
  voxels falls to 57.11% / 45.12% / 67.21%; seven rejected dense nodes remove 13
  required pairs. Every surviving required pair remains a candidate. Some missing
  direct edges are validly replaced by paths through unmatched nodes; direct
  adjacency identity alone must not be equated with anatomical topology.

There is a real mismatch between the node sets used in the successful graph fit
and those produced automatically. Neither the benefit of joint optimization nor
the benefit of graph node refinement has been measured. No matched end-to-end
PN 1.0 improvement has been established. The seven repeatedly inspected neurons
remain development data, not a fresh test set.

## Three choices, with different responsibilities

| Choice | What changes | What it could address | Main limitation | Recommendation now |
|---|---|---|---|---|
| Continue isolated confidence optimization | Optimizer/parameterization or proposal loss; Transformer remains sole selector | Known failure to learn separable fixed labels | More isolated fits do not test the graph interface; zero-label-error need not imply good geometry | Preserve as a possible optimization investigation, not the default next series |
| Joint proposal and edge learning, same heads | Add a real differentiable training connection between existing models; use realistic proposal context and defined final node selection | Missing co-adaptation, graph-context distribution mismatch, node/edge objective interaction | Joint loss may still fail confidence learning; masked extra-edge labels do not teach node deletion | First integrated hypothesis to specify before changing roles |
| Graph-refined node and edge decisions | Retain a proposal pool, add graph node-confidence output and explicit final selection | Context-dependent node rejection and node/edge consistency | New responsibility, extra supervision/optimization choices; cannot recover image structure never represented; may repeat confidence failures | Explicit architectural alternative if agreed, not a silent repair |

No choice is a guaranteed fix. In particular, a graph node head is not justified
simply because the current final linear confidence head fit failed: the capacity
witness already demonstrates that the fixed labels are representable without it.

## Recommended minimal joint contract

1. **Transformer owns unique proposal confidence.** Its label remains membership
   in the agreed target-node assignment, not a claim that every unmatched query
   is image background. Center/radius supervision remains explicit. Do not quietly
   turn all curve-near unmatched queries into positives without redesigning edge
   correspondence and the node representation.
2. **Use actual model proposals in graph reasoning.** An initial implementation
   option is to carry the existing 512 per-crop queries, with soft confidence
   features, through graph scoring and apply final node decisions afterward.
   This delays the irreversible graph-input cutoff and makes training/inference
   context consistent. It is an explicit interface-policy change to evaluate,
   not wording already specified by the proposal, and it does not itself improve
   confidence or authorize extra nodes in the final SWC.
3. **Train existing weights together.** Proposal and edge losses share a backward
   pass into encoder/query features, geometry and confidence where those paths
   exist. Existing GAT tests show gradients to proposal inputs; no real joint fit
   has used them. GATv2 still predicts edges, not node validity or coordinates.
4. **Keep correspondence, masks and uncertainty explicit.** Matching and neighbour
   indices are discrete; ordinary backpropagation can optimize continuous values
   conditional on them, but does not differentiate the hard membership decision.
   Edges touching unmatched/untrusted endpoints currently have unknown labels.
   Retaining those nodes does not magically provide negative labels or a cleanup
   objective. An integrated protocol must specify which labels/weights are used
   and which gradients are available; no mass relabeling to background is allowed.
5. **Respect forests and incomplete annotations.** Do not force one connected
   component per crop. The dense reference legitimately has 14 visible components;
   absence of local SWC supervision is not verified background. Topology penalties
   require a valid mathematical definition and masks before implementation. A
   generic disconnection penalty copied from the proposal would be inappropriate.
6. **Judge the actual automatic output.** Node acceptance/rejection, reference
   arclength P/R, critical-branch incidence, component splitting/merging, paths
   and SWC validity must be evaluated together. Assignment-label perfection is
   a learnability diagnostic, not the final morphology definition. Oracle plots
   remain controls and never satisfy the automatic gate.

This is a minimal proposal-aligned integration, not a claim to reproduce every
proposal feature. Phase 1 stays frozen initially; topology regularizers and
multiscale details cannot be declared implemented merely because the models are
connected. Adding every suggested term at once would obscure interpretation.

The 512-query budget is separate from the 2,048 input-point budget and any future
whole-neuron consolidation budget. Retaining all queries also requires explicit
memory checks: the current radius-union candidate rule can produce up to 130,816
undirected pairs for 512 nodes. Do not shrink budgets or discard candidates
silently to fit the laptop GPU. A visible timing/memory check would precede any
joint GPU fit, after the experiment is specified and agreed.

## Milestone and stopping decision

The next milestone is one **automatic, jointly trained per-crop reconstruction
prototype on the fixed training cases**, with a declared node/edge contract and
reconstruction-level criteria. It is not another oracle-selected component pass.
The minimal joint variant should be compared with the preserved staged baseline
on identical data, coordinates, output conventions and metrics. Warm-start versus
fresh initialization, optimizer budget, masks, topology terms and thresholds
must be stated before implementation; none is silently inherited as a matched
causal comparison.

If that bounded integrated hypothesis fails its declared automatic criteria,
stop and reconsider the interface/formulation; do not proceed to broad data or
commercial claims on the strength of component fits. Graph node refinement is
then a concrete alternative to compare, rather than an assumed cure. If it
succeeds, broad training/development evaluation and a fresh matched PN 1.0 control
are still required before claiming reconstruction improvement.

## Code anchors and scope

- [Current assignment/classification/geometry loss](../../src/pointneuron/models/query_loss.py#L51)
  and [balanced reduction](../../src/pointneuron/models/balanced_query_loss.py#L11).
- [Proposal update](../../scripts/pointneuron2/branched_query_experiment.py#L211)
  and [isolated GAT update](../../scripts/pointneuron2/frozen_proposal_gat.py#L219).
- [Graph input features](../../src/pointneuron/graph/proposal_graph.py#L51),
  [candidate indexing](../../src/pointneuron/graph/proposal_graph.py#L129), and
  [gradient-path test](../../tests/test_gatv2_connectivity.py#L79).
- [Unknown endpoint labels](../../src/pointneuron/data/connectivity_targets.py#L86)
  and [edge objective](../../src/pointneuron/data/connectivity_targets.py#L122).
- [Hard final node gate / decoder](../../src/pointneuron/graph/connectivity_decoder.py#L46).

Only this readable decision note, its provenance metadata and index links are
written. No runtime source, model, cache, configuration, checkpoint or inference
policy is modified; no training or GPU run is launched.
See [review provenance and unchanged-source verification](node_graph_design_decision_seed0/summary.json).
