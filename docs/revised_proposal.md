# PointNeuron 2.0: revised research proposal and development path

Active document for the [fresh PN2 workspace](../README.md).
New source, scripts, configs, tests and reports belong under `PN2/`.

21 October 2026; scope corrected 22 October 2026. **Discussion draft.**
The user clarified that PN1 serves conceptual comparison, not a required
reimplementation or fresh training arm. Work starts with original raw images
and PN2 Phase 1. The former R/N milestone plan is superseded below. Historical
findings remain unchanged. See the [raw-first Phase 1 draft](pn2_phase1_design.md).
This revises the research specification; it does not authorize an unreviewed
implementation, replace the original PDF, promote the current prototypes, or
launch training. The [evidence audit](research_reset_evidence.md) explains what
is established, uncertain and newly corrected in our interpretation.

## 1. Aim and research claim

Develop fully automatic reconstruction of a single labeled neuron from a 3D
microscopy volume into a geometrically faithful skeleton/forest and valid SWC.
Use PN1's successes and failures to motivate better branch retention and
connection accuracy, while testing PN2 in its actual input context. Robust software
and commercialization remain goals to be earned through broader evidence.

The main candidate retains the intended essence:

```text
3D image + acquisition metadata
    -> automatic image-derived point clouds in overlapping spatial crops
    -> PointNeXt features
    -> Transformer skeleton-node proposals and features
    -> automatic cross-crop consolidation
    -> candidate graph
    -> GATv2 connectivity scores
    -> explicit graph decoding
    -> final skeleton/forest and SWC
```

SWC references enter training and evaluation only. They never select inference
clouds, query counts, accepted nodes, candidate edges or reconstruction roots.
Manual masking/correction is not part of the main automatic quality claim.
Single-neuron input does not imply that every bright object is target structure
or that a fragmented reference should be forced into one connected tree.

**Central hypothesis:** informative automatic point representations plus
learned global proposal context and learned local connectivity can preserve
true branches and avoid wrong connections within a declared computation budget.
PN1 motivates this hypothesis; differing contexts prevent a direct performance
ranking without a separate matched protocol. Each claimed
advantage must appear in final whole-neuron outputs, not only component losses.

## 2. What changes from the original proposal

| Original assumption or aspiration | Revised specification | Reason |
|---|---|---|
| Classical skeleton from a segmented image supplies reliable structure, optionally checked manually | Phase 1 is fully automatic and image-derived; an image skeleton or learned field is a fallible guide whose retained support must be measured | Segmentation/skeleton quality was not established; the product preference is automatic |
| PointNeXt is an improvement over DGCNN | PointNeXt is the retained candidate, compared with the working DGCNN reference and audited against its declared reference implementation | Benchmark performance elsewhere does not establish neuron reconstruction gains |
| DETR-style nodes naturally remove redundant predictions and NMS | Stable graph discretization, explicit used-slot semantics, and measured duplicate/cardinality behavior are required | Interior skeleton samples are representational choices; inference still needs cross-crop consolidation |
| Multi-scale queries and topology-aware training are implied parts of the system | Record which are implemented; add only a specified mechanism supported by a named failure and a controlled comparison | Current prototype does not implement every proposed feature; adding terms is not evidence |
| A cycle/multiple-parent/disconnection penalty ensures biological topology | Separate geometric evidence, learned edge scores, forest validity, branch incidence and uncertain fragmentation | A valid tree can contain wrong shortcuts; forcing connectivity can invent branches |
| All stages are end-to-end differentiable | Jointly optimize differentiable components; explicitly document discrete sampling, matching, candidate indexing and decoding | The actual system contains discrete operations; joint gradients do not optimize every final decision |
| Better modules should produce a generally superior product | Require paired whole-neuron development evidence, replication and new external-domain confirmation | Present experiments establish component findings and tiny-set fitting, not the product claim |

These are prospective revisions for discussion. Historical gates and numerical
results are preserved. No failed experiment is relabeled a success.

## 3. Input, data and supervision contract

### Acquisition and coordinates

Accept a 3D scalar microscopy image with declared XYZ axis order, intensity
encoding and voxel spacing when available. Decode supported formats against
independent references. Preserve the native image and explicit coordinate
transforms through all crops and SWC export.

Use physical units for new calibrated acquisitions. Where calibration is
unavailable, report voxel units and evaluate within that acquisition; do not
invent micrometers or treat equal voxel tolerances as equal physical accuracy
across sources. Anisotropic geometry/radius conventions must be resolved
explicitly before a physical-radius claim. Intensity normalization uses image
evidence, is identical in training and inference, and has no neuron-ID rules.

### Partitions and annotation limitations

Provisionally retain the current 33 training / seven development / eight
PN2-reserved internal neuron identities for continuity. Group correlated volumes, duplicate traces
and related acquisitions together where such relationships are known. The
seven repeatedly inspected neurons are development data. All eight reserved
neurons appeared in historical PN1 training; they are not an untouched common
PN1/PN2 test. Preserve them now and plan new external confirmation separately.
If a related-acquisition/duplicate audit finds cross-partition dependence,
revise the partition prospectively and document changed identities/counts.

References define observed curve geometry, radii and known connections, not
automatically exhaustive background. Reference-empty crops remain unknown.
Separate verified completeness, assumed reference-agreement supervision and
unlabeled regions in the dataset specification. Do not silently turn unknown
image structure into negative examples to fix a residual.

Broader training samples multiple neurons and morphologies. Report image-level
contributions so many crops from a large arbor do not masquerade as independent
neurons. Image-defined training crops can be sampled; development inference
must cover the declared full volume rather than only the existing capped crop
cache. No fixed four-crop task can settle generalization.

## 4. Responsibilities of the retained stages

| Stage | Required output | Training/evidence obligation |
|---|---|---|
| Automatic point preparation | Image-derived XYZ/intensity with broad thin-branch support and recorded actual counts | Compare candidate support before/after sampling; guides must not depend on reference access at inference |
| PointNeXt | Useful per-point descriptors with consistent coordinates and normalization | Audit declared wiring, operators and train/eval behavior; establish useful downstream features in PN2; isolate encoder benefit only in a named ablation if warranted |
| Transformer | A compact, reproducibly defined graph discretization: centers, radii, used-slot scores and features | Match canonical training targets; evaluate continuous geometry and structural coverage separately from exact target assignment |
| Consolidation | One spatially coherent proposal set across overlapping crops | Deterministic image-only policy; test duplication, junctions and crop boundaries before the whole-neuron comparison |
| Candidate graph | Possible connections containing the relevant true branches without an undisclosed cap | Measure candidate ceilings and scale/memory; distinguish missing nodes from missing candidate pairs |
| GATv2 | Scores for candidate connections in the proposal context actually encountered at inference | Train/evaluate on realistic imperfect proposals; compare against the working initialized graph and, if needed, a matched simple edge scorer |
| Graph decoding/export | Automatic graph decisions, explicit fragments/failures and valid SWC | Preserve unsupported gaps instead of forcing one tree; evaluate wrong connections and missed branches independently of serialization validity |

### Geometry and proposal selection

Treat skeleton nodes as an engineering discretization of curves, not as a claim
that every interior node is a distinct anatomical object. Keep endpoints,
junctions and genuine crop-boundary intersections. Use one deterministic,
subdivision-stable interior sampling convention. The current canonical targets
are a useful starting candidate; their eight-voxel spacing is not a physical
resolution specification for future acquisitions.

The score means "this query is a used representative under the target
representation and supervision policy." It is not automatically a calibrated
probability of biological foreground. Evaluate score ordering, operating
behavior and cardinality across neurons. A global operating policy can be fit
on development data and frozen; a threshold chosen to remove the last error
on four training crops is not independent evidence.

One-to-one matching is retained initially. It is not inherently wrong because
the target is sampled. Alternative curve/branch decoders become separate
research proposals only if a bounded, coherent set-prediction comparison shows
that discretization is the limiting factor. Do not adopt a new representation
merely to make a local gate easier.

### Connectivity and imperfect proposals

The intended GAT task is connectivity for automatic proposals, including
missing, duplicate, displaced and extra nodes. Oracle-selected nodes are a
diagnostic upper condition, not the training distribution or production result.

Use reference identities and annotation policy to distinguish known adjacency,
known nonadjacency and unknown relationships. Reconcile node and edge labels
under the same policy: an edge touching a no-object query can be trained as
negative only when that no-object interpretation is justified by the declared
supervision contract. Unknown endpoints must not silently become negatives.
If those labels cannot be supported, restrict the claimed supervised task or
obtain suitable data before claiming robust automatic rejection.

Keep the current GAT edge-scoring role in the initial revision. A graph-based
node-rejection head, direct image evidence along edges, a new structured
objective or branch-token decoder would each be a substantive extension to
explain and compare before implementation. Their benefit is not established by
the last spur. Topology-aware losses remain optional hypotheses; cycle-free
decoding already enforces some structural constraints without proving anatomy.

Whole-neuron GAT inputs must use a consistent coordinate/feature frame after
merging crops. A crop-local descriptor cannot silently carry its old crop
origin as if it were a global graph coordinate. Specify feature association,
deterministic merged-score association, deduplication and cross-boundary edges,
then use the same packet/candidate
builder in training and inference. Training only clean isolated crop graphs
does not qualify an unseen global graph distribution.

### Scale and resource behavior

Input-cloud points, query slots, merged proposals and graph nodes are distinct
budgets. The current 2,048-point, 512-query and native N>=256 settings are
prototype choices. Record capacity failures and actual counts; do not truncate
targets, duplicate input points or drop difficult crops to conceal them.
Short clouds, empty clouds, target overflow, overlapping tiles and whole-volume
graph memory require explicit automatic behavior before broad use. Resolve
the prototype's supported-input policy before a broader run; preserve all
declared cases in its results, including unsupported ones.

## 5. PN1 as conceptual reference; PN2 as the product under development

PN1 supplies useful lessons: preserve image-supported geometry, retain weak
branches, inspect final SWCs and avoid assuming learned connectivity must beat
a geometric method. Its source, checkpoints and reports remain historical
evidence with their actual preprocessing and training exposure disclosed.

Fresh PN1-style retraining, DGCNN integration and the proposed singleton fix
are outside the active PN2 plan. The reference port undertaken on 21-22 October
was a scope detour, not a required product stage. Its results are preserved,
not promoted or reclassified as PN2 results.

The active product is original image -> automatic point preparation ->
PointNeXt -> Transformer proposals -> consolidation -> GATv2 -> graph/SWC.
No historical checkpoint, fixed intensity fraction or PN1 sampling rule becomes
a PN2 default because it was implemented earlier.

Assess PN2's own complete outputs, failures and resource behavior under its
declared context. A future claim that it improves PN1 needs a separately agreed
comparison with compatible inputs, coordinates, exposure and metrics. Existing
cross-context headline scores cannot establish that claim, and such a comparison
does not require rebuilding PN1 now.

For a named PN2 component comparison, hold relevant input pools, downstream
controls and resources fixed. Record data order, normalization, initialization,
augmentation, optimizer history and selection policy. No temporary bypass of
an intended product stage is adopted without discussing its architecture impact.

## 6. Common whole-neuron scorecard

The shared output interface is the actual reconstructed SWC/forest. Freeze the
evaluator before generating the decisive comparison. It must consume explicit
approved identities rather than scanning all dataset annotations.

- **Geometry:** arclength-weighted precision, recall and F1 against unchanged
  continuous reference segments, absolute extra/missed length and distance
  distributions. On the present voxel-only development data, report both two-
  and six-voxel tolerances, with six-voxel F1 as the initial primary geometric
  comparison metric for historical context. It does not establish continuity
  or parentage. Recompute both arms; do not compare the new weighted
  values directly with the old point-sampling headline.
- **Structure:** reference endpoint/junction support, branch continuity,
  disconnected portions, wrong merges and long shortcuts. Implement and test
  correspondence/ambiguity handling before claiming a topology score. A graph
  induced on each model's surviving nodes cannot be the sole topology metric.
- **Output quality:** SWC validity, empty/invalid/timeout/unsupported outputs,
  root/type conventions and radius errors separately. Valid serialization and
  a single root are not anatomical accuracy.
- **Resources and robustness:** time and peak memory for the whole job,
  actual crop/point/proposal/graph counts, source/acquisition strata, full
  per-neuron paired results, macro mean, median and worst cases.

Every declared reference-positive case stays in the comparison denominator.
Empty/invalid/failed reconstructions receive zero reconstruction F1 and a
separate failure status; never silently skip them. Unscorable or incomplete
references receive explicit annotation status, not invented background scores.
Converge geometric quadrature numerically and test invariance to SWC vertex
subdivision. Local approximate ESA/DSA/PDS must retain that label.

The advancement decision must require a useful paired geometric gain at an
agreed precision tradeoff, with no extra execution failures or structural
regression concealed by the mean. Define observable structural errors and
freeze quantitative improvement/regression margins in milestone A after
reviewing the evaluator, PN2's input context and the user's quality priorities,
before decisive results. A paired claim requires a declared comparison; it
does not mandate a freshly trained PN1 arm. Earlier query plans suggested +0.02 mean F1, -0.01 mean
precision and -0.05 worst-case F1 margins; those numbers are not empirically
or biologically justified by this reset and are not silently adopted here.
Report two-voxel errors and branch findings regardless of the aggregate result.
Any later change in frozen margins creates a new prospective experiment rather
than turning an old failure into a pass. Seven development neurons cannot by
themselves support statistical or commercial reliability claims.

## 7. A bounded path forward

| Milestone | Concrete deliverable | Decision before proceeding |
|---|---|---|
| A. Start at original raw images | Verified decoding/coordinates, raw-image inspection and reviewed PN2 Phase 1 design | Agree what the point representation must preserve; no inherited PN1 mask or model |
| B. Qualify automatic point preparation | Image-only candidates, fallible guide and finite-budget clouds with stage-by-stage retention/resource accounting | Locate losses before the encoder; keep SWCs outside inference decisions |
| C. Integrate the intended PN2 stages | PointNeXt/Transformer/GATv2, consolidation and final SWC path with explicit labels, normalization and capacity behavior | Resolve proven interface defects before an expensive fit; no PN1 rebuild prerequisite |
| D. Test one complete candidate | One agreed bounded training recipe and whole-neuron development outputs under frozen policies | Assess geometry, structure, failures and resources; use one named ablation only if the result warrants it |
| E. Confirm a frozen candidate | A second training seed, then PN2-reserved internal cases and new compatible external/source-held-out data | Distinguish development improvement, independent generalization and product readiness |

Milestone C may reuse current code as an explicitly reviewed starting asset. The native/reference
operator and decoder evidence should be reused, not reimplemented from zero.
Audit meaningful invariants: format decoding, translation/unit transforms,
point-order behavior under declared tie rules, train/eval normalization,
canonical targets, realistic edge labels, crop merging, capacity and checkpoint
reload. Separate deliberately different prototype behavior from a proven bug.
In particular, the query head's positional/query injection differs from
official DETR/3DETR. Document its identity; do not call it a failed reproduction
or silently replace its mechanism. A reference-aligned control is justified
only if that mechanism is the hypothesis being tested.

Milestone D must have a runnable whole-neuron inference path before an expensive
fit. Use the existing four-crop case only as a short execution/gradient sanity
check; the present near-exact fit already supplies substantial local capacity
evidence. No further 4,000-update confidence-only or canonical-target extension
is the next milestone.

At D, prepare one broader candidate, not a combinatorial grid. If diagnosis
needs an ablation, use a shared downstream control to locate whether input/
proposal retention or connectivity loses the benefit. Then choose one relevant
comparison within the approved PN2 design. A different backbone or a simple
edge scorer would be a separately discussed ablation, not another automatic
product path. Do not train every permutation.

### Budget and failure discipline

Before each substantial run, record training identities, supervision/augmentation,
initialization and optimizer state, LR schedule, update/epoch budget, inference
policy, outputs and success/failure interpretation. Measure timing on the RTX
3070 Ti Laptop GPU with evaluation/I/O included as appropriate. The user gets
a visible PowerShell command and launches the agreed run. This document does
not invent a reliable duration or finalize a fresh-training LR from the tiny
warm-start fits.

A completed negative experiment ends that hypothesis under its recipe. Permit
one bounded corrective comparison for an independently demonstrated bug or
named causal explanation. If that comparison fails, stop that hypothesis and
revise the design at a decision point. Infrastructure failure may resume
the same run with complete optimizer/RNG state and unchanged cap. Neither kind
of failure automatically authorizes a threshold, loss, architecture or cap change.

No research plan can eliminate failed hypotheses. This plan makes each failure
answer a project-level question and prevents every local error from starting
another open-ended sequence.

## 8. Implementation impact and immediate next action

**Proposed for retention:** verified decoding and coordinate handling; PN1's
conceptual lessons and historical evidence; native PointNeXt/query/GAT implementations and operator
audits; useful field/sampling experiments; canonical geometry utilities;
checkpoint/provenance and final-SWC audit tools.

**Proposed for revision before further training:** benchmark independence,
annotation/label semantics, train/inference normalization, whole-volume
consolidation, realistic graph supervision and the common final-output
evaluator. These are substantive work items to specify together, not changes
silently installed by this document.

**Not adopted:** manual masks as a main pipeline requirement, reference-count
top-k, per-neuron cutoffs, new topology penalties, another representation,
unbounded training extensions, or a new field/tracer replacement pipeline.

The immediate work is **original raw-image inspection and PN2 Phase 1 design**.
The common evaluator and image-only decoder/tiler are reusable infrastructure.
The PN1-only port is archived; neither its cutoff nor its singleton behavior
defines PN2's next task. No new sampling method or image field is promoted by
this scope correction, and no training starts. See the
[Phase 1 design draft](pn2_phase1_design.md) before implementation.

## 9. Research grounding

The original PointNeuron paper supplies the point-cloud/skeleton/connectivity
foundation; the local PN1 reimplementation and its selected geodesic back end
are not identical to the published complete system. The original PN2 PDF is
a hypothesis document, not evidence that the proposed replacements improve it.

Primary references, supporting mechanisms rather than guaranteeing transfer:

- [PointNeXt](https://arxiv.org/abs/2206.04670): training strategies and scalable
  point features. Our S prototype's declared topology, short-cloud behavior,
  normalization and Windows native operators require their own verification.
- [DETR](https://arxiv.org/abs/2005.12872) and
  [authors' implementation](https://github.com/facebookresearch/detr): fixed
  queries and bipartite set supervision. A skeleton target convention and
  corresponding score semantics still have to be specified for this task.
- [3DETR](https://arxiv.org/abs/2109.08141) and
  [authors' implementation](https://github.com/facebookresearch/3detr): a
  relevant point-cloud query detector with geometric query locations and
  positional embeddings. Its object-box results do not validate neuron nodes;
  its decoder choices provide a possible fidelity control, not an automatic
  architecture change.
- [GATv2](https://arxiv.org/abs/2105.14491): dynamic graph attention. Attention
  weights are not themselves undirected reconstruction-edge probabilities;
  the edge head, supervision and final decoder must be evaluated separately.

See the [evidence audit](research_reset_evidence.md) for local source/report
links and the provenance of every claimed project result. The first revision
keeps the intended architecture while requiring measured reconstruction quality
in PN2's declared input context. Cross-system superiority remains a separate
claim requiring a valid comparison.
