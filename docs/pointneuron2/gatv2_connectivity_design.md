# GATv2 connectivity: proposed first implementation contract

17 October update: the user agreed to proceed. The
[first per-crop implementation and CPU checks](gatv2_connectivity_implementation.md)
are complete with zero optimizer updates. This document retains the reviewed
design; no training or whole-neuron integration is established.

16 October 2026. Design following the completed training-only representation
oracle. This document specifies the next component; it does not implement GATv2,
alter existing checkpoints/model policies, or authorize a training run.

## Aim and implementation boundary

Retain image-derived points -> PointNeXt -> Transformer -> learned connectivity
-> SWC. The first connectivity implementation is per crop. It refines a
geometric candidate graph, without requiring a classical tracer installation.
Its inputs, labels, model and decoding must be separate interfaces so a later
automatic image/tracer initializer can add candidate edges/features explicitly.

This follows the proposal's proximity-based candidate graph. The published
PointNeuron's tracer-initialized GAE results do not establish that a geometry-only
initializer suffices. Image evidence along an edge is not supplied independently
in this first module; learned node features may not retain enough such evidence.

The [reference oracle](representation_oracle_results.md) establishes a narrow
ceiling for correct nodes and known connections. It does not validate noisy
proposals, automatic crop correspondence or an inference-time forest decoder.

## 1. Transformer-to-graph input

Reuse the current `QueryProposalOutput`, without changing the query head:

| Field | Per-crop shape | Meaning |
| --- | --- | --- |
| `center_proposals` | Q x 3 | Native-voxel coordinates |
| `radius` | Q x 1 | Positive predicted radius |
| `objectness_logits` | Q x 2 | Current used/no-object scores |
| `local_centers` | Q x 3 | Center relative to crop center, divided by crop radius |
| `decoder_features` | Q x 128 | Transformer features associated with each query |

Q is currently 512. Input-cloud size, query capacity and graph-node count remain
distinct. All Q proposals enter the first graph scorer, including low-confidence
ones. This avoids excluding potential neighbours before scoring, but does not
change the final node acceptance policy below. No top-k/global 128-node cap.

Node input is 133-dimensional: decoder features (128), local centers (3),
radius/8 (1), and the used-minus-no-object logit difference (1). Explicit query
IDs, GT assignments/roles/identities and reference components are excluded from
learned features. IDs remain metadata for bookkeeping and permutation checks.

An adapter must record upstream checkpoint/schema, normalization policy,
cloud/feature hashes, crop center/radius and source partition. It must not
silently convert running-statistic PointNeXt inference to current-crop inference.
The documented batch-one normalization control remains a research policy.

## 2. Candidate graph, independent of labels

Start with the oracle's fixed diagnostic rule: undirected symmetrized 16-nearest
neighbours union pairs within eight native voxels. Create candidates from actual
predicted positions, never target positions. The scorer receives both directions
of each edge for message passing; supervise one undirected edge once.

Every node has a self-message for GATv2. Self-loops are not reconstruction edges
and never receive positive/negative connectivity labels. Preserve coincident
proposals as separate nodes at this stage. Use stable tie handling and record
degree/candidate counts; fail explicitly on invalid/nonfinite inputs.

The oracle's candidate recall was 100% even for kNN alone. That is not a guarantee
for 512 imperfect proposals, where clutter and moved/missing nodes can hide a
true neighbour. Do not insert GT edges to make candidate recall pass. Missing
known positive edges count in the attainable-recall ceiling even though no
candidate logit exists for their BCE loss.

The eight-voxel radius is a starting comparison setting, not a verified gap-
bridging distance or a physically calibrated quantity. A future image-supported
initializer is an explicit extension of this interface.

## 3. Training-only edge supervision

Construct target adjacency by continuous reference-chain ordering using the
tested oracle machinery. Retain full-reference component identity separately
from crop-induced components. Transfer adjacency to predictions through the
existing Hungarian assignments; never pass those assignments into the scorer.

Hungarian matching always assigns targets when Q is sufficient, even when
geometry is bad. Treat a match as trusted for edge labels only if its placement
error is <=2 native voxels. This is a proposed label-reliability setting, not a
new proposal-loss or acceptance threshold. Count and report every rejected
match/target and the resulting edge-label coverage.

For a candidate pair:

| Condition | Edge label |
| --- | --- |
| Both endpoints trusted, matched targets consecutive on a visible reference chain | Positive |
| Both endpoints trusted, same full-reference tree, but targets not directly adjacent in the target graph | Negative for this representation |
| Targets belong to different full-reference fragments | Unknown |
| An endpoint unmatched, poorly placed or unsupervised | Unknown |

The same-tree negatives still rely on the annotated topology being correct.
Different crop components from one full reference tree need not be unrelated:
their connecting path may lie outside the crop. Target adjacency encodes local
visible connections, not arbitrary shortcuts through those outside paths.

Unknown edges remain in message passing and inference candidates, but are masked
out of supervised edge loss. An absent local SWC supplies no background labels.
Do not expand targets or match labels differently to create extra supervision.

This initial loss cannot teach rejection of all unmatched-proposal clutter.
Proposal confidence retains that responsibility; unknown candidates also remain
an inference risk. Log predictions on unknown edges separately. A node-validity
head or additional negative supervision would be a separate design change.

Use masked class-balanced BCE: average positive and known-negative terms
separately, giving each class half the weight when both exist. For a single
observed class, use its mean. Average supervised crops equally. An entirely
unknown batch gets no optimizer step, momentum update or weight decay.
Do not infer probability calibration from this class-balanced objective.

No training duration, optimizer recipe or joint-loss weight is selected here.
Sparse trusted labels are a reason to inspect upstream geometry, not to launch
an edge fit with misleadingly low loss. Current two-crop placement evidence
contains no junctions and is insufficient for a branched connectivity claim.

## 4. GATv2 and explicit edge classifier

Proposed small native-PyTorch prototype: project 133 inputs to width 128, then
two GATv2 blocks with four heads of width 32, residual connections, LayerNorm,
ReLU and zero dropout for the first deterministic checks. These are unmeasured
starting settings, not official benchmark reproduction or tuned architecture.
No environment/package installation is part of this design step.

For directed messages, provide relative XYZ/8, distance/8 and radius difference/8
as five geometric edge attributes. Self-message attributes are zero. GATv2's
query-conditioned attention updates representations using the candidate graph.

Attention coefficients are normalized over a node's neighbours; they are not
independent edge-existence probabilities. Add a separate undirected edge head.
Its 264 symmetric inputs are feature sum and absolute feature difference
(256), absolute relative XYZ/8 (3), distance/8 (1), radius sum/8 and absolute
radius difference/8 (2), and used-probability sum and absolute difference (2).
A 264 -> 128 -> 1 MLP produces one logit per undirected candidate edge.
Endpoint reversal must leave the logit unchanged. The no-message-passing control
uses the same 133 -> 128 node projection and this same edge head.

The GATv2 mechanism and edge-attribute formulation follow the
[official GATv2Conv documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html),
checked 16 October 2026. Validate any native implementation against the declared
equations on synthetic graphs; do not claim compiled-kernel or speed parity.

Required control: the same edge head with the same input features and geometry,
but no GATv2 message passing. Match upstream outputs, candidates, label mask,
loss, decoder and training budget. This tests whether graph context earns its
complexity; fitting GATv2 alone would not establish that contribution.

## 5. Explicit baseline decoding

The first diagnostic decoder should be deterministic:

1. Accept nodes at the existing proposal confidence cutoff 0.5. Keep rejected
   nodes/incident edges in diagnostics so their loss is visible.
2. Keep candidate edges whose endpoints are accepted and whose learned edge
   score is >=0.5. This starting threshold is not calibrated across objectives.
3. Sort by descending edge score and build a maximum-weight spanning forest
   with cycle rejection. Do not complete it to a connected tree or invent
   fallback bridges. Use stable tie handling.
4. Preserve accepted isolated nodes as isolated roots. An empty accepted set is
   an explicit reconstruction failure, not a zero-error SWC.
5. Orient each tree from its highest-confidence node, with deterministic ties,
   solely for parent-before-child SWC serialization. This root convention is
   not a predicted soma or verified biological parent direction.

An oriented forest has at most one parent per node and no cycles. Those are
decoder properties, not evidence that the selected topology is correct. Wrong
early high-scoring edges can still displace correct ones; compare edge quality
before and after decoding. Retain exact oracle-score tests for the decoder.

This preserves the current confidence cutoff and therefore preserves its known
failures too. GATv2 is not presented as a confidence repair. Joint node/edge
selection or accepting low-confidence nodes from connectivity evidence would
be substantive alternatives to discuss and compare separately.

## 6. Whole-neuron integration remains explicit work

The first module scores/decodes one crop. Automatic matching of duplicated
predictions across crops is not solved by the oracle's reference identities.
Do not splice independently decoded crop trees and call the result a validated
whole-neuron reconstruction.

Before full-volume evaluation, define reference-free proposal consolidation,
feature association, cross-crop candidates and global decoding. Preserve crop
and query provenance and measure losses before/after each stage. That stage
must use image-derived coverage at inference; the eight-crop training cache
does not certify whole-arbor coverage. No consolidation distance rule is
silently adopted in this component design.

The historical geodesic initializer/GAE remain archived comparison paths. The
oracle forest exporter remains reference-only. Neither is relabeled GATv2.

## Next implementation and advancement checks

Implement separate graph-input/candidate, edge-supervision, GATv2/edge-head and
forest-decoder APIs with their own versioned checkpoint identity. Preserve the
existing model/checkpoint schemas. First verify synthetic CPU behaviour only:

- Finite forward/backward, deterministic reload and explicit node counts.
- Permutation consistency, translation invariance and undirected edge symmetry.
- Candidate creation unaffected by changing labels; no GT feature inputs.
- Unknown/fragment/unmatched/poor-placement label masking and empty cases.
- Same-score cycles, isolated nodes, disconnected forests and deterministic SWC.
- Oracle-score decoding retains all known reference edges on the fixed cases.

Successful implementation checks do not authorize or substitute for fitting.
Any bounded real fit must state the data, upstream policy, feature/placement
coverage, settings, GPU duration estimate and output paths before launch. Save
optimizer/scheduler/RNG states for genuine resumability. The user launches
agreed substantial GPU commands visibly.

Advance to a reconstruction claim only after reference-free whole-neuron
integration and matched PN1 comparison. Use a fresh PN1 baseline trained on the
same neuron-disjoint partition; keep the repeatedly examined seven-neuron set
as development data. Preserving the PN2-reserved eight requires excluding them
from both new models' fitting and tuning; the historical PN1 weights saw them.
