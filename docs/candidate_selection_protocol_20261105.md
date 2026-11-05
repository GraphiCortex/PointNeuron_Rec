# Experimental candidate selection and connectivity — 5 November 2026

User-approved departure for THIS experimental arm: the Transformer does not
propose nodes. Automatic image-derived candidates feed PointNeXt features and
learned node/edge selectors (trees first, then GATv2). This is a stronger
candidate stage before the diagram's Phase4. It departs from PDF Phase3;
professor discussion is pending. No historical result is changed, and the
failed dense head is not trained/tuned again. Its saved nodes are one fixed
candidate source, not an adopted reconstruction.

The user launches timed experiments from visible PowerShell commands. No GPU
work is launched by the assistant in this arm. Reserved neurons remain untouched.

## Stage0 — fixed CPU headroom measurement, selects nothing

All151 reference-present development boxes/seven neurons, repeatedly inspected
development data, not an untouched final test. The other264 grid boxes remain
annotation-unknown and excluded from this per-box study. No whole-image claim.

Four source arms: preserved pruned classical graph; frozen broad-run learned
nodes; original Sato-top512; their union. Use the same source geometry for later
Stages1/2. Classical crop extraction retains in-box vertices and original
endpoints of edges intersecting the box (image-derived halo). Keep inherited
classical edges and saved Sato/learned forest edges. No SWC defines candidates.

Merge within1 native voxel greedily, priority classical then Sato then learned,
lexicographic XYZ/index ties. Keep a real representative coordinate, OR source
flags and count source memberships. Assign each raw candidate to a representative
within1; do NOT use transitive single-link clusters that collapse a skeleton
chain. Remap inherited edges, dropping self-edges and deduplicating pairs.

Build exact Euclidean k6 and k10 unions plus inherited edges. The PRIMARY
oracle uses k10+inherited (k6 is a subset); k6+inherited coverage is diagnostic
only and cannot select k. Persist all candidates/edges/provenance/guide-root
scores BEFORE opening reference geometry for oracle scoring.

Primary geometric oracle, fixed before results:

- Node probability1 iff distance to FULL supplied trace <=2 voxels, otherwise0.
  Full-reference distance avoids creating projection targets at crop edges.
- Edge probability is the fraction of its positive-length IN-BOX segment
  within2 voxels of the same CLIPPED reference geometry used by the segment
  scorer. Same .5-voxel midpoint arclength quadrature; edges with no in-box
  length are ineligible. No inferred reference connection is added.
- Keep nodes with probability>=.5, edges with probability>=.5 and both endpoints
  retained. Minimum spanning forest of the eligible components with cost
  `-log(p_edge)+.5*(-log(p_node_u)-log(p_node_v))`, clip probabilities to[1e-6,1].
  Spanning eligible components is explicit: unrestricted minimization of these
  nonnegative costs would choose the empty graph. Do not force bridges between
  components. The probability-only decoder accepts no references.
- One simultaneous pass removes only leaf-to-junction terminal chains shorter
  than2 native voxels. Preserve whole short components; do not iteratively
  prune. Report removed nodes/length. Roots use image-guide maxima, NEVER SWC
  roots. SWC type0/radius1 are unsupervised placeholders.

Report this SAME two-voxel-selected oracle at both2 and6 voxels; no separate
six-voxel oracle tuning. Report each arm's graph sizes, source counts/flags,
outside-box halo nodes, k6/k10 candidate-edge recall at2/6, node-eligible-union
recall, oracle P/R/F1, forest components/isolates and pruning loss. Lengths pool
within neuron, then macro P/R/F1 across seven neurons. Retain per-box results.

Gate ONLY on primary UNION oracle macro F1 at2:

- >=.80: proceed to Stage1;
- >=.65 and <.80: proceed only alongside explicit candidate-recall work;
- <.65: stop and report, no selector training.

This is a reference-assisted GEOMETRIC oracle recipe, not an inference result,
not a global optimum over every possible subset, and not correct-connectivity
headroom. Its source geometry is image-derived but its oracle output directly
uses development traces and must never enter real inference or training data.
Candidate-union recall is the true geometric ceiling for selecting those fixed
segments; the oracle additionally has node/edge/pruning constraints. Geometry
can reward wrong joins. Passing does not establish label learnability, selector
accuracy, topology or full-neuron reconstruction. No parameter/source choice
is selected from these oracle comparisons. Save qualified artifacts/journals;
no automatic extension after a cap/failure.

## Stage1 — conditional on Stage0

Build the same pool on all460 reference-present training boxes/33 neurons.
Reuse existing classical outputs; extract missing training classical graphs
only from retained guides with the preserved fixed recipe, never refilter images.
Raw input points remain diagnostic only and are not added to the pool.

Proposed explicit path bound for the requested label rule:32 native voxels
(four original8-voxel grid spacings), fixed before label generation, not chosen
from development. Nodes: distance<=2 is trace-positive, farther is negative in
reference-present boxes. Ambiguous/coincident-component ownership is unknown.
Edges touching known negative nodes are negative. Otherwise positives require
unambiguous endpoints consecutive among projected positive candidates on the
same reference chain/path and path length<=32; no intervening candidate/junction
may be silently skipped. Distinct fragments/components and ambiguous junction
ownership are unknown, never welded/averaged. Same-component but nonconsecutive
or over-bound pairs are negative when the relationship is scorable. Reference-
empty boxes are unknown. Report counts/masked fractions and reason breakdowns
per neuron. Unique known fragments can supply within-fragment labels; relations
across fragments cannot. Actual label implementation follows the Stage0 gate.

Labels mean agreement with supplied traces, not exhaustive biological foreground.
Images can contain real untraced neurites penalized by this restricted task.
No training reference can select candidates, potential edges or roots.

## Stage2 — conditional selector CV, as explicitly chosen by the user

Use node/edge Sato, straight-edge intensity profiles, bounded image-derived
geodesic Sato cost, length/direction, candidate-geometry curvature proxies,
source flags/membership, raw-vote support and PointNeXt features. None is derived
from SWCs at inference. If encoder features are not saved, provide inference-only
feature-export PowerShell for the user; no assistant GPU launch.

The user explicitly chose selector-only leave-neuron-out CV CONDITIONAL on the
frozen upstream encoder and broad-run candidates. Both upstream components
have seen labels from all33 training neurons. Disclose that leakage/conditioning;
do not call this strict full-pipeline held-out CV. The seven development neurons
did not train those upstream components, but their set is repeatedly inspected.

Train gradient-boosted node/edge tree baseline first, with balanced loss/weights.
Choose settings/calibration using only the33 training-neuron leave-one-out folds.
Freeze model and decoding before its ONE development scoring. Under the user's
stop-on-failed-criterion rule, a final tree baseline that misses the reconstruction
criteria stops progression; do not silently continue to GATv2 after a failed
tree gate. If trees pass, train/select GATv2 on training-only folds, freeze and
score development once. GATv2 must also clear the reconstruction criteria AND
exceed the frozen tree model's macro F1 at2 by>=.02 to justify its role. No
development-selected threshold, architecture, checkpoint or extension.

Models/dependencies/resource estimates are prepared only after Stage0/labels;
no premature model training or package changes in this Stage0 preparation.

## Stage3, reporting and final criteria

Use the same probability-only threshold/spanning-forest/node-confidence cost
and fixed one-pass<2 spur pruning defined above; image-only roots, declared
placeholder radii until supervised radius exists. This is per-box until whole-
image assembly is implemented and qualified.

Development F1>=59.33% at2 and>=74.14% at6, AND wins on>=5/7 neurons at EACH
tolerance versus matched classical54.33/69.14. Use full-precision original
baseline+.05, not displayed rounding. Report training metrics without a gate.
Report oracle-F1 fraction captured AND `(modelF1-classicalF1)/(oracleF1-classicalF1)`
where the denominator is positive (otherwise undefined); do not clip negatives.

Wrong joins need explicit denominators. Report decoded edges joining different
unambiguous reference branches/components as requested, but separately identify
valid junction-adjacent branch transitions and unknown fragment relations: a
cross-branch edge is not inherently wrong, and fragmented SWCs do not certify
a biological disconnection. Also report reference-inconsistent shortcuts,
off-reference endpoints and ambiguity/masked fractions. Do not label the old
conservative shortcut fraction an exhaustive wrong-join rate. Break reporting
includes uncovered reference intervals/length and scorable projected adjacency
disconnections, with missing candidate opportunity separated from decoding loss.

Numeric stopping criteria above remain fixed. These additional risk diagnostics
are reported honestly, not converted into new post-hoc numeric gates. Stop on
failed criteria or execution/resource failure; no development tuning, automatic
extension, reserved-neuron access or silent promotion of an alternate pipeline.
