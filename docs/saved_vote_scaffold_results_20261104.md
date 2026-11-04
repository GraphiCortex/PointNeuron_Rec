# Saved update-2000 diagnostics and graph measurement — completed 4 November 2026

**Decision: stop this recipe.** The original dense-head gate remains failed.
The one eligible, post-failure consolidation variant also failed every original
gate check. No extension, second variant, sixteen-box manifest, GPU run or
training follows. This report records the user's completed local CPU launch;
the assistant did not launch or repeat the experiment.

Read alongside the [prespecified follow-up recipe](saved_vote_scaffold_protocol_20261104.md),
[original terminal results](local_curve_terminal_results_20261104.md), and
[metric/proposal/implementation audit](proposal_implementation_audit_20261104.md).
The complete portable measurement and verification receipt are in
[evidence](evidence/saved_vote_scaffold_20261104.json).

## Scope and execution

- Saved update-2000 predictions only: fruitfly alias 32, boxes 27/20; frog
  alias 22, boxes 2/5. Four **training boxes, two neurons**: fit-level evidence.
- User command: `tmp/phase1_cpu_env/Scripts/python.exe -B scripts/measure_saved_vote_scaffold.py`.
- Output: `tmp/saved_vote_scaffold_20261104T182353_539123Z`; report.json/report.md,
  A/B/C component reports, raw support arrays, eight graph/SWC pairs and inference receipt.
- Worker 13.20 seconds; guard 14.52 seconds, exit 0, completed under 1800-second cap.
- No model execution, new training, GPU, image filtering, classical extraction,
  development inference, reserved-data use or whole-image merging.
- Config fingerprint: `a665103c70a9b2b5a38fd40488f3d22c05f63793b590bf0508f45234c16b84dd`.
  Original scientific fingerprint remains
  `8c3d98daaeac38a281defad1206bd152af50ea407515d3b77e0229c8fbf79d45`.
- Follow-up source/input hashes (12) and all eight graph/SWC pairs were checked
  after completion. Candidate-union recall is nested with k; decoded recall is
  no greater than the k=6 union in every box. Forest counts are consistent and
  no next-rung manifest exists. These checks validate execution/provenance,
  not biological connectivity.

The earlier three-second JSON-key failure is preserved separately; it produced
no completed A/B/C result. No output from that failed launch is substituted here.

## A. Where retained-point errors occur

Support means accepted raw votes (score >=.25), within two voxels of the retained
position, including its own vote. The complete evidence also includes the
unthresholded raw-vote histogram, which did not control B.

| Accepted raw supporters | Retained | Beyond reference by >2 | Pooled precision | Neuron-macro precision |
|---|---:|---:|---:|---:|
| 1 | 197 | 126 | 36.04% | 35.24% |
| 2 | 205 | 86 | 58.05% | 59.75% |
| 3–4 | 348 | 99 | 71.55% | 73.03% |
| 5–9 | 526 | 50 | 90.49% | 91.22% |
| 10+ | 302 | 8 | 97.35% | 96.90% |

The 402 retained positions with fewer than three supporters contain 212/369
off-reference positions (57.45%); their pooled precision is 47.26%, versus
86.65% for supported points. Support count is associated with agreement, but
is also associated with useful spatial coverage; it is not a free error filter.

| Source distance to reference | Retained | Off-reference votes | Share of all 369 errors |
|---|---:|---:|---:|
| [0,2) | 491 | 19 | 5.15% |
| [2,4) | 690 | 168 | 45.53% |
| [4,6] | 336 | 130 | 35.23% |
| >6 | 61 | 52 | 14.09% |

All four boxes are reference-present. There is no unknown-box population in
these counts. Nevertheless, distance from an incomplete trace is disagreement
with that trace, not proof that the image contains no biological neurite there.
Far sources do not dominate this failure; most errors come from sources within
the six-voxel offset-supervision domain (including ambiguous projections).

Same isolated-point scorer, neuron-macro P2/C2:

| Image-only control | Point agreement P2 | Curve-to-point coverage C2 |
|---|---:|---:|
| Sato-ranked, original radius-1 consolidation, full | 7.53% | 63.03% |
| Same, cap 512 | 18.26% | 40.60% |
| Classical nodes, radius-1 consolidation, full | 55.73% | 50.21% |
| Same, cap 512 | 55.73% | 50.21% |
| Raw preserved classical graph nodes, no consolidation | 57.30% | 53.30% |

The dense head's original 74.96%/83.32% exceeds these controls on its training
boxes. That supports learned fitting beyond simple Sato ranking here. Unequal
counts/selection policies, no replication and no development result prevent a
generalization or broad superiority claim.

## B. The sole conditional consolidation failed

The criterion was committed before the new support-count results: fewer-than-3
support points must account for >=half the errors and have lower pooled
precision; >6-voxel sources must not dominate. It was met. The previously known
far-source share was disclosed in the protocol, not treated as blinded evidence.

Exactly one variant ran: score >=.25, >=3 accepted raw supporters within two
voxels, then score-ordered radius-2 suppression retaining real vote positions.
This is a **post-failure analysis**, not the frozen original recipe or a
retroactive rescue. No constants were selected from these results.

| Point policy | Accepted positions | Neuron-macro P2 | Neuron-macro C2 |
|---|---:|---:|---:|
| Original radius-1 | 1578 | 74.96% | 83.32% |
| Supported radius-2/minimum-3 variant | 456 | 81.89% | 61.35% |

The unchanged gate requires P2>=90%, C2>=80% and >=90% of the original oracle
(effective macro C2 floor 80.23%), plus >=80% of each original box oracle.
The variant fails precision, absolute coverage, macro oracle ratio and minimum
box ratio. P6=100%/C6=75.36% does not override this verdict.

| Box | Variant P2 | Original C2 | Variant C2 | Variant/original-oracle C2 |
|---|---:|---:|---:|---:|
| Fruitfly 27 | 91.53% | 100.00% | 98.98% | 98.98% |
| Fruitfly 20 | 81.13% | 98.73% | 90.52% | 90.73% |
| Frog 2 | 76.00% | 100.00% | 96.30% | 96.30% |
| Frog 5 | 83.33% | 64.91% | 25.27% | 32.99% |

Across the four box-local reference domains, 230.26 voxel-length units lose
two-voxel point support and 2.46 gain support. These are spatial support changes,
not certified counts of deleted biological branches. Per-clipped-run details
are retained in the evidence. Frog 5 contributes 178.04 lost length; its longest
uncovered run grows from 19.49 to 47.81 voxels. Its incident-chain probe support
falls 16/22 -> 6/22, covered leaf locations 2/3 -> 0/3, covered junctions 4/6 ->
2/6, and boundary locations 2/6 -> 0/6. Apparent-thin/dim support falls from
8.84/29.86 length to 0/29.86; frog 2 retains its separate 3.68/3.68 stratum.
Structural proximity/probes do not certify connectivity or biological thickness.

## C. Crude per-box graph/SWC measurement

Both policies used the same fixed image-only scaffold: Euclidean kNN union,
k=6 for decoding, cached-Sato straight-edge cost
`L*(1+4*(1-mean_normalized_Sato))`, deterministic minimum spanning forest.
No reference edges, labels or roots entered inference. Graph/SWC artifacts were
saved before reference scoring. Roots are image-guide maxima per component;
type 0/radius 1 are unsupervised placeholders. No new gap bridges were added.

Segments use the same scorer/domain as the preserved classical graph, .5-voxel
quadrature, tolerances 2/6. Directed length counts are pooled within neuron;
P/R/F1 are then macro averaged across two neurons. Macro F1 is not recomputed
from macro P/R. All values below are **fit-level geometry**.

| Graph on these same boxes | P2 | R2 | F1 at 2 | P6 | R6 | F1 at 6 |
|---|---:|---:|---:|---:|---:|---:|
| Original nodes + fixed scaffold | 76.16% | 87.64% | 81.49% | 96.71% | 97.54% | 97.12% |
| Supported r2/min3 nodes + same scaffold | 86.90% | 70.95% | 76.43% | 100.00% | 78.54% | 86.39% |
| Preserved classical graph | 56.19% | 53.14% | 52.46% | 68.52% | 67.36% | 66.99% |

| Policy / box | P2 | R2 | F1 at 2 | Components / isolated nodes |
|---|---:|---:|---:|---:|
| Original / fruitfly 27 | 95.38% | 100.00% | 97.64% | 2 / 0 |
| Original / fruitfly 20 | 82.56% | 99.06% | 90.06% | 1 / 0 |
| Original / frog 2 | 81.38% | 100.00% | 89.73% | 1 / 0 |
| Original / frog 5 | 65.92% | 74.00% | 69.73% | 1 / 0 |
| Variant / fruitfly 27 | 96.50% | 100.00% | 98.22% | 2 / 0 |
| Variant / fruitfly 20 | 92.74% | 96.07% | 94.37% | 2 / 0 |
| Variant / frog 2 | 89.98% | 100.00% | 94.73% | 1 / 0 |
| Variant / frog 5 | 78.17% | 40.49% | 53.35% | 2 / 0 |

| Candidate-union R2, selects nothing | k=4 | k=6 | k=10 |
|---|---:|---:|---:|
| Original, neuron macro | 90.86% | 92.60% | 94.53% |
| Variant, neuron macro | 72.27% | 72.50% | 76.33% |
| Original, weakest box (frog 5) | 80.59% | 84.21% | 88.19% |
| Variant, weakest box (frog 5) | 42.72% | 42.83% | 51.08% |

Connecting original nodes recovers some between-anchor curve support, but leaves
substantial extra geometry. Across box-local domains its predicted segment
length is 2789.24 versus reference length 1161.36 (2.40x). High within-tolerance
overlap can coexist with redundant paths and wrong joins. A spanning forest
can force spurious nodes into connected structure; zero isolated nodes does
not mean zero spurious nodes. No wrong-join rate was measured or topology
certified. The classical comparison is now geometrically matched on these
boxes, but the learned coordinates were fitted to them and the classical graph
was not; this is not held-out method superiority.

**Correction to a possible node-gate interpretation:** isolated-node coverage
is not a hard ceiling on reconstructed-curve recall, because edges span gaps
between anchors. The union of permitted fixed candidate segments is the
geometric recall ceiling for an edge-only selector. Even original k=6 union
R2=92.60% exceeds decoded R2=87.64%; neither measures true-edge correctness.
The audit's prospective discussion floors (95% macro/90% every box) are not
met by any reported original k, and have not been adopted as a new gate.

## What this establishes, and the next discussion

Execution: the CPU measurement and SWC round-trip completed; provenance checks
pass. Training learnability: the earlier dense head learned this small fit,
but did not meet its joint localization/coverage gate. Development accuracy:
this follow-up measures none. Final reconstruction: the graph scaffold measures
per-box geometry only; GATv2, learned topology, correct roots/radii, overlapping
crop merging and whole-neuron reconstruction remain unqualified or unbuilt.
The original failed gate stays failed even though segments cover more than
isolated points.

One formulation hypothesis for discussion, **without implementation or run**:
the dense source-affinity/offset objective does not directly train a compact,
coverage-preserving set of curve representatives. A curve-supervised Transformer
could select/refine representative anchors using shared context, with its
prospective assessment based on fixed candidate-graph opportunity, structural
support and spurious connected structure. This retains a meaningful Transformer
proposal role. The present evidence motivates testing that hypothesis; it does
not prove attention is the remedy. A concrete supervision/comparison/budget
would need agreement before implementation. No further launch command is due
under the stopped recipe.

The linked D/E audit records the proposal/repo localization tolerances and the
PDF/diagram/code table, including the experimental bypass/NMS departure,
unsupervised radius, missing topology losses/multi-scale queries, polyline versus
discrete-node supervision, and orange stages. Those findings remain unchanged.
