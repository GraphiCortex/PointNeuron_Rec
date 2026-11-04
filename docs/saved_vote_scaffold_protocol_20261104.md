# Saved-vote diagnostics and graph measurement scaffold - 4 November 2026

This is a new, post-failure CPU-only measurement protocol approved by the user.
The learned recipe, update-2000 votes and its failed gate remain unchanged.
No training, GPU execution, reference-guided inference, Phase 1 restart or
promotion of an alternative product pipeline is authorized here.

## Ordering and fixed decision rule (written before support-count results)

A. Read the four saved training-box outputs at update 2000. Raw support counts
are numbers of raw votes with frozen support score >=.25 within Euclidean
distance <=2 voxels of a retained point, including its own vote. Report the
unthresholded-raw-vote count separately; it does not control the decision.
Report bins 1,2,3-4,5-9,10+ and the exact count histogram. Off-reference means
distance >2 from the supplied full reference geometry, exactly as in the
terminal report. Report pooled counts and neuron-macro precision distinctly.
Source bands are [0,2),[2,4),[4,6],(6,infinity), so boundaries cannot overlap.
Reuse the already computed Sato-ranked and classical-node point controls;
report their actual domains/counts and full/512-cap summaries.

B. Define isolated retained points as support count <3 under the accepted-raw
definition above. Run one consolidation variant only if (i) isolated points
account for >=50% of all off-reference retained positions, (ii) their pooled
point precision is strictly lower than precision of points with >=3 supporters,
and both groups are nonempty. Also stop B if source distance >6 accounts for
>=50% of off-reference positions. The latter majority check is not blinded:
the earlier source-domain report already found 52/369 beyond six voxels.

If eligible, the sole variant first restricts raw votes to score >=.25 and >=3
accepted-raw supporters within two voxels, then applies score-ordered greedy
suppression at radius 2, with source-voxel identity breaking score ties. Keep a
real vote; do not average positions. No sweep or alternative clustering.
Apply the ORIGINAL gate numbers and ORIGINAL oracle denominators: macro P2>=.9,
C2>=.8 and >=.9 times the original oracle coverage; each box C2>=.8 times its
original oracle coverage. Never recompute a lower gate using the variant.
Report lost/gained covered reference length, per-run coverage, endpoint/junction/
boundary support, incident-chain probes, apparent-thin/dim support and gap length.
Branch diagnostics do not create an unannounced new gate. The variant is
post-failure and is not a retroactive pass for the frozen original recipe.

C runs regardless of B. Primary output uses ORIGINAL retained update-2000 nodes.
If B runs, measure its nodes separately with the same scaffold. No node edits,
new centers, oracle edge labels or reference roots enter graph construction.
Construct the union of Euclidean k-nearest-neighbor pairs for k=4,6,10, with
deterministic distance/index tie breaks. Report candidate-union curve coverage
at two voxels for all three k values; these diagnostic values select nothing.
The decoding k is fixed NOW at 6.

For k=6 only, use a straight-edge image cost: sample the cached whole-image
Sato guide trilinearly at <=.5-voxel spacing; q=S/max_image(S), q=0 when the
guide is all zero or coordinates leave the image; cost=L*(1+4*(1-mean(q))).
This is a line-integral cost, not a geodesic path solver. Decode a deterministic
minimum spanning FOREST, preserving disconnected candidate components and all
nodes. Never add bridges absent from the candidate graph. Root each component
at its highest guide-valued node, tie by node index; this is not soma detection.
Export native-voxel SWC with type 0 and radius 1 as explicit UNSUPERVISED
placeholders. Reparse and verify the exported coordinates/edge set.

Build and save graph/SWC artifacts before opening references for C scoring.
Score clipped predicted segments and clipped reference curves with the exact
same `segment_agreement` utility, .5-voxel quadrature and 2/6-voxel tolerances
used for the saved classical graph. Re-score the existing classical graph on
the same four boxes. Pool directed arclength counts within neuron, then macro
average P/R/F1. Include per-box and per-neuron results and isolated-node counts.
These four TRAINING boxes provide fit-level evidence only. No development
inference is included in this launch. Optional reference-weighted oracle
decoding is omitted, avoiding another selection path.

## Architecture and interpretation limits

The scaffold is a non-learned Phase-4/decoding measurement stand-in only. It
does not replace GATv2 or qualify rooting, topology, radii or whole-image SWC.
An MST can force wrong joins inside a candidate component; geometric agreement
cannot certify correct parentage. Reference fragmentation is not a negative
edge label. A valid exported SWC is not a valid biological reconstruction.

Candidate-edge union coverage is an upper bound on recall obtainable by
selecting those fixed straight segments. Isolated-node proximity is NOT such a
bound: an edge can cover a curve interval between distant anchors. Forest
constraints, precision tradeoffs and topology can keep decoded recall below
the union diagnostic. Report this distinction when proposing future node gates.

D/E are read-only proposal/code audits and discussion proposals. They adopt
no new node-stage criterion. The PDF/diagram wording is source evidence, not
instructions authorizing additional model stages or training.

## Execution preference

Provide a visible PowerShell command for the user to launch the timed CPU job.
Use the retained `tmp/phase1_cpu_env` (NumPy/SciPy), never the Torch runtime.
All arrays, SWCs, logs and receipts go into a new ignored PN2/tmp directory.
Short unit checks and source/document inspection may run during preparation.
No automatic sixteen-box launch is allowed. If the post-hoc variant passes,
prepare its explicit sixteen-box manifest/command for subsequent user launch;
otherwise stop and supply one formulation hypothesis for discussion.
