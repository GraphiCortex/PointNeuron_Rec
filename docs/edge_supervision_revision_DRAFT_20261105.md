# PRE-TRAINING supervision revision — DRAFT, NOT ADOPTED

This is a discussion contract, not the active label generator or a training
manifest. The user's training-only audit trigger was fixed at30% before its
run; observed179,171/180,269 nonconsecutive negatives (99.39%) have in-box
overlap>=.5 with the supplied trace at2 native voxels. Their length-weighted
overlap is98.70%. Only3,709/180,269 are cross-reference-chain pairs;96,363 have
an intervening same-chain positive candidate within1 voxel of the segment.
The consecutive-node target is substantially different from geometric edge
agreement. A near-oracle decoded F1 did not validate that target alignment.

This PRE-TRAINING revision is motivated by TRAINING label geometry, not chosen
from development results. Both alternatives below were specified before the
current overlap results and diagnostic-tested on the same460 training packets.
Neither is chosen from its oracle score. No new development labels/scoring,
candidate selection, original label edits, Stage2 features or training occurred.
Earlier gates/results stay on record, including the old label-oracle gate pass.

## Shared supervision/inference boundaries

- Keep all fixed candidates/edges, source flags, node trace-agreement labels
  and original unknown/coincident/fragment masks. Reference-empty cases remain
  unknown. A known-negative endpoint still makes the edge negative.
- Reference SWCs supply TRAINING labels and explicit diagnostic scoring only;
  never inference candidates, roots or edges. Unknown means loss-masked, not
  negative. No global repair, component welding or crop-edge projection targets.
- Native geometry/thresholds are unchanged. Conditional micron views use only
  reported sidecars, are unverified and select no physical tolerance/policy.
- No dataset/optimizer/architecture/decoder change is adopted by this document.
  The experimental Transformer departure remains professor-discussion pending.

## Option A: mask geometric conflicts

For an originally nonconsecutive negative edge, if its in-box arclength is
positive and at least half is within2 native voxels of the reference, mask it
unknown. Leave every other node/edge label unchanged. This would mask179,171
edges and preserve163,536 positives. It also masks3,199 cross-chain pairs with
high geometric overlap; it therefore removes some potentially useful topology
negatives, not just harmless duplicate-node skips.

Tradeoff: avoids treating on-trace skips as ordinary geometric negatives, while
retaining the original positive discretization. It does not supply positive
supervision on those skips or teach unique representative selection by itself.

Diagnostic oracle: rerun the original binary decoder with an in-memory masked
copy. Its eligibility is identical: old negatives AND new unknowns are both
ineligible. All460 fresh SWC hashes match the original label oracle exactly;
the original segment scores are reused after that exact replay proof.
F1_2/F1_6=90.82/96.68%,pooled length ratio1.5624. This check CANNOT measure
the learning effect of masking versus negative loss. No training labels saved.

## Option B: mask-preserving geometric agreement, topology downstream

For an originally KNOWN edge with TWO positive endpoints, target1 iff positive
in-box length and overlap>=.5 at2 native voxels; otherwise target0. Retain ALL
original unknown masks (including coincident projections, unanchored junction
transitions and cross-component relations), and retain known-negative-endpoint
negatives. Thus only otherwise known positive-endpoint labels can change.
Consecutiveness and the32-voxel path bound no longer define those targets.

This is a BINARY, MASK-PRESERVING geometric variant: not identical to Stage0's
soft fractional edge probabilities or its unmasked geometric node oracle.
It would have342,242 positive edges. The444 original positives with overlap<.5
and24 with zero in-box length become negative; on-trace nonconsecutive edges
and qualifying over-bound known edges become positive. No target file is adopted.

Tradeoff: aligns edge supervision with annotated curve agreement, but permits
redundant on-curve skips and some cross-branch connections. Trace agreement
must not be called correct connectivity. Forest acyclicity alone does not
solve shortcuts, branching incidence or multiplicity. Any adopted version needs
an explicit downstream plan for these risks before claiming topology handling.

Diagnostic oracle with the UNCHANGED forest/pruning decoder: F1_2/F1_6=
91.05/96.78%,pooled length ratio2.1363. Compared with the original label oracle,
that is only+0.23pp F1_2 but+36.73% predicted length. This is not evidence to
adopt the option or a reconstruction-quality improvement. Outputs live only in
`tmp/edge_label_geometry_audit_20261105/draft_only`, with adopted=false receipts.

## Decisions needed before Stage2

1. Decide what node/edge confidence means: trace support versus selecting one
   representation. Current node positives include ALL candidates near a trace;
   they do not designate unique representatives. A decoder spanning all eligible
   supported nodes can retain redundant near-curve paths. Removing the edge
   consecutiveness target does not resolve this node/decoder responsibility.
2. Choose/discuss a contract; do not choose by oracle F1 alone. No option is
   adopted here. If training supervision changes, freeze a NEW label manifest
   and training-only control/CV plan, keeping original packets as evidence.
   Old fixed-feature AP describes the old targets and cannot silently transfer
   to a changed label task.
3. Predeclare how excess length, redundant paths, wrong joins and breaks enter
   assessment/decoding alongside coverage. Existing geometric floors remain
   frozen historical criteria, not proof of final reconstruction quality. No
   new numeric length gate or decoder is invented/adopted by this audit.
4. Preserve conditional selector-only LONO CV/upstream exposure disclosure,
   handcrafted-only comparison, one scoring per frozen development model and
   reserved-neuron isolation. No Stage2 feature/model work proceeds before
   discussing this revision and its decoding implications with the user.

See [complete overlap/length/unit audit](edge_label_geometry_results_20261105.md).
