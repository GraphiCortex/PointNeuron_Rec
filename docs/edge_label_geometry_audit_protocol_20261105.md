# Pre-Stage2 edge-label geometry audit: fixed CPU contract

User explicitly requested agent CPU execution before ANY Stage2 feature/model
work. Training-only saved460 packets/33 neurons. No development labels, no
candidate/label changes, no training, filtering or inference. Existing Stage0,
Stage1, label-audit and feature-export evidence stays unchanged. This audit
selects no model, candidate policy or active supervision contract.

## Measurements fixed before results

For exactly180,269 edge_reason=nonconsecutive negatives and163,536 positives,
clip each original candidate segment to the frozen native voxel-cell crop.
Use the EXACT Stage0 .5-native-voxel midpoint quadrature and point-to-continuous-
reference distance<=2. For each neuron and pooled: in-box/supported length,
length-weighted overlap, unweighted edge mean/quantiles, histogram bins
[0,.25),[.25,.5),[.5,.75),[.75,.9),[.9,.99),[.99,1], counts>=.5/.9/.99 and<.5.
Zero in-box length is separate and has undefined overlap; it cannot count as
on-trace. Primary requested trigger denominator is ALL180,269 negatives:
at least30% with positive in-box length AND overlap>=.5. Also retain the full
distribution so the meaning of “on trace” is not hidden by this threshold.

Replay original training projection/chain memberships and saved junction
anchors without changing coordinates. Classify negative pairs as same-chain
skip, same-chain other, or cross-reference-chain (cross-branch pairs). A skip
has another positive candidate whose projection is strictly between endpoints
on a shared reference chain, whose Euclidean position lies within1 voxel of
the segment interior. Exclude endpoints and endpoint parameter ties. A separate
geometric-witness flag counts positive candidates within1 of the segment from
ANY branch, and its intersection with cross-branch pairs. These categories are
not forced into a misleading skip-versus-cross-branch dichotomy. “Cross-branch”
here means distinct reference-chain memberships, not proven biological error.
Witnesses refer to the full original segment, including its image-derived halo;
overlap measurements always use its in-box portion.

## Units and micron view

Current candidate scripts use native XYZ unchanged: effective spacing(1,1,1),
native two/six-voxel tolerances, not two/six microns. General geometry_metrics
can apply spacing; these candidate scripts do not invoke that transformation.
Reported .pixelsize.txt sidecars exist for12/33 TRAINING cases. Parse and hash
only those explicit training-case sidecars, retain exact text/provenance and
XYZ order assumption, label calibration UNVERIFIED. Missing21 cases stay native;
never substitute1um/voxel. No new development scoring for this view.

Transform BOTH candidate/reference geometry by reported anisotropic XYZ spacing
after native clipping. Diagnostic tolerances0.5/1/2um, midpoint step<=0.1um;
no physical criterion/model selected. Report positive/negative edge-overlap
distributions and physical scores/length ratios of the already saved training
label/geometric oracle SWCs on the same subset. Native gates and labels remain
unchanged. These are conditional reported-metadata views, not calibration claims.

## Length accounting, no reconstruction reruns

Read canonical completed PN2 curve/scaffold/transfer/broad/headroom/label-oracle
reports; export every numerical predicted_length/reference_length pair with
source report, metric pointer, available neuron/tile/policy and units. Include
per-box and per-neuron/aggregate entries, candidate-union geometry entries
identified by pointer, with explicit warning they are not independent runs.
Zero-reference ratios undefined. Isolated point-only results have no predicted
segment length and are not assigned a fabricated ratio. Add current draft
and conditional physical oracle entries. No old prediction/graph/scoring rerun
is needed for native length accounting.

## Conditional PRE-TRAINING draft, never applied

If the fixed30% count-share trigger fires, document two options and run only
their training-label diagnostic oracles. Both are specified now and compared
without adopting/choosing from their scores. They are motivated by TRAINING
label geometry, not selected from development results. Keep all node labels,
unknown/coincident/fragment masks, inference geometry and decoder constants.

A: mask only the nonconsecutive negative edges with in-box overlap>=.5 as
unknown. All other labels remain unchanged. This reduces loss contradictions
without teaching on-curve skips as positive. It loses some hard-negative
supervision. Rerun decode; its binary eligibility is identical because old
negatives AND new unknowns are both ineligible. Require exact SWC hash replay
before reusing the original segment metrics. Oracle F1 cannot assess the
training-loss effect of this masking change.

B: for originally KNOWN edges with two positive endpoints, use binary
geometric agreement: positive iff in-box overlap>=.5, negative otherwise.
Keep ALL original unknown masks and all known-negative-endpoint negatives.
Remove consecutiveness/path-bound classification only from otherwise known
positive-endpoint edges. This is a MASK-PRESERVING BINARY geometric variant,
not exactly Stage0's soft fractional probabilities or unmasked geometric nodes.
It can teach on-curve shortcuts/redundant connections; forest validity alone
does not resolve that topology/multiplicity risk. Score its diagnostic forest
with the existing native scorer, report length ratio, not F1 alone.

Neither option produces an adopted label manifest or overwrites any saved
labels; only isolated diagnostic probabilities/receipts/SWCs are generated.
The source labels are hash-checked afterward. Any later training-contract
change requires discussion; no Stage2 preparation/training follows automatically.
Earlier gates stay on record, and oracle geometry never enters real inference.

## Execution

CPU runtime `tmp/phase1_cpu_env/Scripts/python.exe`, output
`tmp/edge_label_geometry_audit_20261105`,3600-second wall cap, durable per-box
records, no automatic retry/extension. Estimate several to tens of minutes,
unmeasured for this new audit. A temporary system-awake request during the
authorized CPU work is released afterward; no global power plan is changed.

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\phase1_cpu_env\Scripts\python.exe' -B '.\scripts\audit_edge_label_geometry.py'
```
