# PN1 versus PN2: matched input evidence comparison

29 October 2026. User-requested input comparison. Fixed before scoring. This
authorizes a bounded CPU preparation/scoring test, not a PN1 rebuild/retrain,
model benchmark or change to the intended PN2 architecture.

## Question and controlled domain

Does PN2 retain more useful source-image and traced-curve evidence than the
current local PN1 implementation at a matched input resource budget?

Use the identical two training source files and four boxes from the completed
[real-cloud audit](real_branch_evidence_results.md). Reuse its 304 measurement
windows, background values, width/contrast strata and source/reference hashes.
No new case selection, annotation edits, alignment fit or reserved-data access.
Source indices/XYZ are common across arms. Correctly decoded source values are
the common reference for evidence scoring, including points selected using the
legacy decoder; legacy attributes are also retained and checked for fidelity.

This operationalizes "information" as source intensity fidelity, retained local
positive contrast, traced-curve support/gaps and distinct observations. It is not
an information-theoretic entropy estimate or a learned reconstruction score.

## Why distinguish decoding from selection

The original `main` decoder still contains the PBD8 bit-order and PBD16 difference
bugs documented in [the reset audit](research_reset_evidence.md). On the same frog
file, its current decoder produces a 0..65535 range and 27,361 voxels above 13107;
the verified reader produces 5..4095. PN1's proposal threshold is
`round(0.2 * datatype_maximum)`: 51 for uint8, 13107 for uint16. A saved guarded30
proposal archive for this exact source records threshold 13107, random sampling,
and cell size 8. Its per-patch point count is not serialized; 2048 is the current
aggregation code default, not a newly claimed historical archive fact.

These are local implementation/encoding issues, not evidence against the original
published PN1 architecture. The comparison must not credit their repair to Sato.
No PN1 file, checkpoint or historical score is modified.

## Seven fixed arms

| Arm | Decoder / candidate rule | Point selection |
|---|---|---|
| PN1 native | Current main decoder; 20% datatype-range gate | Actual PN1 random helper |
| PN1 decode-only control | Verified decoder; same datatype-range gate | Same random helper |
| PN1 source-range random control | Verified decoder; 20% verified whole-image observed maximum | Same random helper |
| PN1 source-range spatial control | Same source-range gate | Actual PN1 spatial helper, cell size 8 |
| PN2 uniform | Verified decoder; full-grid eligibility | Frozen uniform PPS ranking prefix |
| PN2 intensity | Same | Frozen intensity-weighted PPS ranking prefix |
| PN2 Sato | Same | Frozen Sato-weighted PPS ranking prefix |

The two source-range arms are explicit counterfactual PN1-style controls, not
retrospective corrections to PN1's historical result. They isolate structural
selection from datatype mismatch and offer the supported spatial sampler as a
stronger comparison. Their 0.2 fraction is inherited and not tuned per image.

Compile the exact pure selection/threshold functions from hash-pinned `main`
source, without model imports or directory scans. Use its actual legacy decoder
read-only. All inference candidate/selection paths are reference-free.

## Budgets and duplicate handling

Primary requested count: **2048 slots for every arm**. Secondary: each box's
existing dense PN2 count (2430, 6912, 20736, 20736), matched across all arms.
Five fixed seeds per arm and budget give 280 records. No threshold sweep.

PN1's native helper samples with replacement if the eligible pool is too small.
Keep that behavior, record actual/distinct/duplicate counts, and score every
source voxel only once. Empty pools remain explicit empty inputs. Do not shrink
PN2 counts or add artificial PN1 foreground to conceal a candidate shortage.
Requested slots are matched; distinct counts may not be, and that fact is part
of the result rather than hidden as an exact distinct-budget match.

PN2's stored per-box race ordering has nested prefixes, so the 2048-point view is
the same lower-budget PPS selection with unchanged guide, seed and tile. No new
guide or PN2 sampling run is needed. Native PN1 RNG is independently initialized
per box/seed/budget: this is not a replay of historical global patch RNG state.

The common boxes intentionally replace PN1's native overlapping 193-wide crop
schedule to isolate point preparation on identical voxels. Do not run its patch
scheduler or discard common boxes below its 256-foreground minimum; record that
eligibility separately. This is not the exact complete as-shipped PN1 frontend.

## Metrics, checks and interpretation

Reuse contrast mass at radii 1/2/4/8 and the existing strata. Add curve quadrature
at 0.5-voxel maximum spacing: nearest selected source-point support at radii 2/6,
and longest unsupported interval. Break runs when paths leave/reenter a box.
Candidate-pool results before sampling identify hard-gate loss separately from
sampling loss. These are input support proxies, not mathematical ceilings on a
learned model, which can infer centers from surrounding evidence.

Check PN1 helper semantics, decoder discrepancy fixtures, duplicate mass handling,
nested PPS prefixes and clipped-curve gap counting before the run. For the dense
PN2 arms, contrast fractions must exactly replay the previous audit. Preserve
all original outputs. Stop on provenance or replay mismatch, without a fallback
threshold, alternative case or automatic extension. No GPU, weights or training.

Interpret gains against PN1 native as a combined implementation difference;
interpret comparison against the correctly decoded, range-aware random/spatial
controls as the stronger test of selection. Report failures and counterexamples.
The limited six-window apparent-thin/dim stratum remains insufficient for broad
thin/dim or future-input claims. A mixed result must remain mixed.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe `
  .\scripts\compare_pn1_input_evidence.py --output .\tmp\pn1_input_comparison_20261029
```
