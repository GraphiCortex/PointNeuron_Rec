# Matched canonical-target joint fit: prepared contract

21 October 2026. **45 targeted tests and the real-model zero-update CPU check
pass. The user-run GPU timing has now passed and been reviewed; full fitting
is still unlaunched.** See the [timing results and next command](canonical_joint_timing_results.md).
The timing command below documents the completed prerequisite, not a rerun request.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\scripts\pointneuron2\run_canonical_joint_fit.ps1 -Action timing
```

This creates ignored timing logs, predictions and diagnostic SWCs. It retains no
trained checkpoint and does not start training automatically. The CPU prerequisite
has already completed and need not be rerun.

## Question and controlled comparison

Does the existing model learn the branch-aware canonical targets under the
same bounded stable-LR recipe, and does its automatic reconstruction improve
against the original full SWC reference? Those are **separate verdicts**.
Reducing the number of target nodes or required adjacency labels is not itself
a reconstruction improvement.

The comparator is the completed [stable joint fit](stable_joint_learnability_results.md),
ending at historical stage 4,000. The new arm imports the **same historical
stage-1,000 model, Adam moments and RNG**, then changes only the target cache.
It does not continue from the old stage-4,000 terminal model or reset Adam.
Both arms inherit prior training on the legacy targets; this tests a matched
warm-start adaptation, not fresh-model training or mathematical learnability.

| Item | Fixed setting |
|---|---|
| TRAINING neuron/crop identities | 0/0, 0/1, 92/4, 86/3 |
| Distinct image-derived input points | 848, 2,048, 1,219, 2,048; exact cached arrays |
| Only treatment change | Canonical target counts 6, 11, 15, 408; formerly 6, 12, 16, 418 |
| Architecture | Existing native PointNeXt / Transformer / GATv2 prototypes |
| Phase 1 | Frozen, fully automatic image-derived point clouds |
| Initial optimizer clocks | Proposal 5,000; GAT 1,000 |
| LR / weight decay / global gradient clip | 0.00001 / 0.0001 / ten |
| Training loss | Existing balanced proposal loss plus available known edge-class means |
| Unknown edge labels | Masked; an all-unknown graph skips GAT updates and moments |
| Normalization / augmentation / schedule | Existing explicit per-crop policy; no augmentation or scheduler |
| Queries / graph inputs | All 512 queries reach GAT before final node selection |
| Candidate graph | Unchanged 16NN union radius eight, no post-selection rebuilding |
| Node / edge cutoffs | Inclusive 0.5 / 0.5 |
| New update cap | 3,000, ending at historical stage 4,000; no extension |

No development or test neurons enter this fit. It cannot establish whole-neuron
performance, PN 1.0 superiority, unseen generalization or product readiness.
Valid SWC export is required; independent radius accuracy remains unmeasured.

## Prospective preparation eligibility

The earlier [canonical preparation audit](canonical_targets_results.md) remains
failed under its strict normalized-fraction nonregression rule. Its raw report,
review, source hashes, and cache manifest (`training_ready: false`) are unchanged.
The old model's failed learnability result also remains unchanged.

This **new experiment** adopts an explicit eligibility policy before any GPU
updates: retain structural incidence/components, bounded added geometry/radius
error, candidate coverage and capacity; preserve input clouds exactly; require
at least 99% reference-curve precision and recall within two voxels, and no
increase in absolute out-of-tolerance predicted or reference arclength.
The length allowance is 1e-9 coordinate units for floating-point roundoff only.

Absolute extra length is `(1 - precision) × predicted length`; absolute missed
length is `(1 - recall) × reference length`, using the existing maximum-0.5-step
arclength quadrature. These are measured geometric errors relative to the SWC,
not biological ground truth. Normalized precision/recall are still reported.
They avoid treating a smaller denominator with identical bad curve length as a
new geometric failure. No curve samples are removed from evaluation.

The CPU check verifies that this prospective eligibility policy passes. It
authorizes this runner's scoped experiment without changing the cache's general
readiness flag or retroactively rewriting any prior gate. All preparation,
baseline, source checkpoint, input/cache and reference hashes are protected.

## Timing and replay integrity

On the RTX 3070 Ti Laptop GPU, timing must first reproduce the saved old arm's
initial centers, radii, confidence, decoder features, candidates, edge scores
and automatic selection **exactly**. Labels and Hungarian assignments may differ,
because the targets have changed. Reference data never enter model inference.

Then run eight throwaway updates on the new targets, cycle all four crops,
probe at four/eight, and verify exact model/Adam/RNG and prediction disk reload.
The timing report retains a numeric state fingerprint, not a trained checkpoint.
Timing success means execution/reload readiness, not reconstruction success.

A full fit, if launched separately, starts from the original shared parent and
must reproduce all eight timing updates and the exact stage-eight model/Adam/RNG
fingerprint. The old 64-update training trajectory is not required to match:
changed target labels necessarily change subsequent optimization.

The prior matched baseline took 71 minutes. Plan roughly 60–90 minutes on this
laptop GPU, with substantial load/I/O variability. Timing also reports a short
update-only extrapolation; it excludes later probe/evaluation/checkpoint costs
and is not a reliable full-job ETA. Runtime never changes the cap.

## Terminal decisions and stopping

**Canonical training learnability:** at the cap require all four original-form
automatic crop gates, now against their declared canonical target graphs:
every assigned node accepted, zero extras, exact required adjacency and no extra
edges, expected component counts, trusted junctions with correct degree, valid
SWC, and precision/recall within two voxels each at least 99%. Require the same
90%-trusted stability test over the final eight probes and final eight incoming
occurrences per crop. This verifies fitting the new representation.

**Paired geometric reconstruction improvement:** separately compare terminal
automatic reconstructions against the same unchanged full SWC segments used
by the old arm. Require the 99% precision/recall floor, no per-crop increase in
absolute extra or missed length (roundoff only), and an actual decrease in at
least one length. Require canonical learnability too. Do not compare 394 new
dense-crop adjacency labels with 404 old ones as if that were an improvement.

A canonical learnability pass can coexist with a paired geometry failure. That
would mean the easier representation was learned but a geometric tradeoff
remains; it is not grounds for automatic promotion or another extension. Even
a paired pass establishes only this heavily inspected four-crop training result.
Report terminal outputs and losses regardless of outcome; do not pick different
checkpoints per crop or select a favorable intermediate stage as the result.

Stop if any crop's mean matched distance exceeds four voxels at three consecutive
all-crop probes. Probe every four updates through 64, then every twenty. Check
finite state, active gradients and distinct proposal/GAT clocks after each step.
Source changes, replay failures and invalid reloads stop execution. No automatic
LR/threshold/loss change, rollback or budget extension follows a failure.

Full-state checkpoints precede graph/SWC exports at new updates 0, 8, 64, 500,
1,000, 2,000 and 3,000. Explicit resume can recover this fit's intermediate state
after infrastructure failure, preserving counters, RNG and cap. Terminal,
CPU-check, timing and geometry-stop states cannot resume. The new schema cannot
resume through the old stable-fit launcher.

## Verified preparation and later command

The 45 targeted tests cover geometry policy, empty-output failure, protected
structure, mixed Adam clocks, exact disk state/RNG/next-update recovery,
fingerprints, cap/options/source checks and initial inference replay independent
of changed assignments. The native CPU check exports four initial SWCs, probes
the four actual caches, performs zero updates, verifies noninterference and an
exact state/prediction reload, and confirms no CUDA initialization. GPU timing
has subsequently passed as recorded in the linked results; sustained fitting
remains unexecuted.

At the initial CPU state all 440 canonical targets are trusted within two voxels
and every one of the 422 required edges is present among the actual 512-query
candidates. The initial failures are selection and edge scoring/forest decisions,
not an already missing candidate connection. This does not guarantee that these
properties remain true during fitting; probes and failure decomposition continue.

After timing is reviewed, the separate bounded fit command is:

```powershell
& .\scripts\pointneuron2\run_canonical_joint_fit.ps1 -Action train
```

The launcher requires matching completed CPU and GPU timing reports. They are
not rerun automatically. Logs and checkpoints are ignored under
`tmp/pointneuron2/local_runs/canonical_joint_fit_20261021/`.

Visible [configuration](../../configs/pointneuron2/canonical_joint_fit.json),
[runner](../../scripts/pointneuron2/canonical_joint_fit.py),
[launcher](../../scripts/pointneuron2/run_canonical_joint_fit.ps1),
[tests](../../tests/test_canonical_joint_fit.py) and
[CPU validation/provenance](canonical_joint_fit_ready_seed0/validation.json).
