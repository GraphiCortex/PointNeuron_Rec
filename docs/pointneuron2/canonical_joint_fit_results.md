# Canonical joint fit: exact required connectivity, one remaining confidence spur

21 October 2026. **Completed at the agreed cap. Training learnability and paired
improvement gates both remain false.** There are three complete crop passes;
the only terminal failure of the declared reconstruction criteria is one extra
node and spur on crop 0/1. No extension or threshold change is made.

The user-run fit completed all 3,000 new updates, from historical stage 1,000 to
4,000, in 2,485.09 report seconds (41.42 minutes; console includes setup). The
earlier short timing estimate overstated actual runtime; the update ledger and
Adam clocks confirm the entire budget executed. Phase 1, input clouds, model,
loss, LR 0.00001 and 0.5 node/edge decisions are unchanged. Only targets differ
from the [matched legacy-target baseline](stable_joint_learnability_results.md).

## Terminal automatic reconstruction

| Training crop | Accepted targets | Extras | Correct required edges | Nonreference edges | Precision / recall within 2 voxels | Full gate |
|---|---:|---:|---:|---:|---:|---|
| 0/0 | 6/6 | 0 | 5/5 | 0 | 100% / 100% | Pass |
| 0/1 | 11/11 | 1 | 9/9 | 1 | 93.8137% / 100% | Fail |
| 92/4 | 15/15 | 0 | 14/14 | 0 | 100% / 100% | Pass |
| 86/3 | 408/408 | 0 | 394/394 | 0 | 99.9589% / 99.7506% | Pass |

All **440 assigned targets** are accepted and trusted within two voxels. All
**422 required canonical connections** are selected. No extra edge joins two
assigned targets. The one nonreference edge attaches the extra query. Component
counts match the crop forests (1, 2, 1, 14), and all 79 reference junctions are
trusted with the expected degree. All four terminal SWCs serialize their actual
decoded graphs correctly. These are reference-relative training results, not
complete biological ground truth or an independent radius-accuracy assessment.

The dense crop's canonical graph is exact at the terminal checkpoint. The two
old endpoint/interior-sample reversals are no longer terminal failures. There
are fewer required edges after canonicalization, so the change from 404 old
dense edges to 394 new ones is **not itself** a reconstruction improvement.

## Matched geometry: small gains, incomplete required quality

Both arms are assessed against the same unchanged full SWC segments, using the
same half-voxel arclength quadrature and two-voxel distance tolerance.

| Crop / absolute error length | Legacy-target terminal | Canonical terminal |
|---|---:|---:|
| 0/1 extra out-of-tolerance length | 4.384089 | 4.324377 |
| 0/1 missed reference length | 0 | 0 |
| 86/3 extra out-of-tolerance length | 0.975229 | 0.972334 |
| 86/3 missed reference length | 7.221923 | 6.189789 |
| 0/0 and 92/4, both error lengths | 0 | 0 |

No crop's absolute extra or missed length increases. Across these four examples,
extra length decreases by about 0.062607 coordinate units and missed reference
length by 1.032134. These are small measured training-case gains, not substantial
or general reconstruction superiority.

`paired_geometry_noninferior=False` includes the predeclared **99% precision/
recall floor**, as well as absolute-length nonregression. Crop 0/1 fails that
floor despite a slightly shorter spur. `paired_improvement=False` additionally
requires full canonical learnability. The false verdicts therefore do not mean
that every measured geometry value regressed; neither verdict is relabeled.
This failure is a real spur, not the target-preparation denominator effect.

## Remaining failure: query 98 on crop 0/1

Query 98 has softmax confidence **0.679170**, exceeds the fixed 0.5 cutoff,
and is unmatched to a canonical target. It is 6.159 voxels from the nearest
target node and attaches to query 383 through a 6.246-voxel edge. About 4.324
units of the reconstructed curve lie outside the two-voxel reference tolerance.
This is meaningful extra structure on a short crop, despite being one node.

The attachment's edge logit is 2.947391. Its training label is **unknown** because
it touches an unmatched endpoint; the declared GAT edge loss masks it. Proposal
confidence owns rejection of unmatched queries and does receive a negative
classification label. This demonstrates the remaining weakness in that division
of responsibilities on this example, not a failed candidate-generation step.

The same query is unmatched at every saved evaluation. Its confidence is not
monotonic: approximately 0.99760 at stage 1,000, 0.99817 at 2,000, 0.99436 at
3,000, and 0.67917 at 4,000. In the old terminal baseline it scored 0.63689.
Canonicalization therefore does not solve this confidence residual. The saved
evaluations do not establish that its assignment was unchanged at every
intermediate optimizer update. Late score movement does not justify an automatic
extension beyond the agreed cap.

Across all four terminal crops, the **highest unmatched score is 0.679170** and
the **lowest assigned score is 0.898196**. The current outputs thus have a clean
ordering of these 440 assigned versus 1,608 unmatched queries, while the fixed
decision boundary accepts one extra. A separating operating point exists on
these frozen training predictions. No alternative threshold or classifier bias
is installed, and this observation is not validation of a future default cutoff.
It also does not distinguish globally reliable calibration from incomplete
optimization on this small fitting task.

## Trajectory and verification

| Historical stage | Full crop passes | Extras on 0/1 | Correct canonical dense edges |
|---|---:|---:|---:|
| 1,000, imported | 0/4 | 19 | 378/394 |
| 1,008, exact timing replay | 0/4 | 19 | 379/394 |
| 1,064 | 1/4 | 18 | 386/394 |
| 1,500 | 2/4 | 15 | 394/394 |
| 2,000 | 3/4 | 11 | 394/394 |
| 3,000 | 2/4 | 3 | 393/394 |
| 4,000, terminal | 3/4 | 1 | 394/394 |

No saved snapshot passes all four crops. Dense topology briefly loses exact
adjacency at 3,000; placement stability alone does not guarantee monotonic
connectivity accuracy. The terminal result, not a selection of per-crop
checkpoints, is used for both declared verdicts.

The saved-output CPU audit verifies **28 prediction/SWC pairs, seven full-state
checkpoints and 3,000 update records**. It recomputes automatic selection,
inherited candidates/scores, known/unknown supervision, graph/continuous geometry
metrics, absolute lengths, both verdicts and SWC round trips. Initial weights,
Adam moments and RNG match the protected parent. The stage-1,008 checkpoint's
numeric model/Adam/RNG fingerprint matches timing exactly, as do all saved
prediction arrays. Final Adam clocks are proposal 8,000 / GAT 4,000; all moments
and weights are finite. Runtime final state/prediction reload has delta zero.

All incoming training examples and all 163 all-crop probes retain 100% trusted
matches. Both known edge classes are present on all 3,000 updates; an actual
all-unknown GAT skip is not exercised. The final stability window passes. The
terminal checkpoint is intentionally nonresumable. Protected sources, inputs
and baselines are unchanged; the review runs no model forward/update or CUDA.

## Research direction

Keep canonicalization as the current research candidate: this matched run
supports its usefulness for the dense training graph and shows no measured
absolute-geometry regression. It does not establish unseen superiority or make
the overall pipeline ready for broad use.

**The next focus should be the confidence-selection contract.** On these four
examples the remaining declared failure is final node acceptance; this result
does not justify another encoder, Transformer or GAT redesign, another target
rewrite, or extending this fit. Preserve the checkpoint as a stable reference.
The earlier failed 50/25/25 hard-negative comparison remains failed; it is not
recommended again merely because an extra persists.

Before a new experiment, discuss what confidence should mean and how one global
automatic operating policy will be fitted and validated across neurons. A cutoff
or equivalent bias chosen to clean these four examples would not establish
future-input reliability. Confidence calibration on held-out development neurons
becomes informative with a model trained beyond this tiny fitting set. The seven
repeatedly inspected validation neurons remain development data; untouched test
data should stay reserved. No calibration run, broader training, architecture
change or threshold edit is prepared or launched by this review.

Execution succeeds; complete four-crop learnability at 0.5 still fails; small
matched training-case geometry gains are measured. Whole-neuron merging,
generalization, matched PN 1.0 improvement, radius fidelity and product readiness
remain separate unestablished questions.

[Read-only audit](../../scripts/pointneuron2/review_canonical_joint_fit.py),
[verified evidence/provenance](canonical_joint_fit_results_seed0/summary.json),
and [original prospective contract](canonical_joint_fit_ready.md).
Raw model outputs remain in
`tmp/pointneuron2/local_runs/canonical_joint_fit_20261021/model/`.
Terminal checkpoint SHA-256:
`8e6d11d0a8051fb279a761b64ae9efc8fb8173e0182c4bfebbd8c97b2ca95beb`.
