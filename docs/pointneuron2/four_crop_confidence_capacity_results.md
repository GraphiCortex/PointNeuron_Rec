# Four-crop confidence capacity: a shared linear separator exists

18 October 2026. The agreed CPU oracle finds **one shared affine classifier that
correctly separates all 2,048 fixed query labels from the completed balanced
fit**. Its predictions are verified in float32 classifier arithmetic at cutoff
.5. No neural training, new model checkpoint or inference policy is introduced.
The subsequently agreed [frozen-confidence control](four_crop_frozen_confidence_ready.md)
is reviewed separately: its retained 4,000-update fit
[completed but failed the strict gate](four_crop_frozen_confidence_results.md).

This answers the [proposed capacity question](query_assignment_contract_review.md):
the existing final linear function class can represent the desired assignment
membership on these frozen four-crop inputs. It does not establish that the
ordinary loss/optimizer will find that boundary, that joint training will keep
it, or that its predictions generalize.

## Result

| Training crop | Fixed matched positives | Oracle accepted matched | Oracle accepted unmatched | Float32 label errors |
|---|---:|---:|---:|---:|
| 0/0 | 6 | 6 | 0 | 0 |
| 0/1 | 12 | 12 | 0 | 0 |
| 92/4 | 16 | 16 | 0 | 0 |
| 86/3 | 418 | 418 | 0 | 0 |
| Total | 452 | 452 | 0 | 0 |

All 1,596 fixed unmatched queries are rejected. The minimum signed logit margin
in actual float32 verification is **.999975**. The witness is rescaled by the LP
to require unit margin, so this number is not a measure of microscopy robustness
or a calibrated probability. No threshold search or reference-count selection
is used at inference; the table describes a known-label mathematical witness.

The mathematical solve completes in 2.01 seconds, with one LP call and a
predeclared 25-second call limit. Witness parameters exist only temporarily in
memory and are not saved as a classifier or installed into the model.

## Correct features and integrity checks

The source is the balanced fit's final checkpoint, SHA256
`62892ad332259dab76896ff16bb71294d87166c7ad274e6813d12ab9dcc1e58a`.
Saved Transformer decoder features are replayed through that checkpoint's
existing linear/ReLU hidden layer. These **post-ReLU 128-dimensional vectors**
are the inputs to its final classifier. The saved raw decoder features are not
substituted for those inputs.

Replaying the existing final head reproduces the saved logits to at most
9.54e-7 absolute difference, with identical .5 acceptance decisions on every
query of every crop. CPU replay tolerance is 1e-5 absolute/relative to account
for saved GPU versus CPU arithmetic. Only the two head linear operations/ReLU
are evaluated; no image, field, PointNeXt or Transformer forward occurs.

Fixed labels come from the saved final Hungarian assignments, not from the
oracle scores. Source/report/cache/checkpoint/prediction hashes are checked
before and after. The reconstructed feature tensor is 2,048 by 128, with all
128 columns varying. Geometry, reference labels and original weights remain
unchanged. There is no neural backward pass, optimizer update, classifier
checkpoint write, GPU work or development/test access.

## Numerical method and validation

The [audit script](../../scripts/pointneuron2/audit_four_crop_confidence_capacity.py)
centers/scales nonconstant feature columns in float64 and adds an intercept.
It removes only exactly constant columns, retaining even small varying signals.
The LP requests signed margins at least one, minimizing standardized L1 witness
size using positive/negative coefficient parts. A strict separator may be
rescaled to satisfy that constraint; conditioning is a numerical coordinate
change, not a new model feature.

The verified float64 minimum margin is .999999999998. After mapping the witness
back into the original features, temporary float32 confidence rows are evaluated
inside the existing six-output final-head shape, separately on all four crops.
All margins are finite/positive and every original binary label is correct.
The other four output rows are copied without alteration; no model is modified
or serialized.

The raw witness weight L2 norm is 70.43 and absolute bias 23.16. These values
depend on the arbitrary unit-margin scale and are not minimum necessary weight
norms or evidence that weight decay caused the learned classifier's failures.
LP parameters are training-label witnesses, not a deployment alternative.

Eight [known-case tests](../../tests/test_four_crop_confidence_capacity.py) pass:
separable points, a tiny nonconstant informative feature, conflicting identical
features, XOR, timeout handling, failed/nonfinite float32 witnesses, and the
post-hidden-ReLU feature boundary. Numerical infeasibility requires an approximate
dual certificate; timeouts or failed float32 checks remain unresolved. No such
infeasibility conclusion is needed for the real data: its primal witness passes.

## Research interpretation

Do not add a larger confidence head or replace the encoder on the assumption
that this frozen representation cannot express the four-crop labels. That
specific capacity explanation is contradicted by the verified witness.

The unresolved issue is the learning process: objective/regularization tradeoffs,
finite optimization and co-adaptation of features, geometry and assignments.
This audit does not isolate one of those causes. In particular, it does not
prove that assignment changes caused the failed hard-negative run, or that
freezing assignments would solve ordinary training.

Perfect assignment membership would also **not** complete reconstruction. The
balanced geometry still has 14 dense assignments outside the trust tolerance
and 76/77 trusted junctions. The oracle accepts those assigned but poorly placed
queries because those are its fixed positive labels. It is not a geometry-quality
classifier, learned connectivity result, test evaluation or matched PN 1.0 gain.

Keep the balanced fit as the stronger incomplete reference; keep the hard-negative
variant rejected. The subsequent fixed-feature training control is reviewed
above; it fails its declared strict gate. Neither classifier is promoted into
reconstruction, and no extension or new fit is prepared.

Evidence: [full oracle result](four_crop_confidence_capacity_results_seed0/summary.json)
and [test/source verification](four_crop_confidence_capacity_results_seed0/validation.json).
Raw outputs remain under ignored
`tmp/pointneuron2/local_runs/four_crop_confidence_capacity_20261018`.
