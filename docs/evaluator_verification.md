# Initial evaluator verification

**22 October update:** the initial record below remains historical. The complete
current synthetic suite passes **138 tests** in the isolated Phase 1 CPU runtime.
A new analytic branch fixture exposed a rounded KD-tree search boundary that
could omit even the nearest segment fragment. The search now includes a
coordinate-scaled roundoff margin and always retains the known nearest fragment.
This expands candidate search only; exact projections and scoring tolerances
are unchanged. No biological score was used to choose this correction.
PN1 rebuild recommendations at the end of the original record are superseded
by the user's scope clarification in the revised proposal.

21 October 2026. This verifies the fresh measurement implementation, not a
model or reconstruction improvement. No image, dataset SWC, reserved reference,
training or model inference was used in these checks.

## Implementation and review

The standalone `pn2.evaluation` implementation validates SWC forests, loads
explicit manifests and evaluates arclength agreement against continuous
segments. Segment projection distances use an exact conservative spatial
search; length fractions use midpoint quadrature. Per-case results, separate
unit-group summaries, failure counts, source hashes and paired differences
are exported to JSON, with an optional flat CSV.

Independent cross-review of geometry and input/accounting code found and fixed:

- Inconsistent tolerance-key formatting for custom high-precision/scientific
  values. Geometry, aggregation, pairing and CSV now use one shared formatter.
- An ambiguous convergence field. `primary_result_retained` means that the
  requested quadrature step stays primary; actual half-step deltas are reported
  separately and do not receive an automatic pass verdict.
- Lost isolated-node counts in the segmentless-output scoring path. Valid
  segmentless morphology is retained while its curve F1 remains zero.

## Checks and observed outputs

**54 synthetic tests passed** with the retained Python environment. Tests cover
analytic/anisotropic distances, translation, collinear subdivision invariance,
quadrature behavior, indexed distances versus exhaustive projection, malformed
and deep SWC forests, unit transforms, manifest restrictions, missing/invalid/
empty outputs, paired failure accounting, custom tolerances and CLI report
preservation. A crossing/wiring example explicitly demonstrates that perfect
curve geometry can coexist with different graph structure.

The test command is in [tests/README.md](../tests/README.md). Third-party pytest
plugin autoload was disabled because the retained environment contains an
unrelated napari/skimage ABI incompatibility; no dependency was changed.

A separate CLI execution used two synthetic ten-unit straight references at
tolerance one and maximum quadrature step 0.5:

| Synthetic output | Case A F1 | Case B F1 | Macro F1 |
|---|---:|---:|---:|
| Exact curves | 1.0 | 1.0 | 1.0 |
| Five-unit partial curve; second output absent | 0.75 | 0.0 | 0.375 |

For the partial curve, precision is 1 and recall is 0.6: one additional unit of
the reference lies within the declared tolerance. The absent output stays in
the denominator. JSON and CSV were written under ignored
`tmp/evaluator_smoke_20261021/`; this readable record is retained in source.
These values are analytic software checks, not model measurements.

## What remains open

- The seven-case development manifest loads and its listed paths exist. Image
  decoding, acquisition calibration, content provenance and reference quality
  are not certified by those metadata checks.
- Numerical convergence must be inspected on the real outputs before a
  decisive comparison. The reported maximum distance is over quadrature
  samples, not an exact Hausdorff distance.
- Endpoint/junction support and component counts are diagnostics. Branch
  correspondence, continuity, wrong merges, shortcuts and radius accuracy
  remain separate assessment work. Initial distance summaries are weighted
  mean and sampled maximum, not full distributions.
- Whole-volume evaluator runtime/memory and model resource instrumentation
  have not been measured. No fresh repaired-PN1 recipe, PN2 fit, or matched
  quality comparison has been completed.

Milestone A has started. See the [benchmark contract](benchmark_contract.md)
and [PN1 reference specification](baseline_reference.md) for the remaining
decisions before integration and training.
