# Branch-aware target canonicalization: implementation and CPU audit

21 October 2026. **Experimental target preparation implemented; no training
launched.** The strict preparation gate remains failed on a tiny normalized
precision decrease, explained below. The separate cache is explicitly marked
`training_ready: false`. No new GPU launcher or automatic promotion is included.

Subsequent preparation is recorded in the [matched-fit contract](canonical_joint_fit_ready.md).
That separate runner uses a prospective absolute-geometry eligibility policy;
the failed gate documented here is preserved.

This implements the agreed direction for future inputs: remove redundant
optional samples using branch identity, while preserving the intended
PointNeXt → Transformer → GATv2 architecture. Reference SWCs are used only for
training targets and evaluation. Future image inference receives no SWC.

## Policy and scope

The legacy generator, historical caches, models, losses, 512-query budget and
0.5 node/edge cutoffs remain unchanged. The new module is opt-in and versioned
`pointneuron2-canonical-curve-targets-v2`.

- Keep every root, leaf, junction and forced crop-boundary identity, including
  distinct structural nodes at identical coordinates. Never merge nearby branches.
- Retain the existing eight-voxel full-chain arc grid. An optional final arc
  sample less than one-quarter of that spacing (two coordinate units) from the
  same chain's endpoint is eligible for removal. This is an arclength rule,
  not a Euclidean proximity filter.
- Remove it only if replacing its two adjacent target edges with one introduces
  at most 0.25 units of curve deviation and radius-profile deviation. These are
  conservative engineering tolerances for this comparison, not established
  imaging resolution or biological accuracy limits.
- Make eligibility decisions on the full reference chain before cropping.
  Shared interior arc samples therefore have a consistent decision across crops.
  A forced crop boundary always wins if it coincides with an otherwise removable
  sample. Boundaries themselves depend on the crop and are not globally shared.
- Only contract a directed degree-two optional vertex. Verify structural degrees,
  branch incidence and component counts after contraction. Cropped splices are
  checked against a conservative 0.5-unit local deviation bound; violations stop
  preparation. Do not silently truncate targets or replace a difficult crop.

Curve bounds use segment geometry, not a sampled proximity score. The convex
capsule around the replacement chord bounds the intervening existing target
polyline. Radius deviation is measured in the source arclength parameter. This
controls the **additional representation change**, rather than claiming that
the original eight-voxel targets perfectly trace every source SWC bend or taper.
Distances retain the project's existing voxel-coordinate convention; physical
calibration for different acquisition spacings remains future work.

## Training-only audit

All 262 existing training crops from 33 neurons were checked. No development or
test images/SWCs were opened. Source SWCs, split, original count audit and four
fixed point-cloud caches are verified by SHA-256. New caches live in ignored tmp.

| Quantity | Result |
|---|---:|
| Original / canonical targets across cached training crops | 9,559 / 9,344 |
| Optional target instances removed | 215 |
| Crops with a removal | 85/262 |
| Empty, unknown-supervision crops retained as empty | 51 |
| Structural incidence or component changes | 0 |
| Newly missing required candidate edges | 0 |
| Crops over the unchanged 512-target capacity | 0 |

All structural role counts are preserved. The candidate comparison uses the
unchanged 16-neighbor union radius-eight rule. This is a reference-position
candidate feasibility check; it does not prove candidate recall on a future
model's 512 predicted proposals. Exact image-derived inputs, voxel indices,
reference scoring geometry and metadata are preserved in all four new caches.
Unknown annotation is not converted into negative/background supervision.

| Fixed training crop | Old targets | Canonical targets | Crop components | Crop junctions |
|---|---:|---:|---:|---:|
| 0/0 | 6 | 6 | 1 | 0 |
| 0/1 | 12 | 11 | 2 | 0 |
| 92/4 | 16 | 15 | 1 | 2 |
| 86/3 | 418 | 408 | 14 | 77 |

Both optional identities implicated in the dense-crop endpoint/sample reversals
are removed. The change also removes eight other eligible dense-crop samples;
it is not a rule targeting those two query IDs. Their measured additional curve
deviations are 0.033419 and 0.030850 units, with radius deviations 0.080589 and
0.035274. Their structural leaf identities are retained.

These counts describe **targets**, not predictions. This audit neither fixes
the old model's spur nor establishes improved learnability or reconstruction.

## Strict preparation failure and its interpretation

The preparation script prospectively required zero decline in the four crops'
continuous precision and recall within two voxels. Three crops retain 100%/100%.
For the dense crop:

| Metric | Original target graph | Canonical target graph |
|---|---:|---:|
| Recall within 2 | 99.7289493014% | 99.7289493014% |
| Precision within 2 | 99.9170170245% | 99.9170116663% |
| Total target chord length | 2,363.533839864 | 2,363.381236614 |
| Out-of-tolerance predicted chord length | 1.961330708421 | 1.961330708421 |

Precision declines by **0.0000053582 percentage points**. A separate CPU review
verifies that the offending prediction quadrature points **and weights are
exactly identical**, and the reference out-of-tolerance probe mask is identical.
The unchanged bad-length numerator is divided by a slightly smaller total
length. The maximum curve distances are unchanged as well.

This explains why fraction-only strict nonregression is overly sensitive to this
resampling operation. It does not retroactively change the criterion: the raw
gate and cache readiness remain false. The original four-crop model learnability
failure also remains unchanged; no prediction is rescored as a successful run.

For a future matched fit, I recommend declaring representation safety using
structural incidence, bounded added geometry/radius error, unchanged candidate
coverage, absolute missed/extra arclength, and the original 99% precision/recall
floor. Normalized fractions should still be reported. That prospective contract
needs review before preparing a GPU fit; no tolerance sweep or threshold tuning
has been performed. Keep the extra-spur/node-selection requirement separate.

## Implementation correction retained in the evidence

The first CPU prototype bounded **absolute** error against every source SWC
vertex. It removed 43 target instances but retained both problematic dense
samples: their long adjacent intervals already contain curvature and radius
variation omitted by the legacy targets. I corrected the guard to measure the
additional change to the existing target representation, using the same 0.25
numeric tolerances, and reran the training-only audit. It now removes 215.
The first output and its source snapshots are preserved under
`tmp/pointneuron2/local_runs/canonical_targets_20261021/`.
This was a preparation-code correction, not a comparison of trained models or
an adjustment to the model's acceptance thresholds.

## Verification and reproduction

**36 targeted tests pass** across the new canonicalizer and legacy representation
checks. They exercise coincident structures, close separate branches, short
structural chains, curved tails, radius changes, crop boundaries, overlap,
exit/re-entry, zero-length SWC edges, empty annotation and capacity overflow.
The real-data audit and independent residual review complete on CPU with zero
model forwards, zero optimizer updates and no CUDA initialization.

- [Canonical target module](../../src/pointneuron/data/canonical_curve_targets.py).
- [Explicit configuration](../../configs/pointneuron2/canonical_targets.json).
- [Preparation/audit script](../../scripts/pointneuron2/prepare_canonical_targets.py),
  [saved-evidence review](../../scripts/pointneuron2/review_canonical_targets.py),
  [tests](../../tests/test_canonical_curve_targets.py).
- [Compact verified evidence](canonical_targets_results_seed0/summary.json) and
  [initial absolute-guard audit summary](canonical_targets_results_seed0/initial_absolute_guard.json).

From this worktree, using the existing sampling environment:

```powershell
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u .\scripts\pointneuron2\prepare_canonical_targets.py
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u .\scripts\pointneuron2\review_canonical_targets.py
```

These document the completed actions; existing outputs refuse overwrite. To
repeat preparation, supply a fresh ignored `--output` directory. The reviewer
checks the default configured output. Raw summaries and the separate four-crop
cache are under `tmp/pointneuron2/local_runs/canonical_targets_incremental_20261021/`.
At this target-preparation checkpoint no new training launcher existed. The
subsequent matched-fit contract above supplies a separate launcher. The existing
stable-fit launcher belongs to its historical target contract and should not be
reused for this cache.
