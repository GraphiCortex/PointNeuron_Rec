# Discussion with professor: reconstruction from strong image-derived candidates

Prepared for the user to share; NOT sent and NOT an approved architecture change.
All scores below are per-box, reference-present development geometry from seven
neurons, with the same2/6-voxel scorer and within-neuron arclength pooling.

| System | F1 at2 | F1 at6 | k6 candidate-union recall at2 |
|---|---:|---:|---:|
| Raw input cloud + fixed image-cost forest | 12.00% | 31.80% | 89.00% |
| Sato-top512 + same forest | 38.49% | 63.13% | 76.80% |
| Preserved classical graph, matched151 boxes | 54.33% | 69.14% | not this kNN graph |
| Dense head fitted on4 boxes/2 neurons + forest | 23.20% | 56.99% | 38.07% |
| Same head trained on460 boxes/33 neurons,20 epochs | 36.04% | 67.54% | 81.69% |

The broad run recovered candidate coverage but still produced too much extra
geometry: at2 its precision/recall are24.80/74.40%. It failed the prospectively
fixed requirement of+5pp F1 above the strongest no-learning control at both
tolerances and wins on>=5/7 neurons at each (actual2/7 and3/7). Training accuracy
was reported, never gated. We stopped tuning the head; no budget extension or
threshold selection was used to rescue it.

The dense arm experimentally bypassed the original Transformer proposal head.
Neither that bypass nor the non-learned forest was adopted as the product.
Whole-image assembly, learned GATv2, anatomical roots/radii and reliable topology
remain unbuilt/unqualified. Candidate-union coverage is spatial opportunity,
not correct-connectivity recall; conservative shortcut flags are incomplete.
Development has been repeatedly inspected and is not a final untouched test.

**Question for agreement:** Is “learned refinement of strong image-derived
candidates” an acceptable framing of the PN2 research bet, instead of requiring
a learned head to independently generate the skeleton-node set?

In that framing, automatic image-derived points/classical skeletons supply
candidate positions; PointNeXt supplies learned image-point features. Learning
would retain/reject/refine candidates and GATv2 would score candidate connections,
followed by explicit decoding. A Transformer could provide contextual
refinement/confidence over seeded candidates, rather than independent fixed-query
set detection. Its exact role must be agreed; it must not be kept only as a name.

This changes the original proposal-generation claim. It retains the point-cloud
representation/feature-learning question and intended connectivity learning,
but comparative gains must be demonstrated in final geometry AND topology over
the strong classical baseline. Fixed candidates can still miss branches, and
pruning can destroy weak structure. No present result establishes that this
alternative will work. No implementation or new run is authorized by this note.
