# Broad same-head experiment — 5 November 2026

User authorization: reject sixteen boxes; CPU diagnostics first; assistant
launches one broad run with measured budget. See
[science/decision protocol](broad_curve_head_protocol_20261105.md) and
[execution addendum](broad_curve_execution_addendum_20261105.md).

## Completed CPU diagnostics (no policy selected)

Output `tmp/broad_curve_diagnostics_20261105`, successful guard 546.89 seconds.
Same 151 reference-present development boxes/seven neurons; fixed k=6 forest,
native voxels, exact .5-voxel quadrature/2&6 decisions. Full evidence in
[portable CPU report](evidence/broad_curve_cpu_diagnostics_20261105.json).

| Policy | P2 | R2 | F1 at 2 | P6 | R6 | F1 at 6 |
|---|---:|---:|---:|---:|---:|---:|
| Classical pruned, SAME151 | 65.11% | 52.90% | 54.33% | 76.80% | 69.33% | 69.14% |
| Sato-top-512, unchanged sources | 29.02% | 70.25% | 38.49% | 53.96% | 86.13% | 63.13% |
| Original frozen dense/.25/r1 | 18.59% | 33.28% | 23.20% | 54.41% | 62.69% | 56.99% |
| All raw predicted votes, no cutoff/suppression | 5.77% | 61.92% | 10.25% | 19.93% | 97.82% | 31.56% |
| Learned-support top512, source coordinates unchanged | 17.61% | 49.70% | 24.82% | 41.89% | 68.02% | 49.91% |
| EXACT Sato source set moved by learned offsets | 18.89% | 44.93% | 24.47% | 51.67% | 84.38% | 60.51% |

The original classical 53.87%/68.52% was FULL-grid F1, counting unknown-box
prediction length. Cached per-box scores use the same geometry/scorer/crops;
reaggregation to the matched151 gives 54.3311257940%/69.1368664294%. No classical
extraction, image filtering or completed metric audit was repeated.

Removing cutoff/suppression restores some coverage but adds enough extra
geometry to lower F1. Learned support ranking without offsets loses to Sato.
Moving an identical Sato-selected set with learned offsets reduces two-voxel
F1 by 14.02pp and recall by 25.32pp. Thus both outputs of this TWO-neuron fit
show transfer problems; the cutoff alone does not explain them. The offset
contrast includes the consequent candidate/forest geometry changes, not a
direct measurement of supervised vector error. No arm was adopted; broad
training keeps the original recipe exactly.

## Training preparation and measured resources

All460 reference-present training boxes/33 neurons use the original2048-point
packets. Dataset's frozen20 epochs contain9200 updates. Reused16 prior training
labels and generated only444 with the same full-curve projection/ambiguity
code; successful CPU preparation guard122.33s. No sampler/filter restart.

`tmp/broad_curve_head_20261105/prepare/manifest.json` and complete marker bind
every label packet, cohort and schedule. A test rejects a modified schedule or
development substitution, even if someone rewrites the preparation marker.

First streamed GPU timing:16 throwaway updates, p90=.9376s/update, estimated
training8626s, terminal reserve2400s, 50% margin =>16560s. This exceeded the
assistant's initial14400s resource ceiling and stopped before real training.
It is a resource preflight result, not an accuracy/learnability failure.
`timing/report.json` is preserved unchanged.

A separate resident-cache timing now checks the identical16-step loss/gradient
prefix and input immutability, measuring all460 cached batches for VRAM. Only
execution planning changes; no labels, model, losses, optimizer, schedule,
updates, inference rule or pass criterion changes. No timing weights are used.
The measured wall plan has a disclosed six-hour upper bound under the user's
instruction to set the budget from throughput. Actual timing and run status
are recorded below.

Resident-cache timing completed successfully: all16 loss/gradient records
EXACTLY match the streamed trace, and the entire460-batch cache is unchanged.
Peak allocated VRAM170,519,040 bytes; conservative update .929151s, projected
training8548.19s (~2h22m), terminal reserve2400s. The measured overall budget
with50% margin is16440s (4h34m), below the disclosed21600s ceiling. Residency
did not recover the older ~.4s throughput; no unverified causal claim about
that discrepancy is made. Both timing models are discarded.

**Authorized broad pipeline launched by the assistant.** Fresh original model
and Adam; all460 boxes/33 neurons, dataset20 epochs/9200 updates. GPU work writes
under `tmp/broad_curve_head_20261105/train`; guard/visible console under
`resident_train_guard`. A successful GPU stage automatically invokes the
CPU-only terminal evaluation, writing `evaluation/report.json` and report.md
under the same root, with `resident_score_guard`. There is no new training
after this pipeline. At this report update the broad verdict is PENDING;
do not infer development accuracy from loss/progress. Preserve all checkpoints,
RNG/cursor receipts and per-graph terminal records. No terminal checkpoint
selection or training-fit gate occurs.

## Fixed terminal decision

Only terminal checkpoint9200 is judged. Training metrics are reported, NEVER
gated. Strongest fixed no-learning macro comparator at both tolerances is the
matched pruned classical graph. PASS requires development F1 >=59.3311257940%
at2 AND >=74.1368664294% at6, plus strict per-neuron F1 wins on at least5/7 at
EACH tolerance. No 45% floor, point gate, training floor or oracle ratio.

If pass, whole-image assembly and the intended graph stage are next. If fail,
stop tuning this head. A different classical-candidate refinement framing is
discussion only and needs professor agreement for the Transformer's role.
No message to the professor has been sent. Per-box geometry cannot qualify
whole-image assembly, GATv2, topology, rooting or radii.
