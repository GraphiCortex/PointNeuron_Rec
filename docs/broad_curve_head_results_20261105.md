# Broad same-head experiment — 5 November 2026

**FINAL: FAILED. Stop tuning this head.** The one authorized broad run completed
all9200 updates/20 epochs; training metrics were reported, never gated. Its
terminal development F1 is36.04%/67.54% at2/6 voxels, below matched classical
54.33%/69.14%, below the required59.33%/74.14%, with only2/7 and3/7 strict
neuron wins. No extension, new threshold, new head experiment or automatic
assembly/GATv2 work follows this failure. Earlier launch/status notes below
preserve execution history and are superseded by this final verdict.

## Terminal measurements

| Policy / domain | P2 | R2 | F1 at2 | P6 | R6 | F1 at6 |
|---|---:|---:|---:|---:|---:|---:|
| Broad model, TRAINING460 boxes/33 neurons | 26.05% | 78.92% | 37.84% | 62.47% | 97.50% | 73.28% |
| Broad model, DEVELOPMENT151 boxes/7 neurons | 24.80% | 74.40% | 36.04% | 55.86% | 93.03% | 67.54% |
| Matched classical pruned, DEVELOPMENT | 65.11% | 52.90% | 54.33% | 76.80% | 69.33% | 69.14% |
| Fixed Sato-top512 forest, DEVELOPMENT | 29.02% | 70.25% | 38.49% | 53.96% | 86.13% | 63.13% |

Macro F1 is averaged per neuron after pooling directed lengths within neuron.
It is not recomputed from macro P/R. The two tolerances use native voxels and
the same .5-voxel quadrature. Comparator pool and criteria were fixed before
training. Required F1 is evaluated as the full-precision comparator+.05; at2
this is .5933112579402622 and at6 .7413686642936105 in IEEE arithmetic.

| Development alias/context | Broad F1 at2 | Classical at2 | Broad F1 at6 | Classical at6 |
|---|---:|---:|---:|---:|
| 19, utokyo fly | 15.38% | 39.21% | 38.38% | 55.34% |
| 21, frog | 50.77% | 50.22% | 86.03% | 72.86% |
| 35, human Allen | 30.60% | 61.74% | 58.82% | 73.20% |
| 44, cultured cell | 34.02% | 74.82% | 64.42% | 85.41% |
| 79, mouse RGC | 53.09% | 61.93% | 87.78% | 83.35% |
| 94, adult zebrafish | 17.18% | 5.95% | 49.77% | 17.11% |
| 135, Janelia fly | 51.23% | 86.43% | 87.58% | 96.69% |

Both macro-gain tests fail, and both>=5/7-win tests fail. Wins at2 are frog and
adult zebrafish; at6 these plus mouse. The gate was not failed because of
training memorization metrics or exact-node sampling. It was a direct failure
to outperform the strongest fixed no-learning comparator under the agreed
downstream comparison.

## What broader training changed

Compared with the frozen four-box head, development F1 improves23.20->36.04%
at2 and56.99->67.54% at6. Recall improves33.28->74.40% at2 and62.69->93.03% at6.
The k6 candidate-union R2 improves38.07->81.69%, above Sato's76.80% but below
raw inputs'89.00%. Broader training recovered substantial geometric opportunity.
It did not resolve selection/localization precision: broad P2 is24.80%, and
its F1_2 remains below Sato's38.49%. At6 it beats Sato by4.41pp but still loses
to the stronger classical baseline and the fixed+5pp criterion.

Training and development F1_2 are similarly low (37.84/36.04), so the outcome
is not explained merely as another81%-fit/23%-transfer drop. Under this fixed
20-epoch budget the formulation has not demonstrated useful enough precise
selection/localization across the broad cohort. One run cannot isolate the
causal contributions of supervision, features, optimization or annotation
incompleteness, and is not an asymptotic-capacity proof. The practical decision
nevertheless remains STOP under the user's precommitted rule.

Development retains108092 nodes/107700 forest edges, no empty boxes. Total
box-local predicted length268665.36 versus reference43082.02 voxel units
(6.24x) accompanies high recall and low precision. Extra geometry includes
redundancy and off-trace structure; annotation incompleteness means it is not
all certified biological false foreground.

## Connectivity and final-reconstruction limits

Conservative development shortcut flags15/23886 scorable edges. Another81372
edges have an off-reference endpoint (75.55% of forest edges),1978 are
projection-ambiguous,267 connect distinct reference components (unknown),197
have reference paths leaving the box (unknown). A tiny conditional flag rate
does not establish safe topology. Training flags429/70262 are also retained;
they are not a gate or evidence of biological correctness. No edge was selected
or deleted using reference scoring.

These are per-box reference-present results. The264 unknown development boxes
were excluded, not labeled background. Development was repeatedly inspected,
not an untouched final test. Whole-image assembly, fresh learned GATv2,
biological rooting/radii and topology remain unqualified/unbuilt. The assembly/
graph-stage pass condition was not triggered. The dense bypass and non-learned
forest are not promoted as a product pipeline.

## Completed execution and reproducible evidence

- Actual optimizer work6379.46s (~1h46m), all9200 clocks for64 Adam states.
- GPU stage including cache/loading and terminal inference6845.66s (~1h54m),
  successful guard. All611 predictions exported from checkpoint9200 only,
  with model/RNG unchanged during terminal inference.
- Terminal CPU graph/scoring959.19s (~16min), successful guard. Combined GPU
  and terminal CPU stages ~2h10m, below the measured4h34m total allowance.
- All460 training boxes have exactly20 logged exposures in the frozen schedule.
  Resident input tensors remained unchanged. Both16-update timing trajectories
  were discarded; the real first16 match their exact loss/gradient prefix.
- Verified all frozen science/execution configurations,460 label packet hashes,
  611 prediction hashes,611 graph/SWC pairs,611 score records, terminal checkpoint
  marker/hash, aggregates and the independently recomputed terminal gate.
- No extra model, training, filtering or geometric audit was launched after
  failure. No professor message has been sent.

Full-precision [terminal evidence](evidence/broad_curve_terminal_20261105.json)
preserves the report and receipts. Artifacts remain under
`tmp/broad_curve_head_20261105`: train/checkpoint_step009200.pt (SHA256
`bc13efbaef2ab012b344e81b87dd4a445d9f85b4fddfb421ccf4f6b76c30cab9`), terminal
predictions, evaluation/report.json/md,611 graph/SWC packets and per-edge
shortcut records. Checkpoints/labels/scorers remain useful evidence, not a
new adopted model. Source was pushed before training as98ca8df.

The [professor-facing discussion note](professor_candidate_refinement_discussion_20261105.md)
states the alternative framing and architecture question without adopting it.

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
