# Query pilot fails selection; proposed centerline-field direction

**Subsequent scope clarification:** the field-to-skeleton route proposed below
was an alternative experiment, not adoption of a different full architecture.
The agreed intended system retains Phase 1 point-cloud preparation, PointNeXt,
Transformer proposals and GATv2. The field is now being assessed as a possible
Phase 1 guide, with no point-cloud interface implemented yet. This pilot used
DGCNN and does not test or disprove the full proposed PointNeXt-based system.
See [the current architecture and actual status](centerline_framework.md).

12 October 2026. The user completed the matched 30-epoch pilot in **six minutes**.
Both arms received 3,774 optimizer updates. Pairing checks pass; they establish
matched execution, not reconstruction improvement. The actual runtime supersedes
the conservative 15–25 minute planning estimate from the sparse timing sample.

## Fixed final-epoch result

Metrics average annotated crops within each of seven development neurons, then
average neurons equally. They use sparse SWC-derived targets at six-voxel
tolerance. No whole-volume/SWC evaluation was launched.

| Epoch-30 metric | Pointwise | Query |
| --- | ---: | ---: |
| All-slot target coverage | 82.8% | 80.8% |
| Coverage after fixed score 0.5 | 82.8% | **0%** |
| Annotated validation crops with no accepted slots | 0 / 47 | **47 / 47** |
| Mean target distance, voxels | 6.40 | 5.41 |
| Accepted sparse-target proximity | 78.5% | Undefined: no accepted slots |

The query arm accepts no validation slots at the fixed threshold at **any of
the 30 epochs**. Raw coverage grows from 20.6% at epoch one to 80.8% at epoch
30, while training loss falls. Its final predictions nevertheless cannot feed
the current reconstruction adapter. This candidate fails the agreed selection
check and is not promoted. The lower mean distance is partly driven by the
difficult zebrafish case; query coverage improves on three neurons and falls
on four, compared with the pointwise arm. It is not an overall quality win.

The pointwise arm accepts 508–512 slots per annotated validation crop at epoch
30. Sparse-target proximity counts duplicate slots separately and is not
reference-curve or SWC precision. It remains the stronger usable local control,
not proof that its full reconstruction meets the reviewed advancement gates.

## Bounded saved-weight confidence probe

Ordinary inference was inspected on six representative annotated training
crops and two annotated crops from each validation neuron. Inputs use the
declared deterministic selection seed without augmentation. No optimization,
threshold change or graph tracing occurred. The original checkpoint hash is
unchanged; source hashes and the saved run's pairing checks were verified.

All six inspected training crops also accept zero queries at 0.5. This includes
the first two training crops used in the successful fixed-crop experiment,
median target-count crops and the two largest target-count crops. Across these
six crops, mean scores are approximately 0.316–0.372 and maxima 0.392–0.419.
All 14 inspected validation crops similarly accept zero queries.

Geometry-only Hungarian assignment excludes scores from its cost. Scores of
assigned and unassigned slots overlap substantially on many inspected crops.
Assignment AUC varies, including a strong ranking on one dense training crop;
the classifier is not uniformly random, and this probe does not establish
that recalibration could never help. It does show that the failure occurs on
inspected training examples too, not solely on unseen neurons.

With positive class weight eight, a constant score minimizing one crop's
weighted classification loss is `8N / (512 + 7N)` for N matched targets. The
average such prior across the 211 annotated training crops is approximately
0.354. Observed scores near this value are consistent with a weakly
discriminative classifier, not proof of why learning stalled. Training loss
balance, assignment instability, representation and optimization remain
possible contributors. Another long run is not justified by this result alone.

## Proposed conceptual pivot

The user asked whether continued retries are justified and requested a new
concept for this phase. The recommendation is to freeze this unsuccessful
decoder implementation and test **image-native centerline-field learning**.
Retain corrected image decoding, coordinate handling, data partitions, useful
diagnostic infrastructure and PointNeuron's graph/SWC back end.

The proposed flow is:

```text
3D image -> learned centerline probability / proximity field
         -> image-supported skeleton points
         -> PointNeuron graph and SWC reconstruction
```

A compact 3D U-Net is the first baseline candidate. Supervision should derive
from continuous SWC segments, with annotation completeness and uncertain
regions handled explicitly; voxel masks are not supplied automatically by
the current point cache. A topology-aware objective can prioritize continuity
and thin branches. Start with one field-prediction head. Tangent, radius and
learned edge-affinity heads are subsequent hypotheses, not bundled changes.

The motivation is physical: neurites are continuous structures. In the current
set loss, only the winning matched slot is marked used; nearby duplicate slots
are unused. When target geometry and augmentation vary, many placements can
be approximately equivalent. This may destabilize useful/unused discrimination.
The pilot does **not** prove this explanation or rule out better query models.
The proposed field assigns spatially consistent supervision and retains image
context before point-cloud compression.

Published support for the components, not proof of success on this dataset:
[nnU-Net](https://www.nature.com/articles/s41592-020-01008-z) provides an established
biomedical segmentation baseline, and
[clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)
studies topology-aware tubular segmentation, including neuron datasets.

Bound the next effort: agree on one prototype and one matched validation
comparison, keep the graph fixed, and judge branch coverage, erroneous
connections and reconstruction quality. Freeze quantitative gates before
training. A failure should lead to retaining the working control and revising
the hypothesis, rather than repeated cutoff/weight tuning. No new architecture
or training run is implemented or launched in this report.

## Evidence

[Verified result summary](query_pilot_results_seed0/summary.json) contains the
30-epoch trajectories, per-neuron final results, hashes and representative score
probe. The [pilot protocol](query_pilot_ready.md) records the frozen comparison.

```text
Completed run: tmp/pointneuron2/local_runs/query_pilot_20261012_091456_4464/models/
Saved-weight probe: tmp/pointneuron2/local_runs/query_pilot_score_probe_20261012/probe.json
```
