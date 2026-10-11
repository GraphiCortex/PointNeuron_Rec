# Clean baseline validation

**Historical input pipeline:** later investigation found decoder/range defects
affecting this baseline. "Clean" here described the fresh split/training run,
not verified image decoding. See [the repairs](root_cause_findings.md) and
[current progress/comparison](validation_graph_cutoff_results.md).

30-epoch training completed; checkpoint selected at epoch 29 by validation loss.

All requested validation neurons are retained below. PASS describes execution, not reconstruction quality. The eight test neurons remain reserved.

| Sample | Status | Topology F1 | Spatial F1 | Candidate coverage | Selected coverage | Bridges | Foreground cap met |
|---|---|---:|---:|---:|---:|---:|---|
| 19 | PASS | 0.425 | 0.616 | 0.997 | 0.478 | 1 | True |
| 21 | PASS | 0.541 | 0.819 | 0.916 | 0.661 | 1 | True |
| 35 | PASS | 0.316 | 0.095 | 0.147 | 0.023 | 61 | True |
| 44 | PASS | 0.150 | 0.003 | 0.034 | 0.025 | 54 | True |
| 79 | PASS | 0.332 | 0.211 | 0.596 | 0.085 | 84 | False |
| 94 | PASS | 0.173 | 0.028 | 0.133 | 0.032 | 89 | False |
| 135 | PASS | 0.947 | 0.849 | 0.822 | 0.443 | 0 | True |

Metrics and limitations:

- Internal validation on seven mixed-domain neurons; not external testing or a direct comparison to historical stable-domain means.
- Spatial F1: voxel distance threshold 6, resampling step 1; topology F1: GT-induced graph on predicted nodes.
- Selection coverage: distance threshold 8, GT segment sampling step 4, proposal score threshold inherited from graph (0.85).
- Candidate proposals in the selection audit are post-local-filtered and score-filtered; this is not raw network-output coverage.
- Bridge counts refer to fallback graph connections. Tree validity does not establish anatomical correctness.

Means and denominators:

- edge_f1: 0.412 over 7/7 neurons.
- spatial_f1: 0.374 over 7/7 neurons.
- proposal_segment_coverage: 0.521 over 7/7 neurons.
- selected_segment_coverage: 0.249 over 7/7 neurons.
- segment_coverage_drop: 0.271 over 7/7 neurons.
- bridges: 41.429 over 7/7 neurons.

## Interpretation and first intervention

The clean baseline completes but generalizes unevenly across this small mixed-domain validation set. These means cannot be directly compared with PN 1.0's earlier stable-domain means, and this experiment does not isolate the cause of any difference.

Selection removes substantial available segment coverage in samples 19, 21 and 79. Samples 35, 44 and 94 already have weak score-filtered candidate coverage. Samples 79 and 94 also fail the graph foreground cap; their errors cannot be attributed solely to an encoder or edge scorer.

Static XY/XZ projections of samples 19 and 79 were inspected. Sample 19 retains much of the annotated geometry with extra trajectories; sample 79 has many missing annotated branches and long reconstructed connections away from the reference. Nonzero-intensity voxel samples are displayed, not a verified segmentation. Bounds and these projections do not establish full image/annotation agreement.

First compare selection at fixed proposal pools and matched node budgets, then ablate 64/128/256-node budgets separately. Report retained segment/terminal coverage, selected precision, graph topology and bridge dependence. Investigate foreground-cap and annotation-support evidence in parallel on the weak proposal cases before attributing their failures uniquely. Do not tune on the reserved test neurons.
