# Completed 2048/8192 node-budget study

3 November 2026. Both fresh models completed 20 epochs / 9200 updates and all
terminal evaluations. User-observed combined wall time: 20h22m50s. Successful
execution is separate from accuracy: **both training and development screening
gates failed**. The larger budget did not repair fine node prediction under
this schedule. This is one initialization and a fixed training allowance, not
a convergence comparison or a claim that 2048 is universally sufficient.

Raw paired report:
`tmp/node_budget_fit_train_20261102_141941/report.json`.
Protocol and immutable bindings: [ready report](node_budget_fit_ready.md).

| Terminal mean node agreement within two voxels | 2048 | 8192 |
|---|---:|---:|
| Training reference-present precision | 7.65% | 7.44% |
| Training reference-present recall | 29.89% | 21.97% |
| Development full-grid precision | 3.19% | 3.00% |
| Development full-grid recall | 25.68% | 18.42% |
| Development full-grid F1 | 5.40% | 4.90% |
| Development all-query recall before cutoff | 27.65% | 19.81% |

Scores are means over neurons. Training and development precision use different
domains as labeled; do not compare those two precision rows directly. Unknown
boxes are not certified biological background. Full-grid precision describes
reference agreement, not certified foreground precision. No graph/SWC was scored.

At six voxels, training reference-present recall was 80.50% / 79.03%; development
recall was 73.84% / 74.83%. This is coarse localization with substantial excess
predictions, not accurate reconstruction. Threshold filtering cannot explain
most missing two-voxel proposals, because pre-cutoff recall is also low.

Average training loss continued decreasing:

| Epoch | 2048 | 8192 |
|---|---:|---:|
| 1 | 10.424 | 14.271 |
| 10 | 6.825 | 7.331 |
| 15 | 6.093 | 6.500 |
| 20 | 5.724 | 5.898 |

The successful four-box fit had 500 exposures per box; this study had 20. Shared
learning across boxes means these counts cannot establish a required epoch
count, but 20 was a compute allowance, not an established convergence duration.

The next implemented control is an [explicit candidate objective](consistent_node_objective.md).
It preserves the preparation/model/targets and historical results. Its CPU
checks establish mathematical/execution properties only. No subsequent GPU fit
has been launched and no GATv2 readiness is claimed.
