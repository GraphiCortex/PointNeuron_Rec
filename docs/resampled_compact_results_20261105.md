# Resampled matching and compact-oracle audit — completed

User disposition: earlier DraftB REJECTED; earlier DraftA HELD. The CPU job completed in372.34 seconds (6m12s),guard exit0,under3600-second cap. Seven fixtures passed. All5,972 metric entries and their graph bindings verified; aggregate replay exact; all611 original oracle SWCs replay exactly before NMS. Original training labels unchanged. No new development label packets, Stage2 feature/label implementation,model execution,GPU or training. Scoped awake request released.

## Primary results

The original Stage0 gate PASSED at90.42% geometric F1_2 with2.39785x pooled reference length (approximately2.40x). The old pass remains on record; it did not certify compact or uniquely traced reconstruction.

| Method / cohort | Segment F1 at2 | Segment F1 at6 | Resampled F1 at2 | Resampled F1 at6 | Pooled length ratio | Neuron-macro length ratio |
|---|---:|---:|---:|---:|---:|---:|
| stage0_union/development | 90.42% | 95.67% | 41.49% | 46.19% | 2.3979 | 2.4453 |
| compact_union/development | 88.52% | 95.10% | 64.80% | 72.54% | 1.1847 | 1.1613 |
| classical_pruned/development | 54.33% | 69.14% | 47.43% | 54.72% | 0.9175 | 1.2709 |
| label_oracle/training | 90.82% | 96.68% | 54.40% | 61.17% | 1.5624 | 1.7544 |
| compact_label/training | 72.28% | 93.38% | 68.96% | 81.88% | 0.5132 | 0.5262 |

Fixed compact union F1_2=88.5171% clears the ORIGINAL>=80% gate:PROCEED branch. The65–80% candidate-recall branch and<65% stop branch are not triggered. No new resampled-F1/length threshold is invented. Passing this reference-assisted geometric headroom gate does not adopt NMS or a supervision contract. No selector is launched.

The compact union gives up1.91pp segment F1_2 while increasing resampled F1_2 by23.30pp and reducing pooled length from2.40x to1.18x. At6,the corresponding values are95.10% segment F1 and72.54% resampled F1. Remaining mismatches/excess structure are real limitations; neither score certifies correct joins.

The matched classical graph is the preserved image-only pruned graph on the SAME151 reference-present development boxes/seven neurons:segment F1=54.33/69.14%,resampled F1=47.43/54.72%,pooled length=.9175x (neuron-macro1.2709x). Reference-assisted oracle results cannot establish that learning beats this baseline. Development is repeatedly inspected.

## Fixed NMS and label interaction

Eligible-node NMS radius<=2,descending oracle probability,then classical>Sato>learned,then XYZ/index ties. Both node oracles are binary,so eligible probability ties are resolved by source/XYZ. No coordinates moved,edge remapping/rewiring,new kNN edges or relabeling. Original edge eligibility/probabilities,forest cost,spurs<2 and image roots unchanged. This is an induced-graph diagnostic,not a repaired graph.

| Compact oracle | Eligible before NMS | Kept before spanning | Edges between NMS nodes | Decoded edges | Isolated retained nodes | Forest components |
|---|---:|---:|---:|---:|---:|---:|
| Union,development | 40,393 | 17,276 | 24,506 | 15,199 | 492 | 2,077 |
| Original label,training | 118,681 | 54,143 | 69,402 | 23,860 | 18,336 | 30,283 |

For the frozen-label training oracle,segment F1_2 falls from90.82% to72.28%,while resampled F1_2 rises from54.40% to68.96%. Segment recall falls from84.75% to57.16%. Its .5132x length ratio is missing structure,not a compactness success. Many remaining nodes have no eligible label-positive link after suppression; surviving skip edges retain the original negative labels. Counting isolated node samples can give point agreement without reconstructing connecting segments,which is why both metrics are required.

Only the user-requested compact UNION gate is applied here. The original uncompact label-oracle sanity pass remains recorded. No new numeric gate is silently imposed on the compact label diagnostic. Its loss motivates representation-conditional supervision discussion,not an automatic label repair.

## Resampled metric definition

Native XYZ,spacing(1,1,1). Clip to original voxel-cell crop first. Resample maximal undirected degree-two chains every2 native voxels,keeping structural endpoints/junctions once by topological identity and including a short terminal remainder endpoint. Isolates count once. Pure cycles have deterministic XYZ/index phase. Connected exact-zero edges collapse only for sampling; coincident disconnected components stay separate. No metric-side geometric NMS erases duplicates.

At tolerance2 and6,maximum-cardinality bipartite matching yields precision=matched/predicted,recall=matched/reference,F1=2*matched/(predicted+reference). Count matches per box,pool counts within neuron,then macro across neurons. This penalizes duplicate paths/sample count,not anatomical wiring. Endpoints/short branches and graph topology affect sampling counts under the stated fixed convention. Crops are not merged,and no matching crosses their boundaries. Evaluation resamples remain in memory; no development training labels are created.

Length ratios pool clipped native lengths; neuron-macro ratios are separately reported. A ratio below1 can indicate missing branches. No physical spacing sidecar is applied to this voxel metric; the earlier conditional micron audit remains separate.

## All historical graph metrics

5,361 historical graph entries plus611 compact entries are covered. Exact-hash/crop matching caches avoid recomputing identical geometry. Coverage includes four Stage0 arms,training label/geometric oracles,historical held-A/rejected-B artifacts,four-fit forests/controls,development transfer,broad forests,all cached classical raw/pruned training/development graphs,and existing target-node+REFERENCE-edge representation oracles. No extraction/filtering or model inference was repeated.

Candidate-edge unions remain opportunity diagnostics,not reconstructed skeletons; their existing length ledger is retained. Isolated proposal-only experiments are not turned into invented graph decodes. Annotation-unknown grid regions are excluded from matched metrics,not relabeled as new negatives; old full-grid ratios remain in their historical ledger. The exact inventory makes this scope reviewable.

| Method / cohort | Segment F1 at2 | Segment F1 at6 | Resampled F1 at2 | Resampled F1 at6 | Pooled length ratio | Neuron-macro length ratio |
|---|---:|---:|---:|---:|---:|---:|
| broad_dense/development | 36.04% | 67.54% | 16.42% | 21.58% | 6.2361 | 7.7429 |
| broad_dense/training | 37.84% | 73.28% | 16.18% | 21.10% | 6.5035 | 7.9780 |
| classical_pruned/development | 54.33% | 69.14% | 47.43% | 54.72% | 0.9175 | 1.2709 |
| classical_pruned/training | 55.97% | 77.13% | 48.96% | 66.13% | 3.4320 | 4.8698 |
| classical_raw/development | 54.31% | 69.10% | 47.39% | 54.68% | 0.9192 | 1.2752 |
| classical_raw/training | 55.98% | 77.13% | 48.91% | 66.16% | 3.4696 | 4.9328 |
| compact_label/training | 72.28% | 93.38% | 68.96% | 81.88% | 0.5132 | 0.5262 |
| compact_union/development | 88.52% | 95.10% | 64.80% | 72.54% | 1.1847 | 1.1613 |
| fit4_controls_dense/training | 81.49% | 97.12% | 43.72% | 50.69% | 2.4017 | 2.3259 |
| fit4_controls_raw_input/training | 13.33% | 37.74% | 6.23% | 8.35% | 20.9603 | 22.0921 |
| fit4_controls_sato512/training | 31.36% | 45.73% | 18.81% | 23.20% | 3.2529 | 3.3065 |
| fit4_scaffold_original/training | 81.49% | 97.12% | 43.72% | 50.69% | 2.4017 | 2.3259 |
| fit4_scaffold_supported_r2_min3/training | 76.43% | 86.39% | 60.30% | 68.70% | 1.1197 | 1.0530 |
| fit4_transfer_dense/development | 23.20% | 56.99% | 13.12% | 26.83% | 2.8433 | 3.1414 |
| fit4_transfer_raw_input/development | 12.00% | 31.80% | 5.31% | 7.01% | 30.1066 | 31.2326 |
| fit4_transfer_sato512/development | 38.49% | 63.13% | 20.08% | 24.93% | 6.2035 | 6.2370 |
| geometric_oracle/training | 92.38% | 97.21% | 42.62% | 47.37% | 2.2354 | 2.4601 |
| historical_HELD_A/training | 90.82% | 96.68% | 54.40% | 61.17% | 1.5624 | 1.7544 |
| historical_REJECTED_B/training | 91.05% | 96.78% | 43.09% | 48.66% | 2.1363 | 2.3486 |
| label_oracle/training | 90.82% | 96.68% | 54.40% | 61.17% | 1.5624 | 1.7544 |
| reference_edge_target_oracle/development | 99.75% | 100.00% | 98.14% | 98.18% | 0.9740 | 0.9601 |
| reference_edge_target_oracle/training | 99.82% | 100.00% | 97.46% | 97.54% | 0.9635 | 0.9493 |
| stage0_classical/development | 74.00% | 80.66% | 35.37% | 40.84% | 2.3896 | 2.2327 |
| stage0_learned/development | 85.33% | 93.19% | 42.36% | 49.61% | 2.1130 | 2.1494 |
| stage0_sato/development | 83.45% | 90.16% | 43.93% | 50.64% | 2.0291 | 2.0844 |
| stage0_union/development | 90.42% | 95.67% | 41.49% | 46.19% | 2.3979 | 2.4453 |

Rows sharing identical artifacts are not independent replications. Rejected DraftB receives metrics only as a historical record; it is not a prospective contract. The reference-edge target oracle is explicitly reference-assisted,not an inference model.

## Development neuron detail

| Alias | Compact union segment F1_2 | Compact union resampled F1_2 | Compact union length ratio | Classical segment F1_2 | Classical resampled F1_2 | Classical length ratio |
|---|---:|---:|---:|---:|---:|---:|
| 19 | 95.90% | 70.40% | 1.2811 | 39.21% | 18.66% | 3.7435 |
| 21 | 91.88% | 65.44% | 1.2980 | 50.22% | 45.30% | 0.3306 |
| 35 | 88.71% | 64.80% | 1.1887 | 61.74% | 55.74% | 0.7780 |
| 44 | 92.50% | 67.19% | 1.2298 | 74.82% | 69.50% | 0.6963 |
| 79 | 91.77% | 64.55% | 1.2851 | 61.93% | 57.07% | 0.6552 |
| 94 | 64.20% | 53.85% | 0.5736 | 5.95% | 5.40% | 1.9507 |
| 135 | 94.67% | 67.34% | 1.2727 | 86.43% | 80.31% | 0.7421 |

Alias94 remains weakest:compact union64.20% segment F1_2 and.5736x length. The original gate is macro over7 neurons;this individual value is disclosed,not a new per-neuron veto.

## Stage2 draft and training-fold comparison boundary

[Draft candidate-geometry label contract](stage2_candidate_geometry_contract_DRAFT_20261105.md) proposes original dense-pool adjacency control C0 versus representative-conditional adjacency C1 on a COMMON image-only representative graph. Candidate-geometry features include all-candidate intervening count and local spacing; never reference-positive witness counts or SWC projections. Oracle ordering is unavailable at real inference. The proposed image-only representative policy is untested and unadopted.

The draft specifies33-neuron training-only leave-neuron-out paired decoded-F1/length-ratio comparisons,plus resampled F1 and handcrafted-only/+PointNeXt arms. The frozen upstream encoder/candidates saw all33 labels,so CV remains conditional selector-only. No fitted contract comparison is claimed:it requires the implementation/fitting the user asked to keep in draft. The existing33-neuron original/compact ORACLE comparison is saved separately and explicitly labeled NOT CV. DraftA stays held;DraftB stays rejected.

No Stage2 implementation,training,automatic extension or development selector scoring follows. Reserved neurons remain untouched;professor discussion of the experimental Transformer departure remains pending.

[Portable evidence](evidence/resampled_compact_completed_20261105.json); [all per-box metrics CSV](../tmp/resampled_compact_audit_20261105/per_box_metrics.csv); [33-neuron oracle comparison,NOT CV](../tmp/resampled_compact_audit_20261105/paired_training_neuron_oracles_NOT_CV.csv); [inventory](../tmp/resampled_compact_audit_20261105/inventory.json); [fixed execution protocol](resampled_compact_audit_protocol_20261105.md).
