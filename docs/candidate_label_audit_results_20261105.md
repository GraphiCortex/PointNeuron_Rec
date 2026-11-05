# Saved-label oracle and held-out edge-ranking controls — completed

User-requested CPU diagnostic completed on all460 training packets/33 neurons in941.63 seconds (15m42s), guard exit0, under3600-second cap. No old labels, candidates, constants or decoder changed. No model/selector training, GPU work, new development labels or development rescoring. Four new diagnostic fixtures passed.

All460 records,920 diagnostic-only SWCs and460 fixed-feature packets were hash-verified; aggregate replay was exact. Original Stage0/Stage1 source/input bindings remain intact. AC power was connected; the temporary system-awake request was released after completion.

## Label-oracle result and stopping rule

| Recipe and cohort | Macro F1 at2 | Macro F1 at6 |
|---|---:|---:|
| Saved-label oracle,460 training boxes | 90.82% | 96.68% |
| Geometric oracle on SAME460 training boxes | 92.38% | 97.21% |
| Preserved Stage0 geometric oracle,151 development boxes | 90.42% | 95.67% |

User gate PASSED:90.8218% exceeds the exact87.4232% floor (saved Stage0 minus3 percentage points). The reference cohorts differ, so the matched training comparison is more informative about the label contract:1.56pp lower F1 at2 and0.54pp at6. The prescribed gate is unchanged. No extra per-neuron gate was invented.

Label-oracle P2/R2=99.44/84.75%; geometric-oracle P2/R2=99.78/86.99%. The union of all label-positive edges before decoding has85.07% R2; the forest/pruning result has84.75%, a0.32pp reduction. Most of the modest gap is already present in permitted label-positive geometry rather than caused by forest/pruning. This does not causally separate ambiguity masks, consecutiveness, missing kNN links and binary-cost tie effects.

Largest matched F1_2 gaps are alias7:4.36pp,95:3.94pp,93:3.61pp,92:3.45pp. Weakest label-oracle neuron is95 at67.13%, followed by93 at71.07%,1 at71.15% and7 at71.64%. The macro pass does not erase local limitations.

Unknown labels are ineligible in this diagnostic, with no change to saved training masks. This binary-label forest is a supervision sanity check, not a strict mathematical ceiling over all graphs, learnability proof, correctly joined reconstruction or unseen-neuron result. Geometry can score well despite topological mistakes.

## Fixed one-feature controls: AP, no accuracy

| Fixed lower-is-better edge ranking | Neuron-macro average precision |
|---|---:|
| Preserved straight-segment Sato path cost | 16.92% |
| Maximum endpoint distance to nearest input point | 5.44% |
| Neuron-macro positive prevalence (constant-score AP) | 5.27% |

Primary AUPRC is tie-aware noninterpolated average precision; trapezoidal PR area is also saved. These rankings have no fitted parameters, selected sign, calibration or threshold: each neuron is scored separately as the held-out fold. This is a no-learning leave-neuron-out scoring control, not a newly trained CV classifier. The pooled known-edge prevalence is4.03%;5.27% above averages per-neuron prevalence, matching the AP macro aggregation. Do not mix those denominators.

The Sato cost is the preserved straight-segment scaffold cost, NOT a geodesic shortest-path feature. Actual Stage2 geodesic features remain unbuilt. Nearest-source means nearest of the same box's2048 image input points; edge feature is the maximum endpoint distance. The same known held-out edge labels are used for both controls. Their purpose is ranking comparison; no probabilities/decoding rule are fitted here to manufacture control reconstruction scores. Future trained selectors require BOTH held-out AP and decoded segment F1.

## Fragmentation audit

| Alias | Reference components | Overall edge mask | Mask among two-positive-endpoint edges | Cross-component pairs masked / observed |
|---|---:|---:|---:|---:|
| 23 | 51 | 2.21% | 15.04% | 314 /314 |
| 66 | 200 | 5.52% | 16.59% | 1,883 /1,883 |
| 75 | 80 | 4.14% | 15.30% | 743 /743 |

Alias23 does not have an above-pooled overall masking rate. Nevertheless, all2,940 cross-component candidate edges with two unambiguous positive endpoints were unknown:ZERO silently negative or positive labels. Across the full cohort,68,145/411,958 two-positive-endpoint edges are masked (16.54%), versus2.62% of all edges. Numerous edges touching known-negative endpoints dilute the overall rate; their negative labels follow the agreed precedence and do not certify a fragment relation. Unknown/coincident-node ownership stays masked. This validates the implemented labeling rule, not the biological completeness of fragmented references.

## Named boxes without positives

| Training neuron identity | Alias | No positive node labels | No positive edge labels |
|---|---:|---|---|
| e_checked6_chick_uw/DONE_09-2902-04R-01C-60x_merge_c1 | 0 | 37 | 37, 46 |
| e_checked6_chick_uw/DONE_10-2900-control-cell-05.oif-C0 | 1 | none | 22 |
| e_checked6_chick_uw/DONE_fix-P7-4.5h-cell2-60x-zoom1.5_merge_c2 | 7 | 11 | 11 |
| p_checked6_fruitfly_larvae_gmu/done_1_CL-I_X_OREGON_R_ddaD_membrane-GFP | 23 | none | 71, 76 |
| p_checked6_zebrafish_adult_RGC_UW/hard_hard_1_10_Ch2 | 95 | 8 | 8 |

Three node-empty and seven edge-empty boxes span FIVE neurons. They are reference-present boxes. In the three node-empty cases every candidate lies farther than2 voxels from the trace. This is missing local candidate opportunity; it does not assign background to the reference curve itself. Other edge-empty cases have only one positive node or coincident/insufficient positive-endpoint connections. Keep every box and the existing recipe unchanged.

## Per-neuron rates, label oracle and held-out AP

Positive rates below exclude unknown labels; masks use all instances. Counts pool per-box instances, not unique whole-neuron objects. Each AP column uses exactly the same known edge labels.

| Alias | Node positive | Node masked | Edge positive | Edge masked | Label F1_2 | Geometric F1_2 | Sato AP | Nearest-input AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.33% | 0.65% | 4.04% | 2.15% | 86.06% | 87.56% | 11.91% | 3.79% |
| 1 | 12.18% | 0.26% | 2.20% | 0.87% | 71.15% | 73.16% | 9.16% | 2.12% |
| 7 | 14.02% | 0.98% | 2.35% | 1.81% | 71.64% | 76.00% | 8.24% | 2.02% |
| 9 | 11.92% | 0.61% | 2.11% | 1.23% | 87.88% | 89.55% | 13.09% | 1.87% |
| 10 | 14.49% | 0.70% | 2.78% | 1.82% | 83.52% | 86.11% | 7.10% | 2.20% |
| 11 | 16.00% | 0.81% | 2.77% | 1.57% | 92.33% | 94.03% | 14.40% | 2.00% |
| 14 | 11.98% | 0.48% | 2.31% | 1.24% | 85.31% | 86.36% | 7.58% | 2.07% |
| 20 | 1.24% | 0.13% | 0.25% | 0.23% | 94.98% | 96.98% | 6.25% | 1.52% |
| 15 | 19.81% | 1.21% | 4.03% | 3.01% | 90.74% | 92.54% | 22.45% | 3.84% |
| 18 | 27.79% | 1.07% | 6.26% | 3.20% | 94.20% | 96.68% | 30.48% | 6.96% |
| 22 | 11.31% | 0.39% | 2.31% | 1.31% | 82.72% | 84.95% | 3.94% | 2.65% |
| 25 | 30.36% | 1.31% | 7.55% | 5.01% | 97.31% | 97.72% | 20.00% | 6.89% |
| 23 | 21.78% | 0.42% | 5.03% | 2.21% | 96.23% | 96.80% | 16.06% | 4.90% |
| 32 | 30.11% | 0.40% | 7.76% | 3.33% | 99.17% | 99.25% | 20.74% | 7.27% |
| 39 | 7.41% | 0.26% | 1.43% | 0.71% | 87.69% | 89.27% | 9.80% | 1.70% |
| 41 | 19.72% | 0.68% | 4.69% | 2.69% | 96.69% | 97.31% | 17.73% | 4.13% |
| 46 | 12.67% | 0.59% | 2.79% | 1.77% | 96.16% | 96.58% | 21.18% | 2.96% |
| 53 | 42.02% | 2.16% | 11.35% | 7.92% | 99.16% | 99.49% | 40.26% | 13.99% |
| 66 | 37.11% | 1.54% | 8.95% | 5.52% | 93.67% | 95.12% | 27.53% | 9.28% |
| 75 | 30.69% | 1.17% | 7.58% | 4.14% | 94.35% | 96.37% | 17.43% | 7.55% |
| 85 | 34.13% | 4.45% | 7.53% | 10.70% | 96.95% | 99.04% | 23.07% | 8.22% |
| 86 | 32.76% | 1.23% | 7.03% | 4.07% | 93.45% | 94.96% | 24.02% | 8.13% |
| 92 | 19.35% | 0.98% | 3.66% | 2.54% | 84.63% | 88.08% | 8.36% | 3.47% |
| 93 | 8.53% | 0.35% | 1.50% | 0.88% | 71.07% | 74.68% | 3.99% | 1.57% |
| 95 | 10.54% | 0.55% | 1.78% | 1.12% | 67.13% | 71.07% | 5.64% | 1.92% |
| 96 | 19.70% | 0.67% | 3.97% | 2.05% | 92.45% | 93.42% | 16.72% | 3.19% |
| 102 | 29.86% | 1.81% | 7.36% | 6.05% | 98.84% | 98.98% | 17.18% | 7.69% |
| 108 | 31.11% | 3.35% | 7.84% | 9.33% | 97.54% | 99.55% | 25.98% | 9.08% |
| 126 | 33.05% | 2.02% | 8.58% | 6.65% | 99.14% | 99.46% | 31.65% | 9.66% |
| 143 | 28.94% | 2.49% | 7.55% | 6.59% | 99.33% | 99.68% | 16.35% | 6.19% |
| 149 | 36.80% | 3.85% | 9.55% | 9.71% | 98.18% | 99.06% | 18.94% | 10.26% |
| 158 | 38.27% | 3.11% | 10.29% | 9.03% | 99.09% | 99.61% | 21.73% | 11.78% |
| 162 | 33.36% | 2.39% | 8.86% | 6.93% | 98.35% | 99.10% | 19.34% | 8.50% |

Full scores/counts/identities and checks: [portable evidence](evidence/candidate_label_audit_completed_20261105.json), [generated report](../tmp/candidate_label_audit_20261105/report.json). Executed command/definitions: [diagnostic protocol](candidate_label_audit_protocol_20261105.md).

## Encoder and next stage

CPU frozen encoder export also completed and verified:611 packets,390.42 seconds,32D source/query features,weights/RNG unchanged,zero updates,no CUDA. [Feature results](candidate_pointnext_results_20261105.md). Do not rerun it.

No selector training has run. Next preparation remains image-derived handcrafted features (including actual edge intensity/geodesic costs) and gradient-boosted trees before GATv2. Handcrafted-only versus handcrafted+PointNeXt comparison and CV-versus-development gap must expose upstream feature dependence. Selector CV is CONDITIONAL on encoder AND learned candidate positions trained on all33 neurons; not full-pipeline CV. Development is repeatedly inspected; one scoring per frozen model, no tuning afterward. Final classical+5pp floors/wins, GAT>=2pp over trees, wrong-join/break reporting and stop rules stay fixed. Professor discussion of the experimental Phase3 departure remains pending. No automatic training/extension follows this diagnostic pass.
