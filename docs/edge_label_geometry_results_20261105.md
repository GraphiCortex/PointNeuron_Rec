# Training edge-label geometry, length and units audit — completed

**Follow-up:** user REJECTED DraftB and HELD DraftA. The
[fixed compact/resampled audit](resampled_compact_results_20261105.md) is complete;
new Stage2 supervision remains a [draft only](stage2_candidate_geometry_contract_DRAFT_20261105.md).
The alternatives below remain historical diagnostic evidence,not adopted targets.

CPU audit completed in167.63 seconds (2m48s),guard exit0,under3600-second cap. All460 training packets,33 neurons; no Stage2 features/model work, GPU, training or development labels. Four new fixtures passed.460 overlap-array packets and920 draft diagnostic SWCs verified; aggregate replay exact,original label hashes unchanged. Temporary sleep-prevention request released.

## Main result: pre-training revision trigger fires

179,171/180,269 nonconsecutive negative edges (99.39%) have positive in-box length and overlap>=.5 at2 native voxels,far above the user's fixed30% trigger. Every training neuron has96.42–99.90% on-trace negatives by that definition.177,309/180,269 (98.36%) have overlap>=.99. Their length-weighted overlap is98.70%; edge-mean overlap99.27%.43 negatives have no in-box length,reported separately.

The old labels encode consecutiveness in one candidate discretization,not simply geometric disagreement. A near-geometric-oracle F1 did not establish alignment of those training targets. The earlier conclusion that the sanity check supported the whole supervision contract was too strong. The historical oracle gate pass remains unchanged; it is insufficient to resolve this newly measured mismatch.

| Negative-pair category | Count | Share of all negatives | Length-weighted overlap at2 |
|---|---:|---:|---:|
| Same-chain skip,intervening positive within1 voxel | 96,363 | 53.46% | 99.49% |
| Same-chain other,no qualifying close same-chain witness | 80,197 | 44.49% | 99.24% |
| Cross-reference-chain pairs | 3,709 | 2.06% | 72.56% |

A skip witness must lie within1 of the original segment interior AND project strictly between its endpoints on a shared reference chain.100,695 negatives have a geometric positive witness from ANY branch;1,026 cross-chain pairs also have such a witness. These overlapping facts are not forced into a false skip/cross-branch dichotomy. Cross-chain labels do not independently prove biological wrong joins.

## Positive edges and overlap distributions

Of163,536 positive edges,444 (0.2715%) have in-box overlap<.5;24 have zero in-box length (undefined overlap,separate from the444). Positive length-weighted overlap is99.68%; edge-mean overlap99.69%. There is a small positive-label geometric mismatch as well as the much larger negative-target mismatch.

| In-box overlap bin | Nonconsecutive negatives | Positives |
|---|---:|---:|
| [0.0,0.25) | 466 | 383 |
| [0.25,0.5) | 589 | 61 |
| [0.5,0.75) | 927 | 153 |
| [0.75,0.9) | 830 | 104 |
| [0.9,0.99) | 105 | 15 |
| [0.99,1.0] | 177,309 | 162,796 |

All quantiles,per-neuron histograms,and raw per-edge index/overlap/length/witness arrays are retained in the generated report/packets. The native integration is EXACTLY Stage0's clipped .5-voxel midpoint quadrature,not a new score.

## Per-neuron native results

Weighted overlap = supported in-box arclength / in-box arclength. On-trace share uses overlap>=.5 and ALL negative edges as denominator. Undefined zero-length edges are separately counted in JSON.

| Alias | Negative edges | Negative weighted overlap | Negative on-trace share | Same-chain skips <=1 | Cross-chain pairs | Positive edges | Positive weighted overlap | Positive <.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10,755 | 97.50% | 99.01% | 6,112 | 354 | 9,256 | 99.74% | 15 |
| 1 | 4,640 | 99.28% | 99.53% | 2,507 | 19 | 4,791 | 99.73% | 13 |
| 7 | 3,577 | 96.02% | 98.49% | 2,140 | 110 | 3,152 | 99.33% | 10 |
| 9 | 3,908 | 98.90% | 99.46% | 2,425 | 18 | 3,220 | 99.70% | 12 |
| 10 | 1,720 | 98.54% | 98.90% | 981 | 47 | 1,550 | 99.61% | 1 |
| 11 | 2,595 | 96.48% | 96.42% | 1,743 | 21 | 1,830 | 97.89% | 45 |
| 14 | 1,811 | 99.38% | 99.67% | 1,022 | 21 | 1,610 | 99.93% | 2 |
| 20 | 2,419 | 99.38% | 99.42% | 1,091 | 15 | 2,378 | 99.72% | 8 |
| 15 | 4,929 | 96.66% | 99.27% | 2,750 | 84 | 4,091 | 99.70% | 5 |
| 18 | 1,936 | 99.61% | 99.74% | 1,054 | 1 | 1,697 | 99.96% | 1 |
| 22 | 894 | 98.90% | 99.66% | 403 | 8 | 878 | 100.00% | 0 |
| 25 | 38,014 | 99.34% | 99.64% | 20,013 | 327 | 36,058 | 99.80% | 80 |
| 23 | 10,982 | 99.78% | 99.85% | 5,575 | 0 | 10,986 | 99.91% | 13 |
| 32 | 7,572 | 99.72% | 99.87% | 3,863 | 51 | 7,145 | 99.79% | 13 |
| 39 | 701 | 99.50% | 99.86% | 302 | 6 | 720 | 99.86% | 1 |
| 41 | 2,229 | 99.26% | 99.73% | 1,128 | 88 | 1,981 | 99.92% | 0 |
| 46 | 1,528 | 99.39% | 99.74% | 907 | 7 | 1,081 | 100.00% | 0 |
| 53 | 4,467 | 99.08% | 99.55% | 2,250 | 199 | 4,069 | 99.84% | 5 |
| 66 | 19,054 | 99.46% | 99.73% | 10,569 | 0 | 16,065 | 99.79% | 35 |
| 75 | 12,516 | 99.77% | 99.90% | 6,959 | 0 | 10,708 | 99.95% | 5 |
| 85 | 2,991 | 96.54% | 98.66% | 716 | 1,386 | 2,243 | 99.72% | 7 |
| 86 | 10,051 | 97.53% | 98.54% | 6,216 | 366 | 8,153 | 99.78% | 18 |
| 92 | 2,882 | 96.29% | 98.58% | 1,458 | 87 | 2,743 | 99.57% | 2 |
| 93 | 664 | 99.08% | 99.40% | 263 | 2 | 758 | 99.24% | 2 |
| 95 | 2,012 | 95.37% | 98.71% | 928 | 2 | 2,203 | 97.92% | 10 |
| 96 | 6,108 | 98.80% | 98.92% | 3,685 | 86 | 4,783 | 99.44% | 30 |
| 102 | 1,294 | 99.60% | 99.69% | 591 | 52 | 1,399 | 99.98% | 0 |
| 108 | 1,605 | 96.68% | 97.26% | 787 | 170 | 1,675 | 96.54% | 62 |
| 126 | 5,545 | 99.17% | 99.48% | 2,659 | 127 | 5,256 | 99.65% | 19 |
| 143 | 2,334 | 98.93% | 99.27% | 1,157 | 5 | 2,281 | 99.77% | 2 |
| 149 | 1,725 | 99.32% | 99.83% | 789 | 14 | 1,704 | 100.00% | 0 |
| 158 | 3,850 | 99.21% | 99.45% | 1,693 | 18 | 4,059 | 99.84% | 5 |
| 162 | 2,961 | 99.13% | 99.43% | 1,627 | 18 | 3,013 | 99.19% | 23 |

## Predicted / reference length: earlier results

The ledger contains11,340 metric entries across10 canonical completed reports plus current draft/physical results: [every per-box and aggregate ratio](../tmp/edge_label_geometry_audit_20261105/all_prior_length_ratios.csv). Each entry has source report,pointer,available case/tile/arm/policy,units and zero-reference handling. Nested aggregate entries/candidate-union entries are not independent experiments. Isolated-point-only results have no predicted segment length and no ratio is fabricated.

Ratios below pool clipped lengths across the declared boxes. They are annotation-relative per-box measurements,not whole-image assembled neuron lengths. The classical matched controls use the same151 reference-present development boxes; the older full-grid classical report also has unknown-region output and is separately retained in the ledger.

| Method | Cohort | Pooled length ratio | Neuron-macro length ratio | F1 at2 |
|---|---|---:|---:|---:|
| four-fit dense | four training boxes,2neurons | 2.4017 | 2.3259 | 81.49% |
| four-fit raw_input | four training boxes,2neurons | 20.9603 | 22.0921 | 13.33% |
| four-fit sato512 | four training boxes,2neurons | 3.2529 | 3.3065 | 31.36% |
| four-fit dense | 151 development boxes,7neurons | 2.8433 | 3.1414 | 23.20% |
| four-fit raw_input | 151 development boxes,7neurons | 30.1066 | 31.2326 | 12.00% |
| four-fit sato512 | 151 development boxes,7neurons | 6.2035 | 6.2370 | 38.49% |
| four-fit original | four training boxes,2neurons | 2.4017 | 2.3259 | 81.49% |
| four-fit supported_r2_min3 | four training boxes,2neurons | 1.1197 | 1.0530 | 76.43% |
| broad dense | training | 6.5035 | 7.9780 | 37.84% |
| broad dense | development | 6.2361 | 7.7429 | 36.04% |
| raw_input | matched151 development boxes | 30.1066 | 31.2326 | 12.00% |
| sato512 | matched151 development boxes | 6.2035 | 6.2370 | 38.49% |
| classical_raw | matched151 development boxes | 0.9192 | 1.2752 | 54.31% |
| classical_pruned | matched151 development boxes | 0.9175 | 1.2709 | 54.33% |
| Stage0 classical oracle | 151 development boxes | 2.3896 | 2.2327 | 74.00% |
| Stage0 sato oracle | 151 development boxes | 2.0291 | 2.0844 | 83.45% |
| Stage0 learned oracle | 151 development boxes | 2.1130 | 2.1494 | 85.33% |
| Stage0 union oracle | 151 development boxes | 2.3979 | 2.4453 | 90.42% |
| label_oracle | 460 training boxes | 1.5624 | 1.7544 | 90.82% |
| geometric_oracle | 460 training boxes | 2.2354 | 2.4601 | 92.38% |
| DRAFT mask_only | 460 training boxes | 1.5624 | 1.7544 | 90.82% |
| DRAFT geometric_known | 460 training boxes | 2.1363 | 2.3486 | 91.05% |

The four-fit original/dense rows reference the same fitted geometry and are intentionally not called separate replications. Earlier fit F1_2=81.49% accompanies2.40x reference length; broad-head development36.04% accompanies6.24x. Stage0 union oracle90.42% accompanies2.40x. These scores never certified a clean single tracing. Geometry P/R/F1 credits multiple nearby segments separately and does not directly penalize multiplicity. Annotation incompleteness can also affect length; ratios alone are not anatomical error certification.

## Conditional pre-training drafts: NOT APPLIED

Both options were defined before this audit and were run only as isolated training diagnostic oracles. The trigger is based on TRAINING label geometry,not development selection. [Full revised contract draft](edge_supervision_revision_DRAFT_20261105.md).

| Training oracle | F1 at2 | F1 at6 | Pooled length ratio |
|---|---:|---:|---:|
| Original consecutiveness labels | 90.82% | 96.68% | 1.5624 |
| Draft A:mask conflicting negatives | 90.82% | 96.68% | 1.5624 |
| Draft B:mask-preserving known-edge geometric labels | 91.05% | 96.78% | 2.1363 |

Draft A masks179,171 conflicts,including3,199 cross-chain pairs. Its binary decode is provably unchanged: old negatives and proposed unknowns are both ineligible. All460 fresh SWCs match the old oracle hashes exactly; scores are reused after this exact replay. Oracle F1 cannot predict its loss-masking effect.

Draft B keeps all original unknowns/node labels and changes only otherwise known positive-endpoint edges to binary geometric agreement. It is not the unrestricted soft Stage0 oracle. It produces342,242 hypothetical positive edges. The+0.23pp F1_2 comes with+36.73% predicted length. That is not a reconstruction-quality win or a reason to adopt it. A forest spanning supported nodes is not sufficient topology/multiplicity handling. No new label manifest or active decoder was written.

## Scoring spacing and micron-scale view

Candidate scripts use native XYZ unchanged,effective spacing(1,1,1),two/six NATIVE VOXELS and .5-voxel quadrature. They do not apply pixel-size sidecars. The general geometry_metrics function can scale coordinates,but it is not the path used by candidate scripts. A2-voxel ball maps to a physical ellipsoid with axis radii2*sx,2*sy,2*sz—not a fixed2um tolerance.

Reported sidecars exist for12/33 TRAINING neurons,224 boxes. Calibration and XYZ ordering are unverified; values/text/paths/SHA256 are preserved. Missing21 cases remain native without fabricated micron conversion. Both reference and prediction are transformed after native clipping. Diagnostic physical tolerances0.5/1/2um use midpoint step<=0.1um. Nothing is selected from this view.

| Same12-neuron subset:oracle | Native F1 at2vox | F1 at0.5um | F1 at1um | F1 at2um | Pooled physical length ratio |
|---|---:|---:|---:|---:|---:|
| label_oracle | 84.90% | 64.82% | 80.74% | 91.89% | 1.8683 |
| geometric_oracle | 87.19% | 66.50% | 81.42% | 92.20% | 2.4615 |

| Physical tolerance | Negative length-weighted overlap | Positive length-weighted overlap |
|---|---:|---:|
| 0.5um | 39.73% | 42.86% |
| 1.0um | 65.75% | 71.47% |
| 2.0um | 86.70% | 91.50% |

Pooled physical lengths weight different acquisitions by their reported lengths; neuron-macro oracle F1 weights each of the12 neurons equally. Physical and native means use different metrics/tolerances; no cross-unit superiority claim is made.

| Training alias | Reported spacing XYZ,um/voxel | Label-oracle F1 at0.5/1/2um | Label physical length ratio | Geometric physical length ratio |
|---|---|---|---:|---:|
| 0 | [0.103, 0.103, 0.3] | 91.62% / 95.57% / 98.49% | 1.6919 | 2.2149 |
| 1 | [0.156, 0.156, 0.3] | 75.39% / 85.26% / 91.84% | 1.0385 | 1.3176 |
| 7 | [0.156, 0.156, 0.3] | 76.95% / 88.25% / 95.15% | 0.8246 | 1.2628 |
| 23 | [0.3, 0.3, 0.5] | 76.17% / 98.26% / 99.62% | 2.5720 | 2.9988 |
| 66 | [0.35, 0.35, 1.0] | 54.04% / 86.50% / 96.98% | 2.7370 | 3.2916 |
| 75 | [0.35, 0.35, 1.0] | 52.52% / 86.65% / 96.57% | 2.9681 | 3.4999 |
| 85 | [2.0, 2.0, 2.01] | 9.57% / 26.00% / 63.27% | 1.4599 | 2.3382 |
| 86 | [0.62, 0.62, 2.0] | 14.35% / 44.79% / 82.58% | 1.8942 | 2.6862 |
| 92 | [0.21, 0.21, 0.25] | 86.56% / 94.44% / 98.88% | 1.1440 | 1.5978 |
| 93 | [0.21, 0.21, 0.25] | 73.07% / 83.03% / 90.53% | 0.8584 | 1.1268 |
| 95 | [0.21, 0.21, 0.25] | 69.57% / 80.35% / 88.77% | 0.6881 | 0.9551 |
| 96 | [0.07, 0.07, 0.25] | 98.04% / 99.81% / 100.00% | 2.3275 | 2.9484 |

## Status and interpretation

Original sources/candidates/labels/thresholds/gates remain intact. This is new pre-training evidence for discussion,not a retroactive change to earlier pass/fail records. No development labels or Stage2 feature/model work were created. Stage2 is held for supervision/decoding discussion; neither draft is adopted. Existing node positives indicate trace support,not unique representative selection. Training/CV and final quality cannot be inferred from oracle execution or overlap alone.

[Portable evidence](evidence/edge_label_geometry_completed_20261105.json); [generated JSON and per-edge arrays](../tmp/edge_label_geometry_audit_20261105/report.json); [per-neuron CSV](../tmp/edge_label_geometry_audit_20261105/per_neuron_native_overlap.csv); [execution/definitions](edge_label_geometry_audit_protocol_20261105.md).
