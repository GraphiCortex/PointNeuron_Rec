# PointNeuron1.0 Guarded30 Baseline Report

Date: 2026-06-28

## Executive Summary

The current PointNeuron1.0 stable baseline is:

```text
guarded30 proposal checkpoint
-> adaptive_connected_coverage_nms geodesic graph
-> SWC generation
```

The guarded30 proposal checkpoint is promoted for the stable Gold166 domain:

```text
tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt
```

The learned connectivity GAE baseline is not promoted. It underperforms the initialized geodesic graph on the eligible connectivity records.

The early "beast" samples are not merely proposal failures. Oracle-node diagnosis showed that PointNeuron1.0's foreground-geodesic forced-MST graph stage breaks even with GT-derived nodes on several of them. These samples are therefore PointNeuron2.0 architecture work, not a tuning issue for the current baseline.

## Promoted Proposal Baseline

Checkpoint:

```text
tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt
```

Training configuration summary:

```text
conservative proposal training
raw coordinate mode
epochs: 30
center weight: 3
offset regularization weight: 0.005
non-worsen weight: 3
best epoch: 27
best validation loss: 5.2155
```

Proposal audit comparison:

```text
old rawcenter:
  proposal_hit 0.3897
  offset_mean  0.5422
  score_p90    0.8489

failed full60:
  proposal_hit 0.3980
  offset_mean  0.9701
  score_p90    0.9265

guarded30:
  proposal_hit 0.3910
  offset_mean  0.6258
  score_p90    0.8616
```

The full60 checkpoint improved validation loss but caused excessive offset drift and bridge collapse in E2E tests. It is not promoted.

## Stable-Domain E2E Results

Selection/graph configuration:

```text
initializer: geodesic
selection mode: adaptive_connected_coverage_nms
min proposal score: 0.85
NMS mode: distance
NMS distance: 18
max graph nodes: 128
```

Old adaptive-v4 baseline versus guarded30:

```text
100-109:
  old F1        0.7822
  old bridges   1.20
  old reachable 0.9659

  guarded30 F1        0.7868
  guarded30 bridges   1.00
  guarded30 reachable 0.9363

110-129:
  old F1        0.7545
  old bridges   1.05
  old reachable 0.9667

  guarded30 F1        0.7827
  guarded30 bridges   0.65
  guarded30 reachable 0.9798

130-162:
  old F1        0.8580
  old bridges   2.1515
  old reachable 0.9308

  guarded30 F1        0.8708
  guarded30 bridges   2.0606
  guarded30 reachable 0.9368
```

Combined stable range:

```text
old adaptive v4 100-162:
  F1        0.8131
  bridges   1.651
  reachable 0.9478

guarded30 adaptive v4 100-162:
  F1        approximately 0.8295
  bridges   approximately 1.444
  reachable approximately 0.9504
```

Conclusion: guarded30 is the promoted PointNeuron1.0 proposal checkpoint for the stable domain.

## Early Beast Samples

The early-domain sweep improved F1 but exposed topology collapse:

```text
guarded30 000-024:
  sample_count 22
  F1           0.4709
  bridges      26.9091
  reachable    0.8049
```

The worst samples were:

```text
7, 16, 22, 23, 24
```

Score-NMS on those same reused proposals made the failure worse:

```text
score_nms early worst:
  F1        0.3075
  bridges   115.4
  reachable 0.1134
```

This showed the adaptive selector was not the main problem. It was partially protecting the graph stage.

## Oracle Beast Diagnosis

Script:

```text
scripts/diagnose_beast_oracle.py
```

Command:

```powershell
py scripts/diagnose_beast_oracle.py `
  --output-root tmp/beast_oracle_diagnosis
```

This test used GT-derived oracle nodes, then ran the same foreground-geodesic graph initializer. Results:

```text
oracle beast samples:
  mean F1        0.5889
  mean bridges   76.6000
  mean reachable 0.3969
```

Per-sample highlights:

```text
sample 7:
  F1        0.2677
  bridges   124
  reachable 0.0236

sample 16:
  F1        0.5433
  bridges   104
  reachable 0.1811

sample 22:
  F1        0.8898
  bridges   0
  reachable 1.0000

sample 23:
  F1        0.6373
  bridges   73
  reachable 0.4252

sample 24:
  F1        0.6063
  bridges   82
  reachable 0.3543
```

Conclusion: most beast samples break the PointNeuron1.0 graph stage even with good nodes. The weakness is architectural:

```text
thresholded foreground geodesic support
+ forced single-tree MST fallback
= fake long bridge edges on saturated / anisotropic / misaligned volumes
```

These samples are PointNeuron2.0 architecture work.

## Connectivity GAE Baseline

Eligibility script:

```text
scripts/build_eligible_connectivity_manifest.py
```

Build command:

```powershell
py scripts/build_eligible_connectivity_manifest.py `
  --summary tmp/e2e_guarded30_adaptive_v4_000_024/summary.json `
  --summary tmp/e2e_guarded30_adaptive_v4_100_109/summary.json `
  --summary tmp/e2e_guarded30_adaptive_v4_110_129/summary.json `
  --summary tmp/e2e_guarded30_adaptive_v4_130_162/summary.json `
  --output-root tmp/connectivity_guarded30_eligible `
  --include-score `
  --skip-existing
```

Eligibility result:

```text
accepted records: 46
rejected records: 39

rejection reasons:
  foreground_cap_not_satisfied: 5
  low_edge_f1: 17
  low_reachable: 13
  too_few_nodes: 4
```

Training command:

```powershell
py scripts/train_connectivity.py `
  --record-manifest tmp/connectivity_guarded30_eligible/eligible_connectivity_manifest.json `
  --epochs 50 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --normalize-node-features `
  --init-decoder-bias `
  --checkpoint tmp/checkpoints/connectivity_guarded30_eligible_50e.pt `
  --device cuda
```

Training result:

```text
records: 46
best epoch: 46
best loss: 0.377377
checkpoint: tmp/checkpoints/connectivity_guarded30_eligible_50e.pt
```

Evaluation:

```text
threshold_precision mean 0.3702
threshold_recall    mean 0.9490
threshold_f1        mean 0.4983

topk_f1             mean 0.5200
init_topk_f1        mean 0.8261
```

Conclusion: the GAE learned a high-recall, low-precision edge signal, but it badly underperforms the initialized geodesic graph. It is not promoted.

## Current PointNeuron1.0 Status

Promoted:

```text
proposal checkpoint:
  tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt

topology path:
  adaptive_connected_coverage_nms geodesic graph
```

Not promoted:

```text
tmp/checkpoints/proposal_conservative_rawcenter_full60_cw3_oreg001_nw1.pt
tmp/checkpoints/connectivity_guarded30_eligible_50e.pt
```

Boundary:

```text
PointNeuron1.0 is suitable for the stable validated Gold166 subset.
PointNeuron1.0 is not architecture-complete for the early beast samples.
Connectivity GAE is not ready to replace the geodesic initializer.
```

## Recommended Next Work

Immediate:

```text
Use guarded30 + adaptive geodesic graph as the PointNeuron1.0 baseline.
Do not run more training for this phase.
Preserve eligibility and oracle diagnosis artifacts for reproducibility.
```

PointNeuron2.0 direction:

```text
Replace forced foreground-geodesic MST authority with learned candidate-edge prediction.
Use foreground geodesic as an optional feature, not the source of truth.
Train connectivity as an edge classifier/reranker over multi-scale candidates.
Add topology constraints after edge scoring rather than forcing a tree before evidence is reliable.
```
