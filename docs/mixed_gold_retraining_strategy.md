# Mixed Gold Skeleton Retraining Strategy

## Diagnosis

The `0-49` Gold stress run showed a domain shift, not a generic end-to-end
failure. The current skeleton checkpoint works well on Janelia-style FlyLight
samples but loses coverage on several non-Janelia Gold domains. Connectivity
should remain frozen until skeleton proposal coverage improves across domains.

## Selected Training Subsets

Two clean aligned subsets are generated:

- `tmp/mixed_gold_source_subset_seed0.json`
  - 56 samples
  - balanced by source group
  - recommended first serious mixed retraining subset
- `tmp/mixed_gold_domain_subset_seed0.json`
  - 68 samples
  - balanced by broader domain family
  - useful if the source-balanced run is too sparse in some families

The source-balanced subset is preferred because it avoids one broad family
hiding several different imaging sources.

## Cache Build

Use topology-centered patches so each domain contributes endpoint and branch
examples instead of relying only on foreground-random crops.

```powershell
py scripts\build_training_cache.py `
  --root data\gold166 `
  --sample-list tmp\mixed_gold_source_subset_seed0.json `
  --output-dir tmp\mixed_gold_source_skeleton_cache `
  --threshold-fraction 0.2 `
  --max-points 512 `
  --patches-per-sample 32 `
  --patch-radius 96 `
  --center-strategy topology `
  --endpoint-fraction 0.35 `
  --branch-fraction 0.20 `
  --min-points 256 `
  --resume
```

## Split

```powershell
py scripts\build_split.py `
  --cache-manifest tmp\mixed_gold_source_skeleton_cache\cache_manifest.json `
  --output tmp\splits\mixed_gold_source_skeleton_seed0.json `
  --seed 0
```

## Train

```powershell
py scripts\train_proposal.py `
  --split-file tmp\splits\mixed_gold_source_skeleton_seed0.json `
  --split train `
  --val-split val `
  --epochs 100 `
  --batch-size 16 `
  --k 20 `
  --loss-mode paper `
  --augment `
  --checkpoint tmp\checkpoints\proposal_mixed_gold_source_aug_100e.pt `
  --device cuda
```

## First Evaluation

Do not run all Gold immediately. Evaluate a small mixed set first:

```powershell
$indices = 0,8,15,23,35,48,57,84,92,143
foreach ($i in $indices) {
  py scripts\run_end_to_end.py `
    --sample-index $i `
    --checkpoint tmp\checkpoints\proposal_mixed_gold_source_aug_100e.pt `
    --output-root tmp\e2e_mixed_gold_source_smoke `
    --device auto
}
```

Then summarize:

```powershell
py scripts\analyze_gold_domains.py `
  --root data\gold166 `
  --run-root tmp\e2e_mixed_gold_source_smoke `
  --output-json tmp\gold_domain_report_mixed_source_smoke.json `
  --output-csv tmp\gold_domain_report_mixed_source_smoke.csv
```

## Go/No-Go

Proceed to connectivity only if skeleton proposal coverage improves materially
outside Janelia. The first target is not perfection; it is removing the
catastrophic failures:

- fruitfly larvae should no longer be near `0.0` proposal coverage
- zebrafish/chick/frog should move clearly above the previous low-coverage band
- Janelia should remain strong, not collapse due mixed-domain training
