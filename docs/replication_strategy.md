# PointNeuron Replication Strategy

This project should proceed by validating the PointNeuron skeleton stage before
running graph connectivity experiments. Connectivity cannot recover missing
skeletal nodes, so APP2 and GAE work should remain blocked until skeleton
coverage is acceptable.

## Paper Constraints

The PointNeuron paper uses this order:

```text
thresholded point cloud
-> random point-cloud patches
-> DGCNN skeleton proposal module
-> coordinate offsets + objectness + radius
-> aggregation + 3D spherical IoU NMS
-> skeletal points
-> graph initialization
-> GAE connectivity refinement
-> SWC generation
```

Important implementation targets:

- Threshold fraction: `0.2`.
- Training crop size: `512` points.
- Training augmentation: rotation and flipping.
- Skeleton module first; train connectivity only after skeleton quality is high.
- Compact skeleton selection should first use predicted-radius spherical NMS, not
  distance heuristics.
- APP2 or another traditional tracer is only graph initialization, not a
  substitute for skeleton prediction.

## Current Finding

The existing proposal aggregation audit showed the skeleton stage is not ready:

```text
usable_for_connectivity: 0/8
mean coverage@8: about 0.29
mean terminal_coverage@8: about 0.15
```

Therefore connectivity training is premature.

## Milestone 1: Paper-Like Skeleton Subset

Select a controlled Janelia-Fly-like aligned subset. In this Gold166 copy, the
Janelia FlyLight volumes are typically around `511 x 511 x 400-600`, so the
dimension caps below are deliberately larger than the paper's reported average
image size.

```powershell
py scripts\select_paper_subset.py --root data\gold166 --include-regex "janelia" --max-width 512 --max-height 512 --max-depth 650 --max-samples 42 --output tmp\paper_subset_indices.json
```

For a stricter subset that resembles the paper's foreground scale, add
foreground counting:

```powershell
py scripts\select_paper_subset.py --root data\gold166 --include-regex "janelia" --max-width 512 --max-height 512 --max-depth 650 --count-foreground --min-foreground 1000 --max-foreground 60000 --max-samples 42 --output tmp\paper_subset_indices.json
```

Use the printed `--sample-index ...` arguments to build the cache.

## Milestone 2: Paper-Like Proposal Cache

Build 512-point patch records with threshold `0.2`. Start with foreground
center sampling because it is closer to random point-cloud cropping; compare
against SWC-centered random sampling if needed.

```powershell
py scripts\build_training_cache.py --root data\gold166 --output-dir tmp\paper_skeleton_cache --threshold-fraction 0.2 --max-points 512 --patches-per-sample 32 --patch-radius 96 --center-strategy foreground --min-points 256 --resume <PASTE_SAMPLE_INDEX_ARGS>
py scripts\build_split.py --cache-manifest tmp\paper_skeleton_cache\cache_manifest.json --output tmp\splits\paper_skeleton_seed0.json --seed 0
```

## Milestone 3: Skeleton Training

Train the proposal module with augmentation and paper loss weights:

```powershell
py scripts\train_proposal.py --split-file tmp\splits\paper_skeleton_seed0.json --split train --val-split val --epochs 100 --batch-size 16 --k 20 --loss-mode paper --augment --checkpoint tmp\checkpoints\proposal_paper_skeleton_aug.pt --device cuda
```

The paper trained much longer, but this milestone should first prove that
validation loss and full-sample coverage move in the right direction.

## Milestone 4: Full-Sample Skeleton Evaluation

Evaluate only skeleton quality. Use paper-style spherical NMS first:

```powershell
py scripts\aggregate_proposals.py --root data\gold166 --sample-index <INDEX> --checkpoint tmp\checkpoints\proposal_paper_skeleton_aug.pt --threshold-fraction 0.2 --max-points 512 --local-nms-mode sphere --global-nms-mode sphere --local-iou-threshold 0.1 --global-iou-threshold 0.1 --global-top-proposals 2048 --output tmp\paper_skeleton_eval\sample_<INDEX>_proposals.npz --html-output ""
```

Required go/no-go signal before connectivity:

```text
coverage@8 should approach or exceed 0.65
terminal_coverage@8 must materially exceed the current ~0.15
```

## Milestone 5: Connectivity Only After Skeleton Passes

After the skeleton stage passes:

1. Build oracle connectivity records using ground-truth SWC initialization.
2. Train and validate GAE.
3. Only then repair APP2/NeuTube/LCM initialization compatibility.
4. Generate SWC from predicted graph.

Until then, APP2 failures are a distraction from the main bottleneck.
