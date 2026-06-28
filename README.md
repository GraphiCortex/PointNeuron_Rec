# PointNeuron Reimplementation on Gold166

This repository is a working reimplementation and experimental extension of the
PointNeuron reconstruction pipeline on the Gold166 / BigNeuron-style neuron
data. It documents the path from raw 3D neuron volumes and SWC labels to a
PointNeuron-style reconstruction pipeline:

```text
raw volume
-> thresholded point cloud / proposal aggregation
-> skeleton node proposal model
-> geodesic graph initialization
-> SWC reconstruction
```

The project started as a replication of the PointNeuron paper pipeline, then
iteratively tested where that pipeline works, where it fails, and which parts
should be considered PointNeuron1.0 versus future PointNeuron2.0 architecture
work.

The current conclusion is:

```text
PointNeuron1.0 stable baseline:
  guarded30 proposal checkpoint
  + adaptive_connected_coverage_nms geodesic graph
  + SWC generation

Not promoted:
  the full60 proposal checkpoint
  the first connectivity GAE checkpoint

Out of PointNeuron1.0 scope:
  early "beast" samples that break foreground-geodesic graphing even with oracle nodes
```

## Background

The original PointNeuron paper reconstructs neurons by converting voxel volumes
into point clouds, learning skeleton proposals, predicting graph connectivity,
and generating a final SWC tree. The paper's conceptual stages are:

```text
1. Voxel-to-point conversion
2. Point-cloud feature encoding
3. Skeleton proposal
4. Connectivity prediction
5. SWC reconstruction
```

This repository follows that structure, but the implementation evolved through
several practical discoveries:

1. The raw Gold166 data required careful sample scanning, SWC selection, volume
   decoding, and alignment checks.
2. The first proposal models could detect skeleton-like points, but offset drift
   and confidence calibration strongly affected the final graph.
3. A foreground-geodesic initializer was much stronger than a naive learned
   connectivity baseline in the stable domain.
4. Some very large or unusual early samples break the PointNeuron1.0 graph
   assumption itself. This was verified using oracle nodes from GT SWCs.

The final result is not a universal PointNeuron2.0 system. It is a documented
PointNeuron1.0 baseline with known scope and known limitations.

## Current Baseline

The promoted proposal checkpoint is:

```text
tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt
```

Its saved training arguments are:

```text
split_file: tmp\splits\mixed_gold_source_coverage_combined_filtered_min15_seed0.json
split: train
val_split: val
epochs: 30
batch_size: 2
k: 20
lr: 0.0001
weight_decay: 0.0001
proposal_coordinate_mode: raw
loss_mode: conservative
conservative_init: True
conservative_center_weight: 3.0
offset_regularization_weight: 0.005
non_worsen_weight: 3.0
augment: True
amp: True
best_epoch: 27
best_val_loss: 5.215460332957181
```

The promoted end-to-end graph path is:

```text
initializer: geodesic
graph selection: adaptive_connected_coverage_nms
min proposal score: 0.85
NMS mode: distance
NMS distance: 18
max graph nodes: 128
```

## What Was Replicated

### 1. Data Foundation

The repository scans Gold166-style data and selects one usable SWC per sample.
SWC priority is:

```text
1. *swc_sorted.swc
2. stamped SWCs
3. base SWCs
```

If a preferred SWC is structurally invalid, the scanner falls back to a lower
priority valid SWC. The code also checks that SWC coordinates fit inside the raw
volume bounds.

Important files:

```text
src/pointneuron/data/gold166.py
  Scans the dataset, pairs volumes with selected SWCs, and writes manifests.

src/pointneuron/data/swc.py
  Parses, validates, and writes SWC trees.

src/pointneuron/data/vaa3d_raw.py
  Reads Vaa3D .v3draw / .v3dpbd volume data.

src/pointneuron/data/alignment.py
  Checks whether SWC coordinates are aligned with volume dimensions.

scripts/build_gold166_manifest.py
  Builds a JSON manifest of usable Gold166 samples.

scripts/check_sample_alignment.py
  Verifies volume/SWC alignment sample by sample.
```

### 2. Voxel-to-Point Conversion

The first PointNeuron-style transformation is implemented as thresholded
foreground extraction:

```text
volume voxels with intensity > threshold
-> (x, y, z, intensity) point records
```

Important files:

```text
src/pointneuron/data/point_cloud.py
  Converts raw volumes into point clouds.

scripts/build_point_cloud.py
  Builds inspectable point-cloud CSVs.

scripts/visualize_sample.py
  Renders foreground points and SWC skeletons to HTML.
```

### 3. Training Cache and Splits

The project builds patch-level training records from foreground points and SWC
nodes. These records are used for PointNeuron-style DGCNN / proposal training.

Important files:

```text
src/pointneuron/data/training_cache.py
  Creates and stores patch-level training records.

src/pointneuron/data/torch_dataset.py
  Loads cached records as PyTorch datasets.

src/pointneuron/data/splits.py
  Builds deterministic train/validation/test splits.

scripts/build_training_cache.py
  Generates .npz training records.

scripts/build_split.py
  Creates deterministic split JSON files.

scripts/inspect_dataset.py
  Checks tensor shapes and loader behavior.
```

### 4. Encoder and Skeleton Proposal

The project implements a DGCNN/EdgeConv-style encoder and a proposal head for
objectness, center offsets, and radius prediction. This corresponds to the
PointNeuron skeleton proposal stage.

Important files:

```text
src/pointneuron/models/dgcnn.py
  DGCNN / EdgeConv encoder.

src/pointneuron/models/proposal.py
  Proposal head for objectness, centers, and radii.

src/pointneuron/models/proposal_loss.py
  Proposal losses, including the conservative proposal losses used later.

scripts/train_proposal.py
  Trains the proposal network.

scripts/audit_proposal_checkpoint.py
  Measures proposal quality and offset behavior.

scripts/aggregate_proposals.py
  Applies a trained proposal checkpoint over a whole volume.

scripts/visualize_proposals.py
  Visualizes predicted proposal centers against SWC labels.
```

### 5. Graph Initialization and SWC Reconstruction

The strongest PointNeuron1.0 graph path is a foreground-geodesic initializer.
It selects proposal nodes, builds foreground paths through the image volume, and
creates a graph that can be converted to SWC.

Important files:

```text
src/pointneuron/graph/initialization.py
  Shared geometric graph construction utilities.

scripts/initialize_geodesic_graph.py
  Builds proposal-node graphs using foreground geodesic distances.

scripts/generate_swc_from_graph.py
  Converts graph edges and paths into an SWC reconstruction.

scripts/visualize_reconstruction.py
  Renders GT, proposals, graph nodes, and reconstructed SWC to HTML.

scripts/run_end_to_end.py
  Runs proposal aggregation, geodesic graph initialization, SWC generation,
  and visualization for one or more samples.

scripts/evaluate_geodesic_baseline.py
  Compares initialized graph topology against GT-induced proposal connectivity.
```

### 6. Connectivity GAE Baseline

The repository also contains a PointNeuron-style connectivity graph
autoencoder. It can train on connectivity records, but the first clean
experiment showed that it does not beat the geodesic initializer.

Important files:

```text
src/pointneuron/models/connectivity.py
  Graph autoencoder used for the learned connectivity baseline.

scripts/build_connectivity_record.py
  Builds a training record from an initialized graph and a GT target graph.

scripts/build_eligible_connectivity_manifest.py
  Filters validated E2E graph outputs into graph-eligible connectivity records.

scripts/train_connectivity.py
  Trains the connectivity GAE.

scripts/evaluate_connectivity.py
  Evaluates trained connectivity checkpoints on connectivity records.

scripts/predict_connectivity.py
  Applies a trained connectivity checkpoint to produce predicted graph edges.
```

## Main Experimental History

### Initial Replication

The first milestone was to reproduce the PointNeuron data flow:

```text
Gold166 volume and SWC scan
-> point cloud construction
-> patch training cache
-> DGCNN encoder inspection
-> proposal training
-> proposal visualization
```

This established that the raw data could be decoded and that the model could
learn PointNeuron-style proposal outputs.

### Proposal Model Issues

The early proposal checkpoints had two major issues:

```text
1. Offset drift:
   A checkpoint could reduce validation loss while moving proposal centers too far.

2. Confidence inflation:
   More confident proposals were not always more useful for graph topology.
```

The failed full60 checkpoint is the clearest example:

```text
tmp/checkpoints/proposal_conservative_rawcenter_full60_cw3_oreg001_nw1.pt
```

It improved validation loss to approximately:

```text
val_loss: 5.0645
```

but caused worse bridge and reachability behavior in E2E graph tests:

```text
old hard probe:
  F1        0.3937
  bridges   15.25
  reachable 0.8878

full60 hard probe:
  F1        0.4232
  bridges   40.25
  reachable 0.6831
```

Conclusion: validation loss alone was not a reliable promotion criterion.
End-to-end topology had to be tested.

### Guarded30 Proposal Checkpoint

The guarded30 checkpoint was trained to be more conservative:

```text
tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt
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

Guarded30 did not make proposal metrics dramatically better, but it gave the
graph stage better end-to-end behavior than full60.

### Stable-Domain E2E Results

The main validated stable range was samples 100-162.

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

Combined stable-domain result:

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

Conclusion: guarded30 is promoted for PointNeuron1.0 stable-domain proposals.

## Early Beast Samples and Architectural Boundary

When guarded30 was tested on early-domain samples 0-24, the result was mixed:

```text
guarded30 000-024:
  samples   22
  F1        0.4709
  bridges   26.9091
  reachable 0.8049
```

The worst samples were:

```text
7, 16, 22, 23, 24
```

Score-NMS on those same reused proposals was worse:

```text
score_nms early worst:
  F1        0.3075
  bridges   115.4
  reachable 0.1134
```

This showed that the adaptive selector was not the primary problem. It was
actually protecting the graph stage.

To separate proposal failure from graph-stage failure, the project added:

```text
scripts/diagnose_beast_oracle.py
```

This script builds oracle proposal nodes directly from GT SWCs and then runs the
same foreground-geodesic initializer. If the graph stage worked with good nodes,
then the beast failure would be proposal-side. It did not.

Oracle diagnosis result:

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

Conclusion:

```text
PointNeuron1.0 has an architectural weakness on these beast samples.
Even GT-derived nodes do not make the current foreground-geodesic forced-MST
stage reliable.
```

The failure mode is:

```text
saturated / huge / anisotropic volume
-> foreground threshold adaptation becomes extreme
-> eligible geodesic pairs collapse
-> MST still forces one tree
-> fake long bridge edges are created
```

These samples should not be used as clean PointNeuron1.0 connectivity training
targets. They are better treated as PointNeuron2.0 architecture work.

## Connectivity GAE Baseline

After promoting guarded30 for proposals, the project built an eligible
connectivity dataset from graph outputs that passed topology checks.

Eligibility command:

```powershell
py scripts\build_eligible_connectivity_manifest.py `
  --summary tmp\e2e_guarded30_adaptive_v4_000_024\summary.json `
  --summary tmp\e2e_guarded30_adaptive_v4_100_109\summary.json `
  --summary tmp\e2e_guarded30_adaptive_v4_110_129\summary.json `
  --summary tmp\e2e_guarded30_adaptive_v4_130_162\summary.json `
  --output-root tmp\connectivity_guarded30_eligible `
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

Connectivity training command:

```powershell
py scripts\train_connectivity.py `
  --record-manifest tmp\connectivity_guarded30_eligible\eligible_connectivity_manifest.json `
  --epochs 50 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --normalize-node-features `
  --init-decoder-bias `
  --checkpoint tmp\checkpoints\connectivity_guarded30_eligible_50e.pt `
  --device cuda
```

Training result:

```text
records: 46
best_epoch: 46
best_loss: 0.377377
checkpoint: tmp/checkpoints/connectivity_guarded30_eligible_50e.pt
```

Evaluation command:

```powershell
$records = (Get-ChildItem tmp\connectivity_guarded30_eligible\records\*_connectivity.npz).FullName

py scripts\evaluate_connectivity.py `
  --records $records `
  --checkpoint tmp\checkpoints\connectivity_guarded30_eligible_50e.pt `
  --csv-output tmp\connectivity_guarded30_eligible\connectivity_eval_50e.csv `
  --device cuda
```

Evaluation result:

```text
threshold_precision mean 0.3702
threshold_recall    mean 0.9490
threshold_f1        mean 0.4983

topk_f1             mean 0.5200
init_topk_f1        mean 0.8261
```

Conclusion:

```text
The GAE learned a high-recall, low-precision edge signal.
It does not beat the initialized geodesic graph.
It is not promoted as a PointNeuron1.0 connectivity stage.
```

## Repository Layout

```text
configs/
  Local configuration templates.

data/
  Local dataset location. Gold166 data is expected under data/gold166.

docs/
  Reference PDFs and strategy notes.

scripts/
  Command-line entry points for data inspection, training, evaluation, graph
  initialization, visualization, and diagnostics.

src/pointneuron/
  Importable Python package.

tests/
  Unit tests for key graph/proposal behaviors.

tmp/
  Generated artifacts: manifests, splits, training caches, checkpoints,
  E2E outputs, HTML visualizations, and reports. This is working output, not
  source code.
```

## Important Scripts

Data and inspection:

```text
scripts/check_environment.py
scripts/build_gold166_manifest.py
scripts/inspect_volume.py
scripts/check_sample_alignment.py
scripts/build_point_cloud.py
scripts/visualize_sample.py
```

Training cache and splits:

```text
scripts/build_training_cache.py
scripts/build_split.py
scripts/inspect_dataset.py
scripts/inspect_encoder.py
```

Proposal model:

```text
scripts/train_proposal.py
scripts/audit_proposal_checkpoint.py
scripts/aggregate_proposals.py
scripts/visualize_proposals.py
```

Graph and reconstruction:

```text
scripts/initialize_geodesic_graph.py
scripts/initialize_image_supported_graph.py
scripts/generate_swc_from_graph.py
scripts/visualize_reconstruction.py
scripts/run_end_to_end.py
scripts/evaluate_geodesic_baseline.py
```

Diagnostics:

```text
scripts/audit_selection_stage.py
scripts/probe_selection_oracle.py
scripts/diagnose_geodesic_edges.py
scripts/diagnose_beast_oracle.py
```

Connectivity:

```text
scripts/build_connectivity_record.py
scripts/build_eligible_connectivity_manifest.py
scripts/train_connectivity.py
scripts/evaluate_connectivity.py
scripts/predict_connectivity.py
```

## How to Use This Repository

### 1. Check the Environment

Create `configs/local.json` from the example file and set the local Vaa3D path
if needed.

```powershell
py scripts\check_environment.py --config configs\local.json
```

Install the package and common dependencies in your Python environment:

```powershell
py -m pip install -e .
py -m pip install scipy
```

For training and CUDA runs, install a compatible PyTorch build separately.

### 2. Build or Inspect the Gold166 Manifest

```powershell
py scripts\build_gold166_manifest.py `
  --root data\gold166 `
  --output tmp\gold166_manifest.json
```

Inspect one sample:

```powershell
py scripts\inspect_volume.py --sample-index 0 --decode

py scripts\check_sample_alignment.py `
  --sample-index 0 `
  --decode-volume
```

### 3. Build and Visualize a Point Cloud

```powershell
py scripts\build_point_cloud.py `
  --sample-index 0 `
  --threshold 0 `
  --max-points 4096 `
  --output tmp\sample0_points.csv

py scripts\visualize_sample.py `
  --sample-index 0 `
  --threshold 0 `
  --max-points 8192 `
  --output tmp\visualizations\sample0.html
```

### 4. Run the Current PointNeuron1.0 Baseline

For a single sample:

```powershell
py scripts\run_end_to_end.py `
  --sample-index 110 `
  --output-root tmp\e2e_guarded30_single `
  --checkpoint tmp\checkpoints\proposal_conservative_guarded30_cw3_oreg005_nw3.pt `
  --initializer geodesic `
  --graph-selection-mode adaptive_connected_coverage_nms `
  --graph-min-proposal-score 0.85 `
  --graph-nms-mode distance `
  --graph-nms-distance 18 `
  --graph-max-nodes 128 `
  --device cuda
```

For a validated range:

```powershell
py scripts\run_end_to_end.py `
  --sample-range 110-129 `
  --output-root tmp\e2e_guarded30_adaptive_v4_110_129 `
  --checkpoint tmp\checkpoints\proposal_conservative_guarded30_cw3_oreg005_nw3.pt `
  --initializer geodesic `
  --graph-selection-mode adaptive_connected_coverage_nms `
  --graph-min-proposal-score 0.85 `
  --graph-nms-mode distance `
  --graph-nms-distance 18 `
  --graph-max-nodes 128 `
  --device cuda
```

Evaluate topology:

```powershell
py scripts\evaluate_geodesic_baseline.py `
  --summary tmp\e2e_guarded30_adaptive_v4_110_129\summary.json `
  --csv-output tmp\e2e_guarded30_adaptive_v4_110_129\topology_report.csv `
  --json-output tmp\e2e_guarded30_adaptive_v4_110_129\topology_report.json
```

### 5. Run the Beast Oracle Diagnostic

This checks whether the graph stage works when nodes are replaced by GT-derived
oracle nodes.

```powershell
py scripts\diagnose_beast_oracle.py `
  --output-root tmp\beast_oracle_diagnosis
```

Interpretation:

```text
verdict: graph_stage_can_handle_good_nodes
  The main issue is proposal / selection / domain normalization.

verdict: graph_stage_or_foreground_geodesic_is_a_pointneuron1_weakness
  The current graph architecture itself is insufficient for these samples.
```

The observed result was the second verdict.

### 6. Build Eligible Connectivity Records

This is only for reproducing the failed connectivity baseline. It should not be
treated as the promoted reconstruction path.

```powershell
py scripts\build_eligible_connectivity_manifest.py `
  --summary tmp\e2e_guarded30_adaptive_v4_000_024\summary.json `
  --summary tmp\e2e_guarded30_adaptive_v4_100_109\summary.json `
  --summary tmp\e2e_guarded30_adaptive_v4_110_129\summary.json `
  --summary tmp\e2e_guarded30_adaptive_v4_130_162\summary.json `
  --output-root tmp\connectivity_guarded30_eligible `
  --include-score `
  --skip-existing
```

Train the GAE baseline:

```powershell
py scripts\train_connectivity.py `
  --record-manifest tmp\connectivity_guarded30_eligible\eligible_connectivity_manifest.json `
  --epochs 50 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --normalize-node-features `
  --init-decoder-bias `
  --checkpoint tmp\checkpoints\connectivity_guarded30_eligible_50e.pt `
  --device cuda
```

Evaluate it:

```powershell
$records = (Get-ChildItem tmp\connectivity_guarded30_eligible\records\*_connectivity.npz).FullName

py scripts\evaluate_connectivity.py `
  --records $records `
  --checkpoint tmp\checkpoints\connectivity_guarded30_eligible_50e.pt `
  --csv-output tmp\connectivity_guarded30_eligible\connectivity_eval_50e.csv `
  --device cuda
```

Expected conclusion:

```text
The GAE underperforms the geodesic initializer and is not promoted.
```

## Current Project Status

Promoted:

```text
tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt
adaptive_connected_coverage_nms geodesic graph path
```

Not promoted:

```text
tmp/checkpoints/proposal_conservative_rawcenter_full60_cw3_oreg001_nw1.pt
tmp/checkpoints/connectivity_guarded30_eligible_50e.pt
```

Main limitation:

```text
PointNeuron1.0 cannot reliably reconstruct the early beast samples because
foreground-geodesic forced-MST graphing fails even with oracle nodes.
```

Recommended next research direction:

```text
PointNeuron2.0 should replace the forced foreground-geodesic MST authority with
a learned candidate-edge classifier/reranker, where foreground geodesic evidence
is only one feature rather than the source of truth.
```

## Final Takeaway

This repository now contains a reproducible PointNeuron1.0 baseline:

```text
guarded30 proposal model
-> adaptive geodesic graph
-> SWC reconstruction
```

It also documents a clear boundary:

```text
stable validated domain: works well enough to promote
early beast domain: PointNeuron1.0 architectural weakness
connectivity GAE: trained and evaluated, but not strong enough to replace the graph initializer
```
