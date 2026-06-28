# PointNeuron Reimplementation on Gold166

This repository contains a working reimplementation of the PointNeuron neuron
reconstruction pipeline on Gold166 / BigNeuron-style data.

The current PointNeuron1.0 baseline is:

```text
raw volume
-> foreground point cloud
-> guarded30 skeleton proposal checkpoint
-> adaptive geodesic graph initialization
-> SWC reconstruction
```

The detailed project report is written in LaTeX:

```text
docs/pointneuron1_report.tex
```

## Current Status

Promoted PointNeuron1.0 proposal checkpoint:

```text
tmp/checkpoints/proposal_conservative_guarded30_cw3_oreg005_nw3.pt
```

Promoted topology path:

```text
initializer: geodesic
graph selection: adaptive_connected_coverage_nms
min proposal score: 0.85
NMS mode: distance
NMS distance: 18
max graph nodes: 128
```

Not promoted:

```text
tmp/checkpoints/proposal_conservative_rawcenter_full60_cw3_oreg001_nw1.pt
tmp/checkpoints/connectivity_guarded30_eligible_50e.pt
```

The learned connectivity GAE was trained and evaluated, but it underperformed
the initialized geodesic graph. The early "beast" samples were also diagnosed
as a PointNeuron1.0 architectural limitation: the current foreground-geodesic
forced-tree graph stage fails on several of them even with oracle GT-derived
nodes.

## Repository Layout

```text
configs/              Local configuration templates.
data/                 Local dataset location. Gold166 is expected under data/gold166.
docs/                 Reference PDFs, notes, and the LaTeX project report.
scripts/              Command-line tools for data, training, evaluation, and visualization.
src/pointneuron/      Importable Python package.
tests/                Unit tests for selected model and graph behavior.
tmp/                  Generated artifacts, checkpoints, reports, and visualizations.
```

## Setup

Create a local config file from the example, then set local paths such as the
Vaa3D executable if needed:

```powershell
Copy-Item configs\local.example.json configs\local.json
py scripts\check_environment.py --config configs\local.json
```

Install the package and common Python dependencies:

```powershell
py -m pip install -e .
py -m pip install scipy
```

For CUDA training and inference, install a CUDA-compatible PyTorch build for
your machine.

## Data Preparation and Inspection

Build a Gold166 manifest:

```powershell
py scripts\build_gold166_manifest.py `
  --root data\gold166 `
  --output tmp\gold166_manifest.json
```

Inspect one sample and check volume/SWC alignment:

```powershell
py scripts\inspect_volume.py --sample-index 0 --decode

py scripts\check_sample_alignment.py `
  --sample-index 0 `
  --decode-volume
```

Build and visualize a foreground point cloud:

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

## Run the Current Baseline

Single sample:

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

Validated range example:

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

## Diagnostics

Run the oracle-node diagnosis for early hard samples:

```powershell
py scripts\diagnose_beast_oracle.py `
  --output-root tmp\beast_oracle_diagnosis
```

This distinguishes proposal failure from graph-stage architectural failure by
replacing learned proposal nodes with GT-derived oracle nodes and reusing the
same foreground-geodesic graph initializer.

## Connectivity Baseline Reproduction

Build eligible connectivity records:

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

The expected conclusion is that this GAE checkpoint underperforms the geodesic
initializer and should not be used as the promoted reconstruction path.

## Detailed Report

The full report for a reader familiar with the PointNeuron paper is:

```text
docs/PointNeuron1_Report.pdf
```

The report explains the replication process, experimental decisions, failures,
metrics, promoted baseline, rejected checkpoints, and recommended PointNeuron2.0
direction.
