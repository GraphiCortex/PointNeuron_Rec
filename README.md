# PointNeuron 2.0 development

This is the real PN 2.0 working folder inside the original PointNeuron directory.
It is a separate Git checkout on branch `pointneuron2`, sharing the original Git
history. Source files, configs, reports and local run artifacts are physically
stored here. The original parent checkout remains on `main`.

Start here:

- [Latest: joint fit stops at update 1,014 on moving edge supervision; automatic gates remain unmet](docs/pointneuron2/joint_saved_adam_fit_stop_results.md)
- [Current: joint fit stopped after 148 updates; Windows I/O repair and recovery from checkpoint eight](docs/pointneuron2/joint_saved_adam_fit_io_recovery.md)
- [Prepared: capped joint fit with saved proposal Adam state and mixed-state resume](docs/pointneuron2/joint_saved_adam_fit_ready.md)
- [Completed: GPU timing/reload passes; automatic reconstruction remains unsolved](docs/pointneuron2/joint_saved_adam_timing_results.md)
- [Prepared: GPU timing with saved proposal Adam state, per-step diagnostics and exact reload](docs/pointneuron2/joint_saved_adam_timing_ready.md)
- [Completed: eight-update joint CPU stability check passes with saved proposal Adam state](docs/pointneuron2/joint_saved_adam_stability_results.md)
- [Completed: saved optimizer state preserves crop-0 geometry in the four-update CPU restart comparison](docs/pointneuron2/proposal_optimizer_state_results.md)
- [Stopped: joint GPU timing; proposal-only CPU control also shows early geometry drift](docs/pointneuron2/joint_query_graph_timing_results.md)
- [Prepared: bounded joint proposal/GATv2 prototype with strict automatic reconstruction gates](docs/pointneuron2/joint_query_graph_ready.md)
- [Design review: node selection, graph responsibility and missing joint training](docs/pointneuron2/node_graph_design_decision.md)
- [Automatic-selection integration: conditional graph success does not survive realistic nodes](docs/pointneuron2/gat_automatic_selection_results.md)
- [Latest: per-crop GATv2/control, edge labels and forest decoder implemented; CPU checks pass without training](docs/pointneuron2/gatv2_connectivity_implementation.md)
- [Reviewed design: GATv2 inputs, edge supervision, scorer and forest decoder](docs/pointneuron2/gatv2_connectivity_design.md)
- [Latest: training-only node/graph/SWC representation check passes; learned connectivity remains open](docs/pointneuron2/representation_oracle_results.md)
- [Protocol: reference-only feasibility cases, criteria and CPU command](docs/pointneuron2/representation_oracle_protocol.md)
- [Latest: LR replay accepts 17/18 matches with no extras; one confidence miss remains](docs/pointneuron2/confidence_lr_results.md)
- [Completed protocol: matched confidence replay at LR .001](docs/pointneuron2/confidence_lr_ready.md)
- [Latest: one confidence crop passes; linear oracle verifies capacity on both crops](docs/pointneuron2/query_confidence_results.md)
- [Completed protocol: frozen-placement confidence diagnostic and visible commands](docs/pointneuron2/query_confidence_ready.md)
- [Latest: per-crop normalization is consistent; confidence carries query identities between crops](docs/pointneuron2/pointnext_cropnorm_results.md)
- [Latest: PointNeXt/query fit fails inference; saved-weight audit locates normalization and confidence issues](docs/pointneuron2/pointnext_query_results.md)
- [Ready: native PointNeXt/Transformer prototype; separate input preparation and GPU timing](docs/pointneuron2/pointnext_query_ready.md)
- [Design for review: PointNeXt per-point interface and two-crop Transformer learnability plan](docs/pointneuron2/pointnext_query_design.md)
- [Latest: half-image quota fixes dilution locally but loses dim support; not promoted](docs/pointneuron2/field_image_reservation_results.md)
- [Completed protocol: cached-field image-reservation allocation comparison](docs/pointneuron2/field_image_reservation_ready.md)
- [Latest: attention cleanup survives sampling, but dim branches and path gaps remain](docs/pointneuron2/field_attention_sampling_results.md)
- [Latest: skip attention cleans targeted support but loses some branches; fixed-budget comparison next](docs/pointneuron2/field_attention_results.md)
- [Approved protocol: fresh baseline versus two skip gates; training completed](docs/pointneuron2/field_attention_ready.md)
- [Latest: development sampling gains survive at equal counts; image-pool dilution and branch losses remain](docs/pointneuron2/field_pilot_sampling_results.md)
- [Latest: field train/development pilot completed; candidate support improves, finite-budget comparison next](docs/pointneuron2/field_pilot_results.md)
- [Completed protocol: approved 33-training / 7-development field pilot and four-control sampling comparison](docs/pointneuron2/field_pilot_ready.md)
- [Latest: candidate expansion helps; 50/50 allocation loses some branch support](docs/pointneuron2/field_sampling_results.md)
- [Completed protocol: isolated Phase 1 four-control input comparison](docs/pointneuron2/field_sampling_ready.md)
- [Approved design: field-guided point-selection interface and matched comparison](docs/pointneuron2/phase1_field_guided_sampling_design.md)
- [Latest: eight-crop field learnability passes; measured improvements and reproducibility caveat](docs/pointneuron2/centerline_duration_results.md)
- [Intended architecture: Phase 1 point cloud, PointNeXt, Transformer and GATv2; isolated field-guide experiment](docs/pointneuron2/centerline_framework.md)
- [Completed protocol: 1,600-update duration-only field check](docs/pointneuron2/centerline_duration_ready.md)
- [Latest: ideal-input oracle confirms good crowded-case geometry; learned field remains the bottleneck](docs/pointneuron2/centerpath_oracle_results.md)
- [Latest: simple center-path accuracy improves; crowded geometry remains unresolved](docs/pointneuron2/field_centerpaths_results.md)
- [Latest: negative-loss control improves all crops; remaining width and placement errors](docs/pointneuron2/centerline_loss_control_results.md)
- [Completed protocol: matched centerline negative-loss control and visible GPU command](docs/pointneuron2/centerline_loss_control_ready.md)
- [Latest: field learns structure but fails centerline localization; diagnosis and proposed loss control](docs/pointneuron2/centerline_fit_results.md)
- [Completed protocol: centerline-field prototype, inspection images and visible GPU command](docs/pointneuron2/centerline_fit_ready.md)
- [Latest: completed query pilot fails selection; proposed centerline-field direction](docs/pointneuron2/query_pilot_results.md)
- [Completed protocol: matched 30-epoch train/validation pilot](docs/pointneuron2/query_pilot_ready.md)
- [Latest: pooled LayerNorm passes the two-crop learnability check](docs/pointneuron2/query_layernorm_results.md)
- [Ready to run: approved query-only pooled LayerNorm control](docs/pointneuron2/query_layernorm_ready.md)
- [Latest: 2,000-update fit learns confidence; pooled normalization limits inference](docs/pointneuron2/query_extended_fit_results.md)
- [Ready to run: approved 2,000-update fit with saved checks at 500/1,000/2,000](docs/pointneuron2/query_extended_fit_ready.md)
- [Latest: completed dropout-zero fit, improved geometry and remaining confidence gap](docs/pointneuron2/query_dropout_zero_results.md)
- [Latest: 500-update GPU fit, confirmed dropout dependence and approved control](docs/pointneuron2/query_gpu_fit_results.md)
- [Current: query decoder implementation, checked results and next GPU command](docs/pointneuron2/query_decoder_implementation.md)
- [Reviewed: query targets and matched architecture comparison](docs/pointneuron2/query_decoder_design.md)
- [Latest: allocation repair regression, off-reference paths and supervision findings](docs/pointneuron2/remaining_budget_results.md)
- [Completed: fixed-budget allocation repair versus saved neuron 79 SWC](docs/pointneuron2/remaining_budget_comparison.md)
- [Confirmed: actual neuron 79 component starvation and remaining losses](docs/pointneuron2/graph_component_trace_results.md)
- [Completed: unchanged graph-component allocation recording](docs/pointneuron2/graph_component_recording.md)
- [Latest: saved graph/path audit and component-allocation counterexamples](docs/pointneuron2/hybrid_graph_path_audit.md)
- [Measured: neuron 79 hybrid SWC improvement and remaining limits](docs/pointneuron2/hybrid_swc_results.md)
- [Completed: neuron 79 score/hybrid SWC protocol](docs/pointneuron2/hybrid_swc_comparison.md)
- [Latest: fixed confidence/coverage hybrid and reconstruction decision](docs/pointneuron2/hybrid_global_comparison.md)
- [Latest: matched global spatial-selection control and its tradeoffs](docs/pointneuron2/global_coverage_comparison.md)
- [Measured: where full-volume branch support is lost](docs/pointneuron2/full_volume_selection_audit.md)
- [Latest: seven-neuron results, progress versus PN1 and conceptual bottlenecks](docs/pointneuron2/validation_graph_cutoff_results.md)
- [Completed: seven-neuron cutoff protocol and local command history](docs/pointneuron2/validation_graph_cutoffs.md)
- [Measured: neuron 21 reconstruction results and SWC projections](docs/pointneuron2/graph_cutoff_results.md)
- [Completed preparation: neuron 21 paired reconstruction](docs/pointneuron2/graph_cutoff_comparison.md)
- [Latest: score filtering and suppression audit, with proposed next check](docs/pointneuron2/selection_audit.md)
- [Current: corrected-input training results and score-filtering findings](docs/pointneuron2/repair_pilot_results.md)
- [Completed preparation: corrected-input coordinate pilot](docs/pointneuron2/repair_pilot_ready.md)
- [Current: confirmed decoder/range/coordinate defects and repairs](docs/pointneuron2/root_cause_findings.md)
- [Corrected candidate-pool and spatial-selection diagnostics](docs/pointneuron2/mixed_candidate_comparison.md)
- [Historical reconstruction comparison](docs/pointneuron2/sampling_ablation_reconstruction_results.md)
- [Historical training pilot results](docs/pointneuron2/sampling_ablation_pilot_results.md)
- [Historical matched sampler pilot](docs/pointneuron2/sampling_ablation.md)
- [Historical automatic foreground comparison](docs/pointneuron2/foreground_comparison.md)
- [Automatic preprocessing: current direction and next comparison](docs/pointneuron2/automatic_preprocessing.md)
- [Current pipeline status](docs/pointneuron2/README.md)
- [Current input-quality findings and next research discussion](docs/pointneuron2/input_audit.md)
- [PN 2.0 scripts and local commands](scripts/pointneuron2/README.md)
- [Training integration results](docs/pointneuron2/sampling_training.md)
- [Optional skeleton inspection tool](docs/pointneuron2/manual_review.md)

Entry scripts live in `scripts/pointneuron2`; shared algorithms live in
`src/pointneuron`; settings live in `configs/pointneuron2`. Figures and readable
reports live in `docs/pointneuron2`. Logs, caches, checkpoints and the local
Python environment live in ignored `tmp/pointneuron2`. The large source dataset
and historical PN 1.0 artifacts remain in the original parent checkout and are
referenced explicitly by PN 2.0 runners.

The main pipeline should run automatically from the input image through
reconstruction. Manual mask correction may be considered later if measured
failures justify it; the existing parameter viewer is an optional diagnostic,
not a required training or inference step.

The image-reader, intensity-range and coordinate repairs are implemented. The
corrected raw/local-coordinate pilot has completed training. Local validation
loss is modestly lower, but crop geometry gains remain small and the current
graph score cutoff discards substantial reference coverage. The completed
selection audit isolates that loss after sphere suppression. A paired full-volume
check on validation neuron 21 has completed: lower-cutoff spatial F1 improves
from 0.7880 to 0.8326, with lower precision and topology F1. The seven-neuron
extension completed: mean spatial F1 0.6598 to 0.6671, two gains, one loss and four
unchanged cases. Earlier global proposal selection and sparse anchor selection
limit the effect of the graph cutoff; no new cutoff is promoted.
The saved-output audit locates major global-selection losses on neurons 35/79,
while neuron 94 already lacks proposals before global selection. A matched
spatial control recovers proposal support on 35/79 but worsens the proximity
proxy on neuron 19. No new selector or reconstruction result is promoted.
The fixed 50/50 hybrid retains most recovered support on 35/79 and partly improves
the pure-coverage proximity tradeoff, but still regresses on neuron 19. One
neuron-79 full-SWC control has completed in the user's visible terminal: spatial
F1 0.6560 -> 0.6862, with higher precision and recall. Much recovered proposal
support still fails to become the final SWC; the fragmented reference also limits
the topology metric. See the measured results above. No selector is promoted.
Earlier runs are historical results on incorrectly decoded inputs and require
reassessment. Verified changes are pushed to the `pointneuron2` GitHub branch.
We discuss substantive
script/strategy changes and agree on long training runs together; small local
checks may run as part of an agreed task. Stage 1 has not established a
reconstruction improvement yet.

## Inherited PN 1.0 reference

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

Build the read-only sample diagnostic table from existing local artifacts:

```powershell
py scripts\build_sample_diagnostics.py
```

Outputs are under `tmp/diagnostics`: a per-sample CSV/JSON, a failure summary,
and an exact source manifest. The [schema and triage rules](docs/sample_diagnostics.md)
explain missing values, provenance, and the distinction between completed runs
and diagnostic `review_status` (`CLEAR` / `REVIEW`), separate from actual
`execution_status`. Alignment/data-pairing issues are distinguished from
foreground preprocessing. This command does not train or run inference.

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
