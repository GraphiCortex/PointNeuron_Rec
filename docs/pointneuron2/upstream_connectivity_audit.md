# Frozen upstream connectivity feasibility

17 October 2026. This bounded CPU audit asks whether the existing two-crop
PointNeXt/Transformer checkpoint supplies enough trusted proposals to train
connectivity on branched examples while keeping that checkpoint frozen. It does
not fit a model, compare GATv2 quality, or estimate generalization. The diagnostic
and criteria below were fixed before running it.

The intended architecture remains image-derived cloud -> PointNeXt -> Transformer
-> GATv2 -> forest/SWC. No production model, normalization default, selector,
target representation or loss is changed by this script.

## Cases and policies

| Training neuron/crop | Purpose | Targets |
| --- | --- | ---: |
| 0 / 0 | Original fitted control | 6 |
| 0 / 1 | Original fitted control | 12 |
| 92 / 4 | Modest branched case from the representation oracle | 16 |
| 86 / 3 | Dense branched case from the representation oracle | 418 |

Cases were selected for structural coverage, before looking at upstream
predictions. They are deliberately limited training examples, not a representative
benchmark. The upstream query model was fitted only to the first two crops;
the image-field model previously saw the wider training split.

Use the saved 2,000-update query parameters with the **explicit batch-one
current-crop normalization variant**. That remains a research control. Use its
original confidence head, not an undocumented overlay of the later confidence
replay. The central geometry gate is evaluated before confidence filtering, so
a confidence-only update cannot rescue a geometric failure.

Reuse the exact original control clouds. For the two new cases, run the same
fresh-baseline field weights on CPU, then expanded-random sampling with up to
2,048 distinct points and seed `sample_index * 10000 + patch_index`. Original
control fields used CUDA with cuDNN TF32 enabled; there is **no claim that CPU
and GPU field outputs or selected clouds are identical**. This audits the actual
saved/new clouds, not paired CPU-versus-GPU field behavior.

Freeze all clouds before opening any SWC. Only the four selected training
image caches and three selected training SWCs are opened. The manifests contain
other split identities but development/test image caches and references are not
opened. PointNeXt forwards, field forwards, checkpoint hashes, feature hashes,
source hashes, exact assignments and per-edge labels are recorded. No optimizer,
GATv2 forward or GPU workload is created.

## Measurements and fixed decision rule

Report reference-curve proximity at 2/6 voxels for image foreground, the expanded
candidate pool, the selected cloud, all proposals and accepted proposals. These
are point-support measurements, not reconstructed-edge accuracy. Distances use
native voxel coordinates; physical isotropy is not established.

Retain the existing Hungarian assignment and query loss. A match supplies
connectivity supervision only within two voxels. Build reference-free kNN-16
union radius-eight candidates on all 512 proposals. Report:

- Trusted one-to-one target matches, independently of node confidence.
- Visible reference junctions (degree >= 3) with trusted matches.
- Known positive/negative/unknown candidate counts.
- Required edges lost to untrusted endpoints versus missing candidate pairs.
- Attainable recall over **all** reference edges, both before and after the .5
  node cutoff, even with a perfect edge classifier.
- Any-query target proximity separately; this can reuse a query and is not a
  substitute for a one-to-one assignment.

A branched crop passes frozen-upstream feasibility if >=95% of targets have
trusted matches, >=95% of all target edges have trusted candidate endpoints,
every visible junction has a trusted match, and positive and negative edge
supervision are both nonempty. Both branched crops must pass. The 95% threshold
is a declared practical feasibility criterion, not a biological quality standard.
Confidence losses are reported separately; passing the geometry gate would not
establish a usable end-to-end model.

**Stopping rule:** one inference audit, no threshold search, no replacement
crops and no confidence replay. If the gate fails, do not train a graph scorer
on this frozen checkpoint and then interpret low edge loss as reconstruction
learnability. Inspect whether input support or proposal placement is limiting,
then discuss the smallest branched upstream or joint-learning experiment with
the user. Failure is not evidence that PointNeXt, Transformers or GATv2 cannot
learn the task. If the gate passes, prepare a matched frozen-feature GATv2 versus
pairwise fit with agreed budget and success criteria; do not launch it here.

## CPU reproduction

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\pointneuron2"
& .\tmp\pointneuron2\sampling_env\Scripts\python.exe -u `
  .\scripts\pointneuron2\audit_upstream_connectivity.py `
  --config .\configs\pointneuron2\upstream_connectivity_audit.json `
  --data-root "C:\Users\Seywe\Documents\GitHub\PointNeuron\data\gold166" `
  --output-dir .\tmp\pointneuron2\local_runs\upstream_connectivity_repeat
```

Use a fresh output directory; existing evidence is never overwritten. Generated
clouds, predictions, plots and source snapshots stay in ignored `tmp/pointneuron2`.
The reviewed results and selected inspection plots will be copied to visible
documentation after completion.
