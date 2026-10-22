# Training-reference contract audit

23 October 2026. A read-only inventory of the 33 explicit training references
finds a bounded graph-encoding issue that must be understood before constructing
connectivity labels. No annotations, sampling settings or metrics were changed.
No images, development references or reserved references were opened by this
audit. Counts describe the files, not verified biological connectivity.

## Findings

All 33 sources parse as structurally valid forests using the already documented
`opaque_integer` reference policy. Unknown signed integer type codes are retained
as metadata; no anatomical interpretation or node filtering is applied.

Thirty sources have one encoded component and at least one encoded junction.
The other three have multiple components, no encoded junctions, and exactly
coincident coordinates across components:

| Historical alias | Source | Components | Encoded endpoints | Cross-component shared-coordinate groups | Endpoints coincident with a node in another component |
|---|---|---:|---:|---:|---:|
| 23 | Fruitfly, done_1_CL-I_X_OREGON_R_ddaD_membrane-GFP | 51 | 102 | 26 | 76 |
| 66 | Mouse RGC, HO_140921c14 | 200 | 400 | 114 | 299 |
| 75 | Mouse RGC, sv_140918c8 | 80 | 160 | 47 | 122 |

The pattern is consistent with an encoding that splits paths at shared
locations. That is a hypothesis, not established source provenance. Exact
coordinate coincidence alone does not prove biological connection, and this
audit does not weld nodes or relabel disconnected pairs. Geometry agreement can
still be measured against these files. Topology targets require an explicit
interpretation of what their component boundaries mean.

The three file headers contain only the generic `vaa3d_traced_neuron` name,
an empty comment and column names. They do not document the reason for the
component splits or custom type codes. Header inspection therefore does not
resolve the biological interpretation.

Eleven sources contain negative type metadata. Four sources (aliases 9, 93,
95 and 96) report radius 1 at every node. Constant radii do not prove placeholder
values, but their anatomical accuracy and measurement provenance are unverified.
Do not silently turn either type codes or reported radii into trusted anatomical
labels.

The historical four-crop learnability images, aliases 0, 86 and 92, are among
the 30 single-component sources. The three encoding cases above therefore do
not explain those earlier failures by themselves. Nor does a valid forest prove
that every branch is complete, correctly traced or aligned with its image.

## Consequences for the next model

Before GAT supervision, document which relations are known positive, known
negative or unknown under the source annotation convention. In particular,
unconnected IDs from different components are not automatically certified
negative connections. Verify the source convention before proposing a repair;
do not choose a welding tolerance merely to make a training gate pass.

Keep reference completeness separate from graph validity. An SWC may trace one
target without labeling every visible structure. An unmatched image region or
proposal is not automatically verified biological background. The fully
automatic product still needs a clear single-neuron input contract.

This is a prerequisite for defensible labels, not evidence that a new loss,
different architecture or longer training will succeed. The intended
PointNeXt / Transformer / GATv2 stages remain unchanged.

## Reproduction and checks

The visible [audit script](../scripts/audit_training_references.py) reads only an
explicit inventory whose entries all declare `training`. It preserves malformed
or missing references as unscorable entries, checks source hashes before and
after reading, requires a fresh output path, and makes no repairs.

```powershell
& .\tmp\phase1_cpu_env\Scripts\python.exe .\scripts\audit_training_references.py `
  --inventory .\configs\training_inventory.json `
  --output .\tmp\reference_graph_inventory_20261023\report.json
```

That output path already exists; preserve it. The executed synthetic smoke check
retained all four nodes and both edges of two chains sharing a location,
identified their shared endpoints, retained a cyclic reference as unscorable,
and confirmed unchanged source bytes. The actual 33-reference inventory
completed successfully. This checks execution and counting, not biological
annotation correctness.

Raw reports and fixtures remain under ignored
`tmp/reference_graph_inventory_20261023/`. The compact visible
[audit evidence](evidence/training_reference_audit_20261023.json) preserves all
per-source counts and hashes. The inventory SHA-256 is
`90b46f6983c6be2f5d742549832b1837246dd7a95686db0c1def0bb8736cac58`;
the script SHA-256 is
`6c7e90a11813ff06b7d9330f2405a964e9261c960100f9d4a36479a57c894d0d`.
