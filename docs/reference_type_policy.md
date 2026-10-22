# Geometry-only reference types: policy declared before rescoring

The bounded rescore is now complete; see the
[two-image results and integrity checks](phase1_geometry_rescore_results.md).
The prospective policy below was committed before those scores.

22 October 2026. The user authorized one bounded CPU rescore of the frozen
two-image sampling probe after this compatibility investigation. No new point
selection, filters, training, coordinate alignment or parameter search belongs
to this follow-up. The original strict-reader result stays preserved.

## Finding and limits of the evidence

The fruitfly reference has 466 rows, including 46 with type `-73`. The header
names Vaa3D's `sort_neuron_swc` plugin but does not explain the codes. Their
anatomical meaning, any signed-byte-overflow origin, or an annotator's intent
to mark unreliable branches has **not** been established.

Upstream source inspection establishes a narrower compatibility argument:

- [Vaa3D's reader and writer](https://github.com/Vaa3D/v3d_external/blob/215c7803a3aeb7c7a5772f3db2380ee6737d4edc/v3d_main/basic_c_fun/basic_surf_objs.cpp#L288-L378)
  parse the type field as an integer and write that value without a nonnegative
  restriction or anatomical remapping. Retrieved blob:
  `44ff5d3454d80c6398ff754b04159160a477581c`.
- [The sort implementation](https://github.com/Vaa3D/vaa3d_tools/blob/475e4ca92d4e51de10f1c05d80cef6615432c087/released_plugins/v3d_plugins/sort_neuron_swc/sort_swc.h#L886-L909)
  copies type metadata; its exporter writes the integer value. Retrieved blob:
  `6e89cf3e4b89b69c93d5c1757693e97e9410f4dc`.
- [The inspected regular rendering path](https://github.com/Vaa3D/v3d_external/blob/215c7803a3aeb7c7a5772f3db2380ee6737d4edc/v3d_main/3drenderer/renderer_obj.cpp#L2523-L2552)
  selects a fallback colour for out-of-range type indices. This does not prove
  that this particular code was intended as a colour tag. Retrieved blob:
  `240345cb547d291e0f1dd2f2a9c9396ef2ac9c7a`.

These pinned sources were retrieved during this audit. The exact software
version that produced the 2015 reference is unknown. No upstream implementation
is copied into PN2. The evidence supports preserving source metadata, not
claiming that negative codes are standard anatomical classes.

## Explicit policy

The new `opaque_integer` reading mode is **opt-in and reference-only in this
probe**. It preserves type codes as signed 64-bit integers. It does not set
them to zero, convert them modulo 256, infer axon/dendrite labels, remove nodes,
or relabel the source file. Every provided node and parent link contributes to
geometry agreement. This is the declared interpretation of the supplied curve;
its biological completeness/correctness remains unverified.

All other validation remains: seven columns, integer IDs/parents/types within
int64, positive unique IDs, existing parents, no self-parent/cycle, finite
coordinates/radii and nonnegative radii. Zero radii and multiple roots retain
the existing conventions. The `nonnegative` default remains active for the
common reconstruction evaluator and predicted SWC validation. There is no
automatic fallback from one mode to the other.

The current geometry metrics use coordinates and parent edges, not neurite
type classes. Therefore the opaque metadata can be preserved without changing
the metric definition. Unknown tags must not be used as anatomical training
labels later without a separate justified mapping.

## Read-only source audit before new scores

An independent column-by-column check compared the parsed objects with the raw
rows, including IDs, types, coordinates, radii and parent IDs. All values and
source bytes were preserved. The forest checks pass for both inputs under the
new explicit policy:

| Reference | Nodes | Components | Length, native voxels | Endpoints | Junctions |
|---|---:|---:|---:|---:|---:|
| Fruitfly alias 32 | 466 | 1 | 2636.62 | 18 | 16 |
| Series021 alias 46 | 509 | 1 | 789.71 | 11 | 9 |

Reference byte hashes:

- Fruitfly: `d5d02064d5bc42f27f60270365d46c83e89fa191bab9603af2e7f1d0df651d3b`
- Series021: `e4f72445f008af58dbaee2504f4cfaa40d2fc6200345a359ea60cac89c98a121`

The [full pre-score audit](phase1/reference_type_audit_20261022.json) is retained
in visible source, with the raw copy under ignored
`tmp/reference_type_audit_20261022/audit.json`. This is structural/syntax evidence,
not certification of the reference anatomy.

## Frozen comparison and checks

Apply the same explicit policy to **both original case IDs**, all three methods,
both densities and all five seeds: 60 saved clouds. Keep the original recipe,
counts, selection seeds, tolerances, quadrature and advancement criteria exactly
as declared in [the first protocol](phase1_sampling_protocol.md). Do not choose
another reference or omit negative-tagged nodes. Verify the original artifacts
and source files remain unchanged. Verify that Series021's replayed geometry
scores exactly match its existing scored outputs, since its parsed geometry is
identical under both policies.

Synthetic checks require signed metadata preservation, strict default rejection,
continued rejection of malformed geometry and equality of spatial scores when
only metadata changes. The integration check preserves the original strict
report and frozen packets. New reports record policy, type histograms, source
hashes, retained node counts and parser/metric code hashes.

New evaluation artifacts and overlays go into a fresh
`tmp/phase1_sampling_20261022/evaluation_opaque_integer/` directory. The original
`evaluation/` and its overlays are not overwritten. The original Boolean stays
false; the new report has its own verdict. No full input generation is repeated.

```powershell
Set-Location "C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2"
& .\scripts\run_phase1_sampling.ps1 -Action evaluate-geometry
```

Expected duration from the measured old evaluation: roughly 15-20 minutes on
CPU; no GPU or model training. If structural validation, source equality or the
stable-case replay fails, stop and report it without repairing the reference or
changing point selection. Even a passed numerical gate would justify only
broader input qualification, conditional on the documented reference semantics.
