# New implementation

The standalone `pn2` package contains the common reconstruction
evaluator in `pn2/evaluation/`: strict SWC parsing, explicit benchmark/output
manifests, continuous-curve geometry and failure-aware comparison reports.
It does not import historical model code or run inference/training.

Native SWC coordinates are parsed once; declared XYZ spacing is applied once
by geometry. Distances use continuous segments, with arclength-weighted midpoint
quadrature for length statistics. Endpoint/junction support is geometric and
does not establish correct wiring.

`pn2/inputs/` adds explicit image-only cases, the retained verified Vaa3D decoder,
native XYZ provenance and deterministic covering tile inventories. It imports
no historical package, SWC parser or model. Physical spacing is declared
metadata; array/tile boxes remain native voxel coordinates. This interface
does not yet select point clouds or implement reconstruction.

`pn2/inputs/clouds.py` adds an explicit PN1 reference input control with distinct
image-voxel selection and short-input accounting. `pn2/reference/` contains the
retained encoder/proposal operators, a local-head adapter and a fresh untrained
CPU forward audit. Model code requires the optional PyTorch dependency already
present in the retained runtime. The base image/evaluation interfaces remain
independent of it. No training loss or graph/export implementation is ported yet.

The working design is the [revised proposal](../docs/revised_proposal.md).
