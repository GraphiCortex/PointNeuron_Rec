# New implementation

The standalone `pn2` package contains the common reconstruction
evaluator in `pn2/evaluation/`: strict SWC parsing, explicit benchmark/output
manifests, continuous-curve geometry and failure-aware comparison reports.
It does not import historical model code or run inference/training.

Native SWC coordinates are parsed once; declared XYZ spacing is applied once
by geometry. Distances use continuous segments, with arclength-weighted midpoint
quadrature for length statistics. Endpoint/junction support is geometric and
does not establish correct wiring.

The common evaluator and predicted SWC validation keep nonnegative type codes
by default. The sampling probe's separately declared
[reference-only policy](../docs/reference_type_policy.md) can preserve signed
integer type metadata verbatim for geometry agreement. It retains every node
and all structural checks; no anatomical type labels are inferred or changed.

`pn2/inputs/` adds explicit image-only cases, the retained verified Vaa3D decoder,
native XYZ provenance and deterministic covering tile inventories. It imports
no historical package, SWC parser or model. Physical spacing is declared
metadata; array/tile boxes remain native voxel coordinates. This interface
now also contains a reference-free three-control CPU sampling probe. It does
not implement reconstruction or select a validated product sampling policy.

22 October scope correction: PN1-only `clouds.py` and `reference/` were removed
from active source and are recoverable at pushed commit `48519ff`. Current
source contains no PN1 model or foreground/sampling default. The approved
bounded [sampling protocol](../docs/phase1_sampling_protocol.md) tests uniform,
intensity and published Sato weights without reference input. A separate
`evaluation/cloud_support.py` measures saved point support and gaps using SWCs.
Neither execution nor those support proxies validate an encoder, training loss,
graph model or end-to-end reconstruction.

The working design is the [revised proposal](../docs/revised_proposal.md).

The explicit learned component now adds `models/node_objective.py`, a node-only
one-to-one position/used-slot loss with no radius or topology terms. Torch is
required only when this optional module or the experiment runner is imported;
image preparation and the standalone evaluator retain their model-free paths.
`evaluation/node_proposals.py` supplies one-to-one proposal metrics using NumPy /
SciPy. `supervision/curve_nodes.py` supplies reference-only rectangular targets.
The model factory in the visible experiment script reuses hash-pinned historical
encoder/head source read-only with fresh weights; this is not a vendored product
backend or a trained whole-neuron inference implementation.
