# New implementation

The standalone `pn2` package currently contains the common reconstruction
evaluator in `pn2/evaluation/`: strict SWC parsing, explicit benchmark/output
manifests, continuous-curve geometry and failure-aware comparison reports.
It does not import historical model code or run inference/training.

Native SWC coordinates are parsed once; declared XYZ spacing is applied once
by geometry. Distances use continuous segments, with arclength-weighted midpoint
quadrature for length statistics. Endpoint/junction support is geometric and
does not establish correct wiring.

The working design is the [revised proposal](../docs/revised_proposal.md).
