# PointNeuron 2.0 — fresh research start

All fresh development lives in the **[PN2 folder](PN2/README.md)**, starting from
the [revised proposal](PN2/docs/revised_proposal.md).
The intended automatic architecture remains image-derived point clouds →
PointNeXt → Transformer proposals → GATv2 connectivity → skeleton/SWC.

Start here:

- [Fresh-start status and cleanup record](docs/pointneuron2/fresh_start.md)
- [Revised proposal and bounded development path](PN2/docs/revised_proposal.md)
- [What PN 1.0 and PN 2.0 actually establish](PN2/docs/research_reset_evidence.md)

The current milestone is a common whole-neuron evaluator and a repaired
PN 1.0-style baseline specification. No production PN 2.0 checkpoint is promoted,
no historical fit is resumed, and no new training run is scheduled.

## Workspace

This real nested checkout is on branch `pointneuron2`. The original parent
checkout stays on `main`. New source, scripts, tests, configurations and readable
reports go under `PN2/`; new generated artifacts go under ignored `PN2/tmp/`.
Substantial model changes and GPU runs are discussed before implementation/launch.

Useful source and verification tools are retained at their existing paths.
[Historical experiments](HISTORY.md) preserve the record; their old ready/next
labels and run commands are retired. Many old generated inputs were removed.

The cleanup preserves the working Python environment, pinned PointNeXt reference
sources and one compact diagnostic archive. The archive is historical evidence,
not a default initialization or a runnable continuation chain.
