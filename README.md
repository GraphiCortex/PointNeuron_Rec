# PointNeuron 2.0 — fresh research start

We are starting the next development cycle from the [revised proposal](docs/pointneuron2/revised_proposal.md).
The intended automatic architecture remains image-derived point clouds →
PointNeXt → Transformer proposals → GATv2 connectivity → skeleton/SWC.

Start here:

- [Fresh-start status and cleanup record](docs/pointneuron2/fresh_start.md)
- [Revised proposal and bounded development path](docs/pointneuron2/revised_proposal.md)
- [What PN 1.0 and PN 2.0 actually establish](docs/pointneuron2/research_reset_evidence.md)

The current milestone is a common whole-neuron evaluator and a repaired
PN 1.0-style baseline specification. No production PN 2.0 checkpoint is promoted,
no historical fit is resumed, and no new training run is scheduled.

## Workspace

This real nested checkout is on branch `pointneuron2`. The original parent
checkout stays on `main`. Keep source, tests, configurations and readable reports
visible; put generated runs, caches and environments in ignored `tmp/` folders.
Substantial model changes and GPU runs are discussed before implementation/launch.

Useful source and verification tools are retained at their existing paths.
[Historical experiments](HISTORY.md) preserve the record; their old ready/next
labels and run commands are retired. Many old generated inputs were removed.

The cleanup preserves the working Python environment, pinned PointNeXt reference
sources and one compact diagnostic archive. The archive is historical evidence,
not a default initialization or a runnable continuation chain.
