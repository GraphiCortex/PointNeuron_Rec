# Configuration status after the research reset

New configurations belong in **[PN2/configs](../../PN2/configs/README.md)**.
This parent folder retains historical recipes and identity metadata.

The active plan is [the revised proposal](../../docs/pointneuron2/revised_proposal.md).
The existing experiment JSON files are historical recipes, not current training
defaults. Their generated dependencies may have been removed by the cleanup.

`partition_seed0.json` preserves the current neuron identities for benchmark
planning. Its 33/7/8 groups are disjoint within PN2, but all 48 neurons appeared
in historical PN1 training. Preserve that exposure distinction. Old cache paths
are not a substitute for the explicit source manifest in the new benchmark.

A repaired-PN1 reference recipe and a fresh PN2 training recipe will be specified
under the shared evaluation/data contract before new substantial runs. No new
model initialization, threshold or training budget is selected by this cleanup.
