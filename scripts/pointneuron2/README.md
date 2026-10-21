# PointNeuron 2.0 — scripts for the fresh development cycle

The active direction is the [revised proposal](../../docs/pointneuron2/revised_proposal.md).
The next implementation milestone is a common explicit-ID whole-neuron evaluator
and a repaired PN1-style baseline specification. No new training entry point or
training command has been prepared for this fresh cycle.

Existing experiment scripts remain as historical implementation references and
shared utilities; they are not recommended next commands. Many import one another,
so their paths are retained. Old cached inputs/checkpoint chains have been removed.
[The historical script index](HISTORY.md) records earlier protocols and results.

`maintenance/prepare_cleanup.py` inventories only the named retired PN2 scopes,
builds a compact verified archive, and never deletes files.
`maintenance/apply_cleanup.ps1` previews by default; its explicit `-Apply` action
checks the archive, protected files and exact scoped paths before deleting.
The 21 October cleanup is complete; its plan cannot be applied again automatically.
See [fresh-start status](../../docs/pointneuron2/fresh_start.md).

Keep future scripts/configs/reports visible and generated artifacts in ignored
`tmp/pointneuron2/` directories. Substantial GPU runs get a separate visible
PowerShell command for the user to launch after the experiment is agreed.
