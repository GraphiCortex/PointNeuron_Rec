# Fresh start: status and retained assets

**Workspace location:** all fresh work is now under [PN2](../../PN2/README.md),
including its active [proposal](../../PN2/docs/revised_proposal.md) and new
ignored `PN2/tmp/` outputs. This document records the earlier completed cleanup.

21 October 2026. The user requested a fresh PN2 start and deletion of obsolete
remnants for storage and clarity. Cleanup is complete. The [revised proposal](revised_proposal.md)
is the working research direction; model changes and substantial runs still need
an explicit, reviewable experimental specification.

## What was removed

- 113 exact obsolete output/scratch targets: 27,424 regular files, about **4.00 GB**.
- Old PN2 run folders, crop/field caches, intermediate checkpoint series,
  repeated timing/smoke/test outputs and bytecode caches.
- Explicitly named PN2 staging/transfer folders and the earlier workspace backup
  under the original checkout's ignored `tmp/`; PN1 artifacts were excluded.
- Three tracked LaTeX build byproducts (`.aux`, `.log`, `.out`); historical source
  and PDF artifacts remain available.

A verified **69.5 MB** archive offsets the deletion, giving approximately
**3.93 GB net logical storage recovery**. Filesystem allocation and unrelated
system activity can make drive-free-space changes differ from logical bytes.

## What remains and why

- The original dataset, reference files and original PN1 checkout/promoted
  checkpoint, split and measured results: the foundation of the comparison.
- Revised proposal, evidence audit, tracked summaries/overlays and Git history:
  the positive and negative findings should not be rediscovered from scratch.
- Useful model/data/geometry code, tests and configs at their current paths:
  several historical runners supply shared helpers; deleting them saves little
  storage and can break imports.
- `tmp/pointneuron2/sampling_env`: one working Python environment, retaining
  existing installed dependencies instead of repeating setup.
- `tmp/pointneuron2/pointnext_reference`: small pinned reference sources used
  by independent implementation checks.
- `tmp/pointneuron2/preserved_reference/research_before_reset_20261021.zip`:
  historical reports/ledgers/source snapshots, the final canonical diagnostic
  checkpoint and outputs, matched stable outputs, exact four-crop cache, and
  the paired field epoch-30 checkpoints. Archive entries and retained hashes
  were verified before deletion; the archive hash was checked afterward.

The archive is diagnostic evidence, **not fresh-training initialization**.
Most old experiments are no longer runnable in place because caches and
intermediate optimizer chains were intentionally removed. Nothing is silently
resumed. The historical indexes are [workspace](../../HISTORY.md),
[research](HISTORY.md) and [scripts](../../scripts/pointneuron2/HISTORY.md).

## Next milestone

Start Milestone A of the revised proposal: define the common explicit-ID
whole-neuron evaluator, source/exposure manifest and repaired PN1-style reference
recipe. Judge final curves, branches, wrong connections, failures and resources
on the same data. Then finish the automatic whole-volume PN2 interfaces and
prepare one broader paired comparison. Repeated perfection of four training
crops is not the next objective.

No model implementation, loss, target rule, normalization, cutoff or training
budget changes in this cleanup. No inference or training was launched.
The [machine-readable cleanup record](research_reset_20261021/cleanup.json)
records exact target paths, archive digest and protected-file validation.
