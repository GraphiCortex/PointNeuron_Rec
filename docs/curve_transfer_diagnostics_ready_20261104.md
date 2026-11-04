# Frozen-checkpoint transfer and matched forests — ready, not run

The user parked the curve-supervised Transformer hypothesis. The next step is
to measure learning contribution and transfer while holding the original
consolidation/scaffold fixed. No new training or node-stage criterion is adopted.
The original and supported-variant failures remain on record. See the
[diagnostic protocol](curve_transfer_diagnostics_protocol_20261104.md).

## Existing evidence answers the two-voxel input question

These values were already computed by the completed formulation audit. They
are read from saved reports, not a repeated audit. The earlier six-voxel headline
did not adequately expose this localization/density limitation.

| Input support within 2 voxels | 2048 points | 8192 points |
|---|---:|---:|
| Training, neuron macro | 71.10% | 92.82% |
| Development, neuron macro | 70.24% | 91.98% |
| Training apparent-thin/dim, neuron macro | 39.87% | 70.17% |
| Development apparent-thin/dim, neuron macro | 40.56% | 83.23% |
| Exact fitted frog alias 22, box 5: all curve | 36.05% | 74.46% |
| Same box: apparent-thin/dim curve | 16.68% | 69.53% |

The exact frog thin/dim stratum is 29.8648 voxel-length units: raw support
4.9819 at 2048 and 20.7650 at 8192. The original learned points support 8.8404
there. More points improve spatial opportunity, and learned offsets already
improve on raw two-voxel input proximity locally. Neither observation isolates
the sole bottleneck or proves that additional points carry sufficient image
evidence. Width/contrast strata are fixed image-derived proxies, not anatomical
caliber labels. Far-source (>6) votes account for only 14.09% of retained-point
errors in the saved fit; offset errors at distances 2–6 still matter.

Source: `tmp/curve_formulation_audit_20261103/report.json` summary/partitions and
the exact case/tile row in boxes.jsonl. Source hashes and the full relevant
saved fields are copied into [portable input evidence](evidence/curve_transfer_input_support_20261104.json).

## What is missing and now prepared

The previous no-learning control used the same isolated-point scorer. It was
not passed through the forest. The previous 81.49% versus classical 52.46%
segment F1 compares fitted coordinates to an untuned classical method on the
training boxes and cannot establish transfer or fair learned superiority.
The local report's statement that isolated-point results support learned
fitting is narrower than a claim that learning improves downstream geometry.

The new launcher compares frozen dense nodes, all raw input points and
Sato-ranked top-512 points, using radius-1 consolidation and the same k=6
Sato-cost forest. Counts are disclosed and not equalized. No support-count
filter or changed threshold is introduced. It reports per-box/per-neuron
segment metrics, diagnostic candidate unions and conservative reference-path
shortcut flags, including unknown/off-reference categories. No edge is changed
using reference scores. The saved fit dense graphs are reused.

Development uses the original 2048-point inputs for all 151 reference-present
boxes from seven development neurons, not the 264 unknown boxes. These neurons
did not train this four-box checkpoint, but this cohort is repeatedly inspected
development data, not a new untouched test. No result is yet available. CPU
numerical replay on the four fitted boxes must pass before development forward.
No GPU, backward, optimizer, fine tuning, best-checkpoint selection or subsequent
training is invoked. SWC root/type/radius placeholders remain unqualified.

Eight short CPU unit/integration tests passed, covering continuous paths,
detours, ambiguity, fragmentation, boundaries, subdivision, exact frog identity,
and the actual three-policy graph/scoring/report boundary. Source/input binding
preflight passes. These tests do not establish real control scores, CPU model
parity, development accuracy or a biological wrong-join rate. The timed stages
have not been launched by the assistant.

## User-local PowerShell commands

First, CPU-only fit controls and shortcut diagnostics. Expected seconds to a
few minutes based on the earlier 13-second scaffold, but the denser controls
have not been timed. Hard cap 30 minutes:

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\phase1_cpu_env\Scripts\python.exe' -B '.\scripts\measure_curve_transfer.py' --stage fit
```

Then, frozen development inference plus matched controls/scoring. CPU-only,
including the retained Torch runtime as a CPU subprocess. Runtime is unmeasured;
hard overall cap two hours. No automatic GPU fallback or extension:

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\phase1_cpu_env\Scripts\python.exe' -B '.\scripts\measure_curve_transfer.py' --stage development
```

Each creates a fresh `tmp/curve_transfer_<stage>_<UTC timestamp>` with report.md,
report.json, graph/SWC artifacts, shortcut details and inference receipts. A
sibling `_wall_guard` folder holds the relayed console log and completion/cap
status. Failed/partial output is retained and not automatically retried.

## Decision after these measurements

Do not implement a new head or prepare sixteen boxes now. Assess whether dense
nodes improve downstream geometry beyond the no-learning controls, whether
that contribution transfers, and which reference-path shortcut/coverage failures
remain. A rank reversal from two consolidation policies shows that improving
point precision alone can hurt reconstruction; it does not establish that point
precision is irrelevant or that stray nodes are harmless.

If a new sixteen-box experiment becomes justified, its downstream acceptance
criteria would be a disclosed post-hoc revision relative to the failed original
gate, then frozen prospectively before new training. Require paired evaluation
on untrained neurons, numeric absolute and comparison floors, worst-neuron and
structural reporting, and explicit shortcut/unknown handling. These diagnostics
adopt no such floors and cannot retroactively pass either old recipe.
