# CPU diagnostic launches — Stage2 stays stopped

Use the existing Stage2 CPU environment. No installation,training,GPU,GATv2 or
Stage2 restart. All thresholds and definitions are fixed in the
[protocol](post_stage2_diagnostics_protocol_20261106.md) and
[config](../configs/post_stage2_diagnostics_20261106.json).
Actual runtimes are unmeasured;caps are limits,not predicted completion times.
Run commands separately. Benchmark scoring depends on benchmark download;
the other diagnostic actions are independent. Nothing launches the next action.

## Gold-to-ridge offsets and five-neuron visual packet

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\post_stage2_diagnostics.py' --action ridge_review
```

CPU40 images;expect tens of minutes to hours;hard cap2h. Original image decoding
and ridge Hessians are distinct diagnostics;retained Sato filters are untouched.
Output `tmp\post_stage2_diagnostics_20261106\ridge_review\report.{json,md}`,
per-neuron NPZ/JSON and `index.html` linking the five XY/XZ/YZ review sheets.
Flags are possible untraced structure;afterward inspect the PNGs in this shared
workspace before making visual claims. No relabeling.

## External BigNeuron outputs: download/extract,then score

The existing gold166 download has images/manual annotations. Automatic benchmark
outputs are a separate [official release](https://github.com/BigNeuron/Data/releases/tag/gold166_bt_v1.0).
This command downloads726,174,044 bytes (about693MiB) and extracts SWC contents
only for the seven development identities. Other archive entries are skipped;
no reserved morphology is parsed. Download duration depends on the network;
hard cap30min. Archive and extracted seven-case files remain in the new tmp root.

```powershell
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\post_stage2_diagnostics.py' --action benchmark_download
```

Then:

```powershell
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\post_stage2_diagnostics.py' --action benchmark_score
```

CPU,hard cap2h;likely minutes to hours depending on output sizes and algorithm
variants. Native-coordinate checks precede scoring;no fitted alignment.
Missing/invalid case results are disclosed and count as zero. If any of the seven
identities cannot be mapped unambiguously,the action writes a mapping report
instead of fabricating a complete comparison. The duplicated Cell1 basename is
handled conservatively;archive layout has not been examined in a timed run yet.
Primary151-box and separate whole-image results,best/median/per-variant tables
are saved under `benchmark_score\report.{json,md}`. Native origin remains a
disclosed check limitation pending image-overlay confirmation.

## Resampled-F1 decomposition

```powershell
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\post_stage2_diagnostics.py' --action decompose
```

CPU saved C0/C1 and policy graphs only;expect tens of minutes to hours;cap2h.
Original scores are replayed before fixed isolate/spur/duplicate-path ablations.
Full per-neuron F1 deltas,matching counts,length ratios and category counts:
`decompose\report.{json,md}`. No ablation is adopted as postprocessing.

## Classical-anchored editing headroom

```powershell
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\post_stage2_diagnostics.py' --action editing
```

CPU151 development boxes;expect hours;hard cap4h. Fixed bounded,reference-assisted
prune-only and prune-plus-connected-addition oracles;length<=1.25x per box.
The search is not a proven global maximum. Report(b) resampled F1_2>=60% permits
discussion/drafting an editing contract;also state whether(a) alone reaches it.
No automatic drafting,training or new stage follows. Output
`editing\report.{json,md}` plus explicitly reference-assisted SWCs.

All outputs are under `PN2\tmp\post_stage2_diagnostics_20261106`;watchdog logs
are in similarly named sibling guard folders. Keep the machine awake. Actions
are write-once and must run one at a time. Preserve partial outputs and guard
reports on failure;no deleting evidence,retry,extension or parameter change.
The original Stage2 STOP/gate/artifacts remain unchanged after every action.
