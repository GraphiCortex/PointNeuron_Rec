# Benchmark identity repair — CPU,first scoring only

The official726,174,044-byte archive has already downloaded successfully. Reuse
it;do not download it again. Original extraction preserved96 outputs for only
three case identities,so benchmark scoring correctly wrote an incomplete-mapping
report and evaluated NO algorithms. Other completed diagnostics do not need reruns.

Observed archive naming differs from the original image folders:
`checked3_human_culturedcell_.../Series017/01_103.v3dpbd_app2new3.swc`.
The leading01 is a per-case file ordinal and103 is a dataset ID,not the algorithm.
Keeping those prefixes would split one implementation into different per-neuron
"algorithms". The repair resolves identity and implementation names before the
FIRST benchmark metric is computed;this is an execution/data-inventory fix,not
selection from benchmark accuracy.

Six approved cases have unique IDs in the authors' published scaling table:
SLP251,frog245,Series017103,mouse122,adult zebrafish276,Janelia52. The three
previously named cases independently corroborate251/103/52 in existing extracted
metadata. All96 earlier named-case files must be corroborated by the repair.
Other cases' SWC contents are not parsed/extracted;only archive headers are scanned.

Human in_house1 has a shared Cell1 image basename. Its chosen gold annotation
timestamp differs from the published ID256 annotation,so256 is NOT assigned from
the basename alone. Require exact normalized source-group/case-leaf identity:
`human_allen_confocal/in_house1`,excluding the independent-trial2/HO copy. If this
is absent,the action reports that seven-case mapping remains incomplete;it does
not exclude the case,replace the reference or fabricate complete scores. Candidate
human-family header names are retained to diagnose a remaining alias problem.

Implementation variants retain actual method/parameter suffixes but remove
ordinal/dataset prefixes. Filename seedXYZ coordinates are recorded separately;
seed-encoded mode remains a separate variant. Seed origin is unknown here,so
this optimized external collection is not certified as a reference-free independent
baseline. If a case/method has replicas,all contribute through mean primitive
length/matching counts before normal within-neuron/macro aggregation. No best
replicate is selected;individual replica results remain saved. These aggregation
rules are fixed before ANY benchmark scoring.

Coordinate/scoring rules are unchanged:native XYZ identity,no fitted registration,
>=99% inside native bounds,2/6-voxel segment scores and2-voxel-spaced skeleton
matching at2/6. Bounds do not prove subvoxel origin;results remain provisional
until coordinate overlays/provenance support it. All seven identities stay in
each method summary;missing/invalid/empty algorithm outputs score zero.
Primary results use the same151 boxes;whole-image scores are separate. Report
per implementation/per neuron,best fixed variant,median variant macro and per-
neuron best/median as descriptive context,not an adopted best-of composite.

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\stage2_cpu_env\Scripts\python.exe' -B '.\scripts\recover_bigneuron_benchmark.py'
```

CPU only,**two-hour hard cap**. Runtime unmeasured;archive scan plus FIRST scoring
may take minutes to hours. Output is a NEW write-once root
`tmp\bigneuron_benchmark_recovery_20261106`,with `inventory\manifest.json`,
`scoring\report.{json,md}` and an owned-process guard log. Original archive,
96 extracted files,incomplete-scoring report and all successful diagnostic
outputs remain unchanged. No model fitting,GPU,new development training labels,
editing implementation or Stage2 continuation. No automatic retry/extension.
