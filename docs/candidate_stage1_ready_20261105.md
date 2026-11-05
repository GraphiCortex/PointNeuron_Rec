# Stage0 completed; Stage1 ready for local CPU launch

**Completed update,5 November:** Stage1 succeeded locally in7m34s console time,
460 training labels and611 candidate pools. Packet bindings verified; no
development labels/model/training/GPU. Do not rerun preparation below.
Read [Stage1 results and next feature-export command](candidate_stage1_results_20261105.md).
Preparation instructions below are preserved launch history.

Stage0 completed on151 development boxes/seven repeatedly inspected neurons in
575.09 seconds. The original union gate passed, with no source/threshold selected.

| Geometric oracle arm | F1 at2 | F1 at6 | Candidate nodes / edges |
|---|---:|---:|---:|
| Classical | 74.00% | 80.66% | 21,901 / 129,375 |
| Sato512 | 83.45% | 90.16% | 77,312 / 480,560 |
| Broad learned nodes | 85.33% | 93.19% | 108,092 / 675,107 |
| Union | 90.42% | 95.67% | 160,634 / 1,015,242 |

Union candidate-edge coverage at2 is88.49% for k6+inherited and90.44% for
k10+inherited. Primary oracle recall is83.88%, precision99.73%. Alias94 is
weakest (oracle F1_2 67.97%); it is disclosed, not a new veto or tuning source.
See [portable evidence](evidence/candidate_headroom_completed_20261105.json).

The oracle uses reference-assisted choices. It establishes useful geometric
headroom, not learnability, image-only accuracy, correct joins or final quality.
The broad head's failed reconstruction result remains failed. The union's
advantage does not prove the learned source is necessary: no source is adopted
from these diagnostic comparisons. The already agreed union stays fixed.

Read the [Stage1 contract](candidate_stage1_contract_20261105.md). Run first:

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '.\tmp\phase1_cpu_env\Scripts\python.exe' -B '.\scripts\prepare_candidate_selection.py'
```

CPU only, no dependency installation, no model/GPU/training. New work is27
missing classical extractions from retained guides,460 training unions/labels;
existing151 development unions are reused. Estimated several to tens of minutes,
UNMEASURED for this script; hard cap3600 seconds. Outputs:
`tmp/candidate_stage1_20261105/{candidate_manifest,label_manifest,report}.json`,
readable report.md, candidates, label packets and provenance. Guard console/
receipt live in sibling `candidate_stage1_20261105_guard_<UTC>`.
On failure/cap, preserve evidence and report it; no automatic retry/training.

After Stage1 succeeds and its counts are inspected, the inference-only feature
export command is available (CPU default, no training):

```powershell
Set-Location 'C:\Users\Seywe\Documents\GitHub\PointNeuron\PN2'
& '..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe' -B '.\scripts\export_candidate_pointnext.py' --device cpu
```

All611 image-only input packets, immutable broad checkpoint,32D PointNeXt
features and3NN candidate queries. CPU runtime UNMEASURED; allow several to
tens of minutes with hard3600-second cap. Output
`tmp/candidate_pointnext_20261105/complete.json` plus611 feature packets and
guard logs. Optional `--device cuda:0` is a user-launched inference choice,
not training; no assistant GPU launch is authorized or performed.
This command cannot run before completed Stage1. It does not launch trees/GAT.

Implemented fixture checks cover geometry/reference separation, training-only
label access, cache replay/mismatch rejection, adjacency/bounds, ambiguous
components/junctions, interpolation ties and extrapolation. These are execution
checks, not measured Stage1 counts or a selector result. No timed run was
launched by the assistant. GBT/GAT and remaining feature construction follow
inspection of these outputs, with the original final floors unchanged.
