# New configurations

Fresh benchmark manifests and reviewed experiment configurations belong here.
No fresh training recipe is selected yet.

[sampler_learnability_inputs.json](sampler_learnability_inputs.json) fixes the
two-arm, four-training-box input preparation for the
[learned comparison protocol](../docs/sampler_learnability_protocol.md).
It preserves existing 2048-slot packets and discloses repeated control points.
It is not an executable model/optimizer recipe and does not authorize training.

[phase1_sampling_probe.json](phase1_sampling_probe.json) freezes the three-arm
CPU input diagnostic, not a training/product policy. Its
[protocol](../docs/phase1_sampling_protocol.md) explains the scales, count budgets,
controls, measurements and stopping rules.
[phase1_cpu_requirements.txt](phase1_cpu_requirements.txt) declares direct runtime
pins; [phase1_cpu_lock.txt](phase1_cpu_lock.txt) records the complete installed
dependency set for Python 3.11 on Windows. Both belong to the isolated new runtime.

[phase1_qualification_cases.json](phase1_qualification_cases.json) fixes four
additional image-only training cases, aliases 22/75/85/126, using the unchanged
sampling recipe. The [qualification plan and resource preflight](../docs/phase1_qualification_plan.md)
record why whole-volume generation is currently held. No clouds or training are
created by this configuration.

[development_benchmark.json](development_benchmark.json) lists seven existing
development identities, explicit image/reference paths, native voxel units and
training-exposure metadata. Identity joins use stable sample IDs. Its source
inventory has 163 records and must not be indexed as the filtered 155-record
inventory. File existence was checked; image decoding, calibration and source
content provenance remain separate work.

[training_inventory.json](training_inventory.json) lists the provisional 33
training identities and explicit source paths. It is source metadata, not a
cache or executable training recipe. The development-only evaluator rejects its
training partition. No image/reference contents were read to create it.

[input_tiling.example.json](input_tiling.example.json) declares a complete
voxel-grid example for input audits; it is not a frozen model-preparation
recipe. [reused_components.json](reused_components.json) pins the reused decoder
source and destination hashes, including explicit LF normalization.

[phase1_raw_review.json](phase1_raw_review.json) lists two original training
images without reference fields for the raw inspection script. No segmentation
or sampling policy is selected by this file.

The PN1 cloud-control config is removed from active source, preserved in commit
`48519ff` with its [historical result](../docs/reference_forward_interface.md).
Its fixed cutoff and point budget are not PN2 defaults.

The retained [historical neuron partition](../../pointneuron2/configs/pointneuron2/partition_seed0.json)
is reference metadata for the next benchmark specification. Its 33/7/8 groups
are disjoint within PN2, but all 48 neurons appeared in historical PN1 training.
Old cache paths must not be assumed to be valid fresh-workspace inputs.
