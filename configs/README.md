# New configurations

Fresh benchmark manifests and reviewed experiment configurations belong here.
No fresh training recipe is selected yet.

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

[reference_cloud_control.example.json](reference_cloud_control.example.json)
names the inherited image-max-fraction/uniform-selection reference control.
It binds cloud behavior explicitly and contains no supervision or optimizer
settings. The [real forward check](../docs/reference_forward_interface.md)
remains unsupported on singleton clouds; this is not a promoted PN2 policy.

The retained [historical neuron partition](../../pointneuron2/configs/pointneuron2/partition_seed0.json)
is reference metadata for the next benchmark specification. Its 33/7/8 groups
are disjoint within PN2, but all 48 neurons appeared in historical PN1 training.
Old cache paths must not be assumed to be valid fresh-workspace inputs.
