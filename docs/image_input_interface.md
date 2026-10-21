# Image-only loading and whole-volume tile inventory

21 October 2026. First bounded implementation from the
[baseline integration plan](baseline_integration_plan.md). This is an input
execution interface. It does not prepare model clouds, train or reconstruct.

## Implemented behavior

The standalone `pn2.inputs` package accepts explicit image paths, decodes a
scalar Vaa3D raw/PBD volume and enumerates every spatial tile. It imports no
model, dataset scanner, SWC parser or evaluator. It cannot accept a reference
path as an input-case field. Existing reference-bearing inventories must first
be projected into the strict image-only schema below; they are not accepted
unchanged.

The verified `vaa3d-pbd-v2` decoder module was copied from the historical
checkout, with only CRLF-to-LF normalization. Operators remain unchanged.
[reused_components.json](../configs/reused_components.json) records the source
revision, original byte hash and destination LF hash. No historical source was
edited. PBD16 four/five-bit extensions remain explicitly unsupported.

The image-only loader hashes and decodes the same in-memory source-byte
snapshot. It records source/decoded byte hashes, dimensions, type, byte order
and reader version. No intensity transformation, spatial resampling or padding
is applied. The decoded array is read-only.

## Image cases and coordinates

```json
{
  "schema_version": 1,
  "cases": [
    {
      "case_id": "my-explicit-image",
      "image_path": "relative/path/to/image.v3dpbd",
      "partition": "development",
      "coordinates": {"mode": "native", "units": "voxel"}
    }
  ]
}
```

Paths resolve relative to the JSON file. Duplicate IDs, duplicate JSON keys,
unknown fields, nonfinite numbers and reference-bearing fields are rejected
before images are opened. Current allowed roles are `training`, `development`
and `synthetic`; reserved/test roles are rejected. The role is a caller
declaration, not proof that an arbitrary path is an eligible dataset identity.
Use reviewed source inventories when constructing actual run manifests.

Acquisition coordinates may instead be declared as
`{"mode":"spacing","units":"um","spacing_xyz":[sx,sy,sz]}` with positive
finite XYZ scales. This declares metadata; it does not verify calibration.
Array and tile coordinates remain native voxel indices. A later metric frame
uses `metric_xyz = native_xyz * spacing_xyz`. The loader applies no implicit
scaling or resampling, and makes no scalar physical-radius claim.

XYZ voxel `(x,y,z)` is array `[z,y,x]`; voxel centers have integer coordinates
with origin `(0,0,0)`. Tile boxes use inclusive starts and exclusive stops in
native XYZ. A tile-local point maps to global XYZ by adding `start_xyz`.
The report distinguishes native voxel units from declared metric units.

## Covering grid and tile accounting

The [example tiling config](../configs/input_tiling.example.json) declares exact
voxel sizes `[97,97,97]` and strides `[48,48,48]`. A 97-voxel axis has center
48, matching an inclusive radius-48 box. This is an interface example, not a
frozen R/N preparation policy or a universal physical scale.

Every axis starts at zero. Regular starts advance by the explicit stride, with
an additional final start at `max(0, dimension - tile_size)` if needed. Strides
larger than tile sizes are rejected because they leave gaps. The Cartesian
product covers the whole image; X varies fastest, then Y, then Z. A smaller
image uses its actual shape with one tile along that axis. There is no padding,
duplicate final tile, foreground-selected bounding box or tile-count cap.

Every tile is written to a JSONL inventory, including its ID, shape, bounds,
center, global origin, voxel count, raw nonzero count and intensity min/max.
`all_zero` and `nonzero_signal` describe numerical voxel values only. They do
not label background or neuronal foreground. No threshold, sparse-tile
rejection, cloud sampling or point duplication is performed.

The summary keeps every declared case. Successful cases are
`decoded_and_tiled`; missing/corrupt/unreadable inputs are `failed`, unsupported
formats/types/channels/extensions are `unsupported`. Overall status fails when
any case fails. A partial JSONL inventory remains visible after an expected
case failure. Fresh output directories are mandatory; previous reports are not
overwritten. Programming errors and interruptions still propagate; this is
not a checkpoint/resume runner.

`model_processed_tiles` is zero for every case. Complete **geometric tile
coverage** means that image indices were accounted for; it establishes neither
foreground retention nor branch coverage by future point clouds.

## Visible command

From the PN2 root, after creating the explicit image-only manifest:

```powershell
& ..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe .\scripts\audit_image_inputs.py `
  --cases .\tmp\my_input_check\images.json `
  --tiling .\configs\input_tiling.example.json `
  --output-dir .\tmp\my_input_check\audit
```

The `images.json` path is an input to prepare, not an existing model result.
The command uses CPU and writes `report.json` plus per-case tile JSONL files.
It writes no SWCs or training caches. Its reports are not model-output
manifests for the reconstruction evaluator.

## Verification and observed compatibility

The full fresh suite passed **123 tests**: 54 existing evaluator checks and
69 input checks. Fixtures cover all 256 packed PBD8 bytes with partial groups
and carry, PBD run boundaries, PBD16 signed codes and bit/byte boundaries,
both byte orders, corrupt/unsupported encodings, asymmetric XYZ orientation,
native/physical metadata separation, small images, final-boundary coverage,
input/output manifests and failure accounting. Tests open no dataset images
or references. A fresh subprocess confirms that input imports load neither
PyTorch nor the SWC evaluator.

A separate real compatibility check read the 43-byte headers of the **33
explicit training images**, then chose the smallest decoded byte count (case-ID
tie break) to bound a CPU execution check. It did not choose by reconstruction
results. Historical alias **32**, stable identity
`p_checked6_fruitfly_larvae_gmu/done_err_3_CL-I_MT_X_MYR-GFP_ddaE_MT-mCherry_membrane-GFP.czi_C_1`,
was decoded and inventoried:

| Observation | Value |
|---|---|
| Native XYZ dimensions | 1017 x 429 x 27 |
| Encoding/type | PBD, uint8, little endian |
| Decoded bytes | 11,779,911 |
| Written/expected tiles | 168 / 168 |
| First box | `[0,0,0]` to `[97,97,27]` |
| Final box | `[920,332,0]` to `[1017,429,27]` |
| Per-case decode/inventory time | 1.59 seconds on CPU, excluding process startup |
| Model-processed tiles | 0 |

Source SHA256:
`3b2cc2f0078d74c44875a4abbc69c74367b5ba9ff2d60538b898b22ff7510be7`.
Decoded-byte SHA256:
`48d6f60f3555f3a57b12da202cc5f8c32dd359250bbdaf4916cadc8793b998bc`.
The header preflight, strict image-case manifest and raw audit outputs are under
ignored `tmp/input_interface_20261021/`. No SWC contents, reserved images,
foreground rule, model inference or training were used. This is one real-file
compatibility result, not independent real-file verification of every decoder
mode or a reconstruction-quality measurement.

## Remaining work before a model run

The entire decoded image is currently held in memory, one case at a time.
Tile enumeration streams to disk, but image decoding is not out-of-core.
Decoded byte count is reported; peak process memory has not been measured and
can exceed that count because compressed/decoded buffers coexist.

Point-cloud preparation, short-cloud behavior, model normalization/reload,
proposal aggregation, geodesic memory/path integration and SWC export remain
to be ported and checked. The segment-based supervision proposal remains a
separate design decision before cache creation. There is no new learned
sampler, topology loss, foreground rule or training recipe in this port.
