# PointNeuron
An attempt at reconstructing PointNeuron Reconstruction Pipeline using Gold dataset.

## Data foundation

The first milestone builds a training-ready Gold166 manifest by pairing each raw
volume with one validated SWC reconstruction.

```powershell
python scripts\check_environment.py --config configs\local.json
python scripts\build_gold166_manifest.py --root data\gold166 --output tmp\gold166_manifest.json --validate-swc
```

Create `configs/local.json` from `configs/local.example.json` and set the local
Vaa3D executable path before running the checks.

SWC priority is: `*swc_sorted.swc`, then stamped SWCs, then base SWCs. The
scanner falls back to a lower-priority SWC when a preferred file is structurally
invalid.
