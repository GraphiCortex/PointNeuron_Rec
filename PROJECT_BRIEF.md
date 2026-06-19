# PROJECT_BRIEF

Goal:
Faithful reproduction of the original PointNeuron pipeline on Gold166.

Current phase:
Stage 1 — data audit, loader, voxel-to-point conversion, SWC pairing.

Immediate next task:
Build dataset inspection utilities and a loader that can:
1. find image/SWC pairs
2. report file counts and missing pairs
3. load one sample
4. convert foreground voxels to a point cloud
5. support debug visualization

Non-negotiables:
- Start paper-faithful.
- Do not redesign architecture yet.
- Keep every preprocessing choice explicit and easy to change later.