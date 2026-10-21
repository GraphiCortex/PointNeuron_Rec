# Workspace layout

21 October 2026. The user requested PN2 as a sibling of the earlier worktree.

| Location under `C:\Users\Seywe\Documents\GitHub\PointNeuron` | Role | Branch |
|---|---|---|
| Original checkout root | PN1 source and dataset assets | `main` |
| `pointneuron2/` | Historical prototypes, reports, retained runtime and evidence archive | `pointneuron2-history` |
| `PN2/` | All fresh PN2 source, commands, configs, tests and reports | `pointneuron2` |

The fresh project root is `PN2/`, not `pointneuron2/PN2/`. It is a linked Git
worktree sharing history with the original repository, not a second independent
repository. Historical files removed from the new checkout remain in the old
reference checkout and in Git history. The original dataset and environment
have not moved.

From the fresh root, the retained Python executable is
`..\pointneuron2\tmp\pointneuron2\sampling_env\Scripts\python.exe`.
New generated artifacts belong in ignored `PN2/tmp/`. Parent/sibling source
reuse must be explicit; the shared evaluator is a standalone `pn2` package.

22 October clarification: PN1 serves conceptual comparison only. Original raw
images and the PN2 Phase 1 design are the active starting point. The initial
PN1-reference milestone was a scope detour; its reports/history remain, and
its source port has been removed from the active tree. The image-only I/O,
tiling and evaluator are reusable. No GPU training is authorized by setup.
