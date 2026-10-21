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

The user authorized starting Milestone A: a common reconstruction evaluator,
explicit development manifest and repaired-PN1 reference specification.
This is not authorization to launch a new GPU training run automatically.
