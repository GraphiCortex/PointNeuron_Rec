# AGENTS.md

Project: faithful reimplementation of the original PointNeuron pipeline.

Rules:
- Rebuild the original PointNeuron first. Do not modernize architecture unless explicitly asked.
- Dataset: Gold166.
- Stay as close as possible to the paper’s preprocessing, module order, and training logic.
- Before changing thresholds, patch sizes, losses, or graph initialization, explain why.
- Keep code modular and readable.
- Prefer small diffs over large rewrites.
- For each task, update PROJECT_BRIEF.md if an assumption changes.