from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PointNeuronConfig:
    vaa3d_path: Path
    gold166_root: Path


def load_config(path: str | Path) -> PointNeuronConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return PointNeuronConfig(
        vaa3d_path=Path(payload["vaa3d_path"]),
        gold166_root=Path(payload.get("gold166_root", "data/gold166")),
    )

