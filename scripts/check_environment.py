from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointneuron.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local PointNeuron environment paths.")
    parser.add_argument("--config", default="configs/local.json", help="Local JSON config path.")
    args = parser.parse_args()

    config = load_config(args.config)
    checks = {
        "vaa3d_path": config.vaa3d_path,
        "gold166_root": config.gold166_root,
    }

    missing = 0
    for key, path in checks.items():
        exists = path.exists()
        print(f"{key}: {path} ({'ok' if exists else 'missing'})")
        missing += 0 if exists else 1

    return 0 if missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

