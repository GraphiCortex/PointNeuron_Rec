"""Compatibility entry point; PN 2.0 scripts now live in scripts/pointneuron2."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pointneuron2.sampling_probe import main

if __name__ == "__main__":
    raise SystemExit(main())
