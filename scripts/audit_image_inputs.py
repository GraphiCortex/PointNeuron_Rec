"""Decode explicit image cases and inventory every tile on CPU; no model runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.inputs.audit import audit_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="Strict image-only JSON; no reference fields")
    parser.add_argument("--tiling", type=Path, required=True, help="Explicit voxel tile sizes and strides")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory; never overwritten")
    args = parser.parse_args(argv)
    report = audit_inputs(args.cases, args.tiling, args.output_dir)
    print(f"Input audit {report['status']}; no foreground selection or model executed.")
    print(f"Report: {args.output_dir.resolve() / 'report.json'}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
