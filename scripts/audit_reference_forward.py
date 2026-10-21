"""CPU image/cloud/proposal execution check; fresh untrained reference, no optimization."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--tiling", type=Path, required=True)
    parser.add_argument("--clouds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory, never overwritten")
    args = parser.parse_args(argv)
    import torch
    from pn2.reference.forward_audit import audit_reference_forward
    # One CPU thread makes a bounded local check reproducible; no GPU is selected.
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report = audit_reference_forward(args.cases, args.tiling, args.clouds, args.output_dir)
    finally:
        torch.set_num_threads(previous)
    print(f"Reference execution {report['status']}; fresh untrained parameters, no optimizer or reconstruction.")
    print(f"Report: {args.output_dir.resolve() / 'report.json'}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
