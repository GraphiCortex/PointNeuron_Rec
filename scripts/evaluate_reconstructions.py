"""Evaluate saved SWCs for explicit development identities; never run a model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pn2.evaluation.manifest import load_benchmark, load_predictions
from pn2.evaluation.runner import csv_rows, evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True,
                        help="Repeat for paired models; first is the baseline")
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSON report path; never overwritten")
    parser.add_argument("--csv", type=Path, help="Optional fresh flat per-case CSV path")
    parser.add_argument("--tolerances", type=float, nargs="+", default=[2., 6.],
                        help="Distances in each case's declared units; 2/6 are historical voxel settings, not universal physical tolerances")
    parser.add_argument("--maximum-step", type=float, default=.5, help="Arclength quadrature spacing in each case's declared units")
    parser.add_argument("--check-convergence", action="store_true", help="Also report half-step metrics/deltas; primary metric stays at requested step")
    args = parser.parse_args(argv)
    destinations = [path.resolve() for path in (args.output, args.csv) if path is not None]
    if len(set(destinations)) != len(destinations):
        parser.error("JSON and CSV destinations must differ")
    for path in destinations:
        if path.exists():
            parser.error(f"Preserve existing outputs; choose a fresh report path: {path}")
    benchmark = load_benchmark(args.benchmark)
    predictions = [load_predictions(path, benchmark) for path in args.predictions]
    report = evaluate(benchmark, predictions, tolerances=tuple(args.tolerances),
                      maximum_step=args.maximum_step, check_convergence=args.check_convergence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        rows = csv_rows(report)
        with args.csv.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Evaluated {report['declared_case_count']} explicit cases; no models executed.")
    for model in report["models"]:
        for group in model["macro"]:
            f1 = group["f1"]["mean"]
            print(f"{model['model_id']} tolerance={group['tolerance']:g} {group['units']} "
                  f"scorable={group['scorable_reference_cases']}/{group['declared_cases']} "
                  f"mean_F1={f1 if f1 is not None else 'unscorable'} statuses={group['prediction_status_counts']}")
    print(f"Report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
