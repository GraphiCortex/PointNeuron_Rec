"""CPU-only saved-index storage audit; no model edits, optimizer or GPU execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def audit():
    import torch

    root = Path(__file__).resolve().parents[1]
    source = root.parent / "pointneuron2/src/pointneuron/models/pointnext_ops.py"
    expected = "968fd3af88446d92c2dc3295eb462362794910f829ee755a61474b1b3fb7886d"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Historical operator source changed; review before auditing")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("pinned_pointnext_ops", source)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    queries, supports, chunk = 1024, 256, 64
    query = torch.rand(1, queries, 3)
    support = torch.rand(1, supports, 3)
    features = torch.rand(1, supports, 8, requires_grad=True)
    saved = []

    def pack(value):
        saved.append({"shape": list(value.shape), "dtype": str(value.dtype),
                      "logical_bytes": value.numel()*value.element_size(),
                      "storage_bytes": value.untyped_storage().nbytes(),
                      "pointer": value.untyped_storage().data_ptr()})
        return value

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        output = ops.interpolate_features(query, support, features, chunk_size=chunk)
        output.square().mean().backward()
    indices = [row for row in saved if row["dtype"] == "torch.int64"
               and row["shape"] == [1, chunk, 3]]
    unique = {row["pointer"]: row["storage_bytes"] for row in indices}
    logical = sum(row["logical_bytes"] for row in indices)
    backing = sum(unique.values())
    assert output.shape == (1, queries, 8)
    assert bool(torch.isfinite(output).all()) and bool(torch.isfinite(features.grad).all())
    assert len(indices) == len(unique) == queries // chunk
    assert logical == queries*3*8 and backing == queries*supports*8
    return {"status": "confirmed_full_sort_storage_retained", "device": "cpu",
            "torch": torch.__version__, "source_sha256": expected,
            "queries": queries, "supports": supports, "channels": 8, "chunk_size": chunk,
            "output_shape": list(output.shape), "finite_output_and_gradient": True,
            "saved_index_tensors": len(indices), "unique_index_storages": len(unique),
            "logical_index_bytes": logical, "backing_index_storage_bytes": backing,
            "one_saved_index_tensor": {key: value for key, value in indices[0].items() if key != "pointer"},
            "final_restoration_full_sort_storage_bytes_by_input_budget": {
                str(n): n*(n//4)*8 for n in (8192, 16384, 32768, 65536)},
            "scope": "CPU storage behavior verified; large-budget values are source-derived extrapolations, not GPU measurements"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=False)
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
