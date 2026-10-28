"""CPU-only exact parity and compact storage gate for the 64k execution replay."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import probe_pointnext_capacity as probe
from pointnext_compact_interpolation import make_interpolator


def check(config_path):
    config, source = probe.load(config_path)
    baseline = json.loads((probe.ROOT / "configs/pointnext_capacity_probe.json").read_text())
    assert config["interpolation_variant"] == "compact_indices_v1" and config["budgets"] == [65536]
    assert all(config[key] == value for key, value in baseline.items() if key not in ("purpose", "budgets"))
    torch, Encoder, _ = probe.imports(source)
    from pointneuron.models import pointnext_native as native, pointnext_ops as ops
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    corrected = make_interpolator(ops)
    rows = []
    fixtures = [("random", 1, 1024, 256, 64), ("tied", 2, 13, 7, 5),
                ("one_support", 1, 11, 1, 4), ("two_supports", 2, 11, 2, 4),
                ("three_supports", 1, 11, 3, 4)]
    for name, batch, queries, supports, chunk in fixtures:
        query = torch.rand(batch, queries, 3)
        support = torch.rand(batch, supports, 3)
        if name == "tied":
            query.zero_(); support.zero_()  # Stable support-index tie order.
        values = torch.rand(batch, supports, 8)
        results = []
        for function in (ops.interpolate_features, corrected):
            features = values.clone().requires_grad_()
            neighbors, saved = [], []
            gather = ops.gather_points
            def record_gather(features, indices):
                neighbors.append(indices.tolist())
                return gather(features, indices)
            def pack(value):
                if name == "random" and value.dtype == torch.int64 and list(value.shape) == [1, chunk, 3]:
                    saved.append((value.untyped_storage().data_ptr(), value.untyped_storage().nbytes()))
                return value
            with patch.object(ops, "gather_points", record_gather):
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                    output = function(query, support, features, chunk_size=chunk)
                    output.square().mean().backward()
            assert bool(torch.isfinite(output).all()) and bool(torch.isfinite(features.grad).all())
            results.append((output.detach(), features.grad, neighbors, sum(dict(saved).values())))
        assert torch.equal(results[0][0], results[1][0]), f"Forward mismatch: {name}"
        assert torch.equal(results[0][1], results[1][1]), f"Gradient mismatch: {name}"
        assert results[0][2] == results[1][2], f"Neighbor mismatch: {name}"
        row = {"fixture": name, "neighbors_equal": True, "outputs_exact": True, "feature_gradients_exact": True}
        if name == "random":
            old, new = results[0][3], results[1][3]
            assert old == queries*supports*8 and new == queries*3*8
            row.update(baseline_backing_bytes=old, corrected_backing_bytes=new)
        rows.append(row)
    # Whole encoder: same initial parameters, real historical operator bindings.
    initial = Encoder()
    positions, attributes = probe.cloud(torch, 256, config, "cpu")
    models = [copy.deepcopy(initial), copy.deepcopy(initial)]
    outputs = []
    original = native.interpolate_features
    try:
        for model, function in zip(models, (ops.interpolate_features, corrected)):
            native.interpolate_features = function
            model.train()
            output = model(positions, attributes)
            output.square().mean().backward()
            assert bool(torch.isfinite(output).all())
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            outputs.append(output.detach())
        assert torch.equal(outputs[0], outputs[1])
        assert all(torch.equal(a.grad, b.grad) for a, b in zip(models[0].parameters(), models[1].parameters()))
        assert all(torch.equal(models[0].state_dict()[key], value) for key, value in models[1].state_dict().items())
        for model, function in zip(models, (ops.interpolate_features, corrected)):
            native.interpolate_features = function
            model.eval()
            with torch.no_grad():
                outputs.append(model(positions, attributes))
        assert torch.equal(outputs[2], outputs[3])
    finally:
        native.interpolate_features = original
    return {"status": "passed", "device": "cpu", "torch": torch.__version__, "operator_fixtures": rows,
            "whole_encoder_256_points": {"train_outputs_exact": True, "all_parameter_gradients_exact": True,
                                         "state_and_bn_buffers_exact": True, "eval_outputs_exact": True},
            "historical_source_sha256": config["source_sha256"],
            "corrected_source_sha256": config["additional_source_sha256"],
            "scope": "No GPU or optimizer; exact CPU parity and retained-index storage only"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check(args.config)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    probe.write(directory / "report.json", result)
    print(json.dumps(result, indent=2))
