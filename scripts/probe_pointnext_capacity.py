"""Bounded, untrained execution probe of the pinned historical native encoder.

No SWC, real-image selection, optimizer, checkpoint, or model-quality metric.
Each GPU budget gets a fresh process; timeout/OOM stops the sweep explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    # Write-once files avoid replacing a file another Windows process is reading.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    source = (ROOT / config["historical_source"]).resolve()
    for relative, expected in config["source_sha256"].items():
        actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Historical diagnostic source changed: {relative}")
    return config, source


def imports(source):
    # Read-only source reuse is explicit; suppress bytecode writes in history.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import torch
    from pointneuron.models.pointnext_native import PointNextNativeEncoder
    from pointneuron.models.pointnext_ops import ball_query_indices
    return torch, PointNextNativeEncoder, ball_query_indices


def pack(torch, xyz, intensity, *, center, scale, image_min, image_max):
    """Candidate diagnostic adapter; raw XYZ/intensity remain outside the model."""
    if scale <= 0 or image_max < image_min:
        raise ValueError("Invalid normalization metadata")
    local = (xyz - xyz.new_tensor(center)) / scale
    normalized = torch.zeros_like(intensity) if image_max == image_min else (
        (intensity - image_min) / (image_max - image_min))
    if not bool(torch.isfinite(normalized).all()) or bool(((normalized < 0) | (normalized > 1)).any()):
        raise ValueError("Intensity outside declared source-image range")
    return local, torch.cat((local, normalized[..., None]), -1)


def cloud(torch, count, config, device):
    side = config["box_side"]
    generator = torch.Generator().manual_seed(config["seed"])
    # Nested unique source-grid supports, fixed random order independent of intensity.
    ids = torch.randperm(side**3, generator=generator)[:count]
    xyz = torch.stack((ids % side, (ids // side) % side, ids // side**2), -1).float()[None].to(device)
    # Synthetic image I(x,y,z)=x+2y+3z, known full-image range [0,6*(side-1)].
    intensity = (xyz * xyz.new_tensor([1, 2, 3])).sum(-1)
    return pack(torch, xyz, intensity, center=[(side-1)/2]*3,
                scale=config["coordinate_scale"], image_min=0, image_max=6*(side-1))


def check(torch, Encoder, ball, config):
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    xyz = torch.tensor([[[0., 0., 0.], [127., 127., 127.]]])
    p, a = pack(torch, xyz, torch.tensor([[10., 110.]]), center=[63.5]*3,
                scale=64, image_min=10, image_max=110)
    assert torch.equal(a[..., :3], p) and a[0, 0, 3] == 0 and a[0, 1, 3] == 1
    _, constant = pack(torch, xyz, torch.ones(1, 2)*7, center=[63.5]*3,
                       scale=64, image_min=7, image_max=7)
    assert torch.equal(constant[..., 3], torch.zeros(1, 2))
    support = torch.tensor([[[0., 0., 0.], [6.3/64, 0., 0.], [6.5/64, 0., 0.]]])
    members = ball(support, support[:, :1], .1, 32)
    assert set(members.flatten().tolist()) == {0, 1}
    # First-K in support order, including a deliberately distant first member.
    x = torch.linspace(.09, 0, 40)
    support = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), -1)[None]
    members = ball(support, torch.zeros(1, 1, 3), .1, 32)
    assert members.flatten().tolist() == list(range(32))
    reverse = support.flip(1)
    reversed_members = ball(reverse, torch.zeros(1, 1, 3), .1, 32)
    assert set(support[0, members.flatten(), 0].tolist()) != set(reverse[0, reversed_members.flatten(), 0].tolist())
    model = Encoder()
    positions, attributes = cloud(torch, 256, config, "cpu")
    output = model(positions, attributes)
    assert output.shape == (1, 256, 32) and bool(torch.isfinite(output).all())
    output.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
    assert any(bool((g != 0).any()) for g in gradients)
    return {"status": "passed", "device": "cpu", "points": 256,
            "shape": list(output.shape), "checks": ["coordinate/intensity adapter", "constant image",
            "6.3-voxel included / 6.5-voxel excluded at r=6.4", "first-32 support-order grouping",
            "finite forward and parameter backward without optimizer"], "torch": torch.__version__}


def worker(args, config, torch, Encoder):
    directory = Path(args.output)
    torch.manual_seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback for the GPU sweep")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = Encoder().to(device)
    observed_shapes = {}
    for stage, layer in enumerate(model.abstraction, 1):
        def record_shape(module, inputs, output, stage=stage):
            observed_shapes[str(stage)] = [list(value.shape) for value in output]
        layer.register_forward_hook(record_shape)
    xyz, attributes = cloud(torch, args.points, config, device)
    free, total = torch.cuda.mem_get_info(device)
    report = {"status": "completed", "points": args.points, "batch": 1, "dtype": "float32",
              "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "free_bytes_after_setup": free, "total_bytes": total,
              "parameter_count": sum(p.numel() for p in model.parameters()), "modes": {}}
    n = args.points
    report["hierarchy_points"] = [n // 4**i for i in range(5)]
    for mode in ("eval_forward", "train_forward_backward"):
        write(directory / f"{mode}_started.json", {"phase": mode, "points": args.points})
        training = mode == "train_forward_backward"
        model.train(training)  # Ordinary reference BN; no cropnorm policy adoption.
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for iteration in range(config["warmups"] + config["repeats"]):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.set_grad_enabled(training):
                output = model(xyz, attributes)
                if training:
                    output.square().mean().backward()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if output.shape != (1, args.points, 32) or not bool(torch.isfinite(output).all()):
                raise RuntimeError("Invalid encoder output")
            if training and any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError("Invalid encoder parameter gradient")
            if iteration >= config["warmups"]:
                durations.append(elapsed)
            print(f"N={args.points} {mode} iteration={iteration+1} seconds={elapsed:.3f}", flush=True)
            del output
        row = {"seconds": durations, "mean_seconds": sum(durations)/len(durations),
               "output_shape": [1, args.points, 32],
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        report["modes"][mode] = row
        write(directory / f"{mode}_result.json", row)
    report["observed_abstraction_shapes"] = observed_shapes
    for stage, shapes in observed_shapes.items():
        expected = [[1, args.points // 4**int(stage), 3],
                    [1, args.points // 4**int(stage), 32 * 2**int(stage)]]
        if shapes != expected:
            raise RuntimeError(f"Unexpected stage {stage} shapes: {shapes}")
    write(directory / "report.json", report)


def sweep(args, config):
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "config.json", config)
    rows, stopped = [], None
    for count in config["budgets"]:
        if stopped:
            rows.append({"points": count, "status": f"not_run_after_{stopped}"})
            continue
        target = directory / str(count)
        target.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--action", "worker",
                   "--config", str(Path(args.config).resolve()), "--points", str(count), "--output", str(target)]
        started = time.perf_counter()
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with (target / "console.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", env=environment)
            def relay():
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
            reader = threading.Thread(target=relay, daemon=True)
            reader.start()
            try:
                code = process.wait(timeout=config["worker_timeout_seconds"])
                status = "completed" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code, status = process.returncode, "timeout"
            except KeyboardInterrupt:
                process.kill()
                process.wait()
                raise
            finally:
                reader.join()
        result_path = target / "report.json"
        row = json.loads(result_path.read_text()) if result_path.exists() else {
            "points": count, "status": "failed_missing_report" if status == "completed" else status}
        if code != 0 and row["status"] == "completed":
            row["status"] = "failed_process"
        row.update(exit_code=code, wall_seconds=time.perf_counter()-started)
        write(target / "process_result.json", row)
        rows.append(row)
        print(f"BUDGET N={count} status={row['status']} wall={row['wall_seconds']:.1f}s", flush=True)
        if row["status"] != "completed":
            stopped = row["status"]
    write(directory / "report.json", {"scope": "Untrained native encoder only", "rows": rows,
                                      "status": "stopped" if stopped else "completed"})
    print(f"Report: {directory / 'report.json'}", flush=True)
    return 1 if stopped else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["check", "sweep", "worker"], required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/pointnext_capacity_probe.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", type=int)
    args = parser.parse_args()
    config, source = load(args.config)
    if args.action == "sweep":
        return sweep(args, config)
    torch, Encoder, ball = imports(source)
    if args.action == "check":
        directory = Path(args.output)
        directory.mkdir(parents=True, exist_ok=False)
        result = check(torch, Encoder, ball, config)
        result["source_sha256"] = config["source_sha256"]
        write(directory / "report.json", result)
        print(json.dumps(result, indent=2))
        return 0
    if args.points not in config["budgets"]:
        raise ValueError("Worker point count must match a declared budget")
    try:
        worker(args, config, torch, Encoder)
    except Exception as error:
        status = "cuda_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed"
        write(Path(args.output) / "report.json", {"status": status, "points": args.points,
                                               "error": str(error), "traceback": traceback.format_exc()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
