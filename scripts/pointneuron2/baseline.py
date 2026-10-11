"""Prepare a neuron-disjoint baseline and optionally train it from scratch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pointneuron.data.splits import SplitRatios, validate_split_file, write_split_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path, help="Original checkout containing existing cache manifests and NPZ records.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pointneuron2/proposal_baseline.json")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "tmp/pointneuron2")
    parser.add_argument("--train", action="store_true", help="Train the fresh baseline after preparation.")
    parser.add_argument("--smoke", action="store_true", help="Train one epoch with two training batches and full validation in a separate smoke directory.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.artifact_root.resolve()
    manifest = root / config["cache_manifest"]
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    output = args.output_root.resolve()
    split_path = output / "splits" / f"proposal_neuron_seed{config['seed']}.json"
    manifest_records = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    missing = [r["path"] for r in manifest_records if not (root / r["path"]).is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} cache records; first: {missing[0]}")
    # Existing partitions are immutable. Compare to the intended provenance.
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        if any([
            payload.get("identity_rule") != "strip_legacy_patch_suffix_v1",
            payload.get("cache_manifest_sha256") != manifest_hash,
            payload.get("seed") != config["seed"],
            payload.get("group_by") != config["group_by"],
            payload.get("record_root") != str(root),
            payload.get("ratios") != config["split_ratios"],
            payload.get("dataset_id") != config["dataset_id"],
        ]):
            raise ValueError("Existing split provenance differs; choose a new output root")
        validate_split_file(split_path, require_neuron_split=True)
    else:
        write_split_file(manifest, split_path, ratios=SplitRatios(**config["split_ratios"]), seed=config["seed"], group_by=config["group_by"], record_root=root, dataset_id=config["dataset_id"])
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        payload["cache_manifest_sha256"] = manifest_hash
        split_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"split_file: {split_path}", flush=True)
    print(f"neurons/groups: {payload['group_counts']}", flush=True)
    print(f"patches: {payload['counts']}", flush=True)
    if not (args.train or args.smoke):
        return 0
    run_root = output / ("baseline_smoke" if args.smoke else "baseline_seed" + str(config["seed"]))
    if (run_root / "run.json").exists():
        raise ValueError("Run already exists; choose a new output root to avoid overwriting results")
    training = dict(config["training"])
    if args.smoke:
        training.update(epochs=1, limit_batches=2)
    training.update(seed=config["seed"], split_file=str(split_path), checkpoint=str(run_root / "proposal.pt"), require_neuron_split=True)
    command = [sys.executable, "-u", str(REPO_ROOT / "scripts/train_proposal.py")]
    for key, value in training.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    run_root.mkdir(parents=True, exist_ok=True)
    source_files = sorted((REPO_ROOT / "src/pointneuron").rglob("*.py")) + [REPO_ROOT / "scripts/train_proposal.py", Path(__file__).resolve()]
    source_hashes = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    run = {"status": "running", "smoke_only": args.smoke, "config": config, "command": command, "cache_manifest_sha256": manifest_hash, "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(), "source_sha256": source_hashes, "python": sys.version}
    run_path = run_root / "run.json"
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    try:
        with (run_root / "training.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        run.update(status="completed" if return_code == 0 else "failed", return_code=return_code)
    except BaseException:
        if "process" in locals() and process.poll() is None:
            process.terminate()
            process.wait()
        run["status"] = "interrupted"
        raise
    finally:
        run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
