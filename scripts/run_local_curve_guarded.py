"""Future launcher with sleep-aware wall enforcement. Does not resume or auto-run.

The frozen scientific driver is kept byte-identical. Explicit new output paths
are required. A separate user decision is needed to continue the interrupted fit.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading

import psutil
import local_curve_experiment as experiment
from pn2.inputs.node_cache import digest, write_once
from pn2.training.local_curve_control import wait_wall
from pn2.training.process_control import terminate_owned_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['timing', 'train'])
    parser.add_argument('--arm', choices=['dense', 'transformer'], required=True)
    parser.add_argument('--rung', choices=['4', '16'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c = experiment.control()
    output = experiment.output_path(args.output)
    if output.exists():
        raise FileExistsError('Existing evidence must not be overwritten')
    logs = output.with_name(output.name+'_wall_guard')
    logs.mkdir(parents=True, exist_ok=False)
    timeout = c['timing_wall_seconds'] if args.action == 'timing' else c['rungs'][args.rung]['wall_seconds_per_arm']
    command = [sys.executable, '-B', str(Path(experiment.__file__).resolve()), args.action,
               '--worker', '--arm', args.arm, '--rung', args.rung, '--output', str(output)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (logs/'console.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', creationflags=flags)
        owner = psutil.Process(process.pid)
        def relay():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
        reader = threading.Thread(target=relay, daemon=True)
        reader.start()
        try:
            receipt = wait_wall(process, timeout, lambda: terminate_owned_tree(owner))
        except BaseException:
            terminate_owned_tree(owner)
            process.wait(timeout=5)
            raise
        finally:
            reader.join(timeout=10)
        if reader.is_alive():
            raise RuntimeError('Owned worker output pipe did not close')
    receipt.update(command=command, output=str(output),
        guard_sources={str(p.relative_to(experiment.ROOT)): digest(p) for p in (
            Path(__file__).resolve(), experiment.ROOT/'src/pn2/training/local_curve_control.py')})
    write_once(logs/'report.json', receipt)
    if receipt['status'] != 'completed':
        raise SystemExit(f'Guarded worker {receipt["status"]}; no automatic continuation')


if __name__ == '__main__':
    main()
