"""Visible, sleep-aware owned subprocess guard; no automatic retry."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import psutil

from pn2.inputs.node_cache import write_once
from .local_curve_control import wait_wall
from .process_control import terminate_owned_tree


def run(command, logs, seconds, metadata):
    logs = Path(logs); logs.mkdir(parents=True, exist_ok=False)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (logs/'console.log').open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', creationflags=flags)
        owner = psutil.Process(process.pid)
        write_once(logs/'owner.json', {'pid': process.pid, 'created': owner.create_time(), 'command': command})
        def relay():
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line, end='', flush=True)
        thread = threading.Thread(target=relay, daemon=True); thread.start()
        wall, mono = time.time(), time.perf_counter()
        try:
            receipt = wait_wall(process, seconds, lambda: terminate_owned_tree(owner))
        except BaseException as error:
            terminate_owned_tree(owner); process.wait(timeout=5)
            receipt = {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'guard_error',
                       'error': type(error).__name__, 'exit_code': process.returncode,
                       'elapsed_seconds': max(time.time()-wall, time.perf_counter()-mono)}
        finally:
            thread.join(timeout=10)
        receipt.update(metadata, maximum_seconds=seconds)
        write_once(logs/'report.json', receipt)
    if thread.is_alive() or receipt['status'] != 'completed':
        raise RuntimeError(f'Owned worker {receipt["status"]}; preserve evidence; no automatic retry')
    return receipt
