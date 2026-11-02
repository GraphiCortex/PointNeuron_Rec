"""A timeout must stop descendants, not merely a Windows venv launcher."""
import sys
from pathlib import Path

import psutil
from pn2.training.process_control import launch_worker


def test_owned_sleeping_child_tree_is_stopped_on_timeout(tmp_path):
    script="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"
    code,status=launch_worker([sys.executable,'-c',script],tmp_path,2)
    assert status=='timeout' and code!=0
    pid=int((tmp_path/'console.log').read_text().strip())
    assert not psutil.pid_exists(pid)
