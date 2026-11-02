"""Stop only this controller's worker tree, including Windows venv redirectors."""
import os
import subprocess
import threading

import psutil


def terminate_owned_tree(owner):
    try:children=owner.children(recursive=True)
    except psutil.NoSuchProcess:children=[]
    targets=list(reversed(children))+[owner]
    for process in targets:
        try:process.kill()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(targets,timeout=5)
    if alive:raise RuntimeError(f'Could not stop owned worker processes: {[p.pid for p in alive]}')


def launch_worker(command,folder,timeout):
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (folder/'console.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,text=True,encoding='utf-8',errors='replace',creationflags=flags)
        owner=psutil.Process(process.pid)
        def relay():
            try:
                for line in process.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
            except (OSError,ValueError):pass
        reader=threading.Thread(target=relay,daemon=True);reader.start()
        try:
            code=process.wait(timeout=timeout);status='completed' if code==0 else 'failed'
        except subprocess.TimeoutExpired:
            terminate_owned_tree(owner);code=process.wait(timeout=5);status='timeout'
        except BaseException:
            terminate_owned_tree(owner);process.wait(timeout=5);raise
        finally:
            reader.join(timeout=10)
        if reader.is_alive():raise RuntimeError('Worker output pipe did not close; inspect owned worker termination')
    return code,status
