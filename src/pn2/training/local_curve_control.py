"""Sleep-aware wall guard for future local-curve workers; old controllers unchanged."""
import time


def wait_wall(process, timeout, terminate, *, wall=time.time, monotonic=time.perf_counter,
              sleep=time.sleep, period=.5):
    """Check both clocks, not a single OS wait whose timeout can exclude sleep.

    Nothing can execute while the host is suspended. On resumption the guard
    checks the elapsed clocks before waiting again. A completed job discovered
    beyond its cap is also budget-limited, not automatically eligible to advance.
    ``terminate`` must act only on the process tree this caller created.
    """
    if timeout <= 0 or period <= 0:
        raise ValueError('Positive wall limit and polling period required')
    started_wall, started_mono = wall(), monotonic()
    while True:
        elapsed = max(wall()-started_wall, monotonic()-started_mono, 0.)
        code = process.poll()
        if elapsed >= timeout:
            if code is None:
                terminate()
                code = process.wait(timeout=5)
            return {'status': 'wall_limit', 'exit_code': code, 'elapsed_seconds': elapsed,
                    'maximum_seconds': timeout, 'started_epoch': started_wall}
        if code is not None:
            return {'status': 'completed' if code == 0 else 'failed', 'exit_code': code,
                    'elapsed_seconds': elapsed, 'maximum_seconds': timeout, 'started_epoch': started_wall}
        sleep(min(period, timeout-elapsed))
