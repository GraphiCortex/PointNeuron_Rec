from pn2.training.local_curve_control import wait_wall


class Process:
    def __init__(self, code=None):
        self.code = code
        self.stopped = False

    def poll(self):
        return self.code

    def stop(self):
        self.stopped = True
        self.code = -1

    def wait(self, timeout):
        assert self.stopped
        return self.code


def clock(values):
    return iter(values).__next__


def test_large_wall_jump_stops_owned_worker_without_new_wait():
    process = Process()
    sleeps = []
    result = wait_wall(process, 10, process.stop, wall=clock([100, 101, 20000]),
                       monotonic=clock([0, 1, 2]), sleep=sleeps.append)
    assert process.stopped
    assert result['status'] == 'wall_limit'
    assert sleeps == [.5]


def test_monotonic_jump_also_stops_worker_if_wall_clock_moves_back():
    process = Process()
    result = wait_wall(process, 10, process.stop, wall=clock([100, 90]),
                       monotonic=clock([0, 20]), sleep=lambda _: None)
    assert process.stopped and result['status'] == 'wall_limit'


def test_completion_discovered_after_limit_is_not_automatic_pass():
    process = Process(code=0)
    result = wait_wall(process, 10, process.stop, wall=clock([100, 120]),
                       monotonic=clock([0, 20]), sleep=lambda _: None)
    assert not process.stopped
    assert result['status'] == 'wall_limit'


def test_normal_success_does_not_terminate_worker():
    process = Process(code=0)
    result = wait_wall(process, 10, process.stop, wall=clock([100, 101]),
                       monotonic=clock([0, 1]), sleep=lambda _: None)
    assert not process.stopped
    assert result['status'] == 'completed'
