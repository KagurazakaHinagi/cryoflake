"""Allocation bounds for large validation inputs and pending worker tasks."""

import threading
import tracemalloc

import numpy as np
import pytest

from cryoflake import common


@pytest.mark.parametrize("order", ["contiguous", "fortran", "strided"])
def test_finite_check_uses_bounded_workspace(order):
    values = np.ones(8 * 1024 * 1024, dtype=np.float32)
    if order == "fortran":
        values = values.reshape(2048, 4096).T
    elif order == "strided":
        values = values[::2]

    tracemalloc.start()
    try:
        assert common.array_all_finite(values)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # A full boolean mask would require 4–8 MiB. Iteration and its optional
    # strided input buffer stay bounded independently of the array's size.
    assert peak_bytes < 2 * 1024 * 1024


@pytest.mark.parametrize(
    "values",
    [
        np.empty((0, 3)),
        np.asarray(1.0),
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([1 + 2j, complex(np.inf, 1)]),
        np.asarray([[np.nan, 1], [2, 3]]).T,
        np.arange(600_000, dtype=np.float32)[::-2],
    ],
)
def test_chunked_finite_check_matches_numpy(values):
    assert common.array_all_finite(values) == bool(np.isfinite(values).all())


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_finite_check_visits_elements_after_the_first_chunk(invalid):
    values = np.ones(600_000, dtype=np.float32)
    values[-1] = invalid
    assert not common.array_all_finite(values)


def test_pending_parallel_tasks_do_not_allocate_a_queue_entry_per_task(monkeypatch):
    tasks = list(range(30_000))
    worker_started = threading.Event()
    release_workers = threading.Event()
    failures = []
    result = []

    def work(value):
        worker_started.set()
        assert release_workers.wait(timeout=5)
        return value

    def run():
        try:
            result.extend(
                common.run_tasks_with_progress(
                    tasks, work, 2, "Testing", wait_for_workers_on_error=True
                )
            )
        except BaseException as exc:
            failures.append(exc)

    class QuietProgress:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_postfix(self, **kwargs):
            pass

        def update(self, *args):
            pass

    monkeypatch.setattr(common, "tqdm", QuietProgress)
    runner = threading.Thread(target=run)
    tracemalloc.start()
    try:
        runner.start()
        assert worker_started.wait(timeout=5)
        _, pending_peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        release_workers.set()
        runner.join(timeout=10)

    assert not runner.is_alive()
    assert not failures
    assert result == tasks
    # Task and result reference arrays need about 16 bytes/task. An eager
    # queue of (index, task) records adds several times that amount.
    assert pending_peak_bytes < len(tasks) * 32 + 128 * 1024
