"""Allocation bounds for large validation inputs and pending worker tasks."""

import threading
import tracemalloc

import numpy as np
import pytest

from cryoflake import common
from cryoflake.preprocess.reuse import _coordinates_have_grid_targets


@pytest.mark.parametrize("target_in_last_row", [False, True])
def test_grid_target_check_uses_bounded_workspace_and_checks_last_chunk(
    target_in_last_row,
):
    coordinates = np.full((1_000_000, 3), 10.0, dtype=np.float32)
    coordinates[0] = np.nan
    coordinates[1] = np.finfo(np.float32).max
    if target_in_last_row:
        coordinates[-1] = [1, 1, 1]
    tracemalloc.start()
    try:
        found = _coordinates_have_grid_targets(
            (coordinates,), (4, 4, 4), np.zeros(3, dtype=np.float32), 1.0
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert found is target_in_last_row
    # The old filtered coordinate copy alone needed nearly 12 MB. Chunking
    # bounds all temporary coordinate and boolean arrays independently of N.
    assert peak_bytes < 4 * 1024 * 1024


@pytest.mark.parametrize(
    "coordinate, expected",
    [
        ([0, 0, 0], True),
        ([3.999, 2.999, 1.999], True),
        ([4, 0, 0], False),
        ([0, 3, 0], False),
        ([0, 0, 2], False),
        ([-0.001, 0, 0], False),
    ],
)
def test_grid_target_check_preserves_xyz_bounds_for_zyx_grid(coordinate, expected):
    assert (
        _coordinates_have_grid_targets(
            (np.asarray([coordinate], dtype=np.float32),),
            (2, 3, 4),
            np.zeros(3, dtype=np.float32),
            1.0,
        )
        is expected
    )


@pytest.mark.parametrize(
    "coordinate, origin, voxel_size, expected",
    [
        ([0, 0, 0], [0, 0, 0], 1e-100, True),
        ([0, 0, 0], [0, 0, 0], np.nextafter(0.0, 1.0), True),
        ([-1, 0, 0], [0, 0, 0], 1e100, False),
        ([-1e-40, 0, 0], [0, 0, 0], 1e30, False),
        ([1, 0, 0], [0, 0, 0], 1e-100, False),
        ([1e38, 0, 0], [-1e38, 0, 0], 1e38, True),
        ([3e38, 0, 0], [-3e38, 0, 0], 1e39, True),
        ([1, 0, 0], [0, 0, 0], 1e308, True),
    ],
)
def test_grid_target_check_preserves_finite_spacing_range(
    coordinate, origin, voxel_size, expected
):
    with np.errstate(all="raise"):
        assert (
            _coordinates_have_grid_targets(
                (np.asarray([coordinate], dtype=np.float32),),
                (4, 4, 4),
                np.asarray(origin, dtype=np.float32),
                voxel_size,
            )
            is expected
        )


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
