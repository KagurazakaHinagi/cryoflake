"""Cross-pipeline helpers retain stable hashes and bounded worker behavior."""

import hashlib
import threading

import pytest

from cryoflake.common import (
    clean_token,
    finite_float,
    format_gib,
    integer_token,
    json_sha256,
    run_tasks_with_progress,
    sha256_file,
    validate_num_workers,
)


def test_sha256_file_handles_multiple_chunks_and_empty_files(tmp_path):
    path = tmp_path / "large.bin"
    payload = bytes(range(256)) * 9000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()


def test_json_hash_ignores_mapping_order_but_preserves_sequence_order():
    assert json_sha256({"b": [1, 2], "a": 3}) == json_sha256({"a": 3, "b": [1, 2]})
    assert json_sha256([1, 2]) != json_sha256([2, 1])


def test_common_scalar_conversions_handle_missing_and_invalid_values():
    assert clean_token(" ? ") == ""
    assert clean_token(" ALA ") == "ALA"
    assert finite_float("nan", default=1.5) == 1.5
    assert integer_token("?", "12", default=0) == 12
    assert format_gib(3 * 1024**3) == "3.0 GiB"


@pytest.mark.parametrize("value", [0, -1, "abc", None, True, 1.5, float("inf")])
def test_worker_count_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        validate_num_workers(value, "workers")


def test_parallel_worker_results_preserve_input_order_and_deliver_callbacks():
    first_started = threading.Event()
    second_finished = threading.Event()
    callbacks = []

    def process(value):
        if value == 0:
            first_started.set()
            assert second_finished.wait(timeout=2)
        else:
            assert first_started.wait(timeout=2)
            second_finished.set()
        return value * 3

    assert run_tasks_with_progress(
        [0, 1], process, 2, "Testing", result_callback=callbacks.append
    ) == [0, 3]
    assert sorted(callbacks) == [0, 3]


@pytest.mark.parametrize("num_workers", [1, 2])
def test_worker_results_preserve_none_values(num_workers):
    callbacks = []

    assert run_tasks_with_progress(
        [0, 1, 2],
        lambda value: None if value % 2 == 0 else value,
        num_workers,
        "Testing",
        result_callback=callbacks.append,
    ) == [None, 1, None]
    assert callbacks.count(None) == 2
    assert callbacks.count(1) == 1


@pytest.mark.parametrize("removed_keyword", ["worker_fn", "success_fn", "result_fn"])
def test_task_runner_rejects_removed_callback_keywords(removed_keyword):
    with pytest.raises(TypeError, match=removed_keyword):
        run_tasks_with_progress(
            [], lambda value: value, 1, "Testing", **{removed_keyword: lambda: None}
        )


@pytest.mark.parametrize("num_workers", [0, -1, 1.5, float("inf")])
def test_task_runner_rejects_invalid_worker_count_before_starting(num_workers):
    with pytest.raises(ValueError, match="num_workers must be a positive integer"):
        run_tasks_with_progress([1], lambda value: value, num_workers, "Testing")


def test_worker_failure_cancels_the_remaining_queue():
    stop = threading.Event()
    seen = []

    def process(value):
        seen.append(value)
        raise RuntimeError("failed task")

    with pytest.raises(RuntimeError, match="failed task"):
        run_tasks_with_progress([1, 2, 3], process, 1, "Testing", stop_event=stop)
    assert stop.is_set()
    assert seen == [1]


@pytest.mark.parametrize("num_workers", [1, 2])
def test_task_progress_tracks_completion_and_successes(monkeypatch, num_workers):
    from cryoflake import common

    class Progress:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.completed = 0
            self.successes = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_postfix(self, *, success, refresh):
            self.successes.append(success)

        def update(self, increment):
            self.completed += increment

    instances = []

    def make_progress(**kwargs):
        progress = Progress(**kwargs)
        instances.append(progress)
        return progress

    monkeypatch.setattr(common, "tqdm", make_progress)

    assert run_tasks_with_progress(
        [1, 2, 3],
        lambda value: value,
        num_workers,
        "Testing",
        success_predicate=lambda value: value != 2,
    ) == [1, 2, 3]
    assert len(instances) == 1
    progress = instances[0]
    assert progress.options == {"total": 3, "desc": "Testing", "unit": "item"}
    assert progress.completed == 3
    assert progress.successes[0] == 0
    assert progress.successes[-1] == 2


def test_empty_tasks_do_not_start_progress_or_workers(monkeypatch):
    from cryoflake import common

    def unexpected_call(*args, **kwargs):
        pytest.fail("Empty tasks must not start progress or workers")

    monkeypatch.setattr(common, "tqdm", unexpected_call)
    assert run_tasks_with_progress([], unexpected_call, 2, "Testing") == []


def test_parallel_callback_failure_cancels_pending_tasks_and_waits_for_active_workers():
    stop = threading.Event()
    second_started = threading.Event()
    second_finished = threading.Event()
    callbacks = []

    def work(value):
        if value == 0:
            assert second_started.wait(timeout=2)
        elif value == 1:
            second_started.set()
            assert stop.wait(timeout=2)
            second_finished.set()
        else:
            pytest.fail("Cancelled pending tasks must not start")
        return value

    def record(value):
        callbacks.append(value)
        if value == 0:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        run_tasks_with_progress(
            [0, 1, 2],
            work,
            2,
            "Testing",
            result_callback=record,
            stop_event=stop,
            wait_for_workers_on_error=True,
        )
    assert stop.is_set()
    assert second_finished.is_set()
    assert sorted(callbacks) == [0, 1]


def test_partial_worker_start_failure_cancels_and_joins_started_workers(monkeypatch):
    from cryoflake import common

    stop = threading.Event()
    started = threading.Event()
    finished = threading.Event()
    threads = []
    original_start = threading.Thread.start

    def start(thread):
        threads.append(thread)
        if len(threads) == 2:
            assert started.wait(timeout=2)
            raise RuntimeError("thread startup failed")
        original_start(thread)

    def work(value):
        started.set()
        try:
            assert stop.wait(timeout=2)
        finally:
            finished.set()
        return value

    monkeypatch.setattr(common.threading.Thread, "start", start)
    try:
        with pytest.raises(RuntimeError, match="thread startup failed"):
            run_tasks_with_progress(
                [0, 1],
                work,
                2,
                "Testing",
                stop_event=stop,
                wait_for_workers_on_error=True,
            )
        assert stop.is_set()
        assert finished.is_set()
        assert not threads[0].is_alive()
    finally:
        stop.set()
        threads[0].join(timeout=2)


def test_successful_task_run_joins_worker_teardown(monkeypatch):
    from cryoflake import common

    threads = []
    release = threading.Event()
    exited = threading.Event()
    original_thread = threading.Thread

    class WorkerThread(original_thread):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            threads.append(self)

        def run(self):
            try:
                super().run()
            finally:
                assert release.wait(timeout=2)
                exited.set()

        def join(self, *args, **kwargs):
            release.set()
            super().join(*args, **kwargs)

    monkeypatch.setattr(common.threading, "Thread", WorkerThread)
    try:
        assert run_tasks_with_progress([0], lambda value: value, 2, "Testing") == [0]
        assert exited.is_set()
        assert not threads[0].is_alive()
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=2)


def test_worker_start_interrupt_after_spawn_still_joins_worker(monkeypatch):
    from cryoflake import common

    stop = threading.Event()
    finished = threading.Event()
    threads = []
    original_start = threading.Thread.start

    def start(thread):
        threads.append(thread)
        original_start(thread)
        raise KeyboardInterrupt("interrupted startup")

    def work(value):
        assert stop.wait(timeout=2)
        finished.set()
        return value

    monkeypatch.setattr(common.threading.Thread, "start", start)
    try:
        with pytest.raises(KeyboardInterrupt, match="interrupted startup"):
            run_tasks_with_progress(
                [0, 1],
                work,
                2,
                "Testing",
                stop_event=stop,
                wait_for_workers_on_error=True,
            )
        assert stop.is_set()
        assert finished.is_set()
        assert not threads[0].is_alive()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2)
