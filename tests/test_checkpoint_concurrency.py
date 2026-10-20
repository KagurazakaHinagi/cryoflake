from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cryoflake.preprocess.checkpoint import PreprocessCheckpoint


def test_checkpoint_closes_ownership_file_when_flock_acquisition_fails(
    tmp_path, monkeypatch
):
    fcntl = pytest.importorskip("fcntl")
    opened_lock_files = []
    path_open = Path.open

    def track_open(path, *args, **kwargs):
        stream = path_open(path, *args, **kwargs)
        if path.name.endswith(".checkpoint.lock"):
            opened_lock_files.append(stream)
        return stream

    def fail_flock(*args):
        raise OSError("simulated flock failure")

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(fcntl, "flock", fail_flock)
    with pytest.raises(OSError, match="simulated flock failure"):
        PreprocessCheckpoint(tmp_path / "summary.csv", {})

    assert len(opened_lock_files) == 1
    assert opened_lock_files[0].closed


def test_checkpoint_requires_supported_ownership_locks(tmp_path, monkeypatch):
    opened_lock_files = []
    path_open = Path.open

    def track_open(path, *args, **kwargs):
        stream = path_open(path, *args, **kwargs)
        if path.name.endswith(".checkpoint.lock"):
            opened_lock_files.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setitem(sys.modules, "fcntl", None)
    with pytest.raises(RuntimeError, match="requires fcntl file locks"):
        PreprocessCheckpoint(tmp_path / "summary.csv", {})
    assert len(opened_lock_files) == 1
    assert opened_lock_files[0].closed
    assert not (tmp_path / "summary.csv.checkpoint.json").exists()


def test_checkpoint_closes_ownership_file_when_flock_release_fails(
    tmp_path, monkeypatch
):
    fcntl = pytest.importorskip("fcntl")
    checkpoint = PreprocessCheckpoint(tmp_path / "summary.csv", {})
    ownership_file = checkpoint._ownership_file
    flock = fcntl.flock

    def fail_unlock(fd, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError("simulated unlock failure")
        return flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", fail_unlock)
    try:
        with pytest.raises(OSError, match="simulated unlock failure"):
            checkpoint.close()
        assert checkpoint._journal_file.closed
        assert ownership_file.closed
        assert checkpoint._ownership_file is None
        monkeypatch.setattr(fcntl, "flock", flock)
        with PreprocessCheckpoint(tmp_path / "summary.csv", {}):
            pass
    finally:
        monkeypatch.setattr(fcntl, "flock", flock)
        ownership_file.close()


def test_failed_checkpoint_completion_retains_recovery_files_and_releases_lock(
    tmp_path,
):
    summary_path = tmp_path / "summary.csv"
    checkpoint = PreprocessCheckpoint(summary_path, {})
    checkpoint.record({"row_index": 1, "state": "upload_pending"})
    metadata = checkpoint.metadata_path.read_bytes()
    journal = checkpoint.journal_path.read_bytes()
    checkpoint.fail_from_callback(OSError("callback failed"))
    try:
        with pytest.raises(RuntimeError, match="callback failed"):
            checkpoint.complete()
        assert checkpoint._journal_file.closed
        assert checkpoint._ownership_file is None
        assert checkpoint.metadata_path.read_bytes() == metadata
        assert checkpoint.journal_path.read_bytes() == journal
        with PreprocessCheckpoint(summary_path, {}) as recovered:
            assert recovered.records[1]["state"] == "upload_pending"
    finally:
        checkpoint.close()


def test_checkpoint_records_are_independent_snapshots(tmp_path):
    with PreprocessCheckpoint(tmp_path / "summary.csv", {}) as checkpoint:
        record = {"row_index": 1, "summary_row": {"success": "true"}}
        checkpoint.record(record)
        record["summary_row"]["success"] = "false"
        assert checkpoint.records[1]["summary_row"]["success"] == "true"

        snapshot = checkpoint.records
        snapshot[1]["summary_row"]["success"] = "false"
        assert checkpoint.records[1]["summary_row"]["success"] == "true"


def test_concurrent_checkpoint_records_are_recoverable(tmp_path):
    summary_path = tmp_path / "summary.csv"
    row_indices = range(1, 101)
    with PreprocessCheckpoint(summary_path, {}) as checkpoint:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(
                executor.map(
                    lambda row: checkpoint.record({"row_index": row}), row_indices
                )
            )
        assert set(checkpoint.records) == set(row_indices)

    with PreprocessCheckpoint(summary_path, {}) as checkpoint:
        assert set(checkpoint.records) == set(row_indices)


def test_checkpoint_close_waits_for_inflight_record(tmp_path):
    checkpoint = PreprocessCheckpoint(tmp_path / "summary.csv", {})
    stream = checkpoint._journal_file
    writing = threading.Event()
    release_write = threading.Event()
    close_started = threading.Event()
    close_done = threading.Event()

    class PausedStream:
        def write(self, payload):
            writing.set()
            assert release_write.wait(5)
            return stream.write(payload)

        def flush(self):
            return stream.flush()

        def close(self):
            return stream.close()

    checkpoint._journal_file = PausedStream()

    def close_checkpoint():
        close_started.set()
        checkpoint.close()
        close_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        record_future = executor.submit(checkpoint.record, {"row_index": 1})
        assert writing.wait(5)
        close_future = executor.submit(close_checkpoint)
        try:
            assert close_started.wait(5)
            assert not close_done.is_set()
        finally:
            release_write.set()
        record_future.result(timeout=5)
        close_future.result(timeout=5)
    assert stream.closed
    assert checkpoint.records[1]["row_index"] == 1


def test_checkpoint_ownership_is_exclusive_between_processes(tmp_path):
    pytest.importorskip("fcntl")
    summary_path = tmp_path / "summary.csv"
    script = """
import sys
from cryoflake.preprocess.checkpoint import PreprocessCheckpoint
try:
    with PreprocessCheckpoint(sys.argv[1], {}):
        print('acquired')
except RuntimeError as exc:
    if 'Another preprocessing run owns checkpoint' not in str(exc):
        raise
    print('owned')
"""
    with PreprocessCheckpoint(summary_path, {}):
        owned = subprocess.run(
            [sys.executable, "-c", script, str(summary_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        assert owned.stdout.strip() == "owned"
    available = subprocess.run(
        [sys.executable, "-c", script, str(summary_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert available.stdout.strip() == "acquired"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork inheritance")
def test_forked_checkpoint_cannot_release_parent_ownership(tmp_path):
    pytest.importorskip("fcntl")
    summary_path = tmp_path / "summary.csv"
    script = """
import os
import signal
import sys
from cryoflake.preprocess.checkpoint import PreprocessCheckpoint
summary_path = sys.argv[1]
with PreprocessCheckpoint(summary_path, {}) as checkpoint:
    child_pid = os.fork()
    if child_pid == 0:
        signal.alarm(5)
        try:
            checkpoint.close()
        except RuntimeError as exc:
            os._exit(0 if 'process' in str(exc) else 2)
        except BaseException:
            os._exit(3)
        os._exit(1)
    _, status = os.waitpid(child_pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    try:
        replacement = PreprocessCheckpoint(summary_path, {})
    except RuntimeError as exc:
        assert 'Another preprocessing run owns' in str(exc)
    else:
        replacement.close()
        raise AssertionError('Child released parent checkpoint ownership')
"""
    subprocess.run(
        [sys.executable, "-c", script, str(summary_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
