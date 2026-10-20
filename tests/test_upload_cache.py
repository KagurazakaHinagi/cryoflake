"""Upload staging and cleanup operate only on managed artifact directories."""

import threading

import pytest

from cryoflake.storage.upload_cache import BatchedArtifactUploadCache


def test_upload_failures_retain_artifact_directory(tmp_path):
    def fail_upload(local_path, remote_uri):
        raise OSError("remote unavailable")

    with BatchedArtifactUploadCache(tmp_path, fail_upload, num_workers=1) as cache:
        artifact = cache.allocate_artifact_dir(1, "artifact")
        artifact.mkdir()
        metadata = artifact / "metadata.json"
        metadata.write_bytes(b"retained")
        result = cache.stage(artifact, "s3://bucket/artifact")

    with pytest.raises(RuntimeError, match="cached artifact retained"):
        result.result(timeout=1)
    assert metadata.read_bytes() == b"retained"


def test_stage_rejects_single_files_before_upload(tmp_path):
    uploaded = []
    with BatchedArtifactUploadCache(
        tmp_path,
        lambda *args: uploaded.append(args),
        num_workers=1,
    ) as cache:
        artifact = cache.allocate_artifact_dir(1, "artifact")
        artifact.write_bytes(b"unsupported single file")
        with pytest.raises(ValueError, match="requires an artifact directory"):
            cache.stage(artifact, "s3://bucket/artifact")
        assert uploaded == []
        assert artifact.read_bytes() == b"unsupported single file"
        cache.discard(artifact)
        assert not artifact.parent.exists()


def test_discard_never_deletes_unmanaged_files(tmp_path):
    unmanaged = tmp_path / "important"
    unmanaged.write_bytes(b"keep")
    with BatchedArtifactUploadCache(
        tmp_path, lambda *args: None, num_workers=1
    ) as cache:
        cache.discard(unmanaged)
    assert unmanaged.read_bytes() == b"keep"


def test_discard_cleans_managed_rows_without_an_artifact(tmp_path):
    with BatchedArtifactUploadCache(
        tmp_path, lambda *args: None, num_workers=1
    ) as cache:
        artifact = cache.allocate_artifact_dir(1, "artifact")
        temporary_output = artifact.parent / "interrupted.tmp"
        temporary_output.write_bytes(b"partial")
        cache.discard(artifact)
        cache.discard(artifact)
        assert not artifact.parent.exists()


def test_discard_never_follows_artifact_symlinks_to_unmanaged_paths(tmp_path):
    unmanaged = tmp_path / "important"
    unmanaged.mkdir()
    content = unmanaged / "data"
    content.write_bytes(b"keep")
    with BatchedArtifactUploadCache(
        tmp_path, lambda *args: None, num_workers=1
    ) as cache:
        artifact = cache.allocate_artifact_dir(1, "artifact")
        artifact.symlink_to(unmanaged, target_is_directory=True)
        cache.discard(artifact)
    assert content.read_bytes() == b"keep"


def test_discard_never_follows_artifact_symlinks_to_another_managed_row(tmp_path):
    with BatchedArtifactUploadCache(
        tmp_path, lambda *args: None, num_workers=1
    ) as cache:
        artifact = cache.allocate_artifact_dir(1, "artifact")
        other_artifact = cache.allocate_artifact_dir(2, "artifact")
        other_artifact.mkdir()
        content = other_artifact / "data"
        content.write_bytes(b"keep")
        artifact.symlink_to(other_artifact, target_is_directory=True)

        cache.discard(artifact)

        assert content.read_bytes() == b"keep"


def test_cancelled_staged_upload_retains_cache_and_does_not_fail_other_uploads(
    tmp_path,
):
    uploaded = []
    with BatchedArtifactUploadCache(
        tmp_path,
        lambda local_path, remote_uri: uploaded.append(remote_uri),
        num_workers=1,
        stop_on_failure=True,
    ) as cache:
        cancelled_artifact = cache.allocate_artifact_dir(1, "cancelled")
        cancelled_artifact.mkdir()
        cancelled = cache.stage(cancelled_artifact, "s3://bucket/cancelled")
        completed_artifact = cache.allocate_artifact_dir(2, "completed")
        completed_artifact.mkdir()
        completed = cache.stage(completed_artifact, "s3://bucket/completed")
        assert cancelled.cancel()

    assert cancelled.cancelled()
    assert cancelled_artifact.is_dir()
    assert completed.result(timeout=1) is None
    assert uploaded == ["s3://bucket/completed"]
    cache.raise_if_failed()


def test_running_upload_cannot_be_cancelled(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def upload(local_path, remote_uri):
        started.set()
        assert release.wait(timeout=5)

    with BatchedArtifactUploadCache(
        tmp_path, upload, num_workers=1, batch_size=1
    ) as cache:
        artifact = cache.allocate_artifact_dir(1, "artifact")
        artifact.mkdir()
        result = cache.stage(artifact, "s3://bucket/artifact")
        try:
            assert started.wait(timeout=2)
            assert not result.cancel()
        finally:
            release.set()

    assert result.result(timeout=1) is None


@pytest.mark.parametrize("row_index", [True, 0, -1, 1.5, float("inf")])
def test_allocate_artifact_dir_refuses_invalid_rows(tmp_path, row_index):
    with BatchedArtifactUploadCache(
        tmp_path, lambda *args: None, num_workers=1
    ) as cache:
        with pytest.raises(ValueError, match="row_index must be a positive integer"):
            cache.allocate_artifact_dir(row_index, "artifact")
        assert list(cache.cache_dir.iterdir()) == []


@pytest.mark.parametrize("file_name", ["", ".", ".."])
def test_allocate_artifact_dir_refuses_empty_or_parent_artifact_names(
    tmp_path, file_name
):
    with BatchedArtifactUploadCache(
        tmp_path, lambda *args: None, num_workers=1
    ) as cache:
        with pytest.raises(ValueError, match="artifact_name must name an artifact"):
            cache.allocate_artifact_dir(1, file_name)
        assert list(cache.cache_dir.iterdir()) == []


def test_concurrent_close_waits_for_uploads_and_callbacks(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    joining = threading.Event()
    second_returned = threading.Event()
    callback_finished = threading.Event()

    def upload(*args):
        started.set()
        assert release.wait(timeout=5)

    cache = BatchedArtifactUploadCache(tmp_path, upload, num_workers=1, batch_size=1)
    artifact = cache.allocate_artifact_dir(1, "artifact")
    artifact.mkdir()
    result = cache.stage(artifact, "s3://bucket/artifact")
    result.add_done_callback(lambda _: callback_finished.set())
    original_join = cache._dispatcher.join

    def join(*args, **kwargs):
        joining.set()
        return original_join(*args, **kwargs)

    monkeypatch.setattr(cache._dispatcher, "join", join)
    first = threading.Thread(target=cache.close)
    second = threading.Thread(target=lambda: (cache.close(), second_returned.set()))
    try:
        assert started.wait(timeout=2)
        first.start()
        assert joining.wait(timeout=2)
        second.start()
        assert not second_returned.wait(timeout=0.1)
    finally:
        release.set()
        first.join(timeout=3)
        if second.ident is not None:
            second.join(timeout=3)
        cache.close()
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_returned.is_set()
    assert callback_finished.is_set()


def test_close_wakes_stage_waiting_for_backpressure(tmp_path):
    started = threading.Event()
    release = threading.Event()
    staging = threading.Event()
    stage_finished = threading.Event()
    stage_errors = []

    def upload(*args):
        started.set()
        assert release.wait(timeout=5)

    cache = BatchedArtifactUploadCache(
        tmp_path, upload, num_workers=1, batch_size=1, max_pending_batches=1
    )
    first_path = cache.allocate_artifact_dir(1, "first")
    first_path.mkdir()
    second_path = cache.allocate_artifact_dir(2, "second")
    second_path.mkdir()
    first_result = cache.stage(first_path, "s3://bucket/first")

    def stage():
        staging.set()
        try:
            cache.stage(second_path, "s3://bucket/second")
        except RuntimeError as exc:
            stage_errors.append(exc)
        finally:
            stage_finished.set()

    producer = threading.Thread(target=stage)
    closer = threading.Thread(target=cache.close)
    try:
        assert started.wait(timeout=2)
        producer.start()
        assert staging.wait(timeout=2)
        assert not stage_finished.wait(timeout=0.1)
        closer.start()
        assert stage_finished.wait(timeout=2)
        assert len(stage_errors) == 1
        assert "closure" in str(stage_errors[0])
        assert second_path.is_dir()
    finally:
        release.set()
        producer.join(timeout=3)
        if closer.ident is not None:
            closer.join(timeout=3)
        cache.close()
    assert first_result.result(timeout=1) is None


def test_upload_cleanup_preserves_sibling_artifact_in_same_row(tmp_path):
    slow_started = threading.Event()
    fast_finished = threading.Event()

    def upload(local_path, remote_uri):
        if remote_uri.endswith("fast"):
            assert slow_started.wait(timeout=2)
        else:
            slow_started.set()
            assert fast_finished.wait(timeout=2)
            assert (local_path / "data").read_bytes() == b"keep until uploaded"

    with BatchedArtifactUploadCache(
        tmp_path, upload, num_workers=2, batch_size=2
    ) as cache:
        fast_path = cache.allocate_artifact_dir(1, "fast")
        fast_path.mkdir()
        slow_path = cache.allocate_artifact_dir(1, "slow")
        slow_path.mkdir()
        (slow_path / "data").write_bytes(b"keep until uploaded")
        fast = cache.stage(fast_path, "s3://bucket/fast")
        fast.add_done_callback(lambda _: fast_finished.set())
        slow = cache.stage(slow_path, "s3://bucket/slow")

    assert fast.result(timeout=1) is None
    assert slow.result(timeout=1) is None
    assert not cache.cache_dir.exists()


def test_last_sibling_completion_removes_empty_row(tmp_path, monkeypatch):
    cleaned = threading.Barrier(2)
    cache = BatchedArtifactUploadCache(
        tmp_path, lambda *args: None, num_workers=2, batch_size=2
    )
    original_remove = cache._remove_cached_output

    def remove(*args, **kwargs):
        result = original_remove(*args, **kwargs)
        cleaned.wait(timeout=2)
        return result

    monkeypatch.setattr(cache, "_remove_cached_output", remove)
    paths = [cache.allocate_artifact_dir(1, name) for name in ("first", "second")]
    for path in paths:
        path.mkdir()
    first = cache.stage(paths[0], "s3://bucket/first")
    second = cache.stage(paths[1], "s3://bucket/second")
    cache.close()

    assert first.result(timeout=1) is None
    assert second.result(timeout=1) is None
    assert not cache.cache_dir.exists()


def test_submission_failure_waits_for_running_upload(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    submission_failed = threading.Event()
    cache = BatchedArtifactUploadCache(
        tmp_path,
        lambda *args: (started.set(), release.wait(timeout=5)),
        num_workers=1,
        batch_size=2,
    )
    original_submit = cache._executor.submit
    submissions = 0

    def submit(*args, **kwargs):
        nonlocal submissions
        submissions += 1
        if submissions == 2:
            submission_failed.set()
            raise RuntimeError("submission unavailable")
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(cache._executor, "submit", submit)
    first_path = cache.allocate_artifact_dir(1, "first")
    first_path.mkdir()
    second_path = cache.allocate_artifact_dir(2, "second")
    second_path.mkdir()
    first = cache.stage(first_path, "s3://bucket/first")
    second = cache.stage(second_path, "s3://bucket/second")
    try:
        assert started.wait(timeout=2)
        assert submission_failed.wait(timeout=2)
        assert not first.done()
    finally:
        release.set()
        cache.close()
    assert first.result(timeout=1) is None
    with pytest.raises(RuntimeError, match="submission unavailable"):
        second.result(timeout=1)
    assert second_path.is_dir()


def test_worker_start_failure_never_runs_rejected_queued_upload(tmp_path, monkeypatch):
    uploaded = []
    cache = BatchedArtifactUploadCache(
        tmp_path,
        lambda _, remote: uploaded.append(remote),
        num_workers=1,
        batch_size=1,
    )
    first_path = cache.allocate_artifact_dir(1, "first")
    first_path.mkdir()
    second_path = cache.allocate_artifact_dir(2, "second")
    second_path.mkdir()
    original_start = threading.Thread.start

    def fail_start(_):
        raise RuntimeError("worker cannot start")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    first = cache.stage(first_path, "s3://bucket/rejected")
    try:
        with pytest.raises(RuntimeError, match="worker cannot start"):
            first.result(timeout=2)
        monkeypatch.setattr(threading.Thread, "start", original_start)
        second = cache.stage(second_path, "s3://bucket/accepted")
    finally:
        monkeypatch.setattr(threading.Thread, "start", original_start)
        cache.close()
    assert second.result(timeout=1) is None
    assert uploaded == ["s3://bucket/accepted"]
    assert first_path.is_dir()


def test_dispatcher_start_failure_releases_created_resources(tmp_path, monkeypatch):
    shutdown_calls = []
    from cryoflake.storage import upload_cache

    original_shutdown = upload_cache.ThreadPoolExecutor.shutdown

    def shutdown(executor, *args, **kwargs):
        shutdown_calls.append(executor)
        return original_shutdown(executor, *args, **kwargs)

    monkeypatch.setattr(upload_cache.ThreadPoolExecutor, "shutdown", shutdown)
    monkeypatch.setattr(
        upload_cache.threading.Thread,
        "start",
        lambda _: (_ for _ in ()).throw(RuntimeError("cannot start dispatcher")),
    )
    with pytest.raises(RuntimeError, match="cannot start dispatcher"):
        BatchedArtifactUploadCache(tmp_path, lambda *args: None, num_workers=1)
    assert len(shutdown_calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_dispatcher_start_interruption_joins_already_started_thread(
    tmp_path, monkeypatch
):
    started_threads = []
    original_start = threading.Thread.start

    def interrupted_start(thread):
        started_threads.append(thread)
        original_start(thread)
        raise KeyboardInterrupt("dispatcher start interrupted")

    monkeypatch.setattr(threading.Thread, "start", interrupted_start)
    with pytest.raises(KeyboardInterrupt, match="dispatcher start interrupted"):
        BatchedArtifactUploadCache(tmp_path, lambda *args: None, num_workers=1)
    assert len(started_threads) == 1
    assert not started_threads[0].is_alive()
    assert list(tmp_path.iterdir()) == []


def test_allocate_artifact_dir_rejects_closed_cache_without_creating_rows(tmp_path):
    cache = BatchedArtifactUploadCache(tmp_path, lambda *args: None, num_workers=1)
    cache.close()
    with pytest.raises(RuntimeError, match="closure"):
        cache.allocate_artifact_dir(1, "artifact")
    assert not cache.cache_dir.exists()


@pytest.mark.parametrize("close_from", ["upload", "callback"])
def test_upload_or_callback_can_request_close_without_deadlock(tmp_path, close_from):
    finished = threading.Event()

    def upload(*args):
        if close_from == "upload":
            cache.close()

    cache = BatchedArtifactUploadCache(tmp_path, upload, num_workers=1, batch_size=1)
    artifact = cache.allocate_artifact_dir(1, "artifact")
    artifact.mkdir()
    result = cache.stage(artifact, "s3://bucket/artifact")

    def callback(_):
        if close_from == "callback":
            cache.close()
        finished.set()

    result.add_done_callback(callback)
    assert finished.wait(timeout=3)
    cache.close()
    assert result.result(timeout=1) is None
    assert not cache._dispatcher.is_alive()
    assert cache._executor._shutdown


def test_fatal_completion_callback_does_not_abandon_other_futures(tmp_path):
    def interrupted(_):
        raise KeyboardInterrupt("completion callback interrupted")

    with BatchedArtifactUploadCache(
        tmp_path,
        lambda *args: None,
        num_workers=1,
        batch_size=2,
        stop_on_failure=True,
    ) as cache:
        first_path = cache.allocate_artifact_dir(1, "first")
        first_path.mkdir()
        second_path = cache.allocate_artifact_dir(2, "second")
        second_path.mkdir()
        first = cache.stage(first_path, "s3://bucket/first")
        first.add_done_callback(interrupted)
        second = cache.stage(second_path, "s3://bucket/second")

    assert first.result(timeout=1) is None
    assert second.result(timeout=1) is None
    with pytest.raises(RuntimeError, match="completion callback interrupted"):
        cache.raise_if_failed()


def test_discard_preserves_staged_sibling_and_unstaged_allocation(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def upload(local_path, _):
        started.set()
        assert release.wait(timeout=3)
        assert (local_path / "data").read_bytes() == b"upload me"

    cache = BatchedArtifactUploadCache(tmp_path, upload, num_workers=1, batch_size=1)
    active_path = cache.allocate_artifact_dir(1, "active")
    active_path.mkdir()
    (active_path / "data").write_bytes(b"upload me")
    failed_path = cache.allocate_artifact_dir(1, "failed")
    failed_path.mkdir()
    allocated_path = cache.allocate_artifact_dir(1, "allocated")
    result = cache.stage(active_path, "s3://bucket/active")
    try:
        assert started.wait(timeout=2)
        cache.discard(active_path)
        cache.discard(failed_path)
        assert not failed_path.exists()
        assert (active_path / "data").read_bytes() == b"upload me"
    finally:
        release.set()
        cache.close()
    assert result.result(timeout=1) is None
    assert allocated_path.parent.is_dir()
    cache.discard(allocated_path)
    assert not cache.cache_dir.exists()


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_discard_sibling_preserves_retained_upload(tmp_path, outcome):
    def upload(*args):
        raise OSError("remote unavailable")

    cache = BatchedArtifactUploadCache(tmp_path, upload, num_workers=1)
    retained_path = cache.allocate_artifact_dir(1, "retained")
    retained_path.mkdir()
    data = retained_path / "data"
    data.write_bytes(b"retry me")
    discarded_path = cache.allocate_artifact_dir(1, "discarded")
    result = cache.stage(retained_path, "s3://bucket/retained")
    if outcome == "cancelled":
        assert result.cancel()
    cache.close()
    if outcome == "failed":
        with pytest.raises(RuntimeError, match="remote unavailable"):
            result.result(timeout=1)

    cache.discard(discarded_path)
    assert data.read_bytes() == b"retry me"
    cache.discard(retained_path)
    assert not cache.cache_dir.exists()


def test_failed_upload_claim_allows_explicit_retry(tmp_path):
    attempts = 0

    def upload(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary failure")

    cache = BatchedArtifactUploadCache(tmp_path, upload, num_workers=1, batch_size=1)
    artifact = cache.allocate_artifact_dir(1, "artifact")
    artifact.mkdir()
    first = cache.stage(artifact, "s3://bucket/artifact")
    with pytest.raises(RuntimeError, match="temporary failure"):
        first.result(timeout=2)
    second = cache.stage(artifact, "s3://bucket/artifact")
    cache.close()
    assert second.result(timeout=1) is None
    assert attempts == 2
    assert not cache.cache_dir.exists()


def test_stage_rejects_concurrent_duplicate_artifact_uploads(tmp_path):
    with BatchedArtifactUploadCache(
        tmp_path, lambda *args: None, num_workers=1
    ) as cache:
        artifact = cache.allocate_artifact_dir(1, "artifact")
        artifact.mkdir()
        result = cache.stage(artifact, "s3://bucket/first")
        with pytest.raises(RuntimeError, match="already staged"):
            cache.stage(artifact, "s3://bucket/second")
        with pytest.raises(RuntimeError, match="while its upload is staged"):
            cache.allocate_artifact_dir(1, "artifact")
    assert result.result(timeout=1) is None


def test_upload_failure_wakes_blocked_stage_before_other_upload_finishes(tmp_path):
    running = threading.Event()
    fail = threading.Event()
    release = threading.Event()
    stage_finished = threading.Event()
    stage_errors = []

    def upload(_, remote_uri):
        if remote_uri.endswith("failed"):
            assert fail.wait(timeout=3)
            raise OSError("remote unavailable")
        running.set()
        assert release.wait(timeout=3)

    cache = BatchedArtifactUploadCache(
        tmp_path,
        upload,
        num_workers=2,
        batch_size=2,
        max_pending_batches=1,
        stop_on_failure=True,
    )
    paths = [cache.allocate_artifact_dir(index, "artifact") for index in range(1, 4)]
    for path in paths:
        path.mkdir()
    failed = cache.stage(paths[0], "s3://bucket/failed")
    successful = cache.stage(paths[1], "s3://bucket/running")

    def stage():
        try:
            cache.stage(paths[2], "s3://bucket/never-uploaded")
        except RuntimeError as exc:
            stage_errors.append(exc)
        finally:
            stage_finished.set()

    producer = threading.Thread(target=stage)
    try:
        assert running.wait(timeout=2)
        producer.start()
        assert not stage_finished.wait(timeout=0.1)
        fail.set()
        assert stage_finished.wait(timeout=2)
        assert "asynchronous upload failure" in str(stage_errors[0])
        assert not successful.done()
    finally:
        fail.set()
        release.set()
        producer.join(timeout=3)
        cache.close()
    assert successful.result(timeout=1) is None
    with pytest.raises(RuntimeError, match="remote unavailable"):
        failed.result(timeout=1)


def test_upload_reentrant_stage_refuses_blocking_on_its_own_batch(tmp_path):
    stage_errors = []

    def upload(*args):
        try:
            cache.stage(second_path, "s3://bucket/second")
        except RuntimeError as exc:
            stage_errors.append(exc)

    cache = BatchedArtifactUploadCache(
        tmp_path, upload, num_workers=1, batch_size=1, max_pending_batches=1
    )
    first_path = cache.allocate_artifact_dir(1, "first")
    first_path.mkdir()
    second_path = cache.allocate_artifact_dir(2, "second")
    second_path.mkdir()
    result = cache.stage(first_path, "s3://bucket/first")
    cache.close()
    assert result.result(timeout=1) is None
    assert len(stage_errors) == 1
    assert "backpressure" in str(stage_errors[0]) or "closure" in str(stage_errors[0])
    assert second_path.is_dir()
