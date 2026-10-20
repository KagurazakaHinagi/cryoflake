from __future__ import annotations

import queue
import shutil
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from cryoflake.common import validate_num_workers

DEFAULT_UPLOAD_BATCH_SIZE = 100


@dataclass(frozen=True)
class _CachedArtifactUpload:
    local_artifact_dir: Path
    artifact_dir_identity: Path
    remote_artifact_reference: str
    result: Future[Optional[str]]


class BatchedArtifactUploadCache:
    """Stage local artifacts and upload bounded batches on background threads."""

    _STOP = object()

    def __init__(
        self,
        cache_parent: Path,
        upload_callback: Callable[[Path, str], None],
        *,
        num_workers: int,
        batch_size: int = DEFAULT_UPLOAD_BATCH_SIZE,
        max_pending_batches: int = 2,
        stop_on_failure: bool = False,
    ) -> None:
        self.num_workers = validate_num_workers(num_workers, "num_workers")
        self.batch_size = validate_num_workers(batch_size, "batch_size")
        self.max_pending_batches = validate_num_workers(
            max_pending_batches, "max_pending_batches"
        )
        cache_parent = Path(cache_parent)
        cache_parent.mkdir(parents=True, exist_ok=True)
        self.cache_parent = cache_parent.resolve()
        self.cache_dir = Path(
            tempfile.mkdtemp(prefix=".upload_cache_", dir=str(cache_parent))
        )
        self._upload_callback = upload_callback
        self._stop_on_failure = bool(stop_on_failure)
        self._failure_lock = threading.Lock()
        self._first_failure: Optional[BaseException] = None
        self._pending_batches = 0
        self._batch_queue: queue.Queue[object] = queue.Queue()
        self._current_batch: list[_CachedArtifactUpload] = []
        self._state_changed = threading.Condition()
        self._claimed_paths: dict[Path, Optional[Future[Optional[str]]]] = {}
        self._claimed_rows: dict[Path, int] = {}
        self._closed = False
        self._worker_context = threading.local()
        executor = None
        dispatcher = None
        try:
            executor = self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="cryoflake-upload",
            )
            dispatcher = self._dispatcher = threading.Thread(
                target=self._dispatch_batches,
                name="cryoflake-upload-dispatcher",
                daemon=True,
            )
            self._dispatcher.start()
        except BaseException:
            with self._state_changed:
                self._closed = True
                self._batch_queue.put(self._STOP)
                self._state_changed.notify_all()
            try:
                if dispatcher is not None and dispatcher.ident is not None:
                    dispatcher.join()
                elif executor is not None:
                    executor.shutdown(wait=True)
            finally:
                self._remove_empty_directory(self.cache_dir)
            raise

    def allocate_artifact_dir(self, row_index: int, artifact_name: str) -> Path:
        row_index = validate_num_workers(row_index, "row_index")
        artifact_name = Path(artifact_name).name
        if artifact_name in {"", ".", ".."}:
            raise ValueError("artifact_name must name an artifact directory")
        row_directory = self.cache_dir / f"row_{row_index:08d}"
        with self._state_changed:
            if self._closed:
                raise RuntimeError("Cannot allocate an upload after cache closure")
            artifact_dir_identity = row_directory.resolve() / artifact_name
            if self._claimed_paths.get(artifact_dir_identity) is not None:
                raise RuntimeError(
                    f"Cannot allocate an artifact while its upload is staged: "
                    f"{row_directory / artifact_name}"
                )
            row_directory.mkdir(parents=True, exist_ok=True)
            if artifact_dir_identity not in self._claimed_paths:
                self._claim_path(artifact_dir_identity, None)
        return row_directory / artifact_name

    def stage(
        self, local_artifact_dir: Path, remote_artifact_reference: str
    ) -> Future[Optional[str]]:
        local_artifact_dir = Path(local_artifact_dir)
        if not local_artifact_dir.is_dir():
            raise ValueError(
                f"Upload cache requires an artifact directory: {local_artifact_dir}"
            )
        result: Future[Optional[str]] = Future()
        artifact_dir_identity = local_artifact_dir.resolve()
        cached_upload = _CachedArtifactUpload(
            local_artifact_dir=local_artifact_dir,
            artifact_dir_identity=artifact_dir_identity,
            remote_artifact_reference=str(remote_artifact_reference),
            result=result,
        )
        with self._state_changed:
            while True:
                if self._closed:
                    raise RuntimeError("Cannot stage an upload after cache closure")
                self.raise_if_failed()
                if self._claimed_paths.get(artifact_dir_identity) is not None:
                    raise RuntimeError(
                        f"Upload artifact is already staged: {local_artifact_dir}"
                    )
                if (
                    self._current_batch
                    or self._pending_batches < self.max_pending_batches
                ):
                    break
                if threading.current_thread() is self._dispatcher or getattr(
                    self._worker_context, "active", False
                ):
                    raise RuntimeError(
                        "Cannot wait for upload backpressure from an upload callback"
                    )
                self._state_changed.wait()
            if not local_artifact_dir.is_dir():
                raise ValueError(
                    f"Upload cache requires an artifact directory: {local_artifact_dir}"
                )
            if not self._current_batch:
                self._pending_batches += 1
            self._claim_path(artifact_dir_identity, result)
            self._current_batch.append(cached_upload)
            if len(self._current_batch) == self.batch_size:
                self._batch_queue.put(tuple(self._current_batch))
                self._current_batch = []
        return result

    def raise_if_failed(self) -> None:
        if not self._stop_on_failure:
            return
        with self._failure_lock:
            failure = self._first_failure
        if failure is not None:
            raise RuntimeError(
                f"Stopping after asynchronous upload failure: {failure}"
            ) from failure

    def flush(self) -> None:
        with self._state_changed:
            if self._closed:
                return
            if self._current_batch:
                self._batch_queue.put(tuple(self._current_batch))
                self._current_batch = []

    def discard(self, local_artifact_dir: Path) -> None:
        local_artifact_dir = Path(local_artifact_dir)
        with self._state_changed:
            managed_row_directory = self._managed_row_directory(local_artifact_dir)
            if managed_row_directory is None:
                return
            artifact_dir_identity = managed_row_directory / local_artifact_dir.name
            if self._claimed_paths.get(artifact_dir_identity) is not None:
                return
            self._release_path_claim(artifact_dir_identity)
            remove_row = managed_row_directory not in self._claimed_rows
            self._remove_cached_output(local_artifact_dir, remove_row=remove_row)

    def close(self) -> None:
        """Finish accepted uploads; upload callbacks may request closure too."""
        with self._state_changed:
            if not self._closed:
                if self._current_batch:
                    self._batch_queue.put(tuple(self._current_batch))
                    self._current_batch = []
                self._closed = True
                self._batch_queue.put(self._STOP)
                self._state_changed.notify_all()
        if threading.current_thread() is not self._dispatcher and not getattr(
            self._worker_context, "active", False
        ):
            self._dispatcher.join()

    def __enter__(self) -> "BatchedArtifactUploadCache":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _dispatch_batches(self) -> None:
        try:
            while True:
                batch = self._batch_queue.get()
                try:
                    if batch is self._STOP:
                        return
                    self._dispatch_batch(batch)
                finally:
                    self._batch_queue.task_done()
        finally:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            finally:
                self._remove_empty_directory(self.cache_dir)

    def _dispatch_batch(self, batch: object) -> None:
        uploads = tuple(batch)
        try:
            submitted = {}
            submission_failure = None
            for upload in uploads:
                if not upload.result.set_running_or_notify_cancel():
                    self._release_upload_claim(upload, retain=True)
                    continue
                if submission_failure is None:
                    accepted: Future[bool] = Future()
                    try:
                        worker = self._executor.submit(
                            self._upload_one, upload, accepted
                        )
                    except BaseException as exc:
                        # submit() may have queued work before worker startup
                        # raises. A rejected upload must never run later.
                        accepted.set_result(False)
                        self._record_failure(exc)
                        submission_failure = exc
                    else:
                        accepted.set_result(True)
                        submitted[worker] = upload
                if submission_failure is not None:
                    self._finish_upload(upload, error=submission_failure)
            for completed in as_completed(submitted):
                upload = submitted[completed]
                try:
                    cleanup_warning = completed.result()
                except BaseException as exc:
                    self._record_failure(exc)
                    self._finish_upload(upload, error=exc)
                else:
                    self._finish_upload(upload, cleanup_warning=cleanup_warning)
        finally:
            with self._state_changed:
                self._pending_batches -= 1
                self._state_changed.notify_all()

    def _finish_upload(
        self,
        upload: _CachedArtifactUpload,
        *,
        cleanup_warning: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        self._release_upload_claim(
            upload, retain=error is not None or cleanup_warning is not None
        )
        try:
            if error is not None:
                upload.result.set_exception(error)
            else:
                upload.result.set_result(cleanup_warning)
        except BaseException as exc:
            # A callback can raise SystemExit/KeyboardInterrupt after the result
            # is published. It must not abandon other uploads or their futures.
            self._record_failure(exc)

    def _release_upload_claim(
        self, upload: _CachedArtifactUpload, *, retain: bool = False
    ) -> None:
        with self._state_changed:
            if self._claimed_paths.get(upload.artifact_dir_identity) is upload.result:
                if retain:
                    self._claimed_paths[upload.artifact_dir_identity] = None
                    return
                self._release_path_claim(upload.artifact_dir_identity)
                row_directory = self._managed_row_directory(upload.local_artifact_dir)
                if (
                    row_directory is not None
                    and row_directory not in self._claimed_rows
                ):
                    self._remove_empty_directory(row_directory)
                    self._remove_empty_directory(row_directory.parent)

    def _claim_path(self, path: Path, result: Optional[Future[Optional[str]]]) -> None:
        if path not in self._claimed_paths:
            self._claimed_rows[path.parent] = self._claimed_rows.get(path.parent, 0) + 1
        self._claimed_paths[path] = result

    def _release_path_claim(self, path: Path) -> None:
        if path not in self._claimed_paths:
            return
        del self._claimed_paths[path]
        remaining = self._claimed_rows[path.parent] - 1
        if remaining:
            self._claimed_rows[path.parent] = remaining
        else:
            del self._claimed_rows[path.parent]

    def _record_failure(self, exc: BaseException) -> None:
        with self._failure_lock:
            if self._first_failure is None:
                self._first_failure = exc
        with self._state_changed:
            self._state_changed.notify_all()

    def _upload_one(
        self, upload: _CachedArtifactUpload, accepted: Future[bool]
    ) -> Optional[str]:
        if not accepted.result():
            return None
        self._worker_context.active = True
        try:
            try:
                self._upload_callback(
                    upload.local_artifact_dir, upload.remote_artifact_reference
                )
            except BaseException as exc:
                raise RuntimeError(
                    f"Upload failed for {upload.remote_artifact_reference}; cached artifact retained "
                    f"at {upload.local_artifact_dir}: {exc}"
                ) from exc
            cleanup_error = self._remove_cached_output(upload.local_artifact_dir)
            if cleanup_error is None:
                return None
            return (
                f"Upload succeeded for {upload.remote_artifact_reference}, but the local cache "
                f"could not be removed at {upload.local_artifact_dir}: {cleanup_error}"
            )
        finally:
            self._worker_context.active = False

    def _remove_cached_output(
        self, local_artifact_dir: Path, *, remove_row: bool = False
    ) -> Optional[str]:
        managed_row_directory = self._managed_row_directory(local_artifact_dir)
        if managed_row_directory is not None:
            try:
                shutil.rmtree(
                    managed_row_directory if remove_row else local_artifact_dir
                )
            except FileNotFoundError:
                return None
            except OSError as exc:
                return str(exc)
            with self._state_changed:
                claimed_siblings = self._claimed_rows.get(managed_row_directory, 0) - (
                    (managed_row_directory / local_artifact_dir.name)
                    in self._claimed_paths
                )
                if not remove_row and not claimed_siblings:
                    self._remove_empty_directory(managed_row_directory)
                self._remove_empty_directory(managed_row_directory.parent)
        else:
            try:
                local_artifact_dir.lstat()
            except FileNotFoundError:
                return None
            except OSError as exc:
                return str(exc)
            return (
                "refusing to remove an uploaded path outside a "
                f"managed cache root: {local_artifact_dir}"
            )
        return None

    def _managed_row_directory(self, local_artifact_dir: Path) -> Optional[Path]:
        try:
            if any(
                path.is_symlink()
                for path in (
                    local_artifact_dir,
                    local_artifact_dir.parent,
                    local_artifact_dir.parent.parent,
                )
            ):
                return None
            resolved_path = local_artifact_dir.resolve()
        except (OSError, RuntimeError):
            return None
        row_directory = resolved_path.parent
        cache_directory = row_directory.parent
        if (
            cache_directory.parent != self.cache_parent
            or not cache_directory.name.startswith(".upload_cache_")
            or len(row_directory.name) < len("row_00000000")
            or not row_directory.name.startswith("row_")
            or not row_directory.name[len("row_") :].isdigit()
        ):
            return None
        return row_directory

    @staticmethod
    def _remove_empty_directory(path: Path) -> None:
        try:
            path.rmdir()
        except OSError:
            pass
