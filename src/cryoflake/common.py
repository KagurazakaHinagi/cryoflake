"""Shared hashing, progress reporting, and bounded worker execution."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, TypeVar, cast

import numpy as np
from tqdm import tqdm

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")
_ARRAY_CHECK_BUFFER_ELEMENTS = 256 * 1024


def array_all_finite(array: np.ndarray) -> bool:
    """Check every element using bounded buffers, including strided arrays."""
    if array.size <= _ARRAY_CHECK_BUFFER_ELEMENTS:
        return bool(np.isfinite(array).all())
    mask = np.empty(min(array.size, _ARRAY_CHECK_BUFFER_ELEMENTS), dtype=bool)
    for chunk in np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=_ARRAY_CHECK_BUFFER_ELEMENTS,
    ):
        selected_mask = mask[: chunk.size]
        np.isfinite(chunk, out=selected_mask)
        if not selected_mask.all():
            return False
    return True


def emit_progress(
    progress_callback: Optional[Callable[[str], None]],
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)


def validate_num_workers(num_workers: int, argument_name: str) -> int:
    try:
        normalized = int(num_workers)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{argument_name} must be a positive integer") from exc
    if (
        normalized <= 0
        or isinstance(num_workers, bool)
        or (not isinstance(num_workers, str) and normalized != num_workers)
    ):
        raise ValueError(f"{argument_name} must be a positive integer")
    return normalized


def path_key(path: os.PathLike) -> str:
    return str(Path(path).resolve())


def paths_must_be_distinct(
    paths: Mapping[str, os.PathLike | str | None],
) -> None:
    """Reject aliased file paths, including symlinks and existing hard links."""
    resolved_paths: list[tuple[str, Path]] = []
    for name, value in paths.items():
        if value is None:
            continue
        path = Path(value).resolve()
        for other_name, other_path in resolved_paths:
            if path == other_path or (
                path.exists() and other_path.exists() and path.samefile(other_path)
            ):
                raise ValueError(f"{other_name} and {name} must use distinct files")
        resolved_paths.append((name, path))


def run_tasks_with_progress(
    tasks: Sequence[TaskT],
    worker_callback: Callable[[TaskT], ResultT],
    num_workers: int,
    progress_description: str,
    success_predicate: Optional[Callable[[ResultT], bool]] = None,
    wait_for_workers_on_error: bool = False,
    stop_event: Optional[threading.Event] = None,
    result_callback: Optional[Callable[[ResultT], None]] = None,
) -> list[ResultT]:
    num_workers = validate_num_workers(num_workers, "num_workers")
    if not tasks:
        return []
    if success_predicate is None:

        def success_predicate(result):
            return True

    if stop_event is None:
        stop_event = threading.Event()

    if num_workers == 1:
        results = []
        success_count = 0
        try:
            with tqdm(
                total=len(tasks),
                desc=progress_description,
                unit="item",
            ) as progress:
                progress.set_postfix(success=success_count, refresh=False)
                for task in tasks:
                    if stop_event.is_set():
                        raise RuntimeError("Task execution was cancelled")
                    result = worker_callback(task)
                    if result_callback is not None:
                        result_callback(result)
                    if success_predicate(result):
                        success_count += 1
                    progress.set_postfix(success=success_count, refresh=False)
                    results.append(result)
                    progress.update(1)
        except BaseException:
            stop_event.set()
            raise
        return results

    # Snapshot task references before workers start, without an additional
    # queued tuple and integer per pending task. The result list is the only
    # required per-task output allocation.
    task_snapshot = tuple(tasks)
    pending_tasks = iter(enumerate(task_snapshot))
    pending_lock = threading.Lock()
    result_queue: queue.Queue[
        tuple[int, Optional[ResultT], Optional[BaseException]]
    ] = queue.Queue()
    results: list[Optional[ResultT]] = [None] * len(task_snapshot)

    def worker() -> None:
        while not stop_event.is_set():
            with pending_lock:
                try:
                    result_index, task = next(pending_tasks)
                except StopIteration:
                    return
            try:
                if stop_event.is_set():
                    return
                result = worker_callback(task)
                if result_callback is not None:
                    result_callback(result)
                result_queue.put((result_index, result, None))
            except BaseException as exc:
                stop_event.set()
                result_queue.put((result_index, None, exc))
                return

    worker_threads = [
        threading.Thread(target=worker, daemon=True)
        for _ in range(min(num_workers, len(task_snapshot)))
    ]
    attempted_threads = []
    completed_count = 0
    success_count = 0
    succeeded = False
    try:
        for worker_thread in worker_threads:
            # start() can be interrupted after spawning the OS thread. Track
            # it before calling so that an already-running worker is joined.
            attempted_threads.append(worker_thread)
            worker_thread.start()
        with tqdm(
            total=len(task_snapshot),
            desc=progress_description,
            unit="item",
        ) as progress:
            progress.set_postfix(success=success_count, refresh=False)
            while completed_count < len(task_snapshot):
                try:
                    result_index, result, exc = result_queue.get(timeout=0.1)
                except queue.Empty:
                    if not any(
                        worker_thread.is_alive() for worker_thread in worker_threads
                    ):
                        break
                    continue

                completed_count += 1
                if exc is not None:
                    stop_event.set()
                    progress.set_postfix(success=success_count, refresh=False)
                    progress.update(1)
                    raise exc
                if success_predicate(result):
                    success_count += 1
                progress.set_postfix(success=success_count, refresh=False)
                progress.update(1)
                results[result_index] = result
        if completed_count != len(task_snapshot):
            raise RuntimeError(
                "Task execution was cancelled before all tasks completed"
            )
        succeeded = True
    except BaseException:
        stop_event.set()
        raise
    finally:
        if succeeded or wait_for_workers_on_error:
            for worker_thread in attempted_threads:
                if worker_thread.ident is not None:
                    worker_thread.join()

    # Every slot was filled once: None can itself be a valid worker result.
    return cast(list[ResultT], results)


def sha256_file(path: str | os.PathLike) -> str:
    """Hash a file in bounded chunks, including large map and archive files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    """Return a deterministic SHA-256 for a JSON-serializable value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def format_gib(byte_count: int) -> str:
    """Format a byte count in binary gibibytes for progress reports."""
    return f"{byte_count / 1024**3:.1f} GiB"


def clean_token(value) -> str:
    """Normalize optional CIF/text fields, treating ``.`` and ``?`` as missing."""
    value = "" if value is None else str(value).strip()
    return "" if value in {".", "?"} else value


def finite_float(value, default: float = 0.0) -> float:
    """Convert a scalar to a finite float, using a default for absent values."""
    import math

    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def integer_token(*values, default: int = 0) -> int:
    """Return the first valid integer token, skipping absent or malformed fields."""
    for value in values:
        value = clean_token(value)
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return int(default)
