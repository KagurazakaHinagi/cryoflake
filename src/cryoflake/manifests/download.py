import gzip
import os
import shutil
import socket
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from cryoflake.common import (
    emit_progress,
    path_key,
    run_tasks_with_progress,
    validate_num_workers,
)
from cryoflake.manifests.manifest import ManifestEntry


def download_url(
    url: str,
    output_path: os.PathLike,
    force_download: bool = False,
) -> Path:
    output_path = Path(output_path)
    if _download_output_is_reusable(output_path) and not force_download:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with urlopen(url) as response:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                shutil.copyfileobj(response, output_file)
        temporary_path.replace(output_path)
    except HTTPError as exc:
        exc.close()
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def _download_output_is_reusable(output_path: Path) -> bool:
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        return False
    if output_path.suffix.lower() != ".gz":
        return True
    try:
        with gzip.open(output_path, "rb") as input_file:
            return bool(input_file.read(1))
    except Exception:
        return False


def _download_not_found_error(
    url: str,
    output_path: os.PathLike,
    exc: HTTPError,
    label: str,
) -> str:
    reason = getattr(exc, "reason", None) or getattr(exc, "msg", "")
    if reason:
        return (
            f"HTTP 404 while downloading {label} from {url} to {output_path}: {reason}"
        )
    return f"HTTP 404 while downloading {label} from {url} to {output_path}"


def _download_url_error(
    url: str,
    output_path: os.PathLike,
    exc: URLError,
    label: str,
) -> str:
    reason = getattr(exc, "reason", None) or str(exc)
    return (
        f"Network error while downloading {label} from {url} to {output_path}: {reason}"
    )


def _is_recoverable_url_error(exc: URLError) -> bool:
    reason = getattr(exc, "reason", None)
    if not isinstance(reason, socket.gaierror):
        return False
    recoverable_codes = {socket.EAI_NONAME}
    if hasattr(socket, "EAI_NODATA"):
        recoverable_codes.add(socket.EAI_NODATA)
    return reason.errno in recoverable_codes


def _record_download_error(
    download_errors: dict[str, str],
    url: str,
    output_path: os.PathLike,
    exc: HTTPError | URLError,
    label: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    if isinstance(exc, HTTPError) and exc.code == 404:
        error = _download_not_found_error(url, output_path, exc, label)
    else:
        error = _download_url_error(url, output_path, exc, label)
    download_errors[path_key(output_path)] = error
    warnings.warn(error, RuntimeWarning, stacklevel=2)
    emit_progress(progress_callback, f"Warning: {error}")


@dataclass(frozen=True)
class _ManifestDownloadTask:
    url: str
    output_path: Path
    download_label: str


@dataclass(frozen=True)
class _ManifestDownloadResult:
    task: _ManifestDownloadTask
    download_error: Optional[HTTPError | URLError] = None


def _run_manifest_download_task(
    task: _ManifestDownloadTask,
    force_download: bool,
) -> _ManifestDownloadResult:
    try:
        download_url(task.url, task.output_path, force_download=force_download)
    except HTTPError as exc:
        if exc.code != 404:
            raise
        return _ManifestDownloadResult(task=task, download_error=exc)
    except URLError as exc:
        if not _is_recoverable_url_error(exc):
            raise
        return _ManifestDownloadResult(task=task, download_error=exc)
    return _ManifestDownloadResult(task=task)


def _run_manifest_download_tasks(
    tasks: Iterable[_ManifestDownloadTask],
    force_download: bool,
    num_workers: int,
    progress_description: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    tasks = list(tasks)
    download_errors: dict[str, str] = {}
    if not tasks:
        return download_errors

    results = run_tasks_with_progress(
        tasks,
        lambda task: _run_manifest_download_task(task, force_download=force_download),
        num_workers=num_workers,
        progress_description=progress_description,
        success_predicate=lambda result: result.download_error is None,
        wait_for_workers_on_error=True,
    )

    for result in results:
        if result.download_error is None:
            continue
        _record_download_error(
            download_errors,
            result.task.url,
            result.task.output_path,
            result.download_error,
            label=result.task.download_label,
            progress_callback=progress_callback,
        )
    return download_errors


def _download_manifest_entries_parallel(
    entries: list[ManifestEntry],
    maps_dir: Path,
    structures_dir: Path,
    force_download: bool,
    num_workers: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    map_tasks = [
        _ManifestDownloadTask(
            url=entry.map_url,
            output_path=maps_dir / entry.map_file_name,
            download_label="map",
        )
        for entry in entries
    ]
    download_errors = _run_manifest_download_tasks(
        map_tasks,
        force_download=force_download,
        num_workers=num_workers,
        progress_description="Downloading maps",
        progress_callback=progress_callback,
    )

    downloaded_wwpdb_ids = set()
    structure_entries = []
    for entry in entries:
        if entry.wwpdb_id is None:
            continue
        if path_key(maps_dir / entry.map_file_name) in download_errors:
            continue
        if entry.wwpdb_id in downloaded_wwpdb_ids:
            continue
        downloaded_wwpdb_ids.add(entry.wwpdb_id)
        structure_entries.append(entry)

    structure_tasks = [
        _ManifestDownloadTask(
            url=entry.structure_url,
            output_path=structures_dir / entry.structure_file_name,
            download_label="structure",
        )
        for entry in structure_entries
    ]
    download_errors.update(
        _run_manifest_download_tasks(
            structure_tasks,
            force_download=force_download,
            num_workers=num_workers,
            progress_description="Downloading structures",
            progress_callback=progress_callback,
        )
    )
    return download_errors


def download_manifest_entries(
    entries: Iterable[ManifestEntry],
    maps_dir: os.PathLike,
    structures_dir: os.PathLike,
    force_download: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    num_workers: int = 1,
) -> dict[str, str]:
    entries = list(entries)
    maps_dir = Path(maps_dir)
    structures_dir = Path(structures_dir)
    num_workers = validate_num_workers(num_workers, "num_workers")
    if num_workers > 1:
        return _download_manifest_entries_parallel(
            entries,
            maps_dir=maps_dir,
            structures_dir=structures_dir,
            force_download=force_download,
            num_workers=num_workers,
            progress_callback=progress_callback,
        )

    downloaded_wwpdb_ids = set()
    failed_wwpdb_ids = set()
    download_errors: dict[str, str] = {}
    total_downloads = len(entries)
    for entry_index, entry in enumerate(entries, start=1):
        emit_progress(progress_callback, f"Downloading {entry_index}/{total_downloads}")
        map_path = maps_dir / entry.map_file_name
        try:
            download_url(entry.map_url, map_path, force_download=force_download)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            _record_download_error(
                download_errors,
                entry.map_url,
                map_path,
                exc,
                label="map",
                progress_callback=progress_callback,
            )
            continue
        except URLError as exc:
            if not _is_recoverable_url_error(exc):
                raise
            _record_download_error(
                download_errors,
                entry.map_url,
                map_path,
                exc,
                label="map",
                progress_callback=progress_callback,
            )
            continue
        if (
            entry.wwpdb_id is not None
            and entry.wwpdb_id not in downloaded_wwpdb_ids
            and entry.wwpdb_id not in failed_wwpdb_ids
        ):
            structure_path = structures_dir / entry.structure_file_name
            try:
                download_url(
                    entry.structure_url,
                    structure_path,
                    force_download=force_download,
                )
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                _record_download_error(
                    download_errors,
                    entry.structure_url,
                    structure_path,
                    exc,
                    label="structure",
                    progress_callback=progress_callback,
                )
                failed_wwpdb_ids.add(entry.wwpdb_id)
                continue
            except URLError as exc:
                if not _is_recoverable_url_error(exc):
                    raise
                _record_download_error(
                    download_errors,
                    entry.structure_url,
                    structure_path,
                    exc,
                    label="structure",
                    progress_callback=progress_callback,
                )
                failed_wwpdb_ids.add(entry.wwpdb_id)
                continue
            downloaded_wwpdb_ids.add(entry.wwpdb_id)
    return download_errors
