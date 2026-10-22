from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import uuid
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

from cryoflake.common import json_sha256, sha256_file
from cryoflake.manifests.splits import GroupedSplitRecord
from cryoflake.preprocess.summary import PREPROCESS_SUMMARY_FIELDNAMES
from cryoflake.storage.store import (
    preprocessed_artifact_member_reference,
    preprocessed_artifact_root,
)

PREPROCESS_CHECKPOINT_FORMAT_VERSION = "C2610210"
_GROUPED_RECORD_FIELDS = {
    "artifact_reference",
    "artifact_group_id",
    "component_group_ids",
    "interface_group_ids",
}
_CHECKPOINT_RECORD_FIELDS = {
    "row_index",
    "state",
    "summary_row",
    "artifact_reference",
    "target_artifact_reference",
    "grouped_split_record",
    "cached_artifact_dir",
    "cache_identity",
    "success_status",
    "artifact_voxel_size",
    "grid_shape",
    "entry_label",
    "artifact_identity",
    "checkpoint_generation",
}


class _InvalidCheckpointJSON(ValueError):
    def __init__(
        self,
        message: str,
        row_index: Optional[int],
        generation: Optional[str],
        *,
        ambiguous_row_index: bool,
    ) -> None:
        super().__init__(message)
        self.row_index = row_index
        self.generation = generation
        self.ambiguous_row_index = ambiguous_row_index


def _parse_checkpoint_json(serialized: str | bytes) -> object:
    problem = None
    last_object = None
    last_pairs = []

    def object_from_pairs(pairs):
        nonlocal problem, last_object, last_pairs
        result = {}
        for key, value in pairs:
            if key in result and problem is None:
                problem = f"Duplicate checkpoint JSON key: {key}"
            result[key] = value
        last_object, last_pairs = result, pairs
        return result

    def reject_constant(value):
        nonlocal problem
        problem = problem or f"Non-finite checkpoint JSON value: {value}"
        return None

    def finite_float(value):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else reject_constant(value)

    parsed = json.loads(
        serialized,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
    if problem is None:
        return parsed
    row_index = generation = None
    ambiguous_row_index = False
    if isinstance(parsed, dict) and parsed is last_object:
        indices = [value for key, value in last_pairs if key == "row_index"]
        ambiguous_row_index = len(indices) > 1
        generations = [
            value for key, value in last_pairs if key == "checkpoint_generation"
        ]
        if len(indices) == 1 and type(indices[0]) is int and indices[0] > 0:
            row_index = indices[0]
        if len(generations) == 1 and isinstance(generations[0], str):
            generation = generations[0]
    # Parsed values with duplicate keys or non-finite numbers never become
    # recovery data. Only unambiguous root identifiers can retire older rows.
    raise _InvalidCheckpointJSON(
        problem, row_index, generation, ambiguous_row_index=ambiguous_row_index
    )


def _checkpoint_record_fields_are_canonical(record: Mapping[str, object]) -> bool:
    if not set(record).issubset(_CHECKPOINT_RECORD_FIELDS):
        return False
    if "summary_row" in record:
        summary = record["summary_row"]
        if not isinstance(summary, Mapping) or not set(summary).issubset(
            PREPROCESS_SUMMARY_FIELDNAMES
        ):
            return False
    grouped = record.get("grouped_split_record")
    return grouped is None or (
        isinstance(grouped, Mapping) and set(grouped) == _GROUPED_RECORD_FIELDS
    )


class CheckpointContractMismatchError(RuntimeError):
    """Raised when existing progress belongs to a different preprocessing run."""


def _checkpoint_row_index(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("checkpoint row_index must be a positive integer")
    return value


def _path_value(path_value: object) -> Optional[str]:
    if path_value is None:
        return None
    return str(path_value)


def _grouped_record(
    record: Optional[GroupedSplitRecord],
) -> Optional[dict[str, object]]:
    if record is None:
        return None
    return {
        "artifact_reference": str(record.artifact_reference),
        "artifact_group_id": record.artifact_group_id,
        "component_group_ids": list(record.component_group_ids),
        "interface_group_ids": list(record.interface_group_ids),
    }


def _local_cache_identity(path: Path) -> Optional[dict[str, object]]:
    try:
        if not path.is_dir():
            return None
        members = []
        for member in sorted(path.iterdir(), key=lambda value: value.name):
            if not member.is_file():
                continue
            stat_result = member.stat()
            members.append(
                {
                    "name": member.name,
                    "size": stat_result.st_size,
                    "mtime_ns": stat_result.st_mtime_ns,
                    "sha256": sha256_file(member),
                }
            )
        if not members:
            return None
        return {"kind": "artifact_directory", "members": members}
    except OSError:
        return None


def checkpoint_record_for_result(
    result: object,
    *,
    artifact_identity: Optional[Mapping[str, object]] = None,
) -> dict[str, object]:
    upload_future = getattr(result, "upload_future")
    artifact_reference = getattr(result, "artifact_reference")
    summary_row = dict(getattr(result, "summary_row"))
    if upload_future is not None:
        state = "upload_pending"
    elif artifact_reference is not None and summary_row.get("success") == "true":
        state = "durable_success"
    else:
        state = "terminal_failure"

    cached_artifact_dir = getattr(result, "cached_artifact_dir")
    cache_identity = None
    if cached_artifact_dir is not None:
        cache_identity = _local_cache_identity(Path(cached_artifact_dir))

    summary_row["split"] = ""
    return {
        "row_index": getattr(result, "row_index"),
        "state": state,
        "summary_row": summary_row,
        "artifact_reference": _path_value(artifact_reference),
        "target_artifact_reference": summary_row.get("artifact_reference", ""),
        "grouped_split_record": _grouped_record(
            getattr(result, "grouped_split_record")
        ),
        "cached_artifact_dir": _path_value(cached_artifact_dir),
        "cache_identity": cache_identity,
        "success_status": getattr(result, "success_status"),
        "artifact_voxel_size": getattr(result, "artifact_voxel_size"),
        "grid_shape": getattr(result, "grid_shape"),
        "entry_label": getattr(result, "entry_label"),
        "artifact_identity": (
            None if artifact_identity is None else dict(artifact_identity)
        ),
    }


def artifact_reference_from_checkpoint(path_value: str) -> Path | str:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Checkpoint artifact_reference must be a nonempty string")
    artifact_reference = preprocessed_artifact_root(path_value)
    # Validate remote authorities as well as the root/member spelling.
    preprocessed_artifact_member_reference(artifact_reference, "metadata.json")
    return artifact_reference


def grouped_record_from_checkpoint(
    payload: object,
) -> Optional[GroupedSplitRecord]:
    if not isinstance(payload, Mapping) or set(payload) != _GROUPED_RECORD_FIELDS:
        return None
    path_value = payload.get("artifact_reference")
    artifact_group_id = payload.get("artifact_group_id")
    if not isinstance(artifact_group_id, str) or not artifact_group_id.strip():
        return None
    component_group_ids = payload.get("component_group_ids")
    interface_group_ids = payload.get("interface_group_ids")
    if not isinstance(component_group_ids, list) or not isinstance(
        interface_group_ids,
        list,
    ):
        return None
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (*component_group_ids, *interface_group_ids)
    ):
        return None
    try:
        return GroupedSplitRecord(
            artifact_reference=artifact_reference_from_checkpoint(path_value),
            artifact_group_id=artifact_group_id,
            component_group_ids=tuple(component_group_ids),
            interface_group_ids=tuple(interface_group_ids),
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _managed_upload_cache_path(
    cache_path: Path,
    *,
    cache_parent: Path,
    row_index: int,
) -> bool:
    try:
        if any(
            path.is_symlink()
            for path in (cache_path, cache_path.parent, cache_path.parent.parent)
        ):
            return False
        resolved_path = cache_path.resolve(strict=True)
        resolved_parent = cache_parent.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    row_directory = resolved_path.parent
    cache_directory = row_directory.parent
    return (
        cache_directory.parent == resolved_parent
        and cache_directory.name.startswith(".upload_cache_")
        and row_directory.name == f"row_{row_index:08d}"
    )


def checkpoint_cache_matches(
    record: Mapping[str, object],
    *,
    cache_parent: Path,
) -> bool:
    cache_path_value = str(record.get("cached_artifact_dir") or "")
    expected_identity = record.get("cache_identity")
    if not cache_path_value or not isinstance(expected_identity, Mapping):
        return False
    try:
        row_index = _checkpoint_row_index(record["row_index"])
    except (KeyError, TypeError, ValueError):
        return False
    cache_path = Path(cache_path_value)
    if not _managed_upload_cache_path(
        cache_path,
        cache_parent=cache_parent,
        row_index=row_index,
    ):
        return False
    current_identity = _local_cache_identity(cache_path)
    return current_identity == dict(expected_identity)


class PreprocessCheckpoint:
    """Append-only per-row progress journal for resumable preprocessing."""

    def __init__(
        self,
        summary_csv: os.PathLike,
        contract: Mapping[str, Any],
        *,
        reset: bool = False,
    ) -> None:
        self.contract = dict(contract)
        # Reject non-JSON and non-finite contract values before acquiring
        # ownership or creating progress files.
        json.dumps(self.contract, allow_nan=False)
        self.fingerprint = json_sha256(self.contract)
        summary_path = Path(summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path = Path(f"{summary_path}.checkpoint.json")
        self.journal_path = Path(f"{summary_path}.checkpoint.jsonl")
        self.lock_path = Path(f"{summary_path}.checkpoint.lock")
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._closed = False
        self._failure: Optional[BaseException] = None
        self._records: dict[int, dict[str, Any]] = {}
        self._ownership_file = None

        self._acquire_ownership()
        try:
            matching_metadata = None if reset else self._matching_metadata()
            checkpoint_exists = (
                self.metadata_path.exists() or self.journal_path.exists()
            )
            if matching_metadata is not None and self.journal_path.is_file():
                self._generation = str(matching_metadata["generation"])
                self._repair_journal_tail()
                self._records = self._read_records()
                self._journal_file = self.journal_path.open(
                    "a",
                    encoding="utf-8",
                )
            elif checkpoint_exists and not reset and matching_metadata is None:
                raise CheckpointContractMismatchError(
                    "Existing preprocessing checkpoint does not match the "
                    f"current run: {self.metadata_path}. Keep it for its "
                    "original configuration, choose a different --summary-csv, "
                    "or explicitly reset preprocessing progress with "
                    "--reset-preprocess-checkpoint."
                )
            else:
                if reset and checkpoint_exists:
                    self._warn_about_retained_upload_caches()
                self._generation = uuid.uuid4().hex
                self._reset_journal()
                self._write_metadata_atomically()
                self._journal_file = self.journal_path.open(
                    "a",
                    encoding="utf-8",
                )
        except BaseException:
            self._release_ownership()
            raise

    @property
    def records(self) -> dict[int, dict[str, Any]]:
        self._check_owner_process()
        with self._lock:
            return deepcopy(self._records)

    def _check_owner_process(self) -> None:
        # A fork inherits both the file description and possibly a locked
        # threading.Lock. Reject use before touching either: LOCK_UN from the
        # child would release its parent's checkpoint ownership.
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "Preprocessing checkpoint belongs to another process; "
                "create a checkpoint in the process that will use it"
            )

    def _acquire_ownership(self) -> None:
        ownership_file = self.lock_path.open("a+")
        try:
            import fcntl
        except ImportError as exc:
            ownership_file.close()
            raise RuntimeError(
                "Preprocessing checkpoint ownership requires fcntl file locks"
            ) from exc
        try:
            fcntl.flock(
                ownership_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            ownership_file.close()
            raise RuntimeError(
                f"Another preprocessing run owns checkpoint {self.journal_path}"
            ) from exc
        except BaseException:
            ownership_file.close()
            raise
        self._ownership_file = ownership_file

    def _release_ownership(self) -> None:
        ownership_file = self._ownership_file
        if ownership_file is None:
            return
        self._ownership_file = None
        try:
            import fcntl

            fcntl.flock(ownership_file.fileno(), fcntl.LOCK_UN)
        finally:
            ownership_file.close()

    def _matching_metadata(self) -> Optional[dict[str, Any]]:
        if not self.metadata_path.is_file():
            return None
        try:
            metadata = _parse_checkpoint_json(self.metadata_path.read_text())
        except (OSError, TypeError, ValueError):
            return None
        if (
            isinstance(metadata, Mapping)
            and set(metadata)
            == {"format_version", "fingerprint", "generation", "contract"}
            and type(metadata.get("format_version")) is str
            and metadata.get("format_version") == PREPROCESS_CHECKPOINT_FORMAT_VERSION
            and metadata.get("fingerprint") == self.fingerprint
            and isinstance(metadata.get("contract"), Mapping)
            and json_sha256(metadata["contract"]) == self.fingerprint
            and isinstance(metadata.get("generation"), str)
            and metadata["generation"]
        ):
            return metadata
        return None

    def _warn_about_retained_upload_caches(self) -> None:
        cache_paths = set()
        try:
            with self.journal_path.open("rb") as input_file:
                for line in input_file:
                    try:
                        record = _parse_checkpoint_json(line)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    if not isinstance(record, Mapping):
                        continue
                    cache_path = str(record.get("cached_artifact_dir") or "")
                    if cache_path and Path(cache_path).exists():
                        cache_paths.add(cache_path)
        except FileNotFoundError:
            return
        if not cache_paths:
            return
        preview = ", ".join(sorted(cache_paths)[:3])
        if len(cache_paths) > 3:
            preview += f", and {len(cache_paths) - 3} more"
        warnings.warn(
            "Resetting preprocessing progress leaves retained upload caches "
            f"unchanged; review them before deletion: {preview}",
            RuntimeWarning,
            stacklevel=3,
        )

    def _reset_journal(self) -> None:
        with self.journal_path.open("wb") as output_file:
            output_file.flush()
            os.fsync(output_file.fileno())

    def _repair_journal_tail(self) -> None:
        with self.journal_path.open("rb+") as journal_file:
            journal_file.seek(0, os.SEEK_END)
            end_position = journal_file.tell()
            if end_position == 0:
                return
            journal_file.seek(-1, os.SEEK_END)
            if journal_file.read(1) == b"\n":
                return

            search_end = end_position
            truncate_position = 0
            while search_end > 0:
                chunk_start = max(0, search_end - 64 * 1024)
                journal_file.seek(chunk_start)
                chunk = journal_file.read(search_end - chunk_start)
                newline_index = chunk.rfind(b"\n")
                if newline_index >= 0:
                    truncate_position = chunk_start + newline_index + 1
                    break
                search_end = chunk_start
            journal_file.truncate(truncate_position)
            journal_file.flush()
            os.fsync(journal_file.fileno())

    def _read_records(self) -> dict[int, dict[str, Any]]:
        records = {}
        try:
            with self.journal_path.open("rb") as input_file:
                for line in input_file:
                    try:
                        record = _parse_checkpoint_json(line)
                        row_index = _checkpoint_row_index(record["row_index"])
                    except _InvalidCheckpointJSON as exc:
                        if exc.generation in {None, self._generation}:
                            if exc.ambiguous_row_index:
                                records.clear()
                            elif exc.row_index is not None:
                                records.pop(exc.row_index, None)
                        continue
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ):
                        continue
                    if record.get("checkpoint_generation") == self._generation:
                        if _checkpoint_record_fields_are_canonical(record):
                            records[row_index] = record
                        else:
                            records.pop(row_index, None)
        except FileNotFoundError:
            pass
        return records

    def _write_metadata_atomically(self) -> None:
        metadata = {
            "format_version": PREPROCESS_CHECKPOINT_FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "generation": self._generation,
            "contract": self.contract,
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.metadata_path.parent,
                prefix=f".{self.metadata_path.name}.",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                json.dump(
                    metadata, output_file, sort_keys=True, indent=2, allow_nan=False
                )
                output_file.write("\n")
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, self.metadata_path)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    def record(self, record: Mapping[str, Any]) -> None:
        self._check_owner_process()
        row_index = _checkpoint_row_index(record["row_index"])
        if not _checkpoint_record_fields_are_canonical(record):
            raise ValueError(
                "Checkpoint record contains unexpected or malformed fields"
            )
        normalized_record = deepcopy(dict(record))
        normalized_record["row_index"] = row_index
        normalized_record["checkpoint_generation"] = self._generation
        payload = json.dumps(
            normalized_record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            self._raise_if_failed_locked()
            if self._closed:
                raise RuntimeError("Cannot write to a closed checkpoint")
            try:
                self._journal_file.write(payload + "\n")
                self._journal_file.flush()
            except BaseException as exc:
                self._failure = exc
                raise
            self._records[row_index] = normalized_record

    def record_from_callback(self, record: Mapping[str, Any]) -> None:
        try:
            self.record(record)
        except BaseException as exc:
            self.fail_from_callback(exc)

    def fail_from_callback(self, exc: BaseException) -> None:
        self._check_owner_process()
        with self._lock:
            if self._failure is None:
                self._failure = exc

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                f"Preprocessing checkpoint write failed: {self._failure}"
            ) from self._failure

    def raise_if_failed(self) -> None:
        self._check_owner_process()
        with self._lock:
            self._raise_if_failed_locked()

    def close(self) -> None:
        self._check_owner_process()
        with self._lock:
            if self._closed:
                return
            try:
                self._journal_file.close()
            finally:
                self._closed = True
                self._release_ownership()

    def complete(self) -> None:
        self._check_owner_process()
        with self._lock:
            if self._closed:
                return
            try:
                self._journal_file.close()
                self._raise_if_failed_locked()
                for path in (self.journal_path, self.metadata_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
            finally:
                self._closed = True
                self._release_ownership()

    def __enter__(self) -> "PreprocessCheckpoint":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
