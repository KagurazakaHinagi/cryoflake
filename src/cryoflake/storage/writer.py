"""Atomic local filesystem writes for the NPZ+JSON storage codec."""

from __future__ import annotations

import os
import stat
import uuid
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from cryoflake.common import sha256_file
from cryoflake.storage.codec import (
    CORE_FILE_NAME,
    MAX_METADATA_BYTES,
    METADATA_FILE_NAME,
    SEQUENCE_FILE_NAME,
    SEQUENCE_ROOT_KEYS,
    _metadata_bytes,
    flatten_payload,
    read_metadata,
)
from cryoflake.storage.locks import (
    preprocessed_artifact_write_locks,
)


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _preserve_replaced_mode(temporary_path: Path, path: Path) -> None:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        return
    temporary_path.chmod(mode)


def _stage_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_output_path(path)
    try:
        with temporary_path.open("xb") as output_file:
            # Field names are data, including names such as ``file`` and
            # ``allow_pickle`` that are parameters of NumPy's savez helpers.
            with zipfile.ZipFile(
                output_file, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for key, array in arrays.items():
                    with archive.open(f"{key}.npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, array, allow_pickle=False)
            output_file.flush()
            os.fsync(output_file.fileno())
        _preserve_replaced_mode(temporary_path, path)
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_output_path(path)
    try:
        with temporary_path.open("xb") as output_file:
            output_file.write(payload)
            output_file.flush()
            os.fsync(output_file.fileno())
        _preserve_replaced_mode(temporary_path, path)
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _commit_staged_file(temporary_path: Path, path: Path) -> None:
    os.replace(temporary_path, path)


def _checked_metadata_bytes(metadata: Mapping[str, object]) -> bytes:
    serialized = _metadata_bytes(metadata)
    if len(serialized) > MAX_METADATA_BYTES:
        raise ValueError(
            f"NPZ+JSON metadata exceeds the {MAX_METADATA_BYTES}-byte format limit"
        )
    return serialized


def _commit_staged_files(
    staged_targets: list[tuple[Path, Path]],
) -> None:
    """Commit a local multi-file update and roll back ordinary I/O failures."""

    backups: dict[Path, Path] = {}
    attempted_targets = []
    try:
        for _, target in staged_targets:
            if not target.exists() and not target.is_symlink():
                continue
            backup = _temporary_output_path(target).with_suffix(".backup")
            os.link(target, backup, follow_symlinks=False)
            backups[target] = backup

        for staged_path, target in staged_targets:
            attempted_targets.append(target)
            _commit_staged_file(staged_path, target)
    except BaseException:
        rollback_error = None
        for target in reversed(attempted_targets):
            backup = backups.pop(target, None)
            try:
                if backup is not None:
                    if (target.exists() or target.is_symlink()) and os.path.samestat(
                        backup.lstat(), target.lstat()
                    ):
                        backup.unlink()
                    else:
                        os.replace(backup, target)
                else:
                    target.unlink(missing_ok=True)
            except BaseException as exc:
                if rollback_error is None:
                    rollback_error = exc
        for target, backup in list(backups.items()):
            try:
                backup.unlink(missing_ok=True)
            except BaseException as exc:
                if rollback_error is None:
                    rollback_error = exc
        if rollback_error is not None:
            raise RuntimeError(
                "NPZ+JSON commit failed and its local rollback was incomplete"
            ) from rollback_error
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def write_npz_json(
    artifact_dir: Path,
    artifact_mapping: Mapping[str, object],
) -> None:
    """Write a canonical artifact with embeddings in a separate sequence store."""

    with preprocessed_artifact_write_locks(artifact_dir):
        _write_npz_json_unlocked(
            artifact_dir,
            artifact_mapping,
        )


def _write_npz_json_unlocked(
    artifact_dir: Path,
    artifact_mapping: Mapping[str, object],
) -> None:
    metadata, stores = flatten_payload(artifact_mapping)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = artifact_dir / METADATA_FILE_NAME
    core_path = artifact_dir / CORE_FILE_NAME
    sequence_path = artifact_dir / SEQUENCE_FILE_NAME
    staged_paths = []
    try:
        staged_core = _stage_npz(core_path, stores["core"])
        staged_paths.append(staged_core)
        staged_sequence = None
        if stores["sequence"]:
            staged_sequence = _stage_npz(
                sequence_path,
                stores["sequence"],
            )
            staged_paths.append(staged_sequence)
        metadata["store_hashes"] = {
            "core": sha256_file(staged_core),
            **(
                {"sequence": sha256_file(staged_sequence)}
                if staged_sequence is not None
                else {}
            ),
        }
        serialized_metadata = _checked_metadata_bytes(metadata)
        staged_metadata = _stage_bytes(
            metadata_path,
            serialized_metadata,
        )
        staged_paths.append(staged_metadata)

        staged_targets = [(staged_core, core_path)]
        if staged_sequence is not None:
            staged_targets.append((staged_sequence, sequence_path))
        staged_targets.append((staged_metadata, metadata_path))
        _commit_staged_files(staged_targets)
        staged_paths.clear()
    finally:
        for staged_path in staged_paths:
            staged_path.unlink(missing_ok=True)
    if not stores["sequence"]:
        sequence_path.unlink(missing_ok=True)


def write_sequence_npz_json(
    metadata_path: Path,
    sequence_payload: Mapping[str, object],
) -> None:
    """Replace the split sequence store without rewriting the core NPZ."""

    metadata_path = Path(metadata_path)
    with preprocessed_artifact_write_locks(metadata_path.parent):
        _write_sequence_npz_json_unlocked(
            metadata_path,
            sequence_payload,
        )


def _write_sequence_npz_json_unlocked(
    metadata_path: Path,
    sequence_payload: Mapping[str, object],
) -> None:
    metadata = read_metadata(metadata_path)
    unexpected = set(sequence_payload) - SEQUENCE_ROOT_KEYS
    if unexpected:
        raise ValueError(
            "Sequence sidecar contains unsupported top-level keys: "
            f"{sorted(unexpected)}"
        )

    sequence_metadata, stores = flatten_payload(sequence_payload)
    sequence_items = sequence_metadata["tree"]["items"]
    existing_items = metadata["tree"]["items"]
    metadata["tree"]["items"] = {
        key: value
        for key, value in existing_items.items()
        if key not in SEQUENCE_ROOT_KEYS
    }
    metadata["tree"]["items"].update(sequence_items)

    sequence_path = metadata_path.with_name(SEQUENCE_FILE_NAME)
    if stores["sequence"]:
        metadata["files"]["sequence"] = SEQUENCE_FILE_NAME
        metadata["store_generations"]["sequence"] = sequence_metadata[
            "store_generations"
        ]["sequence"]
    else:
        metadata["files"].pop("sequence", None)
        metadata["store_generations"].pop("sequence", None)
    staged_paths = []
    try:
        staged_sequence = None
        if stores["sequence"]:
            staged_sequence = _stage_npz(
                sequence_path,
                stores["sequence"],
            )
            staged_paths.append(staged_sequence)
            metadata["store_hashes"]["sequence"] = sha256_file(staged_sequence)
        else:
            metadata["store_hashes"].pop("sequence", None)
        serialized_metadata = _checked_metadata_bytes(metadata)
        staged_metadata = _stage_bytes(
            metadata_path,
            serialized_metadata,
        )
        staged_paths.append(staged_metadata)
        staged_targets = []
        if staged_sequence is not None:
            staged_targets.append((staged_sequence, sequence_path))
        staged_targets.append((staged_metadata, metadata_path))
        _commit_staged_files(staged_targets)
        staged_paths.clear()
    finally:
        for staged_path in staged_paths:
            staged_path.unlink(missing_ok=True)
    if not stores["sequence"]:
        sequence_path.unlink(missing_ok=True)
